"""MLflow context manager — log params, metrics, datasets, pipelines and model cards.

Usage::

    from iad.ml.tracking import MLflowTracker

    with MLflowTracker(experiment="telco-churn") as tracker:
        tracker.log_params(card.best_params)
        tracker.log_metrics(card.metrics)
        tracker.log_pipeline(pipeline)
        tracker.log_model_card(card)

If MLflow is unavailable (the optional dep is missing or the configured
``tracking_uri`` is unreachable) the tracker degrades gracefully into a
no-op — the surrounding code keeps working and a warning is logged.

The default tracking URI is taken from ``Settings.MLFLOW_TRACKING_URI``;
if unset, MLflow writes to ``./mlruns/`` next to the running process
(MLflow's stock behaviour).
"""
from __future__ import annotations

import json
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from iad.config.settings import get_settings
from iad.core.logging import get_logger
from iad.core.paths import safe_filename
from iad.ml.tracking.runs import RunMetadata

logger = get_logger("iad.ml.tracking.mlflow")


def mlflow_available() -> bool:
    """Return True if ``mlflow`` (or ``mlflow-skinny``) is importable."""
    try:
        import mlflow  # type: ignore[import-not-found]  # noqa: F401
        return True
    except ImportError:
        return False


class MLflowTracker:
    """Context manager that wraps a single MLflow run.

    Args:
        experiment: experiment name. Created if it does not exist.
        run_name: optional human-readable run name.
        tracking_uri: explicit URI. Defaults to ``Settings.MLFLOW_TRACKING_URI``
            and finally to MLflow's local ``./mlruns/`` directory.
        tags: extra tags attached to the run.
    """

    def __init__(
        self,
        experiment: str = "iad_default",
        *,
        run_name: str | None = None,
        tracking_uri: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> None:
        self.experiment = experiment
        self.run_name = run_name
        self.tracking_uri = tracking_uri or get_settings().MLFLOW_TRACKING_URI
        self.tags = dict(tags or {})
        self._mlflow = None
        self._active_run = None
        self._run_metadata: RunMetadata | None = None
        self._enabled = mlflow_available()

    # ------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def run_id(self) -> str | None:
        return self._run_metadata.run_id if self._run_metadata else None

    # ------------------------------------------------------------------
    def __enter__(self) -> MLflowTracker:
        if not self._enabled:
            logger.warning(
                "MLflow not installed; tracker running in no-op mode. "
                "Install mlflow or mlflow-skinny to enable tracking."
            )
            return self
        try:
            import mlflow  # type: ignore[import-not-found]

            self._mlflow = mlflow
            if self.tracking_uri:
                mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment)
            self._active_run = mlflow.start_run(run_name=self.run_name)
            for k, v in self.tags.items():
                mlflow.set_tag(k, str(v))
            run = self._active_run
            self._run_metadata = RunMetadata(
                run_id=run.info.run_id,
                experiment_id=run.info.experiment_id,
                experiment_name=self.experiment,
                tracking_uri=mlflow.get_tracking_uri(),
                artifact_uri=run.info.artifact_uri,
            )
            logger.info(
                "mlflow run started",
                extra={
                    "ctx_run_id": self.run_id,
                    "ctx_experiment": self.experiment,
                    "ctx_uri": mlflow.get_tracking_uri(),
                },
            )
        except Exception as exc:  # pragma: no cover — degrade gracefully
            logger.warning("MLflow setup failed: %s — tracker disabled", exc)
            self._enabled = False
        return self

    def __exit__(self, exc_type, exc_value, exc_tb) -> None:
        if not self._enabled or self._mlflow is None:
            return
        try:
            if exc_value is not None:
                self._mlflow.set_tag("error", repr(exc_value)[:500])
            self._mlflow.end_run(status="FAILED" if exc_type else "FINISHED")
            logger.info("mlflow run ended", extra={"ctx_run_id": self.run_id})
        except Exception as exc:  # pragma: no cover
            logger.warning("MLflow end_run failed: %s", exc)

    # ==================================================================
    # Logging API — every method is a safe no-op when ``enabled`` is False
    # ==================================================================
    def log_params(self, params: dict[str, Any]) -> None:
        if not self._enabled or self._mlflow is None:
            return
        with suppress(Exception):
            for key, value in params.items():
                self._mlflow.log_param(key, _coerce_scalar(value))

    def log_metrics(self, metrics: dict[str, float], step: int = 0) -> None:
        if not self._enabled or self._mlflow is None:
            return
        with suppress(Exception):
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self._mlflow.log_metric(k, float(v), step=step)

    def log_artifact(self, path: Path | str) -> None:
        if not self._enabled or self._mlflow is None:
            return
        with suppress(Exception):
            self._mlflow.log_artifact(str(path))

    def log_dataset(self, df: pd.DataFrame, name: str = "dataset") -> None:
        if not self._enabled or self._mlflow is None:
            return
        try:
            with tempfile.TemporaryDirectory() as tmp:
                csv_path = Path(tmp) / f"{safe_filename(name)}.csv"
                df.to_csv(csv_path, index=False)
                self._mlflow.log_artifact(str(csv_path), artifact_path="datasets")
                self._mlflow.log_param(f"{name}_shape", f"{df.shape[0]}x{df.shape[1]}")
                self._mlflow.log_param(f"{name}_columns", ",".join(df.columns.astype(str)))
        except Exception as exc:  # pragma: no cover
            logger.debug("log_dataset failed: %s", exc)

    def log_pipeline(self, pipeline: Pipeline, name: str = "model") -> None:
        if not self._enabled or self._mlflow is None:
            return
        # MLflow's modern API is mlflow.sklearn.log_model; fall back gracefully
        # for older / minimal installs (mlflow-skinny may lack the sklearn
        # adapter, in which case we persist via joblib as an artifact).
        try:
            from mlflow import sklearn as mlflow_sklearn  # type: ignore[import-not-found]

            mlflow_sklearn.log_model(pipeline, artifact_path=name)
            return
        except Exception as exc:
            logger.debug("mlflow.sklearn.log_model unavailable: %s", exc)
        try:
            import joblib

            with tempfile.TemporaryDirectory() as tmp:
                joblib_path = Path(tmp) / f"{safe_filename(name)}.joblib"
                joblib.dump(pipeline, joblib_path)
                self._mlflow.log_artifact(str(joblib_path), artifact_path=name)
        except Exception as exc:  # pragma: no cover
            logger.warning("log_pipeline fallback failed: %s", exc)

    def log_model_card(self, model_card) -> None:
        """Persist a :class:`iad.ml.training.ModelCard` as a JSON artifact."""
        if not self._enabled or self._mlflow is None:
            return
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "model_card.json"
                path.write_text(
                    json.dumps(model_card.to_dict(), indent=2, default=str), encoding="utf-8"
                )
                self._mlflow.log_artifact(str(path))
        except Exception as exc:  # pragma: no cover
            logger.warning("log_model_card failed: %s", exc)


def _coerce_scalar(value: Any) -> Any:
    """MLflow ``log_param`` accepts scalars and short strings only."""
    if value is None or isinstance(value, (int, float, bool, str)):
        return value
    return str(value)[:250]
