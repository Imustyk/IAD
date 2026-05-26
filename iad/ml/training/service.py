"""TrainingService — orchestrates preprocessing + leaderboard + reporting.

Composition
-----------
* :class:`iad.ml.training.registry.ModelRegistry` — provides candidate models.
* :func:`iad.ml.preprocessing.build_auto_preprocessor` — provides the column
  transformer pipeline that lives upstream of every model.
* :mod:`iad.ml.evaluation` — computes consistent metrics across all models.
* :mod:`iad.ml.tracking` — optionally logs every run to MLflow.

The service is consumed by:

* the Streamlit predictive-modeling page (Phase 4),
* the FastAPI ``/train`` endpoint (Phase 5).
"""
from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from iad import __version__ as IAD_VERSION
from iad.config.settings import get_settings
from iad.core.exceptions import TrainingError
from iad.core.logging import get_logger
from iad.core.validation import (
    validate_columns_present,
    validate_dataframe,
    validate_target_column,
)
from iad.ml.evaluation.metrics import (
    classification_metrics,
    primary_metric_name,
    regression_metrics,
    scoring_for,
)
from iad.ml.evaluation.reports import (
    build_calibration_report,
    build_confusion_matrix_report,
    build_regression_report,
)
from iad.ml.preprocessing.pipelines.auto import build_auto_preprocessor
from iad.ml.training.registry import ModelRegistry, ModelSpec, Task
from iad.ml.training.reports import LeaderboardEntry, TrainingResult
from iad.ml.training.reproducibility import (
    ModelCard,
    SeedManager,
    capture_environment,
    fingerprint_dataframe,
)

logger = get_logger("iad.training")  # iad.training is the dedicated log channel


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TrainingConfig:
    """Knobs for a single training invocation."""

    task: Task
    test_size: float = 0.2
    cv_folds: int = 5
    random_state: int = 42
    selected_models: list[str] | None = None  # None → all available
    primary_metric: str | None = None  # None → primary_metric_name(task)
    cross_validate_best: bool = True
    track_mlflow: bool = False
    mlflow_experiment: str = "iad_default"
    notes: str = ""
    feature_selection: bool = False
    max_features: int | None = None

    def primary(self) -> str:
        return self.primary_metric or primary_metric_name(self.task)


def _validate_task_target(task: Task, target: str, y: pd.Series) -> None:
    """Reject regression on text/category targets (common UI mis-inference)."""
    non_numeric = (
        pd.api.types.is_object_dtype(y)
        or pd.api.types.is_string_dtype(y)
        or pd.api.types.is_bool_dtype(y)
        or str(y.dtype).startswith("category")
    )
    if task == "regression" and non_numeric:
        raise TrainingError(
            f"column {target!r} is non-numeric; use classification",
            user_message=(
                f"Target column «{target}» contains text or category labels. "
                "Select **classification** as the task type, or choose a numeric target."
            ),
        )


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------
class TrainingService:
    """Train + benchmark a leaderboard of models on a preprocessing pipeline."""

    def __init__(
        self,
        registry: ModelRegistry | None = None,
    ) -> None:
        self.registry = registry or ModelRegistry.default()

    # ------------------------------------------------------------------
    def train(
        self,
        df: pd.DataFrame,
        *,
        target: str,
        config: TrainingConfig,
        feature_columns: Iterable[str] | None = None,
        preprocessor: Pipeline | None = None,
    ) -> TrainingResult:
        """Run training and produce a :class:`TrainingResult`.

        Args:
            df: input DataFrame including ``target``.
            target: name of the target column.
            config: :class:`TrainingConfig` instance.
            feature_columns: when ``None`` use every non-target column.
            preprocessor: optional preprocessor pipeline. When ``None`` we
                build one via ``build_auto_preprocessor``.

        Returns:
            :class:`TrainingResult` with leaderboard, best pipeline, model
            card and the diagnostics required by the UI / API.
        """
        # ---- 1. Input validation -----------------------------------------
        validate_dataframe(df, min_rows=20, min_cols=2)
        validate_target_column(df, target)
        if feature_columns is None:
            feature_columns = [c for c in df.columns if c != target]
        else:
            feature_columns = list(feature_columns)
            validate_columns_present(df, feature_columns + [target])
        if not feature_columns:
            raise TrainingError("at least one feature column is required")

        SeedManager.set_global_seed(config.random_state)
        settings = get_settings()
        primary_metric = config.primary()

        logger.info(
            "training started",
            extra={
                "ctx_target": target,
                "ctx_task": config.task,
                "ctx_n_features": len(feature_columns),
                "ctx_primary_metric": primary_metric,
                "ctx_random_state": config.random_state,
                "ctx_iad_version": IAD_VERSION,
            },
        )

        # ---- 2. Train/test split -----------------------------------------
        data = df[list(feature_columns) + [target]].dropna(subset=[target]).copy()
        X = data[feature_columns]
        y = data[target]
        _validate_task_target(config.task, target, y)
        stratify = y if (config.task == "classification" and y.nunique(dropna=True) > 1) else None
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=config.test_size,
            random_state=config.random_state,
            stratify=stratify,
        )

        # ---- 3. Preprocessor ----------------------------------------------
        if preprocessor is None:
            preprocessor, schema_groups = build_auto_preprocessor(
                X_train,
                target=None,
                task=config.task,
                feature_selection=config.feature_selection,
                max_features=config.max_features,
            )
        else:
            schema_groups = {
                "numeric": X_train.select_dtypes(include=["number"]).columns.tolist(),
                "categorical": [c for c in X_train.columns if c not in X_train.select_dtypes(include=["number"]).columns],
                "datetime": [c for c in X_train.columns if pd.api.types.is_datetime64_any_dtype(X_train[c])],
                "target_encoded": [],
            }

        # ---- 4. Iterate over candidate models -----------------------------
        candidates = self._select_specs(config)
        leaderboard: list[LeaderboardEntry] = []
        best_pipeline: Pipeline | None = None
        best_entry: LeaderboardEntry | None = None
        best_score = -np.inf

        for spec in candidates:
            entry, pipeline = self._train_single(
                spec=spec,
                preprocessor=preprocessor,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                config=config,
                primary_metric=primary_metric,
            )
            leaderboard.append(entry)
            if entry.error is not None or pipeline is None:
                continue
            score = entry.metrics.get(primary_metric, float("-inf"))
            if score > best_score:
                best_score = score
                best_pipeline = pipeline
                best_entry = entry

        if best_pipeline is None or best_entry is None:
            raise TrainingError(
                "no model could be trained successfully",
                user_message="None of the candidate models trained successfully on this dataset.",
                leaderboard=[e.model_name for e in leaderboard],
            )

        leaderboard.sort(
            key=lambda e: e.metrics.get(primary_metric, float("-inf")),
            reverse=True,
        )

        # ---- 5. Cross-validation for the winner ---------------------------
        cv_metrics: dict[str, float] = {}
        if config.cross_validate_best:
            try:
                scores = cross_val_score(
                    best_pipeline,
                    X,
                    y,
                    cv=config.cv_folds,
                    scoring=scoring_for(config.task),
                    n_jobs=1,
                )
                cv_metrics = {
                    f"cv_{primary_metric}_mean": round(float(scores.mean()), 4),
                    f"cv_{primary_metric}_std": round(float(scores.std()), 4),
                }
            except Exception as exc:  # pragma: no cover — informational
                logger.warning("cross-validation failed: %s", exc)

        # Refresh the best entry to include CV metrics.
        best_entry = LeaderboardEntry(
            model_name=best_entry.model_name,
            family=best_entry.family,
            metrics=best_entry.metrics,
            cv_metrics=cv_metrics,
            train_time_seconds=best_entry.train_time_seconds,
            error=None,
        )

        # ---- 6. Diagnostics -----------------------------------------------
        y_pred = best_pipeline.predict(X_test)
        confusion = None
        regression_rep = None
        calibration = None
        test_predictions = X_test.copy()
        test_predictions["actual"] = y_test.values
        test_predictions["predicted"] = y_pred

        if config.task == "classification":
            confusion = build_confusion_matrix_report(y_test, y_pred)
            try:
                if hasattr(best_pipeline.named_steps["model"], "predict_proba"):
                    proba = best_pipeline.predict_proba(X_test)
                    if proba.shape[1] == 2:
                        calibration = build_calibration_report(y_test, proba[:, 1])
            except Exception as exc:  # pragma: no cover — defensive
                logger.debug("calibration skipped: %s", exc)
        else:
            regression_rep = build_regression_report(y_test, y_pred)

        # ---- 7. Feature importance ----------------------------------------
        importance = self._extract_feature_importance(best_pipeline, list(feature_columns))

        # ---- 8. Model card ------------------------------------------------
        card = ModelCard(
            name=best_entry.model_name,
            task=config.task,
            target=target,
            features=list(feature_columns),
            schema_groups=schema_groups,
            metrics=best_entry.metrics,
            cv_metrics=cv_metrics,
            best_params=self._extract_estimator_params(best_pipeline),
            seed=config.random_state,
            dataset_fingerprint=fingerprint_dataframe(data),
            n_rows=int(data.shape[0]),
            n_columns=int(data.shape[1]),
            iad_version=IAD_VERSION,
            environment=capture_environment(),
            train_time_seconds=best_entry.train_time_seconds,
            notes=config.notes,
        )

        # ---- 9. Optional MLflow ------------------------------------------
        run_id: str | None = None
        if config.track_mlflow:
            try:
                from iad.ml.tracking.mlflow_tracker import MLflowTracker

                with MLflowTracker(experiment=config.mlflow_experiment) as tracker:
                    tracker.log_params({"primary_metric": primary_metric, **card.best_params})
                    tracker.log_metrics({**card.metrics, **card.cv_metrics})
                    tracker.log_pipeline(best_pipeline, name="best_model")
                    tracker.log_model_card(card)
                    run_id = tracker.run_id
            except Exception as exc:  # pragma: no cover — best effort
                logger.warning("MLflow tracking failed: %s", exc)
        if run_id is not None:
            card = ModelCard(
                **{**card.to_dict(), "mlflow_run_id": run_id, "environment": card.environment},
            )

        result = TrainingResult(
            task=config.task,
            target=target,
            features=list(feature_columns),
            schema_groups=schema_groups,
            leaderboard=leaderboard,
            best_pipeline=best_pipeline,
            best_entry=best_entry,
            test_predictions=test_predictions,
            confusion_matrix=confusion,
            regression_report=regression_rep,
            calibration=calibration,
            feature_importance=importance,
            model_card=card,
            extra={"settings_version": settings.APP_VERSION},
        )
        logger.info(
            "training complete",
            extra={
                "ctx_best_model": best_entry.model_name,
                "ctx_best_score": best_score,
                "ctx_cv_metrics": cv_metrics,
            },
        )
        return result

    # ==================================================================
    # Private helpers
    # ==================================================================
    def _select_specs(self, config: TrainingConfig) -> list[ModelSpec]:
        candidates = self.registry.all_for(config.task)
        if config.selected_models:
            keep = set(config.selected_models)
            candidates = [s for s in candidates if s.name in keep]
            if not candidates:
                raise TrainingError(
                    "selected_models did not match any registered model",
                    selected=list(keep),
                    available=self.registry.names(config.task),
                )
        return candidates

    def _train_single(
        self,
        *,
        spec: ModelSpec,
        preprocessor: Pipeline,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        config: TrainingConfig,
        primary_metric: str,
    ) -> tuple[LeaderboardEntry, Pipeline | None]:
        started = time.perf_counter()
        try:
            estimator = spec.factory()
            # Clone preprocessor so later candidates do not refit a shared instance
            # (which would invalidate an earlier champion pipeline).
            pipeline = Pipeline(
                steps=[("preprocessor", clone(preprocessor)), ("model", estimator)]
            )
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            if config.task == "classification":
                proba = (
                    pipeline.predict_proba(X_test)
                    if hasattr(estimator, "predict_proba")
                    else None
                )
                metrics = classification_metrics(y_test, y_pred, y_proba=proba)
            else:
                metrics = regression_metrics(y_test, y_pred)
            elapsed = time.perf_counter() - started
            entry = LeaderboardEntry(
                model_name=spec.name,
                family=spec.family,
                metrics=metrics,
                cv_metrics={},
                train_time_seconds=round(elapsed, 4),
            )
            logger.info(
                "model trained",
                extra={
                    "ctx_model": spec.name,
                    "ctx_score": metrics.get(primary_metric),
                    "ctx_train_time_s": round(elapsed, 3),
                },
            )
            return entry, pipeline
        except Exception as exc:
            logger.warning("model %s failed to train: %s", spec.name, exc)
            return (
                LeaderboardEntry(
                    model_name=spec.name,
                    family=spec.family,
                    metrics={},
                    cv_metrics={},
                    train_time_seconds=round(time.perf_counter() - started, 4),
                    error=str(exc),
                ),
                None,
            )

    def _extract_feature_importance(
        self, pipeline: Pipeline, feature_columns: list[str]
    ) -> pd.DataFrame | None:
        try:
            model = pipeline.named_steps["model"]
            if hasattr(model, "feature_importances_"):
                values = np.asarray(model.feature_importances_, dtype=float)
            elif hasattr(model, "coef_"):
                coef = np.asarray(model.coef_, dtype=float)
                values = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
            else:
                return None

            names = self._resolve_feature_names(pipeline, feature_columns, len(values))
            if len(values) != len(names):
                logger.debug(
                    "feature_importance length mismatch: %d values vs %d names",
                    len(values),
                    len(names),
                )
                return None
            return (
                pd.DataFrame({"feature": names, "importance": values})
                .sort_values("importance", ascending=False)
                .reset_index(drop=True)
            )
        except Exception as exc:  # pragma: no cover
            logger.debug("feature importance unavailable: %s", exc)
            return None

    @staticmethod
    def _resolve_feature_names(
        pipeline: Pipeline, feature_columns: list[str], expected_length: int
    ) -> list[str]:
        """Best-effort resolution of feature names matching the model's coefficients."""
        preprocessor = pipeline.named_steps.get("preprocessor")
        # 1) Modern sklearn: ask the preprocessor with the original input names.
        if preprocessor is not None:
            for attempt in (lambda: preprocessor.get_feature_names_out(feature_columns),
                            lambda: preprocessor.get_feature_names_out()):
                try:
                    names = list(attempt())
                    if len(names) == expected_length:
                        return [str(n) for n in names]
                except Exception:
                    continue
        # 2) Some estimators expose feature_names_in_ themselves.
        model = pipeline.named_steps.get("model")
        if model is not None and hasattr(model, "feature_names_in_"):
            names = list(model.feature_names_in_)
            if len(names) == expected_length:
                return [str(n) for n in names]
        # 3) Fall back to the raw input feature columns when the count matches.
        if len(feature_columns) == expected_length:
            return list(feature_columns)
        # 4) Generic placeholders so the importance table still renders.
        return [f"feature_{i}" for i in range(expected_length)]

    def _extract_estimator_params(self, pipeline: Pipeline) -> dict[str, object]:
        try:
            params = pipeline.named_steps["model"].get_params(deep=False)
            return {k: v for k, v in params.items() if isinstance(v, (int, float, str, bool, type(None)))}
        except Exception:
            return {}
