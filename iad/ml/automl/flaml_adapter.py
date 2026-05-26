"""FLAML backend — Microsoft's lightweight AutoML.

Why FLAML as the default backend?
    * Pure Python, ~30MB install, no compiled deps beyond what we already
      have (numpy, lightgbm, xgboost). Plays nicely with our Phase 2
      preprocessing layer because FLAML accepts pre-engineered features.
    * Time-budget driven (optimises within ``time_budget`` seconds), which
      matches a SaaS UX: "I'll wait 5 minutes for the best model".
    * Ships its own tuners (BlendSearch / CFO) — we don't need to wire
      Optuna for the AutoML path.
"""
from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Literal

import pandas as pd

from iad.core.exceptions import TrainingError
from iad.core.logging import get_logger
from iad.ml.automl.base import AutoMLBackend, AutoMLResult

logger = get_logger("iad.training")


def flaml_available() -> bool:
    try:
        import flaml  # type: ignore[import-not-found]  # noqa: F401
        return True
    except ImportError:
        return False


_FLAML_DEFAULT_METRIC = {
    "classification": "macro_f1",
    "regression": "r2",
}


class FLAMLBackend(AutoMLBackend):
    name = "flaml"

    def __init__(self, *, estimator_list: list[str] | None = None) -> None:
        self.estimator_list = estimator_list

    @classmethod
    def is_available(cls) -> bool:
        return flaml_available()

    def fit(
        self,
        X: pd.DataFrame,
        y: Iterable,
        *,
        task: Literal["classification", "regression"] = "classification",
        time_budget: int = 60,
        metric: str | None = None,
    ) -> AutoMLResult:
        if not self.is_available():
            raise TrainingError(
                "flaml is not installed",
                user_message="FLAML AutoML backend is not installed (`pip install flaml`).",
            )
        from flaml import AutoML  # type: ignore[import-not-found]

        scoring = metric or _FLAML_DEFAULT_METRIC[task]
        automl = AutoML()
        kwargs = {
            "X_train": X,
            "y_train": pd.Series(y),
            "task": task,
            "time_budget": time_budget,
            "metric": scoring,
            "estimator_list": self.estimator_list,
            "log_file_name": None,
            "verbose": 0,
            "seed": 42,
        }
        # Drop None / falsy keys to avoid FLAML deprecation warnings.
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        started = time.perf_counter()
        logger.info(
            "flaml automl starting",
            extra={"ctx_task": task, "ctx_time_budget_s": time_budget, "ctx_metric": scoring},
        )
        automl.fit(**kwargs)
        elapsed = time.perf_counter() - started

        leaderboard = self._build_leaderboard(automl)
        result = AutoMLResult(
            backend=self.name,
            task=task,
            best_model=automl.model.estimator if automl.model else automl,
            best_metric_name=scoring,
            best_score=float(getattr(automl, "best_loss", 0.0)) * (-1 if scoring.startswith("neg_") else 1),
            leaderboard=leaderboard,
            elapsed_seconds=round(elapsed, 3),
            extra={
                "best_estimator": automl.best_estimator,
                "best_config": automl.best_config,
            },
        )
        logger.info(
            "flaml automl done",
            extra={
                "ctx_elapsed_s": round(elapsed, 3),
                "ctx_best_estimator": automl.best_estimator,
                "ctx_best_config": automl.best_config,
            },
        )
        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _build_leaderboard(automl) -> pd.DataFrame:
        rows = []
        try:
            best_models = automl.best_iteration_per_estimator
        except Exception:
            best_models = {}
        try:
            estimator_records = automl.estimator_list  # type: ignore[attr-defined]
        except Exception:
            estimator_records = []
        if isinstance(estimator_records, list):
            for estimator in estimator_records:
                rows.append(
                    {
                        "estimator": estimator,
                        "best_iteration": best_models.get(estimator),
                    }
                )
        if not rows:
            rows.append({"estimator": automl.best_estimator})
        return pd.DataFrame(rows)
