"""PyCaret AutoML adapter — optional install path.

Why optional and not default?
    PyCaret has a large transitive dependency surface (~1GB after deps) and
    is opinionated about its own preprocessing pipeline, which would
    duplicate work already done in :mod:`iad.ml.preprocessing`. We wrap it
    behind the same ``AutoMLBackend`` ABC so users who already run PyCaret
    in their team can switch backends without touching the calling code.

    To enable:  ``pip install pycaret``.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import pandas as pd

from iad.core.exceptions import TrainingError
from iad.core.logging import get_logger
from iad.ml.automl.base import AutoMLBackend, AutoMLResult

logger = get_logger("iad.training")


def pycaret_available() -> bool:
    try:
        import pycaret  # type: ignore[import-not-found]  # noqa: F401
        return True
    except ImportError:
        return False


class PyCaretBackend(AutoMLBackend):
    """Thin adapter around ``pycaret.classification`` / ``pycaret.regression``."""

    name = "pycaret"

    @classmethod
    def is_available(cls) -> bool:
        return pycaret_available()

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
                "pycaret is not installed",
                user_message=(
                    "PyCaret AutoML backend is not installed. "
                    "Run `pip install pycaret` to enable it."
                ),
            )

        if task == "classification":
            from pycaret import classification as pyc  # type: ignore[import-not-found]
        else:
            from pycaret import regression as pyc  # type: ignore[import-not-found]

        data = X.copy()
        data["__target__"] = pd.Series(y).values
        pyc.setup(
            data=data,
            target="__target__",
            session_id=42,
            verbose=False,
            html=False,
            log_experiment=False,
        )
        compare = pyc.compare_models(
            n_select=3, sort=metric or ("F1" if task == "classification" else "R2")
        )
        best_model = compare[0] if isinstance(compare, list) else compare
        leaderboard = pyc.pull()
        finalised = pyc.finalize_model(best_model)
        score = float(leaderboard.iloc[0].get(metric or ("F1" if task == "classification" else "R2"), 0.0))

        logger.info(
            "pycaret automl done",
            extra={"ctx_task": task, "ctx_best_estimator": type(finalised).__name__},
        )
        return AutoMLResult(
            backend=self.name,
            task=task,
            best_model=finalised,
            best_metric_name=metric or ("F1" if task == "classification" else "R2"),
            best_score=score,
            leaderboard=leaderboard,
            elapsed_seconds=0.0,
        )
