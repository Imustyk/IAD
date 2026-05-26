"""AutoML abstraction — pluggable backends behind one interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd


@dataclass(frozen=True)
class AutoMLResult:
    """Output of every backend's ``fit`` method."""

    backend: str
    task: Literal["classification", "regression"]
    best_model: Any
    best_metric_name: str
    best_score: float
    leaderboard: pd.DataFrame = field(default_factory=pd.DataFrame)
    elapsed_seconds: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


class AutoMLBackend(ABC):
    """Abstract AutoML backend.

    Subclasses ``FLAMLBackend``, ``PyCaretBackend`` (and a future
    ``AutoGluonBackend``) implement :meth:`fit`. Streamlit / FastAPI keep
    one code path that consumes ``AutoMLResult`` regardless of which engine
    actually trained the model.
    """

    name: str = "abstract"

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: Iterable,
        *,
        task: Literal["classification", "regression"] = "classification",
        time_budget: int = 60,
        metric: str | None = None,
    ) -> AutoMLResult: ...

    @classmethod
    def is_available(cls) -> bool:
        return False
