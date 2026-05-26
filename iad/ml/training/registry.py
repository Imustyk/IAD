"""Pluggable model registry for the training service.

Adding a new estimator to the platform = registering a ``ModelSpec``. Optional
heavy libraries (XGBoost, LightGBM, CatBoost) are imported lazily and skipped
gracefully when not installed — the platform stays usable on a minimal env
(sklearn-only) and unlocks more models as additional packages are added.

Why a registry instead of two hardcoded dicts (the legacy ``CLASSIFIERS`` /
``REGRESSORS``)?
    * Each spec carries metadata (family, supports_proba, optional, native
      categorical support) that the leaderboard, the tuning module and the
      explainability module all consume.
    * The Optuna search-space helpers in ``iad.ml.tuning`` look up specs by
      name; the registry is a single source of truth.
    * Phase 4's UI can introspect ``ModelRegistry.default()`` to render
      checkboxes, badges ("XGBoost — optional"), and per-model docs.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from iad.core.logging import get_logger

logger = get_logger("iad.ml.training.registry")


Task = Literal["classification", "regression"]


@dataclass(frozen=True)
class ModelSpec:
    """Metadata + factory for a candidate estimator."""

    name: str
    task: Task
    family: str  # "linear" | "tree" | "ensemble" | "boost" | "neighbours"
    factory: Callable[..., BaseEstimator]
    optional: bool = False  # True if the underlying library may be absent
    supports_proba: bool = True
    supports_categorical_native: bool = False
    description: str = ""
    default_kwargs: dict[str, Any] = field(default_factory=dict)


class ModelRegistry:
    """Mutable registry of :class:`ModelSpec`."""

    def __init__(self) -> None:
        self._specs: dict[tuple[Task, str], ModelSpec] = {}

    # ------------------------------------------------------------------
    def register(self, spec: ModelSpec) -> None:
        key = (spec.task, spec.name)
        if key in self._specs:
            raise ValueError(f"model already registered: {key}")
        self._specs[key] = spec
        logger.debug("registered model %s/%s", spec.task, spec.name)

    def unregister(self, task: Task, name: str) -> None:
        self._specs.pop((task, name), None)

    def get(self, task: Task, name: str) -> ModelSpec:
        try:
            return self._specs[(task, name)]
        except KeyError as exc:
            raise KeyError(f"unknown model {name!r} for task {task!r}") from exc

    def names(self, task: Task) -> list[str]:
        return [n for (t, n) in self._specs.keys() if t == task]

    def all_for(self, task: Task) -> list[ModelSpec]:
        return [s for (t, _), s in self._specs.items() if t == task]

    def __len__(self) -> int:
        return len(self._specs)

    def __contains__(self, key: tuple[Task, str]) -> bool:
        return key in self._specs

    # ------------------------------------------------------------------
    @classmethod
    def default(cls) -> ModelRegistry:
        """Build the registry with every spec known to the platform.

        Optional libraries (XGBoost, LightGBM, CatBoost) are tried lazily; if
        they cannot be imported the corresponding spec is skipped and a
        ``debug`` log line is emitted.
        """
        registry = cls()
        for spec in _BUILTIN_SPECS:
            registry.register(spec)
        for loader in _OPTIONAL_LOADERS:
            try:
                for spec in loader():
                    registry.register(spec)
            except ImportError as exc:
                logger.debug("skipping optional model loader: %s", exc)
        logger.info("model registry built", extra={"ctx_n_models": len(registry)})
        return registry


# ---------------------------------------------------------------------------
# Built-in (sklearn-only) specs
# ---------------------------------------------------------------------------
def _merge(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """User-supplied kwargs override registered defaults — that's the contract
    Optuna's search expects, and what every sklearn estimator's __init__ accepts.
    """
    merged = dict(defaults)
    merged.update(overrides)
    return merged


_BUILTIN_SPECS: tuple[ModelSpec, ...] = (
    # ---- Classification ----
    ModelSpec(
        name="Logistic Regression",
        task="classification",
        family="linear",
        factory=lambda **kw: LogisticRegression(**_merge({"max_iter": 2000}, kw)),
        description="L2-regularised logistic regression. Fast baseline.",
    ),
    ModelSpec(
        name="Random Forest",
        task="classification",
        family="ensemble",
        factory=lambda **kw: RandomForestClassifier(
            **_merge({"n_estimators": 200, "random_state": 42, "n_jobs": -1}, kw)
        ),
        description="Bagged decision trees. Robust default.",
    ),
    ModelSpec(
        name="Gradient Boosting",
        task="classification",
        family="boost",
        factory=lambda **kw: GradientBoostingClassifier(
            **_merge({"random_state": 42}, kw)
        ),
        description="Sklearn gradient boosting (slow but well-tested).",
    ),
    ModelSpec(
        name="Hist Gradient Boosting",
        task="classification",
        family="boost",
        factory=lambda **kw: HistGradientBoostingClassifier(
            **_merge({"random_state": 42}, kw)
        ),
        description="Histogram-based GBM, native NaN handling.",
    ),
    ModelSpec(
        name="Extra Trees",
        task="classification",
        family="ensemble",
        factory=lambda **kw: ExtraTreesClassifier(
            **_merge({"n_estimators": 200, "random_state": 42, "n_jobs": -1}, kw)
        ),
        description="Extremely Randomised Trees. Reduces variance.",
    ),
    ModelSpec(
        name="Decision Tree",
        task="classification",
        family="tree",
        factory=lambda **kw: DecisionTreeClassifier(**_merge({"random_state": 42}, kw)),
        description="Single decision tree. Most interpretable baseline.",
    ),
    ModelSpec(
        name="K-Nearest Neighbors",
        task="classification",
        family="neighbours",
        factory=lambda **kw: KNeighborsClassifier(**kw),
        description="Distance-based; sensitive to scaling.",
    ),
    # ---- Regression ----
    ModelSpec(
        name="Linear Regression",
        task="regression",
        family="linear",
        factory=lambda **kw: LinearRegression(**kw),
        description="Ordinary least squares baseline.",
        supports_proba=False,
    ),
    ModelSpec(
        name="Ridge Regression",
        task="regression",
        family="linear",
        factory=lambda **kw: Ridge(**_merge({"alpha": 1.0}, kw)),
        description="L2-regularised linear regression.",
        supports_proba=False,
    ),
    ModelSpec(
        name="ElasticNet",
        task="regression",
        family="linear",
        factory=lambda **kw: ElasticNet(
            **_merge({"alpha": 0.001, "l1_ratio": 0.5, "max_iter": 5000}, kw)
        ),
        description="L1+L2 regularised regression. Good for sparse signals.",
        supports_proba=False,
    ),
    ModelSpec(
        name="Random Forest",
        task="regression",
        family="ensemble",
        factory=lambda **kw: RandomForestRegressor(
            **_merge({"n_estimators": 200, "random_state": 42, "n_jobs": -1}, kw)
        ),
        description="Bagged decision trees.",
        supports_proba=False,
    ),
    ModelSpec(
        name="Extra Trees",
        task="regression",
        family="ensemble",
        factory=lambda **kw: ExtraTreesRegressor(
            **_merge({"n_estimators": 200, "random_state": 42, "n_jobs": -1}, kw)
        ),
        description="Extremely Randomised Trees.",
        supports_proba=False,
    ),
    ModelSpec(
        name="Gradient Boosting",
        task="regression",
        family="boost",
        factory=lambda **kw: GradientBoostingRegressor(**_merge({"random_state": 42}, kw)),
        description="Sklearn gradient boosting.",
        supports_proba=False,
    ),
    ModelSpec(
        name="Hist Gradient Boosting",
        task="regression",
        family="boost",
        factory=lambda **kw: HistGradientBoostingRegressor(**_merge({"random_state": 42}, kw)),
        description="Histogram-based GBM.",
        supports_proba=False,
    ),
    ModelSpec(
        name="Decision Tree",
        task="regression",
        family="tree",
        factory=lambda **kw: DecisionTreeRegressor(**_merge({"random_state": 42}, kw)),
        description="Single decision tree.",
        supports_proba=False,
    ),
    ModelSpec(
        name="K-Nearest Neighbors",
        task="regression",
        family="neighbours",
        factory=lambda **kw: KNeighborsRegressor(**kw),
        description="Distance-based.",
        supports_proba=False,
    ),
)


# ---------------------------------------------------------------------------
# Optional loaders — each returns a tuple of specs or raises ImportError.
# ---------------------------------------------------------------------------
def _load_xgboost_specs() -> tuple[ModelSpec, ...]:
    from xgboost import XGBClassifier, XGBRegressor  # type: ignore[import-not-found]

    xgb_defaults = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.1,
        "random_state": 42,
        "tree_method": "hist",
    }
    return (
        ModelSpec(
            name="XGBoost",
            task="classification",
            family="boost",
            factory=lambda **kw: XGBClassifier(
                **_merge({**xgb_defaults, "eval_metric": "mlogloss"}, kw)
            ),
            optional=True,
            description="Industry-standard gradient boosting (XGBoost).",
        ),
        ModelSpec(
            name="XGBoost",
            task="regression",
            family="boost",
            factory=lambda **kw: XGBRegressor(**_merge(xgb_defaults, kw)),
            optional=True,
            supports_proba=False,
            description="Industry-standard gradient boosting (XGBoost).",
        ),
    )


def _load_lightgbm_specs() -> tuple[ModelSpec, ...]:
    from lightgbm import LGBMClassifier, LGBMRegressor  # type: ignore[import-not-found]

    lgbm_defaults = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    return (
        ModelSpec(
            name="LightGBM",
            task="classification",
            family="boost",
            factory=lambda **kw: LGBMClassifier(
                **_merge({**lgbm_defaults, "max_depth": -1}, kw)
            ),
            optional=True,
            supports_categorical_native=True,
            description="Microsoft LightGBM. Fast, native categorical handling.",
        ),
        ModelSpec(
            name="LightGBM",
            task="regression",
            family="boost",
            factory=lambda **kw: LGBMRegressor(**_merge(lgbm_defaults, kw)),
            optional=True,
            supports_proba=False,
            supports_categorical_native=True,
            description="Microsoft LightGBM.",
        ),
    )


def _load_catboost_specs() -> tuple[ModelSpec, ...]:
    from catboost import CatBoostClassifier, CatBoostRegressor  # type: ignore[import-not-found]

    cat_defaults = {
        "iterations": 300,
        "depth": 6,
        "learning_rate": 0.05,
        "random_seed": 42,
        "verbose": False,
    }
    return (
        ModelSpec(
            name="CatBoost",
            task="classification",
            family="boost",
            factory=lambda **kw: CatBoostClassifier(**_merge(cat_defaults, kw)),
            optional=True,
            supports_categorical_native=True,
            description="Yandex CatBoost. Strong on small to mid-sized data.",
        ),
        ModelSpec(
            name="CatBoost",
            task="regression",
            family="boost",
            factory=lambda **kw: CatBoostRegressor(**_merge(cat_defaults, kw)),
            optional=True,
            supports_proba=False,
            supports_categorical_native=True,
            description="Yandex CatBoost.",
        ),
    )


_OPTIONAL_LOADERS: tuple[Callable[[], tuple[ModelSpec, ...]], ...] = (
    _load_xgboost_specs,
    _load_lightgbm_specs,
    _load_catboost_specs,
)
