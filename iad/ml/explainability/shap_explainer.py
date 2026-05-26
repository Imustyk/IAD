"""SHAP explainer for sklearn pipelines.

Routing strategy (chosen automatically):

* **TreeExplainer** — for tree ensembles (RF, ExtraTrees, GBM, HistGB,
  XGBoost, LightGBM, CatBoost, DecisionTree). O(N · M · TreeDepth) — fast.
* **LinearExplainer** — for linear models (LogisticRegression, Ridge,
  ElasticNet, LinearRegression).
* **KernelExplainer** — universal but slow O(N · 2^M); used as the fallback.
  Sample size is capped to keep latency reasonable.

Public surface:

* :meth:`SHAPExplainer.from_pipeline` — preferred constructor.
* :meth:`global_importance`           — mean(|SHAP|) per (transformed) feature.
* :meth:`explain_row`                 — :class:`SHAPExplanation` for one row.
* :meth:`waterfall_data`              — DataFrame ready for a Plotly waterfall.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from iad.core.logging import get_logger

logger = get_logger("iad.ml.explainability.shap")


_TREE_FAMILIES = {
    "RandomForestClassifier", "RandomForestRegressor",
    "ExtraTreesClassifier", "ExtraTreesRegressor",
    "GradientBoostingClassifier", "GradientBoostingRegressor",
    "HistGradientBoostingClassifier", "HistGradientBoostingRegressor",
    "DecisionTreeClassifier", "DecisionTreeRegressor",
    "XGBClassifier", "XGBRegressor",
    "LGBMClassifier", "LGBMRegressor",
    "CatBoostClassifier", "CatBoostRegressor",
}
_LINEAR_FAMILIES = {
    "LogisticRegression", "Ridge", "Lasso", "ElasticNet", "LinearRegression",
}


@dataclass(frozen=True)
class SHAPExplanation:
    """Numeric SHAP output for a single observation."""

    feature_names: list[str]
    shap_values: np.ndarray  # shape (n_features,) or (n_classes, n_features)
    base_value: float | np.ndarray
    feature_values: np.ndarray
    is_classifier: bool
    predicted_class: Any | None = None

    def top_contributions(self, n: int = 10) -> pd.DataFrame:
        if self.shap_values.ndim == 2:
            arr = self.shap_values[0]  # arbitrary first class for ranking
        else:
            arr = self.shap_values
        idx = np.argsort(np.abs(arr))[::-1][:n]
        return pd.DataFrame(
            {
                "feature": [self.feature_names[i] for i in idx],
                "value": [self.feature_values[i] for i in idx],
                "shap": [arr[i] for i in idx],
            }
        )


class SHAPExplainer:
    """Wraps a SHAP explainer chosen automatically from the pipeline's model.

    Args:
        pipeline: a fitted sklearn pipeline ending with a model step.
        background: a representative sample of training data; required for
            KernelExplainer / LinearExplainer.
        max_background_rows: cap to keep KernelExplainer responsive.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        background: pd.DataFrame,
        *,
        max_background_rows: int = 100,
    ) -> None:
        try:
            import shap  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover — dep guard
            raise ImportError(
                "shap is not installed. `pip install shap` to enable explainability."
            ) from exc

        self._shap = shap
        self.pipeline = pipeline
        self._preprocessor = pipeline[:-1] if len(pipeline.steps) > 1 else None
        self._model = pipeline.named_steps.get("model", pipeline.steps[-1][1])

        if background is None or background.empty:
            raise ValueError("background DataFrame must not be empty")
        if len(background) > max_background_rows:
            background = background.sample(n=max_background_rows, random_state=42)
        self.background = background

        self._transformed_background = self._transform(background)
        self._feature_names = self._build_feature_names()
        self._strategy: Literal["tree", "linear", "kernel"] = self._choose_strategy()
        self._explainer = self._build_explainer()
        logger.info(
            "shap explainer ready",
            extra={
                "ctx_strategy": self._strategy,
                "ctx_n_features": len(self._feature_names),
                "ctx_background_rows": len(background),
            },
        )

    # ------------------------------------------------------------------
    @classmethod
    def from_pipeline(
        cls,
        pipeline: Pipeline,
        sample: pd.DataFrame,
        *,
        max_background_rows: int = 100,
    ) -> SHAPExplainer:
        return cls(pipeline, background=sample, max_background_rows=max_background_rows)

    # ------------------------------------------------------------------
    @property
    def is_classifier(self) -> bool:
        return hasattr(self._model, "predict_proba")

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    @property
    def strategy(self) -> str:
        return self._strategy

    # ------------------------------------------------------------------
    def global_importance(self, sample: pd.DataFrame | None = None) -> pd.DataFrame:
        """Mean(|SHAP|) per feature, sorted by importance."""
        data = sample if sample is not None else self.background
        transformed = self._transform(data)
        values = self._raw_shap_values(transformed)
        # values may be a list (per class) or array. Aggregate to a 2D matrix.
        matrix = self._values_to_matrix(values)
        importance = np.mean(np.abs(matrix), axis=0)
        out = pd.DataFrame({"feature": self._feature_names, "mean_abs_shap": importance})
        return out.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    def explain_row(self, row: pd.Series | pd.DataFrame) -> SHAPExplanation:
        if isinstance(row, pd.Series):
            row = row.to_frame().T
        transformed = self._transform(row)
        values = self._raw_shap_values(transformed)
        if isinstance(values, list):  # per-class list (older shap)
            stacked = np.stack([v[0] for v in values], axis=0)
            base_value = np.asarray([self._explainer_expected_value(i) for i in range(len(values))])
            shap_arr = stacked
        elif values.ndim == 3:  # (n_samples, n_features, n_classes)
            shap_arr = np.transpose(values[0], (1, 0))  # (n_classes, n_features)
            base_value = self._explainer_expected_value()
        else:
            shap_arr = values[0]
            base_value = self._explainer_expected_value()

        predicted_class: Any | None = None
        if self.is_classifier:
            try:
                proba = self.pipeline.predict_proba(row)
                predicted_class = self._model.classes_[int(np.argmax(proba, axis=1)[0])]
            except Exception:  # pragma: no cover
                predicted_class = None

        return SHAPExplanation(
            feature_names=list(self._feature_names),
            shap_values=np.asarray(shap_arr),
            base_value=base_value,
            feature_values=np.asarray(transformed[0]),
            is_classifier=self.is_classifier,
            predicted_class=predicted_class,
        )

    # ------------------------------------------------------------------
    def waterfall_data(self, row: pd.Series | pd.DataFrame, n_top: int = 10) -> pd.DataFrame:
        explanation = self.explain_row(row)
        df = explanation.top_contributions(n=n_top)
        df["sign"] = np.sign(df["shap"])
        return df

    # ==================================================================
    # Internals
    # ==================================================================
    def _transform(self, data: pd.DataFrame) -> np.ndarray:
        if self._preprocessor is None:
            return np.asarray(data.values, dtype=float)
        try:
            transformed = self._preprocessor.transform(data)
        except Exception as exc:
            raise ValueError(f"preprocessor.transform failed: {exc}") from exc
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        return np.asarray(transformed, dtype=float)

    def _build_feature_names(self) -> list[str]:
        expected = self._transformed_background.shape[1]
        if self._preprocessor is not None:
            for attempt in (
                lambda: self._preprocessor.get_feature_names_out(list(self.background.columns)),
                lambda: self._preprocessor.get_feature_names_out(),
            ):
                try:
                    names = list(attempt())
                    if len(names) == expected:
                        return [str(n) for n in names]
                except Exception:
                    continue
        return [f"f{i}" for i in range(expected)]

    def _choose_strategy(self) -> Literal["tree", "linear", "kernel"]:
        cls_name = type(self._model).__name__
        if cls_name in _TREE_FAMILIES:
            return "tree"
        if cls_name in _LINEAR_FAMILIES:
            return "linear"
        return "kernel"

    def _build_explainer(self):
        if self._strategy == "tree":
            try:
                return self._shap.TreeExplainer(self._model)
            except Exception as exc:  # pragma: no cover — fall back gracefully
                logger.debug("TreeExplainer failed, falling back to KernelExplainer: %s", exc)
                self._strategy = "kernel"
        if self._strategy == "linear":
            try:
                return self._shap.LinearExplainer(
                    self._model, self._transformed_background
                )
            except Exception as exc:  # pragma: no cover
                logger.debug("LinearExplainer failed, falling back to KernelExplainer: %s", exc)
                self._strategy = "kernel"
        # KernelExplainer — slowest but universal.
        # KernelExplainer expects raw (untransformed) input when given the
        # whole pipeline; we wrap so it operates on transformed rows.
        return self._shap.KernelExplainer(
            self._build_predict_fn(),
            self._transformed_background,
        )

    def _build_predict_fn(self):
        model = self._model
        if self.is_classifier:
            def fn(x: np.ndarray) -> np.ndarray:
                return model.predict_proba(x)
        else:
            def fn(x: np.ndarray) -> np.ndarray:
                return model.predict(x)
        return fn

    def _raw_shap_values(self, transformed: np.ndarray):
        try:
            return self._explainer.shap_values(transformed)
        except TypeError:
            # Some new shap APIs expect a different signature.
            return self._explainer(transformed).values  # pragma: no cover

    @staticmethod
    def _values_to_matrix(values) -> np.ndarray:
        if isinstance(values, list):
            return np.mean(np.abs(np.stack(values, axis=0)), axis=0)
        if isinstance(values, np.ndarray) and values.ndim == 3:
            return np.mean(np.abs(values), axis=2)
        return np.asarray(values)

    def _explainer_expected_value(self, index: int | None = None):
        ev = getattr(self._explainer, "expected_value", 0.0)
        if isinstance(ev, np.ndarray):
            if index is not None and index < len(ev):
                return float(ev[index])
            return float(ev[0]) if ev.size else 0.0
        if isinstance(ev, list):
            return float(ev[index or 0])
        return float(ev) if ev is not None else 0.0
