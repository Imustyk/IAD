"""Local LIME explanations for tabular models.

LIME perturbs around a single observation, fits a sparse linear surrogate
in the neighbourhood, and returns per-feature contributions. Slower per
explanation than SHAP TreeExplainer but model-agnostic.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class LIMEExplanation:
    """Local explanation for a single observation."""

    feature_names: list[str]
    contributions: list[tuple[str, float]]
    predicted_class: object | None = None
    score: float | None = None  # local fidelity / R²

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.contributions, columns=["feature", "weight"])


class LIMEExplainer:
    """Tabular LIME wrapper for sklearn pipelines.

    Args:
        pipeline: a fitted sklearn pipeline.
        background: training data for the perturbation distribution.
        mode: ``"classification"`` or ``"regression"``.
        feature_names: optional override; defaults to ``background.columns``.
        class_names: optional override for classifier class labels.
        random_state: seed for the LIME RNG.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        background: pd.DataFrame,
        *,
        mode: Literal["classification", "regression"] = "classification",
        feature_names: Iterable[str] | None = None,
        class_names: Iterable[str] | None = None,
        random_state: int = 42,
    ) -> None:
        try:
            from lime.lime_tabular import LimeTabularExplainer  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover — dep guard
            raise ImportError(
                "lime is not installed. `pip install lime` to enable LIME explanations."
            ) from exc

        self.pipeline = pipeline
        self.mode = mode
        self.feature_names = list(feature_names or background.columns)
        self.background = background
        self.class_names = list(class_names) if class_names is not None else None
        self._explainer = LimeTabularExplainer(
            training_data=background.values,
            mode=mode,
            feature_names=self.feature_names,
            class_names=self.class_names,
            discretize_continuous=False,
            random_state=random_state,
        )

    # ------------------------------------------------------------------
    def explain(
        self,
        row: pd.Series | pd.DataFrame,
        *,
        num_features: int = 10,
        num_samples: int = 1000,
        label: int | None = None,
    ) -> LIMEExplanation:
        """Explain a single observation."""
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        if not isinstance(row, pd.Series):
            raise TypeError("row must be a pandas Series or single-row DataFrame")
        x = row.reindex(self.background.columns).values.astype(float)

        if self.mode == "classification":
            predict_fn = self._proba_fn(row)
        else:
            predict_fn = self._predict_fn(row)

        explanation = self._explainer.explain_instance(
            data_row=x,
            predict_fn=predict_fn,
            num_features=num_features,
            num_samples=num_samples,
            labels=(label,) if label is not None else (1,) if self.mode == "classification" else None,
        )
        contributions = explanation.as_list(
            label=label if label is not None else (1 if self.mode == "classification" else None)
        )

        predicted_class: object | None = None
        score: float | None = None
        if self.mode == "classification":
            try:
                proba = self.pipeline.predict_proba(row.to_frame().T)
                predicted_class = (
                    self.class_names[int(np.argmax(proba, axis=1)[0])]
                    if self.class_names
                    else int(np.argmax(proba, axis=1)[0])
                )
            except Exception:  # pragma: no cover
                predicted_class = None
        try:
            score = float(explanation.score)
        except Exception:  # pragma: no cover
            score = None

        return LIMEExplanation(
            feature_names=list(self.feature_names),
            contributions=[(str(name), float(weight)) for name, weight in contributions],
            predicted_class=predicted_class,
            score=score,
        )

    # ------------------------------------------------------------------
    def _proba_fn(self, template_row: pd.Series):
        columns = list(self.background.columns)

        def fn(x_array: np.ndarray) -> np.ndarray:
            df = pd.DataFrame(x_array, columns=columns)
            return self.pipeline.predict_proba(df)

        return fn

    def _predict_fn(self, template_row: pd.Series):
        columns = list(self.background.columns)

        def fn(x_array: np.ndarray) -> np.ndarray:
            df = pd.DataFrame(x_array, columns=columns)
            return np.asarray(self.pipeline.predict(df), dtype=float)

        return fn
