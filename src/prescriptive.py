"""Prescriptive analytics: what-if scenarios and recommendations."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .predictive import predict


def what_if_scenario(
    pipeline,
    base_row: pd.Series,
    feature: str,
    grid: np.ndarray,
    feature_columns: list[str],
    task_type: str,
) -> pd.DataFrame:
    """Sweep one feature across a grid keeping all others fixed."""
    rows = []
    for value in grid:
        row = base_row.copy()
        row[feature] = value
        rows.append(row)
    scenario_df = pd.DataFrame(rows)
    return predict(pipeline, scenario_df, feature_columns, task_type)


def two_factor_scenario(
    pipeline,
    base_row: pd.Series,
    feature_a: str,
    grid_a: np.ndarray,
    feature_b: str,
    grid_b: np.ndarray,
    feature_columns: list[str],
    task_type: str,
) -> pd.DataFrame:
    """Sweep two features in a grid keeping the rest fixed."""
    records = []
    for va in grid_a:
        for vb in grid_b:
            row = base_row.copy()
            row[feature_a] = va
            row[feature_b] = vb
            records.append(row)
    scenario_df = pd.DataFrame(records)
    preds = predict(pipeline, scenario_df, feature_columns, task_type)
    return preds


def generate_recommendations(report) -> list[str]:
    """Translate the training report into plain-language recommendations."""
    recs: list[str] = []
    if report is None:
        return recs

    metrics = report.metrics or {}
    if report.task_type == "classification":
        acc = metrics.get("accuracy")
        f1 = metrics.get("f1_macro")
        if acc is not None and acc < 0.7:
            recs.append(
                f"Accuracy of the best model ({report.best_model_name}) is only "
                f"{acc:.2%}. Consider gathering more data, engineering richer "
                f"features or trying a more flexible model."
            )
        elif acc is not None:
            recs.append(
                f"The best model **{report.best_model_name}** reaches {acc:.2%} "
                f"accuracy and {f1:.2f} macro F1 — strong enough to power a first "
                f"version of the SaaS scoring service."
            )
    else:
        r2 = metrics.get("r2")
        if r2 is not None and r2 < 0.4:
            recs.append(
                f"R² of {r2:.2f} indicates the predictors explain only a small "
                f"share of the variance in **{report.target}**. Investigate "
                f"non-linear features, interactions or domain-specific signals."
            )
        elif r2 is not None:
            recs.append(
                f"The best model **{report.best_model_name}** explains "
                f"{r2:.0%} of the variance in **{report.target}** with RMSE "
                f"{metrics.get('rmse', float('nan')):.3f}."
            )

    if report.feature_importance is not None and not report.feature_importance.empty:
        top = report.feature_importance.head(3)
        feature_list = ", ".join(f"`{f}`" for f in top["feature"])
        recs.append(
            f"Focus operational decisions on the top drivers: {feature_list}. "
            f"Even small interventions on these levers should move the KPI."
        )

    if report.cv_metrics:
        std_key = next((k for k in report.cv_metrics if k.endswith("_std")), None)
        mean_key = next((k for k in report.cv_metrics if k.endswith("_mean")), None)
        if std_key and mean_key:
            std = report.cv_metrics[std_key]
            mean = report.cv_metrics[mean_key]
            if mean and std / max(abs(mean), 1e-9) > 0.2:
                recs.append(
                    f"Cross-validated score is unstable (mean={mean:.2f}, "
                    f"std={std:.2f}). Increase the dataset size or use stratified "
                    f"sampling to stabilise generalisation."
                )

    if not recs:
        recs.append(
            "No automatic recommendations were generated. Train a model first or "
            "review the metrics on the Predictive Modeling page."
        )
    return recs
