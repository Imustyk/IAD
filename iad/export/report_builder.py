"""Build :class:`AnalyticsReport` from Streamlit session artifacts."""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd

from iad.export.models import AnalyticsReport, ReportSection
from iad.state.session import (
    KEY_BUSINESS_CASE,
    KEY_DATASET,
    KEY_DATASET_NAME,
    KEY_MODEL_BUNDLE,
    KEY_TARGET_COLUMN,
    KEY_TASK_TYPE,
    KEY_TRAINING_REPORT,
)


def build_report_from_session(session: dict[str, Any]) -> AnalyticsReport:
    """Construct a report snapshot from ``st.session_state``-like mapping."""
    df = session.get(KEY_DATASET)
    dataset_name = session.get(KEY_DATASET_NAME)
    shape: tuple[int, int] | None = None
    if df is not None and hasattr(df, "shape"):
        shape = (int(df.shape[0]), int(df.shape[1]))

    training = session.get(KEY_TRAINING_REPORT)
    metrics: dict[str, float] = {}
    model_name: str | None = None
    sections: list[ReportSection] = []

    if training is not None:
        model_name = getattr(training, "best_model_name", None)
        raw_metrics = getattr(training, "metrics", None) or {}
        metrics = {k: float(v) for k, v in raw_metrics.items() if isinstance(v, (int, float))}
        cv = getattr(training, "cv_metrics", None) or {}
        if cv:
            sections.append(
                ReportSection(
                    title="Cross-validation",
                    body="Hold-out and cross-validated scores from the training run.",
                    metrics={k: float(v) for k, v in cv.items() if isinstance(v, (int, float))},
                )
            )
        leaderboard = None
        if hasattr(training, "leaderboard_frame"):
            try:
                leaderboard = training.leaderboard_frame()
            except Exception:
                leaderboard = None
        if leaderboard is not None and hasattr(leaderboard, "head"):
            top = leaderboard.head(5)
            bullets = tuple(
                f"{row.get('model_name', row.get('model', 'model'))}: "
                + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items() if k != "model_name")
                for _, row in top.iterrows()
            )
            sections.append(
                ReportSection(
                    title="Leaderboard (top 5)",
                    body="Models ranked by the primary optimization metric.",
                    bullets=bullets[:5],
                )
            )

    business = session.get(KEY_BUSINESS_CASE) or {}
    if isinstance(business, dict) and any(str(business.get(k, "")).strip() for k in ("title", "problem", "objective")):
        bullets = tuple(
            f"{label}: {business[key]}"
            for key, label in (
                ("problem", "Problem"),
                ("objective", "Objective"),
                ("kpis", "KPIs"),
                ("stakeholders", "Stakeholders"),
            )
            if str(business.get(key, "")).strip()
        )
        sections.insert(
            0,
            ReportSection(
                title=business.get("title") or "Business understanding",
                body=str(business.get("data_sources", "") or "CRISP-DM — business understanding snapshot."),
                bullets=bullets,
            ),
        )

    if df is not None:
        missing_pct = float(df.isna().sum().sum() / max(df.size, 1) * 100)
        dup = int(df.duplicated().sum())
        sections.insert(
            0,
            ReportSection(
                title="Dataset profile",
                body="High-level data quality indicators at export time.",
                metrics={
                    "missing_cells_pct": round(missing_pct, 2),
                    "duplicate_rows": float(dup),
                    "numeric_columns": float(len(df.select_dtypes(include="number").columns)),
                },
            ),
        )

    bundle = session.get(KEY_MODEL_BUNDLE)
    if bundle is not None and model_name is None:
        model_name = type(bundle).__name__

    return AnalyticsReport(
        title="IAD Analytics Report",
        subtitle="Automated export from the Data Science SaaS workspace",
        dataset_name=str(dataset_name) if dataset_name else None,
        dataset_shape=shape,
        target_column=session.get(KEY_TARGET_COLUMN),
        task_type=session.get(KEY_TASK_TYPE),
        model_name=model_name,
        sections=sections,
        metrics=metrics,
        generated_at_iso=datetime.now(tz=UTC).isoformat(),
    )


def build_report_from_dataframe(
    df: pd.DataFrame,
    *,
    title: str = "Dataset Report",
    dataset_name: str | None = None,
) -> AnalyticsReport:
    """Minimal report when only a dataframe is available."""
    missing_pct = float(df.isna().sum().sum() / max(df.size, 1) * 100)
    return AnalyticsReport(
        title=title,
        dataset_name=dataset_name,
        dataset_shape=(int(df.shape[0]), int(df.shape[1])),
        sections=[
            ReportSection(
                title="Dataset profile",
                body="Exported without an active training session.",
                metrics={
                    "missing_cells_pct": round(missing_pct, 2),
                    "duplicate_rows": float(int(df.duplicated().sum())),
                },
            )
        ],
        generated_at_iso=datetime.now(tz=UTC).isoformat(),
    )
