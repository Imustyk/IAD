"""Model leaderboard and model-card UI components."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from iad.frontend.components.charts import feature_importance_bar, render_plotly


@dataclass(frozen=True)
class LeaderboardEntry:
    """Single row in a model leaderboard."""

    rank: int
    model_name: str
    primary_metric: float
    primary_metric_label: str
    cv_mean: float | None = None
    cv_std: float | None = None
    train_time_s: float | None = None
    is_best: bool = False
    family: str | None = None
    extra: dict[str, Any] | None = None


def render_model_card(entry: LeaderboardEntry) -> None:
    """Render one model result card."""
    best_class = " best" if entry.is_best else ""
    badge = '<span class="iad-badge best">Best</span>' if entry.is_best else ""
    family_badge = (
        f'<span class="iad-badge family">{entry.family}</span>' if entry.family else ""
    )
    cv_text = ""
    if entry.cv_mean is not None:
        std = f" ± {entry.cv_std:.4f}" if entry.cv_std is not None else ""
        cv_text = f'<div style="font-size:0.85rem;color:var(--iad-text-muted)">CV: {entry.cv_mean:.4f}{std}</div>'
    time_text = ""
    if entry.train_time_s is not None:
        time_text = (
            f'<div style="font-size:0.85rem;color:var(--iad-text-muted)">'
            f"Train time: {entry.train_time_s:.2f}s</div>"
        )

    st.markdown(
        f"""
        <div class="iad-model-card{best_class} iad-animate-in">
          <div class="iad-model-card-header">
            <strong>#{entry.rank} {entry.model_name}</strong>
            <span>{badge}{family_badge}</span>
          </div>
          <div style="font-size:1.25rem;font-weight:700;color:var(--iad-primary)">
            {entry.primary_metric_label}: {entry.primary_metric:.4f}
          </div>
          {cv_text}
          {time_text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_leaderboard(entries: list[LeaderboardEntry]) -> None:
    """Render full model leaderboard."""
    if not entries:
        st.info("No models trained yet. Run training on the Predictive Modeling page.")
        return
    for entry in entries:
        render_model_card(entry)


def render_leaderboard_table(
    leaderboard_df: pd.DataFrame,
    *,
    metric_col: str,
    ascending: bool = False,
) -> None:
    """Render leaderboard as a sortable dataframe."""
    if leaderboard_df.empty:
        st.info("No results to display.")
        return
    sorted_df = leaderboard_df.sort_values(metric_col, ascending=ascending)
    st.dataframe(sorted_df, use_container_width=True, hide_index=True)


def render_leaderboard_chart(
    leaderboard_df: pd.DataFrame,
    *,
    model_col: str = "model",
    metric_col: str = "primary_metric",
    title: str = "Model comparison",
) -> None:
    if leaderboard_df.empty:
        return
    fig = px.bar(
        leaderboard_df.sort_values(metric_col, ascending=True),
        x=metric_col,
        y=model_col,
        orientation="h",
        title=title,
        text=metric_col,
    )
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig.update_layout(yaxis=dict(autorange="reversed"), uniformtext_minsize=8)
    render_plotly(fig)


def render_feature_importance(
    names: list[str],
    values: list[float],
    *,
    top_n: int = 15,
    title: str = "Top features",
) -> None:
    feature_importance_bar(names, values, title=title, top_n=top_n)


def entries_from_leaderboard_df(
    leaderboard_df: pd.DataFrame,
    *,
    task_type: str,
    best_model_name: str,
) -> list[LeaderboardEntry]:
    """Build cards from a leaderboard DataFrame (legacy or enterprise)."""
    metric_key = "roc_auc" if task_type == "classification" else "r2"
    if metric_key not in leaderboard_df.columns:
        for alt in ("accuracy", "f1", "rmse", "mae"):
            if alt in leaderboard_df.columns:
                metric_key = alt
                break
    metric_label = metric_key.upper().replace("_", " ")

    entries: list[LeaderboardEntry] = []
    for idx, row in enumerate(leaderboard_df.to_dict(orient="records"), start=1):
        name = str(row.get("model") or row.get("model_name") or f"Model {idx}")
        metric_val = float(row.get(metric_key, row.get("score", 0.0)) or 0.0)
        entries.append(
            LeaderboardEntry(
                rank=idx,
                model_name=name,
                primary_metric=metric_val,
                primary_metric_label=metric_label,
                cv_mean=row.get("cv_mean") or row.get(f"cv_{metric_key}"),
                cv_std=row.get("cv_std"),
                train_time_s=row.get("train_time_s") or row.get("fit_time"),
                is_best=(name == best_model_name),
                family=row.get("family"),
            )
        )
    return entries


def entries_from_training_report(
    report: Any,
    *,
    task_type: str | None = None,
) -> list[LeaderboardEntry]:
    """Convert report object or dict to leaderboard cards."""
    if hasattr(report, "leaderboard"):
        lb = report.leaderboard
        tt = task_type or getattr(report, "task_type", None) or getattr(report, "task", "classification")
        best = getattr(report, "best_model_name", None) or ""
        if isinstance(lb, pd.DataFrame):
            return entries_from_leaderboard_df(lb, task_type=str(tt), best_model_name=str(best))
    if isinstance(report, dict):
        rows = report.get("leaderboard") or report.get("results") or []
        tt = task_type or report.get("task_type", "classification")
        best = report.get("best_model") or report.get("best_model_name", "")
        if isinstance(rows, pd.DataFrame):
            return entries_from_leaderboard_df(rows, task_type=str(tt), best_model_name=str(best))
    return []
