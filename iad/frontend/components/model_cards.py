"""Model leaderboard and model-card UI components."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from iad.frontend.components.charts import render_plotly
from iad.frontend.components.html_render import esc
from iad.frontend.streamlit_compat import dataframe as st_dataframe


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
        f'<span class="iad-badge family">{esc(entry.family)}</span>' if entry.family else ""
    )
    cv_text = ""
    if entry.cv_mean is not None:
        std = f" ± {entry.cv_std:.4f}" if entry.cv_std is not None else ""
        cv_text = f'<div class="iad-model-meta">CV: {entry.cv_mean:.4f}{std}</div>'
    time_text = ""
    if entry.train_time_s is not None:
        time_text = f'<div class="iad-model-meta">Train time: {entry.train_time_s:.2f}s</div>'

    st.markdown(
        f"""
        <div class="iad-model-card{best_class}">
          <div class="iad-model-card-header">
            <strong>#{entry.rank} {esc(entry.model_name)}</strong>
            <span>{badge}{family_badge}</span>
          </div>
          <div class="iad-model-metric">
            {esc(entry.primary_metric_label)}: {entry.primary_metric:.4f}
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
    sorted_df = leaderboard_df.sort_values(metric_col, ascending=ascending)
    st_dataframe(sorted_df, stretch=True, hide_index=True)


def entries_from_leaderboard_df(
    leaderboard_df: pd.DataFrame,
    *,
    task_type: str,
    best_model_name: str,
    primary_metric_col: str | None = None,
) -> list[LeaderboardEntry]:
    """Build :class:`LeaderboardEntry` list from a training report frame."""
    if leaderboard_df.empty:
        return []
    metric_col = primary_metric_col or (
        "roc_auc" if task_type == "classification" else "r2"
    )
    if metric_col not in leaderboard_df.columns:
        numeric = leaderboard_df.select_dtypes(include="number").columns
        metric_col = numeric[0] if len(numeric) else leaderboard_df.columns[0]

    entries: list[LeaderboardEntry] = []
    for rank, row in enumerate(leaderboard_df.itertuples(), start=1):
        name = (
            getattr(row, "model_name", None)
            or getattr(row, "model", None)
            or (row[0] if hasattr(row, "__getitem__") else None)
            or getattr(row, "Index", str(rank))
        )
        score = float(getattr(row, metric_col, 0.0))
        cv_mean = getattr(row, "cv_mean", None)
        cv_std = getattr(row, "cv_std", None)
        train_time = getattr(row, "train_time_seconds", None) or getattr(row, "train_time_s", None)
        family = getattr(row, "family", None)
        entries.append(
            LeaderboardEntry(
                rank=rank,
                model_name=str(name),
                primary_metric=score,
                primary_metric_label=metric_col.replace("_", " ").upper(),
                cv_mean=float(cv_mean) if cv_mean is not None and pd.notna(cv_mean) else None,
                cv_std=float(cv_std) if cv_std is not None and pd.notna(cv_std) else None,
                train_time_s=float(train_time) if train_time is not None and pd.notna(train_time) else None,
                is_best=str(name) == best_model_name,
                family=str(family) if family is not None else None,
            )
        )
    return entries


def render_leaderboard_chart(
    leaderboard_df: pd.DataFrame,
    *,
    metric_col: str,
    title: str = "Model comparison",
) -> None:
    if leaderboard_df.empty or metric_col not in leaderboard_df.columns:
        return
    name_col = "model_name" if "model_name" in leaderboard_df.columns else leaderboard_df.columns[0]
    fig = px.bar(
        leaderboard_df.sort_values(metric_col, ascending=True),
        x=metric_col,
        y=name_col,
        orientation="h",
        title=title,
        color=metric_col,
        color_continuous_scale="Blues",
    )
    fig.update_layout(showlegend=False, height=max(280, 44 * len(leaderboard_df)))
    render_plotly(fig)
