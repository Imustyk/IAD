"""Plotly chart wrappers with consistent styling and caching."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from iad.frontend.streamlit_compat import plotly_chart as _plotly_chart

def _template() -> str:
    return "plotly_white"


def apply_chart_theme(fig: go.Figure) -> go.Figure:
    """Apply consistent layout defaults to a Plotly figure."""
    fig.update_layout(
        template=_template(),
        font=dict(family="Inter, system-ui, sans-serif"),
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="closest",
    )
    return fig


@st.cache_data(show_spinner=False, ttl=300)
def cached_figure(
    chart_type: str,
    data_hash: str,
    fig_json: str,
) -> go.Figure:
    """Cache a serialised figure by type + data fingerprint.

    ``data_hash`` should be a stable hash of the underlying data (e.g. from
    ``pandas.util.hash_pandas_object`` or a shape/dtype tuple for small frames).
    """
    import json

    return go.Figure(json.loads(fig_json))


def render_plotly(
    fig: go.Figure,
    *,
    use_container_width: bool = True,
    key: str | None = None,
) -> None:
    """Render a themed Plotly figure."""
    apply_chart_theme(fig)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        autosize=True,
    )
    _plotly_chart(fig, key=key, stretch=use_container_width)


def histogram(
    df: pd.DataFrame,
    column: str,
    *,
    nbins: int = 40,
    title: str | None = None,
) -> None:
    fig = px.histogram(df, x=column, nbins=nbins, title=title or f"Distribution of {column}")
    render_plotly(fig)


def boxplot(
    df: pd.DataFrame,
    x: str | None,
    y: str,
    *,
    title: str | None = None,
) -> None:
    fig = px.box(df, x=x, y=y, title=title or f"Box plot — {y}")
    render_plotly(fig)


def bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    color: str | None = None,
    title: str | None = None,
) -> None:
    fig = px.bar(df, x=x, y=y, color=color, title=title)
    render_plotly(fig)


def heatmap(
    matrix: pd.DataFrame,
    *,
    title: str = "Correlation heatmap",
    colorscale: str = "RdBu_r",
) -> None:
    fig = px.imshow(
        matrix,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale=colorscale,
        title=title,
    )
    render_plotly(fig)


def scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    color: str | None = None,
    size: str | None = None,
    title: str | None = None,
) -> None:
    fig = px.scatter(df, x=x, y=y, color=color, size=size, title=title)
    render_plotly(fig)


def line_chart(
    df: pd.DataFrame,
    x: str,
    y: str | list[str],
    *,
    title: str | None = None,
) -> None:
    fig = px.line(df, x=x, y=y, title=title)
    render_plotly(fig)


def confusion_matrix_heatmap(
    matrix: list[list[int]],
    labels: list[str],
    *,
    title: str = "Confusion matrix",
) -> None:
    # px.imshow treats ``labels`` / categorical x,y as axis *titles* in recent Plotly — use Heatmap.
    text = [[str(cell) for cell in row] for row in matrix]
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=labels,
            y=labels,
            text=text,
            texttemplate="%{text}",
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
            colorscale="Blues",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Predicted",
        yaxis_title="Actual",
        xaxis=dict(type="category"),
        yaxis=dict(type="category", autorange="reversed"),
    )
    render_plotly(fig)


def feature_importance_bar(
    names: list[str],
    values: list[float],
    *,
    title: str = "Feature importance",
    top_n: int = 20,
) -> None:
    pairs = sorted(zip(names, values, strict=False), key=lambda p: abs(p[1]), reverse=True)[:top_n]
    imp_df = pd.DataFrame(pairs, columns=["feature", "importance"])
    fig = px.bar(
        imp_df,
        x="importance",
        y="feature",
        orientation="h",
        title=title,
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    render_plotly(fig)


def shap_waterfall(
    base_value: float,
    contributions: dict[str, float],
    prediction: float,
    *,
    title: str = "SHAP explanation",
) -> None:
    """Render a simplified waterfall-style bar chart for local SHAP values."""
    features = list(contributions.keys())
    values = list(contributions.values())
    fig = go.Figure(go.Waterfall(
        name="SHAP",
        orientation="v",
        measure=["absolute"] * len(features) + ["total"],
        x=features + ["Prediction"],
        y=values + [prediction],
        connector={"line": {"color": "var(--iad-border)"}},
        increasing={"marker": {"color": "#10b981"}},
        decreasing={"marker": {"color": "#ef4444"}},
        totals={"marker": {"color": "#4f46e5"}},
    ))
    fig.update_layout(title=title, showlegend=False)
    fig.add_hline(y=base_value, line_dash="dash", line_color="#94a3b8",
                  annotation_text=f"base={base_value:.3f}")
    render_plotly(fig)
