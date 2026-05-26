"""Predictive modeling page — train, compare and persist ML models."""
from __future__ import annotations

import io

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from iad.config import get_settings
from iad.core.logging import get_logger
from iad.frontend.components import alerts
from iad.frontend.components.charts import confusion_matrix_heatmap, feature_importance_bar, render_plotly
from iad.frontend.components.metric_cards import MetricSpec, render_metric_row
from iad.frontend.components.model_cards import (
    entries_from_leaderboard_df,
    render_leaderboard,
    render_leaderboard_chart,
)
from iad.frontend.components.progress import progress_bar
from iad.frontend.components.tables import render_dataframe
from iad.frontend.layouts.page import section, setup_page
from iad.frontend.performance.background import poll_active_job, render_job_status, submit_training_job
from iad.frontend.services.training import train_enterprise, train_legacy
from iad.performance.memory import MemoryFootprint
from iad.ml.training.registry import ModelRegistry
from iad.state.session import (
    KEY_FEATURE_COLUMNS,
    KEY_MODEL_BUNDLE,
    KEY_TARGET_COLUMN,
    KEY_TASK_TYPE,
    KEY_TRAINING_REPORT,
)
from src.predictive import CLASSIFIERS, REGRESSORS
from src.utils import detect_task_type, require_dataset

settings = get_settings()
logger = get_logger("iad.frontend.predictive")

setup_page(
    "Predictive Modeling",
    caption="Step 4 — train and benchmark ML models; keep the best for inference.",
)

df = require_dataset()
if df is None:
    st.stop()

_mem = MemoryFootprint.from_dataframe(df)
st.caption(f"Training data footprint: {_mem.memory_mb} MB ({_mem.rows:,} rows)")

if render_job_status():
    st.info("Training is running in the background. This page will update when complete.")
    _job = poll_active_job()
    if _job and _job.status.value == "completed" and _job.result is not None:
        pipeline, report = _job.result
        st.session_state[KEY_MODEL_BUNDLE] = pipeline
        st.session_state[KEY_TRAINING_REPORT] = report
        alerts.success(f"Background training complete — **{report.best_model_name}**.")
        st.rerun()
    st.stop()

columns = df.columns.tolist()
suggested = st.session_state.get(KEY_TARGET_COLUMN) or columns[-1]
target = st.selectbox(
    "Target column (what to predict)",
    columns,
    index=columns.index(suggested) if suggested in columns else len(columns) - 1,
)
st.session_state[KEY_TARGET_COLUMN] = target

inferred = detect_task_type(df[target])
task_type = st.radio(
    "Task type",
    ["classification", "regression"],
    index=0 if inferred == "classification" else 1,
    horizontal=True,
)
st.session_state[KEY_TASK_TYPE] = task_type

feature_pool = [c for c in columns if c != target]
features = st.multiselect("Features used by the model", feature_pool, default=feature_pool)
if not features:
    alerts.warning("Select at least one feature column to train a model.")
    st.stop()
st.session_state[KEY_FEATURE_COLUMNS] = features

use_enterprise = st.toggle(
    "Enterprise ML engine (XGBoost, LightGBM, auto-preprocessing, MLflow-ready)",
    value=settings.UI_ENABLE_ENTERPRISE_ML,
    help="When off, uses the original sklearn-only training path for full backward compatibility.",
)

with st.expander("Training settings", expanded=False):
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    with col_b:
        cv_folds = st.slider("Cross-validation folds", 3, 10, 5, 1)
    with col_c:
        random_state = st.number_input("Random state", 0, 10_000, 42, 1)

    if use_enterprise:
        registry = ModelRegistry.default()
        candidate_models = registry.names(task_type)  # type: ignore[arg-type]
    else:
        candidate_models = list(
            (CLASSIFIERS if task_type == "classification" else REGRESSORS).keys()
        )
    selected_models = st.multiselect(
        "Models to evaluate",
        candidate_models,
        default=candidate_models,
    )

use_background = st.checkbox(
    "Run training in background (keeps UI responsive)",
    value=settings.PERF_BACKGROUND_TRAINING,
)

train_clicked = st.button("Train models", type="primary")

if train_clicked:
    try:

        def _run_training() -> tuple[object, object]:
            if use_enterprise:
                return train_enterprise(
                    df=df,
                    target=target,
                    feature_columns=features,
                    task_type=task_type,  # type: ignore[arg-type]
                    test_size=test_size,
                    random_state=int(random_state),
                    cv_folds=cv_folds,
                    selected_models=selected_models or None,
                )
            return train_legacy(
                df=df,
                target=target,
                feature_columns=features,
                task_type=task_type,  # type: ignore[arg-type]
                test_size=test_size,
                random_state=int(random_state),
                cv_folds=cv_folds,
                selected_models=selected_models or None,
            )

        if use_background and settings.PERF_BACKGROUND_TRAINING:
            submit_training_job(_run_training, name="predictive-training")
            alerts.info("Training started in the background. Stay on this page or return shortly.")
            st.rerun()
        else:
            with progress_bar("Training and benchmarking models…") as update:
                update(0.1, "Preparing data…")
                update(0.3, "Training models…")
                pipeline, report = _run_training()
                update(1.0, "Complete")
            st.session_state[KEY_MODEL_BUNDLE] = pipeline
            st.session_state[KEY_TRAINING_REPORT] = report
            alerts.success(
                f"Training complete — best model: **{report.best_model_name}** "
                f"({report.engine} engine)."
            )
        if not use_background and settings.DATABASE_ENABLED and report.training_result is not None:
            try:
                from iad.backend.services import PersistenceService

                persisted = PersistenceService().persist_training_result(
                    report.training_result,
                    experiment_name=f"{target}-{task_type}",
                )
                st.caption(
                    f"Saved to database — experiment `{persisted.experiment_id[:8]}…` "
                    f"({persisted.metric_count} metrics)."
                )
            except Exception as persist_exc:
                logger.warning("database persist skipped: %s", persist_exc)
    except Exception as exc:
        alerts.error(f"Training failed: {exc}", show_details=True, exc=exc)

report = st.session_state.get(KEY_TRAINING_REPORT)
pipeline = st.session_state.get(KEY_MODEL_BUNDLE)

if report is None:
    alerts.info("Train models with the button above to see the leaderboard and diagnostics.")
    st.stop()

section("Model leaderboard")
entries = entries_from_leaderboard_df(
    report.leaderboard,
    task_type=report.task_type,
    best_model_name=report.best_model_name,
)
render_leaderboard(entries)
_metric_col = "roc_auc" if report.task_type == "classification" else "r2"
if _metric_col not in report.leaderboard.columns:
    for _alt in ("accuracy", "f1", "rmse", "mae"):
        if _alt in report.leaderboard.columns:
            _metric_col = _alt
            break
if _metric_col in report.leaderboard.columns:
    render_leaderboard_chart(
        report.leaderboard,
        metric_col=_metric_col,
        title="Model comparison",
    )

section(f"Best model: {report.best_model_name}")
render_metric_row([
    MetricSpec(k, f"{v:.4f}") for k, v in list(report.metrics.items())[:4]
])

if report.cv_metrics:
    render_metric_row([
        MetricSpec(f"CV {k}", f"{v:.4f}") for k, v in list(report.cv_metrics.items())[:4]
    ])

left, right = st.columns(2)
with left:
    if report.task_type == "classification" and report.confusion_matrix is not None:
        st.markdown("**Confusion matrix (test set)**")
        labels = list(report.confusion_matrix.index.astype(str))
        confusion_matrix_heatmap(
            report.confusion_matrix.values.tolist(),
            labels,
        )
    elif report.task_type == "regression" and report.residuals is not None:
        st.markdown("**Predicted vs Actual**")
        fig = px.scatter(
            report.residuals,
            x="actual",
            y="predicted",
            trendline="ols",
            title="Predicted vs actual",
        )
        max_v = max(report.residuals["actual"].max(), report.residuals["predicted"].max())
        min_v = min(report.residuals["actual"].min(), report.residuals["predicted"].min())
        fig.add_shape(
            type="line",
            x0=min_v,
            y0=min_v,
            x1=max_v,
            y1=max_v,
            line=dict(dash="dash"),
        )
        render_plotly(fig)

with right:
    if report.task_type == "regression" and report.residuals is not None:
        st.markdown("**Residual distribution**")
        fig = px.histogram(
            report.residuals,
            x="residual",
            nbins=40,
            title="Residual histogram",
            marginal="box",
        )
        render_plotly(fig)
    if report.feature_importance is not None and not report.feature_importance.empty:
        st.markdown("**Top feature importances**")
        top = report.feature_importance.head(20)
        feature_importance_bar(
            top["feature"].astype(str).tolist(),
            top["importance"].astype(float).tolist(),
        )

with st.expander("Test-set predictions"):
    if report.test_predictions is not None:
        render_dataframe(report.test_predictions.head(200))

with st.expander("Persist the trained model"):
    if pipeline is not None:
        buffer = io.BytesIO()
        joblib.dump(
            {
                "pipeline": pipeline,
                "task_type": report.task_type,
                "target": report.target,
                "features": report.features,
                "best_model_name": report.best_model_name,
            },
            buffer,
        )
        buffer.seek(0)
        st.download_button(
            "Download model (.joblib)",
            data=buffer,
            file_name=f"model_{report.best_model_name.lower().replace(' ', '_')}.joblib",
            mime="application/octet-stream",
        )
