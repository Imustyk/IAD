"""Apply Model page — single-row and batch inference on new data."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import streamlit as st

from iad.core.error_handler import handle_error
from iad.core.paths import safe_filename
from iad.frontend.layouts.page import setup_page
from iad.ml.training.persistence import load_bundle
from src.predictive import predict
from src.utils import SESSION_KEYS, require_dataset


setup_page(
    "Apply Model to New Data",
    icon="🎯",
    caption="Step 5 — score single rows, batch CSVs, or load a saved model bundle.",
)


# ---------------------------------------------------------------------------
# Load model from session or upload a saved bundle
# ---------------------------------------------------------------------------
pipeline = st.session_state.get(SESSION_KEYS["model_bundle"])
report = st.session_state.get(SESSION_KEYS["training_report"])
features = st.session_state.get(SESSION_KEYS["feature_columns"])
task_type = st.session_state.get(SESSION_KEYS["task_type"])
target = st.session_state.get(SESSION_KEYS["target_column"])

with st.expander("📦 Load a saved model bundle (.joblib)"):
    uploaded = st.file_uploader("Upload .joblib", type=["joblib"], key="model_upload")
    if uploaded is not None:
        try:
            from iad.backend.security.upload_policy import validate_model_upload
            from iad.config.settings import get_settings

            validate_model_upload(uploaded.name, uploaded.size)
            settings = get_settings()
            upload_dir = settings.DATA_DIR / "uploads" / "model_bundles"
            upload_dir.mkdir(parents=True, exist_ok=True)
            dest = upload_dir / safe_filename(uploaded.name)
            dest.write_bytes(uploaded.getvalue())
            pipeline_loaded, card = load_bundle(dest)
            st.session_state[SESSION_KEYS["model_bundle"]] = pipeline_loaded
            st.session_state[SESSION_KEYS["task_type"]] = card.task
            st.session_state[SESSION_KEYS["feature_columns"]] = list(card.features)
            st.session_state[SESSION_KEYS["target_column"]] = card.target
            st.success(
                f"Loaded model **{card.name}** ({card.task}) — target `{card.target}`."
            )
            pipeline = pipeline_loaded
            features = list(card.features)
            task_type = card.task
            target = card.target
        except Exception as exc:
            handle_error(exc)

if pipeline is None or not features:
    st.warning(
        "No trained model in this session. Train one on the **Predictive "
        "Modeling** page or upload a saved bundle above."
    )
    st.stop()

st.success(
    f"Active model — task: **{task_type}**, target: **{target}**, "
    f"features: {len(features)}"
)

df = require_dataset()


tab_single, tab_batch = st.tabs(["✏️ Single record", "📂 Batch CSV"])


with tab_single:
    st.markdown("Enter feature values to score a single new observation.")

    inputs: dict = {}
    cols = st.columns(2)
    for i, feat in enumerate(features):
        target_col = cols[i % 2]
        with target_col:
            if df is not None and feat in df.columns:
                series = df[feat]
                if pd.api.types.is_numeric_dtype(series):
                    median = float(series.median()) if series.notna().any() else 0.0
                    minv = float(series.min()) if series.notna().any() else 0.0
                    maxv = float(series.max()) if series.notna().any() else 1.0
                    inputs[feat] = st.number_input(
                        feat,
                        value=median,
                        min_value=minv if minv < median else None,
                        max_value=maxv if maxv > median else None,
                    )
                else:
                    options = series.dropna().unique().tolist()
                    if not options:
                        inputs[feat] = st.text_input(feat, value="")
                    else:
                        inputs[feat] = st.selectbox(feat, options)
            else:
                inputs[feat] = st.text_input(feat, value="")

    if st.button("Predict", type="primary"):
        try:
            row = pd.DataFrame([inputs])
            result = predict(pipeline, row, features, task_type)
            st.subheader("Prediction")
            pred_value = result.iloc[0]["prediction"]
            st.metric("Predicted value", f"{pred_value}")

            proba_cols = [c for c in result.columns if c.startswith("proba_")]
            if proba_cols:
                proba_df = (
                    result[proba_cols]
                    .iloc[0]
                    .rename_axis("class")
                    .reset_index(name="probability")
                )
                proba_df["class"] = proba_df["class"].str.replace("proba_", "")
                st.dataframe(proba_df.sort_values("probability", ascending=False),
                             use_container_width=True)
            with st.expander("Full row with prediction"):
                st.dataframe(result, use_container_width=True)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


with tab_batch:
    st.markdown(
        "Upload a CSV/Excel/Parquet file containing the same feature columns. "
        "Missing columns will be filled with NaN and imputed by the pipeline."
    )

    uploaded_batch = st.file_uploader(
        "Upload batch file",
        type=["csv", "tsv", "txt", "xlsx", "xls", "parquet"],
        key="batch_upload",
    )

    use_loaded_dataset = False
    if df is not None:
        use_loaded_dataset = st.checkbox(
            "Use the dataset already loaded in this session as a batch",
            value=False,
        )

    batch_df: pd.DataFrame | None = None
    if uploaded_batch is not None:
        try:
            name = uploaded_batch.name.lower()
            buffer = io.BytesIO(uploaded_batch.read())
            if name.endswith((".xlsx", ".xls")):
                batch_df = pd.read_excel(buffer)
            elif name.endswith(".parquet"):
                batch_df = pd.read_parquet(buffer)
            elif name.endswith(".tsv"):
                batch_df = pd.read_csv(buffer, sep="\t")
            else:
                batch_df = pd.read_csv(buffer)
        except Exception as exc:
            st.error(f"Failed to load batch file: {exc}")
    elif use_loaded_dataset and df is not None:
        batch_df = df.copy()

    if batch_df is not None:
        st.write(f"Batch contains {batch_df.shape[0]:,} rows.")
        st.dataframe(batch_df.head(10), use_container_width=True)
        if st.button("Score the batch", type="primary"):
            try:
                scored = predict(pipeline, batch_df, features, task_type)
                st.success(f"Scored {len(scored):,} rows.")
                st.dataframe(scored.head(200), use_container_width=True)
                csv_bytes = scored.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download scored CSV",
                    data=csv_bytes,
                    file_name="predictions.csv",
                    mime="text/csv",
                )
            except Exception as exc:
                st.error(f"Batch prediction failed: {exc}")
