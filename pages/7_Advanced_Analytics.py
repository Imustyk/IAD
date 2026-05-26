"""Advanced analytics — NLP, forecasting, clustering, anomaly, recommendations."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from iad.core.error_handler import handle_error
from iad.frontend.components.charts import render_plotly
from iad.frontend.layouts.page import setup_page
from iad.ml.anomaly import AnomalyService
from iad.ml.clustering import ClusteringService
from iad.ml.forecasting import ForecastingService, prophet_available
from iad.ml.forecasting.prepare import discover_datetime_columns
from iad.ml.nlp import NLPService
from iad.ml.nlp.availability import sentence_transformers_available, vader_available
from iad.ml.recommendation import RecommendationService
from src.utils import numeric_columns, require_dataset

setup_page(
    "Advanced Analytics",
    caption="Phase 11 — NLP, time series, clustering, anomaly detection, and recommendations.",
)

df = require_dataset()
if df is None:
    st.stop()

text_cols = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype) == "string"]
num_cols = numeric_columns(df)
date_cols = discover_datetime_columns(df)

tab_nlp, tab_ts, tab_cluster, tab_anomaly, tab_rec = st.tabs(
    ["NLP", "Time series", "Clustering", "Anomaly", "Recommendations"]
)

nlp = NLPService()
forecasting = ForecastingService()
clustering = ClusteringService()
anomaly = AnomalyService()
recommendations = RecommendationService()


def _run(action: str, fn) -> None:
    try:
        fn()
    except Exception as exc:
        handle_error(exc)


with tab_nlp:
    if not text_cols:
        st.info("No text columns detected. Load a dataset with free-text fields.")
    else:
        text_col = st.selectbox("Text column", text_cols, key="adv_nlp_col")
        sub_sent, sub_emb, sub_topic = st.tabs(["Sentiment", "Embeddings", "Topics"])

        with sub_sent:
            if not vader_available():
                st.warning("Install `vaderSentiment` for sentiment analysis.")
            elif st.button("Run sentiment", key="btn_sentiment"):
                def _sentiment() -> None:
                    st.session_state["adv_sentiment"] = nlp.sentiment(df, text_col)

                _run("Sentiment analysis", _sentiment)
            if report := st.session_state.get("adv_sentiment"):
                st.metric("Mean compound", f"{report.summary['mean_compound']:.3f}")
                st.dataframe(report.scores.head(200), use_container_width=True)
                dist = report.distribution_frame()
                if not dist.empty:
                    fig = px.bar(dist, x="label", y="share", title="Sentiment distribution")
                    render_plotly(fig)

        with sub_emb:
            methods = ["tfidf_svd"]
            if sentence_transformers_available():
                methods.append("sentence_transformer")
            emb_method = st.selectbox("Method", methods, key="adv_emb_method")
            n_dims = st.slider("Dimensions", 2, 5, 2, key="adv_emb_dims")
            if st.button("Compute embeddings", key="btn_embeddings"):
                def _embed() -> None:
                    st.session_state["adv_embeddings"] = nlp.embeddings(
                        df, text_col, method=emb_method, n_components=n_dims
                    )

                _run("Embeddings", _embed)
            if emb := st.session_state.get("adv_embeddings"):
                st.caption(emb.model_name or emb.method)
                plot_cols = [c for c in emb.matrix.columns if c.startswith("dim_")]
                if len(plot_cols) >= 2:
                    fig = px.scatter(
                        emb.matrix,
                        x=plot_cols[0],
                        y=plot_cols[1],
                        hover_data=["text_preview"],
                        title="Text embedding projection",
                    )
                    render_plotly(fig)

        with sub_topic:
            n_topics = st.slider("Number of topics", 2, 15, 5, key="adv_n_topics")
            if st.button("Discover topics", key="btn_topics"):
                def _topics() -> None:
                    st.session_state["adv_topics"] = nlp.topics(df, text_col, n_topics=n_topics)

                _run("Topic modeling", _topics)
            if topics := st.session_state.get("adv_topics"):
                for topic in topics.topics:
                    words = ", ".join(topic["top_words"][:8])  # type: ignore[index]
                    st.markdown(f"**Topic {topic['topic_id']}:** {words}")
                if topics.document_topics is not None:
                    st.dataframe(topics.document_topics.head(100), use_container_width=True)


with tab_ts:
    if not date_cols:
        st.warning(
            "No date/time column detected in this dataset. "
            "Load data with real dates (e.g. monthly sales) or use **Data loading** to parse a column as datetime."
        )
    if not num_cols:
        st.info("Need a numeric value column for forecasting.")
    elif not date_cols:
        pass
    else:
        dt_col = st.selectbox("Datetime column", date_cols, key="adv_dt_col")
        value_options = [c for c in num_cols if c != dt_col]
        if not value_options:
            st.error("No numeric column available for values besides the datetime column.")
            st.stop()
        val_col = st.selectbox("Value column", value_options, key="adv_val_col")
        horizon = st.slider("Forecast horizon", 3, 60, 14, key="adv_horizon")

        if st.button("Decompose series", key="btn_decompose"):
            def _decompose() -> None:
                st.session_state["adv_decomp"] = forecasting.decompose(
                    df, datetime_column=dt_col, value_column=val_col
                )

            _run("Decomposition", _decompose)
        if decomp := st.session_state.get("adv_decomp"):
            comp_df = pd.DataFrame(
                {
                    "observed": decomp.observed,
                    "trend": decomp.trend,
                    "seasonal": decomp.seasonal,
                    "residual": decomp.residual,
                }
            )
            fig = px.line(comp_df, title="Seasonal decomposition")
            render_plotly(fig)

        forecast_methods = ["arima"]
        if prophet_available():
            forecast_methods.append("prophet")
        method = st.selectbox("Forecast method", forecast_methods, key="adv_forecast_method")
        if st.button("Generate forecast", key="btn_forecast"):
            def _forecast() -> None:
                st.session_state["adv_forecast"] = forecasting.forecast(
                    df,
                    datetime_column=dt_col,
                    value_column=val_col,
                    method=method,
                    horizon=horizon,
                )

            _run("Forecasting", _forecast)
        if fc := st.session_state.get("adv_forecast"):
            if fc.metrics:
                st.json(fc.metrics)
            hist = fc.history.reset_index()
            fut = fc.forecast.reset_index()
            hist_plot = hist.rename(columns={"actual": "value"}).assign(series="history")
            fut_plot = fut.rename(columns={"yhat": "value"}).assign(series="forecast")
            combined = pd.concat([hist_plot, fut_plot], ignore_index=True)
            x_col = combined.columns[0]
            fig = px.line(combined, x=x_col, y="value", color="series", title="Forecast")
            render_plotly(fig)


with tab_cluster:
    if len(num_cols) < 2:
        st.info("Select at least two numeric columns for clustering.")
    else:
        features = st.multiselect(
            "Features",
            num_cols,
            default=num_cols[: min(5, len(num_cols))],
            key="adv_cluster_features",
        )
        algo = st.radio("Algorithm", ["kmeans", "dbscan"], horizontal=True, key="adv_cluster_algo")
        projection = st.selectbox("2-D projection", ["pca", "umap"], key="adv_cluster_proj")
        k = st.slider("Clusters (k)", 2, 12, 3, key="adv_k")
        eps = st.number_input("DBSCAN eps", 0.1, 5.0, 0.5, 0.1, key="adv_eps")
        min_samples = st.slider("DBSCAN min_samples", 2, 20, 5, key="adv_min_samples")

        if st.button("Run clustering", key="btn_cluster") and features:
            def _cluster() -> None:
                if algo == "kmeans":
                    st.session_state["adv_cluster"] = clustering.kmeans(
                        df,
                        feature_columns=features,
                        n_clusters=k,
                        projection=projection,
                    )
                else:
                    st.session_state["adv_cluster"] = clustering.dbscan(
                        df,
                        feature_columns=features,
                        eps=eps,
                        min_samples=min_samples,
                        projection=projection,
                    )

            _run("Clustering", _cluster)

        if cl := st.session_state.get("adv_cluster"):
            st.json(cl.metrics)
            if cl.projection is not None:
                dims = [c for c in cl.projection.columns if c.startswith(("pc", "umap"))]
                if len(dims) >= 2:
                    fig = px.scatter(
                        cl.projection,
                        x=dims[0],
                        y=dims[1],
                        color=cl.projection["cluster"].astype(str),
                        title=f"{cl.method} clusters",
                    )
                    render_plotly(fig)
            if cl.cluster_sizes is not None:
                st.dataframe(cl.cluster_sizes, use_container_width=True)


with tab_anomaly:
    if not num_cols:
        st.info("Need numeric columns for multivariate anomaly detection.")
    else:
        feat = st.multiselect(
            "Features (empty = all numeric)",
            num_cols,
            default=num_cols[: min(8, len(num_cols))],
            key="adv_anomaly_features",
        )
        method = st.radio(
            "Detector",
            ["isolation_forest", "one_class_svm"],
            horizontal=True,
            key="adv_anomaly_method",
        )
        contamination = st.slider("Expected anomaly rate", 0.01, 0.25, 0.05, key="adv_contam")

        if st.button("Detect anomalies", key="btn_anomaly"):
            def _anomaly() -> None:
                cols = feat or None
                if method == "isolation_forest":
                    st.session_state["adv_anomaly"] = anomaly.isolation_forest(
                        df, feature_columns=cols, contamination=contamination
                    )
                else:
                    st.session_state["adv_anomaly"] = anomaly.one_class_svm(
                        df, feature_columns=cols, nu=contamination
                    )

            _run("Anomaly detection", _anomaly)

        if ar := st.session_state.get("adv_anomaly"):
            st.json(ar.metrics)
            flagged = ar.flagged_frame(df)
            st.dataframe(flagged[flagged["is_anomaly"]].head(100), use_container_width=True)


with tab_rec:
    st.caption("Requires user · item · rating interaction columns (long format).")
    all_cols = list(df.columns)
    user_col = st.selectbox("User column", all_cols, key="adv_rec_user")
    item_col = st.selectbox("Item column", all_cols, key="adv_rec_item")
    rating_col = st.selectbox("Rating column", num_cols or all_cols, key="adv_rec_rating")
    users = df[user_col].dropna().unique().tolist() if user_col in df.columns else []
    if users:
        target = st.selectbox("Target user", users[:500], key="adv_rec_target")
        rec_method = st.radio(
            "Method",
            ["user_collaborative", "cosine_item"],
            horizontal=True,
            key="adv_rec_method",
        )
        if st.button("Recommend", key="btn_rec"):
            def _rec() -> None:
                if rec_method == "user_collaborative":
                    st.session_state["adv_rec"] = recommendations.user_collaborative(
                        df,
                        user_column=user_col,
                        item_column=item_col,
                        rating_column=rating_col,
                        target_user=target,
                    )
                else:
                    st.session_state["adv_rec"] = recommendations.cosine_similarity(
                        df,
                        user_column=user_col,
                        item_column=item_col,
                        rating_column=rating_col,
                        target_user=target,
                    )

            _run("Recommendations", _rec)
        if rec := st.session_state.get("adv_rec"):
            st.dataframe(rec.recommendations, use_container_width=True)
    else:
        st.info("No users found in the selected column.")
