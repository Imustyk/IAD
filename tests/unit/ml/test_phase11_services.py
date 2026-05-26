"""Phase 11 service facades and error-path coverage."""
from __future__ import annotations

import pandas as pd
import pytest

from iad.core.exceptions import AnalyticsError, SchemaError
from iad.ml.anomaly import AnomalyService
from iad.ml.clustering import ClusteringService
from iad.ml.clustering.reduction import project_pca
from iad.ml.forecasting import ForecastingService
from iad.ml.forecasting.prepare import prepare_series
from iad.ml.nlp import NLPService
from iad.ml.nlp.embeddings import embed_text_column
from iad.ml.nlp.sentiment import analyze_sentiment
from iad.ml.recommendation import RecommendationService
from iad.ml.recommendation.matrix import build_user_item_matrix


@pytest.mark.unit
def test_nlp_service_sentiment(text_df) -> None:
    pytest.importorskip("vaderSentiment")
    svc = NLPService()
    report = svc.sentiment(text_df, "review")
    assert report.method == "vader"


@pytest.mark.unit
def test_nlp_unknown_method(text_df) -> None:
    with pytest.raises(AnalyticsError):
        analyze_sentiment(text_df, "review", method="unknown")


@pytest.mark.unit
def test_nlp_missing_column() -> None:
    with pytest.raises(SchemaError):
        embed_text_column(pd.DataFrame({"a": [1]}), "missing")


@pytest.mark.unit
def test_forecasting_unknown_method(ts_df) -> None:
    svc = ForecastingService()
    with pytest.raises(AnalyticsError):
        svc.forecast(
            ts_df,
            datetime_column="month",
            value_column="sales",
            method="invalid",
        )


@pytest.mark.unit
def test_prepare_series_invalid() -> None:
    with pytest.raises(SchemaError):
        prepare_series(
            pd.DataFrame({"a": [1, 2]}),
            datetime_column="missing",
            value_column="a",
        )


@pytest.mark.unit
def test_sentiment_empty_column() -> None:
    pytest.importorskip("vaderSentiment")
    with pytest.raises(SchemaError):
        from iad.ml.nlp.sentiment import analyze_sentiment_vader

        analyze_sentiment_vader(pd.Series(["", "  "]), column="t")


@pytest.mark.unit
def test_prepare_series(ts_df) -> None:
    series = prepare_series(ts_df, datetime_column="month", value_column="sales")
    assert len(series) == len(ts_df)


@pytest.mark.unit
def test_clustering_service(iris_df) -> None:
    features = [c for c in iris_df.columns if iris_df[c].dtype.kind in "iuf"][:3]
    report = ClusteringService().kmeans(iris_df, feature_columns=features, n_clusters=2)
    assert report.method == "kmeans"


@pytest.mark.unit
def test_anomaly_flagged_frame(iris_df) -> None:
    report = AnomalyService().one_class_svm(iris_df, nu=0.1)
    flagged = report.flagged_frame(iris_df)
    assert "is_anomaly" in flagged.columns


@pytest.mark.unit
def test_clustering_report_labeled(iris_df) -> None:
    features = [c for c in iris_df.columns if iris_df[c].dtype.kind in "iuf"][:3]
    report = ClusteringService().dbscan(iris_df, feature_columns=features, eps=2.0)
    labeled = report.labeled_frame(iris_df)
    assert "cluster" in labeled.columns


@pytest.mark.unit
def test_arima_holdout(ts_df) -> None:
    from iad.ml.forecasting.arima import forecast_arima

    report = forecast_arima(
        ts_df,
        datetime_column="month",
        value_column="sales",
        horizon=3,
        order=(1, 0, 0),
        holdout=6,
    )
    assert "holdout_mae" in report.metrics


@pytest.mark.unit
def test_project_pca(iris_df) -> None:
    features = [c for c in iris_df.columns if iris_df[c].dtype.kind in "iuf"][:3]
    matrix = iris_df[features]
    coords = project_pca(matrix, n_components=2)
    assert coords.shape[1] == 2


@pytest.mark.unit
def test_anomaly_service(iris_df) -> None:
    report = AnomalyService().isolation_forest(iris_df, contamination=0.1)
    assert "anomaly_share" in report.metrics


@pytest.mark.unit
def test_recommendation_matrix(interactions) -> None:
    matrix = build_user_item_matrix(
        interactions, user_column="user", item_column="item", rating_column="rating"
    )
    assert matrix.shape[0] >= 2


@pytest.mark.unit
def test_recommendation_service(interactions) -> None:
    rec = RecommendationService().cosine_similarity(
        interactions,
        user_column="user",
        item_column="item",
        rating_column="rating",
        target_user="u1",
    )
    assert not rec.recommendations.empty


@pytest.fixture
def text_df() -> pd.DataFrame:
    return pd.DataFrame(
        {"review": ["good", "bad", "neutral", "great", "awful"]}
    )


@pytest.fixture
def ts_df() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=36, freq="MS")
    return pd.DataFrame({"month": dates, "sales": range(len(dates))})


@pytest.fixture
def interactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user": ["u1", "u1", "u2", "u2"],
            "item": ["a", "b", "a", "c"],
            "rating": [5.0, 2.0, 4.0, 3.0],
        }
    )
