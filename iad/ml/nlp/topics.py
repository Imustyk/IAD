"""Topic modeling — LDA on text columns."""
from __future__ import annotations

import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from iad.core.exceptions import AnalyticsError, SchemaError
from iad.core.logging import get_logger
from iad.ml.nlp.reports import TopicReport

logger = get_logger("iad.ml.nlp.topics")


def fit_lda_topics(
    series: pd.Series,
    *,
    column: str | None = None,
    n_topics: int = 5,
    max_features: int = 2000,
    top_words: int = 10,
    random_state: int = 42,
) -> TopicReport:
    """Fit Latent Dirichlet Allocation and return top terms per topic."""
    col = column or str(series.name or "text")
    text = series.fillna("").astype(str)
    if len(text) < 3:
        raise SchemaError(
            "Need at least 3 non-empty documents for topic modeling.",
            user_message="Load more rows or choose a denser text column.",
        )

    n_topics = max(2, min(n_topics, len(text) - 1))
    vectorizer = CountVectorizer(max_features=max_features, stop_words="english")
    dtm = vectorizer.fit_transform(text)
    if dtm.shape[1] == 0:
        raise AnalyticsError(
            "Bag-of-words matrix is empty.",
            user_message="Text may be too short for topic modeling.",
        )

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=random_state,
        learning_method="batch",
        max_iter=20,
    )
    lda.fit(dtm)
    vocab = vectorizer.get_feature_names_out()
    topics: list[dict[str, object]] = []
    for topic_idx, topic_vector in enumerate(lda.components_):
        top_indices = topic_vector.argsort()[::-1][:top_words]
        words = [vocab[i] for i in top_indices]
        weights = [float(topic_vector[i]) for i in top_indices]
        topics.append(
            {
                "topic_id": topic_idx,
                "top_words": words,
                "weights": weights,
            }
        )

    doc_topics = pd.DataFrame(
        lda.transform(dtm),
        index=text.index,
        columns=[f"topic_{i}" for i in range(n_topics)],
    )
    dominant = doc_topics.idxmax(axis=1).rename("dominant_topic")
    doc_topics = doc_topics.join(dominant)

    logger.info("lda topics fitted", extra={"column": col, "n_topics": n_topics})
    return TopicReport(
        column=col,
        n_topics=n_topics,
        topics=topics,
        document_topics=doc_topics,
        method="lda",
    )


def analyze_topics(
    df: pd.DataFrame,
    column: str,
    *,
    n_topics: int = 5,
) -> TopicReport:
    if column not in df.columns:
        raise SchemaError(
            f"Column {column!r} not found.",
            user_message="Select a valid text column.",
        )
    return fit_lda_topics(df[column], column=column, n_topics=n_topics)
