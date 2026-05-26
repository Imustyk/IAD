"""NLP analytics service — sentiment, embeddings, topic modeling."""
from __future__ import annotations

import pandas as pd

from iad.ml.nlp.embeddings import embed_text_column
from iad.ml.nlp.reports import EmbeddingReport, SentimentReport, TopicReport
from iad.ml.nlp.sentiment import analyze_sentiment
from iad.ml.nlp.topics import analyze_topics


class NLPService:
    """Facade for text analytics operations."""

    def sentiment(
        self,
        df: pd.DataFrame,
        column: str,
        *,
        method: str = "vader",
    ) -> SentimentReport:
        return analyze_sentiment(df, column, method=method)

    def embeddings(
        self,
        df: pd.DataFrame,
        column: str,
        *,
        method: str = "tfidf_svd",
        n_components: int = 2,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> EmbeddingReport:
        return embed_text_column(
            df,
            column,
            method=method,
            n_components=n_components,
            model_name=model_name,
        )

    def topics(
        self,
        df: pd.DataFrame,
        column: str,
        *,
        n_topics: int = 5,
    ) -> TopicReport:
        return analyze_topics(df, column, n_topics=n_topics)
