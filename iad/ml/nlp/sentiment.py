"""Sentiment analysis for text columns."""
from __future__ import annotations

import pandas as pd

from iad.core.exceptions import AnalyticsError, SchemaError
from iad.core.logging import get_logger
from iad.ml.nlp.availability import vader_available
from iad.ml.nlp.reports import SentimentReport

logger = get_logger("iad.ml.nlp.sentiment")

_LABELS = ("negative", "neutral", "positive")


def _label_from_compound(compound: float, *, pos: float, neg: float) -> str:
    if compound >= pos:
        return "positive"
    if compound <= neg:
        return "negative"
    return "neutral"


def analyze_sentiment_vader(
    series: pd.Series,
    *,
    column: str | None = None,
    positive_threshold: float = 0.05,
    negative_threshold: float = -0.05,
) -> SentimentReport:
    """Score text with VADER (lexicon-based, fast, no GPU)."""
    if not vader_available():
        raise AnalyticsError(
            "vaderSentiment is not installed.",
            user_message="Install NLP extras: pip install vaderSentiment",
        )

    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    col = column or str(series.name or "text")
    text = series.fillna("").astype(str)
    if text.str.strip().eq("").all():
        raise SchemaError(
            f"Column {col!r} has no non-empty text.",
            user_message="Choose a text column with content.",
        )

    analyzer = SentimentIntensityAnalyzer()
    rows: list[dict[str, object]] = []
    for idx, value in text.items():
        scores = analyzer.polarity_scores(value)
        compound = float(scores["compound"])
        rows.append(
            {
                "index": idx,
                "text": value[:200],
                "neg": scores["neg"],
                "neu": scores["neu"],
                "pos": scores["pos"],
                "compound": compound,
                "label": _label_from_compound(
                    compound, pos=positive_threshold, neg=negative_threshold
                ),
            }
        )

    frame = pd.DataFrame(rows)
    summary = {
        "mean_compound": float(frame["compound"].mean()),
        "positive_share": float((frame["label"] == "positive").mean()),
        "negative_share": float((frame["label"] == "negative").mean()),
        "neutral_share": float((frame["label"] == "neutral").mean()),
    }
    logger.info("sentiment complete", extra={"column": col, "n_rows": len(frame)})
    return SentimentReport(column=col, scores=frame, summary=summary, method="vader")


def analyze_sentiment(
    df: pd.DataFrame,
    column: str,
    *,
    method: str = "vader",
) -> SentimentReport:
    """Dispatch sentiment analysis for a DataFrame column."""
    if column not in df.columns:
        raise SchemaError(
            f"Column {column!r} not found.",
            user_message="Select a valid text column.",
        )
    if method != "vader":
        raise AnalyticsError(
            f"Unsupported sentiment method {method!r}.",
            user_message="Only VADER sentiment is currently supported.",
        )
    return analyze_sentiment_vader(df[column], column=column)
