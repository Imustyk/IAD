"""Phase 11 NLP tests."""
from __future__ import annotations

import pandas as pd
import pytest

from iad.ml.nlp.availability import vader_available
from iad.ml.nlp.embeddings import embed_tfidf_svd
from iad.ml.nlp.sentiment import analyze_sentiment_vader
from iad.ml.nlp.topics import fit_lda_topics


@pytest.fixture
def text_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "review": [
                "great product love it",
                "terrible experience hate it",
                "okay nothing special",
                "amazing quality fast shipping",
                "worst purchase ever",
            ]
        }
    )


@pytest.mark.unit
def test_sentiment_vader(text_df) -> None:
    pytest.importorskip("vaderSentiment")
    if not vader_available():
        pytest.skip("vader not available")
    report = analyze_sentiment_vader(text_df["review"], column="review")
    assert len(report.scores) == 5
    assert "compound" in report.scores.columns


@pytest.mark.unit
def test_embeddings_tfidf(text_df) -> None:
    report = embed_tfidf_svd(text_df["review"], n_components=2)
    assert report.matrix.shape[0] == 5
    assert "dim_1" in report.matrix.columns


@pytest.mark.unit
def test_lda_topics(text_df) -> None:
    report = fit_lda_topics(text_df["review"], n_topics=2)
    assert report.n_topics == 2
    assert len(report.topics) == 2
