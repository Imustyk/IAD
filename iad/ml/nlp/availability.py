"""Optional NLP dependency probes."""
from __future__ import annotations

_VADER: bool | None = None
_SENTENCE_TRANSFORMERS: bool | None = None


def vader_available() -> bool:
    global _VADER
    if _VADER is None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # noqa: F401

            _VADER = True
        except ImportError:
            _VADER = False
    return _VADER


def sentence_transformers_available() -> bool:
    global _SENTENCE_TRANSFORMERS
    if _SENTENCE_TRANSFORMERS is None:
        try:
            from sentence_transformers import SentenceTransformer  # noqa: F401

            _SENTENCE_TRANSFORMERS = True
        except ImportError:
            _SENTENCE_TRANSFORMERS = False
    return _SENTENCE_TRANSFORMERS
