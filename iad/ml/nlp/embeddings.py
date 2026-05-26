"""Text embeddings — TF-IDF + SVD baseline, optional sentence-transformers."""
from __future__ import annotations

import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from iad.core.exceptions import AnalyticsError, SchemaError
from iad.core.logging import get_logger
from iad.ml.nlp.availability import sentence_transformers_available
from iad.ml.nlp.reports import EmbeddingReport

logger = get_logger("iad.ml.nlp.embeddings")


def embed_tfidf_svd(
    series: pd.Series,
    *,
    column: str | None = None,
    n_components: int = 2,
    max_features: int = 5000,
    random_state: int = 42,
) -> EmbeddingReport:
    """Lightweight embeddings via TF-IDF + truncated SVD (no GPU)."""
    col = column or str(series.name or "text")
    text = series.fillna("").astype(str)
    if text.str.strip().eq("").all():
        raise SchemaError(
            f"Column {col!r} has no non-empty text.",
            user_message="Choose a text column with content.",
        )

    n_components = min(n_components, max(1, len(text) - 1))
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    matrix = vectorizer.fit_transform(text)
    if matrix.shape[1] == 0:
        raise AnalyticsError(
            "TF-IDF produced no features.",
            user_message="Text may be too short or repetitive for embedding.",
        )

    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    reduced = svd.fit_transform(matrix)
    columns = [f"dim_{i + 1}" for i in range(reduced.shape[1])]
    frame = pd.DataFrame(reduced, index=text.index, columns=columns)
    frame.insert(0, "text_preview", text.str.slice(0, 80))

    return EmbeddingReport(
        column=col,
        matrix=frame,
        method="tfidf_svd",
        n_features=int(matrix.shape[1]),
        model_name=f"tfidf({max_features})+svd({n_components})",
    )


def embed_sentence_transformer(  # pragma: no cover — optional dependency; install `.[nlp]`
    series: pd.Series,
    *,
    column: str | None = None,
    model_name: str = "all-MiniLM-L6-v2",
    n_components: int | None = 2,
    random_state: int = 42,
) -> EmbeddingReport:
    """Dense embeddings via sentence-transformers (optional dependency)."""
    if not sentence_transformers_available():
        raise AnalyticsError(
            "sentence-transformers is not installed.",
            user_message="Install NLP extras: pip install sentence-transformers",
        )

    from sentence_transformers import SentenceTransformer

    col = column or str(series.name or "text")
    text = series.fillna("").astype(str).tolist()
    model = SentenceTransformer(model_name)
    vectors = model.encode(text, show_progress_bar=False)

    if n_components and n_components < vectors.shape[1]:
        svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        vectors = svd.fit_transform(vectors)

    columns = [f"dim_{i + 1}" for i in range(vectors.shape[1])]
    frame = pd.DataFrame(vectors, index=series.index, columns=columns)
    frame.insert(0, "text_preview", series.fillna("").astype(str).str.slice(0, 80))
    logger.info("sentence-transformer embeddings", extra={"column": col, "model": model_name})
    return EmbeddingReport(
        column=col,
        matrix=frame,
        method="sentence_transformer",
        n_features=vectors.shape[1],
        model_name=model_name,
    )


def embed_text_column(
    df: pd.DataFrame,
    column: str,
    *,
    method: str = "tfidf_svd",
    n_components: int = 2,
    model_name: str = "all-MiniLM-L6-v2",
) -> EmbeddingReport:
    """Create embeddings for a text column."""
    if column not in df.columns:
        raise SchemaError(
            f"Column {column!r} not found.",
            user_message="Select a valid text column.",
        )
    series = df[column]
    if method == "tfidf_svd":
        return embed_tfidf_svd(series, column=column, n_components=n_components)
    if method == "sentence_transformer":
        return embed_sentence_transformer(
            series, column=column, model_name=model_name, n_components=n_components
        )
    raise AnalyticsError(
        f"Unknown embedding method {method!r}.",
        user_message="Choose tfidf_svd or sentence_transformer.",
    )
