"""Item-based cosine similarity recommendations."""
from __future__ import annotations

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from iad.core.exceptions import AnalyticsError, SchemaError
from iad.core.logging import get_logger
from iad.ml.recommendation.matrix import build_user_item_matrix, normalize_entity_id
from iad.ml.recommendation.reports import RecommendationReport

logger = get_logger("iad.ml.recommendation.similarity")


def cosine_item_recommendations(
    df: pd.DataFrame,
    *,
    user_column: str,
    item_column: str,
    rating_column: str,
    target_user: str | int,
    top_n: int = 10,
) -> RecommendationReport:
    """Recommend items similar to those the user already rated (item-item cosine)."""
    matrix = build_user_item_matrix(
        df,
        user_column=user_column,
        item_column=item_column,
        rating_column=rating_column,
    )
    target_key = normalize_entity_id(target_user)
    if target_key not in matrix.index:
        raise SchemaError(
            f"User {target_user!r} not in interaction data.",
            user_message="Pick a user with valid ratings in the selected columns.",
        )

    item_vectors = matrix.fillna(0).T.values
    items = list(matrix.columns)
    sim = cosine_similarity(item_vectors)
    sim_df = pd.DataFrame(sim, index=items, columns=items)

    user_ratings = matrix.loc[target_key].dropna()
    if user_ratings.empty:
        raise AnalyticsError(
            "User has no ratings.",
            user_message="Choose a user with interaction history.",
        )

    scores: dict[object, float] = {}
    rated_items = set(user_ratings.index)
    for item in items:
        if item in rated_items:
            continue
        similar = sim_df[item].drop(index=list(rated_items), errors="ignore")
        if similar.empty:
            continue
        weighted = float((similar * user_ratings.reindex(similar.index).fillna(0)).sum())
        norm = float(similar.abs().sum())
        scores[item] = weighted / norm if norm > 0 else 0.0

    ranked = (
        pd.Series(scores, name="score")
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
        .rename(columns={"index": item_column})
    )

    logger.info(
        "cosine item recommendations",
        extra={"user": target_key, "n_recs": len(ranked)},
    )
    return RecommendationReport(
        method="cosine_item",
        target_user=target_key,
        recommendations=ranked,
        metrics={"candidate_items": float(len(scores))},
        similarity_matrix=sim_df,
    )
