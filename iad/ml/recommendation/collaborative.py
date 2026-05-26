"""User-based collaborative filtering."""
from __future__ import annotations

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from iad.core.exceptions import AnalyticsError, SchemaError
from iad.core.logging import get_logger
from iad.ml.recommendation.matrix import build_user_item_matrix, normalize_entity_id
from iad.ml.recommendation.reports import RecommendationReport

logger = get_logger("iad.ml.recommendation.collaborative")


def user_based_collaborative_filtering(
    df: pd.DataFrame,
    *,
    user_column: str,
    item_column: str,
    rating_column: str,
    target_user: str | int,
    top_n: int = 10,
    min_common: int = 1,
) -> RecommendationReport:
    """Recommend items using similar users (user-user collaborative filtering)."""
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

    filled = matrix.fillna(0.0)
    user_sim = cosine_similarity(filled.values)
    users = list(matrix.index)
    sim_df = pd.DataFrame(user_sim, index=users, columns=users)

    target_idx = users.index(target_key)
    similarities = sim_df.iloc[target_idx].drop(target_key)
    neighbors = similarities[similarities > 0].sort_values(ascending=False)
    if neighbors.empty:
        raise AnalyticsError(
            "No similar users found.",
            user_message="Try cosine item similarity or add more interactions.",
        )

    target_rated = set(matrix.loc[target_key].dropna().index)
    candidate_scores: dict[object, float] = {}
    weight_sums: dict[object, float] = {}

    for neighbor, sim in neighbors.items():
        if sim < 1e-9:
            continue
        neighbor_ratings = matrix.loc[neighbor].dropna()
        for item, rating in neighbor_ratings.items():
            if item in target_rated:
                continue
            candidate_scores[item] = candidate_scores.get(item, 0.0) + float(sim * rating)
            weight_sums[item] = weight_sums.get(item, 0.0) + float(abs(sim))

    filtered = {
        item: candidate_scores[item] / weight_sums[item]
        for item in candidate_scores
        if weight_sums.get(item, 0.0) >= min_common
    }

    ranked = (
        pd.Series(filtered, name="predicted_rating")
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
        .rename(columns={"index": item_column})
    )

    logger.info(
        "user-based CF",
        extra={"user": target_key, "neighbors": len(neighbors), "n_recs": len(ranked)},
    )
    return RecommendationReport(
        method="user_collaborative",
        target_user=target_key,
        recommendations=ranked,
        metrics={
            "neighbor_count": float(len(neighbors)),
            "candidate_items": float(len(filtered)),
        },
        similarity_matrix=sim_df,
    )
