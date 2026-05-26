"""Recommendation service."""
from __future__ import annotations

import pandas as pd

from iad.ml.recommendation.collaborative import user_based_collaborative_filtering
from iad.ml.recommendation.reports import RecommendationReport
from iad.ml.recommendation.similarity import cosine_item_recommendations


class RecommendationService:
    """Collaborative filtering and similarity recommendations."""

    def user_collaborative(
        self,
        df: pd.DataFrame,
        *,
        user_column: str,
        item_column: str,
        rating_column: str,
        target_user: str | int,
        top_n: int = 10,
    ) -> RecommendationReport:
        return user_based_collaborative_filtering(
            df,
            user_column=user_column,
            item_column=item_column,
            rating_column=rating_column,
            target_user=target_user,
            top_n=top_n,
        )

    def cosine_similarity(
        self,
        df: pd.DataFrame,
        *,
        user_column: str,
        item_column: str,
        rating_column: str,
        target_user: str | int,
        top_n: int = 10,
    ) -> RecommendationReport:
        return cosine_item_recommendations(
            df,
            user_column=user_column,
            item_column=item_column,
            rating_column=rating_column,
            target_user=target_user,
            top_n=top_n,
        )
