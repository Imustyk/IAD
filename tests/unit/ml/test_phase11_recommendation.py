"""Phase 11 recommendation tests."""
from __future__ import annotations

import pandas as pd
import pytest

from iad.ml.recommendation.collaborative import user_based_collaborative_filtering
from iad.ml.recommendation.similarity import cosine_item_recommendations


@pytest.fixture
def interactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user": ["u1", "u1", "u2", "u2", "u3", "u3"],
            "item": ["a", "b", "a", "c", "b", "c"],
            "rating": [5.0, 3.0, 4.0, 2.0, 5.0, 1.0],
        }
    )


@pytest.mark.unit
def test_user_collaborative(interactions) -> None:
    report = user_based_collaborative_filtering(
        interactions,
        user_column="user",
        item_column="item",
        rating_column="rating",
        target_user="u1",
        top_n=3,
    )
    assert not report.recommendations.empty


@pytest.mark.unit
def test_cosine_item(interactions) -> None:
    report = cosine_item_recommendations(
        interactions,
        user_column="user",
        item_column="item",
        rating_column="rating",
        target_user="u1",
        top_n=3,
    )
    assert report.method == "cosine_item"
