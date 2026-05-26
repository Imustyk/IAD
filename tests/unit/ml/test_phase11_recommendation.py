"""Phase 11 recommendation tests."""
from __future__ import annotations

import pandas as pd
import pytest

from iad.ml.recommendation.collaborative import user_based_collaborative_filtering
from iad.ml.recommendation.matrix import list_interaction_users, normalize_entity_id
from iad.ml.recommendation.similarity import cosine_item_recommendations


@pytest.mark.unit
def test_normalize_entity_id_coerces_numeric_types() -> None:
    assert normalize_entity_id(3) == "3"
    assert normalize_entity_id(3.0) == "3"
    assert normalize_entity_id("3") == "3"


@pytest.mark.unit
def test_user_collaborative_accepts_int_target_user() -> None:
    df = pd.DataFrame(
        {
            "user": [1, 1, 2, 2, 3, 3],
            "item": ["a", "b", "a", "c", "b", "c"],
            "rating": [5.0, 3.0, 4.0, 2.0, 5.0, 1.0],
        }
    )
    report = user_based_collaborative_filtering(
        df,
        user_column="user",
        item_column="item",
        rating_column="rating",
        target_user=3,
        top_n=3,
    )
    assert report.target_user == "3"
    assert report.method == "user_collaborative"


@pytest.mark.unit
def test_list_interaction_users_excludes_invalid_ratings() -> None:
    df = pd.DataFrame(
        {
            "user": [1, 2, 3],
            "item": ["a", "b", "c"],
            "rating": [5.0, None, "bad"],
        }
    )
    users = list_interaction_users(
        df,
        user_column="user",
        item_column="item",
        rating_column="rating",
    )
    assert users == ["1"]


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
