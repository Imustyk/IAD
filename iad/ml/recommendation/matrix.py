"""User-item matrix utilities."""
from __future__ import annotations

import pandas as pd

from iad.core.exceptions import SchemaError


def _select_column(df: pd.DataFrame, column: str) -> pd.Series:
    """Return a 1-D series for ``column`` (handles duplicate column names in ``df``)."""
    if column not in df.columns:
        raise SchemaError(
            f"Column {column!r} not found.",
            user_message="Map user, item, and rating columns.",
        )
    selected = df[column]
    if isinstance(selected, pd.DataFrame):
        if selected.shape[1] != 1:
            raise SchemaError(
                f"Column {column!r} is ambiguous (duplicate names in the dataset).",
                user_message="User, item, and rating must be three different columns.",
            )
        selected = selected.iloc[:, 0]
    return selected.squeeze()


def normalize_entity_id(value: object) -> str:
    """Consistent string key for user/item IDs (matches pivot index labels)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    text = str(value).strip()
    if text.endswith(".0") and text.replace(".0", "", 1).replace("-", "", 1).isdigit():
        return text[:-2]
    return text


def list_interaction_users(
    df: pd.DataFrame,
    *,
    user_column: str,
    item_column: str,
    rating_column: str,
) -> list[str]:
    """Users with at least one valid rating after matrix cleaning."""
    matrix = build_user_item_matrix(
        df,
        user_column=user_column,
        item_column=item_column,
        rating_column=rating_column,
    )
    return list(matrix.index)


def build_user_item_matrix(
    df: pd.DataFrame,
    *,
    user_column: str,
    item_column: str,
    rating_column: str,
) -> pd.DataFrame:
    """Pivot interactions into a users × items matrix (mean rating if duplicates)."""
    if len({user_column, item_column, rating_column}) < 3:
        raise SchemaError(
            "User, item, and rating columns must be distinct.",
            user_message="Choose three different columns for user, item, and rating.",
        )

    frame = pd.DataFrame({
        "user": _select_column(df, user_column),
        "item": _select_column(df, item_column),
        "rating": _select_column(df, rating_column),
    })

    frame["rating"] = pd.to_numeric(frame["rating"], errors="coerce")
    frame["user"] = frame["user"].map(normalize_entity_id)
    frame["item"] = frame["item"].map(normalize_entity_id)
    frame = frame.dropna(subset=["user", "item", "rating"])
    frame = frame[(frame["user"] != "") & (frame["item"] != "")]
    if frame.empty:
        raise SchemaError(
            "No valid interactions after cleaning.",
            user_message="Rating column must be numeric. User and item columns need valid values.",
        )

    matrix = frame.pivot_table(
        index="user",
        columns="item",
        values="rating",
        aggfunc="mean",
    )
    return matrix
