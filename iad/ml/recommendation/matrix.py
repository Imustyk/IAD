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
    frame["user"] = frame["user"].astype(str)
    frame["item"] = frame["item"].astype(str)
    frame = frame.dropna(subset=["user", "item", "rating"])
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
