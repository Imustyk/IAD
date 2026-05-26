"""User-item matrix utilities."""
from __future__ import annotations

import pandas as pd

from iad.core.exceptions import SchemaError


def build_user_item_matrix(
    df: pd.DataFrame,
    *,
    user_column: str,
    item_column: str,
    rating_column: str,
) -> pd.DataFrame:
    """Pivot interactions into a users × items matrix (mean rating if duplicates)."""
    for col in (user_column, item_column, rating_column):
        if col not in df.columns:
            raise SchemaError(
                f"Column {col!r} not found.",
                user_message="Map user, item, and rating columns.",
            )

    frame = df[[user_column, item_column, rating_column]].copy()
    frame[rating_column] = pd.to_numeric(frame[rating_column], errors="coerce")
    frame = frame.dropna()
    if frame.empty:
        raise SchemaError(
            "No valid interactions after cleaning.",
            user_message="Check rating column is numeric.",
        )

    matrix = frame.pivot_table(
        index=user_column,
        columns=item_column,
        values=rating_column,
        aggfunc="mean",
    )
    return matrix
