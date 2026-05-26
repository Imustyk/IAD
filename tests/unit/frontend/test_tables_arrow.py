"""Arrow-safe dataframe display tests."""
from __future__ import annotations

import pandas as pd
import pytest


@pytest.mark.unit
def test_arrow_safe_mixed_object_column() -> None:
    from iad.frontend.components.tables import arrow_safe_dataframe

    df = pd.DataFrame({"sample": ["malignant", "3.14", "benign"]})
    safe = arrow_safe_dataframe(df)
    assert all(isinstance(v, str) for v in safe["sample"])


@pytest.mark.unit
def test_format_schema_sample() -> None:
    from iad.frontend.components.tables import format_schema_sample

    assert format_schema_sample(pd.Series(["malignant", "benign"])) == "malignant"
    assert format_schema_sample(pd.Series([1.5, 2.0])) == "1.5"
    assert format_schema_sample(pd.Series([pd.NA], dtype="object")) == ""
