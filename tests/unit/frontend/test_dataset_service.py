"""Frontend dataset service tests."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from iad.state.session import KEY_DATASET, KEY_DATASET_NAME, init_session_state, state_get


@pytest.fixture(autouse=True)
def _session():
    init_session_state()
    yield


@pytest.mark.unit
def test_store_dataset(iris_df) -> None:
    from iad.frontend.services.dataset import store_dataset

    view = store_dataset(iris_df, "iris-test")
    assert view.total_rows == len(iris_df)
    assert state_get(KEY_DATASET) is not None
    assert state_get(KEY_DATASET_NAME) == "iris-test"


@pytest.mark.unit
def test_load_uploaded(iris_df, monkeypatch) -> None:
    from iad.frontend.services.dataset import load_uploaded

    uploaded = MagicMock()
    uploaded.name = "iris.csv"
    uploaded.read.return_value = iris_df.to_csv(index=False).encode()
    view = load_uploaded(uploaded)
    assert view.total_rows == len(iris_df)


@pytest.mark.unit
def test_load_from_legacy_loader(iris_df) -> None:
    from iad.frontend.services.dataset import load_from_legacy_loader

    view = load_from_legacy_loader(lambda: iris_df.copy(), "legacy")
    assert view.footprint.columns == len(iris_df.columns)


@pytest.mark.unit
def test_loader_must_return_dataframe() -> None:
    from iad.frontend.services.dataset import load_from_legacy_loader

    with pytest.raises(TypeError):
        load_from_legacy_loader(lambda: [1, 2, 3], "bad")  # type: ignore[list-item]
