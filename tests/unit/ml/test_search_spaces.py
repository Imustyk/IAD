"""Optuna search space registry tests."""
from __future__ import annotations

import optuna
import pytest

from iad.ml.tuning.search_spaces import _SEARCH_SPACES, has_search_space, suggest_params


@pytest.mark.unit
@pytest.mark.parametrize("model_name", list(_SEARCH_SPACES.keys()))
def test_suggest_params_all_models(model_name: str) -> None:
    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    params = suggest_params(model_name, trial)
    assert isinstance(params, dict)
    assert params


@pytest.mark.unit
def test_has_search_space() -> None:
    assert has_search_space("Random Forest")
    assert not has_search_space("Unknown Model XYZ")


@pytest.mark.unit
def test_suggest_params_unknown_raises() -> None:
    study = optuna.create_study()
    trial = study.ask()
    with pytest.raises(KeyError):
        suggest_params("Not A Model", trial)
