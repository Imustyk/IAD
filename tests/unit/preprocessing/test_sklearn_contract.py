"""sklearn contract compliance.

Every transformer must round-trip through ``sklearn.base.clone`` so it can
participate in cross-validation, GridSearchCV, OptunaSearchCV, etc. The
``clone`` function calls ``__init__`` with the exact parameters returned by
``get_params`` and then asserts ``init(p).get_params() == p``. Any transformer
that normalises its arguments inside ``__init__`` (mutable defaults,
``None`` → ``[]``, type coercion) breaks this contract.
"""
from __future__ import annotations

import pytest
from sklearn.base import clone

from iad.ml.preprocessing import (
    AutoFeatureSelector,
    DatetimeFeatureExtractor,
    MulticollinearityReducer,
    RareCategoryGrouper,
    SkewnessCorrector,
    SmoothedTargetEncoder,
)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: DatetimeFeatureExtractor(),
        lambda: DatetimeFeatureExtractor(columns=["ts"], drop_original=False, coerce=False),
        lambda: RareCategoryGrouper(),
        lambda: RareCategoryGrouper(columns=["a"], min_frequency=0.05, max_categories=10),
        lambda: SkewnessCorrector(),
        lambda: SkewnessCorrector(skew_threshold=0.5),
        lambda: MulticollinearityReducer(),
        lambda: MulticollinearityReducer(threshold=0.9, method="spearman", protect=["x"]),
        lambda: MulticollinearityReducer(threshold=0.9, method="spearman", protect=None),
        lambda: SmoothedTargetEncoder(),
        lambda: SmoothedTargetEncoder(columns=["a"], smoothing=2.0, n_folds=3),
        lambda: AutoFeatureSelector(),
        lambda: AutoFeatureSelector(task="regression", max_features=5),
    ],
)
def test_transformer_is_clone_safe(factory) -> None:
    instance = factory()
    cloned = clone(instance)
    assert cloned is not instance
    assert cloned.get_params() == instance.get_params()
