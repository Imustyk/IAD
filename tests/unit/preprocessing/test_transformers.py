"""Unit tests for every Phase 2 transformer."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from iad.ml.preprocessing import (
    AutoFeatureSelector,
    DatetimeFeatureExtractor,
    MulticollinearityReducer,
    RareCategoryGrouper,
    SkewnessCorrector,
    SmoothedTargetEncoder,
    TransformerNotFittedError,
)


# ---------------------------------------------------------------------------
# DatetimeFeatureExtractor
# ---------------------------------------------------------------------------
def test_datetime_extractor_creates_calendar_features() -> None:
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2024-01-15", "2024-06-30", "2024-12-25"]),
            "x": [1, 2, 3],
        }
    )
    extractor = DatetimeFeatureExtractor().fit(df)
    out = extractor.transform(df)
    assert "ts__year" in out.columns
    assert "ts__month" in out.columns
    assert "ts__is_weekend" in out.columns
    assert "ts" not in out.columns  # dropped by default
    assert out.loc[0, "ts__year"] == 2024
    assert out.loc[0, "ts__month"] == 1


def test_datetime_extractor_preserves_when_drop_original_false() -> None:
    df = pd.DataFrame({"ts": pd.to_datetime(["2024-01-15"])})
    out = DatetimeFeatureExtractor(drop_original=False).fit_transform(df)
    assert "ts" in out.columns
    assert "ts__year" in out.columns


def test_datetime_extractor_coerces_strings_when_enabled() -> None:
    df = pd.DataFrame({"ts": ["2024-03-01", "2024-04-01"]})
    out = DatetimeFeatureExtractor(columns=["ts"], coerce=True).fit_transform(df)
    assert "ts__year" in out.columns
    assert out.loc[0, "ts__month"] == 3


def test_datetime_extractor_raises_if_used_unfitted() -> None:
    extractor = DatetimeFeatureExtractor()
    with pytest.raises(TransformerNotFittedError):
        extractor.transform(pd.DataFrame({"ts": pd.to_datetime(["2024-01-01"])}))


# ---------------------------------------------------------------------------
# RareCategoryGrouper
# ---------------------------------------------------------------------------
def test_rare_category_grouper_replaces_below_threshold() -> None:
    df = pd.DataFrame({"city": ["NY"] * 90 + ["LA"] * 8 + ["TX"] * 1 + ["XX"] * 1})
    grouper = RareCategoryGrouper(min_frequency=0.05).fit(df)
    out = grouper.transform(df)
    assert set(out["city"].unique()) == {"NY", "LA", "Other"}


def test_rare_category_max_categories_caps_keep_set() -> None:
    df = pd.DataFrame(
        {"x": ["a"] * 50 + ["b"] * 30 + ["c"] * 15 + ["d"] * 5}
    )
    grouper = RareCategoryGrouper(min_frequency=0.0, max_categories=2).fit(df)
    out = grouper.transform(df)
    kept = set(out["x"].unique())
    assert "Other" in kept
    assert len(kept - {"Other"}) <= 2


def test_rare_category_grouper_unfitted_raises() -> None:
    with pytest.raises(TransformerNotFittedError):
        RareCategoryGrouper().transform(pd.DataFrame({"x": ["a"]}))


def test_rare_category_grouper_validates_inputs() -> None:
    with pytest.raises(ValueError):
        RareCategoryGrouper(min_frequency=2.0)
    with pytest.raises(ValueError):
        RareCategoryGrouper(max_categories=0)


# ---------------------------------------------------------------------------
# SkewnessCorrector
# ---------------------------------------------------------------------------
def test_skewness_corrector_reduces_skew() -> None:
    rng = np.random.default_rng(0)
    skewed = rng.exponential(scale=2.0, size=500)
    df = pd.DataFrame({"x": skewed})
    before = float(df["x"].skew())
    out = SkewnessCorrector(skew_threshold=0.5).fit_transform(df)
    after = float(out["x"].skew())
    assert before > 1.0
    assert abs(after) < abs(before)


def test_skewness_corrector_handles_negative_values() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=200) * 10})
    df.loc[0:5, "x"] = -50  # heavy negative tail
    out = SkewnessCorrector(skew_threshold=0.0).fit_transform(df)
    assert np.isfinite(out["x"]).all()


def test_skewness_corrector_passes_through_low_skew() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=500)})
    sk = SkewnessCorrector(skew_threshold=2.0).fit(df)
    assert sk.skewed_columns_ == []
    out = sk.transform(df)
    pd.testing.assert_series_equal(out["x"], df["x"], check_dtype=False)


# ---------------------------------------------------------------------------
# MulticollinearityReducer
# ---------------------------------------------------------------------------
def test_multicollinearity_drops_highly_correlated() -> None:
    rng = np.random.default_rng(0)
    base = rng.normal(size=200)
    df = pd.DataFrame(
        {
            "a": base,
            "b": base + 1e-6 * rng.normal(size=200),  # essentially a copy
            "c": rng.normal(size=200),
        }
    )
    reducer = MulticollinearityReducer(threshold=0.99).fit(df)
    out = reducer.transform(df)
    assert "a" in out.columns or "b" in out.columns
    assert not ({"a", "b"}.issubset(out.columns))


def test_multicollinearity_protect_keeps_column() -> None:
    rng = np.random.default_rng(0)
    base = rng.normal(size=200)
    df = pd.DataFrame({"a": base, "b": base + 1e-6 * rng.normal(size=200)})
    reducer = MulticollinearityReducer(threshold=0.99, protect=["a"]).fit(df)
    out = reducer.transform(df)
    assert "a" in out.columns


def test_multicollinearity_validation() -> None:
    with pytest.raises(ValueError):
        MulticollinearityReducer(threshold=1.5)
    with pytest.raises(ValueError):
        MulticollinearityReducer(method="invalid")


# ---------------------------------------------------------------------------
# SmoothedTargetEncoder — the leakage-safety test
# ---------------------------------------------------------------------------
def test_target_encoder_basic_fit_transform() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"city": rng.choice(["A", "B", "C"], size=300)})
    y = pd.Series(rng.integers(0, 2, size=300))
    enc = SmoothedTargetEncoder(columns=["city"], smoothing=5.0, n_folds=5).fit_transform(df, y)
    assert "city__te" in enc.columns
    assert enc["city__te"].notna().all()


def test_target_encoder_does_not_leak() -> None:
    """Encoded value of a row must not equal the row's own target.

    The classic leakage: with no smoothing and no K-folding, a unique
    category whose only row has y=1 would be encoded as 1.0 — directly
    revealing the label. Smoothed K-fold encoding must avoid this.
    """
    df = pd.DataFrame({"cat": [f"cat_{i}" for i in range(20)]})
    y = pd.Series(np.arange(20) % 2)  # alternating 0/1
    enc = SmoothedTargetEncoder(columns=["cat"], smoothing=10.0, n_folds=5)
    encoded = enc.fit_transform(df, y)
    # Encoded values should be close to the global mean (0.5) thanks to smoothing,
    # not equal to the row-specific target.
    assert (encoded["cat__te"].abs() < 1.0).all()
    assert (encoded["cat__te"].between(0.2, 0.8)).all()


def test_target_encoder_unseen_category_falls_back_to_global_mean() -> None:
    df_train = pd.DataFrame({"cat": ["A"] * 50 + ["B"] * 50})
    y = pd.Series([1] * 50 + [0] * 50)
    enc = SmoothedTargetEncoder(columns=["cat"], smoothing=5.0).fit(df_train, y)
    new = pd.DataFrame({"cat": ["A", "Z", "B"]})
    out = enc.transform(new)
    expected_global = float(y.mean())
    # Z is unseen → must equal global mean (within smoothing tolerance).
    assert out.loc[1, "cat__te"] == pytest.approx(expected_global)


def test_target_encoder_requires_y() -> None:
    enc = SmoothedTargetEncoder(columns=["x"])
    with pytest.raises(Exception):
        enc.fit(pd.DataFrame({"x": ["a", "b"]}), None)


def test_target_encoder_validates_params() -> None:
    with pytest.raises(ValueError):
        SmoothedTargetEncoder(smoothing=-1.0)
    with pytest.raises(ValueError):
        SmoothedTargetEncoder(n_folds=1)


# ---------------------------------------------------------------------------
# AutoFeatureSelector
# ---------------------------------------------------------------------------
def test_feature_selector_drops_low_variance() -> None:
    df = pd.DataFrame(
        {
            "useful": np.random.default_rng(0).normal(size=200),
            "constant": [1.0] * 200,
        }
    )
    y = pd.Series(np.random.default_rng(0).integers(0, 2, size=200))
    sel = AutoFeatureSelector(
        task="classification", variance_threshold=0.0, use_model_based=False
    ).fit(df, y)
    assert "constant" not in sel.selected_


def test_feature_selector_drops_correlated() -> None:
    rng = np.random.default_rng(0)
    base = rng.normal(size=200)
    df = pd.DataFrame({"a": base, "b": base + 1e-9 * rng.normal(size=200)})
    y = pd.Series(rng.integers(0, 2, size=200))
    sel = AutoFeatureSelector(
        task="classification", correlation_threshold=0.99, use_model_based=False
    ).fit(df, y)
    assert len(sel.selected_) == 1


def test_feature_selector_caps_max_features() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({f"f_{i}": rng.normal(size=200) for i in range(10)})
    y = pd.Series(rng.integers(0, 2, size=200))
    sel = AutoFeatureSelector(task="classification", max_features=3).fit(df, y)
    assert len(sel.selected_) <= 3
