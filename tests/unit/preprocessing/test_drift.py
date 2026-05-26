"""Drift detection — metrics + DriftDetector."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from iad.ml.preprocessing import DriftDetectionError, DriftDetector
from iad.ml.preprocessing.drift.metrics import (
    chi_square_drift,
    jensen_shannon_divergence,
    ks_statistic,
    population_stability_index,
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def test_ks_statistic_zero_for_identical() -> None:
    x = np.linspace(0, 1, 1000)
    stat, p = ks_statistic(x, x)
    assert stat == 0.0
    assert p > 0.99


def test_ks_statistic_one_for_disjoint() -> None:
    a = np.zeros(1000)
    b = np.ones(1000)
    stat, p = ks_statistic(a, b)
    assert stat > 0.9
    assert p < 0.01


def test_psi_zero_for_identical() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000)
    b = rng.normal(size=5000)
    psi = population_stability_index(a, b, bins=10)
    assert psi < 0.1


def test_psi_increases_for_shift() -> None:
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, size=5000)
    b = rng.normal(2, 1, size=5000)
    psi = population_stability_index(a, b, bins=10)
    assert psi > 0.25


def test_psi_categorical() -> None:
    a = pd.Series(["A"] * 70 + ["B"] * 30)
    b = pd.Series(["A"] * 30 + ["B"] * 70)
    psi = population_stability_index(a, b, categorical=True)
    assert psi > 0.25


def test_jensen_shannon_zero_for_identical_categorical() -> None:
    s = pd.Series(["x", "x", "y", "y", "z"])
    assert jensen_shannon_divergence(s, s, categorical=True) == pytest.approx(0.0, abs=1e-9)


def test_jensen_shannon_positive_when_different() -> None:
    a = pd.Series(["x", "x", "x", "y"])
    b = pd.Series(["y", "y", "y", "x"])
    assert jensen_shannon_divergence(a, b, categorical=True) > 0


def test_chi_square_p_value_low_for_drift() -> None:
    a = pd.Series(["A"] * 90 + ["B"] * 10)
    b = pd.Series(["A"] * 30 + ["B"] * 70)
    stat, p = chi_square_drift(a, b)
    assert stat > 0
    assert p < 0.05


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------
def test_detector_requires_fit() -> None:
    detector = DriftDetector()
    with pytest.raises(DriftDetectionError):
        detector.detect(pd.DataFrame({"a": [1, 2, 3]}))


def test_detector_no_drift_on_resampled_iris() -> None:
    from src.data_loader import load_sample

    df = load_sample("Iris (classification)")
    detector = DriftDetector(psi_threshold=0.25, ks_p_threshold=0.05).fit(df)
    sample = df.sample(frac=0.5, random_state=42)
    report = detector.detect(sample)
    assert report.n_columns_checked > 0
    # Resampled data must not look drifted on the majority of columns.
    assert report.drift_share < 0.4


def test_detector_detects_synthetic_drift() -> None:
    rng = np.random.default_rng(42)
    ref = pd.DataFrame(
        {
            "income": rng.normal(50_000, 10_000, size=2000),
            "country": rng.choice(["US", "UK", "DE", "FR"], size=2000, p=[0.5, 0.2, 0.2, 0.1]),
        }
    )
    cur = pd.DataFrame(
        {
            "income": rng.normal(80_000, 10_000, size=2000),  # mean shifted
            "country": rng.choice(
                ["US", "UK", "DE", "FR"], size=2000, p=[0.1, 0.2, 0.2, 0.5]
            ),  # distribution flipped
        }
    )
    detector = DriftDetector(psi_threshold=0.25).fit(ref)
    report = detector.detect(cur)
    assert report.overall_drift_detected
    drifted_columns = {c.column for c in report.columns if c.drift_detected}
    assert "income" in drifted_columns
    assert "country" in drifted_columns


def test_detector_skips_high_cardinality_columns() -> None:
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"id": [f"u_{i}" for i in range(500)], "x": rng.normal(size=500)})
    cur = pd.DataFrame({"id": [f"u_{i + 1000}" for i in range(500)], "x": rng.normal(size=500)})
    detector = DriftDetector(max_categorical_cardinality=50).fit(ref)
    report = detector.detect(cur)
    columns = {c.column for c in report.columns}
    assert "id" not in columns  # skipped
    assert "x" in columns


def test_detector_flags_small_samples() -> None:
    ref = pd.DataFrame({"x": np.linspace(0, 1, 1000)})
    cur = pd.DataFrame({"x": np.linspace(0, 1, 5)})
    detector = DriftDetector(min_sample_size=30).fit(ref)
    report = detector.detect(cur)
    assert any("sample size" in f for f in report.flags)


def test_detector_raises_on_no_shared_columns() -> None:
    ref = pd.DataFrame({"a": [1, 2, 3]})
    cur = pd.DataFrame({"b": [1, 2, 3]})
    detector = DriftDetector().fit(ref)
    with pytest.raises(DriftDetectionError):
        detector.detect(cur)


def test_drift_report_to_frame_has_expected_columns() -> None:
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"x": rng.normal(size=500)})
    cur = pd.DataFrame({"x": rng.normal(size=500) + 0.5})
    detector = DriftDetector().fit(ref)
    report = detector.detect(cur)
    df = report.to_frame()
    assert {"column", "kind", "psi", "ks_stat", "severity", "drift_detected"}.issubset(df.columns)
