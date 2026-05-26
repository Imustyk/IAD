"""High-level drift detection coordinator.

A ``DriftDetector`` is fit on a *reference* DataFrame (typically the training
set) and then asked to score a *current* DataFrame (typically a recent batch
of inference traffic or a re-uploaded dataset).

For each shared column it picks the right metric automatically:

* numeric column → KS test (statistic + p-value) **and** PSI (binned),
* categorical column → PSI (categorical) **and** JS divergence,

and returns a structured ``DriftReport`` with per-column results plus a
top-level ``overall_drift_detected`` flag based on configurable thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from iad.core.logging import get_logger
from iad.ml.preprocessing.drift.metrics import (
    jensen_shannon_divergence,
    ks_statistic,
    population_stability_index,
)
from iad.ml.preprocessing.exceptions import DriftDetectionError

logger = get_logger("iad.ml.preprocessing.drift.detector")


# Industry-standard PSI interpretation thresholds.
PSI_NO_DRIFT = 0.10
PSI_MODERATE_DRIFT = 0.25


@dataclass(frozen=True)
class ColumnDriftResult:
    """Per-column drift outcome."""

    column: str
    column_kind: Literal["numeric", "categorical"]
    psi: float
    ks_statistic: float | None
    ks_p_value: float | None
    js_divergence: float | None
    drift_detected: bool
    severity: Literal["none", "moderate", "significant"]
    reason: str = ""


@dataclass(frozen=True)
class DriftReport:
    """Full drift report for a (reference, current) pair."""

    n_columns_checked: int
    n_columns_with_drift: int
    overall_drift_detected: bool
    psi_threshold: float
    ks_p_threshold: float
    columns: list[ColumnDriftResult] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    n_reference_rows: int = 0
    n_current_rows: int = 0

    @property
    def drift_share(self) -> float:
        return self.n_columns_with_drift / max(self.n_columns_checked, 1)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "column": c.column,
                    "kind": c.column_kind,
                    "psi": round(c.psi, 4),
                    "ks_stat": None if c.ks_statistic is None else round(c.ks_statistic, 4),
                    "ks_p": None if c.ks_p_value is None else round(c.ks_p_value, 4),
                    "js": None if c.js_divergence is None else round(c.js_divergence, 4),
                    "severity": c.severity,
                    "drift_detected": c.drift_detected,
                }
                for c in self.columns
            ]
        ).sort_values(["drift_detected", "psi"], ascending=[False, False]).reset_index(drop=True)


class DriftDetector:
    """Compare a reference DataFrame against a current DataFrame.

    Args:
        psi_threshold: PSI value above which a column is flagged.
        ks_p_threshold: KS p-value below which a numeric column is flagged.
        max_categorical_cardinality: skip categorical drift when a column has
            more unique values than this — those columns are usually IDs.
        min_sample_size: when either side has fewer than this many rows the
            detector emits a flag and returns NaN scores.
    """

    def __init__(
        self,
        *,
        psi_threshold: float = PSI_MODERATE_DRIFT,
        ks_p_threshold: float = 0.05,
        max_categorical_cardinality: int = 200,
        min_sample_size: int = 30,
    ) -> None:
        self.psi_threshold = psi_threshold
        self.ks_p_threshold = ks_p_threshold
        self.max_categorical_cardinality = max_categorical_cardinality
        self.min_sample_size = min_sample_size
        self._reference: pd.DataFrame | None = None

    def fit(self, reference: pd.DataFrame) -> DriftDetector:
        if not isinstance(reference, pd.DataFrame):
            raise DriftDetectionError("reference must be a pandas DataFrame")
        if reference.empty:
            raise DriftDetectionError("reference DataFrame is empty")
        self._reference = reference.copy()
        logger.info("drift detector fitted", extra={"ctx_n_ref": len(reference)})
        return self

    def detect(self, current: pd.DataFrame) -> DriftReport:
        if self._reference is None:
            raise DriftDetectionError("DriftDetector must be fit() before detect()")
        if not isinstance(current, pd.DataFrame):
            raise DriftDetectionError("current must be a pandas DataFrame")

        ref = self._reference
        flags: list[str] = []

        if len(ref) < self.min_sample_size or len(current) < self.min_sample_size:
            flags.append(
                f"sample size below {self.min_sample_size} (ref={len(ref)}, cur={len(current)})"
            )

        shared = [c for c in ref.columns if c in current.columns]
        if not shared:
            raise DriftDetectionError(
                "no shared columns between reference and current frames",
                ref_columns=list(ref.columns),
                cur_columns=list(current.columns),
            )

        results: list[ColumnDriftResult] = []
        for col in shared:
            kind: Literal["numeric", "categorical"] = (
                "numeric" if pd.api.types.is_numeric_dtype(ref[col]) else "categorical"
            )

            if kind == "categorical":
                cardinality = ref[col].nunique(dropna=True)
                if cardinality > self.max_categorical_cardinality:
                    flags.append(
                        f"skipped column {col!r} (cardinality {cardinality} > "
                        f"{self.max_categorical_cardinality}, likely an ID)"
                    )
                    continue
                psi = population_stability_index(ref[col], current[col], categorical=True)
                js = jensen_shannon_divergence(ref[col], current[col], categorical=True)
                ks_stat = None
                ks_p = None
            else:
                psi = population_stability_index(ref[col], current[col])
                js = None
                ks_stat, ks_p = ks_statistic(ref[col], current[col])

            severity, drift, reason = self._severity(psi, ks_p)
            results.append(
                ColumnDriftResult(
                    column=col,
                    column_kind=kind,
                    psi=psi,
                    ks_statistic=ks_stat,
                    ks_p_value=ks_p,
                    js_divergence=js,
                    drift_detected=drift,
                    severity=severity,
                    reason=reason,
                )
            )

        n_with_drift = sum(1 for r in results if r.drift_detected)
        report = DriftReport(
            n_columns_checked=len(results),
            n_columns_with_drift=n_with_drift,
            overall_drift_detected=n_with_drift > 0,
            psi_threshold=self.psi_threshold,
            ks_p_threshold=self.ks_p_threshold,
            columns=results,
            flags=flags,
            n_reference_rows=len(ref),
            n_current_rows=len(current),
        )
        logger.info(
            "drift report ready",
            extra={
                "ctx_drift_cols": n_with_drift,
                "ctx_total_cols": len(results),
                "ctx_share": round(report.drift_share, 4),
            },
        )
        return report

    # ------------------------------------------------------------------
    def _severity(
        self, psi: float, ks_p: float | None
    ) -> tuple[Literal["none", "moderate", "significant"], bool, str]:
        if psi != psi:  # NaN check
            return "none", False, "psi unavailable"
        if psi >= self.psi_threshold:
            return "significant", True, f"PSI={psi:.3f} >= {self.psi_threshold}"
        if psi >= PSI_NO_DRIFT:
            ks_drift = ks_p is not None and ks_p == ks_p and ks_p < self.ks_p_threshold
            if ks_drift:
                return "moderate", True, f"PSI={psi:.3f}, KS p={ks_p:.3f} < {self.ks_p_threshold}"
            return "moderate", False, f"PSI={psi:.3f} (moderate, not flagged)"
        return "none", False, f"PSI={psi:.3f}"
