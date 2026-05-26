"""Lightweight run metadata for the tracker."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RunMetadata:
    """Returned by :class:`MLflowTracker.__exit__` for downstream services."""

    run_id: str
    experiment_id: str
    experiment_name: str
    tracking_uri: str
    artifact_uri: str | None = None
