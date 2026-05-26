"""Export metric dictionaries to CSV or JSON."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from iad.export.models import ExportFormat, ExportResult


def export_metrics_table(
    metrics: dict[str, float],
    destination: Path,
    *,
    fmt: ExportFormat = ExportFormat.CSV,
) -> ExportResult:
    """Serialize a flat metrics dict to CSV or JSON."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if fmt == ExportFormat.JSON:
        destination.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        content_type = "application/json"
    elif fmt == ExportFormat.CSV:
        frame = pd.DataFrame([{"metric": k, "value": v} for k, v in metrics.items()])
        frame.to_csv(destination, index=False)
        content_type = "text/csv"
    else:
        raise ValueError(f"metrics export does not support {fmt}")

    return ExportResult(
        format=fmt,
        path=destination,
        size_bytes=destination.stat().st_size,
        content_type=content_type,
    )
