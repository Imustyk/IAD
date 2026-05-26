"""Unit tests for metric card specs (no Streamlit runtime)."""
from __future__ import annotations

from iad.frontend.components.metric_cards import MetricSpec


def test_metric_spec_frozen() -> None:
    spec = MetricSpec("Rows", "1,000", delta="+5%", delta_direction="positive", icon="📋")
    assert spec.label == "Rows"
    assert spec.value == "1,000"
    assert spec.delta_direction == "positive"
