"""MLflow tracker — both in noop mode and (when available) end-to-end."""
from __future__ import annotations

import pytest

from iad.ml.tracking import MLflowTracker, mlflow_available


def test_tracker_noop_when_disabled(monkeypatch) -> None:
    """Even without MLflow installed, the tracker must not raise."""
    # Force the tracker into disabled mode.
    tracker = MLflowTracker(experiment="ut")
    monkeypatch.setattr(tracker, "_enabled", False)
    with tracker as t:
        t.log_params({"alpha": 0.1})
        t.log_metrics({"accuracy": 0.9})
        t.log_artifact("/no/such/file")
    assert tracker.run_id is None


@pytest.mark.skipif(not mlflow_available(), reason="mlflow not installed")
def test_tracker_logs_run_to_local_directory(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file:{tmp_path / 'mlruns'}")
    with MLflowTracker(experiment="iad_test", tracking_uri=str(tmp_path / "mlruns")) as tracker:
        assert tracker.enabled
        tracker.log_params({"alpha": 0.1, "beta": "x"})
        tracker.log_metrics({"f1_macro": 0.9, "accuracy": 0.92})
    assert tracker.run_id is not None
    # mlruns directory created in the tmp path
    assert any(p.name == "mlruns" for p in tmp_path.iterdir())
