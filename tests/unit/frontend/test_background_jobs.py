"""Background training Streamlit integration tests."""
from __future__ import annotations

import time

import pytest

from iad.performance.jobs import BackgroundJobRunner
from iad.state.session import init_session_state


@pytest.fixture(autouse=True)
def _jobs(monkeypatch):
    monkeypatch.setenv("IAD_PERF_BACKGROUND_TRAINING", "true")
    from iad.config.settings import get_settings

    get_settings.cache_clear()
    init_session_state()
    BackgroundJobRunner.reset()
    yield
    BackgroundJobRunner.reset()
    get_settings.cache_clear()


@pytest.mark.unit
def test_submit_and_poll() -> None:
    from iad.frontend.performance.background import (
        clear_job_state,
        poll_active_job,
        submit_training_job,
    )
    from iad.state.session import KEY_MODEL_BUNDLE, KEY_TRAINING_REPORT, state_get

    def train() -> tuple[str, object]:
        from iad.frontend.services.training import UnifiedTrainingReport

        time.sleep(0.05)
        report = UnifiedTrainingReport(
            task_type="classification",
            target="y",
            features=["x"],
            leaderboard=__import__("pandas").DataFrame(),
            best_model_name="Test",
            metrics={},
            cv_metrics={},
        )
        return ("pipeline", report)

    record = submit_training_job(train, name="unit-test")
    assert record.job_id

    for _ in range(50):
        polled = poll_active_job()
        if polled and polled.status.name in ("COMPLETED", "FAILED"):
            break
        time.sleep(0.05)

    assert state_get(KEY_MODEL_BUNDLE) == "pipeline"
    assert state_get(KEY_TRAINING_REPORT).best_model_name == "Test"
    from iad.state.session import KEY_MODEL_BUNDLE_BYTES

    assert state_get(KEY_MODEL_BUNDLE_BYTES)
    clear_job_state()
    assert poll_active_job() is None


@pytest.mark.unit
def test_render_job_status_no_job(monkeypatch) -> None:
    from iad.frontend.performance import background

    monkeypatch.setattr(background, "st", __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock())
    assert background.render_job_status() is False


@pytest.mark.unit
def test_sync_background_training_completed(monkeypatch) -> None:
    from iad.frontend.performance import background
    from iad.frontend.performance.background import KEY_ACTIVE_JOB
    from iad.performance.jobs import JobRecord, JobStatus
    from iad.state.session import KEY_MODEL_BUNDLE, KEY_TRAINING_REPORT, state_get, state_set

    mock_st = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
    monkeypatch.setattr(background, "st", mock_st)
    monkeypatch.setattr(background, "_poll_training_fragment", lambda: None)

    state_set(KEY_ACTIVE_JOB, "job-1")
    from iad.frontend.services.training import UnifiedTrainingReport

    report = UnifiedTrainingReport(
        task_type="classification",
        target="y",
        features=["x"],
        leaderboard=__import__("pandas").DataFrame(),
        best_model_name="Test",
        metrics={},
        cv_metrics={},
    )
    completed = JobRecord(
        job_id="job-1", name="test", status=JobStatus.COMPLETED, result=("p", report)
    )

    class _Runner:
        def get(self, job_id: str):
            return completed if job_id == "job-1" else None

    monkeypatch.setattr(background.BackgroundJobRunner, "instance", classmethod(lambda cls: _Runner()))

    assert background.sync_background_training() == "completed"
    assert state_get(KEY_MODEL_BUNDLE) == "p"
    assert state_get(KEY_TRAINING_REPORT).best_model_name == "Test"
    assert state_get(KEY_ACTIVE_JOB) is None
