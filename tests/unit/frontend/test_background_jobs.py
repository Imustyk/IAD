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

    def train() -> str:
        time.sleep(0.05)
        return "ok"

    record = submit_training_job(train, name="unit-test")
    assert record.job_id

    for _ in range(50):
        polled = poll_active_job()
        if polled and polled.status.name in ("COMPLETED", "FAILED"):
            break
        time.sleep(0.05)

    clear_job_state()
    assert poll_active_job() is None


@pytest.mark.unit
def test_render_job_status_no_job(monkeypatch) -> None:
    from iad.frontend.performance import background

    monkeypatch.setattr(background, "st", __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock())
    assert background.render_job_status() is False
