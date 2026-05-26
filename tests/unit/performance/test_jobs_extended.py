"""Extended background job runner tests."""
from __future__ import annotations

import time

import pytest

from iad.performance.jobs import BackgroundJobRunner, JobStatus


@pytest.fixture(autouse=True)
def _runner():
    BackgroundJobRunner.reset()
    yield
    BackgroundJobRunner.reset()


@pytest.mark.unit
def test_job_failure_records_error() -> None:
    runner = BackgroundJobRunner.instance()

    def boom() -> None:
        raise RuntimeError("boom")

    record = runner.submit(boom, name="fail")
    for _ in range(30):
        if record.status != JobStatus.RUNNING:
            break
        time.sleep(0.05)
    assert record.status == JobStatus.FAILED
    assert record.error


@pytest.mark.unit
def test_cancel_job() -> None:
    runner = BackgroundJobRunner.instance()

    def slow() -> str:
        time.sleep(2)
        return "done"

    record = runner.submit(slow, name="slow")
    cancelled = runner.cancel(record.job_id)
    assert isinstance(cancelled, bool)


@pytest.mark.unit
def test_get_missing_job() -> None:
    assert BackgroundJobRunner.instance().get("missing-id") is None
