"""Background job runner tests."""
from __future__ import annotations

import time

from iad.performance.jobs import BackgroundJobRunner, JobStatus


def test_background_job_completes() -> None:
    BackgroundJobRunner.reset()
    runner = BackgroundJobRunner(max_workers=1)

    def work(x: int) -> int:
        time.sleep(0.05)
        return x * 2

    record = runner.submit(work, name="test", x=21)
    for _ in range(50):
        updated = runner.get(record.job_id)
        assert updated is not None
        if updated.status == JobStatus.COMPLETED:
            assert updated.result == 42
            break
        time.sleep(0.02)
    else:
        raise AssertionError("job did not complete in time")
    BackgroundJobRunner.reset()
