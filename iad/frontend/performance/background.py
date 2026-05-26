"""Streamlit integration for background training jobs."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import streamlit as st

from iad.config.settings import get_settings
from iad.performance.jobs import BackgroundJobRunner, JobRecord, JobStatus
from iad.state.session import state_get, state_set

KEY_ACTIVE_JOB = "iad_active_training_job"
KEY_JOB_RESULT = "iad_training_job_result"


def submit_training_job(
    fn: Callable[..., Any],
    *,
    name: str = "model-training",
    **kwargs: Any,
) -> JobRecord[Any]:
    """Submit training to the background pool and store job id in session."""
    runner = BackgroundJobRunner.instance()
    record = runner.submit(fn, name=name, **kwargs)
    state_set(KEY_ACTIVE_JOB, record.job_id)
    state_set(KEY_JOB_RESULT, None)
    return record


def poll_active_job() -> JobRecord[Any] | None:
    """Return the current job record and sync completed results to session."""
    job_id = state_get(KEY_ACTIVE_JOB)
    if not job_id:
        return None
    record = BackgroundJobRunner.instance().get(job_id)
    if record is None:
        return None
    if record.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
        state_set(KEY_JOB_RESULT, record.to_dict())
        if record.status != JobStatus.RUNNING:
            state_set(KEY_ACTIVE_JOB, None)
    return record


def render_job_status(container: Any | None = None) -> bool:
    """Render progress for an active job. Returns True when a job is in flight."""
    settings = get_settings()
    if not settings.PERF_BACKGROUND_TRAINING:
        return False

    record = poll_active_job()
    if record is None:
        return False

    target = container or st
    if record.status == JobStatus.RUNNING:
        target.progress(record.progress or 0.1, text=record.message or "Training…")
        target.caption(f"Background job `{record.job_id[:8]}…` — UI stays responsive.")
        return True
    if record.status == JobStatus.COMPLETED:
        target.success("Background training finished.")
        return False
    if record.status == JobStatus.FAILED:
        target.error(record.error or "Training failed.")
        return False
    return False


def clear_job_state() -> None:
    state_set(KEY_ACTIVE_JOB, None)
    state_set(KEY_JOB_RESULT, None)
