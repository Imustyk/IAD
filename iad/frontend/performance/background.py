"""Streamlit integration for background training jobs."""
from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from typing import Any, Literal

import streamlit as st

from iad.config.settings import get_settings
from iad.performance.jobs import BackgroundJobRunner, JobRecord, JobStatus
from iad.frontend.services.training import persist_training_to_session
from iad.state.session import state_get, state_set

KEY_ACTIVE_JOB = "iad_active_training_job"
KEY_JOB_RESULT = "iad_training_job_result"
_POLL_INTERVAL = timedelta(seconds=2)

TrainingJobPhase = Literal["idle", "running", "completed", "failed"]


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


def _apply_training_result(record: JobRecord[Any]) -> None:
    """Persist a finished training job into the same session keys as sync training."""
    if record.result is None:
        return
    pipeline, report = record.result
    persist_training_to_session(pipeline, report)


def poll_active_job() -> JobRecord[Any] | None:
    """Return the current job record and sync finished results into session state."""
    job_id = state_get(KEY_ACTIVE_JOB)
    if not job_id:
        return None
    record = BackgroundJobRunner.instance().get(job_id)
    if record is None:
        state_set(KEY_ACTIVE_JOB, None)
        return None
    if record.status == JobStatus.COMPLETED:
        _apply_training_result(record)
        state_set(KEY_JOB_RESULT, record.to_dict())
        state_set(KEY_ACTIVE_JOB, None)
    elif record.status in (JobStatus.FAILED, JobStatus.CANCELLED):
        state_set(KEY_JOB_RESULT, record.to_dict())
        state_set(KEY_ACTIVE_JOB, None)
    return record


@st.fragment(run_every=_POLL_INTERVAL)
def _poll_training_fragment() -> None:
    """Re-run while a background job is active so completion updates without a click."""
    job_id = state_get(KEY_ACTIVE_JOB)
    if not job_id:
        return
    record = poll_active_job()
    if record is None or record.status != JobStatus.RUNNING:
        st.rerun()


def sync_background_training() -> TrainingJobPhase:
    """Poll the active job, render status, and auto-refresh until it finishes.

    Returns:
        * ``idle`` — no background job tracked in session.
        * ``running`` — job in progress (caller should ``st.stop()``).
        * ``completed`` — results written to session; caller should ``st.rerun()``.
        * ``failed`` — job failed; error shown (caller should ``st.stop()``).
    """
    settings = get_settings()
    if not settings.PERF_BACKGROUND_TRAINING:
        return "idle"
    if not state_get(KEY_ACTIVE_JOB):
        return "idle"

    record = poll_active_job()
    if record is None:
        return "idle"

    if record.status == JobStatus.RUNNING:
        st.progress(record.progress or 0.1, text=record.message or "Training…")
        st.caption(f"Background job `{record.job_id[:8]}…` — UI stays responsive.")
        _poll_training_fragment()
        return "running"

    if record.status == JobStatus.COMPLETED:
        return "completed"

    if record.status == JobStatus.FAILED:
        st.error(record.error or "Training failed.")
        return "failed"

    return "idle"


def render_job_status(container: Any | None = None) -> bool:
    """Render progress for an active job. Returns True when a job is in flight."""
    phase = sync_background_training()
    if phase == "running":
        return True
    if phase == "completed":
        target = container or st
        target.success("Background training finished.")
    return False


def clear_job_state() -> None:
    state_set(KEY_ACTIVE_JOB, None)
    state_set(KEY_JOB_RESULT, None)
