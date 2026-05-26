"""Background job execution — thread pool for CPU-bound ML work."""
from __future__ import annotations

import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from threading import Lock
from typing import Any, Generic, TypeVar

from iad.config.settings import get_settings
from iad.core.logging import get_logger

logger = get_logger("iad.performance.jobs")

T = TypeVar("T")


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobRecord(Generic[T]):
    """Tracked background job with result or error."""

    job_id: str
    name: str
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: T | None = None
    error: str | None = None
    progress: float = 0.0
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "name": self.name,
            "status": self.status.value,
            "progress": self.progress,
            "message": self.message,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class BackgroundJobRunner:
    """Process-wide thread pool for training and heavy analytics.

    Streamlit reruns the script on interaction; job state is stored in
    ``st.session_state`` via :func:`iad.frontend.performance.background` helpers.
    """

    _instance: BackgroundJobRunner | None = None
    _lock = Lock()

    def __init__(self, max_workers: int | None = None) -> None:
        settings = get_settings()
        workers = max_workers or settings.PERF_MAX_WORKERS
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="iad-job")
        self._jobs: dict[str, JobRecord[Any]] = {}
        self._futures: dict[str, Future[Any]] = {}

    @classmethod
    def instance(cls) -> BackgroundJobRunner:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Shutdown pool — tests only."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance._executor.shutdown(wait=False, cancel_futures=True)
                cls._instance = None

    def submit(
        self,
        fn: Callable[..., T],
        *,
        name: str,
        job_id: str | None = None,
        **kwargs: Any,
    ) -> JobRecord[T]:
        jid = job_id or uuid.uuid4().hex
        record: JobRecord[T] = JobRecord(job_id=jid, name=name)
        self._jobs[jid] = record  # type: ignore[assignment]

        def _wrapper() -> T:
            record.status = JobStatus.RUNNING
            record.started_at = datetime.now(UTC)
            record.message = "Running…"
            try:
                result = fn(**kwargs)
                record.result = result
                record.status = JobStatus.COMPLETED
                record.progress = 1.0
                record.message = "Complete"
                return result
            except Exception as exc:
                record.status = JobStatus.FAILED
                record.error = str(exc)
                record.message = "Failed"
                logger.exception("background job %s failed", jid)
                raise
            finally:
                record.completed_at = datetime.now(UTC)

        future = self._executor.submit(_wrapper)
        self._futures[jid] = future
        logger.info("submitted background job %s (%s)", jid, name)
        return record

    def get(self, job_id: str) -> JobRecord[Any] | None:
        record = self._jobs.get(job_id)
        if record is None:
            return None
        future = self._futures.get(job_id)
        if future is not None and future.done() and record.status == JobStatus.RUNNING:
            try:
                future.result()
            except Exception:
                pass
        return record

    def cancel(self, job_id: str) -> bool:
        future = self._futures.get(job_id)
        if future is None:
            return False
        cancelled = future.cancel()
        if cancelled:
            record = self._jobs.get(job_id)
            if record:
                record.status = JobStatus.CANCELLED
        return cancelled
