"""Performance timing helpers — logs + Prometheus + optional Sentry spans."""
from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps
from typing import ParamSpec, TypeVar

from iad.core.logging import get_logger
from iad.core.observability.prometheus import observe_ml_operation

logger = get_logger("iad.observability.performance")

P = ParamSpec("P")
R = TypeVar("R")


@contextmanager
def timed_block(operation: str, *, log_level: int | None = None) -> Iterator[None]:
    """Time a block; record duration to logs and Prometheus."""
    start = time.perf_counter()
    outcome = "success"
    try:
        yield
    except Exception:
        outcome = "error"
        raise
    finally:
        elapsed = time.perf_counter() - start
        observe_ml_operation(operation=operation, outcome=outcome, duration_seconds=elapsed)
        logger.log(
            log_level or 20,
            "%s completed",
            operation,
            extra={"ctx_operation": operation, "ctx_duration_s": round(elapsed, 4), "ctx_outcome": outcome},
        )


def observe_duration(operation: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator wrapping callables with :func:`timed_block`."""

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        @wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            with timed_block(operation):
                return fn(*args, **kwargs)

        return wrapper

    return decorator
