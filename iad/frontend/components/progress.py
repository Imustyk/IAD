"""Progress bars and background-job helpers for long-running UI tasks."""
from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TypeVar

import streamlit as st

T = TypeVar("T")


@contextmanager
def progress_bar(
    label: str = "Working…",
    *,
    total_steps: int | None = None,
) -> Iterator[Callable[[float, str | None], None]]:
    """Context manager yielding an update callback ``(fraction, message)``.

    Example::

        with progress_bar("Training models", total_steps=5) as update:
            for i, model in enumerate(models):
                train(model)
                update((i + 1) / 5, f"Trained {model}")
    """
    bar = st.progress(0.0, text=label)
    status = st.empty()

    def update(fraction: float, message: str | None = None) -> None:
        fraction = max(0.0, min(1.0, fraction))
        bar.progress(fraction, text=message or label)
        if message:
            status.markdown(f'<p class="iad-progress-label">{message}</p>', unsafe_allow_html=True)

    try:
        yield update
        bar.progress(1.0, text="Done")
        status.empty()
    except Exception:
        bar.empty()
        status.empty()
        raise


@contextmanager
def spinner(label: str = "Please wait…"):
    """Thin wrapper around ``st.spinner``."""
    with st.spinner(label):
        yield


def run_with_progress(
    steps: list[str],
    fn: Callable[[Callable[[int, str], None],], T],
) -> T:
    """Run *fn* with a stepped progress bar.

    *fn* receives ``advance(step_index, step_label)``.
    """
    n = len(steps)
    with progress_bar(steps[0] if steps else "Working…", total_steps=n) as update:

        def advance(i: int, msg: str | None = None) -> None:
            update((i + 1) / max(n, 1), msg or (steps[i] if i < n else None))

        return fn(advance)


def simulated_steps(
    steps: list[str],
    delay: float = 0.05,
) -> None:
    """For demos/tests — walk through labeled steps with small delays."""
    with progress_bar(steps[0] if steps else "Processing…", total_steps=len(steps)) as update:
        for i, step in enumerate(steps):
            update((i + 1) / len(steps), step)
            time.sleep(delay)
