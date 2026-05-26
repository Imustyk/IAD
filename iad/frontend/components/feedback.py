"""Loading and interaction feedback helpers."""
from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TypeVar

import streamlit as st

T = TypeVar("T")


@contextmanager
def loading(message: str = "Loading…") -> Iterator[None]:
    """Show a spinner while a block runs."""
    with st.spinner(message):
        yield


def run_with_spinner(fn: Callable[[], T], message: str = "Please wait…") -> T:
    with st.spinner(message):
        return fn()
