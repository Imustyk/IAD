"""Optional forecasting dependency probes."""
from __future__ import annotations

_PROPHET: bool | None = None


def prophet_available() -> bool:
    global _PROPHET
    if _PROPHET is None:
        try:
            from prophet import Prophet  # noqa: F401

            _PROPHET = True
        except ImportError:
            _PROPHET = False
    return _PROPHET
