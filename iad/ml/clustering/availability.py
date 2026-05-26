"""Optional clustering dependency probes."""
from __future__ import annotations

_UMAP: bool | None = None


def umap_available() -> bool:
    global _UMAP
    if _UMAP is None:
        try:
            import umap  # noqa: F401

            _UMAP = True
        except ImportError:
            _UMAP = False
    return _UMAP
