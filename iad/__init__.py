"""IAD — Intelligent Analytics & Data Science platform.

This is the production Python package for the application. All new code lives
here; the legacy ``src/`` and ``pages/`` modules continue to work unchanged
during the migration.

Layered architecture
====================
- ``iad.config``    — pydantic-settings configuration (Phase 1).
- ``iad.core``      — cross-cutting infrastructure: logging, exceptions,
                      validation, error handling, paths (Phase 1).
- ``iad.state``     — typed Streamlit session state accessors (Phase 1).
- ``iad.ml``        — preprocessing, training, evaluation, explainability,
                      tuning, tracking (Phase 2-3).
- ``iad.frontend``  — Streamlit components, layouts, styles (Phase 4).
- ``iad.backend``   — FastAPI API, schemas, repositories, security (Phase 5+).
- ``iad.services``  — application use-case orchestration (Phase 5+).
"""
from __future__ import annotations

__all__ = ["__version__"]
__version__ = "0.2.0"
