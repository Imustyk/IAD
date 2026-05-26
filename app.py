"""Data Science SaaS — Streamlit application entry point.

Run:
    streamlit run app.py
"""
from __future__ import annotations

from iad.config import get_settings
from iad.core import get_logger
from iad.core.logging import configure_logging
from iad.core.observability import init_observability
from iad.frontend.layouts.dashboard import render_home_dashboard
from iad.frontend.layouts.page import setup_page

configure_logging()
init_observability(service="iad-streamlit")
settings = get_settings()
logger = get_logger("iad.frontend.home")

# setup_page: config → session → auth → CSS → sidebar → title shell
setup_page("Home", icon="📊", caption="End-to-end analytics workspace — from raw data to deployed models.")

logger.info(
    "rendering home page",
    extra={"ctx_environment": settings.ENVIRONMENT, "ctx_app_version": settings.APP_VERSION},
)

render_home_dashboard()
