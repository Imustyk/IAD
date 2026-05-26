"""Streamlit mock for unit-testing UI modules without a runtime."""
from __future__ import annotations

import sys
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock


class _SessionState(dict):
    def __getattr__(self, key: str) -> Any:
        return self.get(key)

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def install_streamlit_mock() -> MagicMock:
    """Install a minimal ``streamlit`` mock into ``sys.modules``."""
    st = MagicMock()
    st.session_state = _SessionState()
    st.cache_data = lambda **kwargs: (lambda fn: fn)
    st.cache_resource = lambda **kwargs: (lambda fn: fn)
    st.columns = lambda spec: [MagicMock() for _ in range(spec if isinstance(spec, int) else len(spec))]
    st.tabs = lambda labels: [MagicMock() for _ in labels]
    st.expander = lambda *a, **k: MagicMock(__enter__=lambda s: s, __exit__=lambda *x: None)
    st.spinner = lambda *a, **k: MagicMock(__enter__=lambda s: s, __exit__=lambda *x: None)
    st.form = lambda *a, **k: MagicMock(__enter__=lambda s: s, __exit__=lambda *x: None)
    st.progress = MagicMock(return_value=MagicMock())
    st.empty = MagicMock(return_value=MagicMock())
    st.toggle = MagicMock(return_value=False)
    st.button = MagicMock(return_value=False)
    st.markdown = MagicMock()
    st.caption = MagicMock()
    st.title = MagicMock()
    st.subheader = MagicMock()
    st.info = MagicMock()
    st.success = MagicMock()
    st.warning = MagicMock()
    st.error = MagicMock()
    st.stop = MagicMock()
    st.rerun = MagicMock()
    st.set_page_config = MagicMock()
    st.plotly_chart = MagicMock()
    st.dataframe = MagicMock()
    st.metric = MagicMock()
    st.divider = MagicMock()
    st.sidebar = MagicMock()
    st.file_uploader = MagicMock(return_value=None)
    st.selectbox = MagicMock(return_value=None)
    st.text_input = MagicMock(return_value="")
    st.download_button = MagicMock()

    module = ModuleType("streamlit")
    for name in dir(st):
        if not name.startswith("_"):
            setattr(module, name, getattr(st, name))
    sys.modules["streamlit"] = module
    return st


def remove_streamlit_mock() -> None:
    sys.modules.pop("streamlit", None)
