"""Streamlit API compatibility (width vs legacy use_container_width)."""
from __future__ import annotations

from typing import Any

import streamlit as st


def plotly_chart(fig: Any, *, key: str | None = None, stretch: bool = True) -> None:
    width = "stretch" if stretch else "content"
    try:
        st.plotly_chart(fig, width=width, key=key)
    except TypeError:
        st.plotly_chart(fig, use_container_width=stretch, key=key)


def dataframe(df: Any, *, height: int | None = None, hide_index: bool = True, stretch: bool = True, **kwargs: Any) -> None:
    width = "stretch" if stretch else "content"
    call_kwargs: dict[str, Any] = {"hide_index": hide_index, **kwargs}
    if height is not None:
        call_kwargs["height"] = height
    try:
        st.dataframe(df, width=width, **call_kwargs)
    except TypeError:
        st.dataframe(df, use_container_width=stretch, **call_kwargs)


def form_submit_button(label: str, *, type: str = "secondary", stretch: bool = True, **kwargs: Any) -> bool:
    try:
        return st.form_submit_button(label, type=type, width="stretch" if stretch else "content", **kwargs)
    except TypeError:
        return st.form_submit_button(label, type=type, use_container_width=stretch, **kwargs)


def button(label: str, *, type: str = "secondary", stretch: bool = True, **kwargs: Any) -> bool:
    try:
        return st.button(label, type=type, width="stretch" if stretch else "content", **kwargs)
    except TypeError:
        return st.button(label, type=type, use_container_width=stretch, **kwargs)


def download_button(label: str, *, data: Any, file_name: str, mime: str | None = None, stretch: bool = True, **kwargs: Any) -> bool:
    try:
        return st.download_button(
            label,
            data=data,
            file_name=file_name,
            mime=mime,
            width="stretch" if stretch else "content",
            **kwargs,
        )
    except TypeError:
        return st.download_button(
            label,
            data=data,
            file_name=file_name,
            mime=mime,
            use_container_width=stretch,
            **kwargs,
        )


def page_link(container: Any, path: str, *, label: str, icon: str | None = None, stretch: bool = True) -> None:
    width = stretch
    try:
        if icon:
            container.page_link(path, label=label, icon=icon, width=width)
        else:
            container.page_link(path, label=label, width=width)
    except TypeError:
        if icon:
            container.page_link(path, label=label, icon=icon, use_container_width=stretch)
        else:
            container.page_link(path, label=label, use_container_width=stretch)
