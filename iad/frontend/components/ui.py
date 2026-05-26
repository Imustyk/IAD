"""Dashboard UI blocks — rendered via embedded HTML panels."""
from __future__ import annotations

from iad.config.settings import get_settings
from iad.frontend.components.html_render import esc, render_html_panel


def render_hero(
    *,
    title: str | None = None,
    subtitle: str | None = None,
    badge: str | None = None,
) -> None:
    settings = get_settings()
    title = title or settings.APP_NAME
    subtitle = subtitle or "Professional analytics workspace — load data, train models, export reports."
    badge = badge or f"v{settings.APP_VERSION} · {settings.ENVIRONMENT}"
    body = (
        f'<div class="iad-wrap"><div class="iad-hero">'
        f'<span class="iad-hero-badge">{esc(badge)}</span>'
        f"<h2>{esc(title)}</h2>"
        f"<p>{esc(subtitle)}</p>"
        f"</div></div>"
    )
    render_html_panel(body, height=130)


def render_section_header(title: str, description: str | None = None) -> None:
    desc = f"<p>{esc(description)}</p>" if description else ""
    body = (
        f'<div class="iad-wrap iad-section">'
        f"<h3>{esc(title)}</h3>{desc}"
        f"</div>"
    )
    render_html_panel(body, height=72 if description else 48)


def render_page_header(title: str, caption: str | None = None) -> None:
    cap = f"<p>{esc(caption)}</p>" if caption else ""
    body = (
        f'<div class="iad-wrap iad-page-hdr">'
        f"<h1>{esc(title)}</h1>{cap}"
        f"</div>"
    )
    render_html_panel(body, height=100 if caption else 72)


def render_empty_state(
    title: str,
    message: str,
    *,
    hint: str | None = None,
) -> None:
    hint_html = f"<p style='margin-top:0.5rem;font-size:0.75rem'>{esc(hint)}</p>" if hint else ""
    body = (
        f'<div class="iad-wrap"><div class="iad-empty">'
        f"<strong>{esc(title)}</strong>"
        f"<p>{esc(message)}</p>{hint_html}"
        f"</div></div>"
    )
    render_html_panel(body, height=140)
