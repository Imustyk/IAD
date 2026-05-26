"""Dashboard UI blocks — rendered as styled HTML in the main document."""
from __future__ import annotations

from iad.config.settings import get_settings
from iad.frontend.components.html_render import esc, render_html_panel


def _panel(inner: str) -> str:
    return f'<div class="iad-embed-root"><div class="iad-wrap">{inner}</div></div>'


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
    inner = (
        f'<div class="iad-hero">'
        f'<div class="iad-hero-inner">'
        f'<span class="iad-hero-badge">{esc(badge)}</span>'
        f"<h2>{esc(title)}</h2>"
        f"<p>{esc(subtitle)}</p>"
        f"</div></div>"
    )
    render_html_panel(_panel(inner))


def render_section_header(title: str, description: str | None = None) -> None:
    desc = f'<p class="iad-section-desc">{esc(description)}</p>' if description else ""
    inner = f'<div class="iad-section iad-surface"><h3>{esc(title)}</h3>{desc}</div>'
    render_html_panel(_panel(inner))


def render_page_header(title: str, caption: str | None = None) -> None:
    cap = f'<p class="iad-page-caption">{esc(caption)}</p>' if caption else ""
    inner = f'<div class="iad-page-hdr iad-surface"><h1>{esc(title)}</h1>{cap}</div>'
    render_html_panel(_panel(inner))


def render_empty_state(
    title: str,
    message: str,
    *,
    hint: str | None = None,
) -> None:
    hint_html = f'<p class="iad-empty-hint">{esc(hint)}</p>' if hint else ""
    inner = (
        f'<div class="iad-empty iad-surface">'
        f"<strong>{esc(title)}</strong>"
        f"<p>{esc(message)}</p>{hint_html}"
        f"</div>"
    )
    render_html_panel(_panel(inner))
