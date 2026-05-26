"""Design system — tokens, component CSS, theme injection."""
from iad.frontend.styles.theme import (
    DEFAULT_THEME,
    THEME_KEY,
    get_theme,
    inject_css,
    page_config,
    set_theme,
    theme_toggle_widget,
)

__all__ = [
    "DEFAULT_THEME",
    "THEME_KEY",
    "get_theme",
    "inject_css",
    "page_config",
    "set_theme",
    "theme_toggle_widget",
]
