"""Streamlit routes and grouped sidebar navigation."""
from __future__ import annotations

from dataclasses import dataclass

HOME = "app.py"
DATA_LOADING = "pages/1_Data_Loading.py"
DESCRIPTIVE = "pages/2_Descriptive_Analysis.py"
DIAGNOSTIC = "pages/3_Diagnostic_Analysis.py"
PREDICTIVE = "pages/4_Predictive_Modeling.py"
APPLY_MODEL = "pages/5_Apply_Model.py"
PRESCRIPTIVE = "pages/6_Prescriptive_Analysis.py"
ADVANCED = "pages/7_Advanced_Analytics.py"
EXPORT = "pages/8_Export_Reports.py"


@dataclass(frozen=True)
class NavItem:
    path: str
    label: str
    icon: str


@dataclass(frozen=True)
class NavGroup:
    title: str
    items: tuple[NavItem, ...]


NAV_GROUPS: tuple[NavGroup, ...] = (
    NavGroup("Overview", (NavItem(HOME, "Dashboard", ":material/dashboard:"),)),
    NavGroup("Data", (NavItem(DATA_LOADING, "Data loading", ":material/database:"),)),
    NavGroup(
        "Analytics",
        (
            NavItem(DESCRIPTIVE, "Descriptive analytics", ":material/bar_chart:"),
            NavItem(DIAGNOSTIC, "Diagnostic analytics", ":material/query_stats:"),
            NavItem(PREDICTIVE, "Predictive analytics", ":material/model_training:"),
            NavItem(APPLY_MODEL, "Apply model", ":material/play_circle:"),
            NavItem(PRESCRIPTIVE, "Prescriptive analytics", ":material/lightbulb:"),
            NavItem(ADVANCED, "Advanced analytics", ":material/science:"),
        ),
    ),
    NavGroup("Reports", (NavItem(EXPORT, "Export and reports", ":material/description:"),)),
)

NAV_PAGES: list[tuple[str, str]] = [
    (item.path, item.label) for group in NAV_GROUPS for item in group.items
]
