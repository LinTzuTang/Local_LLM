"""Day / night switch at the top right of every page.

Streamlit has no Python call to change the theme; its own menu stores the choice in the browser
(localStorage "stActiveTheme-<path>-v2" = "Light" | "Dark") and reads it when the page loads.
This button writes the same value and reloads. The choice is also kept under "labTheme", so a
page opened directly at another address follows it. Until the first click the system setting
decides (both themes are in .streamlit/config.toml).
"""

from functools import cache
from pathlib import Path

import streamlit as st
import ui
from i18n import tr

DIR = Path(__file__).parent / "static" / "theme_switch"


@cache
def _component():
    return st.components.v2.component(
        "lal_theme_switch",
        html=(DIR / "theme_switch.html").read_text(encoding="utf-8"),
        css=(DIR / "theme_switch.css").read_text(encoding="utf-8"),
        js=(DIR / "theme_switch.js").read_text(encoding="utf-8"),
    )


def switch() -> None:
    _component()(
        key="lal_theme_switch",
        data={
            "current": ui.theme(),
            "labels": {"Light": tr("白天", "Day"), "Dark": tr("黑夜", "Night")},
            "tips": {
                "Light": tr("切換成白天模式", "Switch to day mode"),
                "Dark": tr("切換成黑夜模式", "Switch to night mode"),
            },
        },  # fmt: skip
    )
