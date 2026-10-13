"""Chinese / English UI text.

Copy stays next to the code that shows it: `tr("中文", "English")` returns the string for the
language picked with the switch in the top-right corner. The choice is kept in session state
and in the `?lang=` query parameter, so a link to any page opens in the same language.
"""

import streamlit as st

LANGS = {"zh": "中文", "en": "EN"}
DEFAULT = "zh"


def current() -> str:
    if "lang" not in st.session_state:
        wanted = st.query_params.get("lang", DEFAULT)
        st.session_state.lang = wanted if wanted in LANGS else DEFAULT
    return st.session_state.lang


def is_en() -> bool:
    return current() == "en"


def tr(zh: str, en: str) -> str:
    return en if is_en() else zh


def pick(text: dict | str | None) -> str:
    """A {"zh": ..., "en": ...} value from the JSON configs (plain strings pass through)."""
    if isinstance(text, dict):
        return text.get(current()) or text.get(DEFAULT) or next(iter(text.values()), "")
    return text or ""


def _changed() -> None:
    choice = st.session_state.get("_lang_switch") or current()
    st.session_state.lang = choice
    st.query_params["lang"] = choice


def switch() -> None:
    """Language toggle; main.py puts it in the top-right corner of every page."""
    lang = current()
    st.query_params["lang"] = lang
    st.session_state["_lang_switch"] = lang
    st.segmented_control(
        "Language",
        list(LANGS),
        format_func=LANGS.get,
        key="_lang_switch",
        on_change=_changed,
        label_visibility="collapsed",
    )
