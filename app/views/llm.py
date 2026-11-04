"""Language models: compare several side by side, or chat with one. The two modes are the former
Compare models and Chat pages (llm_compare_page.py, llm_chat_page.py), run here as scripts so
they keep their own state, background jobs and live fragments."""

import runpy
from pathlib import Path

import streamlit as st
import ui
from common import kept, require_ollama
from i18n import tr

ui.page_header(
    "balance",
    tr("語言模型比較", "Language models"),
    tr(
        "同一個問題並排比較多個地端語言模型的答案與速度，或直接和一個模型對話。語音模型請到「語音模型比較」。",
        "Compare several local language models side by side on one prompt, or chat with one. "
        "Speech models have their own page, Compare speech models.",
    ),
    eyebrow=tr("模型", "Models"),
)
require_ollama()

MODES = {
    "compare": (tr("並排比較", "Side by side"), "balance", "llm_compare_page.py"),
    "chat": (tr("對話測試", "Chat"), "chat", "llm_chat_page.py"),
}
st.session_state.setdefault("llm_mode", "compare")
mode = (
    st.segmented_control(
        tr("模式", "Mode"), list(MODES), label_visibility="collapsed",
        format_func=lambda m: f":material/{MODES[m][1]}: {MODES[m][0]}", **kept("llm_mode"),
    )
    or "compare"
)  # fmt: skip
runpy.run_path(str(Path(__file__).resolve().parents[1] / MODES[mode][2]), run_name="__main__")
