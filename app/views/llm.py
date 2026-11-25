"""Language models: compare several side by side, chat with one, test tool calling, run the
multi-step or the coding tests. The modes are separate scripts (llm_compare_page.py,
llm_chat_page.py, llm_tools_page.py, llm_steps_page.py, llm_code_page.py), run here so they keep
their own state, background jobs and live fragments. The measured scores are on the Scores page
(views/scores.py)."""

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
        "同一個問題並排比較多個地端語言模型的答案與速度、直接和一個模型對話，"
        "現場測工具呼叫、多步驟任務與寫程式。語音模型請到「語音模型比較」；評測結果在「評測分數」。",
        "Compare several local language models side by side on one prompt, chat with one, test "
        "tool calling, multi-step tasks and coding live. Speech models have their own page, "
        "Compare speech models; measured results are on Scores.",
    ),
    eyebrow=tr("模型", "Models"),
)
st.page_link("views/scores.py", label=tr("評測結果與選模型建議 → 評測分數",
             "Results and which model to pick → Scores"), icon=":material/leaderboard:",
             query_params={"kind": "llm"})  # fmt: skip
require_ollama()

MODES = {
    "compare": (tr("並排比較", "Side by side"), "balance", "llm_compare_page.py"),
    "chat": (tr("對話測試", "Chat"), "chat", "llm_chat_page.py"),
    "tools": (tr("工具呼叫測試", "Tool calling"), "build", "llm_tools_page.py"),
    "steps": (tr("多步驟測試", "Multi-step"), "route", "llm_steps_page.py"),
    "code": (tr("寫程式測試", "Coding"), "code", "llm_code_page.py"),
}
if st.session_state.get("llm_mode") not in MODES:  # "scores" moved to its own page
    st.session_state.llm_mode = "compare"
mode = (
    st.segmented_control(
        tr("模式", "Mode"), list(MODES), label_visibility="collapsed",
        format_func=lambda m: f":material/{MODES[m][1]}: {MODES[m][0]}", **kept("llm_mode"),
    )
    or "compare"
)  # fmt: skip
runpy.run_path(str(Path(__file__).resolve().parents[1] / MODES[mode][2]), run_name="__main__")
