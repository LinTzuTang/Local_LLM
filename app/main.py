"""Local Agent Lab: uv run streamlit run app/main.py"""

from pathlib import Path

import i18n
import jobs
import streamlit as st
import ui
from i18n import tr

st.set_page_config(page_title="Local Agent Lab", page_icon=":material/science:", layout="wide")
st.logo(str(Path(__file__).parent / "static" / "logo.svg"), size="large")
ui.inject_css()

i18n.current()


def page(path: str, zh: str, en: str, icon: str, **kw) -> st.Page:
    return st.Page(path, title=tr(zh, en), icon=icon, **kw)


pages = {
    "": [page("views/home.py", "首頁", "Home", ":material/home:", default=True)],
    tr("模型", "Models"): [
        page("views/models.py", "模型庫", "Model library", ":material/apps:"),
        page("views/chat.py", "對話測試", "Chat", ":material/chat:"),
        page("views/compare.py", "模型比較", "Compare models", ":material/balance:"),
        page("views/speech.py", "語音模型比較", "Compare speech models", ":material/graphic_eq:"),
        page("views/media.py", "語音與文件", "Speech & documents", ":material/mic:"),
    ],
    "Agents": [
        page("views/agent.py", "Agent 工作室", "Agent studio", ":material/smart_toy:"),
        page("views/agents_overview.py", "Agents 一覽", "All agents", ":material/groups:"),
        page("views/agent_compare.py", "多 Agent 比較", "Compare agents", ":material/compare:"),
        page("views/tools.py", "工具與技能", "Tools & skills", ":material/build:"),
        page("views/company.py", "公司資料庫", "Company database", ":material/database:"),
    ],
    tr("多 Agent 場景", "Scenarios"): [
        page("views/scenarios.py", "場景展示", "Scenario gallery", ":material/hub:"),
        page("views/knowledge.py", "知識庫 RAG", "Knowledge base", ":material/library_books:"),
    ],
    tr("總控", "Control"): [
        page("views/control.py", "總控 Coding Agent", "Coding agent", ":material/terminal:"),
    ],
    tr("更多", "More"): [
        page("views/roadmap.py", "導入路線圖", "Adoption roadmap", ":material/map:"),
        page("views/history.py", "執行紀錄", "Run history", ":material/history:"),
    ],
}
current = st.navigation(pages, position="top")

# Top-right corner of every page: background tasks and the language switch.
_, tasks_col, lang_col = st.columns([8, 1.7, 1.3], vertical_alignment="center")
with tasks_col:
    jobs.header_popover()
with lang_col:
    i18n.switch()
current.run()
