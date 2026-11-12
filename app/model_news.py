"""New models the catalog does not list yet: a weekly background check and the home page notice.

The check itself is local_llm.model_watch; this file only schedules it while the platform runs
and shows what it found.
"""

import threading
import time

import pandas as pd
import streamlit as st
import ui
from i18n import tr

from local_llm import model_watch

WAKE_EVERY = 6 * 3600  # seconds; the check runs only when the last one is a week old


@st.cache_resource
def start_background() -> threading.Thread:
    """One thread per server process, whichever page is opened first."""

    def loop() -> None:
        time.sleep(60)  # let the platform finish starting first
        while True:
            model_watch.check_if_due()
            time.sleep(WAKE_EVERY)

    thread = threading.Thread(target=loop, name="model-watch", daemon=True)
    thread.start()
    return thread


def _about(m: dict) -> str:
    if m["source"] == "ollama":
        return m.get("summary") or ""
    likes = tr(f"{m.get('likes', 0)} 個讚", f"{m.get('likes', 0)} likes")
    chinese = tr(" · 標示支援中文", " · lists Chinese") if m.get("chinese") else ""
    return f"{ui.cap_text(m.get('category', ''))} · {likes}{chinese}"


@st.dialog(tr("新模型", "New models"), width="large")
def _list(rows: list[dict]) -> None:
    st.caption(
        tr(
            "還沒列在模型庫目錄裡的模型。這裡只是提示，要不要加進模型庫需要人工整理。",
            "Models the catalog does not list yet. This is only a notice; adding one to the "
            "model library is done by hand.",
        )
    )
    table = pd.DataFrame(
        [
            {
                tr("模型", "Model"): m["name"],
                tr("來源", "Source"): "Ollama" if m["source"] == "ollama" else "Hugging Face",
                tr("日期", "Date"): (m.get("date") or "")[:10],
                tr("說明", "About"): _about(m),
                tr("連結", "Link"): m["url"],
            }
            for m in rows
        ]
    )
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        height=min(38 + 35 * len(rows), 600),
        column_config={
            tr("連結", "Link"): st.column_config.LinkColumn(display_text=tr("開啟", "Open"))
        },
    )


def home_notice() -> None:
    state = model_watch.load()
    rows = model_watch.unread(state)
    if not rows:
        return
    ollama = sum(m["source"] == "ollama" for m in rows)
    hf = len(rows) - ollama
    when = (state.get("last_check") or "")[:10]
    with st.container(border=True, key="model_news"):
        text, see, done = st.columns([6, 1.2, 1.2], vertical_alignment="center")
        text.markdown(
            ":material/new_releases: "
            + tr(
                f"**發現 {len(rows)} 個新模型**（Ollama {ollama} 個、Hugging Face {hf} 個），"
                f"模型庫還沒列。上次檢查：{when}",
                f"**{len(rows)} new models found** (Ollama {ollama}, Hugging Face {hf}) "
                f"that the model library does not list yet. Last check: {when}",
            )
        )
        if see.button(tr("查看", "View"), key="model_news_view", width="stretch"):
            _list(rows)
        if done.button(tr("知道了", "Dismiss"), key="model_news_ack", width="stretch"):
            model_watch.acknowledge()
            st.rerun()
