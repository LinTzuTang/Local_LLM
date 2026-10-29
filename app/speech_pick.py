"""Pick a model: which speech model to use in which situation, local and cloud side by side.

Nothing is run here: data/lab/recommendations.json is written from the measured results
(the other modes' benchmarks and docs/SPEECH_EVALUATION.md). Models whose results are close
are recommended together. Local cells are blue, cloud cells orange (the ui palette).
"""

import html
import json

import streamlit as st
import ui
from common import PROJECT_DIR
from i18n import tr

DATA = PROJECT_DIR / "data" / "lab" / "recommendations.json"


def load() -> dict | None:
    try:
        return json.loads(DATA.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def table(rows: list[dict]) -> str:
    local, cloud = ui.where_names()
    head = (
        f"<th>{tr('任務', 'Task')}</th><th>{tr('語言', 'Language')}</th>"
        f'<th style="background:{ui.LOCAL_COLOR};color:#fff">{local}</th>'
        f'<th style="background:{ui.CLOUD_COLOR};color:#fff">{cloud}</th>'
        f"<th>{tr('怎麼比', 'Measured on')}</th>"
    )
    body = []
    for r in rows:
        task, lang = html.escape(tr(*r["task"])), html.escape(tr(*r["lang"]))
        body.append(
            f"<tr><td><b>{task}</b></td><td>{lang}</td>"
            + f'<td style="background:{ui.LOCAL_TINT}">{html.escape(tr(*r["local"]))}</td>'
            + f'<td style="background:{ui.CLOUD_TINT}">{html.escape(tr(*r["cloud"]))}</td>'
            + f'<td style="color:#64748B">{html.escape(tr(*r["note"]))}</td></tr>'
        )  # fmt: skip
    return (
        '<table style="width:100%;border-collapse:collapse;font-size:0.92rem">'
        f'<thead><tr style="text-align:left">{head}</tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table>"
        "<style>table td,table th{border:1px solid #E2E8F0;padding:6px 8px;vertical-align:top}"
        "</style>"
    )


def pick_page(url: str | None = None, status: dict | None = None) -> None:
    data = load()
    if not data:
        st.info(tr("找不到建議表（data/lab/recommendations.json）。",
                   "No recommendations found (data/lab/recommendations.json)."))  # fmt: skip
        return
    st.markdown(
        tr(
            f"**依情境選模型**（{data['date']} 的實測結果）。藍色＝本機（資料不出這台電腦），"
            "橘色＝雲端（錄音或文字要上傳、要付費）；結果差不多的模型一起列出。括號裡是錯字率、"
            "開口時間或翻譯分數，細節在各模式的「評測總表」。",
            f"**Which model for which situation** (results of {data['date']}). Blue = on this "
            "machine (data stays here), orange = cloud (audio or text is uploaded, paid); models "
            "with similar results are listed together. In brackets: error rate, time to talk or "
            "translation score; the details are in each mode's Benchmark.",
        )
    )
    st.markdown(table(data["rows"]), unsafe_allow_html=True)
