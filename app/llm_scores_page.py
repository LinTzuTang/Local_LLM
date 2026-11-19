"""Scores mode of the Language models page (views/llm.py): our own test results and Artificial
Analysis's scores for the same models, each with how the test is done (what it measures, the
method, an example question, its limits), so a number is never shown without its method.

Our results and every method text come from data/lab/llm_benchmarks.json (written by
scripts/llm_eval/export_scores.py from the status document). AA's numbers are read from the copy
this server fetched (src/local_llm/aa_scores.py; an administrator can refresh it here).
Run as a script by views/llm.py (runpy), like a page of its own.
"""

import json
from pathlib import Path

import access
import pandas as pd
import streamlit as st
import ui
from common import kept
from i18n import current, pick, tr

from local_llm import aa_scores

CATALOG = Path(__file__).resolve().parents[1] / "data" / "lab" / "llm_benchmarks.json"


@st.cache_data
def catalog(mtime: float) -> dict:
    return json.loads(CATALOG.read_text(encoding="utf-8"))


def method_card(item: dict, limits: dict | None = None) -> None:
    """What it measures / how / an example / limits, the same four parts for every test."""
    with st.container(border=True):
        st.markdown(f"**{tr('測什麼', 'What it measures')}**　{pick(item['measures'])}")
        st.markdown(f"**{tr('怎麼測', 'Method')}**　{pick(item['method'])}")
        if pick(item.get("example")):
            st.markdown(f"**{tr('例題', 'Example')}**")
            st.markdown(pick(item["example"]))
        if pick(limits or item.get("limits")):
            st.markdown(f"**{tr('限制', 'Limits')}**　{pick(limits or item['limits'])}")
        if pick(item.get("platform")):
            st.caption(":material/play_circle: " + pick(item["platform"]))


if not CATALOG.exists():
    st.warning(
        tr("找不到 data/lab/llm_benchmarks.json。", "data/lab/llm_benchmarks.json is missing.")
    )
    st.stop()
cat = catalog(CATALOG.stat().st_mtime)
local_name, cloud_name = ui.where_names()
where_col = tr("跑在哪", "Where")

SOURCES = {
    "ours": tr("我們的實測", "Our tests"),
    "aa": tr("外部評測：Artificial Analysis", "External: Artificial Analysis"),
}
source = (
    st.segmented_control(
        tr("來源", "Source"), list(SOURCES), format_func=SOURCES.get,
        label_visibility="collapsed", **kept("scores_source"),
    )
    or "ours"
)  # fmt: skip

# ------------------------------------------------------------------ our tests
if source == "ours":
    st.caption(
        tr(
            f"我們自己跑的測試（更新到 {cat['updated']}）。"
            "地端模型是 4-bit 量化，在 Mac 或 GPU 主機上跑；"
            "雲端模型是對照組。每個測試都先說明測法，再看數字。",
            f"Tests we ran ourselves (as of {cat['updated']}). Local models are 4-bit builds on "
            "the Mac or the GPU machine; cloud models are the reference. Each test states its "
            "method before its numbers.",
        )
    )
    tests = {t["id"]: t for t in cat["ours"]}
    chosen = st.selectbox(
        tr("測試", "Test"), list(tests), format_func=lambda i: pick(tests[i]["name"]),
        **kept("scores_test"),
    )  # fmt: skip
    t = tests[chosen]
    method_card(t)
    lang = current() if current() in t["columns"] else "zh"
    df = pd.DataFrame([r[lang] for r in t["rows"]], columns=t["columns"][lang])
    df.insert(
        1, where_col, [cloud_name if r["where"] == "cloud" else local_name for r in t["rows"]]
    )
    st.dataframe(ui.where_rows(df, where_col), hide_index=True, width="stretch",
                 height=min(38 * (len(df) + 1) + 2, 800))  # fmt: skip
    st.caption(
        tr(
            "順序和報告相同（大致由高到低）。完整說明與結論在測試進度文件與 HackMD 報告。",
            "In the report's order (roughly best first). The full discussion is in the status "
            "document and the HackMD reports.",
        )
    )
    st.stop()

# ------------------------------------------------------------------ Artificial Analysis
aa = cat["aa"]
st.markdown(pick(aa["about"]))
with st.container(border=True):
    st.markdown(f"**{tr('怎麼測', 'Method')}**　{pick(aa['method'])}")
    st.markdown(f"**{tr('限制', 'Limits')}**　{pick(aa['limits'])}")
    st.markdown(
        f"[{tr('AA 排行榜', 'AA leaderboard')}]({aa['url']})　·　"
        f"[{tr('AA 的方法說明', 'AA methodology')}]({aa['methodology_url']})"
    )

data = aa_scores.load()
if access.is_admin():
    c1, c2 = st.columns([1, 4], vertical_alignment="center")
    if c1.button(tr("重新抓取", "Refresh"), icon=":material/refresh:", key="aa_refresh"):
        bar = st.progress(0.0, tr("讀取 AA 的模型頁面…", "Reading AA's model pages…"))
        try:
            data = aa_scores.fetch_all(progress=lambda d, n: bar.progress(d / n))
        except Exception as exc:  # noqa: BLE001  (network, or AA changed its page)
            st.error(tr(f"抓取失敗：{exc}", f"Could not fetch: {exc}"))
        bar.empty()
    c2.caption(
        tr(
            "從 AA 網站讀每個模型的頁面（約 20 頁，一秒一頁）。只有管理者看得到這個按鈕。",
            "Reads each model's page on AA's site (about 20 pages, one a second). "
            "Administrators only.",
        )
    )
if not data:
    st.info(
        tr(
            "這台伺服器還沒有抓 AA 的分數。請管理者按上面的「重新抓取」，"
            "或在伺服器上執行 `uv run python -m local_llm.aa_scores`。",
            "This server has not fetched AA's scores yet. An administrator can press Refresh "
            "above, or run `uv run python -m local_llm.aa_scores` on the server.",
        ),
        icon=":material/info:",
    )

metrics = {m["field"]: m for m in aa["metrics"]}
if data:
    st.caption(tr(f"資料日期：{data['fetched']}", f"Fetched: {data['fetched']}"))
    shown = st.multiselect(
        tr("要看的評測", "Evaluations to show"),
        list(metrics),
        default=[f for f, m in metrics.items() if m.get("default")],
        format_func=lambda f: pick(metrics[f]["name"]),
        **kept("scores_aa_metrics"),
    )
    only_local = st.toggle(tr("只看地端模型", "Local models only"), **kept("scores_aa_local"))

    def cell(value, unit: str) -> float | None:
        if value is None:
            return None
        return round(value * 100, 1) if unit == "pct" else round(value, 1 if unit == "index" else 0)

    rows = []
    for name, m in data["models"].items():
        local = aa_scores.is_local(m)
        if only_local and not local:
            continue
        row = {tr("模型", "Model"): name, where_col: local_name if local else cloud_name}
        for f in shown:
            unit = metrics[f]["unit"]
            label = pick(metrics[f]["name"]) + (" %" if unit == "pct" else "")
            row[label] = cell(m.get(f), unit)
        rows.append(row)
    df = pd.DataFrame(rows)
    if shown and not df.empty:
        first = df.columns[2]
        df = df.sort_values(first, ascending=False, na_position="last")
    st.dataframe(ui.where_rows(df, where_col).format(precision=1), hide_index=True,
                 width="stretch", height=min(38 * (len(df) + 1) + 2, 900))  # fmt: skip
    st.caption(
        tr(
            "百分比欄位是 0–100；Elo 與綜合指數是分數。點欄位標題可以排序。「None」是 AA 沒有測。",
            "Percent columns are 0–100; Elo and the index are scores. Click a header to sort. "
            '"None" means AA has not tested it.',
        )
    )

st.markdown(f"#### {tr('各項評測在量什麼', 'What each evaluation measures')}")
st.caption(pick(aa["example_note"]))
for m in metrics.values():
    with st.container(border=True):
        st.markdown(f"**{pick(m['name'])}**　{pick(m['measures'])}")
        if pick(m["example"]):
            st.caption(f"{tr('例', 'Example')}：{pick(m['example'])}")
