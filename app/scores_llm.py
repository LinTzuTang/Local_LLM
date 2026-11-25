"""Language models on the Scores page (views/scores.py): our own test results and Artificial
Analysis's scores for the same models, each with how the test is done, so a number is never shown
without its method.

Our results, the "pick a model" table, every method text, each test's last test date and state
come from data/lab/llm_benchmarks.json (written by scripts/llm_eval/export_scores.py from the
status document). AA's numbers are read from the copy this server fetched
(src/local_llm/aa_scores.py; an administrator can refresh it here).
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
def _load(mtime: float) -> dict:
    return json.loads(CATALOG.read_text(encoding="utf-8"))


def catalog() -> dict | None:
    return _load(CATALOG.stat().st_mtime) if CATALOG.exists() else None


def where_col() -> str:
    return tr("地端／雲端", "Local / cloud")


def height(rows: int) -> int:
    return min(35 * (rows + 1) + 3, 800)


def results_table(columns: dict, rows: list[dict]) -> None:
    """One results table: local rows blue, cloud rows orange (ui.where_rows). The local/cloud
    column only when the table has cloud rows."""
    local_name, cloud_name = ui.where_names()
    lang = current() if current() in columns else "zh"
    df = pd.DataFrame([r[lang] for r in rows], columns=columns[lang])
    if any(r["where"] == "cloud" for r in rows):
        df.insert(
            1, where_col(), [cloud_name if r["where"] == "cloud" else local_name for r in rows]
        )
        st.dataframe(ui.where_rows(df, where_col()), hide_index=True, width="stretch",
                     height=height(len(df)))  # fmt: skip
    else:
        st.dataframe(df, hide_index=True, width="stretch", height=height(len(df)))


def test_results(t: dict) -> None:
    """A test's results: one table, several (capacity), or none yet (running or planned)."""
    if t.get("tables"):
        for part in t["tables"]:
            st.markdown(f"**{pick(part['title'])}**")
            if pick(part.get("note")):
                st.caption(pick(part["note"]))
            results_table(part["columns"], part["rows"])
    elif t.get("rows"):
        results_table(t["columns"], t["rows"])
        if any(r["where"] == "cloud" for r in t["rows"]):
            zh, en = "；雲端模型（橘色）是對照組。", "; cloud models (orange) are the reference."
        else:
            zh, en = "；這項沒有測雲端模型。", "; no cloud models in this test."
        st.caption(
            tr("地端模型是 4-bit 量化，在 Mac 或 GPU 主機（模型名後標「GPU 主機」）上跑" + zh,
               "Local models are 4-bit builds on the Mac or the GPU machine (marked after the "
               "name)" + en)
        )  # fmt: skip
    elif t.get("status") == "running":
        st.info(tr("已經在測或測完了，結果整理好就會放上來。",
                   "Being run or just run; the results appear here once written up."),
                icon=":material/progress_activity:")  # fmt: skip
    else:
        st.info(tr("尚未測，預計做法見上面的說明。", "Not run yet; the plan is described above."),
                icon=":material/schedule:")  # fmt: skip


def aa_section(cat: dict) -> None:
    """Artificial Analysis: what it is, its scores for the models we tested, and each metric."""
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
    local_name, cloud_name = ui.where_names()
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
            if unit == "pct":
                return round(value * 100, 1)
            return round(value, 1 if unit == "index" else 0)

        rows = []
        for name, m in data["models"].items():
            local = aa_scores.is_local(m)
            if only_local and not local:
                continue
            row = {tr("模型", "Model"): name, where_col(): local_name if local else cloud_name}
            for f in shown:
                unit = metrics[f]["unit"]
                label = pick(metrics[f]["name"]) + (" %" if unit == "pct" else "")
                row[label] = cell(m.get(f), unit)
            rows.append(row)
        df = pd.DataFrame(rows)
        if shown and not df.empty:
            first = df.columns[2]
            df = df.sort_values(first, ascending=False, na_position="last")
        st.dataframe(ui.where_rows(df, where_col()).format(precision=1, na_rep="—"),
                     hide_index=True, width="stretch", height=height(len(df)))  # fmt: skip
        st.caption(
            tr(
                "百分比欄位是 0–100；Elo 與綜合指數是分數。點欄位標題可以排序。「—」是 AA 沒有測。",
                "Percent columns are 0–100; Elo and the index are scores. Click a header to sort. "
                '"—" means AA has not tested it.',
            )
        )

    st.markdown(f"#### {tr('各項評測在量什麼', 'What each evaluation measures')}")
    st.caption(pick(aa["example_note"]))
    for m in metrics.values():
        with st.container(border=True):
            st.markdown(f"**{pick(m['name'])}**　{pick(m['measures'])}")
            if pick(m["example"]):
                st.caption(f"{tr('例', 'Example')}：{pick(m['example'])}")
