"""Language models on the Scores page (views/scores.py): our own test results and Artificial
Analysis's scores for the same models, each with how the test is done, so a number is never shown
without its method.

Our results, the "pick a model" table, every method text, each test's last test date and state
come from data/lab/llm_benchmarks.json (written by scripts/llm_eval/export_scores.py from the
status document). AA's numbers are read from the copy this server fetched
(src/local_llm/aa_scores.py; an administrator can refresh it here).
"""

import json
import re
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
    # the first column carries the model with its machine and notes: wide, so they are not cut off
    wide = {columns[lang][0]: st.column_config.TextColumn(width="large")}
    if any(r["where"] == "cloud" for r in rows):
        df.insert(
            1, where_col(), [cloud_name if r["where"] == "cloud" else local_name for r in rows]
        )
        st.dataframe(ui.where_rows(df, where_col()), hide_index=True, width="stretch",
                     height=height(len(df)), column_config=wide)  # fmt: skip
    else:
        st.dataframe(df, hide_index=True, width="stretch", height=height(len(df)),
                     column_config=wide)  # fmt: skip


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
        first = tr("由最好排到最差。", "Best first. ") if t.get("sorted") else ""
        st.caption(
            first + tr("地端模型是 4-bit 量化；秒數後面括號是在哪台機器量的" + zh,
                       "Local models are 4-bit builds; the machine a time was measured on is "
                       "in brackets after it" + en)
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
        # Streamlit shows a missing number as "None" whatever the Styler says: write the dash in
        scores = list(df.columns[2:])
        df[scores] = df[scores].astype(object).where(df[scores].notna(), "—")
        styled = ui.where_rows(df, where_col()).format(lambda x: ui.trim_number(x, 1), na_rep="—")
        st.dataframe(styled, hide_index=True, width="stretch", height=height(len(df)),
                     column_config={tr("模型", "Model"): st.column_config.TextColumn(
                         width="medium")})  # fmt: skip
        st.caption(
            tr(
                "百分比欄位是 0–100；Elo 與綜合指數是分數。"
                "依第一個評測由高到低排。「—」是 AA 沒有測。",
                "Percent columns are 0–100; Elo and the index are scores. "
                "Sorted by the first evaluation. "
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


def _board(c: dict, rows: list[dict]) -> None:
    """One AA list as a table: model, creator, local/cloud (when AA says), then its measures."""
    if not rows:
        st.caption(tr("這一類沒有開源權重的模型。", "No open-weight models on this board."))
        return
    local_name, cloud_name = ui.where_names()
    out = []
    for r in rows:
        row = {tr("模型", "Model"): r.get("model"), tr("開發商", "Creator"): r.get("creator")}
        if r.get("parameters") is not None:
            row[tr("參數（B）", "Parameters (B)")] = r.get("parameters")
        if any(x.get("open_weights") is not None for x in rows):
            row[where_col()] = local_name if r.get("open_weights") else cloud_name
        for m in c.get("metrics", []):
            v = r.get(m["field"])
            if isinstance(v, (int, float)) and m.get("unit") == "pct" and abs(v) <= 1:
                v = round(v * 100, 1)
            elif isinstance(v, float):
                v = round(v, 0 if m.get("unit") in ("elo", "usd") and abs(v) >= 100 else 1)
            label = pick(m["name"]) + (" %" if m.get("unit") == "pct" else "")
            label += tr("（越低越好）", " (lower is better)") if m.get("lower_better") else ""
            row[label] = v
        out.append(row)
    df = pd.DataFrame(out)
    shown = (ui.where_rows(df, where_col()) if where_col() in df.columns
             else df.style.format(ui.trim_number, na_rep="—"))  # fmt: skip
    st.dataframe(shown, hide_index=True, width="stretch", height=height(len(df)))


def aa_media_section(media: dict | None, category: str) -> None:
    """One of AA's leaderboards from local_llm.aa_media (language models, speech, image, video,
    music): how AA tests it, the top of the board, the top open-weight models, and for language
    models the top small open models (4B-40B, the size we run)."""
    if not media or category not in (media.get("categories") or {}):
        st.info(tr("這一類的 AA 資料還沒抓到。", "This AA leaderboard has not been fetched yet."))
        return
    c = media["categories"][category]
    if pick(c.get("about")):
        st.markdown(f"**{tr('AA 怎麼測', 'How AA tests it')}**　{pick(c['about'])}")
    total = f"　·　{tr('榜上共', 'On the board:')} {c['total']}" if c.get("total") else ""
    st.caption(
        tr(f"資料日期：{c.get('fetched') or media.get('fetched', '—')}",
           f"Fetched: {c.get('fetched') or media.get('fetched', '—')}")
        + total + (f"　·　[{tr('AA 原始頁面', 'AA page')}]({c['url']})" if c.get("url") else "")
    )  # fmt: skip
    lists = [(tr(f"總排名前 {len(c.get('rows', []))}", f"Top {len(c.get('rows', []))} overall"),
              c.get("rows", []))]  # fmt: skip
    if "open_rows" in c:
        lists.append((tr("開源權重（可以自己架）", "Open weights (we could host)"), c["open_rows"]))
    if "small_rows" in c:
        lists.append((tr("小模型 4B–40B（我們主要測的大小）", "Small, 4B-40B (the size we run)"),
                      c["small_rows"]))  # fmt: skip
    if len(lists) == 1:
        _board(c, lists[0][1])
    else:
        for tab, (_, rows) in zip(st.tabs([name for name, _ in lists]), lists, strict=True):
            with tab:
                _board(c, rows)
    st.caption(
        tr(
            "依 AA 的主要指標排序。開源權重的模型標成地端（可以自己架），其餘是雲端服務。",
            "In AA's order by its main measure. Open-weight models are marked local (we could "
            "host them), the rest are cloud services.",
        )
    )


# ---------------------------------------------------------------- compare chosen models
# The headline figure of each test for the side-by-side summary:
# (test id, table index, columns joined with ／, label).
SUMMARY = [
    ("bfcl_en", 0, [1], ("英文工具呼叫 BFCL（官方權重）",
                         "English tool calling, BFCL (official weighting)")),
    ("tools_zh", 0, [1], ("繁中工具呼叫 BFCL", "Traditional Chinese tool calling, BFCL")),
    ("tools_zh", 0, [3], ("ACEBench 中文", "ACEBench Chinese")),
    ("company_tools", 0, [1, 2, 3], ("公司工具 A／B／C", "Company tools A / B / C")),
    ("agentic", 0, [1], ("網路搜尋＋記憶（65 題）", "Web search + memory (65)")),
    ("stance", 0, [4], ("台灣立場：照中國說法的題數（沒加提示，越少越好）",
                        "Taiwan stance: China-line answers (no prompt, fewer is better)")),
    ("lcb", 0, [1], ("寫程式：LiveCodeBench（60 題）", "Coding: LiveCodeBench (60)")),
    ("aider", 0, [1], ("寫程式：Aider（75 題）", "Coding: Aider (75)")),
    ("zh_qa", 0, [6], ("中文：不知道就說不知道、照文件回答（60 題）",
                       "Chinese: saying \"I don't know\", answering from documents (60)")),
    ("zh_writing", 0, [1], ("中文辦公寫作（1–5 分）", "Chinese office writing (1-5)")),
    ("ncb", 0, [2], ("寫程式：用中文問（NaturalCodeBench 58 題）",
                     "Coding asked in Chinese (NaturalCodeBench, 58)")),
    ("speed", 0, [1], ("速度：公司工具每題秒數（GPU 主機，越低越好）",
                       "Speed: seconds per company-tool task (GPU machine, lower is better)")),
    ("capacity", 2, [3], ("一張 RTX PRO 6000：模擬真實使用者在線人數",
                          "One RTX PRO 6000: simulated users online")),
]  # fmt: skip
# Rows that are another setting of the same model, not its usual one.
_OTHER_SETTING = ("關思考", "原廠取樣", "溫度", "只跑", "turned off", "vendor sampling")
# Rows that are tests (thinking), abliterated builds, or a re-cut of tools_zh (zh_en_gap).
# moe: its rows are groups; coding_fixed: its rows are Ollama tags, not the names used elsewhere
_SKIP_TESTS = {"thinking", "abliterated", "zh_en_gap", "moe", "coding_fixed"}


def model_key(name: str) -> str | None:
    """The model a row is about: its name without the bracketed setting or machine. None for rows
    that are not one model (several models in one row, numbers of users)."""
    base = name.replace("*", "")
    base = base.split("（")[0].split(" (")[0].strip()
    base = {"Llama-Breeze2 8B": "Breeze 2 8B", "Gemma 4 31B 原版": "Gemma 4 31B",
            "Qwen3.8 27B 原版": "Qwen3.8 27B"}.get(base, base)  # fmt: skip
    if not base or "／" in base or "、" in base or base[0].isdigit():
        return None
    return base


def _parts(t: dict) -> list[dict]:
    if t.get("tables"):
        return t["tables"]
    return [{"columns": t["columns"], "rows": t["rows"]}] if t.get("rows") else []


def model_index(cat: dict) -> tuple[dict, dict]:
    """{model: {(test id, table index): [rows]}} over every test whose rows are models, and
    {model: its English name}, models in the order they first appear (best of English BFCL
    first)."""
    found: dict[str, dict] = {}
    english: dict[str, str] = {}
    for t in cat["ours"]:
        if t["id"] in _SKIP_TESTS:
            continue
        for i, part in enumerate(_parts(t)):
            if part["columns"]["zh"][0] != "模型":
                continue
            for r in part["rows"]:
                key = model_key(r["zh"][0])
                if not key:
                    continue
                found.setdefault(key, {}).setdefault((t["id"], i), []).append(r)
                english.setdefault(key, model_key(r["en"][0]) or key)
    return found, english


def _usual(rows: list[dict]) -> dict:
    """The row with the model's usual setting: not a thinking-off / other-sampling run, and for
    the stance test the one without the Taiwan prompt."""
    for r in rows:
        note = r["zh"][0] + " " + r["en"][0]
        if any(w in note for w in _OTHER_SETTING):
            continue
        if len(r["zh"]) > 1 and r["zh"][1] == "有":  # stance: system prompt on
            continue
        return r
    return rows[0]


def compare_section(cat: dict) -> None:
    """Pick several models; one summary table of each test's headline figure, then every test's
    full rows for just those models."""
    found, english = model_index(cat)
    name = (lambda k: k) if current() == "zh" else english.get
    st.session_state.setdefault(
        "scores_compare",
        [m for m in ("Gemma 4 31B", "Qwen3.8 27B", "Qwen3.6 35B-A3B coding", "Claude Sonnet 5.5")
         if m in found],
    )  # fmt: skip
    chosen = st.multiselect(
        tr("勾選要比較的模型（可以多選）", "Models to compare (choose several)"),
        list(found), format_func=name, **kept("scores_compare"),
        placeholder=tr("選模型…", "Choose models…"),
    )  # fmt: skip
    if not chosen:
        st.info(tr("先勾選至少一個模型。", "Choose at least one model."),
                icon=":material/checklist:")  # fmt: skip
        return
    by_id = {t["id"]: t for t in cat["ours"]}
    local_name, cloud_name = ui.where_names()

    def head(m: str) -> str:
        rows = [r for rs in found[m].values() for r in rs]
        return name(m) + ("（" + cloud_name + "）" if rows and rows[0]["where"] == "cloud" else "")

    lang = current() if current() in ("zh", "en") else "zh"
    table, more = [], False
    for test_id, part, cols, label in SUMMARY:
        if test_id not in by_id:
            continue
        row = {tr("測試", "Test"): tr(*label)}
        for m in chosen:
            rows = found[m].get((test_id, part))
            if not rows:
                row[head(m)] = "—"
                continue
            r = _usual(rows)
            cell = "／".join(str(r[lang][c]) for c in cols if c < len(r[lang]))
            cell = re.sub(r"（[^）]*）| \([^)]*\)", "", cell)  # notes are in the full table below
            if "重跑" in r["zh"][0]:  # a partly failed run (Inkling's search)
                cell += tr("（部分重跑中）", " (partly being rerun)")
            if len(rows) > 1:
                cell += " ＊"
                more = True
            row[head(m)] = cell
        table.append(row)
    st.markdown(f"**{tr('總覽：各測試的主要分數', 'Summary: the main figure of each test')}**")
    st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch", height=height(len(table)))
    notes = [tr("「—」是這個模型沒有測這一項。", '"—" means the model was not in that test.')]
    if more:
        notes.append(tr("＊ 這個模型在這一項有不只一種設定（例如關思考、加提示），"
                        "這裡放平常的設定，"
                        "下面的完整結果全部列出。",
                        "* More than one setting of this model in that test (e.g. thinking off, "
                        "a prompt); the usual one is shown here, all of them below."))  # fmt: skip
    notes.append(tr("消融版請看「消融版 vs 原版」那一項。",
                    'Abliterated builds: see the "abliterated vs original" test.'))  # fmt: skip
    st.caption(" ".join(notes))

    full = tr("各測試的完整結果（只留勾選的模型）", "Each test in full (chosen models only)")
    st.markdown(f"**{full}**")
    for t in cat["ours"]:
        if t["id"] in _SKIP_TESTS:
            continue
        shown = []
        for i, part in enumerate(_parts(t)):
            rows = [r for m in chosen for r in found[m].get((t["id"], i), [])]
            keep = [r for r in part["rows"] if any(r is x for x in rows)]  # the test's own order
            if keep:
                shown.append((part, keep))
        if not shown:
            continue
        with st.expander(f"{pick(t['name'])}　·　{t.get('date') or '—'}"):
            if pick(t.get("headline")):
                st.caption(pick(t["headline"]))
            for part, keep in shown:
                if pick(part.get("title")):
                    st.markdown(f"**{pick(part['title'])}**")
                results_table(part["columns"], keep)
