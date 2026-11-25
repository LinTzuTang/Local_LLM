"""Scores: every measured result in one place. Pick a kind of model (language, speech, document
OCR); read "which model for what" first, then the list of tests (state, last run, one-sentence
conclusion), then any test: what it measures, how it was run and the results.

Nothing runs here and no server is needed (no Ollama or gateway check): it only reads
data/lab/llm_benchmarks.json (language models, AA), data/lab/recommendations.json and
data/lab/scores_methods.json (speech and OCR texts, dates), and the *_benchmark.json files the
speech and OCR views draw. The test pages (Language models, Compare speech models, Other models)
only run tests now and link here with ?kind=llm|speech|ocr.
"""

import html
import json
from pathlib import Path

import asr_bench
import ocr_compare
import scores_llm
import speech_bench
import speech_meeting
import speech_pick
import speech_simul
import speech_tts
import streamlit as st
import ui
import voice_bench
from common import kept
from i18n import pick, tr

METHODS = Path(__file__).resolve().parents[2] / "data" / "lab" / "scores_methods.json"

ui.page_header(
    "leaderboard",
    tr("評測分數", "Scores"),
    tr(
        "我們實際測過的結果。先選模型類型，看「依情境選模型」；再看「做了哪些測試」的清單，"
        "選一項看測了什麼、怎麼測、結果。每項都標狀態與最後測試日期，測試還在持續進行。",
        "What we measured. Pick a kind of model and read which model to use for what; then the "
        "list of tests, and choose one for what it measures, how it was run and the results. "
        "Every test shows its state and when it was last run; testing goes on.",
    ),
    eyebrow=tr("模型", "Models"),
)


def state(t: dict) -> str:
    """done; more models being added to a finished test; run, being written up; or planned."""
    s = t.get("status", "done")
    if s == "running":
        return "adding" if (t.get("rows") or t.get("tables")) else "writing"
    return s


STATES = {  # state -> (zh, en, icon, badge colour)
    "done": ("完成", "Done", "check_circle", "green"),
    "adding": ("完成・補測新模型中", "Done, adding models", "add_circle", "blue"),
    "writing": ("已跑完・整理中", "Run, being written up", "progress_activity", "orange"),
    "planned": ("待測", "Planned", "schedule", "gray"),
}


def state_name(t: dict) -> str:
    zh, en, _, _ = STATES[state(t)]
    return tr(zh, en)


def state_badge(t: dict) -> str:
    _, _, icon, color = STATES[state(t)]
    return f":{color}-badge[:material/{icon}: {state_name(t)}]"


def option_label(t: dict) -> str:
    """A test in the picker: its name, then its date or state."""
    name = pick(t["name"])
    if state(t) == "planned":
        return f"{name}　·　{tr('待測', 'planned')}"
    date = (t.get("date") or "")[5:]  # 2026-11-23 -> 11-23
    if state(t) == "writing":
        return f"{name}　·　{tr('整理中', 'being written up')}"
    return f"{name}　·　{date}" if date else name


def method_card(t: dict) -> None:
    """State and date, the conclusion, then what / why / how / example / limits."""
    planned = state(t) == "planned"
    with st.container(border=True):
        date = t.get("date") or "—"
        st.markdown(f"{state_badge(t)}　{tr('最後測試：', 'Last run: ')}{date}")
        if pick(t.get("pending")):
            st.caption(f":material/progress_activity: {pick(t['pending'])}")
        if pick(t.get("headline")):
            st.markdown(f"**{tr('結論', 'Conclusion')}**　**{pick(t['headline'])}**")
        parts = [
            ("measures", tr("測什麼", "What it measures")),
            ("why", tr("為什麼要測", "Why")),
            ("method", tr("打算怎麼測", "How we plan to run it") if planned else tr("怎麼測", "Method")),
        ]  # fmt: skip
        for key, label in parts:
            if pick(t.get(key)):
                st.markdown(f"**{label}**　{pick(t[key])}")
        if pick(t.get("example")):
            st.markdown(f"**{tr('例題', 'Example')}**")
            st.markdown(pick(t["example"]))
        if pick(t.get("limits")):
            st.markdown(f"**{tr('限制', 'Limits')}**　{pick(t['limits'])}")
        if pick(t.get("todo")):
            st.caption(f":material/pending: {pick(t['todo'])}")
        if pick(t.get("platform")):
            st.caption(f":material/play_circle: {pick(t['platform'])}")


TABLE_CSS = (
    "<style>.lab-uses td,.lab-uses th{border:1px solid var(--lab-border);padding:6px 8px;"
    "vertical-align:top}.lab-uses td small{color:var(--lab-muted)}</style>"
)


def html_table(head: str, body: list[str]) -> None:
    st.markdown(
        '<div style="overflow-x:auto"><table class="lab-uses" style="width:100%;'
        'border-collapse:collapse;font-size:0.92rem">'
        f'<thead><tr style="text-align:left">{head}</tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table></div>" + TABLE_CSS,
        unsafe_allow_html=True,
    )


def uses_table(columns: dict, rows: list[dict], names: dict | None = None) -> None:
    """Use | local pick | why | cloud: the local column blue, the cloud column orange (as on the
    speech recommendations). `names` (test id -> name) adds which tests each row rests on."""
    cells = [html.escape(tr(columns["zh"][i], columns["en"][i])) for i in range(4)]
    head = (
        f"<th>{cells[0]}</th>"
        f'<th style="background:{ui.LOCAL_COLOR};color:#fff">{cells[1]}</th>'
        f"<th>{cells[2]}</th>"
        f'<th style="background:{ui.CLOUD_COLOR};color:#fff">{cells[3]}</th>'
    )
    body = []
    for r in rows:
        use, local, why, cloud = (html.escape(c) for c in tr(r["zh"], r["en"]))
        refs = [names[i] for i in r.get("tests", []) if names and i in names]
        if refs:
            why += "<br><small>" + tr("見：", "See: ") + html.escape("、".join(refs)) + "</small>"
        body.append(
            f"<tr><td><b>{use}</b></td>"
            f'<td style="background:{ui.LOCAL_TINT}">{local}</td>'
            f'<td style="color:var(--lab-muted)">{why}</td>'
            f'<td style="background:{ui.CLOUD_TINT}">{cloud}</td></tr>'
        )
    html_table(head, body)


def overview(tests: list[dict]) -> None:
    """Every test of this kind on one table: state, last run, conclusion (or, if planned, what it
    will answer)."""
    newest = max((t["date"] for t in tests if t.get("date")), default=None)
    st.markdown(f"### {tr('做了哪些測試', 'What we tested')}")
    counts = {s: sum(state(t) == s for t in tests) for s in STATES}
    st.caption(
        tr(
            f"共 {len(tests)} 項：完成 {counts['done'] + counts['adding']}（其中 {counts['adding']} 項還在補測新模型）、"
            f"整理中 {counts['writing']}、待測 {counts['planned']}。這一類最近一次測試：{newest or '—'}。",
            f"{len(tests)} tests: {counts['done'] + counts['adding']} done ({counts['adding']} "
            f"still adding models), {counts['writing']} being written up, {counts['planned']} "
            f"planned. Latest run in this group: {newest or '—'}.",
        )
    )  # fmt: skip
    head = "".join(
        f"<th>{h}</th>"
        for h in (tr("測試", "Test"), tr("狀態", "State"), tr("最後測試", "Last run"),
                  tr("結論（待測：要回答什麼）", "Conclusion (planned: the question)"))
    )  # fmt: skip
    body = []
    for t in tests:
        said = pick(t.get("headline")) or pick(t.get("measures")) or ""
        body.append(
            f"<tr><td><b>{html.escape(pick(t['name']))}</b></td>"
            f"<td style='white-space:nowrap'>{html.escape(state_name(t))}</td>"
            f"<td style='white-space:nowrap'>{html.escape(t.get('date') or '—')}</td>"
            f"<td>{html.escape(said)}</td></tr>"
        )
    html_table(head, body)


def pick_test(kind: str, tests: list[dict], first: str | None = None) -> dict:
    by_id = {t["id"]: t for t in tests}
    st.markdown(f"### {tr('看一項測試', 'One test in detail')}")
    if first in by_id:
        st.session_state.setdefault(f"scores_test_{kind}", first)
    chosen = st.selectbox(
        tr("選一項看測了什麼、怎麼測、結果", "Choose a test: what it measures, how, the results"),
        list(by_id), format_func=lambda i: option_label(by_id[i]), **kept(f"scores_test_{kind}"),
    )  # fmt: skip
    return by_id[chosen]


GLOSSARY = [
    ("token", "模型讀寫文字的單位，中文大約一個字一到兩個 token。「每秒 20 token」大約是每秒十幾個中文字。",
     "The unit models read and write text in; a Chinese character is about one or two tokens."),
    ("agent／工具呼叫", "模型不只回答，還會自己決定呼叫哪個工具（查資料、開卡片、查天氣），拿到結果再繼續做。",
     "Agent / tool calling: the model decides which tool to call (look something up, open a card), reads the result and carries on."),
    ("地端／雲端", "地端＝模型跑在公司自己的機器（Mac 或 GPU 主機），資料不出公司；雲端＝送到外部服務（Claude、GPT、Gemini、Azure 等），要付費。",
     "Local = on the company's own machines (the Mac or the GPU machine), data stays in-house; cloud = an outside, paid service."),
    ("4-bit／FP8 量化", "把模型的數字存得比較省，記憶體小、速度快，分數通常只掉一點。地端多用 4-bit，vLLM 用 FP8。",
     "Quantization: storing the model's numbers more compactly (4-bit, FP8) for less memory and more speed, usually at a small cost."),
    ("vLLM／Ollama", "兩種跑模型的伺服器軟體。Ollama 簡單、平台現在用它；vLLM 專門給很多人同時用。",
     "Two kinds of model server: Ollama is simple and what the platform uses now; vLLM is built for many users at once."),
    ("思考（開／關）", "模型回答前先寫一段推理。開著通常比較準、但比較慢。",
     "Thinking: the model reasons in writing before answering; usually more accurate, but slower."),
    ("RAG", "先從文件裡找出相關段落，再交給模型回答，而不是把整份文件丟進去。",
     "Retrieval first: find the relevant passages, then let the model answer from them."),
    ("消融版", "把模型「拒絕回答」的傾向拿掉的改造版，用來問一般模型會擋的問題（例如 Heretic ARA）。",
     "Abliterated build: a model with its tendency to refuse removed (e.g. Heretic ARA)."),
    ("BFCL／ACEBench", "公開的工具呼叫題庫：給一句話和一份工具清單，看模型挑的工具與參數對不對。",
     "Public tool-calling benchmarks: a request and a tool list; is the chosen tool with its arguments right?"),
    ("LiveCodeBench／Aider", "寫程式題庫：前者從零寫演算法，後者在既有專案裡把程式寫完、通過單元測試。",
     "Coding benchmarks: algorithms from scratch, and finishing code in existing projects."),
    ("CER／WER", "語音辨識的錯字率（中文按字、英文按詞），越低越好。",
     "Error rates of speech recognition (characters for Chinese, words for English); lower is better."),
    ("DER", "講者辨識錯誤率：漏掉、誤判、認錯人的時間比例，越低越好。",
     "Diarization error rate: share of time missed, falsely detected or given to the wrong speaker."),
    ("chrF", "翻譯和標準譯文的相似度，越高越好。",
     "Similarity of a translation to the reference translation; higher is better."),
    ("Elo", "兩兩比較算出的分數，差 400 分表示十次約輸九次。",
     "A score from pairwise comparisons; 400 points apart means losing about nine times in ten."),
]  # fmt: skip


def glossary() -> None:
    with st.expander(tr("名詞解釋", "Glossary"), icon=":material/menu_book:"):
        for term, zh, en in GLOSSARY:
            st.markdown(f"**{term}**　{tr(zh, en)}")


def methods() -> dict:
    try:
        return json.loads(METHODS.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


KINDS = {
    "llm": (tr("語言模型", "Language models"), "chat"),
    "speech": (tr("語音模型", "Speech models"), "graphic_eq"),
    "ocr": (tr("文件 OCR", "Document OCR"), "document_scanner"),
}
if st.query_params.get("kind") in KINDS:  # a link from a test page (?kind=speech)
    st.session_state.scores_kind = st.query_params["kind"]
    del st.query_params["kind"]
st.session_state.setdefault("scores_kind", "llm")
kind = (
    st.segmented_control(
        tr("模型類型", "Kind of model"), list(KINDS),
        format_func=lambda k: f":material/{KINDS[k][1]}: {KINDS[k][0]}",
        **kept("scores_kind"),
    )
    or "llm"
)  # fmt: skip
st.markdown(f"### {tr('依情境選模型', 'Which model for what')}")

# ---------------------------------------------------------------- language models
if kind == "llm":
    cat = scores_llm.catalog()
    if not cat:
        st.warning(
            tr("找不到 data/lab/llm_benchmarks.json。", "data/lab/llm_benchmarks.json is missing.")
        )
        st.stop()
    uses = cat["uses"]
    aa = {"id": "aa", "name": {"zh": "外部評測：Artificial Analysis", "en": "External: Artificial Analysis"},
          "status": "done", "date": None,
          "headline": {"zh": "AA 用英文、完整精度測各家模型；短任務地端追上了，長時間自主工作還差很遠。",
                       "en": "AA tests full-precision models in English; local models have caught up on short tasks, not on long autonomous work."}}  # fmt: skip
    with_results = [t for t in cat["ours"] if t.get("rows") or t.get("tables")]
    later = [t for t in cat["ours"] if not (t.get("rows") or t.get("tables"))]
    tests = [*with_results, aa, *later]
    st.caption(
        tr(
            f"取自語言模型總報告（{uses.get('date') or cat['updated']} 更新）。地端建議是 4-bit 量化、"
            "在我們自己的機器上跑；「見：」是依據來自哪一項測試。",
            f"From the language-model report (updated {uses.get('date') or cat['updated']}). "
            'Local picks are 4-bit builds on our own machines; "See" names the tests behind each.',
        )
    )
    uses_table(uses["columns"], uses["rows"], {t["id"]: pick(t["name"]) for t in tests})
    if pick(uses.get("note")):
        st.caption(f":material/help: {pick(uses['note'])}")
    overview(tests)
    t = pick_test("llm", tests, first="company_tools")
    if t["id"] == "aa":
        scores_llm.aa_section(cat)
    else:
        method_card(t)
        scores_llm.test_results(t)
    glossary()
    st.stop()

info = methods()
# ---------------------------------------------------------------- speech models
if kind == "speech":
    rec = speech_pick.load()
    if rec:
        st.caption(
            tr(
                f"{rec['date']} 的實測結果。藍色＝地端（資料不出公司），橘色＝雲端（錄音要上傳、要付費）；"
                "結果差不多的模型一起列出。括號裡是錯字率、開口時間或翻譯分數。",
                f"Results of {rec['date']}. Blue = local (data stays in-house), orange = cloud "
                "(audio is uploaded, paid); models with similar results are listed together. In "
                "brackets: error rate, time to talk or translation score.",
            )
        )
        st.markdown(speech_pick.table(rec["rows"]), unsafe_allow_html=True)
    views = {
        "asr": asr_bench.benchmark_view,
        "diar": speech_meeting.benchmark_view,
        "live": speech_bench.benchmark_page,
        "simul": speech_simul.bench_view,
        "tts": speech_tts.benchmark_view,
        "voice": voice_bench.section,
    }
    tests = (info.get("speech") or {}).get("tests") or []
    overview(tests)
    todo = [f"{pick(t['name'])}：{pick(t['todo'])}" for t in tests if pick(t.get("todo"))]
    if todo:
        st.caption(":material/pending: " + tr("還沒測：", "Not tested yet: ") + "　".join(todo))
    t = pick_test("speech", tests)
    method_card(t)
    views[t["id"]]()
    glossary()
    st.stop()

# ---------------------------------------------------------------- document OCR
ocr = info.get("ocr") or {}
if ocr.get("uses"):
    st.caption(
        tr(
            f"{ocr['uses']['date']} 的實測結果（Apple M5 Pro）。數字是和人工標註的差異（0＝完全一樣，越低越好）與每頁秒數。",
            f"Results of {ocr['uses']['date']} (Apple M5 Pro). Figures: difference from the human "
            "labels (0 = identical, lower is better) and seconds per page.",
        )
    )
    uses_table(ocr["uses"]["columns"], ocr["uses"]["rows"])
tests = ocr.get("tests") or []
overview(tests)
st.markdown(f"### {tr('測試內容與結果', 'The test and its results')}")
t = tests[0]  # one test: no picker
method_card(t)
ocr_compare.bench_view()
glossary()
