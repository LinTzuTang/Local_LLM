"""Multi-step mode of the Language models page (views/llm.py): BFCL multi-turn, BFCL web search,
BFCL memory and the company-tools selection test on a few questions, run by the same batch scripts
as the report (src/local_llm/step_bench.py) in a background job.

The cloud switch only gates what needs the cloud: web search (Google search through Vertex) and
cloud models. Multi-turn, memory and the company test use simulated tools, so local models run them
with the switch off. A test whose files are not on this server is not offered.
Run as a script by views/llm.py (runpy), like a page of its own.
"""

import time

import jobs
import streamlit as st
from common import cloud_enabled, installed_names, kept, memory_guard, model_label, ollama_url
from i18n import tr

from local_llm import catalog, ollama_admin, step_bench

MAX_MODELS = 3
MAX_QUESTIONS = 10

KIND_LABEL = {
    "multi_turn": tr("BFCL 多輪", "BFCL multi-turn"),
    "search": tr("網路搜尋", "Web search"),
    "memory": tr("長期記憶", "Memory"),
    "company": tr("公司系統挑工具", "Company tools"),
}
KIND_HELP = {
    "multi_turn": tr(
        "多輪對話裡連續呼叫工具（模擬的檔案系統、交易、訂票等），每一輪都要接著前面的結果做；"
        "含缺工具、缺參數、長上下文。題目是我們英文 BFCL 抽樣的 100 題。",
        "Several rounds of tool calls in one conversation (simulated file system, trading, "
        "booking…), "
        "each building on the last; includes missing tools, missing parameters and long context. "
        "The 100 questions of our English BFCL sample.",
    ),
    "search": tr(
        "模型自己用搜尋工具查資料、必要時抓網頁讀，再回答。搜尋走 Vertex 的 Google 搜尋（雲端），"
        "查過的查詢會沿用快取。題目是我們抽樣的 20 題。",
        "The model searches, reads pages when it needs to, then answers. Search goes through "
        "Google search on Vertex (cloud); queries already made come from the cache. Our 20 "
        "sampled questions.",
    ),
    "memory": tr(
        "先跑該情境的前置對話讓模型把資料寫進記憶，再問之前說過的事。記憶在本機模擬，不需要雲端。",
        "First the scenario's earlier conversations, in which the model writes to memory, then "
        "questions about them. Memory is simulated here; no cloud needed.",
    ),
    "company": tr(
        "公司內部專案管理系統的 74 個真實工具定義（回傳假資料，不會碰到真的系統）："
        "從口語中文的要求挑對工具、填對參數，需要時先查出 ID。",
        "The company project-management system's 74 real tool definitions (answers are made "
        "up; the "
        "real system is never called): pick the right tool from a colloquial Chinese request, fill "
        "the arguments, look up ids first when needed.",
    ),
}
CATEGORY_LABEL = {
    "multi_turn_base": tr("多輪：一般", "Multi-turn: base"),
    "multi_turn_miss_func": tr("多輪：缺工具（要說做不到）", "Multi-turn: missing tool"),
    "multi_turn_miss_param": tr("多輪：缺參數（要追問）", "Multi-turn: missing parameter"),
    "multi_turn_long_context": tr("多輪：長上下文", "Multi-turn: long context"),
    "web_search_base": tr("網路搜尋", "Web search"),
    "D": tr("查詢", "Look-ups"),
    "S": tr("單一動作", "One action"),
    "L": tr("多步驟（要先查再做）", "Several steps (look up first)"),
    "P": tr("日期與時數換算", "Dates and hours"),
    "N": tr("不需要工具", "No tool needed"),
    "A": tr("資訊不足要反問", "Missing information: ask"),
}
SCENARIO_LABEL = {
    "finance": tr("財務", "Finance"),
    "healthcare": tr("醫療", "Healthcare"),
    "notetaker": tr("筆記", "Note-taking"),
}
BACKEND_LABEL = {
    "kv": tr("鍵值", "Key-value"),
    "vector": tr("向量", "Vector"),
    "rec_sum": tr("遞迴摘要", "Recursive summary"),
}
CONDITION_LABEL = {
    "A": tr("A：只給這個系統的 74 個工具", "A: the system's 74 tools only"),
    "B": tr("B：混入其他系統，約 150 個", "B: mixed with other systems, ~150"),
    "C": tr("C：篩成 29 個＋讓模型自己搜尋工具", "C: 29 tools plus a tool search"),
}

st.caption(
    tr(
        "多步驟的工具呼叫測試，用的是和報告完全相同的批次程式；每次挑幾題、在背景跑，可以切換頁面。"
        "地端模型用原廠建議的取樣設定（報告裡多數地端模型用 BFCL 的溫度 0.001，結果可能略有不同）。"
        "雲端開關只管會用到雲端的部分：網路搜尋與雲端模型。",
        "Multi-step tool-calling tests, run by the same batch code as the report; pick a few "
        "questions, they run in the background while you switch pages. Local models use their "
        "model-card sampling (most local runs in the report used BFCL's 0.001, so results can "
        "differ). The cloud switch only gates what uses the cloud: web search and cloud models.",
    )
)

cloud = cloud_enabled()
keys = step_bench.env_keys()
missing = {k: step_bench.missing(k) for k in step_bench.KINDS}
if all(missing.values()):
    st.info(
        tr(
            "這台伺服器沒有多步驟測試需要的檔案（BFCL 執行環境與題目）。",
            "This server does not have the files these tests need (the BFCL environment and "
            "questions).",
        ),
        icon=":material/info:",
    )
    st.stop()


def blocked(kind: str) -> str | None:
    if missing[kind]:
        return tr("這台伺服器沒有這項測試的檔案", "Not available on this server")
    if kind == "search" and not cloud:
        return tr("要先打開上方的「雲端模型」開關", "Turn on the Cloud switch first")
    if kind == "search" and "GOOGLE_APPLICATION_CREDENTIALS" not in keys:
        return tr("伺服器沒有設定 Google 雲端金鑰", "No Google cloud key on this server")
    return None


offered = [k for k in step_bench.KINDS if not (missing[k] and k == "company")]
st.session_state.setdefault("steps_kind", next(k for k in offered if not missing[k]))
kind = (
    st.segmented_control(
        tr("測試", "Test"),
        offered,
        format_func=lambda k: KIND_LABEL[k] + (" 🔒" if blocked(k) else ""),
        **kept("steps_kind"),
    )
    or offered[0]
)
st.caption(KIND_HELP[kind])
reason = blocked(kind)
if reason:
    st.warning(reason, icon=":material/lock:")
    st.stop()


@st.cache_data(ttl=600)
def questions(kind: str, backend: str = "kv") -> list[dict] | dict:
    if kind == "multi_turn":
        return step_bench.multi_turn_questions()
    if kind == "search":
        return step_bench.search_questions()
    if kind == "memory":
        return step_bench.memory_sets(backend)
    return step_bench.company_questions()


def preview(q: dict) -> str:
    first = next((t for t in q["turns"] if t), "")
    more = len(q["turns"]) - 1
    return first.replace("\n", " ")[:80] + (
        tr(f"（共 {more + 1} 輪）", f" ({more + 1} turns)") if more else ""
    )


# ------------------------------------------------------------------ questions
prereqs: list[str] = []
cond = "A"
if kind == "memory":
    c1, c2 = st.columns(2)
    backend = c1.selectbox(
        tr("記憶方式", "Memory backend"),
        list(step_bench.MEMORY_BACKENDS),
        format_func=lambda b: BACKEND_LABEL[b],
        **kept("steps_backend"),
    )
    sets = questions("memory", backend)
    scenario = c2.selectbox(
        tr("情境", "Scenario"),
        list(sets),
        format_func=lambda s: SCENARIO_LABEL.get(s, s),
        **kept("steps_scenario"),
    )
    pool = sets[scenario]["questions"]
    prereqs = sets[scenario]["prereqs"]
    st.caption(
        tr(
            f"會先跑 {len(prereqs)} 段前置對話（不計分），再問下面選的題目。",
            f"Runs {len(prereqs)} earlier conversations first (not scored), then the questions "
            "below.",
        )
    )
else:
    pool = questions(kind)
    if kind == "company":
        cond = st.radio(
            tr("工具數量", "Tool list"),
            list(CONDITION_LABEL),
            format_func=lambda c: CONDITION_LABEL[c],
            horizontal=True,
            **kept("steps_cond"),
        )
    cats = list(dict.fromkeys(q["category"] for q in pool))
    if len(cats) > 1:
        cat = st.selectbox(
            tr("類型", "Category"),
            cats,
            format_func=lambda c: (
                f"{CATEGORY_LABEL.get(c, c)}（{sum(q['category'] == c for q in pool)}）"
            ),
            **kept(f"steps_cat_{kind}"),
        )
        pool = [q for q in pool if q["category"] == cat]

by_id = {q["id"]: q for q in pool}
pick_key = f"steps_pick_{kind}"
st.session_state.setdefault(pick_key, [])
st.session_state[pick_key] = [i for i in st.session_state[pick_key] if i in by_id][
    :MAX_QUESTIONS
] or list(by_id)[:2]
picked = st.multiselect(
    tr(
        f"題目（最多 {MAX_QUESTIONS} 題，可打字搜尋）",
        f"Questions (up to {MAX_QUESTIONS}, type to search)",
    ),
    list(by_id),
    format_func=lambda i: preview(by_id[i]),
    max_selections=MAX_QUESTIONS,
    key=pick_key,
)
for i in picked:
    q = by_id[i]
    if len(q["turns"]) > 1:
        with st.expander(f"{preview(q)}"):
            for n, t in enumerate(q["turns"], 1):
                st.markdown(f"**{tr('第', 'Turn ')}{n}{tr(' 輪', '')}**：{t}")

# ------------------------------------------------------------------ models
url = ollama_url()
names = installed_names()


@st.cache_data(ttl=300)
def tool_models(url: str, names: tuple[str, ...]) -> list[str]:
    return [n for n in names if catalog.is_chat(n) and ollama_admin.supports_tools(n, url)]


local = tool_models(url, tuple(names))
cloud_models = []
if cloud:
    cloud_models = [
        k
        for k, (_, env) in step_bench.CLOUD.items()
        if env in keys and not (kind == "company" and k.startswith("gemini"))
    ]
installed = {catalog.normalize(m) for m in names}
options = local + cloud_models
st.session_state.setdefault("steps_models", [m for m in [catalog.default_model()] if m in local])
st.session_state.steps_models = [m for m in st.session_state.steps_models if m in options]
selected = st.multiselect(
    tr(
        f"模型（最多 {MAX_MODELS} 個；地端只列支援工具呼叫、已下載的模型"
        + ("；雲端模型要打開上方「雲端模型」開關）" if not cloud else "）"),
        f"Models (up to {MAX_MODELS}; downloaded local models that support tools"
        + ("; turn on Cloud for cloud models)" if not cloud else ")"),
    ),
    options,
    format_func=lambda m: (
        step_bench.CLOUD[m][0] + tr("（雲端）", " (cloud)")
        if m in step_bench.CLOUD
        else model_label(m, installed)
    ),
    max_selections=MAX_MODELS,
    **kept("steps_models"),
)
memory_guard([m for m in selected if m not in step_bench.CLOUD])

est = step_bench.estimate_seconds(kind, len(picked), len(prereqs)) * max(len(selected), 1)
st.caption(
    tr(
        f"粗估：約 {max(est // 60, 1)} 分鐘"
        "（以 Gemma 4 31B 在 Mac 上的速度估；小模型和雲端較快）。",
        f"Rough estimate: about {max(est // 60, 1)} min (Gemma 4 31B speed on the Mac; small "
        "and cloud "
        "models are faster).",
    )
)


def work(job: jobs.Job, base_url: str) -> None:
    s = job.state
    for n, r in enumerate(s["results"]):
        if job.cancel.is_set():
            break
        r["status"] = "running"

        def progress(done: int, total: int, r=r, n=n) -> None:
            r["done"] = done
            job.detail = f"{n + 1}/{len(s['results'])} {r['label']} · {done}/{total}"
            job.progress = (n + min(done / max(total, 1), 1)) / len(s["results"])

        out = step_bench.run(
            s["kind"],
            r["model"],
            s["ids"],
            cond=s["cond"],
            run_dir=step_bench.new_run_dir(job.id, r["model"]),
            ollama_url=base_url,
            progress=progress,
            cancel=job.cancel,
        )
        r.update(out, status="done")


def render(job: jobs.Job) -> None:
    s = job.state
    st.markdown(
        f"**{KIND_LABEL[s['kind']]}**"
        + (f" · {CONDITION_LABEL[s['cond']]}" if s["kind"] == "company" else "")
        + tr(f" · {s['n']} 題", f" · {s['n']} questions")
    )
    for r in s["results"]:
        with st.container(border=True):
            rows = r.get("rows") or []
            ok = sum(x["ok"] for x in rows)
            head = f"**{r['label']}**"
            if r["status"] == "done":
                head += tr(f"　答對 {ok}/{len(rows)}", f"   {ok}/{len(rows)} correct")
            st.markdown(head)
            if r["status"] == "waiting":
                st.caption(tr("等待中…", "Waiting…"))
                continue
            if r["status"] == "running":
                st.caption(
                    tr(f"執行中…已完成 {r.get('done', 0)} 段", f"Running… {r.get('done', 0)} done")
                )
                continue
            for x in rows:
                q = s["questions"].get(x["id"], {})
                mark = "✅" if x["ok"] else "❌"
                sec = f" · {x['seconds']} s" if x.get("seconds") else ""
                with st.expander(f"{mark} {preview(q) if q else x['id']}{sec}"):
                    if not x["ok"] and x["why"]:
                        st.error(x["why"], icon=":material/cancel:")
                    for t, steps in enumerate(x["turns"], 1):
                        asked = (q.get("turns") or [])[t - 1 : t]
                        if len(x["turns"]) > 1:
                            label = tr(f"第 {t} 輪", f"Turn {t}")
                            st.markdown(f"**{label}**：{asked[0] if asked else ''}")
                        for stp in steps:
                            if "calls" in stp:
                                st.code(
                                    "\n".join(str(c) for c in stp["calls"])
                                    or tr("（沒有呼叫工具）", "(no call)")
                                )
                            elif stp.get("text"):
                                st.markdown(stp["text"])
            st.caption(tr(f"紀錄：`{r.get('log', '')}`", f"Log: `{r.get('log', '')}`"))
    if job.running:
        a, b = st.columns([5, 1])
        a.info(
            f"{job.detail or tr('準備中', 'Preparing')} · {job.elapsed() / 60:.1f} min",
            icon=":material/progress_activity:",
        )
        if b.button(tr("停止", "Stop"), key=f"stop_{job.id}"):
            job.cancel.set()
    elif job.error:
        st.error(job.error)


@st.fragment(run_every=3)
def live(job: jobs.Job) -> None:
    if not job.running:
        st.rerun()
    render(job)


job = jobs.get(st.session_state.get("steps_job"))
busy = bool(job and job.running)
if st.button(
    tr("開始測試", "Run"),
    type="primary",
    icon=":material/play_arrow:",
    disabled=busy or not selected or not picked,
):
    ids = (prereqs + picked) if kind == "memory" else picked
    job = jobs.Job(
        "steps",
        tr(
            f"{KIND_LABEL[kind]} {len(picked)} 題", f"{KIND_LABEL[kind]}, {len(picked)} question(s)"
        ),
        "views/llm.py",
        state={
            "kind": kind,
            "cond": cond,
            "ids": ids,
            "n": len(picked),
            "questions": {i: by_id[i] for i in picked},
            "started": time.time(),
            "results": [
                {
                    "model": m,
                    "label": step_bench.CLOUD[m][0] if m in step_bench.CLOUD else m,
                    "status": "waiting",
                }
                for m in selected
            ],
        },
    )
    jobs.start(job, lambda j: work(j, url))
    st.session_state.steps_job = job.id
    st.rerun()

if job and job.running:
    live(job)
elif job:
    render(job)
