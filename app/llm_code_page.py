"""Coding mode of the Language models page (views/llm.py): LiveCodeBench v6 on a few questions,
run by the same batch script and tester as the report (src/local_llm/code_bench.py) in a
background job. Local models only (the report has no cloud run of this test).
Run as a script by views/llm.py (runpy), like a page of its own.
"""

import time

import jobs
import streamlit as st
from common import installed_names, kept, memory_guard, model_label, ollama_url
from i18n import tr

from local_llm import catalog, code_bench, ollama_admin

MAX_MODELS = 3
MAX_QUESTIONS = 6
DIFFICULTY = {
    "easy": tr("簡單", "Easy"),
    "medium": tr("中等", "Medium"),
    "hard": tr("困難", "Hard"),
}
WHY = {
    "no answer (stopped or failed)": tr(
        "沒有回答（已停止或失敗）", "No answer (stopped or failed)"
    ),
    "hit the 32,768-token cap before finishing": tr(
        "一直思考到 32,768 token 上限，沒交出程式", "Hit the 32,768-token cap before finishing"
    ),
    "no code block in the answer": tr("回答裡沒有程式區塊", "No code block in the answer"),
    "not scored": tr("沒有評分", "Not scored"),
    "failed at least one test": tr("至少一組測資沒過", "Failed at least one test"),
}

st.caption(
    tr(
        "LiveCodeBench v6：讀題目寫一支 Python 程式，公開與隱藏測資全部通過才算對。"
        "用的是和報告相同的批次程式與評分程式；挑幾題在背景跑，可以切換頁面。只跑地端模型。",
        "LiveCodeBench v6: write a Python program from the statement; it must pass every public "
        "and hidden test. Run by the same batch script and tester as the report; pick a few "
        "questions, they run in the background while you switch pages. Local models only.",
    )
)
with st.expander(tr("這個測試怎麼測", "How this test works"), icon=":material/science:"):
    st.markdown(
        tr(
            "**測什麼**：模型能不能從題目敘述寫出一支正確的 Python 程式。\n\n"
            "**題目**：LiveCodeBench v6 在 2025-01～04 新增的程式競賽題（AtCoder、LeetCode）。"
            "用新題是為了避開模型訓練時看過的題目。這裡列的是批次測試每個模型都做的同一批 60 題"
            "（簡單、中等、困難各 20）。\n\n"
            "**怎麼跑**：用 LiveCodeBench 官方提示詞，每題只答一次；模型用自己的預設取樣、思考開，"
            "回答上限 32,768 token，撞到上限算錯。取回答裡最後一個程式區塊，"
            "用 LiveCodeBench 的評分程式跑公開與隱藏測資（每組 6 秒），**全部通過才算對**。\n\n"
            "**怎麼讀**：簡單題多數模型都會；差別在困難題和「思考停不下來」。"
            "這是從零寫一支演算法程式，不是在既有專案裡改檔案，所以不等於 coding agent 的表現。"
            "一題可能要幾分鐘到二十分鐘。",
            "**What it measures**: can the model write a correct Python program from a problem "
            "statement.\n\n"
            "**Questions**: the contest problems (AtCoder, LeetCode) LiveCodeBench v6 added for "
            "2025-01 to 04; new problems, so the model is unlikely to have seen them in training. "
            "Listed here are the 60 every model answered in the batch runs (20 easy, 20 medium, "
            "20 hard).\n\n"
            "**How it runs**: LiveCodeBench's own prompt, one answer per question; the model's "
            "default sampling, thinking on, a 32,768-token cap (an answer cut there is wrong). "
            "The last code block of the answer is run by LiveCodeBench's tester on the public and "
            "hidden tests (6 s each): **all must pass**.\n\n"
            "**How to read it**: most models solve the easy ones; they differ on the hard ones "
            "and on thinking that never stops. This is writing one algorithm from scratch, not "
            "editing files in a project, so it is not the same as coding-agent performance. One "
            "question can take from a few minutes to twenty.",
        )
    )

why_not = code_bench.missing()
if why_not:
    st.info(
        tr(
            f"這台伺服器沒有這項測試需要的檔案（{why_not}）。",
            f"This server does not have the files this test needs ({why_not}).",
        ),
        icon=":material/info:",
    )
    st.stop()


@st.cache_data(ttl=600)
def questions() -> list[dict]:
    return code_bench.questions()


@st.cache_data(ttl=600)
def batch() -> dict[str, dict[str, bool]]:
    return code_bench.batch_results()


def preview(q: dict) -> str:
    return f"{DIFFICULTY[q['difficulty']]} · {q['title']} · {q['platform']} {q['date']}"


# ------------------------------------------------------------------ questions
pool = questions()
seen = batch()
level = st.segmented_control(
    tr("難度", "Difficulty"),
    list(DIFFICULTY),
    format_func=lambda d: f"{DIFFICULTY[d]}（{sum(q['difficulty'] == d for q in pool)}）",
    **kept("code_level"),
) or "easy"  # fmt: skip
by_id = {q["id"]: q for q in pool if q["difficulty"] == level}
pick_key = f"code_pick_{level}"
st.session_state.setdefault(pick_key, [])
st.session_state[pick_key] = [i for i in st.session_state[pick_key] if i in by_id][
    :MAX_QUESTIONS
] or list(by_id)[:1]
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
    got = seen.get(i, {})
    with st.expander(q["title"]):
        if got:
            st.caption(
                tr(
                    f"批次測試：{len(got)} 個模型有 {sum(got.values())} 個答對。",
                    f"Batch runs: {sum(got.values())} of {len(got)} models solved it.",
                )
            )
        st.markdown(q["content"])
        if q["starter"]:
            st.code(q["starter"], language="python")

# ------------------------------------------------------------------ models
url = ollama_url()
names = installed_names()
local = [n for n in names if catalog.is_chat(n)]
installed = {catalog.normalize(m) for m in names}
st.session_state.setdefault("code_models", [m for m in [catalog.default_model()] if m in local])
st.session_state.code_models = [m for m in st.session_state.code_models if m in local]
selected = st.multiselect(
    tr(
        f"模型（最多 {MAX_MODELS} 個；只列已下載的地端模型）",
        f"Models (up to {MAX_MODELS}; downloaded local models)",
    ),
    local,
    format_func=lambda m: model_label(m, installed),
    max_selections=MAX_MODELS,
    **kept("code_models"),
)
think = st.toggle(
    tr("思考模式（批次測試的設定）", "Thinking (as in the batch runs)"),
    value=True,
    help=tr(
        "關掉後，會思考的模型改成直接作答"
        "（批次測試裡 Qwen3.6 coding 關思考反而從 29 題進步到 38 題）。"
        "不會思考的模型不受影響。",
        "Off: models that can think answer directly (in the batch runs Qwen3.6 coding went from "
        "29 to 38 correct with thinking off). Models that cannot think are not affected.",
    ),
    **kept("code_think"),
)
memory_guard(selected)
est = code_bench.estimate_minutes([by_id[i]["difficulty"] for i in picked]) * max(len(selected), 1)
st.caption(
    tr(
        f"粗估：約 {est} 分鐘（以批次測試的平均估；思考停不下來的模型一題可能超過 20 分鐘）。",
        f"Rough estimate: about {est} min (batch-run averages; a model whose thinking never "
        "stops can take over 20 min on one question).",
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

        out = code_bench.run(
            r["model"],
            s["ids"],
            think=r["think"],
            run_dir=code_bench.new_run_dir(job.id, r["model"]),
            ollama_url=base_url,
            progress=progress,
            cancel=job.cancel,
        )
        r.update(out, status="done")


def render(job: jobs.Job) -> None:
    s = job.state
    n = len(s["ids"])
    st.markdown("**LiveCodeBench v6** · " + tr(f"{n} 題", f"{n} questions"))
    for r in s["results"]:
        with st.container(border=True):
            rows = r.get("rows") or []
            head = f"**{r['label']}**" + ("" if r["think"] else tr("（關思考）", " (thinking off)"))
            if r["status"] == "done":
                ok = sum(x["ok"] for x in rows)
                head += tr(f"　答對 {ok}/{len(rows)}", f"   {ok}/{len(rows)} correct")
            st.markdown(head)
            if r["status"] == "waiting":
                st.caption(tr("等待中…", "Waiting…"))
                continue
            if r["status"] == "running":
                st.caption(
                    tr(f"執行中…已完成 {r.get('done', 0)} 題", f"Running… {r.get('done', 0)} done")
                )
                continue
            for x in rows:
                q = s["questions"].get(x["id"], {})
                mark = "✅" if x["ok"] else "❌"
                bits = [f"{x['seconds']:.0f} s" if x.get("seconds") else "",
                        f"{x['tokens']} tokens" if x.get("tokens") else ""]  # fmt: skip
                tail = " · ".join(b for b in bits if b)
                with st.expander(
                    f"{mark} {q.get('title', x['id'])}" + (f" · {tail}" if tail else "")
                ):
                    if not x["ok"] and x["why"]:
                        st.error(WHY.get(x["why"], x["why"]), icon=":material/cancel:")
                    before = s["batch"].get(x["id"], {}).get(r["key"])
                    if before is not None:
                        st.caption(
                            tr(
                                "批次測試時這個模型這題：",
                                "This model on this question in the batch run: ",
                            )
                            + (tr("答對", "solved") if before else tr("答錯", "not solved"))
                        )
                    if x["code"]:
                        st.code(x["code"], language="python")
                    elif x["answer"]:
                        st.markdown(tr("**回答的結尾**", "**End of the answer**"))
                        st.text(x["answer"][-1500:])
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


job = jobs.get(st.session_state.get("code_job"))
busy = bool(job and job.running)
if st.button(
    tr("開始測試", "Run"),
    type="primary",
    icon=":material/play_arrow:",
    disabled=busy or not selected or not picked,
):
    modes = {m: ollama_admin.thinking_mode(m, url) for m in selected}
    job = jobs.Job(
        "code",
        tr(f"寫程式測試 {len(picked)} 題", f"Coding test, {len(picked)} question(s)"),
        "views/llm.py",
        state={
            "ids": picked,
            "questions": {i: by_id[i] for i in picked},
            "batch": {i: seen.get(i, {}) for i in picked},
            "started": time.time(),
            "results": [
                {
                    "model": m,
                    "label": m,
                    # a model that cannot think runs the same either way; one that always
                    # thinks cannot be switched off
                    "think": think or modes[m] != "optional",
                    "key": code_bench.key(m, think or modes[m] != "optional"),
                    "status": "waiting",
                }
                for m in selected
            ],
        },
    )
    jobs.start(job, lambda j: work(j, url))
    st.session_state.code_job = job.id
    st.rerun()

if job and job.running:
    live(job)
elif job:
    render(job)
