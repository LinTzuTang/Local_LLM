import json
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import i18n
import jobs
import streamlit as st
import ui
from agent_ui import render_step
from common import kept, model_picker, ollama_url, require_ollama, save_run, warn_if_no_tools
from i18n import tr
from langchain_core.messages import AIMessage, ToolMessage

from local_llm import catalog, company_db
from local_llm.agents import AGENTS, build
from local_llm.config import Settings

ui.page_header(
    "compare",
    tr("多 Agent 比較", "Compare agents"),
    tr(
        "同一個問題同時問你選的 agent（最多 4 個），並排比較答案、權限、步驟與時間。"
        "預設是三個部門 agent：同一個模型、同樣的工具，只差資料庫身分，最能看出權限如何影響答案。",
        "Ask up to four agents the same question at once and compare answers, permissions, steps "
        "and time side by side. The default is the three department agents — same model and "
        "tools, different database identity — which shows how permissions shape the answer.",
    ),
)
require_ollama()

lang = i18n.current()
st.session_state.setdefault("compare_agents", ["company_sales", "company_hr", "company_support"])
st.session_state.compare_agents = [k for k in st.session_state.compare_agents if k in AGENTS]
selected = st.multiselect(
    tr("要比較的 agent", "Agents to compare"),
    list(AGENTS),
    format_func=lambda k: AGENTS[k].label(lang),
    max_selections=4,
    **kept("compare_agents"),
)
DEPARTMENTS = selected or ["company_sales"]
if any(AGENTS[k].db_role for k in DEPARTMENTS) and (db_error := company_db.status()):
    st.error(tr(f"連不到公司資料庫：{db_error}", f"Cannot reach the company database: {db_error}"))
    st.stop()
EXAMPLES = {
    "zh": [
        "2026 年營收最高的前五個城市是哪些？",
        "各部門的在職人數與平均月薪",
        "目前還沒解決的緊急工單有幾張？",
        "伺服馬達系列的成本與牌價",
        "業務部有哪些人？列出姓名與職稱",
    ],
    "en": [
        "Which five cities had the highest revenue in 2026?",
        "Headcount and average monthly salary per department",
        "How many urgent tickets are still unresolved?",
        "Cost and list price of the servo motor series",
        "Who works in Sales? List names and job titles",
    ],
}
# Outcome of one department's run; the run files keep the id, the page shows the label.
VERDICTS = {
    "error": ":orange[:material/error:] " + tr("執行失敗", "Failed"),
    "stopped": ":gray[:material/stop_circle:] " + tr("已停止", "Stopped"),
    "denied": ":red[:material/lock:] " + tr("權限不足", "Permission denied"),
    "answered": ":green[:material/check_circle:] " + tr("有回答", "Answered"),
}

# An English answer saying the role may not see the data (Chinese answers mention 權限).
REFUSAL_EN = re.compile(
    r"permission|not (allowed|authori[sz]ed|permitted)|(cannot|can't|unable to) access|no access",
    re.IGNORECASE,
)

c1, c2, c3 = st.columns([1.3, 2, 1])
MODES = {
    "shared": tr("全部用同一個模型", "One model for all"),
    "own": tr("各自的預設模型", "Each agent's own model"),
}
mode = c1.segmented_control(
    tr("模型", "Model"), list(MODES), format_func=MODES.get, default="shared"
)
with c2:
    model = (
        model_picker(tr("共用模型", "Shared model"), key="dept_model") if mode == "shared" else None
    )
    if mode == "shared":
        warn_if_no_tools(model, needs_tools=any(AGENTS[k].tools for k in DEPARTMENTS))
    else:
        st.caption(
            "、".join(
                f"{AGENTS[k].label(lang)}：`{AGENTS[k].model or catalog.default_model()}`"
                for k in DEPARTMENTS
            )
        )
max_steps = c3.slider(tr("最多步數", "Max steps"), 4, 30, 14)
if mode == "shared" and not model:
    st.info(tr("請先選擇已下載的模型。", "Pick a downloaded model first."))
    st.stop()

job = jobs.get(st.session_state.get("agent_compare_job"))
busy = bool(job and job.running)
examples = list(
    dict.fromkeys(
        EXAMPLES[lang] + [x for k in DEPARTMENTS for x in AGENTS[k].example_prompts(lang)[:1]]
    )
)
st.caption(tr("試試看：", "Try:"))
clicked = [ex for i, ex in enumerate(examples) if st.button(ex, disabled=busy, key=f"cmp_ex_{i}")]
question = st.chat_input(
    tr("輸入要問這些 agent 的問題…", "Ask these agents a question…"), disabled=busy
) or (clicked[0] if clicked else None)


def department_work(job: jobs.Job, base_url: str, max_steps: int, progress: str) -> None:
    """Worker thread: runs the three agents at the same time (no Streamlit calls).

    `progress` is the status line, translated on the page, with {done} and {total} fields.
    """
    state = job.state

    def run_one(key: str) -> None:
        dept = state["depts"][key]
        config = Settings("ollama", dept["model"], base_url, max_tokens=4096, context=16384)
        agent = build(AGENTS[key], config)
        started = time.perf_counter()
        try:
            for update in agent.stream(
                {"messages": [{"role": "user", "content": state["question"]}]},
                config={
                    "recursion_limit": max_steps,
                    "configurable": {"thread_id": str(uuid.uuid4())},
                },
                stream_mode="updates",
            ):
                for value in update.values():
                    dept["messages"].extend((value or {}).get("messages", []))
                if job.cancel.is_set():
                    dept["stopped"] = True
                    break
        except Exception as exc:
            dept["error"] = f"{type(exc).__name__}: {exc}"
        dept["seconds"] = time.perf_counter() - started
        dept["done"] = True
        finished = sum(d["done"] for d in state["depts"].values())
        job.detail = progress.format(done=finished, total=len(state["depts"]))

    started = time.perf_counter()
    keys = list(state["depts"])
    with ThreadPoolExecutor(max_workers=len(keys)) as pool:
        list(pool.map(run_one, keys))
    state["wall_seconds"] = time.perf_counter() - started
    save_run(
        "agent_compare",
        {
            "model": state["model"],
            "agents": keys,
            "prompt": state["question"],
            "wall_seconds": round(state["wall_seconds"], 2),
            "results": {
                key: {
                    "model": r["model"],
                    "verdict": verdict(r),
                    "seconds": round(r["seconds"], 2),
                    "error": r["error"],
                    "stopped": r.get("stopped", False),
                    "messages": [json.loads(m.model_dump_json()) for m in r["messages"]],
                }
                for key, r in state["depts"].items()
            },
        },
    )


def verdict(result: dict) -> str:
    """One of the VERDICTS ids (language-neutral, so the worker thread can call it)."""
    tool_text = " ".join(str(m.content) for m in result["messages"] if isinstance(m, ToolMessage))
    answer = str(result["messages"][-1].content) if result["messages"] else ""
    if result["error"]:
        return "error"
    if result.get("stopped"):
        return "stopped"
    refused = "權限" in answer or REFUSAL_EN.search(answer)
    if "Permission denied" in tool_text or ("PERMISSION NOTICE" in tool_text and refused):
        # Either the database refused a query, or the agent saw hidden tables/columns and
        # (correctly) answered that it lacks permission.
        return "denied"
    return "answered"


def last_step(messages: list) -> str:
    if not messages:
        return tr("等待模型回應（思考中）", "Waiting for the model (thinking)")
    m = messages[-1]
    if isinstance(m, AIMessage) and m.tool_calls:
        names = [f"`{c['name']}`" for c in m.tool_calls]
        return tr("要求執行 " + "、".join(names), "Calling " + ", ".join(names))
    if isinstance(m, ToolMessage):
        return tr(
            f"`{m.name}` 已執行，等待模型讀取結果",
            f"`{m.name}` ran; waiting for the model to read the result",
        )
    return tr("整理回答中", "Writing the answer")


def show(result: dict) -> None:
    messages = result["messages"]
    llm = sum(isinstance(m, AIMessage) for m in messages)
    tools = sum(isinstance(m, ToolMessage) for m in messages)
    st.markdown(f"**{VERDICTS[verdict(result)]}**")
    st.caption(
        f"{result['seconds']:.1f}s · "
        + tr(
            f"呼叫 LLM {llm} 次 · 執行工具 {tools} 次",
            f"{llm} LLM call{'' if llm == 1 else 's'} · {tools} tool run{'' if tools == 1 else 's'}",
        )
    )
    if result["error"]:
        st.error(result["error"])
    elif messages and not result.get("stopped"):
        st.markdown(messages[-1].content)
    with st.expander(tr("查看每一步", "Every step")):
        for message in messages:
            render_step(message)


def render(job: jobs.Job) -> None:
    state = job.state
    st.markdown(f"#### :material/help: {state['question']}")
    keys = list(state["depts"])
    for col, key in zip(st.columns(len(keys)), keys, strict=True):
        with col.container(border=True):
            label = AGENTS[key].label(i18n.current()) if key in AGENTS else key
            st.markdown(f"**{label}**")
            dept = state["depts"][key]
            role = AGENTS[key].db_role if key in AGENTS else None
            st.markdown(
                ui.badges([], [dept["model"]] + ([f":material/database: {role}"] if role else [])),
                unsafe_allow_html=True,
            )
            if dept["done"]:
                show(dept)
            else:
                steps = sum(isinstance(m, AIMessage) for m in dept["messages"])
                st.info(
                    tr(
                        f"執行中 {job.elapsed():.0f}s · 已呼叫 LLM {steps} 次",
                        f"Running {job.elapsed():.0f}s · {steps} LLM call{'' if steps == 1 else 's'} so far",
                    )
                    + f"  \n{last_step(dept['messages'])}",
                    icon=":material/progress_activity:",
                )
    if job.running:
        a, b = st.columns([5, 1])
        a.caption(
            tr(
                "這些 agent 同時執行中；可以切換到其他頁面，回來再看結果。",
                "The agents are running in parallel; you can switch pages and come back for the "
                "results.",
            )
        )
        if b.button(tr("停止", "Stop"), key=f"stop_{job.id}"):
            job.cancel.set()
    elif "wall_seconds" in state:
        wall = state["wall_seconds"]
        total = sum(d["seconds"] for d in state["depts"].values())
        st.caption(
            tr(
                f"同時執行，總共 {wall:.1f}s（個別加總 {total:.1f}s）",
                f"Ran in parallel: {wall:.1f}s in total ({total:.1f}s if added up)",
            )
        )


@st.fragment(run_every=1)
def live(job: jobs.Job) -> None:
    if not job.running:
        st.rerun()
    render(job)


if question and not busy:
    job = jobs.Job(
        "agent_compare",
        tr(f"Agent 比較：{question[:10]}", f"Agents: {question[:14]}"),
        "views/agent_compare.py",
        state={
            "model": model or "own",
            "question": question,
            "depts": {
                k: {
                    "messages": [],
                    "done": False,
                    "error": None,
                    "stopped": False,
                    "seconds": 0.0,
                    "model": model or AGENTS[k].model or catalog.default_model(),
                }
                for k in DEPARTMENTS
            },
        },
    )
    url, steps = ollama_url(), max_steps
    progress = tr("{done}/{total} 個 agent 完成", "{done}/{total} agents done")
    jobs.start(job, lambda j: department_work(j, url, steps, progress))
    st.session_state.agent_compare_job = job.id
    st.rerun()

if job and job.running:
    live(job)
elif job:
    render(job)
