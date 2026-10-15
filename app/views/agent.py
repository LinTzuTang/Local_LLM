import json
import time
import uuid

import i18n
import jobs
import streamlit as st
import ui
from agent_ui import render_step, was_rejected
from common import (
    kept,
    model_picker,
    ollama_url,
    require_ollama,
    save_run,
    think_control,
    warn_if_no_tools,
)
from i18n import pick, tr
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from local_llm import company_db, skills
from local_llm.agents import (
    AGENTS,
    EXTERNAL_CALL_LIMIT,
    TOOL_INFO,
    Policy,
    all_tools,
    build,
    load_agent_definitions,
    spec_from_dict,
)
from local_llm.config import Settings
from local_llm.lab import store

lang = i18n.current()
ui.page_header(
    "smart_toy",
    tr("Agent 工作室", "Agent studio"),
    tr(
        "在左邊組一個 agent：選模型、寫 system prompt、勾選工具與技能、指定資料庫身分（決定它能看到哪些資料），"
        "右邊直接測試；滿意就儲存，它會出現在「Agents 一覽」、多 Agent 比較與場景裡。",
        "Assemble an agent on the left — model, system prompt, tools, skills and the database "
        "identity that decides what data it can see — and test it on the right. Save it and it "
        "appears in All agents, agent comparison and scenarios.",
    ),
)
require_ollama()

NEW = "__new__"
definitions = {d["id"]: d for d in load_agent_definitions()}
base_key = st.selectbox(
    tr("從哪個 agent 開始", "Start from"),
    [NEW, *definitions],
    format_func=lambda k: (
        tr("+ 新的空白 agent", "+ New blank agent") if k == NEW else AGENTS[k].label(lang)
    ),
    **kept("agent_type"),
)
if base_key not in (NEW, *definitions):
    base_key = NEW
base = definitions.get(base_key) or {
    "id": "my_agent",
    "name": {"zh": "我的 agent", "en": "My agent"},
    "description": {"zh": "", "en": ""},
    "system_prompt": "You are a helpful local assistant. Answer in the language of the user's "
    "message; for Chinese use Traditional Chinese (Taiwan).",
    "tools": [],
    "skills": [],
    "examples": {"zh": [], "en": []},
}


def wkey(field: str) -> str:
    """Widget keys per base agent, so switching agents reloads its settings."""
    return f"studio_{base_key}_{field}"


tool_names = list(all_tools())
identity_list = company_db.identities()
identity_ids = [i["role"] for i in identity_list]
identity_label = {i["role"]: pick(i["label"]) for i in identity_list}
skill_list = skills.list_skills()
skill_names = [s_["name"] for s_ in skill_list]

config_col, chat_col = st.columns([1, 1.45], gap="large")
with config_col.container(border=True):
    st.markdown(f"**{tr('設定', 'Configuration')}**")
    if base.get("model"):
        st.session_state.setdefault("agent_model", base["model"])
    model = model_picker(tr("模型", "Model"), key="agent_model")
    think = think_control(model, key="agent_think")
    system_prompt = st.text_area(
        "System prompt", base["system_prompt"], height=170, key=wkey("prompt")
    )

    def tool_kind(name: str) -> str:
        info = TOOL_INFO.get(name)
        return "network" if info and info.network else ("db" if name.startswith("db_") else "")

    def tool_label(name: str) -> str:
        """Option text (multiselect options cannot show icons): a plain-text kind suffix."""
        kind = tool_kind(name)
        suffix = {"network": tr("連網", "network"), "db": tr("資料庫", "database")}.get(kind)
        return f"{name} · {suffix}" if suffix else name

    def tool_icon(name: str) -> str:
        return {"network": "public", "db": "database"}.get(tool_kind(name), "build")

    chosen_tools = st.multiselect(
        tr("工具", "Tools"),
        tool_names,
        default=[t for t in base.get("tools", []) if t in tool_names],
        format_func=tool_label,
        key=wkey("tools"),
        help=tr(
            ":material/build: 本機　:material/database: 標「資料庫」：公司資料庫（依下方身分的權限）"
            "　:material/public: 標「連網」：會連網，需要允許並核准",
            ":material/build: local · :material/database: tagged “database”: company database "
            "(with the identity below) · :material/public: tagged “network”: goes online, needs "
            "permission and approval",
        ),
    )
    chosen_skills = st.multiselect(
        tr("技能", "Skills"),
        skill_names,
        default=[k for k in base.get("skills", []) if k in skill_names],
        key=wkey("skills"),
        help=tr(
            "技能是可重複使用的工作說明（SKILL.md）；agent 需要時才用 load_skill 讀取。",
            "Skills are reusable instructions (SKILL.md) the agent loads with load_skill only "
            "when needed.",
        ),
    )
    needs_db = any(t.startswith("db_") for t in chosen_tools)
    db_role = None
    if needs_db:
        default_role = base.get("db_role") if base.get("db_role") in identity_ids else None
        db_role = st.selectbox(
            tr("資料庫身分（權限）", "Database identity (permissions)"),
            identity_ids,
            index=identity_ids.index(default_role) if default_role else 0,
            format_func=lambda r: f"{identity_label[r]} · {r}",
            key=wkey("db_role"),
        )
        st.page_link(
            "views/company.py",
            label=tr("看這個身分能讀哪些資料", "See what this identity can read"),
            icon=":material/database:",
        )
    if chosen_tools or chosen_skills:
        st.markdown(
            ui.badges(
                [],
                [f":material/{tool_icon(t)}: {t}" for t in chosen_tools]
                + [f":material/menu_book: {k}" for k in chosen_skills],
            ),
            unsafe_allow_html=True,
        )
    warn_if_no_tools(model, bool(chosen_tools or chosen_skills))
    with st.expander(tr("外部工具控制", "External tool controls")):
        st.caption(
            tr("程式強制，模型無法繞過。", "Enforced in code; the model cannot bypass them.")
        )
        allow_network = st.toggle(tr("允許連網工具", "Allow network tools"), value=False)
        approve = st.toggle(
            tr("每次外部呼叫都要人工核准", "Human approval for every external call"),
            value=True,
            disabled=not allow_network,
        )
        limit = st.number_input(
            tr("每個外部工具每次任務上限", "Max calls per external tool per task"),
            1,
            10,
            EXTERNAL_CALL_LIMIT,
            disabled=not allow_network,
        )
        max_steps = st.slider(tr("最多步數", "Max steps"), 4, 30, 12)
    with st.expander(tr("名稱、說明與範例", "Name, description and examples")):
        n1, n2 = st.columns(2)
        name_zh = n1.text_input("名稱（中文）", base["name"].get("zh", ""), key=wkey("name_zh"))
        name_en = n2.text_input("Name (English)", base["name"].get("en", ""), key=wkey("name_en"))
        desc = base.get("description") or {}
        desc_zh = st.text_input("說明（中文）", desc.get("zh", ""), key=wkey("desc_zh"))
        desc_en = st.text_input("Description (English)", desc.get("en", ""), key=wkey("desc_en"))
        ex = base.get("examples") or {}
        ex_zh = st.text_area(
            tr("範例題目（中文，一行一題）", "Example prompts (Chinese, one per line)"),
            "\n".join(ex.get("zh", [])),
            key=wkey("ex_zh"),
        )
        ex_en = st.text_area(
            tr("範例題目（英文，一行一題）", "Example prompts (English, one per line)"),
            "\n".join(ex.get("en", [])),
            key=wkey("ex_en"),
        )

    draft = {
        **{k: v for k, v in base.items() if k not in ("db_role", "model", "skills")},
        "name": {"zh": name_zh, "en": name_en},
        "description": {"zh": desc_zh, "en": desc_en},
        "system_prompt": system_prompt,
        "tools": chosen_tools,
        "examples": {
            "zh": [x.strip() for x in ex_zh.splitlines() if x.strip()],
            "en": [x.strip() for x in ex_en.splitlines() if x.strip()],
        },
        **({"skills": chosen_skills} if chosen_skills else {}),
        **({"db_role": db_role} if db_role else {}),
        **({"model": model} if model else {}),
    }
    check = store.validate_agent(draft)
    for e in check.errors:
        st.error(e)
    changed = base_key != NEW and {k: v for k, v in draft.items() if k != "model"} != {
        k: v for k, v in base.items() if k != "model"
    }
    s1, s2 = st.columns(2)
    if base_key != NEW and s1.button(
        tr("儲存變更", "Save changes"),
        icon=":material/save:",
        type="primary",
        disabled=not (changed and check.ok),
        use_container_width=True,
    ):
        store.save_agent(draft, source="ui")
        st.toast(tr("已儲存（可在總控頁復原）", "Saved (undo from the control center)"))
        st.rerun()
    with s2.popover(
        tr("另存新 agent", "Save as new"), icon=":material/add:", use_container_width=True
    ):
        new_id = st.text_input(
            tr("新 agent 的 id（英文小寫、底線）", "New agent id (lowercase, underscores)"),
            "my_agent" if base_key == NEW else f"{base_key}_copy",
            key=wkey("new_id"),
        )
        taken = new_id in definitions
        valid_id = bool(new_id) and new_id.replace("_", "").isalnum() and new_id.islower()
        if taken:
            st.warning(tr("這個 id 已經存在。", "This id already exists."))
        if st.button(
            tr("儲存", "Save"),
            type="primary",
            disabled=taken or not valid_id or not check.ok,
            key=wkey("save_new"),
        ):
            store.save_agent({**draft, "id": new_id}, source="ui")
            st.session_state["agent_type"] = new_id
            st.toast(tr(f"已建立 {new_id}", f"Created {new_id}"))
            st.rerun()

key = base_key
spec = spec_from_dict({**draft, "id": base_key if base_key != NEW else "draft"})
policy = Policy(allow_network=allow_network, approve_external=approve, external_limit=int(limit))
clear = chat_col.button(tr("清除對話", "Clear chat"))


def drop_job() -> None:
    """Stop the current task (if any) and forget it."""
    if job := jobs.get(st.session_state.pop("agent_job", None)):
        job.cancel.set()
        if job.status == "paused":
            job.status, job.finished = "cancelled", time.time()


if clear or st.session_state.get("agent_last_type") != key:
    st.session_state.agent_history = []
    st.session_state.pop("agent_last_summary", None)
    drop_job()
    st.session_state.agent_last_type = key

if not model:
    chat_col.info(
        tr(
            "請先選擇已下載的模型，或下載所選模型。",
            "Pick a downloaded model, or download the selected one.",
        )
    )
    st.stop()
if needs_db and (db_error := company_db.status()):
    chat_col.error(
        tr(f"連不到公司資料庫：{db_error}", f"Cannot reach the company database: {db_error}")
    )
    st.stop()
if not check.ok:
    chat_col.warning(tr("請先修正左邊的設定。", "Fix the configuration on the left first."))
    st.stop()

with chat_col:
    history = st.session_state.setdefault("agent_history", [])

    def render_message(msg):
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            if role == "user":
                st.markdown(msg.content)
            else:
                render_step(msg)

    def status_text() -> dict:
        """Status strings for the worker thread, translated here on the page."""
        return {
            "step": tr("第 {n} 步：{label}", "Step {n}: {label}"),
            "calls": tr("模型要求執行 {names}", "model calls {names}"),
            "sep": tr("、", ", "),
            "answers": tr("模型回答", "model answers"),
            "runs": tr("執行 {name}", "running {name}"),
            "tool": tr("工具", "a tool"),
            "paused": tr("等待人工核准", "waiting for approval"),
        }

    def step_label(message, text: dict) -> str:
        if isinstance(message, AIMessage):
            names = text["sep"].join(c["name"] for c in message.tool_calls)
            return text["calls"].format(names=names) if names else text["answers"]
        return text["runs"].format(name=getattr(message, "name", None) or text["tool"])

    def drive(job: jobs.Job, payload) -> None:
        """Worker thread: run the agent until it finishes or pauses for approval (no Streamlit).

        Status strings come from run["text"], translated on the page.
        """
        run, error = job.state, None
        text = run["text"]
        started = time.perf_counter()
        try:
            for update in run["agent"].stream(payload, config=run["config"], stream_mode="updates"):
                for node, value in update.items():
                    if node == "__interrupt__":
                        run["pending"] = value[0].value
                        continue
                    for message in (value or {}).get("messages", []):
                        if message.id and message.id in run["seen"]:
                            continue  # the approval middleware re-emits the message it paused on
                        run["seen"].add(message.id)
                        run["messages"].append(message)
                        job.detail = text["step"].format(
                            n=len(run["messages"]), label=step_label(message, text)
                        )
                if job.cancel.is_set():
                    break
        except Exception as exc:
            error = exc
        run["busy_s"] += time.perf_counter() - started
        if run["pending"] and not error and not job.cancel.is_set():
            job.status, job.detail = "paused", text["paused"]
            return
        run["pending"] = None
        finish(run, stopped=job.cancel.is_set())
        if error:
            raise error

    def finish(run: dict, stopped: bool) -> None:
        """Worker thread: count the task's calls into run["totals"] and save the run."""
        messages, meta = run["messages"], run["meta"]
        llm_calls = sum(isinstance(m, AIMessage) for m in messages)
        tools = [m for m in messages if isinstance(m, ToolMessage) and not was_rejected(m)]
        rejected = sum(isinstance(m, ToolMessage) and was_rejected(m) for m in messages)
        external = sum(bool(TOOL_INFO.get(m.name)) and TOOL_INFO[m.name].network for m in tools)
        run["totals"] = {
            "model": meta["model"],
            "busy_s": run["busy_s"],
            "llm_calls": llm_calls,
            "tool_calls": len(tools),
            "external": external,
            "rejected": rejected,
            "stopped": stopped,
        }
        save_run(
            f"agent-{meta['agent']}",
            {
                **meta,
                "prompt": run["prompt"],
                "seconds": round(run["busy_s"], 2),
                "llm_calls": llm_calls,
                "tool_calls": len(tools),
                "external_calls": external,
                "rejected_calls": rejected,
                "stopped": stopped,
                "decisions": run["decisions"],
                "messages": [json.loads(m.model_dump_json()) for m in messages],
            },
        )

    def plural(n: int, noun: str) -> str:
        return f"{n} {noun}" + ("" if n == 1 else "s")

    def summary_text(totals: dict) -> str:
        """One-line summary of a finished task, in the current language."""
        if "llm_calls" not in totals:
            return ""
        t = totals
        return (
            f"{t['model']} · "
            + tr(
                f"運算 {t['busy_s']:.1f}s（不含等待核准）· 呼叫 LLM API {t['llm_calls']} 次 · "
                f"執行工具 {t['tool_calls']} 次（外部 {t['external']} 次）",
                f"{t['busy_s']:.1f}s compute (excluding approval waits) · "
                f"{plural(t['llm_calls'], 'LLM API call')} · "
                f"{plural(t['tool_calls'], 'tool run')} ({t['external']} external)",
            )
            + (
                tr(f" · 拒絕 {t['rejected']} 次", f" · {t['rejected']} rejected")
                if t["rejected"]
                else ""
            )
            + (
                tr(" · :material/stop_circle: 已停止", " · :material/stop_circle: stopped")
                if t["stopped"]
                else ""
            )
        )

    @st.fragment(run_every=1)
    def live_steps(job: jobs.Job) -> None:
        """Steps of the running task that are not in the history yet."""
        if not job.running:
            st.rerun()
        run = job.state
        for message in run["messages"][run["synced"] :]:
            render_message(message)
        a, b = st.columns([5, 1])
        detail = job.detail or tr("等待模型回應", "waiting for the model")
        a.info(
            tr(
                f"Agent 執行中… {detail} · {job.elapsed():.0f}s（可切換到其他頁面，會在背景繼續）",
                f"Agent running… {detail} · {job.elapsed():.0f}s (you can switch pages; it "
                "continues in the background)",
            ),
            icon=":material/progress_activity:",
        )
        if b.button(tr("停止", "Stop"), key=f"stop_{job.id}"):
            job.cancel.set()

    # Move steps of a finished or paused task into the history (it may have finished while the
    # user was on another page).
    job = jobs.get(st.session_state.get("agent_job"))
    if job and not job.running:
        run = job.state
        history.extend(run["messages"][run["synced"] :])
        run["synced"] = len(run["messages"])
        if job.status != "paused":
            st.session_state.agent_last_summary = {
                **run.get("totals", {}),
                "error": job.error if job.status == "error" else None,
            }
            del st.session_state.agent_job
            job = None

    for msg in history:
        render_message(msg)
    if job and job.running:
        live_steps(job)
    elif history and not job and (last := st.session_state.get("agent_last_summary")):
        summary = summary_text(last)
        if last.get("error"):
            summary = f":orange[:material/error:] {last['error']} · {summary}"
        if summary:
            st.caption(summary)

    # --- Paused: a human decides on the pending external calls --------------------------
    if job and job.status == "paused":
        request = job.state["pending"]
        with st.container(border=True):
            st.warning(
                tr(
                    "Agent 暫停：以下外部呼叫會把資料送出這台電腦，請決定是否執行。",
                    "Agent paused: the external calls below send data off this computer. Decide "
                    "whether to run them.",
                ),
                icon=":material/pause_circle:",
            )
            decision_labels = {"approve": tr("核准", "Approve"), "reject": tr("拒絕", "Reject")}
            with st.form("approval"):
                choices = []
                for i, action in enumerate(request["action_requests"]):
                    runs = TOOL_INFO[action["name"]].text(lang)[0]
                    st.markdown(f":material/public: **`{action['name']}`** — {runs}")
                    args_text = st.text_area(
                        tr(
                            "即將送出的參數（可修改後再核准）",
                            "Arguments to send (edit before approving)",
                        ),
                        json.dumps(action["args"], ensure_ascii=False, indent=2),
                        key=f"args_{i}",
                    )
                    choice = st.radio(
                        tr("決定", "Decision"),
                        list(decision_labels),
                        format_func=decision_labels.get,
                        horizontal=True,
                        key=f"choice_{i}",
                    )
                    reason = st.text_input(
                        tr("拒絕理由（會告訴模型）", "Reason for rejecting (passed to the model)"),
                        key=f"reason_{i}",
                    )
                    choices.append((action, args_text, choice, reason))
                submitted = st.form_submit_button(tr("送出決定", "Submit decision"), type="primary")
        if submitted:
            decisions = []
            for action, args_text, choice, reason in choices:
                if choice == "reject":
                    decisions.append(
                        {
                            "type": "reject",
                            "message": reason
                            or tr("使用者拒絕這次外部呼叫", "The user rejected this external call"),
                        }
                    )
                    continue
                try:
                    args = json.loads(args_text)
                except json.JSONDecodeError:
                    st.error(
                        tr(
                            "參數不是合法的 JSON，請修正後再送出。",
                            "The arguments are not valid JSON; fix them and submit again.",
                        )
                    )
                    st.stop()
                if args == action["args"]:
                    decisions.append({"type": "approve"})
                else:
                    decisions.append(
                        {"type": "edit", "edited_action": {"name": action["name"], "args": args}}
                    )
            job.state["decisions"].append(decisions)
            job.state["pending"] = None
            job.state["text"] = status_text()  # the language may have changed while paused
            jobs.resume(job, lambda j: drive(j, Command(resume={"decisions": decisions})))
            st.rerun()
        st.stop()

    # --- New task -------------------------------------------------------------------------
    spec_examples = spec.example_prompts(lang)
    if spec_examples and not history:
        st.caption(tr("試試看：", "Try:"))
    # A list (not next()) so every example button is rendered on each run.
    clicked = [
        ex
        for i, ex in enumerate(spec_examples)
        if st.button(ex, disabled=job is not None, key=f"agent_ex_{i}")
    ]
    prompt = st.chat_input(
        tr("給 agent 一個任務…", "Give the agent a task…"), disabled=job is not None
    ) or (clicked[0] if clicked else None)

    if prompt and not job:
        config = Settings("ollama", model, ollama_url(), max_tokens=4096, context=16384)
        user_message = HumanMessage(prompt, id=str(uuid.uuid4()))
        history.append(user_message)
        run = {
            "agent": build(spec, config, think=think, policy=policy),
            # A fresh thread per task: the full history is passed in, and the checkpointer only
            # holds the paused state while waiting for approval.
            "config": {
                "configurable": {"thread_id": str(uuid.uuid4())},
                "recursion_limit": max_steps,
            },
            "prompt": prompt,
            "meta": {"agent": key, "model": model, "think": think, "policy": policy.__dict__},
            "messages": [],
            "seen": {m.id for m in history if m.id},
            "synced": 0,
            "decisions": [],
            "busy_s": 0.0,
            "pending": None,
            "text": status_text(),
        }
        payload = {"messages": list(history)}
        job = jobs.start(
            jobs.Job(
                "agent",
                tr(f"{spec.label(lang)}：{prompt[:10]}", f"{spec.label(lang)}: {prompt[:14]}"),
                "views/agent.py",
                state=run,
            ),
            lambda j: drive(j, payload),
        )
        st.session_state.agent_job = job.id
        st.rerun()
