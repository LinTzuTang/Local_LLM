"""Control center: the fully local coding agent (settings only, or settings + code).

The same agent as `uv run local-agent code`: it reads and changes agent / scenario settings and,
in code mode, the project's code. Every write pauses here with a diff; the change log on the
right can restore any configuration version or undo any code change from its backup.
"""

import json
import time

import jobs
import streamlit as st
import ui
from agent_ui import render_step
from common import download_control, installed_names, kept, ollama_url, require_ollama, save_run
from i18n import pick, tr
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from local_llm import ollama_admin
from local_llm.coder.agent import (
    DEFAULT_CONTEXT,
    RECOMMENDED_MODEL,
    CoderSession,
    best_coding_model,
    ordered_models,
    run_stats,
    thinking_options,
)
from local_llm.coder.workspace import PROJECT_ROOT, EditError, Workspace
from local_llm.lab import control, store

ui.page_header(
    "🛠️",
    tr("總控 Coding Agent", "Control-center coding agent"),
    tr(
        "完全在本機執行的 coding agent：用自然語言修改 agent 與多 agent 場景的設定，"
        "切到「設定＋程式碼」還能新增工具、修改程式並跑測試。它會先讀取現有內容再提出修改；"
        "每次寫入都會顯示 diff 與驗證結果，經你核准才會存檔，所有版本與程式備份都能一鍵復原。"
        "也可以在 terminal 用 <code>uv run local-agent code</code>。",
        "A coding agent that runs entirely on this Mac: change agent and scenario settings in "
        "plain language, or switch to “settings + code” to add tools, edit code and run tests. "
        "It reads first, then proposes changes; every write shows a diff and validation result "
        "and is saved only after you approve it. Every version and code backup can be restored. "
        "Also available in the terminal: <code>uv run local-agent code</code>.",
    ),
    eyebrow=tr("總控 · 全本機", "Control · fully local"),
)
require_ollama()

MODE_LABELS = {
    "settings": ("⚙️ 只改設定", "⚙️ Settings only"),
    "code": ("💻 設定＋程式碼", "💻 Settings + code"),
}
CONTEXTS = [32768, 65536, 131072]
job = jobs.get(st.session_state.get("ctl_job"))
busy = job is not None


# --- sidebar: scope, model, thinking ------------------------------------------------------------


def tool_models() -> list[str]:
    names = [n for n in installed_names() if ollama_admin.supports_tools(n, ollama_url())]
    return ordered_models(names)


with st.sidebar:
    st.session_state.setdefault("ctl_mode", "code")
    mode = st.segmented_control(
        tr("範圍", "Scope"),
        list(MODE_LABELS),
        format_func=lambda m: tr(*MODE_LABELS[m]),
        disabled=busy,
        required=True,
        **kept("ctl_mode"),
    )
    mode = mode or "code"
    models = tool_models()
    if st.session_state.get("ctl_model") not in models:
        st.session_state.ctl_model = best_coding_model(models) or (models[0] if models else None)
    model = st.selectbox(
        tr("Coding agent 模型", "Coding-agent model"),
        models,
        format_func=lambda m: m + (" ⭐" if m == best_coding_model(models) else ""),
        disabled=busy,
        **kept("ctl_model"),
    )
    if best_coding_model(models) != RECOMMENDED_MODEL:
        st.info(
            tr(
                f"建議的本機 coding 模型是 {RECOMMENDED_MODEL}（工具呼叫最穩、速度快）。",
                f"The recommended local coding model is {RECOMMENDED_MODEL} "
                "(most reliable tool calls, fast).",
            )
        )
        download_control(RECOMMENDED_MODEL, key="ctl_pull", page="views/control.py")
    think_mode, think_default = thinking_options(model, ollama_url()) if model else ("none", 0)
    think = None
    if think_mode == "optional":
        st.session_state.setdefault("ctl_think", think_default)
        think = st.toggle(
            tr("思考模式", "Thinking"),
            disabled=busy,
            help=tr(
                "建議開啟：實測 qwen3.6 coding 關閉思考後，新增工具的任務工具呼叫多了約 3 倍、總時間反而更長。",
                "Recommended on: with thinking off, qwen3.6 coding needed ~3x more tool calls "
                "and took longer on the add-a-tool task.",
            ),
            **kept("ctl_think"),
        )
    elif think_mode == "always":
        st.caption(tr("這個模型一定會先思考。", "This model always thinks first."))
    st.session_state.setdefault("ctl_ctx", DEFAULT_CONTEXT)
    context = st.select_slider(
        tr("上下文長度（tokens）", "Context window (tokens)"),
        CONTEXTS,
        format_func=lambda n: f"{n // 1024}K",
        disabled=busy,
        help=tr(
            "coding agent 要放下檔案內容與工具結果，建議至少 64K。",
            "A coding agent needs room for files and tool results; 64K or more is recommended.",
        ),
        **kept("ctl_ctx"),
    )
    if st.button(tr("清除對話", "Clear conversation"), disabled=busy and job.running):
        st.session_state.pop("ctl_session", None)
        st.session_state.ctl_history = []
        if old := jobs.get(st.session_state.pop("ctl_job", None)):
            old.cancel.set()
            if old.status == "paused":
                old.status, old.finished = "cancelled", time.time()
        st.rerun()
    st.caption(
        tr(
            f"專案：`{PROJECT_ROOT.name}` · 模型、程式與資料都不離開這台電腦。"
            "不能讀寫 .env、models/、storage/pgdata、.venv。",
            f"Project: `{PROJECT_ROOT.name}` · models, code and data never leave this Mac. "
            ".env, models/, storage/pgdata and .venv are off limits.",
        )
    )

EXAMPLES = {
    "settings": [
        tr(
            "把「混合檢索 RAG」的回答模型改成 qwen3:1.7b",
            "Switch the answer model of the hybrid RAG scenario to qwen3:1.7b",
        ),
        tr(
            "在「混合檢索 RAG」回答之後加一個把答案翻成英文的節點，最後輸出英文版",
            "In the hybrid RAG scenario, add a node after the answer that translates it to "
            "English and make that the output",
        ),
        tr(
            "公司問答路由的 general 分支改成交給「翻譯」agent",
            "In the company router, send the general branch to the translator agent",
        ),
        tr(
            "新增一個「採購詢價助理」agent，只能用計算機和時間工具，幫忙試算報價",
            "Create a 'purchasing assistant' agent that may only use the calculator and clock tools",
        ),
    ],
    "code": [
        tr(
            "新增 unit_convert 工具（公里/英里、公斤/磅、攝氏/華氏），註冊中英文說明，加到「工具」agent，並補上測試",
            "Add a unit_convert tool (km/mi, kg/lb, °C/°F) with zh/en info, give it to the tools "
            "agent and add a test",
        ),
        tr(
            "找出 calculator 工具的實作，說明它怎麼避免執行任意程式碼",
            "Find the calculator tool and explain how it avoids running arbitrary code",
        ),
        tr(
            "執行 ruff 和 pytest，如果有錯誤就修好",
            "Run ruff and pytest and fix whatever fails",
        ),
        tr(
            "把「混合檢索 RAG」的回答模型改成 qwen3:1.7b",
            "Switch the answer model of the hybrid RAG scenario to qwen3:1.7b",
        ),
    ],
}

left, right = st.columns([2.2, 1], gap="large")
history = st.session_state.setdefault("ctl_history", [])
ws = Workspace(PROJECT_ROOT)


def get_session() -> CoderSession | None:
    """The conversation's agent; rebuilt (history kept) when model, scope or settings change."""
    if not model:
        return None
    wanted = (model, mode, think, context)
    session = st.session_state.get("ctl_session")
    if session is None:
        session = CoderSession(model, ollama_url(), mode=mode, context=context, think=think)
    elif st.session_state.get("ctl_session_key") != wanted and not busy:
        session.think, session.context = think, context
        session.rebuild(model=model, mode=mode)
    st.session_state.ctl_session, st.session_state.ctl_session_key = session, wanted
    return session


def render_message(msg) -> None:
    with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant"):
        if isinstance(msg, HumanMessage):
            st.markdown(msg.content)
        else:
            render_step(msg)


def drive(job: jobs.Job, payload) -> None:
    """Worker thread: run until the agent answers or pauses for approval (no Streamlit)."""
    run = job.state

    def on_event(kind: str, data) -> None:
        if kind == "message":
            run["messages"].append(data)
            run["live"] = ""
            if isinstance(data, AIMessage) and data.tool_calls:
                job.detail = ", ".join(c["name"] for c in data.tool_calls)
        else:  # streamed thinking / answer text
            run["live"] = (run["live"] + data)[-1200:]

    run["pending"] = run["session"].run(payload, on_event, job.cancel)
    if run["pending"] and not job.cancel.is_set():
        job.status, job.detail = "paused", "等待核准 / approval"  # worker: no tr()
        return
    run["pending"] = None
    save_run(
        "control",
        {
            "model": run["model"],
            "mode": run["mode"],
            "prompt": run["prompt"],
            "decisions": run["decisions"],
            "stats": run_stats(run["messages"]),
            "seconds": round(time.time() - job.started, 1),
            "messages": [json.loads(m.model_dump_json()) for m in run["messages"]],
        },
    )


@st.fragment(run_every=1)
def live(job: jobs.Job) -> None:
    if not job.running:
        st.rerun()
    run = job.state
    for message in run["messages"][run["synced"] :]:
        render_message(message)
    if run.get("live"):
        st.caption("💭 " + run["live"][-600:])
    a, b = st.columns([5, 1])
    a.info(
        f"⏳ {tr('Coding agent 執行中', 'Coding agent working')}… {job.detail} · "
        f"{job.elapsed():.0f}s"
    )
    if b.button(tr("停止", "Stop"), key=f"stop_{job.id}"):
        job.cancel.set()


def stats_text(stats: dict, seconds: float) -> str:
    speed = f" · {stats['tokens_per_s']} tokens/s" if stats["tokens_per_s"] else ""
    return tr(
        f"⏱ {seconds:.0f} 秒 · {stats['llm_calls']} 次模型呼叫 · {stats['tool_calls']} 次工具呼叫 · "
        f"輸出 {stats['output_tokens']} tokens{speed}",
        f"⏱ {seconds:.0f}s · {stats['llm_calls']} model calls · {stats['tool_calls']} tool calls · "
        f"{stats['output_tokens']} tokens out{speed}",
    )


if job and not job.running:
    run = job.state
    history.extend(run["messages"][run["synced"] :])
    run["synced"] = len(run["messages"])
    if job.status != "paused":
        if job.error:
            st.session_state.ctl_error = job.error
        st.session_state.ctl_stats = (run_stats(run["messages"]), job.elapsed())
        del st.session_state.ctl_job
        job = None


def approval_card(job: jobs.Job) -> None:
    request = job.state["pending"]
    with st.container(border=True):
        st.warning(
            tr(
                "⏸️ Coding agent 想修改以下內容，請檢查 diff 後決定。",
                "⏸️ The coding agent wants to make these changes. Review the diffs.",
            )
        )
        choices = []
        for i, (action, preview) in enumerate(
            zip(request["action_requests"], request["previews"], strict=True)
        ):
            if preview["kind"] == "code":
                verb = tr("新增檔案", "new file") if preview.get("action") == "create" else ""
                st.markdown(
                    f"**✏️ `{action['name']}`** · `{preview['id']}` {verb} "
                    f"<span style='color:#15803D'>+{preview.get('added', 0)}</span> "
                    f"<span style='color:#B91C1C'>−{preview.get('removed', 0)}</span>",
                    unsafe_allow_html=True,
                )
            else:
                target = tr("場景", "scenario") if preview["kind"] == "scenario" else "agent"
                st.markdown(f"**⚙️ `{action['name']}`** · {target} `{preview['id']}`")
            st.code(preview["diff"] or "(no change)", "diff")
            for e in preview["errors"]:
                st.error(e)
            for w in preview["warnings"]:
                st.warning(w)
            if preview["ok"] and preview["kind"] != "code":
                st.success(tr("驗證通過", "Validation passed"))
            choice = st.radio(
                tr("決定", "Decision"),
                ["approve", "reject"],
                index=0 if preview["ok"] else 1,
                format_func=lambda c: {
                    "approve": tr("核准", "Approve"),
                    "reject": tr("拒絕", "Reject"),
                }[c],
                horizontal=True,
                key=f"ctl_choice_{job.id}_{len(job.state['decisions'])}_{i}",
            )
            reason = st.text_input(
                tr(
                    "拒絕理由／改成怎麼做（會告訴 agent）",
                    "Reason / what to do instead (sent to the agent)",
                ),
                key=f"ctl_reason_{job.id}_{len(job.state['decisions'])}_{i}",
            )
            choices.append((choice, reason))
        a, b = st.columns(2)
        submit = a.button(tr("送出決定", "Submit"), type="primary", use_container_width=True)
        approve_all = b.button(tr("全部核准", "Approve all"), use_container_width=True)
        if submit or approve_all:
            decisions = [
                {"type": "approve"}
                if approve_all or c == "approve"
                else {
                    "type": "reject",
                    "message": "The user rejected this change"
                    + (f": {r}" if r else ". Ask what to do instead."),
                }
                for c, r in choices
            ]
            job.state["decisions"].append(decisions)
            job.state["pending"] = None
            jobs.resume(job, lambda j: drive(j, Command(resume={"decisions": decisions})))
            st.rerun()


with left:
    if not history and not job:
        st.markdown(tr("**可以這樣說：**", "**Try asking:**"))
    for msg in history:
        render_message(msg)
    if err := st.session_state.pop("ctl_error", None):
        st.error(err)
    if not job and (last := st.session_state.get("ctl_stats")):
        st.caption(stats_text(*last))
    if job and job.running:
        live(job)
    if job and job.status == "paused":
        approval_card(job)

    # --- new instruction ------------------------------------------------------------------------
    clicked = []
    if not job:
        cols = st.columns(2)
        clicked = [
            ex for i, ex in enumerate(EXAMPLES[mode]) if cols[i % 2].button(ex, key=f"ctl_ex_{i}")
        ]
    prompt = st.chat_input(
        tr("告訴 coding agent 要改什麼…", "Tell the coding agent what to change…"),
        disabled=job is not None or not model,
    ) or (clicked[0] if clicked else None)
    if prompt and not job and (session := get_session()):
        st.session_state.pop("ctl_stats", None)
        payload = session.start(prompt)
        history.append(payload["messages"][-1])
        run = {
            "session": session,
            "model": model,
            "mode": mode,
            "prompt": prompt,
            "messages": [],
            "synced": 0,
            "decisions": [],
            "pending": None,
            "live": "",
        }
        job = jobs.start(
            jobs.Job(
                "control", tr("總控：", "Control: ") + prompt[:12], "views/control.py", state=run
            ),
            lambda j: drive(j, payload),
        )
        st.session_state.ctl_job = job.id
        st.rerun()
    if not models:
        st.error(
            tr(
                f"沒有支援工具呼叫的模型。請下載 {RECOMMENDED_MODEL}。",
                f"No tool-calling model installed. Download {RECOMMENDED_MODEL}.",
            )
        )

# --- change log: configuration versions and code backups -----------------------------------------
with right:
    st.markdown(f"#### {tr('變更紀錄', 'Change log')}")
    if ws.code_changed_since():
        st.warning(
            tr(
                "這次啟動後程式碼有修改：執行 `./scripts/restart_app.sh` 重新啟動平台才會載入。"
                "在那之前，設定的驗證會在新的 Python 行程裡用新程式碼進行。",
                "Code changed since the platform started: run `./scripts/restart_app.sh` to load "
                "it. Until then, settings are validated in a fresh Python process with the new code.",
            )
        )
    config_tab, code_tab = st.tabs([tr("⚙️ 設定版本", "⚙️ Settings"), tr("💻 程式備份", "💻 Code")])
    with config_tab:
        st.caption(
            tr("所有設定修改（UI、總控、復原）都會留版本。", "Every settings change is versioned.")
        )
        names = {s["id"]: pick(s["name"]) for s in store.scenarios()}
        names |= {a["id"]: pick(a["name"]) for a in store.agents()}
        entries = control.history(limit=30)
        if not entries:
            st.info(tr("還沒有修改。", "No changes yet."))
        for entry in entries:
            icon = "🕸️" if entry["kind"] == "scenario" else "🤖"
            source = {
                "control-center": tr("總控", "agent"),
                "ui": "UI",
                "rollback": tr("復原", "undo"),
            }.get(entry["source"], entry["source"])
            title = f"{icon} {names.get(entry['id'], entry['id'])} · {source} · {entry['at'][5:16]}"
            with st.expander(title):
                st.code(store.diff(entry["before"], entry["after"]) or "(no change)", "diff")
                if st.button(
                    tr("復原到這次修改之前", "Restore the version before"), key=entry["file"]
                ):
                    result = control.rollback(entry["file"], fresh=ws.code_changed_since())
                    if result["ok"]:
                        st.toast(tr("已復原", "Restored"))
                        st.rerun()
                    for e in result["errors"]:
                        st.error(e)
        st.page_link(
            "views/scenarios.py", label=tr("到場景頁執行 →", "Run it on the Scenarios page →")
        )
    with code_tab:
        st.caption(
            tr(
                "每次程式修改前都會備份到 `storage/coder/backups/`，可以逐筆復原（由新到舊）。",
                "Every code change is backed up in `storage/coder/backups/` and can be undone "
                "(newest first).",
            )
        )
        changes = ws.changes(30)
        if not changes:
            st.info(tr("還沒有程式修改。", "No code changes yet."))
        for entry in changes:
            icon = "🆕" if entry["action"] == "create" else "✏️"
            undone = tr(" · 已復原", " · undone") if entry.get("undone_at") else ""
            title = (
                f"{icon} {entry['path']} · +{entry['added']} −{entry['removed']} · "
                f"{entry['at'][5:16]}{undone}"
            )
            with st.expander(title):
                st.code(ws.entry_change(entry["id"]).diff() or "(no change)", "diff")
                if not entry.get("undone_at") and st.button(
                    tr("復原這次修改", "Undo this change"), key=f"undo_{entry['id']}"
                ):
                    try:
                        ws.undo(entry["id"])
                        st.toast(tr("已復原", "Undone"))
                        st.rerun()
                    except EditError as exc:
                        st.error(str(exc))
    st.caption(
        f"`{store.SCENARIOS_FILE.relative_to(store.ROOT)}` · `agents.json` · `storage/coder/`"
    )
