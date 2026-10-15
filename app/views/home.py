import time
from datetime import datetime

import flow_ui
import jobs
import psutil
import streamlit as st
import ui
from common import GPU_BUDGET_GB, ollama_url
from i18n import pick, tr

from local_llm import catalog, company_db, gateway_client, ollama_admin
from local_llm.lab import engine, store

st.markdown(
    """
<style>
.lab-hero {padding: 1.6rem 1.8rem; border-radius: 18px; margin-bottom: .8rem;
  background: radial-gradient(1200px 300px at 0% 0%, #EEF2FF 0%, #F8FAFF 45%, #ECFEFF 100%);
  border: 1px solid #E0E7FF;}
.lab-hero h1 {font-size: 2.5rem; font-weight: 800; margin: 0; padding: 0;
  background: linear-gradient(90deg, #4F46E5, #0891B2); -webkit-background-clip: text;
  background-clip: text; color: transparent;}
.lab-hero p {font-size: 1.08rem; color: #475569; margin: .35rem 0 0 0; max-width: 860px;}
.lab-svc {display:flex; gap:10px; flex-wrap:wrap; margin-bottom: .6rem;}
.lab-svc > span {padding: 4px 12px; border-radius: 999px; background:#F8FAFC;
  border: 1px solid #E2E8F0; font-size: .85rem; font-weight: 600; color:#334155;}
.lab-mem {height: 12px; border-radius: 999px; background: #EEF2F7; overflow: hidden;
  display: flex; margin: .3rem 0 .4rem 0;}
.lab-mem div {height: 100%;}
</style>
""",
    unsafe_allow_html=True,
)
st.markdown(
    f"""<div class="lab-hero"><h1>Local Agent Lab</h1>
<p>{
        tr(
            "在一台 Mac 上測試地端 LLM、展示多 Agent 協作——所有模型與資料都留在這台電腦。",
            "Benchmark local LLMs and demo multi-agent workflows on one Mac — every model and "
            "every byte of data stays on this machine.",
        )
    }</p></div>""",
    unsafe_allow_html=True,
)
links = st.columns(5)
links[0].page_link(
    "views/scenarios.py", label=tr("場景展示", "Scenario gallery"), icon=":material/hub:"
)
links[1].page_link(
    "views/agent.py", label=tr("打造一個 Agent", "Build an agent"), icon=":material/smart_toy:"
)
links[2].page_link(
    "views/compare.py", label=tr("比較模型", "Compare models"), icon=":material/balance:"
)
links[3].page_link(
    "views/speech.py",
    label=tr("比較語音模型", "Compare speech models"),
    icon=":material/graphic_eq:",
)
links[4].page_link(
    "views/models.py", label=tr("瀏覽模型庫", "Browse models"), icon=":material/apps:"
)


# --- live system status --------------------------------------------------------------------------
def model_kind(name: str) -> str:
    caps = set((catalog.find(name) or {}).get("caps") or [])
    for key in ("embedding", "decision", "vision"):
        if key in caps:
            return key
    return "chat"


def gateway_process_rss() -> float:
    for proc in psutil.process_iter(["cmdline", "memory_info"]):
        try:
            cmd = " ".join(proc.info.get("cmdline") or [])
        except (psutil.Error, TypeError):
            continue
        if "local_llm.gateway" in cmd and proc.info.get("memory_info"):
            return proc.info["memory_info"].rss / 1e9
    return 0.0


def expires_in(stamp: str | None) -> str:
    if not stamp:
        return tr("常駐", "kept loaded")
    try:
        left = datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp() - time.time()
    except ValueError:
        return ""
    if left > 3600 * 24 * 30:
        return tr("常駐", "kept loaded")
    return tr(f"{max(left, 0) / 60:.0f} 分鐘後卸載", f"unloads in {max(left, 0) / 60:.0f} min")


JOB_ICONS = {
    "running": ":material/progress_activity:",
    "paused": ":material/pause_circle:",
    "done": ":material/check_circle:",
    "error": ":material/error:",
    "cancelled": ":material/stop_circle:",
}
STATUS = {
    "running": ("執行中", "running"),
    "paused": ("等待核准", "waiting for approval"),
    "done": ("完成", "done"),
    "error": ("失敗", "failed"),
    "cancelled": ("已停止", "stopped"),
}


@st.fragment(run_every=3)
def live_status() -> None:
    url = ollama_url()
    version = ollama_admin.version(url)
    loaded = ollama_admin.loaded(url) if version else []
    try:
        gw_version = gateway_client.version()
        gw_loaded = gateway_client.loaded() if gw_version else []
    except Exception:
        gw_version, gw_loaded = None, []
    db_ok = company_db.status() is None
    active = jobs.active()

    def dot(ok: bool) -> str:
        return f'<span class="lab-dot" style="background:{"#16A34A" if ok else "#DC2626"}"></span>'

    st.markdown(
        '<div class="lab-svc">'
        f"<span>{dot(bool(version))}Ollama {version or tr('未啟動', 'down')}</span>"
        f"<span>{dot(bool(gw_version))}{tr('模型閘道', 'Model gateway')} "
        f"{gw_version or tr('未啟動', 'down')}</span>"
        f"<span>{dot(db_ok)}PostgreSQL</span>"
        f"<span>{ui.icon('pending_actions')} {tr('執行中的工作', 'Running tasks')} "
        f"{len(active)}</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    ollama_gb = sum((m.get("size_vram") or m.get("size", 0)) for m in loaded) / 1e9
    gw_gb = gateway_process_rss() if gw_loaded else 0.0
    mem = psutil.virtual_memory()
    total_gb = mem.total / 1e9
    other_gb = max(mem.used / 1e9 - ollama_gb - gw_gb, 0)
    left, right = st.columns([1.15, 1], gap="large")
    with left.container(border=True):
        st.markdown(f"**{tr('記憶體與載入的模型', 'Memory and loaded models')}**")
        bar = "".join(
            f'<div style="width:{w / total_gb * 100:.1f}%;background:{color}"></div>'
            for w, color in ((ollama_gb, "#4F46E5"), (gw_gb, "#DB2777"), (other_gb, "#CBD5E1"))
        )
        st.markdown(f'<div class="lab-mem">{bar}</div>', unsafe_allow_html=True)
        st.markdown(
            '<span class="lab-kv"><span class="lab-dot" style="background:#4F46E5"></span>'
            f"Ollama {ollama_gb:.1f} GB　"
            '<span class="lab-dot" style="background:#DB2777"></span>'
            f"{tr('模型閘道', 'Gateway')} {gw_gb:.1f} GB　"
            '<span class="lab-dot" style="background:#CBD5E1"></span>'
            f"{tr('其他程式', 'Other apps')} {other_gb:.1f} GB　·　"
            f"{tr('共', 'of')} {total_gb:.0f} GB · "
            f"{tr('模型預算', 'model budget')} {GPU_BUDGET_GB:.0f} GB</span>",
            unsafe_allow_html=True,
        )
        if not loaded and not gw_loaded:
            st.caption(
                tr(
                    "沒有模型在記憶體中；第一次使用時會自動載入。",
                    "No model is loaded; models load on first use.",
                )
            )
        for m in loaded:
            x, y = st.columns([5, 1], vertical_alignment="center")
            gb = (m.get("size_vram") or m.get("size", 0)) / 1e9
            x.markdown(
                f"**{flow_ui.short_model(m['name'])}** · {gb:.1f} GB · "
                f"ctx {m.get('context_length', '?')} · {expires_in(m.get('expires_at'))}"
                + ui.badges([model_kind(m["name"])], ["Ollama"]),
                unsafe_allow_html=True,
            )
            if y.button(tr("卸載", "Unload"), key=f"unload_{m['name']}"):
                ollama_admin.unload(m["name"], url)
                st.rerun(scope="fragment")
        for m in gw_loaded:
            x, y = st.columns([5, 1], vertical_alignment="center")
            kind = m.get("task") if m.get("task") in ui.CAPS else "chat"  # asr, diarize, doc, …
            x.markdown(
                f"**{m['id']}** · {m.get('device', '')} · {expires_in(m.get('expires_at'))}"
                + ui.badges([kind], [tr("模型閘道", "gateway")]),
                unsafe_allow_html=True,
            )
            if y.button(tr("卸載", "Unload"), key=f"gw_unload_{m['id']}"):
                gateway_client.unload(m["id"])
                st.rerun(scope="fragment")
    with right.container(border=True):
        st.markdown(f"**{tr('工作', 'Tasks')}**")
        recent = jobs._recent(600)  # running, paused, or finished in the last 10 minutes
        if not recent:
            st.caption(
                tr(
                    "目前沒有工作。對話、Agent、場景、比較和下載都在背景執行，切換頁面不會中斷。",
                    "Nothing is running. Chats, agents, scenarios, comparisons and downloads run "
                    "in the background and keep going when you switch pages.",
                )
            )
        for job in reversed(recent):
            st.page_link(
                job.page,
                label=f"{job.title} · {tr(*STATUS[job.status])} · {job.elapsed():.0f}s",
                icon=JOB_ICONS.get(job.status, ":material/radio_button_unchecked:"),
            )
            if job.running and job.progress is not None:
                st.progress(min(max(job.progress, 0.0), 1.0), text=job.detail or None)
            elif job.running and job.detail:
                st.caption(job.detail)


st.markdown(f"#### {tr('系統現況', 'Live status')}")
live_status()

# --- scenarios ------------------------------------------------------------------------------------
st.markdown(f"#### {tr('多 Agent 場景', 'Multi-agent scenarios')}")
st.caption(
    tr(
        "點「開啟」到場景展示頁執行；每次執行都會即時畫出流程與時間軸。",
        "Open one to run it on the scenario page; every run draws its flow and timeline live.",
    )
)
scenarios = store.scenarios()


def open_scenario(sid: str) -> None:
    st.session_state.scenario_id = sid
    st.session_state["_open_scenario"] = True


for row in range(0, len(scenarios), 3):
    for col, s in zip(st.columns(3, gap="medium"), scenarios[row : row + 3], strict=False):
        need = engine.requirements(s)
        kinds = sorted({flow_ui.node_kind(n) for n in s["nodes"]} - {None})
        models = [flow_ui.short_model(m) for m in sorted(need["ollama"] | need["gateway"])]
        with col.container(border=True, height=450):
            icon = flow_ui.category_icon(s.get("category"))
            st.markdown(f":material/{icon}: **{pick(s['name'])}**")
            st.graphviz_chart(flow_ui.flow_dot(s, compact=True), use_container_width=True)
            st.markdown(ui.badges(kinds, models[:4]), unsafe_allow_html=True)
            st.caption(pick(s.get("description")))
            st.button(
                tr("開啟", "Open"),
                icon=":material/arrow_forward:",
                key=f"open_{s['id']}",
                on_click=open_scenario,
                args=(s["id"],),
                type="primary",
            )
if st.session_state.pop("_open_scenario", False):
    st.switch_page("views/scenarios.py")

# --- about ----------------------------------------------------------------------------------------
with st.expander(tr("關於這個工具與系統架構", "About this tool and its architecture")):
    st.markdown(
        tr(
            "**兩個目的**：① 測試各種地端 LLM 的表現（模型庫、對話、模型比較、語音與文件）；"
            "② 展示多 Agent 場景（Agent 工作室、多 Agent 比較、場景展示、知識庫、總控 Coding Agent）。"
            "所有服務只監聽 127.0.0.1；權限由資料庫角色與檢索過濾強制，不靠 prompt。",
            "**Two goals**: ① benchmark local LLMs (model library, chat, compare, speech & "
            "documents); ② demo multi-agent scenarios (agent studio, agent comparison, scenario "
            "gallery, knowledge base, coding agent). Every service listens on 127.0.0.1 only; "
            "permissions are enforced by database roles and retrieval filters, not prompts.",
        )
    )
    st.graphviz_chart(flow_ui.architecture_dot(), use_container_width=True)
