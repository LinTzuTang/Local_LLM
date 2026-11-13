import html
import os
import time
from datetime import datetime
from pathlib import Path

import flow_ui
import jobs
import lottie
import model_news
import psutil
import streamlit as st
import ui
import voice_assistant
from common import GPU_BUDGET_GB, ollama_url
from i18n import pick, tr

from local_llm import catalog, company_db, gateway_client, machine, ollama_admin
from local_llm.lab import engine, store

st.markdown(
    """
<style>
/* hero: an ink panel with a slow aurora, a grid that fades out, and this machine on the right */
.st-key-lab_hero {position: relative; overflow: hidden; isolation: isolate;
  padding: 2.2rem 2.4rem 3.6rem; border-radius: 22px; margin: .2rem 0 0 0;
  color: #E2E8F2; background-color: var(--lab-ink);
  background-image: linear-gradient(rgba(255,255,255,.035) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,.035) 1px, transparent 1px);
  background-size: 30px 30px;
  box-shadow: 0 24px 50px -30px rgba(14, 26, 28, .6), inset 0 0 0 1px rgba(255,255,255,.06);}
.st-key-lab_hero::before {content: ""; position: absolute; inset: -40%; z-index: -2;
  background:
    radial-gradient(closest-side at 30% 40%, rgba(110, 150, 220, .18), transparent 70%),
    radial-gradient(closest-side at 72% 30%, rgba(150, 140, 215, .13), transparent 70%),
    radial-gradient(closest-side at 60% 75%, rgba(110, 190, 215, .09), transparent 70%);
  filter: blur(10px); animation: lab-drift 22s ease-in-out infinite alternate;}
.st-key-lab_hero::after {content: ""; position: absolute; inset: 0; z-index: -1;
  border-radius: inherit; padding: 1px; pointer-events: none;
  background: conic-gradient(from var(--lab-angle), transparent 0 70%,
    rgba(157, 187, 232, .55) 80%, rgba(189, 179, 232, .5) 86%, transparent 92%);
  -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
  -webkit-mask-composite: xor; mask-composite: exclude; animation: lab-spin 7s linear infinite;}
@keyframes lab-drift {0% {transform: translate3d(-4%, -2%, 0) rotate(0deg);}
  100% {transform: translate3d(5%, 3%, 0) rotate(8deg);}}
.st-key-lab_hero .lab-eyebrow {color: var(--lab-signal);}
.st-key-lab_hero h1 a, .st-key-lab_hero [data-testid="stHeaderActionElements"] {display: none;}
.st-key-lab_hero h1 {font-size: 3.2rem; font-weight: 820; letter-spacing: -0.04em; margin: .4rem 0 0;
  padding: 0; line-height: 1.02; color: #F6F5F1;}
.st-key-lab_hero h1 em {font-style: normal;
  background: linear-gradient(100deg, #A9C4EC 0%, #BDB3E8 55%, #9ED0E4 100%);
  -webkit-background-clip: text; background-clip: text; color: transparent;
  filter: drop-shadow(0 0 12px rgba(110, 150, 220, .25));}
.st-key-lab_hero p {font-size: 1.05rem; color: #A9B4C8; margin: .8rem 0 0; max-width: 600px;}
/* this machine */
.lab-machine {position: relative; border-radius: 14px; padding: 1rem 1.1rem 1.05rem;
  background: linear-gradient(160deg, rgba(255,255,255,.08), rgba(255,255,255,.025));
  border: 1px solid rgba(255,255,255,.12); backdrop-filter: blur(6px);
  box-shadow: inset 0 1px 0 rgba(255,255,255,.08);}
.lab-machine .hd {display: flex; align-items: center; gap: 8px; font-family: var(--lab-mono);
  font-size: .7rem; letter-spacing: .12em; text-transform: uppercase; color: #8F9CB3;}
.lab-machine .kind {margin-left: auto; padding: 2px 9px; border-radius: 999px; font-weight: 700;
  letter-spacing: .06em; color: #0E1A1C; background: var(--kind, #2DD4BF);
  box-shadow: none;}
.lab-machine .live {width: 8px; height: 8px; border-radius: 50%; background: #6FCB9A;
  box-shadow: 0 0 0 0 rgba(111, 203, 154, .5); animation: lab-pulse 2.2s ease-out infinite;}
@keyframes lab-pulse {0% {box-shadow: 0 0 0 0 rgba(111, 203, 154, .45);}
  100% {box-shadow: 0 0 0 10px rgba(111, 203, 154, 0);}}
.lab-machine .ttl {font-size: 1.35rem; font-weight: 760; color: #F6F5F1; margin: .45rem 0 .55rem;
  letter-spacing: -0.01em;}
.lab-machine dl {display: grid; grid-template-columns: auto minmax(0, 1fr); gap: 5px 14px;
  margin: 0;}
.lab-machine dt {font-size: .76rem; color: #8F9CB3; white-space: nowrap;}
.lab-machine dd {margin: 0; font-family: var(--lab-mono); font-size: .8rem; color: #E6ECF5;
  overflow-wrap: anywhere;}
@media (max-width: 900px) {.st-key-lab_hero {padding: 1.5rem;}
  .st-key-lab_hero h1 {font-size: 2.3rem;}}
@media (prefers-reduced-motion: reduce) {.st-key-lab_hero::before, .lab-machine .live {animation: none;}}
/* shortcuts: cards that float over the hero's lower edge */
.st-key-lab_shortcuts {position: relative; z-index: 2; margin: -1.9rem 0 1.2rem; padding: 0 1.1rem;}
.st-key-lab_shortcuts [data-testid="stPageLink"] a {position: relative; flex-direction: column;
  align-items: flex-start; gap: .7rem; border: 1px solid var(--lab-border);
  background: var(--lab-surface); border-radius: 14px; padding: .95rem 1rem .9rem;
  min-height: 6.2rem; box-shadow: 0 12px 28px -20px rgba(10, 20, 22, .55);
  transition: border-color .15s, transform .15s, box-shadow .15s;}
.st-key-lab_shortcuts [data-testid="stPageLink"] a:hover {border-color: var(--lab-primary);
  transform: translateY(-3px); background: var(--lab-surface);
  box-shadow: 0 18px 34px -20px rgba(10, 20, 22, .6);}
.st-key-lab_shortcuts [data-testid="stPageLink"] a span {color: var(--lab-heading); font-weight: 650;
  font-size: .95rem; line-height: 1.3;}
.st-key-lab_shortcuts [data-testid="stPageLink"] a [data-testid="stIconMaterial"] {
  color: var(--lab-primary); background: var(--lab-tint); width: 34px; height: 34px; border-radius: 9px;
  display: inline-flex; align-items: center; justify-content: center; font-size: 20px;}
.st-key-lab_shortcuts [data-testid="stPageLink"] a::after {content: "arrow_forward";
  font-family: "Material Symbols Rounded"; font-size: 18px; position: absolute; right: .9rem;
  bottom: .85rem; color: var(--lab-primary); opacity: 0; transform: translateX(-6px);
  transition: opacity .15s, transform .15s;}
.st-key-lab_shortcuts [data-testid="stPageLink"] a:hover::after {opacity: 1; transform: none;}
/* service readouts */
.lab-svc {display:flex; gap:8px; flex-wrap:wrap; margin-bottom: .7rem;}
.lab-svc > span {padding: 5px 12px; border-radius: 8px; background: var(--lab-surface);
  border: 1px solid var(--lab-border); font-family: var(--lab-mono); font-size: .8rem;
  font-weight: 600; color: var(--lab-text-2);}
.lab-mem {position: relative; height: 10px; border-radius: 999px; background: var(--lab-track); overflow: hidden;
  display: flex; margin: .4rem 0 .5rem 0;}
.lab-mem div {height: 100%;}
/* motion: a light that circles the hero's edge, flowing "Lab", tiles that rise in, bars that
   glide and shine. All off when the system asks for reduced motion. */
@property --lab-angle {syntax: "<angle>"; initial-value: 0deg; inherits: false;}
@keyframes lab-spin {to {--lab-angle: 360deg;}}
.st-key-lab_hero h1 em {background-size: 200% 100%; animation: lab-flow 6s ease-in-out infinite alternate;}
@keyframes lab-flow {from {background-position: 0% 50%;} to {background-position: 100% 50%;}}
.st-key-lab_hero {animation: lab-rise .7s cubic-bezier(.2, .7, .2, 1) both;}
.st-key-lab_shortcuts [data-testid="stColumn"] {animation: lab-rise .6s cubic-bezier(.2, .7, .2, 1) both;}
.st-key-lab_shortcuts [data-testid="stColumn"]:nth-child(1) {animation-delay: .10s;}
.st-key-lab_shortcuts [data-testid="stColumn"]:nth-child(2) {animation-delay: .16s;}
.st-key-lab_shortcuts [data-testid="stColumn"]:nth-child(3) {animation-delay: .22s;}
.st-key-lab_shortcuts [data-testid="stColumn"]:nth-child(4) {animation-delay: .28s;}
.st-key-lab_shortcuts [data-testid="stColumn"]:nth-child(5) {animation-delay: .34s;}
@keyframes lab-rise {from {opacity: 0; transform: translateY(10px);} to {opacity: 1; transform: none;}}
.st-key-lab_shortcuts [data-testid="stPageLink"] a [data-testid="stIconMaterial"] {
  transition: transform .2s;}
.st-key-lab_shortcuts [data-testid="stPageLink"] a:hover [data-testid="stIconMaterial"] {
  transform: scale(1.18);}
.lab-mem div {transition: width .8s cubic-bezier(.2, .7, .2, 1);}
.lab-mem::after {content: ""; position: absolute; inset: 0; width: 30%;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,.18), transparent);
  animation: lab-sheen 3.2s ease-in-out infinite;}
@keyframes lab-sheen {from {transform: translateX(-120%);} to {transform: translateX(420%);}}
@media (prefers-reduced-motion: reduce) {.st-key-lab_hero::after, .st-key-lab_hero h1 em, .st-key-lab_hero,
  .st-key-lab_shortcuts [data-testid="stColumn"], .lab-mem::after {animation: none;}}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def this_machine() -> dict:
    return machine.describe()  # system_profiler / nvidia-smi / GCP metadata: read once


MACHINE_KIND = {  # label and colour of the badge: tells a Mac from the GCP machine at a glance
    "mac": (tr("本機 Mac", "This Mac"), "#9DBBE8"),
    "gcp": (tr("GCP 雲端主機", "GCP VM"), "#E2C285"),
    "linux": (tr("Linux 主機", "Linux host"), "#9ED0E4"),
}
FACT_LABELS = {
    "zone": tr("位置", "Where"),
    "chip": tr("晶片", "Chip"),
    "gpu": "GPU",
    "cpu": "CPU",
    "memory": tr("記憶體", "Memory"),
    "os": tr("系統", "OS"),
    "host": tr("主機名稱", "Host"),
}
info = this_machine()
kind_label, kind_color = MACHINE_KIND[info["kind"]]
facts = "".join(
    f"<dt>{FACT_LABELS.get(k, k)}</dt><dd>{html.escape(v)}"
    + (
        tr("（CPU、GPU 共用）", " (shared by CPU and GPU)")
        if info["kind"] == "mac" and k == "memory"
        else ""
    )
    + "</dd>"
    for k, v in info["facts"]
)
with st.container(key="lab_hero"):
    text_col, anim_col, machine_col = st.columns(
        [1.15, 0.95, 1.1], gap="medium", vertical_alignment="center"
    )
    text_col.markdown(
        '<div class="lab-eyebrow">On-device AI lab</div>'
        "<h1>Local Agent <em>Lab</em></h1>"
        "<p>"
        + tr(
            "在這台機器上測試地端 LLM、展示多 Agent 協作——所有模型與資料都留在本機。",
            "Benchmark local LLMs and demo multi-agent workflows on this machine — every model "
            "and every byte of data stays here.",
        )
        + "</p>",
        unsafe_allow_html=True,
    )
    with anim_col:
        lottie.play("core", height=200)  # this machine as a core; everything stays inside
    machine_col.markdown(
        f'<div class="lab-machine" style="--kind:{kind_color}">'
        f'<div class="hd"><span class="live"></span>{tr("這台機器", "This machine")}'
        f'<span class="kind">{kind_label}</span></div>'
        f'<div class="ttl">{html.escape(info["title"])}</div><dl>{facts}</dl></div>',
        unsafe_allow_html=True,
    )
SHORTCUTS = [  # models first, then building and the demo; the scenarios have their own section below
    ("views/models.py", tr("瀏覽模型庫", "Browse models"), ":material/apps:"),
    ("views/llm.py", tr("比較語言模型", "Compare language models"), ":material/balance:"),
    ("views/speech.py", tr("比較語音模型", "Compare speech models"), ":material/graphic_eq:"),
    ("views/agent.py", tr("打造一個 Agent", "Build an agent"), ":material/smart_toy:"),
    ("views/voice_assistant.py", tr("即時語音助理", "Live voice"), ":material/support_agent:"),
]
links = st.container(key="lab_shortcuts").columns(len(SHORTCUTS))
for col, (path, label, icon) in zip(links, SHORTCUTS, strict=True):
    col.page_link(path, label=label, icon=icon, width="stretch")
model_news.home_notice()  # new models from the weekly check, until dismissed


# --- live system status --------------------------------------------------------------------------
def model_kind(name: str) -> str:
    caps = set((catalog.find(name) or {}).get("caps") or [])
    for key in ("embedding", "decision", "vision"):
        if key in caps:
            return key
    return "chat"


GIB = 2**30  # memory in the units macOS and nvidia-smi use, so 64 GB reads as 64, not 69


def gateway_processes() -> list[psutil.Process]:
    """The model gateway's processes (`uv run` and the Python it starts both match)."""
    found = []
    for proc in psutil.process_iter(["cmdline"]):
        try:
            cmd = " ".join(proc.info.get("cmdline") or [])
        except (psutil.Error, TypeError):
            continue
        if "local_llm.gateway" in cmd:
            found.append(proc)
    return found


def gateway_process_rss() -> float:
    total = 0.0
    for proc in gateway_processes():
        try:
            total += proc.memory_info().rss / GIB
        except psutil.Error:
            pass
    return total


def memory_rows(loaded: list[dict], gw_loaded: list[dict]):
    """What the model memory bar shows: (title, [(label, GB, colour)], total GB, extra note).

    On a Mac the CPU and GPU share one memory, so models count against it. With an NVIDIA GPU
    (the GCP machine) models live in the GPU's own memory, apart from the system memory, so the
    bar is the GPU memory and the system memory is a note under it.
    """
    mem = psutil.virtual_memory()
    ollama_gb = sum((m.get("size_vram") or m.get("size", 0)) for m in loaded) / GIB
    gpu = machine.gpu_memory()
    if gpu is None:
        gw_gb = gateway_process_rss() if gw_loaded else 0.0
        other = max(mem.used / GIB - ollama_gb - gw_gb, 0)
        rows = [("Ollama", ollama_gb, ui.PRIMARY), (tr("模型閘道", "Gateway"), gw_gb, "#C2678F"),
                (tr("其他程式", "Other apps"), other, "var(--lab-other)")]  # fmt: skip
        title = tr("統一記憶體與載入的模型", "Unified memory and loaded models")
        note = tr("CPU、GPU 共用", "shared by CPU and GPU")
        return title, rows, mem.total / GIB, note
    used, total = gpu
    by_pid = machine.gpu_memory_by_pid()
    gw_gb = sum(by_pid.get(p.pid, 0.0) for p in gateway_processes())
    other = max(used - ollama_gb - gw_gb, 0)
    rows = [("Ollama", ollama_gb, ui.PRIMARY), (tr("模型閘道", "Gateway"), gw_gb, "#C2678F"),
            (tr("其他 GPU 程式", "Other GPU apps"), other, "var(--lab-other)")]  # fmt: skip
    title = tr("GPU 顯存與載入的模型", "GPU memory and loaded models")
    note = tr(
        f"系統記憶體另計：已用 {mem.used / GIB:.0f} / {mem.total / GIB:.0f} GB",
        f"system memory is separate: {mem.used / GIB:.0f} of {mem.total / GIB:.0f} GB used",
    )
    return title, rows, total, note


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


PROJECT_DIR = Path(__file__).resolve().parents[2]
# Where model files land: this project's models/ (Ollama and the gateway) and Hugging Face's cache.
MODEL_FOLDERS = (PROJECT_DIR / "models", Path.home() / ".cache" / "huggingface")


@st.cache_data(ttl=600, show_spinner=False)
def folder_gb(path: str) -> float:
    """Disk space a folder takes (allocated blocks, symlinks not followed); cached 10 minutes."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(root, name)).st_blocks * 512
            except OSError:
                pass
    return total / 1e9


def disk_card() -> None:
    disk = psutil.disk_usage(str(PROJECT_DIR))
    total, free = disk.total / 1e9, disk.free / 1e9
    used = total - free  # on macOS `used` counts one APFS volume only; total - free is the disk
    models = sum(folder_gb(str(p)) for p in MODEL_FOLDERS if p.exists())
    other = max(used - models, 0)
    ui.card_title("hard_drive", tr("硬碟", "Disk"))
    bar = "".join(
        f'<div style="width:{w / total * 100:.1f}%;background:{color}"></div>'
        for w, color in ((models, ui.PRIMARY), (other, "var(--lab-other)"))
    )
    st.markdown(f'<div class="lab-mem">{bar}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<span class="lab-kv"><span class="lab-dot" style="background:{ui.PRIMARY}"></span>'
        f"{tr('模型檔', 'Model files')} {models:.0f} GB　"
        '<span class="lab-dot" style="background:var(--lab-other)"></span>'
        f"{tr('其他', 'Other')} {other:.0f} GB　·　"
        f"{tr('剩', 'free')} <b>{free:.0f} GB</b> / {total:.0f} GB</span>",
        unsafe_allow_html=True,
    )
    if free < 15 or free / total < 0.08:
        st.warning(
            tr(
                f"硬碟只剩 {free:.0f} GB：下載大模型可能失敗，先到模型庫刪掉不用的模型。",
                f"Only {free:.0f} GB left: large downloads may fail. Delete unused models in "
                "the model library first.",
            ),
            icon=":material/warning:",
        )


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
        return f'<span class="lab-dot" style="background:{"#4CAF7A" if ok else "#D46A6A"}"></span>'

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

    title, rows, total_gb, note = memory_rows(loaded, gw_loaded)
    left, right = st.columns([1.15, 1], gap="large")
    with left.container(border=True):
        ui.card_title("memory", title)
        bar = "".join(
            f'<div style="width:{gb / total_gb * 100:.1f}%;background:{color}"></div>'
            for _, gb, color in rows
        )
        st.markdown(f'<div class="lab-mem">{bar}</div>', unsafe_allow_html=True)
        st.markdown(
            '<span class="lab-kv">'
            + "　".join(
                f'<span class="lab-dot" style="background:{color}"></span>{label} {gb:.1f} GB'
                for label, gb, color in rows
            )
            + f"<br>{tr('共', 'of')} {total_gb:.0f} GB · "
            f"{tr('模型預算', 'model budget')} {GPU_BUDGET_GB:.0f} GB · {note}</span>",
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
            gb = (m.get("size_vram") or m.get("size", 0)) / GIB
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
        disk_card()
    with right.container(border=True):
        ui.card_title("pending_actions", tr("工作", "Tasks"))
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


ui.section("01", tr("系統現況", "Live status"), tr("每 3 秒更新", "refreshes every 3 s"))
live_status()

# --- scenarios ------------------------------------------------------------------------------------
scenarios = store.scenarios()


def open_scenario(sid: str) -> None:
    st.session_state.scenario_id = sid
    st.session_state["_open_scenario"] = True


# Collapsed like "About this tool" below: a glance at the scenarios, with the Scenario gallery for
# the details.
cards = [*scenarios, None]  # the live voice assistant is a page of its own, not a graph
ui.section("02", tr("場景與架構", "Scenarios and architecture"))
with st.expander(tr(f"多 Agent 場景速覽（目前 {len(cards)} 個）",
                    f"Multi-agent scenarios at a glance ({len(cards)} now)")):  # fmt: skip
    st.page_link(
        "views/scenarios.py",
        label=tr("詳細請看「場景展示」：每個場景的說明、流程、執行與時間軸",
                 "More on the Scenario gallery: each scenario's description, flow, runs and timeline"),
        icon=":material/hub:",
    )  # fmt: skip
    st.caption(
        tr(
            "或直接點下面的「開啟」，到場景展示頁執行該場景；每次執行都會即時畫出流程與時間軸。",
            "Or press Open below to run one on the Scenario gallery; every run draws its flow "
            "and timeline live.",
        )
    )
    for row in range(0, len(cards), 3):
        for col, s in zip(st.columns(3, gap="medium"), cards[row : row + 3], strict=False):
            if s is None:
                voice_assistant.home_card(col)
                continue
            need = engine.requirements(s)
            kinds = sorted({flow_ui.node_kind(n) for n in s["nodes"]} - {None})
            models = [flow_ui.short_model(m) for m in sorted(need["ollama"] | need["gateway"])]
            with col.container(border=True, height=450):
                icon = flow_ui.category_icon(s.get("category"))
                st.markdown(f":material/{icon}: **{pick(s['name'])}**")
                st.graphviz_chart(
                    ui.themed_dot(flow_ui.flow_dot(s, compact=True)), use_container_width=True
                )
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
            "**兩個目的**：① 測試各種地端 LLM 的表現（模型庫、語言模型比較、語音模型比較、其他模型比較）；"
            "② 展示多 Agent 場景（Agent 工作室、多 Agent 比較、場景展示、知識庫、總控 Coding Agent）。"
            "所有服務只監聽 127.0.0.1；權限由資料庫角色與檢索過濾強制，不靠 prompt。",
            "**Two goals**: ① benchmark local LLMs (model library, chat, compare, speech & "
            "documents); ② demo multi-agent scenarios (agent studio, agent comparison, scenario "
            "gallery, knowledge base, coding agent). Every service listens on 127.0.0.1 only; "
            "permissions are enforced by database roles and retrieval filters, not prompts.",
        )
    )
    st.graphviz_chart(ui.themed_dot(flow_ui.architecture_dot()), use_container_width=True)
