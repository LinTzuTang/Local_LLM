"""Shared Streamlit helpers: backend settings, model picker with download, metrics, run files."""

import json
import os
from datetime import datetime
from pathlib import Path

import httpx
import jobs
import streamlit as st
from dotenv import load_dotenv
from i18n import tr

from local_llm import catalog, gateway_client, ollama_admin
from local_llm.streaming import Backend, Stats

load_dotenv()
PROJECT_DIR = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_DIR / "runs"
# Rough GPU budget on a 64GB Mac (macOS gives the GPU about 75% of unified memory).
GPU_BUDGET_GB = float(os.getenv("LOCAL_LLM_GPU_BUDGET_GB", "44"))


def ollama_url() -> str:
    if "ollama_url" not in st.session_state:
        url = os.getenv("LOCAL_LLM_BASE_URL", ollama_admin.DEFAULT_URL)
        st.session_state.ollama_url = url if "11434" in url else ollama_admin.DEFAULT_URL
    return st.session_state.ollama_url


def gateway_url() -> str:
    """Model gateway for speech, document and reranker models (see docs/GATEWAY.md)."""
    return os.getenv("LOCAL_LLM_GATEWAY_URL", gateway_client.DEFAULT_URL)


def ollama_backend() -> Backend:
    return Backend("Ollama", "ollama", ollama_url())


def require_ollama() -> str:
    """Stop the page with instructions when the Ollama server is down."""
    ver = ollama_admin.version(ollama_url())
    if ver is None:
        st.error(
            tr(
                f"連不到 Ollama（{ollama_url()}）。請在 terminal 執行 `./scripts/tmux_up.sh`。",
                f"Cannot reach Ollama ({ollama_url()}). Run `./scripts/tmux_up.sh` in a terminal.",
            )
        )
        st.stop()
    return ver


def installed_names() -> list[str]:
    return sorted(m["name"] for m in ollama_admin.installed(ollama_url()))


def model_label(tag: str, installed: set[str]) -> str:
    info = catalog.find(tag)
    mark = "" if catalog.normalize(tag) in installed else "⬇️ "
    size = f" ({info['size_gb']}GB)" if info else ""
    star = " ⭐" if info and info.get("recommended") else ""
    return f"{mark}{tag}{size}{star}"


def model_options(task: str | None = None, include_catalog: bool = True) -> list[str]:
    """Installed models first (filtered by task when known), then catalog models to download."""
    installed = installed_names()
    wanted = [m["tag"] for m in catalog.models(task)] if task else []
    chat_capable = [n for n in installed if catalog.is_chat(n)]
    ordered = [n for n in chat_capable if not task or any(catalog.same_model(n, w) for w in wanted)]
    ordered += [n for n in chat_capable if n not in ordered]
    if include_catalog:
        normalized = {catalog.normalize(n) for n in installed}
        source = wanted or [m["tag"] for m in catalog.models() if catalog.is_chat(m["tag"])]
        ordered += [t for t in source if catalog.normalize(t) not in normalized]
    return ordered


def kept(key: str) -> dict:
    """Widget kwargs whose value survives page switches.

    Streamlit drops the state of widgets that are not on the current page (also during the
    background-job panel's refreshes), so the value is mirrored into a plain session key.
    """
    widget_key = f"_kept_{key}"
    if key in st.session_state:
        st.session_state[widget_key] = st.session_state[key]
    return {"key": widget_key, "on_change": _mirror, "args": (widget_key, key)}


def _mirror(widget_key: str, key: str) -> None:
    st.session_state[key] = st.session_state[widget_key]


def model_picker(label: str, key: str, task: str | None = None, container=st) -> str | None:
    """Select box over installed + downloadable models; offers a download for the latter.

    Returns the model name only when it is installed and ready.
    """
    options = model_options(task)
    installed = {catalog.normalize(n) for n in installed_names()}
    default = catalog.default_model()
    index = next((i for i, o in enumerate(options) if catalog.same_model(o, default)), 0)
    if st.session_state.get(key) not in options:
        st.session_state[key] = options[index] if options else None
    choice = container.selectbox(
        label, options, format_func=lambda t: model_label(t, installed), **kept(key)
    )
    if choice and catalog.normalize(choice) not in installed:
        info = catalog.find(choice) or {}
        size = info.get("size_gb", "?")
        container.warning(
            tr(f"{choice} 尚未下載（約 {size}GB）。", f"{choice} is not downloaded (~{size} GB).")
        )
        if info.get("fit") == "no":
            container.error(
                tr(
                    "這個模型超過本機記憶體，建議放在 DGX Spark。",
                    "This model does not fit in this Mac's memory; use a DGX Spark.",
                )
            )
        download_control(choice, key=f"{key}_pull", container=container)
        return None
    return choice


def _pull_work(job: jobs.Job, base_url: str) -> None:
    """Worker thread: no Streamlit calls."""
    model, source = job.state["model"], job.state.get("source", "ollama")
    # Both servers stream the same {"status", "total", "completed"} events.
    pull = gateway_client.pull if source == "gateway" else ollama_admin.pull
    for event in pull(model, base_url):
        if job.cancel.is_set():
            return  # closing the stream makes the server stop the download
        total, done = event.get("total"), event.get("completed")
        if total and done:
            job.progress = min(done / total, 1.0)
            job.detail = f"{done / 1e9:.1f} / {total / 1e9:.1f} GB"
        else:
            if event["status"] == "success":
                job.progress = 1.0
            elif source == "ollama":
                job.progress = 0.0
            # Gateway steps without bytes (verifying, converting) keep the bar where it is.
            job.detail = event["status"]


def start_pull(model: str, source: str = "ollama", page: str = "views/models.py") -> jobs.Job:
    """Download in the background; reuses a download of the same model already running.

    source: "ollama" (LLMs, embeddings) or "gateway" (speech, document, reranker models).
    """
    job = jobs.find("pull", model=model)
    if job and job.running:
        return job
    url = gateway_url() if source == "gateway" else ollama_url()
    title = tr(f"下載 {model}", f"Download {model}")
    if source == "gateway":
        title += tr("（模型閘道）", " (gateway)")
    return jobs.start(
        jobs.Job("pull", title, page, state={"model": model, "source": source}),
        lambda j: _pull_work(j, url),
    )


@st.fragment(run_every=1)
def pull_progress(job: jobs.Job, key: str) -> None:
    if not job.running:
        st.rerun()  # refresh the whole page so the model shows up as installed
    detail = job.detail or tr("準備中…", "starting…")
    st.progress(
        job.progress or 0.0,
        text=tr(f"下載 {job.state['model']}：", f"Downloading {job.state['model']}: ") + detail,
    )
    if st.button(tr("取消下載", "Cancel download"), key=f"{key}_cancel"):
        job.cancel.set()


def download_control(
    model: str,
    key: str,
    container=st,
    label: str | None = None,
    source: str = "ollama",
    page: str = "views/models.py",
) -> None:
    """Download button, or live progress when a download of this model is running.

    The download runs in the background, so it continues when the user switches pages.
    source="gateway" downloads through the model gateway instead of Ollama.
    """
    job = jobs.find("pull", model=model)
    if job and job.running:
        if container is st:
            pull_progress(job, key)
        else:
            with container:
                pull_progress(job, key)
        return
    if job and job.status == "error":
        container.error(
            tr(f"上次下載 {model} 失敗：", f"Last download of {model} failed: ") + str(job.error)
        )
    if container.button(label or tr(f"下載 {model}", f"Download {model}"), key=key, type="primary"):
        start_pull(model, source, page)
        st.rerun()


def think_control(model: str | None, key: str, container=st) -> bool:
    """Thinking toggle that reflects what the model actually supports."""
    mode = ollama_admin.thinking_mode(model, ollama_url()) if model else "none"
    # Separate widget state per mode, so a forced-on toggle does not carry over to other models.
    key = f"{key}_{mode}"
    label = tr("思考模式", "Thinking")
    if mode == "always":
        container.toggle(label, value=True, disabled=True, key=key)
        container.caption(
            tr(
                "這個模型一定會先思考，無法關閉（思考內容會另外顯示）。",
                "This model always thinks first; it cannot be switched off (shown separately).",
            )
        )
        return True
    if mode == "none":
        container.toggle(label, value=False, disabled=True, key=key)
        return False
    return container.toggle(
        label,
        value=False,
        key=key,
        help=tr("開啟後回答前會先推理，較慢", "Reason before answering (slower)"),
    )


def warn_if_no_tools(model: str | None, needs_tools: bool, container=st) -> None:
    """Agents with tools need a model whose template supports tool calling."""
    if model and needs_tools and not ollama_admin.supports_tools(model, ollama_url()):
        container.warning(
            tr(
                f"{model} 不支援工具呼叫（例如 TAIDE、Breeze 的 GGUF 版本），"
                "這個 agent 需要查資料庫或呼叫工具，會無法運作。"
                "請改用 qwen3:30b-a3b 等標示 tools 的模型；"
                "這個模型可以在「💬 對話」或「⚖️ 模型比較」使用。",
                f"{model} does not support tool calling (e.g. the GGUF builds of TAIDE and "
                "Breeze), so this agent cannot query the database or call tools. Use a model "
                "with tools such as qwen3:30b-a3b; this one works in Chat and Compare.",
            )
        )


def memory_guard(models: list[str], container=st) -> None:
    """Warn when the models about to run do not fit next to what is already loaded."""
    loaded = {m["name"]: m.get("size_vram") or m.get("size", 0) for m in _safe_loaded()}
    new = sum(
        ((catalog.find(m) or {}).get("size_gb") or 0)
        for m in models
        if not any(catalog.same_model(m, n) for n in loaded)
    )
    used = sum(loaded.values()) / 1e9
    if used + new > GPU_BUDGET_GB and loaded:
        container.warning(
            tr(
                f"已載入 {used:.1f}GB，再載入約 {new:.0f}GB 會超過約 {GPU_BUDGET_GB:.0f}GB 預算，"
                "Mac 可能開始用 swap 而變慢。",
                f"{used:.1f} GB is loaded; another ~{new:.0f} GB exceeds the "
                f"~{GPU_BUDGET_GB:.0f} GB budget and the Mac may start swapping.",
            )
        )
        if container.button(
            tr("先卸載目前載入的模型", "Unload the loaded models first"),
            key=f"unload_{'_'.join(models)}",
        ):
            for name in loaded:
                ollama_admin.unload(name, ollama_url())
            st.rerun()


def _safe_loaded() -> list[dict]:
    try:
        return ollama_admin.loaded(ollama_url())
    except httpx.HTTPError:
        return []


def stats_caption(stats: Stats | None) -> str:
    """Page-side only (uses tr()); workers keep the Stats object and the page formats it."""
    if not stats:
        return ""
    parts = []
    if stats.first_answer_s is not None:
        parts.append(tr("首字", "first token") + f" {stats.first_answer_s:.2f}s")
    if stats.ttft_s is not None and stats.ttft_s != stats.first_answer_s:
        parts.append(tr("開始思考", "thinking starts") + f" {stats.ttft_s:.2f}s")
    if stats.decode_tps:
        parts.append(f"{stats.decode_tps:.1f} tokens/s")
    if stats.output_tokens is not None:
        parts.append(tr(f"輸出 {stats.output_tokens} tokens", f"{stats.output_tokens} tokens out"))
    parts.append(tr("總計", "total") + f" {stats.total_s:.2f}s")
    if stats.load_s and stats.load_s > 0.5:
        parts.append(tr("含載入", "incl. load") + f" {stats.load_s:.1f}s")
    return " · ".join(parts)


def save_run(kind: str, payload: dict) -> Path:
    RUNS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = RUNS_DIR / f"{kind}-{stamp}.json"
    record = {"kind": kind, "recorded_at": datetime.now().astimezone().isoformat(), **payload}
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str) + "\n")
    return path
