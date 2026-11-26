"""Shared Streamlit helpers: backend settings, model picker with download, metrics, run files."""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import access
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


def this_machine(capital: bool = False) -> str:
    """How pages name the machine the platform runs on: the Mac, or the shared GPU machine."""
    if catalog.IS_MAC:
        return tr("這台 Mac", "This Mac" if capital else "this Mac")
    return tr("這台 GPU 主機", "This GPU machine" if capital else "this GPU machine")


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
    # Select box options cannot show icons, so a plain text mark: ↓ to download. (The catalog's
    # hand-set "recommended" flags are not shown: they predate and disagree with our tests.)
    mark = "" if catalog.normalize(tag) in installed else "↓ "
    size = f" ({info['size_gb']}GB)" if info else ""
    return f"{mark}{tag}{size}"


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


@st.cache_data(ttl=60, show_spinner=False)
def cloud_keys() -> dict[str, bool]:
    """Which cloud services this server has keys for (.env / key files): a server without them
    (the shared GCP platform) must not offer cloud models that would only fail when picked."""
    from local_llm import elevenlabs, gcp_auth, neural_tts, voice_cloud

    return {
        "Azure": neural_tts.azure_speech() is not None,
        "Google": gcp_auth.available() or gcp_auth.available(gcp_auth.tts_key_path()),
        "AWS": voice_cloud.aws_credentials() is not None,
        "ElevenLabs": elevenlabs.available(),
    }


def cloud_enabled() -> bool:
    """The top bar's cloud switch (main.py): off unless turned on, kept in the URL (?cloud=1);
    always off on a server with no cloud keys."""
    if not any(cloud_keys().values()):
        return False
    if "cloud_on" not in st.session_state:
        st.session_state.cloud_on = st.query_params.get("cloud") == "1"
    return bool(st.session_state.cloud_on)


def cloud_switch() -> None:
    """Cloud models (Azure, Google, AWS) can be picked only while this is on; everything else
    on the platform runs on this machine. Greyed out on a server with no cloud keys."""
    keys = cloud_keys()
    have = [k for k, ok in keys.items() if ok]
    if not have:
        st.toggle(
            ":material/cloud_off: " + tr("雲端模型", "Cloud"),
            value=False,
            disabled=True,
            key="_cloud_switch_off",
            help=tr(
                "這台平台沒有設定雲端金鑰，所以不能用雲端模型；所有功能都在本機執行。",
                "This server has no cloud keys, so cloud models are not available; everything "
                "runs on this machine.",
            ),
        )
        return
    cloud_enabled()

    def changed() -> None:
        st.session_state.cloud_on = st.session_state._cloud_switch
        if st.session_state.cloud_on:
            st.query_params["cloud"] = "1"
        else:
            st.query_params.pop("cloud", None)

    missing = [k for k, ok in keys.items() if not ok]
    st.session_state._cloud_switch = st.session_state.cloud_on
    st.toggle(
        ":material/cloud: " + tr("雲端模型", "Cloud"),
        key="_cloud_switch",
        on_change=changed,
        help=tr(
            "打開後才能選雲端模型（Azure、Google、AWS、ElevenLabs 的語音對話、語音合成、"
            "同步口譯）。"
            "選了雲端模型，錄音或文字會上傳到該服務；關閉時平台全部在本機執行。"
            f"這台平台有 {'、'.join(have)} 的金鑰"
            + (f"；沒有 {'、'.join(missing)} 的，那些模型不會列出。" if missing else "。"),
            "Turn on to pick cloud models (Azure, Google, AWS, ElevenLabs voice chat, speech "
            "synthesis, "
            "interpreting). With a cloud model the audio or text goes to that service; off, "
            f"everything runs on this machine. This server has keys for {', '.join(have)}"
            + (f"; not for {', '.join(missing)}, whose models are not listed." if missing else "."),
        ),
    )


def local_or_allowed(v: dict) -> bool:
    """A voice or model dict may be offered: it runs here, or the cloud switch is on and this
    server has the key its service needs."""
    if not v.get("cloud"):
        return True
    if not cloud_enabled():
        return False
    from local_llm import neural_tts

    n = neural_tts
    if any(v.get("name") in d for d in (n.GOOGLE_TTS, n.AZURE_TTS, n.GEMINI_TTS, n.ELEVENLABS_TTS)):
        return n.available(v["name"])
    return True


def cloud_hint(what: str, what_en: str) -> None:
    """A caption where cloud options are hidden by the switch (or by a server without keys)."""
    if not any(cloud_keys().values()):
        st.caption(
            tr(
                f":material/cloud_off: 這台平台沒有設定雲端金鑰，雲端{what}無法使用。",
                f":material/cloud_off: This server has no cloud keys, so cloud {what_en} are "
                "not available.",
            )  # fmt: skip
        )
    elif not cloud_enabled():
        st.caption(
            tr(
                f":material/cloud_off: 雲端{what}已隱藏；要比較請打開右上角的「雲端模型」。",
                f":material/cloud_off: Cloud {what_en} are hidden; turn on Cloud (top right) "
                "to compare them.",
            )  # fmt: skip
        )


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


def model_picker(
    label: str, key: str, task: str | None = None, container=st, cloud: bool = False
) -> str | None:
    """Select box over installed + downloadable models; offers a download for the latter.
    cloud: also the cloud language models (cloud_llm.py) while the cloud switch is on.

    Returns the model name only when it is installed and ready (a cloud model always is).
    """
    from local_llm import cloud_llm

    options = model_options(task)
    if cloud and cloud_enabled():
        options += cloud_llm.models()
    installed = {catalog.normalize(n) for n in installed_names()}
    default = catalog.default_model()
    index = next((i for i, o in enumerate(options) if catalog.same_model(o, default)), 0)
    if st.session_state.get(key) not in options:
        st.session_state[key] = options[index] if options else None
    choice = container.selectbox(
        label,
        options,
        format_func=lambda t: (
            cloud_llm.label(t) if cloud_llm.is_cloud(t) else model_label(t, installed)
        ),
        **kept(key),
    )
    if choice and cloud_llm.is_cloud(choice):
        return choice
    if choice and catalog.normalize(choice) not in installed:
        info = catalog.find(choice) or {}
        size = info.get("size_gb", "?")
        container.warning(
            tr(f"{choice} 尚未下載（約 {size}GB）。", f"{choice} is not downloaded (~{size} GB).")
        )
        if catalog.fit(info) == "no":
            container.error(
                tr(
                    f"這個模型超過{this_machine()}的記憶體，放不下。",
                    f"This model does not fit in {this_machine()}'s memory.",
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


def think_control(model: str | None, key: str, container=st, compact: bool = False) -> bool:
    """Thinking toggle that reflects what the model actually supports.

    compact: explain an always-on toggle in its tooltip instead of a caption (for toolbars).
    """
    from local_llm import cloud_llm

    if model and cloud_llm.is_cloud(model):
        mode = "optional"  # every cloud model here can think or not
    else:
        mode = ollama_admin.thinking_mode(model, ollama_url()) if model else "none"
    # Separate widget state per mode, so a forced-on toggle does not carry over to other models.
    key = f"{key}_{mode}"
    label = tr("思考模式", "Thinking")
    if mode == "always":
        note = tr(
            "這個模型一定會先思考，無法關閉（思考內容會另外顯示）。",
            "This model always thinks first; it cannot be switched off (shown separately).",
        )
        container.toggle(label, value=True, disabled=True, key=key, help=note if compact else None)
        if not compact:
            container.caption(note)
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
                "請改用 gemma4:31b 等標示 tools 的模型；"
                "這個模型可以在「對話測試」或「模型比較」使用。",
                f"{model} does not support tool calling (e.g. the GGUF builds of TAIDE and "
                "Breeze), so this agent cannot query the database or call tools. Use a model "
                "with tools such as gemma4:31b; this one works in Chat and Compare.",
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
        # Only models this viewer may unload (the admin: all; a user: those only they use).
        mine = [name for name in loaded if access.may_unload(name)]
        if mine and container.button(
            tr("先卸載目前載入的模型", "Unload the loaded models first")
            if len(mine) == len(loaded)
            else tr(f"先卸載你載入的模型（{len(mine)} 個）", f"Unload yours first ({len(mine)})"),
            key=f"unload_{'_'.join(models)}",
        ):
            for name in mine:
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
    """Write runs/<kind>-<YYYYmmdd-HHMMSS>.json. Two runs of a kind finishing in the same
    second (background jobs) must not overwrite each other: the stamp moves on a second."""
    RUNS_DIR.mkdir(exist_ok=True)
    now = datetime.now()
    record = {"kind": kind, "recorded_at": now.astimezone().isoformat(), **payload}
    text = json.dumps(record, ensure_ascii=False, indent=2, default=str) + "\n"
    for extra in range(120):
        stamp = (now + timedelta(seconds=extra)).strftime("%Y%m%d-%H%M%S")
        path = RUNS_DIR / f"{kind}-{stamp}.json"
        try:
            with path.open("x") as f:  # exclusive: fails if the file exists
                f.write(text)
            return path
        except FileExistsError:
            continue
    raise RuntimeError(f"could not find a free file name for a {kind} run")
