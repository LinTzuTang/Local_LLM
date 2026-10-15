"""Shared pieces of the speech-model page (views/speech.py): audio choice, model pickers,
the Simplified -> Traditional display option, cached audio files and the running-job bar.

Workers never import Streamlit state from here: they get plain values (paths, model ids) and
write to job.state; everything that calls `st.*` or `tr()` runs on the page.
"""

import hashlib
from functools import cache
from pathlib import Path

import jobs
import streamlit as st
from common import PROJECT_DIR, download_control, kept
from i18n import pick, tr

from local_llm import asr_metrics, catalog, gateway_client, media_tools

PAGE = "views/speech.py"
SAMPLES_DIR = media_tools.SAMPLES_DIR
# Bundled recordings with a reference script next to them (<name>.txt); see samples/README.md.
SAMPLES = ["national_day_2026_zh.wav", "meeting_zh.wav"]
# Speaker turns and an English reference translation for the bundled samples (this project's).
REFS_DIR = PROJECT_DIR / "data" / "lab" / "speech_refs"
AUDIO_TYPES = sorted(t.lstrip(".") for t in media_tools.AUDIO_TYPES)
# Augmented and cut audio (noise, telephone, live-simulation chunks), regenerated on demand.
CACHE_DIR = PROJECT_DIR / "storage" / "speech_cache"


# -- names and labels -----------------------------------------------------------------------


def names(task: str = "asr") -> dict[str, str]:
    """Gateway model id -> display name in the current language."""
    return {
        m["id"]: pick({"zh": m.get("name") or m["id"], "en": m.get("name_en") or m.get("name")})
        for m in catalog.gateway_models(task)
    }


def all_names() -> dict[str, str]:
    return names("asr") | names("diarize")


def model_label(model: str, status: dict, labels: dict) -> str:
    info = status.get(model) or {}
    mark = "" if info.get("installed") else "↓ "
    size = f" ({info['size_gb']}GB)" if info.get("size_gb") else ""
    return f"{mark}{labels.get(model, model)}{size}"


def pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def sample_label(name: str) -> str:
    return {
        "national_day_2026_zh.wav": tr(
            "真人：總統國慶演說節錄（4 分 29 秒，照稿宣讀）",
            "Real voice: President's National Day address, excerpt (4 min 29 s, read from script)",
        ),
        "meeting_zh.wav": tr(
            "合成語音：虛構產銷會議（88 秒，兩位講者）",
            "Synthetic voices: fictional production meeting (88 s, two speakers)",
        ),
    }.get(name, name)


def sample_note(name: str) -> str:
    return {
        "national_day_2026_zh.wav": tr(
            "〔總統府提供〕總統府 YouTube 頻道 115 年國慶大會影片 07:14–11:43，原稿取自總統府"
            "新聞的演說全文，依實際口述修改 4 處；來源與授權見 data/lab/samples/README.md。",
            "Courtesy of the Office of the President: its YouTube video of the 2026 National "
            "Day address, 07:14–11:43; the reference is the official full text with 4 edits "
            "where the speaker deviated. Source and licence: data/lab/samples/README.md.",
        ),
        "meeting_zh.wav": tr(
            "88 秒的虛構產銷會議（macOS 合成語音），原稿在同資料夾的 meeting_zh.txt。",
            "An 88-second fictional production meeting (macOS synthetic voices); the script "
            "is meeting_zh.txt in the same folder.",
        ),
    }.get(name, "")


# -- Simplified -> Traditional display -------------------------------------------------------


@cache
def _s2tw():
    import opencc

    return opencc.OpenCC("s2tw")


def show(text: str | None) -> str:
    """Text as displayed: converted to Traditional (OpenCC s2tw) when the page toggle is on.
    Saved runs always keep the model's raw output."""
    if not text:
        return text or ""
    return _s2tw().convert(text) if st.session_state.get("speech_s2tw", True) else text


def s2tw_toggle(container=st) -> None:
    st.session_state.setdefault("speech_s2tw", True)
    container.toggle(
        tr("簡體轉繁體顯示", "Show in Traditional"),
        help=tr(
            "許多模型輸出簡體字；開啟時以 OpenCC s2tw 轉成台灣繁體顯示。"
            "只影響畫面，存檔保留模型原始輸出；字錯率本來就會先統一繁簡再比對。",
            "Many models write Simplified Chinese; when on, transcripts are shown converted to "
            "Taiwan Traditional (OpenCC s2tw). Display only: saved runs keep the raw output, and "
            "the CER unifies the script anyway.",
        ),
        **kept("speech_s2tw"),
    )


# -- gateway models ---------------------------------------------------------------------------


def gateway_status(url: str) -> dict | None:
    """{model id: /api/tags row} for every gateway model, or None (with an error shown)."""
    try:
        return {m["id"]: m for m in gateway_client.models(url)}
    except Exception:
        st.error(
            tr(
                f"連不到模型閘道（{url}），語音模型無法執行。請執行 `./scripts/tmux_up.sh`。",
                f"Cannot reach the model gateway ({url}), so speech models cannot run. Run "
                "`./scripts/tmux_up.sh`.",
            )
        )
        return None


def download_missing(models: list[str], status: dict, prefix: str) -> None:
    """Download buttons for the chosen gateway models that are not on disk yet."""
    missing = [m for m in models if m in status and not status[m]["installed"]]
    if not missing:
        return
    with st.container(border=True):
        st.markdown(
            tr(
                "**尚未下載**（下載在背景進行；沒下載的模型不會參加比較）",
                "**Not downloaded yet** (downloads run in the background; models that are not "
                "downloaded are left out)",
            )
        )
        for m in missing:
            size = status[m]["size_gb"]
            download_control(
                m, key=f"{prefix}_pull_{m}", source="gateway", page=PAGE,
                label=tr(f"下載 {m}（約 {size}GB）", f"Download {m} (~{size} GB)"),
            )  # fmt: skip


def licence_warning(models: list[str], labels: dict) -> None:
    licences = {
        m["id"]: pick({"zh": m.get("license", ""), "en": m.get("license_en") or m.get("license")})
        for m in catalog.gateway_models()
    }
    custom = [m for m in models if not licences.get(m, "").lower().startswith(("apache", "mit"))]
    if custom:
        st.warning(
            tr(
                "下列模型不是 Apache-2.0／MIT 授權，使用前請確認條款：",
                "These models are not Apache-2.0 / MIT licensed; check the terms before use:",
            )
            + "".join(
                f"\n- **{labels.get(m, m)}**{tr('：', ': ')}{licences.get(m, '?')}" for m in custom
            )
        )


def backend(model_id: str) -> str:
    """The gateway backend that serves a model ("mlx-whisper", "mlx-audio", "ollama-audio")."""
    spec = next((m for m in catalog.gateway_models() if m["id"] == model_id), {})
    return spec.get("backend", "")


def translates(model_id: str) -> bool:
    """Whether a gateway speech model can translate speech to English by itself: audio LLMs,
    and Whisper models that kept Whisper's translate task."""
    if backend(model_id) == "ollama-audio":
        return True
    spec = next((m for m in catalog.gateway_models("asr") if m["id"] == model_id), {})
    return backend(model_id) == "mlx-whisper" and spec.get("options", {}).get("translate", True)


def has_timestamps(model_id: str) -> bool:
    """Whether the model returns timed sentence segments (the Whisper family) rather than one
    block of text or long fixed chunks."""
    return backend(model_id) == "mlx-whisper"


# -- audio ------------------------------------------------------------------------------------


def audio_choice(prefix: str, default: str = SAMPLES[0]) -> tuple | None:
    """(audio: Path or bytes, file name, source key, display name), or None until chosen.

    `prefix` keeps each mode's choice separate ("asr", "st", "mt")."""
    sources = {
        "sample": tr("範例錄音", "Sample recording"),
        "upload": tr("上傳檔案", "Upload a file"),
        "record": tr("用麥克風錄音", "Record with the microphone"),
    }
    st.session_state.setdefault(f"{prefix}_source", "sample")
    source = (
        st.segmented_control(
            tr("錄音來源", "Audio"),
            list(sources),
            format_func=sources.get,
            **kept(f"{prefix}_source"),
        )  # fmt: skip
        or "sample"
    )
    if source == "sample":
        st.session_state.setdefault(f"{prefix}_sample", default)
        sample = st.selectbox(
            tr("範例錄音", "Sample recording"),
            SAMPLES,
            format_func=sample_label,
            **kept(f"{prefix}_sample"),
        )
        path = SAMPLES_DIR / sample
        st.audio(str(path))
        st.caption(f"`data/lab/samples/{sample}` · " + sample_note(sample))
        return path, sample, source, sample
    if source == "upload":
        upload = st.file_uploader(
            tr("錄音檔", "Audio file"), type=AUDIO_TYPES, key=f"{prefix}_upload"
        )
    else:
        upload = st.audio_input(tr("按下開始錄音", "Press to record"), key=f"{prefix}_record")
    if not upload or not upload.size:
        return None
    if source == "upload":
        st.audio(upload)
    name = upload.name or "recording.wav"
    return upload.getvalue(), name, source, name


def follow(prefix: str, field: str, chosen: tuple | None, value_for) -> None:
    """Reset a text box to `value_for(sample)` whenever the chosen audio changes (the sample's
    reference, or empty for your own audio)."""
    current = (chosen[2], chosen[3]) if chosen else None
    if st.session_state.get(f"{prefix}_{field}_for") != current:
        sample = chosen[3] if chosen and chosen[2] == "sample" else None
        st.session_state[f"{prefix}_{field}"] = (value_for(sample) if sample else "") or ""
        st.session_state[f"{prefix}_{field}_for"] = current


def sample_reference(sample: str) -> str:
    return asr_metrics.sample_reference(SAMPLES_DIR / sample) or ""


def ref_file(sample: str, suffix: str) -> str:
    """A bundled sample's reference, e.g. meeting_zh.rttm / meeting_zh.en.txt in speech_refs;
    an English reference can also be the sample's official translation next to it
    (national_day_2026_zh.wav -> samples/national_day_2026_en.txt)."""
    path = REFS_DIR / (Path(sample).stem + suffix)
    official = SAMPLES_DIR / (Path(sample).stem.removesuffix("_zh") + "_en.txt")
    if not path.exists() and suffix == ".en.txt" and official.exists():
        path = official
    try:
        return "\n".join(
            line for line in path.read_text().splitlines() if not line.startswith("#")
        ).strip()
    except OSError:
        return ""


def cached_audio(audio, filename: str) -> Path:
    """The chosen audio as a file in the cache (uploads and recordings are bytes)."""
    if isinstance(audio, Path):
        return audio
    digest = hashlib.sha1(audio).hexdigest()[:12]
    suffix = Path(filename).suffix.lower() or ".wav"
    path = CACHE_DIR / digest / f"original{suffix}"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(audio)
    return path


def cache_key(path: Path) -> str:
    """Short content hash of an audio file: the cache folder of its derived files."""
    h = hashlib.sha1()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()[:12]


# -- running jobs -----------------------------------------------------------------------------


def running_bar(job: jobs.Job) -> None:
    a, b = st.columns([5, 1])
    a.info(
        f"{job.detail or tr('準備中', 'Preparing')} · {job.elapsed():.0f}s",
        icon=":material/progress_activity:",
    )
    if b.button(tr("停止", "Stop"), key=f"stop_{job.id}", icon=":material/stop:"):
        job.cancel.set()


def saved_caption(state: dict) -> None:
    if state.get("path"):
        st.caption(tr(f"已儲存：`{state['path']}`", f"Saved: `{state['path']}`"))


def ollama_models(base_url: str) -> list[str]:
    from local_llm import ollama_admin

    try:
        return sorted(m["name"] for m in ollama_admin.installed(base_url))
    except Exception:
        return []
