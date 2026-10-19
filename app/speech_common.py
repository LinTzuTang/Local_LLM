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
# Sample recordings with a reference script next to them (<name>.txt); see samples/README.md.
# Most are bundled in data/lab/samples; LOCAL_SAMPLES are built on this machine only
# (licence forbids derived files in the repo) by scripts/make_speech_samples.py into
# storage/speech_samples, and are listed only when the file exists.
SAMPLES = [
    "national_day_2026_zh.wav",
    "meeting_zh.wav",
    "alimeeting_r8001_zh.flac",
    "palau_hsiao_2026_en.wav",
    "librispeech_20_en.flac",
    "hokkien_moe_20_nan.flac",
]
LOCAL_SAMPLES = {"hokkien_moe_20_nan.flac"}
# Babble noise mixes in other speech; kept to the original recordings so results stay
# comparable with earlier runs.
BABBLE_SAMPLES = SAMPLES[:4]
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


def sample_path(name: str) -> Path:
    """Where a sample lives: data/lab/samples, or storage/speech_samples when built locally."""
    folder = asr_metrics.LOCAL_SAMPLES_DIR if name in LOCAL_SAMPLES else SAMPLES_DIR
    return folder / name


def available_samples() -> list[str]:
    """The samples to offer: every bundled one, local ones only when they have been built."""
    return [s for s in SAMPLES if s not in LOCAL_SAMPLES or sample_path(s).exists()]


def sample_language(name: str) -> str:
    """Language hint for a sample: "en" for English ones, else "zh". Taiwanese Hokkien gets
    "zh" too: none of the installed models has a Hokkien option, and Chinese is closest."""
    return "en" if Path(name).stem.endswith("_en") else "zh"


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
        "palau_hsiao_2026_en.wav": tr(
            "英文：副總統帛琉英語致詞（4 分 59 秒，台灣口音）",
            "English: Vice President's speech in Palau (4 min 59 s, Taiwanese accent)",
        ),
        "alimeeting_r8001_zh.flac": tr(
            "真實會議：AliMeeting 4 人會議（5 分鐘，遠場麥克風，大陸口音）",
            "Real meeting: AliMeeting, 4 speakers (5 min, far-field mic, mainland accent)",
        ),
        "librispeech_20_en.flac": tr(
            "英文有聲書：LibriSpeech 20 句（3 分 6 秒，朗讀）",
            "English audiobook: LibriSpeech, 20 sentences (3 min 6 s, read aloud)",
        ),
        "hokkien_moe_20_nan.flac": tr(
            "台語：教育部台語辭典例句 20 句（1 分 47 秒，只在本機）",
            "Taiwanese Hokkien: MOE dictionary, 20 example sentences (1 min 47 s, local only)",
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
        "palau_hsiao_2026_en.wav": tr(
            "〔總統府提供〕蕭副總統在帛琉歡迎晚宴的英語致詞，原稿取自總統府英文官網全文，"
            "依實際口述修改 3 處；英文以詞錯誤率（WER）評分。來源與授權見 "
            "data/lab/samples/README.md。",
            "Courtesy of the Office of the President: Vice President Hsiao's English remarks "
            "at the Palau welcome banquet; the reference is the official English text with 3 "
            "edits where she deviated. English is scored by word error rate (WER). Source and "
            "licence: data/lab/samples/README.md.",
        ),
        "alimeeting_r8001_zh.flac": tr(
            "AliMeeting 評測集 R8001_M8004 前 5 分鐘（第 1 聲道），4 人圍桌開會、會搶話。"
            "逐字稿與講者時間取自資料集；CC BY-SA 4.0，來源見 data/lab/samples/README.md。",
            "AliMeeting Eval R8001_M8004, first 5 minutes (channel 1): 4 people around a table, "
            "with overlapping speech. Transcript and speaker turns from the dataset; CC BY-SA 4.0, "
            "see data/lab/samples/README.md.",
        ),
        "librispeech_20_en.flac": tr(
            "LibriSpeech test-clean（OpenSLR 12）裡英文評測用的 20 句，句間 0.8 秒靜音接成一段；"
            "錄音來自 LibriVox 有聲書。CC BY 4.0，標示「LibriSpeech (Panayotov et al., 2015)」；"
            "英文以詞錯誤率（WER）評分。來源見 data/lab/samples/README.md。",
            "The 20 LibriSpeech test-clean sentences (OpenSLR 12) of the English evaluation, "
            "joined with 0.8 s of silence; LibriVox audiobook recordings. CC BY 4.0, credit "
            '"LibriSpeech (Panayotov et al., 2015)"; English is scored by word error rate '
            "(WER). Source: data/lab/samples/README.md.",
        ),
        "hokkien_moe_20_nan.flac": tr(
            "教育部《臺灣台語常用詞辭典》例句 20 句，參考答案可選辭典的台語漢字或華語翻譯。"
            "授權 CC BY-ND 3.0 TW"
            "（禁止改作）：接成一段算改作，所以只用 scripts/make_speech_samples.py 在本機產生到 "
            "storage/（不進 repo），請勿轉散布。模型沒有台語選項，語言用最接近的「中文」；多數模型"
            "會寫成華語，字錯率會很高。",
            "20 example sentences of the MOE Dictionary of Frequently-Used Taiwan Taigi; the "
            "reference is the dictionary's Hokkien 漢字 or its Mandarin translation. Licence "
            "CC BY-ND 3.0 TW (no "
            "derivatives): a joined file counts as a derivative, so it is only built locally "
            "by scripts/make_speech_samples.py into storage/ (not in the repo); do not share "
            "it. The models have no Hokkien option, so the closest hint, Chinese, is used; most "
            "models write Mandarin, so error rates are high.",
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
        options = available_samples()
        st.session_state.setdefault(f"{prefix}_sample", default)
        if st.session_state[f"{prefix}_sample"] not in options:
            st.session_state[f"{prefix}_sample"] = default
        sample = st.selectbox(
            tr("範例錄音", "Sample recording"),
            options,
            format_func=sample_label,
            **kept(f"{prefix}_sample"),
        )
        path = sample_path(sample)
        st.audio(str(path))
        st.caption(f"`{path.relative_to(PROJECT_DIR)}` · " + sample_note(sample))
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


def sample_reference(sample: str, variant: str | None = None) -> str:
    return asr_metrics.sample_reference(sample_path(sample), variant=variant) or ""


# Models that write Taiwanese Hokkien speech as Mandarin: scored against the Mandarin
# translation by default (the Hokkien sample's .zh.txt reference).
MANDARIN_FOR_HOKKIEN = {"breeze-asr-26"}


def reference_variant(prefix: str, chosen: tuple | None, models: list[str]) -> None:
    """For a sample with a second reference (the Hokkien sentences' Mandarin translation,
    <name>.zh.txt), a radio choosing which one fills the `<prefix>_reference` box: the
    dictionary's 漢字 or its Mandarin translation, the latter by default when a model that
    writes Mandarin (Breeze-ASR-26) is chosen."""
    sample = chosen[3] if chosen and chosen[2] == "sample" else None
    key = f"{prefix}_ref_variant"
    if not sample or not sample_reference(sample, "zh"):
        st.session_state.pop(f"{key}_for", None)  # coming back refills the reference
        return
    default = "zh" if MANDARIN_FOR_HOKKIEN & set(models) else "nan"
    if st.session_state.get(f"{key}_default") != (sample, default):
        st.session_state[key] = default  # a new sample or model choice resets the default
        st.session_state[f"{key}_default"] = (sample, default)
    options = {
        "nan": tr("台語漢字（辭典原文）", "Hokkien 漢字 (the dictionary's text)"),
        "zh": tr("華語翻譯（辭典對譯）", "Mandarin translation (the dictionary's)"),
    }
    variant = st.radio(
        tr("參考答案", "Reference"),
        list(options),
        format_func=options.get,
        horizontal=True,
        help=tr(
            "Breeze-ASR-26 等模型把台語寫成華語，和台語漢字比對錯誤率會偏高；改用辭典的華語翻譯比對"
            "較公平。換這個選項會重填下面的參考原稿。",
            "Breeze-ASR-26 and similar models write Hokkien speech as Mandarin, so scoring "
            "them against the Hokkien 漢字 overstates their errors; the dictionary's Mandarin "
            "translation is the fairer reference for them. Changing this refills the reference "
            "box below.",
        ),
        **kept(key),
    )
    if st.session_state.get(f"{key}_for") != (sample, variant):
        st.session_state[f"{prefix}_reference"] = sample_reference(
            sample, None if variant == "nan" else variant
        )
        st.session_state[f"{key}_for"] = (sample, variant)


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
