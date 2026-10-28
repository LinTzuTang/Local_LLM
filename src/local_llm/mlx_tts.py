"""Neural text-to-speech models on Apple silicon through mlx-audio (models/speech_extra/tts/).

    from local_llm import mlx_tts
    samples, rate = mlx_tts.generate("各位早安。", mlx_tts.VOICES["Qwen3-TTS Vivian"])

neural_tts.py lists these voices next to its own and renders them through `generate`. Each
voice is one model plus its options:

    Kokoro-82M (hexgrad, Apache-2.0, 0.3 GB): tiny and very fast, Mainland Mandarin speakers
        (misaki[zh] reads the text).
    Qwen3-TTS 0.6B (Alibaba, Apache-2.0, 2.5 GB): CustomVoice has preset speakers; Base
        clones a voice from a few seconds of reference audio and its transcript.
    VoxCPM2 (OpenBMB, Apache-2.0, 2B, 8-bit 3.2 GB): tokenizer-free, clones from reference
        audio, 48 kHz.
    Breeze-TTS-2 (BreezeBlue, research and non-commercial licence, 7.6 GB): Qwen3-TTS codec,
        clones from reference audio.

The cloned voices copy a Taiwanese speaker (data/lab/voices/taiwan_speaker.wav, BreezyVoice's
example speaker), so they speak with a Taiwanese accent. MLX arrays belong to the thread
that made them, so one worker thread loads the models and renders every line; at most
MAX_LOADED models stay in memory (least recently used first out) and all are dropped after
IDLE_S without use.
"""

import queue
import threading
import time
from collections import OrderedDict
from collections.abc import Iterator
from concurrent.futures import Future
from functools import cache
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT / "models" / "speech_extra" / "tts"
MAX_LOADED = 2
IDLE_S = 900.0

# model folder -> (display name, repository, licence, size in GB)
MODELS = {
    "kokoro-82m": ("Kokoro-82M", "mlx-community/Kokoro-82M-bf16", "Apache-2.0", 0.3),
    "qwen3-tts-0.6b-customvoice": ("Qwen3-TTS 0.6B CustomVoice",
                                   "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16",
                                   "Apache-2.0", 2.5),
    "qwen3-tts-0.6b-base": ("Qwen3-TTS 0.6B Base",
                            "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16", "Apache-2.0", 2.5),
    "voxcpm2-8bit": ("VoxCPM2 8-bit", "mlx-community/VoxCPM2-8bit", "Apache-2.0", 3.2),
    "breeze-tts-2": ("Breeze-TTS-2", "mlx-community/Breeze-TTS-2-mlx",
                     "research / non-commercial", 7.6),
    "moss-tts-nano": ("MOSS-TTS-Nano 100M", "mlx-community/MOSS-TTS-Nano-100M", "Apache-2.0", 0.3),
    "omnivoice": ("OmniVoice", "mlx-community/OmniVoice-bfloat16", "Apache-2.0", 2.0),
}  # fmt: skip
CLONE = "clone"  # options marker: reference audio and transcript from neural_tts


def voice(name: str, short: str, locale: str, model: str, source: tuple[str, str],
          sample: str, english: bool = False, fast: bool = False, simplified: bool = False,
          stream: bool = False,
          limits: tuple[str, str] = ("", ""), **options) -> dict:  # fmt: skip
    """`fast`: quick enough to talk live (measured, scripts/speech_eval/run_tts_bench.py);
    `simplified`: the model mispronounces Traditional characters, so it reads the text
    converted to Simplified (Qwen3-TTS: 「客戶」 came out as 「撤務」, CER 22.6% -> 0%);
    `limits`: (Chinese, English) what it cannot do, shown where voices are picked."""
    return {"name": name, "short": short, "locale": locale, "engine": "mlx", "model": model,
            "options": options, "source": source, "sample": sample, "english": english,
            "fast": fast, "simplified": simplified, "limits": limits,
            "stream": stream}  # fmt: skip


KOKORO_LIMITS = (
    "只能念中文（大陸口音）：數字 OK；整句英文改用 macOS Samantha 念，中文句子裡的英文字母（如 CN"
    "C）會被跳過。",
    "Chinese only (Mainland accent): numbers are fine; English sentences go to macOS Samantha, "
    "and Latin letters inside Chinese (e.g. CNC) are skipped.",
)
QWEN_LIMITS = (
    "中英文都能念（官方支援 10 種語言），數字、中英夾雜 OK；繁體字會走音，所以先轉簡體再念；"
    "整句合成每句約 1.2–1.7 秒，助理已改成邊合成邊播，約 0.2 秒開口。",
    "Chinese and English (10 languages), numbers and mixed text are fine; Traditional "
    "characters are mispronounced, so it reads Simplified; 1.2-1.7 s per whole sentence, "
    "but the assistant plays it while it is made and starts in about 0.2 s.",
)
MOSS_LIMITS = (
    "很快（每句約 0.6 秒），但常把句子拉長、重複（88 秒的稿念成 183 秒），"
    "測試台詞辨識錯字率約 45%，不建議使用；繁體字要先轉簡體。",
    "Fast (about 0.6 s a sentence) but often stretches and repeats lines (an 88 s "
    "script took 183 s), 45% errors on the test lines: not recommended; reads Simplified.",
)
OMNI_LIMITS = (
    "中英文、數字都行；繁體字會走音，所以先轉簡體再念；每句約 1.5 秒才開口，即時對話稍慢；"
    "測試台詞串流辨識全對，很適合當測試講者。",
    "Chinese, English and numbers are fine; Traditional characters are mispronounced, so it "
    "reads Simplified; about 1.5 s before each sentence, a little slow to talk live; every test "
    "line was recognised right, a good test speaker.",
)
SLOW_LIMITS = {
    "voxcpm2-8bit": (
        "多語言（30 種），中英夾雜、數字 OK；繁體字要先轉簡體；很慢（每句約 3.6 秒），"
        "只適合預先合成。",
        "30 languages, mixed text and numbers are fine; reads Simplified; slow (about 3.6 s "
        "a sentence), pre-render only.",
    ),
    "breeze-tts-2": (
        "中英文、數字、中英夾雜都 OK；最慢（每句約 5 秒，合成比說話還慢），只適合預先合成；"
        "只限研究、非商業使用。",
        "Chinese, English, numbers and mixed text are fine; slowest (about 5 s a sentence, "
        "slower than speech), pre-render only; research and non-commercial use only.",
    ),
}


VOICES = {
    v["name"]: v
    for v in [
        voice("Kokoro 曉北", "Kokoro 曉北", "zh_CN", "kokoro-82m",
              ("Kokoro-82M・MLX", "Kokoro-82M, MLX"), "我是 Kokoro 的中文女聲。",
              fast=True, limits=KOKORO_LIMITS, voice="zf_xiaobei", lang_code="z"),
        voice("Kokoro 雲希", "Kokoro 雲希", "zh_CN", "kokoro-82m",
              ("Kokoro-82M・MLX", "Kokoro-82M, MLX"), "我是 Kokoro 的中文男聲。",
              fast=True, limits=KOKORO_LIMITS, voice="zm_yunxi", lang_code="z"),
        voice("Qwen3-TTS Vivian", "Qwen3-TTS Vivian", "zh_CN", "qwen3-tts-0.6b-customvoice",
              ("Qwen3-TTS 0.6B 內建聲音・MLX", "Qwen3-TTS 0.6B preset, MLX"),
              "我是 Qwen3-TTS 的 Vivian。", english=True, simplified=True, stream=True,
              limits=QWEN_LIMITS,
              voice="vivian",
              lang_code="chinese"),
        voice("Qwen3-TTS Uncle Fu", "Qwen3-TTS Uncle Fu", "zh_CN", "qwen3-tts-0.6b-customvoice",
              ("Qwen3-TTS 0.6B 內建聲音・MLX", "Qwen3-TTS 0.6B preset, MLX"),
              "我是 Qwen3-TTS 的 Uncle Fu。", english=True, simplified=True, stream=True,
              limits=QWEN_LIMITS,
              voice="uncle_fu",
              lang_code="chinese"),
        voice("Qwen3-TTS 台灣腔（複製）", "Qwen3-TTS 台灣腔", "zh_TW", "qwen3-tts-0.6b-base",
              ("Qwen3-TTS 0.6B 聲音複製・MLX", "Qwen3-TTS 0.6B clone, MLX"),
              "我是複製台灣口音的 Qwen3-TTS。", english=True, simplified=True, stream=True,
              limits=QWEN_LIMITS,
              lang_code="chinese", clone=CLONE),
        voice("VoxCPM2 台灣腔（複製）", "VoxCPM2 台灣腔", "zh_TW", "voxcpm2-8bit",
              ("VoxCPM2 聲音複製・MLX", "VoxCPM2 clone, MLX"),
              "我是複製台灣口音的 VoxCPM2。", english=True, simplified=True,
              limits=SLOW_LIMITS["voxcpm2-8bit"], clone=CLONE),
        voice("MOSS-TTS-Nano 台灣腔（複製）", "MOSS-TTS-Nano 台灣腔", "zh_TW", "moss-tts-nano",
              ("MOSS-TTS-Nano 100M 聲音複製・MLX", "MOSS-TTS-Nano 100M clone, MLX"),
              "我是複製台灣口音的 MOSS-TTS-Nano。", english=True, simplified=True,
              limits=MOSS_LIMITS, clone=CLONE),
        voice("OmniVoice 台灣腔（複製）", "OmniVoice 台灣腔", "zh_TW", "omnivoice",
              ("OmniVoice 聲音複製・MLX", "OmniVoice clone, MLX"),
              "我是複製台灣口音的 OmniVoice。", english=True, simplified=True,
              limits=OMNI_LIMITS,
              language="zh", clone=CLONE),
        # Spark-TTS 0.5B was tried and left out: its clone fails in mlx-audio (BatchNorm) and
        # long lines time the GPU out.
        voice("Breeze-TTS-2 台灣腔（複製）", "Breeze-TTS-2 台灣腔", "zh_TW", "breeze-tts-2",
              ("Breeze-TTS-2 聲音複製・MLX・非商用", "Breeze-TTS-2 clone, MLX, non-commercial"),
              "我是複製台灣口音的 Breeze-TTS-2。", english=True,
              limits=SLOW_LIMITS["breeze-tts-2"], clone=CLONE),
    ]
}  # fmt: skip


def installed(model: str) -> bool:
    folder = MODELS_DIR / model
    return (folder / "config.json").exists() and any(folder.glob("*.safetensors"))


def available(name: str) -> bool:
    v = VOICES.get(name)
    if v is None or not installed(v["model"]):
        return False
    if v["model"] == "kokoro-82m":
        try:
            import misaki.zh  # noqa: F401  (Kokoro's Mandarin front end)
        except ImportError:
            return False
    return True


# -- the worker thread ------------------------------------------------------------------------

_jobs: queue.Queue = queue.Queue()
_started = threading.Lock()
_thread: threading.Thread | None = None


def _options(v: dict) -> dict:
    options = dict(v["options"])
    if options.pop("clone", None) == CLONE:
        from . import neural_tts

        options["ref_audio"] = str(neural_tts.PROMPT)
        options["ref_text"] = neural_tts.PROMPT_TEXT
    return options


@cache
def _converter():
    import opencc

    return opencc.OpenCC("t2s")


def _to_simplified(text: str) -> str:
    return _converter().convert(text)


STREAM_INTERVAL_S = 0.48  # seconds of audio per streamed chunk (Qwen3-TTS: 6 codec frames)


def _render(models: OrderedDict, v: dict, text: str,
            sink: queue.Queue | None = None) -> tuple[np.ndarray, int]:  # fmt: skip
    """Render `text`; with a `sink`, stream: every chunk goes there as it is made
    ((samples, rate)), for the voices whose model streams (`stream` in VOICES)."""
    from mlx_audio.tts.utils import load_model

    model = models.get(v["model"])
    if model is None:
        while len(models) >= MAX_LOADED:
            models.popitem(last=False)
        model = load_model(str(MODELS_DIR / v["model"]))
        models[v["model"]] = model
    models.move_to_end(v["model"])
    if v.get("simplified"):
        text = _to_simplified(text)
    parts, rate, options = [], 24000, _options(v)
    if sink is not None:
        options |= {"stream": True, "streaming_interval": STREAM_INTERVAL_S}
    for result in model.generate(text=text, **options):
        chunk = np.asarray(result.audio, dtype=np.float32).reshape(-1)
        rate = int(result.sample_rate)
        if sink is not None:
            sink.put((chunk, rate))
        parts.append(chunk)
    return (np.concatenate(parts) if parts else np.zeros(0, np.float32)), rate


def _work() -> None:
    import gc

    models: OrderedDict = OrderedDict()
    while True:
        try:
            item = _jobs.get(timeout=IDLE_S)
        except queue.Empty:
            if models:
                models.clear()
                gc.collect()
            continue
        v, text, future, sink = item
        if not future.set_running_or_notify_cancel():
            continue
        try:
            future.set_result(_render(models, v, text, sink))
        except Exception as exc:  # shown where the line was asked for
            future.set_exception(exc)
        finally:
            if sink is not None:
                sink.put(None)


def _submit(v: dict, text: str, sink: queue.Queue | None = None) -> Future:
    global _thread
    with _started:
        if _thread is None or not _thread.is_alive():
            _thread = threading.Thread(target=_work, name="mlx-tts", daemon=True)
            _thread.start()
    future: Future = Future()
    _jobs.put((v, text, future, sink))
    return future


def generate(text: str, v: dict, timeout: float = 600) -> tuple[np.ndarray, int]:
    """(float samples, sample rate) of `text` in voice `v`, rendered on the worker thread."""
    return _submit(v, text).result(timeout=timeout)


def stream(text: str, v: dict, timeout: float = 600) -> Iterator[tuple[np.ndarray, int]]:
    """`text` in voice `v` as (float samples, sample rate) chunks while they are made, so
    playback can start after the first one (voices with `stream`); else one whole chunk."""
    if not v.get("stream"):
        yield generate(text, v, timeout)
        return
    sink: queue.Queue = queue.Queue()
    future = _submit(v, text, sink)
    while (item := sink.get(timeout=timeout)) is not None:
        yield item
    future.result(timeout=timeout)  # raises what the render raised


def warm(v: dict) -> float:
    """Load the voice's model and render one short line; returns the seconds it took."""
    started = time.perf_counter()
    generate("你好。", v)
    return time.perf_counter() - started
