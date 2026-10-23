"""Taiwanese-accented neural voices next to macOS's `say` (tts.py lists and renders them).

    MediaTek VITS（台灣）: MediaTek-Research/vits-mr-20250709 (Apache-2.0, 0.12 GB ONNX), a
        single Taiwanese Mandarin speaker through sherpa-onnx on the CPU, about 0.17 s per
        second of speech: fast enough for the live assistant. It reads Chinese characters only,
        so numbers are spelled out first (26°C -> 二十六度, 33% -> 百分之三十三) and English
        text falls back to macOS Samantha.

More models (Kokoro, Qwen3-TTS, VoxCPM2, Breeze-TTS-2) run through mlx-audio: mlx_tts.py
defines them and VOICES here lists them all. Every voice says where it comes from ("source").

All write 16-bit mono WAV at the sample rate asked for, a little louder than the macOS voices
(they came out quiet: VITS about 22 dB below `say`).

BreezyVoice (MediaTek, CosyVoice-based) was tried and removed (2026-10-22): PyTorch on the CPU
in its own environment, about 12x the audio length (68 s before the first sentence), and it
sometimes dropped or misread words. Its example speaker stays as the clone prompt
(data/lab/voices/taiwan_speaker.wav).
"""

import re
import threading
import wave
from functools import cache
from pathlib import Path

import numpy as np

from . import mlx_tts
from .asr_metrics import spell_numbers

PROJECT = Path(__file__).resolve().parents[2]
VITS_DIR = PROJECT / "models" / "speech_extra" / "vits-mr-20250709"
# The speaker the cloned "台灣腔" voices copy (mlx_tts.py) and its transcript.
PROMPT = PROJECT / "data" / "lab" / "voices" / "taiwan_speaker.wav"
PROMPT_TEXT = (
    "在密碼學中，加密是將明文資訊改變為難以讀取的密文內容，使之不可讀的方法。"
    "只有擁有解密方法的對象，經由解密過程，才能將密文還原為正常可讀的內容。"
)
VITS = "MediaTek VITS（台灣）"
VITS_LIMITS = (
    "只會念中文字：數字先轉成國字再念，整句英文改用 macOS Samantha，中文句子裡的英文字母"
    "（如 CNC）會被跳過；咬字較不清楚、每次念的長度不一（回聽錯字率約 6–10%），短句常吞字，"
    "不適合當測試講者。",
    "Chinese characters only: numbers are spelled out first, English sentences go to macOS "
    "Samantha, Latin letters inside Chinese (e.g. CNC) are skipped; less clear and uneven "
    "(6-10% round-trip errors), drops syllables of short lines, not a good test speaker.",
)
VOICES = {
    VITS: {
        "name": VITS,
        "short": "MediaTek VITS",
        "locale": "zh_TW",
        "engine": "vits",
        "sample": "我是聯發科的台灣華語語音。",
        "fast": True,
        "revision": "+v2",  # renderings before the steadier settings are not reused
        "test_speaker": False,  # drops syllables of short lines: a poor stand-in for the user
        "source": ("聯發科 VITS・sherpa-onnx", "MediaTek VITS, sherpa-onnx"),
        "limits": VITS_LIMITS,
    },
} | mlx_tts.VOICES
NATIVE_RATE = 22050
# Every rendering is brought to TARGET_RMS_DB (macOS Tingting measures about -17), its peaks
# rounded off above KNEE so the gain is not capped by a few loud samples.
TARGET_RMS_DB = -16.0
PEAK = 0.89  # -1 dBFS
KNEE = 0.6
_lock = threading.Lock()


def available(name: str) -> bool:
    if name == VITS:
        return (VITS_DIR / "vits-mr-20250709.onnx").exists()
    if name in mlx_tts.VOICES:
        return mlx_tts.available(name)
    return False


def voices(fast_only: bool = True) -> list[dict]:
    return [dict(v) for n, v in VOICES.items() if available(n) and (v["fast"] or not fast_only)]


_RANGE = re.compile(r"(\d+(?:\.\d+)?)\s*[–—~～-]\s*(\d+(?:\.\d+)?)")
_LATIN = re.compile(r"[A-Za-z]")
_CJK = re.compile(r"[一-鿿]")


def speech_text(text: str) -> str:
    """What these models can read: ranges, degrees and numbers spelled out in Chinese."""
    text = _RANGE.sub(r"\1到\2", text)
    text = re.sub(r"\s*(°C|℃|度C)", "度", text)
    text = text.replace("km", "公里").replace("公尺外", "公尺外")
    text = spell_numbers(text)
    return re.sub(r"(?<=[一-鿿，。、])\s+|\s+(?=[一-鿿，。、])", "", text).strip()


def english(text: str) -> bool:
    return len(_LATIN.findall(text)) > len(_CJK.findall(text))


@cache
def _vits():
    import sherpa_onnx

    # Less randomness than the defaults (0.667, 0.8): with them the same line came out
    # anywhere from 0.71 to 1.17 s and the short ones swallowed syllables (round trip on the
    # 16 test lines: 15.5% -> 6-10%).
    model = sherpa_onnx.OfflineTtsVitsModelConfig(
        model=str(VITS_DIR / "vits-mr-20250709.onnx"),
        lexicon=str(VITS_DIR / "lexicon.txt"),
        tokens=str(VITS_DIR / "tokens.txt"),
        noise_scale=0.5,
        noise_scale_w=0.3,
    )
    config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(vits=model, num_threads=4), max_num_sentences=1
    )
    return sherpa_onnx.OfflineTts(config)


def _resample(x: np.ndarray, rate: int, to: int) -> np.ndarray:
    if rate == to or not len(x):
        return x
    n = int(round(len(x) * to / rate))
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)


def limit(x: np.ndarray) -> np.ndarray:
    """Soft limiter: samples above KNEE bend smoothly towards PEAK, the rest is untouched."""
    a = np.abs(x)
    over = a > KNEE
    if not over.any():
        return x
    y = x.copy()
    room = PEAK - KNEE
    y[over] = np.sign(x[over]) * (KNEE + room * np.tanh((a[over] - KNEE) / room))
    return y


def loudness(x: np.ndarray) -> np.ndarray:
    """Gain to TARGET_RMS_DB over the voiced part, peaks softly limited below PEAK."""
    if not len(x):
        return x
    frame = 320
    n = len(x) // frame
    rms = np.sqrt((x[: n * frame].reshape(n, frame) ** 2).mean(axis=1)) if n else np.array([0.0])
    voiced = rms[rms > rms.max() * 0.1] if rms.max() > 0 else rms
    level = float(np.sqrt((voiced**2).mean())) if len(voiced) else 0.0
    if level <= 0:
        return x
    return limit(x * 10 ** (TARGET_RMS_DB / 20) / level)


def _write(path: Path, x: np.ndarray, rate: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    return path


def _read(path: Path) -> tuple[np.ndarray, int]:
    """A mono WAV as float samples: 16-bit PCM, or 32-bit float (what torchaudio writes)."""
    data, pos, fmt, rate, bits, samples = path.read_bytes(), 12, 1, NATIVE_RATE, 16, b""
    while pos + 8 <= len(data):
        cid, size = data[pos : pos + 4], int.from_bytes(data[pos + 4 : pos + 8], "little")
        body = data[pos + 8 : pos + 8 + size]
        if cid == b"fmt ":
            fmt = int.from_bytes(body[0:2], "little")
            rate = int.from_bytes(body[4:8], "little")
            bits = int.from_bytes(body[14:16], "little")
        elif cid == b"data":
            samples = body
        pos += 8 + size + (size & 1)
    if fmt == 3 or (fmt == 0xFFFE and bits == 32):
        return np.frombuffer(samples, dtype="<f4").astype(np.float32), rate
    return np.frombuffer(samples, dtype="<i2").astype(np.float32) / 32768, rate


def finish(x: np.ndarray, rate: int, fade_s: float = 0.03, pad_s: float = 0.3) -> np.ndarray:
    """A short fade-out and trailing silence: VITS stops mid-sound (the last frame is only
    about 10 dB below speech), which sounds cut off and leaves the recogniser no pause."""
    n = min(len(x), int(fade_s * rate))
    if n:
        x = x.copy()
        x[-n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32)
    return np.concatenate([x, np.zeros(int(pad_s * rate), np.float32)])


def render_vits(text: str, out: Path, sample_rate: int) -> Path:
    with _lock:  # one sherpa-onnx engine, used from several threads
        audio = _vits().generate(speech_text(text), sid=0, speed=1.0)
    x = finish(loudness(np.asarray(audio.samples, dtype=np.float32)), audio.sample_rate)
    return _write(out, _resample(x, audio.sample_rate, sample_rate), sample_rate)


def render_mlx(text: str, voice: str, out: Path, sample_rate: int) -> Path:
    x, rate = mlx_tts.generate(speech_text(text), VOICES[voice])
    return _write(out, _resample(loudness(x), rate, sample_rate), sample_rate)


def render(text: str, voice: str, out: Path, sample_rate: int) -> Path:
    """One line with a neural voice (English lines with macOS Samantha unless the model
    speaks English too)."""
    if english(text) and not VOICES[voice].get("english"):
        from . import tts

        return tts.synthesize(text, "Samantha", None, out=out, sample_rate=sample_rate)
    if VOICES[voice]["engine"] == "mlx":
        return render_mlx(text, voice, out, sample_rate)
    return render_vits(text, out, sample_rate)
