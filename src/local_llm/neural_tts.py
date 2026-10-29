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
# At its own pace VITS reads about 1.5x faster than macOS Tingting (a long line in 2.4 s
# against 3.9 s); 0.65 brings it close, as clear as before (round trip 9.5% vs 10.0%).
VITS_SPEED = 0.65
VITS_LIMITS = (
    "只會念中文字：數字先轉成國字再念，整句英文改用 macOS Samantha，中文句子裡的英文字母"
    "（如 CNC）會被跳過；咬字較不清楚、每次念的長度不一（回聽錯字率約 11%，測試台詞約 17%），"
    "短句常吞字，不適合當測試講者。",
    "Chinese characters only: numbers are spelled out first, English sentences go to macOS "
    "Samantha, Latin letters inside Chinese (e.g. CNC) are skipped; less clear and uneven "
    "(about 11% round-trip errors, 17% on the test lines), drops syllables of short lines, "
    "not a good test speaker.",
)
VOICES = {
    VITS: {
        "name": VITS,
        "short": "MediaTek VITS",
        "locale": "zh_TW",
        "engine": "vits",
        "sample": "我是聯發科的台灣華語語音。",
        "fast": True,
        "revision": "+v3",  # renderings before the steadier, slower settings are not reused
        "test_speaker": False,  # drops syllables of short lines: a poor stand-in for the user
        "source": ("聯發科 VITS・sherpa-onnx", "MediaTek VITS, sherpa-onnx"),
        "limits": VITS_LIMITS,
    },
}
# Google Gemini TTS on Vertex AI (the service account of gcp_auth): text goes to Google.
GEMINI_TTS_LIMITS = (
    "雲端（Google Vertex AI）：要能連網，要念的文字會上傳；整句生成完才回傳，"
    "加上網路往返，每句要等數秒。中英文、數字都行；口音偏大陸。",
    "Cloud (Google Vertex AI): needs the network and the text is uploaded; it returns the "
    "whole sentence at once, so with the round trip each sentence takes seconds. Chinese, "
    "English and numbers are fine; Mainland accent.",
)
GEMINI_TTS = {
    "Gemini 2.5 Flash TTS（雲端）": "gemini-2.5-flash-tts",
    "Gemini 2.5 Pro TTS（雲端）": "gemini-2.5-pro-tts",
    "Gemini 3.1 Flash TTS（雲端）": "gemini-3.1-flash-tts-preview",
}
for _name, _model in GEMINI_TTS.items():
    VOICES[_name] = {
        "name": _name, "short": _name.removesuffix("（雲端）"), "locale": "zh_CN",
        "engine": "gemini", "model": _model, "voice": "Kore", "fast": False, "english": True,
        "sample": "我是 Google 的 Gemini 語音。", "cloud": True,
        "source": ("Google Gemini TTS・雲端", "Google Gemini TTS, cloud"),
        "limits": GEMINI_TTS_LIMITS,
    }  # fmt: skip
# Azure AI Speech neural voices (Taiwan): the key of AZURE_SPEECH_KEY, or AZURE_OPENAI_API_KEY
# when that resource is multi-service, in AZURE_SPEECH_REGION (default eastus2). The REST API
# sends the audio in pieces while it is made, so these voices stream (render_stream).
AZURE_TTS_LIMITS = (
    "雲端（Azure AI Speech，美東區）：要能連網，要念的文字會上傳；邊合成邊回傳，連線建立後"
    "第一段聲音約 0.3 秒（每次開始的第一句約 1–2 秒）。台灣口音，中英文、數字都行；"
    "依字數付費（有免費額度）。",
    "Cloud (Azure AI Speech, East US 2): needs the network and the text is uploaded; audio "
    "streams back while it is made, the first piece in about 0.3 s once connected (1-2 s for "
    "the very first line). Taiwanese accent; Chinese, English and numbers are fine; billed per "
    "character (free tier exists).",
)
AZURE_TTS = {
    "Azure 曉臻（雲端）": "zh-TW-HsiaoChenNeural",
    "Azure 雲哲（雲端）": "zh-TW-YunJheNeural",
    "Azure 曉雨（雲端）": "zh-TW-HsiaoYuNeural",
}
for _name, _voice in AZURE_TTS.items():
    VOICES[_name] = {
        "name": _name, "short": _name.removesuffix("（雲端）"), "locale": "zh_TW",
        "engine": "azure", "voice": _voice, "fast": False, "stream": True, "english": True,
        "sample": "我是 Azure 的台灣語音。", "cloud": True,
        "source": ("Azure AI Speech・雲端", "Azure AI Speech, cloud"),
        "limits": AZURE_TTS_LIMITS,
    }  # fmt: skip
# Google Cloud Text-to-Speech (REST text:synthesize, whole line at once): Taiwan Mandarin has
# Standard and WaveNet voices only; the newer Chirp 3 HD voices are Mainland Mandarin. The key
# is gcp_auth.tts_key_path() (GOOGLE_TTS_CREDENTIALS, else the default service account).
GOOGLE_TTS_LIMITS = (
    "雲端（Google Cloud Text-to-Speech）：要能連網，要念的文字會上傳；整句合成完才回傳。"
    "台灣華語只有 WaveNet／Standard 聲音（較舊），最新的 Chirp 3 HD 只有大陸口音。"
    "中英文、數字都行；"
    "依字數付費。",
    "Cloud (Google Cloud Text-to-Speech): needs the network and the text is uploaded; returns "
    "the whole line at once. Taiwan Mandarin has only the older WaveNet / Standard voices; the "
    "newest Chirp 3 HD voices are Mainland Mandarin. Chinese, English and numbers are fine; "
    "billed per character.",
)
GOOGLE_TTS = {
    "Google 台灣 WaveNet A（雲端）": ("cmn-TW-Wavenet-A", "zh_TW"),
    "Google 台灣 WaveNet B（雲端）": ("cmn-TW-Wavenet-B", "zh_TW"),
    "Google 台灣 WaveNet C（雲端）": ("cmn-TW-Wavenet-C", "zh_TW"),
    "Google Chirp 3 HD Aoede（雲端）": ("cmn-CN-Chirp3-HD-Aoede", "zh_CN"),
}
for _name, (_voice, _locale) in GOOGLE_TTS.items():
    VOICES[_name] = {
        "name": _name, "short": _name.removesuffix("（雲端）"), "locale": _locale,
        "engine": "google", "voice": _voice, "fast": True, "english": True,  # 0.5-0.8 s
        "sample": "我是 Google 的雲端語音。", "cloud": True,
        "source": ("Google Cloud TTS・雲端", "Google Cloud TTS, cloud"),
        "limits": GOOGLE_TTS_LIMITS,
    }  # fmt: skip
VOICES |= mlx_tts.VOICES
NATIVE_RATE = 22050
# Every rendering is brought to TARGET_RMS_DB (macOS Tingting measures about -17), its peaks
# rounded off above KNEE so the gain is not capped by a few loud samples.
TARGET_RMS_DB = -16.0
PEAK = 0.89  # -1 dBFS
KNEE = 0.6
_lock = threading.Lock()


def azure_speech() -> tuple[str, str] | None:
    """(region, key) for Azure AI Speech from the environment or the project's .env."""
    import os

    env = dict(os.environ)
    dotenv = PROJECT / ".env"
    if dotenv.exists():
        for line in dotenv.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                env.setdefault(k.strip(), v.strip().strip("'\""))
    key = env.get("AZURE_SPEECH_KEY") or env.get("AZURE_OPENAI_API_KEY")
    return (env.get("AZURE_SPEECH_REGION") or "eastus2", key) if key else None


def available(name: str) -> bool:
    if name in GOOGLE_TTS:
        from . import gcp_auth

        return gcp_auth.available(gcp_auth.tts_key_path())
    if name in AZURE_TTS:
        return azure_speech() is not None
    if name in GEMINI_TTS:
        from . import gcp_auth

        return gcp_auth.available()
    if name == VITS:
        return (VITS_DIR / "vits-mr-20250709.onnx").exists()
    if name in mlx_tts.VOICES:
        return mlx_tts.available(name)
    return False


def voices(fast_only: bool = True) -> list[dict]:
    """`fast_only`: voices quick enough to talk live: fast ones, and the streaming ones
    (Qwen3-TTS: about 0.2 s to the first chunk, synthesis faster than speech)."""
    live = lambda v: v["fast"] or v.get("stream")  # noqa: E731
    return [dict(v) for n, v in VOICES.items() if available(n) and (live(v) or not fast_only)]


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
        audio = _vits().generate(speech_text(text), sid=0, speed=VITS_SPEED)
    x = finish(loudness(np.asarray(audio.samples, dtype=np.float32)), audio.sample_rate)
    return _write(out, _resample(x, audio.sample_rate, sample_rate), sample_rate)


def render_mlx(text: str, voice: str, out: Path, sample_rate: int) -> Path:
    x, rate = mlx_tts.generate(speech_text(text), VOICES[voice])
    return _write(out, _resample(loudness(x), rate, sample_rate), sample_rate)


def render_gemini(text: str, voice: str, out: Path, sample_rate: int) -> Path:
    """One line through Gemini TTS (Vertex AI generateContent, audio out, 24 kHz PCM)."""
    import base64

    import httpx

    from . import gcp_auth

    v = VOICES[voice]
    token, project = gcp_auth.token()
    url = (f"https://us-central1-aiplatform.googleapis.com/v1/projects/{project}/locations/"
           f"us-central1/publishers/google/models/{v['model']}:generateContent")  # fmt: skip
    body = {"contents": [{"role": "user", "parts": [{"text": text}]}],
            "generationConfig": {"responseModalities": ["AUDIO"], "speechConfig": {
                "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": v["voice"]}}}}}  # fmt: skip
    r = httpx.post(url, json=body, headers={"Authorization": f"Bearer {token}"}, timeout=120)
    r.raise_for_status()
    part = r.json()["candidates"][0]["content"]["parts"][0]["inlineData"]
    match = re.search(r"rate=(\d+)", part.get("mimeType", ""))
    rate = int(match.group(1)) if match else 24000
    x = np.frombuffer(base64.b64decode(part["data"]), dtype="<i2").astype(np.float32) / 32768
    return _write(out, _resample(loudness(x), rate, sample_rate), sample_rate)


AZURE_RATE = 24000


@cache
def _azure_client():
    """One connection kept open between lines: a new TLS handshake to the region costs a few
    hundred milliseconds per sentence."""
    import httpx

    return httpx.Client(timeout=60)


def azure_chunks(text: str, voice: str):
    """Float audio pieces (24 kHz) of one line from Azure AI Speech, as they arrive."""
    from xml.sax.saxutils import escape

    region, key = azure_speech()
    v = VOICES[voice]
    ssml = (f'<speak version="1.0" xml:lang="zh-TW"><voice name="{v["voice"]}">'
            f"{escape(text)}</voice></speak>")  # fmt: skip
    head = {
        "Ocp-Apim-Subscription-Key": key,
        "Content-Type": "application/ssml+xml",
        "X-Microsoft-OutputFormat": "raw-24khz-16bit-mono-pcm",
        "User-Agent": "local-llm",
    }
    url = f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"
    rest = b""
    with _azure_client().stream("POST", url, content=ssml.encode(), headers=head) as r:
        r.raise_for_status()
        for data in r.iter_bytes(4800):
            data, rest = rest + data, b""
            if len(data) % 2:
                data, rest = data[:-1], data[-1:]
            if data:
                yield np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768


def render_azure(text: str, voice: str, out: Path, sample_rate: int) -> Path:
    x = np.concatenate(list(azure_chunks(text, voice)) or [np.zeros(1, np.float32)])
    return _write(out, _resample(loudness(x), AZURE_RATE, sample_rate), sample_rate)


def render_google(text: str, voice: str, out: Path, sample_rate: int) -> Path:
    """One line through Google Cloud Text-to-Speech (24 kHz LINEAR16, a WAV file)."""
    import base64
    import io

    from . import gcp_auth

    v = VOICES[voice]
    token, project = gcp_auth.token(gcp_auth.tts_key_path())
    body = {"input": {"text": text},
            "voice": {"languageCode": "-".join(v["voice"].split("-")[:2]), "name": v["voice"]},
            "audioConfig": {"audioEncoding": "LINEAR16", "sampleRateHertz": 24000}}  # fmt: skip
    head = {"Authorization": f"Bearer {token}", "x-goog-user-project": project}
    r = _google_client().post("https://texttospeech.googleapis.com/v1/text:synthesize",
                              json=body, headers=head)  # fmt: skip
    r.raise_for_status()
    with wave.open(io.BytesIO(base64.b64decode(r.json()["audioContent"]))) as w:
        rate = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768
    return _write(out, _resample(loudness(x), rate, sample_rate), sample_rate)


@cache
def _google_client():
    import httpx

    return httpx.Client(timeout=60)  # one connection kept open, as for Azure


def can_stream(voice: str, text: str) -> bool:
    v = VOICES.get(voice) or {}
    return bool(v.get("stream")) and not (english(text) and not v.get("english"))


def render_stream(text: str, voice: str, sample_rate: int):
    """`text` as 16-bit WAV chunks while the model makes them (voices with `stream`): the
    first one comes in about 0.2 s instead of after the whole sentence. One gain for the whole
    line, set from the first chunk, so the level does not jump between chunks."""
    import io
    import wave

    if VOICES[voice]["engine"] == "azure":
        pieces = ((x, AZURE_RATE) for x in azure_chunks(text, voice))
    else:
        pieces = mlx_tts.stream(speech_text(text), VOICES[voice])
    gain = None
    for x, rate in pieces:
        if not len(x):
            continue
        if gain is None:
            n = len(x) // 320
            rms = (
                np.sqrt((x[: n * 320].reshape(n, 320) ** 2).mean(axis=1)) if n else np.array([0.0])
            )
            voiced = rms[rms > rms.max() * 0.1] if rms.max() > 0 else rms
            level = float(np.sqrt((voiced**2).mean())) if len(voiced) else 0.0
            gain = min(10 ** (TARGET_RMS_DB / 20) / level, 30.0) if level > 0 else 1.0
        y = _resample(limit(x * gain), rate, sample_rate)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes((np.clip(y, -1, 1) * 32767).astype("<i2").tobytes())
        yield buf.getvalue(), len(y) / sample_rate


def render(text: str, voice: str, out: Path, sample_rate: int) -> Path:
    """One line with a neural voice (English lines with macOS Samantha unless the model
    speaks English too)."""
    if english(text) and not VOICES[voice].get("english"):
        from . import tts

        return tts.synthesize(text, "Samantha", None, out=out, sample_rate=sample_rate)
    if VOICES[voice]["engine"] == "gemini":
        return render_gemini(text, voice, out, sample_rate)
    if VOICES[voice]["engine"] == "azure":
        return render_azure(text, voice, out, sample_rate)
    if VOICES[voice]["engine"] == "google":
        return render_google(text, voice, out, sample_rate)
    if VOICES[voice]["engine"] == "mlx":
        return render_mlx(text, voice, out, sample_rate)
    return render_vits(text, out, sample_rate)
