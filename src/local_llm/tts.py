"""Text to speech with macOS's built-in `say`: local, free, no model to download.

    from local_llm import tts
    tts.voices()                          # [{"name", "short", "locale", "sample"}, ...]
    path = tts.synthesize("你好，今天天氣很好。", "Meijia", rate=190)  # 16-bit WAV

`say -v '?'` lists the installed voices; `voices()` keeps the ones for Taiwan Mandarin
(zh_TW), Mainland Mandarin (zh_CN), Cantonese (zh_HK) and US / UK English by default,
leaves out the novelty voices (Bubbles, Zarvox, Bad News, ...) and lists the eight Eloquence
voices (Eddy, Flo, Reed, ...: one synthesiser) as Flo alone. Better neural voices
(Premium / Enhanced) can be added in System Settings > Accessibility > Spoken Content and then
appear here under the same name. Rendering writes 16-bit PCM WAV directly
(`--data-format=LEI16@<rate>`), so no conversion step is needed.
"""

import hashlib
import re
import shutil
import struct
import subprocess
import threading
import time
import wave
from functools import cache
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_DIR / "storage" / "speech_cache" / "tts"
SAMPLE_RATE = 24000
DEFAULT_LOCALES = ("zh_TW", "zh_CN", "zh_HK", "en_US", "en_GB")
# Sound-effect voices from macOS's "Novelty" group: fun, but not speech to compare.
NOVELTY = {
    "Albert", "Bad News", "Bahh", "Bells", "Boing", "Bubbles", "Cellos", "Deranged",
    "Good News", "Hysterical", "Jester", "Organ", "Pipe Organ", "Superstar", "Trinoids",
    "Whisper", "Wobble", "Zarvox",
}  # fmt: skip
# Eight names, one synthesiser (com.apple.eloquence.*): the same timing and prosody, only
# pitch and timbre differ (a sentence takes exactly as long in each), so they sound alike.
# One stands for them: Flo, the clearest in a speech-recognition round trip (2026-10-20,
# Qwen3-ASR-0.6B: zh_TW 3.0% CER, the others 4.5-13.4%; en_US 8% WER, the others 12%).
ELOQUENCE = {"Eddy", "Flo", "Grandma", "Grandpa", "Reed", "Rocko", "Sandy", "Shelley"}
ELOQUENCE_PICK = "Flo"
_LINE = re.compile(r"^(?P<name>.+?)\s+(?P<locale>[a-z]{2,3}_[A-Za-z0-9]+)\s+#\s?(?P<sample>.*)$")


def available() -> bool:
    return shutil.which("say") is not None


def parse_voices(listing: str) -> list[dict]:
    """Rows of `say -v '?'` -> [{"name", "short", "locale", "sample"}], first occurrence of
    each name (a voice installed in two qualities is listed twice with the same name).

    `name` is what `say -v` takes; voices that exist in several languages carry the language
    in brackets ("Eddy (中文（台灣）)", in the system's language), `short` drops it ("Eddy")."""
    out, seen = [], set()
    for line in listing.splitlines():
        m = _LINE.match(line.strip())
        if m and m["name"] not in seen:
            seen.add(m["name"])
            short = re.sub(r"\s*\(.*\)$", "", m["name"]) or m["name"]
            out.append({"name": m["name"], "short": short, "locale": m["locale"],
                        "sample": m["sample"].strip()})  # fmt: skip
    return out


def find(name: str, rows: list[dict] | None = None) -> dict | None:
    """The voice with this `say` name, or the first whose short name matches ("Meijia")."""
    rows = voices(None) if rows is None else rows
    return next((v for v in rows if v["name"] == name), None) or next(
        (v for v in rows if v["short"] == name), None
    )


@cache
def _listing() -> str:
    return subprocess.run(["say", "-v", "?"], capture_output=True, text=True, check=True).stdout


def voices(
    locales: tuple[str, ...] | None = DEFAULT_LOCALES,
    novelty: bool = False,
    eloquence: bool = False,
) -> list[dict]:
    """Installed `say` voices for these locales (None = all), novelty voices left out unless
    asked for and the Eloquence family as Flo alone unless `eloquence`; sorted by the order
    of `locales`, then name. Empty when `say` is missing."""
    if not available():
        return []
    rows = parse_voices(_listing())
    if not novelty:
        rows = [v for v in rows if v["name"] not in NOVELTY]
    if not eloquence:
        rows = [v for v in rows if v["short"] not in ELOQUENCE or v["short"] == ELOQUENCE_PICK]
    if locales is not None:
        rows = [v for v in rows if v["locale"] in locales]
        rows.sort(key=lambda v: (locales.index(v["locale"]), v["name"]))
    return rows


def language(voice: dict | None) -> str:
    """ "en" for an English voice, else "zh" (the scoring language of its speech)."""
    return "en" if voice and voice.get("locale", "").startswith("en") else "zh"


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


def synthesize(
    text: str,
    voice: str,
    rate: int | None = None,
    out: Path | None = None,
    sample_rate: int = SAMPLE_RATE,
) -> Path:
    """Speak `text` with a `say` voice into a 16-bit mono WAV and return its path.

    rate: words per minute (None = the voice's default, about 175-200). Without `out` the
    file goes to storage/speech_cache/tts/, named by a hash of text, voice and rate."""
    if not text.strip():
        raise ValueError("no text to speak")
    if not available():
        raise RuntimeError("macOS `say` is not available on this machine")
    if out is None:
        key = hashlib.sha1(f"{voice}|{rate}|{sample_rate}|{text}".encode()).hexdigest()[:16]
        out = CACHE_DIR / f"{key}.wav"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["say", "-v", voice, "-o", str(out), f"--data-format=LEI16@{sample_rate}"]
    if rate:
        cmd += ["-r", str(int(rate))]
    # The text goes in on stdin (`-f -`): no length limit and no option parsing of the text.
    subprocess.run(cmd + ["-f", "-"], input=text, text=True, capture_output=True, check=True)
    return out


def timed_synthesize(text: str, voice: str, rate: int | None = None) -> dict:
    """synthesize() always rendering anew (no cache), with its timing: {"path", "seconds"
    (synthesis), "duration" (audio), "rtf" (seconds / duration)}."""
    key = hashlib.sha1(f"{voice}|{rate}|{text}|{time.time_ns()}".encode()).hexdigest()[:16]
    started = time.perf_counter()
    path = synthesize(text, voice, rate, out=CACHE_DIR / f"{key}.wav")
    seconds = time.perf_counter() - started
    duration = wav_duration(path)
    return {
        "path": path,
        "seconds": round(seconds, 3),
        "duration": round(duration, 2),
        "rtf": round(seconds / duration, 4) if duration else None,
    }


# -- in-process synthesis (no `say` process per sentence) --------------------------------------
#
# Starting `say` costs about 0.5 s however short the text, which a voice assistant pays on
# every sentence. The same voices are available in-process through AppKit's
# NSSpeechSynthesizer (PyObjC): after the first call a short sentence renders in about
# 0.06 s. It writes AIFF (big-endian PCM), converted here to a 16-bit WAV. Without PyObjC,
# or if it fails, `speak_wav` falls back to `say`.

_ns_lock = threading.Lock()
_ns_synths: dict[str, object] = {}


def _ns_voice_id(voice: str, rows: list[dict] | None = None):
    import AppKit

    row = find(voice, rows)
    short, locale = (row["short"], row["locale"]) if row else (voice, None)
    found = []
    for vid in AppKit.NSSpeechSynthesizer.availableVoices():
        attrs = AppKit.NSSpeechSynthesizer.attributesForVoice_(vid)
        name = re.sub(r"\s*\(.*\)$", "", str(attrs.get(AppKit.NSVoiceName) or ""))
        if name == short and (
            locale is None or attrs.get(AppKit.NSVoiceLocaleIdentifier) == locale
        ):
            found.append(vid)
    # A voice downloaded in better quality is listed next to its compact version under the
    # same name: use the best one installed.
    return max(found, key=voice_quality) if found else None


def voice_quality(identifier) -> int:
    """3 premium, 2 enhanced, 1 anything else (compact, Eloquence), from the voice id
    (com.apple.voice.enhanced.zh-TW.Meijia)."""
    ident = str(identifier)
    return 3 if ".premium." in ident else 2 if ".enhanced." in ident else 1


def read_aiff(data: bytes) -> tuple["object", int]:
    """(int16 samples, sample rate) of a mono 16-bit PCM AIFF / AIFF-C file."""
    import numpy as np

    if data[:4] != b"FORM" or data[8:12] not in (b"AIFF", b"AIFC"):
        raise ValueError("not an AIFF file")
    pos, rate, comp, pcm = 12, None, b"NONE", None
    while pos + 8 <= len(data):
        cid, size = data[pos : pos + 4], struct.unpack(">I", data[pos + 4 : pos + 8])[0]
        body = data[pos + 8 : pos + 8 + size]
        if cid == b"COMM":
            channels, _, bits = struct.unpack(">hIh", body[:8])
            exp, mant = struct.unpack(">HQ", body[8:18])
            rate = round(mant * 2.0 ** ((exp & 0x7FFF) - 16383 - 63))
            comp = body[18:22] if len(body) >= 22 else b"NONE"
            if channels != 1 or bits != 16:
                raise ValueError(f"unsupported AIFF: {channels} channels, {bits} bits")
        elif cid == b"SSND":
            offset = struct.unpack(">I", body[:4])[0]
            pcm = body[8 + offset :]
        pos += 8 + size + (size & 1)
    if rate is None or pcm is None:
        raise ValueError("incomplete AIFF file")
    order = "<i2" if comp == b"sowt" else ">i2"
    return np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype=order).astype(np.int16), rate


def wav_bytes(samples, rate: int) -> bytes:
    import io

    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())
    return out.getvalue()


def _ns_speak(text: str, voice: str, rate: int | None) -> tuple[bytes, float]:
    import tempfile

    import AppKit
    import Foundation

    with _ns_lock:
        synth = _ns_synths.get(voice)
        if synth is None:
            vid = _ns_voice_id(voice)
            if vid is None:
                raise RuntimeError(f"voice {voice!r} not found")
            synth = AppKit.NSSpeechSynthesizer.alloc().initWithVoice_(vid)
            _ns_synths[voice] = synth
        if rate:
            synth.setRate_(float(rate))
        with tempfile.TemporaryDirectory(prefix="tts-") as tmp:
            path = Path(tmp) / "out.aiff"
            if not synth.startSpeakingString_toURL_(text, Foundation.NSURL.fileURLWithPath_(
                    str(path))):  # fmt: skip
                raise RuntimeError("NSSpeechSynthesizer refused the text")
            deadline = time.monotonic() + 30
            loop = Foundation.NSRunLoop.currentRunLoop()
            while synth.isSpeaking() and time.monotonic() < deadline:
                loop.runUntilDate_(Foundation.NSDate.dateWithTimeIntervalSinceNow_(0.005))
            samples, sr = read_aiff(path.read_bytes())
    return wav_bytes(samples, sr), len(samples) / sr


def speak_wav(text: str, voice: str, rate: int | None = None) -> tuple[bytes, float]:
    """(16-bit mono WAV bytes, seconds of audio) for one sentence, as fast as this Mac can:
    in-process NSSpeechSynthesizer, else `say` (cached file)."""
    if not text.strip():
        raise ValueError("no text to speak")
    try:
        return _ns_speak(text, voice, rate)
    except Exception:  # no PyObjC, unknown voice, ...: the reliable path
        path = synthesize(text, voice, rate)
        return path.read_bytes(), wav_duration(path)
