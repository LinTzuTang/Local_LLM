"""Text to speech with macOS's built-in `say`: local, free, no model to download.

    from local_llm import tts
    tts.voices()                          # [{"name", "short", "locale", "sample"}, ...]
    path = tts.synthesize("你好，今天天氣很好。", "Meijia", rate=190)  # 16-bit WAV

`say -v '?'` lists the installed voices; `voices()` keeps the ones for Taiwan Mandarin
(zh_TW), Mainland Mandarin (zh_CN), Cantonese (zh_HK) and US / UK English by default and
leaves out the novelty voices (Bubbles, Zarvox, Bad News, ...). Better neural voices
(Premium / Enhanced) can be added in System Settings > Accessibility > Spoken Content and then
appear here under the same name. Rendering writes 16-bit PCM WAV directly
(`--data-format=LEI16@<rate>`), so no conversion step is needed.
"""

import hashlib
import re
import shutil
import subprocess
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


def voices(locales: tuple[str, ...] | None = DEFAULT_LOCALES, novelty: bool = False) -> list[dict]:
    """Installed `say` voices for these locales (None = all), novelty voices left out unless
    asked for; sorted by the order of `locales`, then name. Empty when `say` is missing."""
    if not available():
        return []
    rows = parse_voices(_listing())
    if not novelty:
        rows = [v for v in rows if v["name"] not in NOVELTY]
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
