"""ElevenLabs (cloud): its speech features, for the platform's ElevenLabs page, the text-to-speech
voices (neural_tts) and the meeting-speaker comparison.

The key is ELEVENLABS_API_KEY (environment or the project's .env; the ElevenLabs CLI's onboarding
writes it there). Everything here sends text or audio to ElevenLabs, so the app only calls it
with the cloud switch on.

Measured 2026-11-05 (scripts/speech_eval/run_elevenlabs.py): Eleven v4 Turbo reads Mandarin as
clearly as v4 (1.6-1.9% round-trip error over 9 Taiwanese voices) and starts in 0.12 s over its
WebSocket; Flash v2.5 misreads 10-19% and ran away on some lines for 3 of 8 voices; Scribe v2 is
close to the local Qwen3-ASR-1.7B but not better; its diarization is weak with 4 speakers.

What this key cannot do (missing permissions): add or save voices (voices_write) and dubbing
(dubbing_write).
"""

import base64
import json
import os
import time
from functools import cache
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
RATE = 24000  # PCM sample rate asked of the synthesis models

# Taiwanese-accent voices of the ElevenLabs voice library (professional voice clones, fine-tuned
# for v4, v4 Turbo, Flash v2.5 and Multilingual v2; the most used ones, 2026-11-05)
VOICES = {
    "Anna Su": ("9lHjugDhwqoxA5MhX0az", "女", "female"),
    "Yui": ("kGjJqO6wdwRN9iJsoeIC", "女", "female"),
    "Tiffy": ("1AKkSX7KMPHIWuz76m0n", "女", "female"),
    "Lee Ting Ting": ("gU2KtIu9OZWy3KqiqNj6", "女", "female"),
    "Lucy": ("r4thHOi5MAZw256OxDPJ", "女", "female"),
    "Yu": ("fQj4gJSexpu8RDE2Ii5m", "男", "male"),
    "Kevin Tu": ("BrbEfHMQu0fyclQR7lfh", "男", "male"),
    "Evan": ("kbrsaic1zriFXx1pgRYN", "男", "male"),
    "Neil Chuang": ("auoHciLZJwKTwYUoRTYz", "男", "male"),
}
# model id -> (Chinese, English) note; first = default
TTS_MODELS = {
    "eleven_v4_turbo": ("v4 Turbo：音質同 v4、0.12 秒開口（實測最適合對話）",
                        "v4 Turbo: v4 quality, starts in 0.12 s (best for conversation)"),
    "eleven_v4": ("v4：最高音質，支援 [whisper] 等語氣標籤",
                  "v4: best quality, audio tags like [whisper]"),
    "eleven_v3": ("v3：上一代表現力模型，支援語氣標籤",
                  "v3: previous expressive model, audio tags"),
    "eleven_multilingual_v2": ("Multilingual v2：上一代，長文最穩",
                               "Multilingual v2: previous, steadiest on long text"),
    "eleven_flash_v2_5": ("Flash v2.5：最快，但中文錯字 10–19%、可能失控",
                          "Flash v2.5: fastest, but 10-19% errors in Mandarin and may run away"),
}  # fmt: skip
DIALOGUE_MODELS = ["eleven_v4", "eleven_v3"]
STT_MODEL = "scribe_v2"
STS_MODEL = "eleven_multilingual_sts_v2"
MISSING = {"dubbing": "dubbing_write", "save_voice": "voices_write"}


def _env() -> dict:
    env = dict(os.environ)
    dotenv = PROJECT / ".env"
    if dotenv.exists():
        for line in dotenv.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                env.setdefault(k.strip(), v.strip().strip("'\""))
    return env


def api_key() -> str | None:
    return _env().get("ELEVENLABS_API_KEY") or None


def available() -> bool:
    import importlib.util

    return api_key() is not None and importlib.util.find_spec("elevenlabs") is not None


@cache
def client():
    from elevenlabs import ElevenLabs

    # the SDK reads the key as a default argument when imported: pass it explicitly
    return ElevenLabs(api_key=api_key())


def usage() -> tuple[int, int]:
    """(credits used this period, credit limit)."""
    sub = client().user.subscription.get()
    return int(sub.character_count), int(sub.character_limit)


def _join(chunks) -> bytes:
    return b"".join(chunks)


def wav_bytes(pcm: bytes, rate: int = RATE) -> bytes:
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


# -- text to speech ------------------------------------------------------------------------------


def turbo_chunks(text: str, voice_id: str, language: str | None = "zh"):
    """Eleven v4 Turbo over the Text to Dialogue WebSocket (its only way in): PCM16 pieces at
    RATE as they come. One connection per call, one voice per connection."""
    from websockets.sync.client import connect

    url = ("wss://api.elevenlabs.io/v1/text-to-dialogue/stream-input"
           f"?model_id=eleven_v4_turbo&output_format=pcm_{RATE}")  # fmt: skip
    if language:
        url += f"&language_code={language}"
    with connect(url, max_size=None, open_timeout=20) as ws:
        ws.send(json.dumps({"voices": [voice_id], "xi_api_key": api_key()}))
        ws.send(json.dumps({"inputs": [{"text": text, "voice_id": voice_id, "new_turn": False}]}))
        ws.send(json.dumps({"close_socket": True}))
        while True:
            msg = json.loads(ws.recv(timeout=60))
            if msg.get("error"):
                raise RuntimeError(f"ElevenLabs: {msg['error']}")
            if msg.get("audio"):
                yield base64.b64decode(msg["audio"])
            if msg.get("is_final"):
                return


def speech_chunks(text: str, voice_id: str, model: str = "eleven_v4_turbo",
                  language: str | None = "zh", stability: float | None = None):  # fmt: skip
    """PCM16 pieces at RATE while the model makes them, for any synthesis model."""
    if model == "eleven_v4_turbo":
        yield from turbo_chunks(text, voice_id, language)
        return
    kwargs = {}
    if language and model != "eleven_multilingual_v2":  # v2 takes no language_code
        kwargs["language_code"] = language
    if stability is not None:
        from elevenlabs import VoiceSettings

        kwargs["voice_settings"] = VoiceSettings(stability=stability, similarity_boost=0.75)
    yield from client().text_to_speech.stream(text=text, voice_id=voice_id, model_id=model,
                                              output_format=f"pcm_{RATE}", **kwargs)  # fmt: skip


def speak(text: str, voice_id: str, model: str = "eleven_v4_turbo", language: str | None = "zh",
          stability: float | None = None) -> dict:  # fmt: skip
    """{"wav", "first_s" (to the first audio), "total_s", "seconds" (of audio)}."""
    started, first, pcm = time.perf_counter(), None, b""
    for piece in speech_chunks(text, voice_id, model, language, stability):
        if first is None and piece:
            first = time.perf_counter() - started
        pcm += piece
    total = time.perf_counter() - started
    return {"wav": wav_bytes(pcm), "first_s": round(first or total, 3), "total_s": round(total, 2),
            "seconds": round(len(pcm) / 2 / RATE, 2)}  # fmt: skip


def dialogue(lines: list[tuple[str, str]], model: str = "eleven_v4") -> bytes:
    """Several voices in one take: [(voice_id, text)] -> MP3."""
    inputs = [{"text": text, "voice_id": voice} for voice, text in lines if text.strip()]
    return _join(client().text_to_dialogue.convert(inputs=inputs, model_id=model,
                                                   output_format="mp3_44100_128"))  # fmt: skip


# -- speech to text ------------------------------------------------------------------------------


def turns(words, gap: float = 0.5) -> list[dict]:
    """Scribe's diarized words -> speaker turns [{"start", "end", "speaker", "text"}]: a turn runs
    on while the speaker stays the same and the pause between words is under `gap` seconds."""
    out = []
    for w in words or []:
        if getattr(w, "type", "word") != "word" or w.start is None:
            continue
        spk = str(w.speaker_id)
        if out and out[-1]["speaker"] == spk and w.start - out[-1]["end"] < gap:
            out[-1]["end"], out[-1]["text"] = w.end, out[-1]["text"] + w.text
        else:
            out.append({"start": w.start, "end": w.end, "speaker": spk, "text": w.text})
    return out


def transcribe(audio: Path | bytes, language: str | None = "zho", diarize: bool = False,
               speakers: int | None = None, events: bool = False) -> dict:  # fmt: skip
    """Scribe v2 on a whole recording: {"text", "segments" (speaker turns when diarized),
    "language", "seconds"}."""
    data = audio.read_bytes() if isinstance(audio, Path) else audio
    started = time.perf_counter()
    kwargs = {"diarize": diarize, "tag_audio_events": events, "timestamps_granularity": "word"}
    if language:
        kwargs["language_code"] = language
    if diarize and speakers:
        kwargs["num_speakers"] = speakers
    r = client().speech_to_text.convert(file=("audio", data), model_id=STT_MODEL, **kwargs)
    return {"text": r.text, "segments": turns(r.words) if diarize else [],
            "language": getattr(r, "language_code", None),
            "seconds": round(time.perf_counter() - started, 2)}  # fmt: skip


def diarize(audio: Path, speakers: int | None = None) -> dict:
    """Speaker turns as the gateway's diarization returns them: {"segments", "speakers",
    "seconds"} (Scribe v2 with diarization; `speakers` None = it estimates the count)."""
    out = transcribe(audio, diarize=True, speakers=speakers)
    segs = out["segments"]
    speakers = len({s["speaker"] for s in segs})
    return {"segments": segs, "speakers": speakers, "seconds": out["seconds"]}


def align(audio: bytes, text: str) -> list[dict]:
    """Forced alignment: when each word of `text` is said in `audio`."""
    r = client().forced_alignment.create(file=("audio", audio), text=text)
    return [{"text": w.text, "start": round(w.start, 2), "end": round(w.end, 2)} for w in r.words]


# -- audio to audio, and generated sound ----------------------------------------------------------


def change_voice(audio: bytes, voice_id: str, remove_noise: bool = False) -> bytes:
    """Speech to speech: the same words, said in another voice -> MP3."""
    out = client().speech_to_speech.convert(
        voice_id=voice_id, audio=("audio", audio), model_id=STS_MODEL,
        remove_background_noise=remove_noise, output_format="mp3_44100_128")  # fmt: skip
    return _join(out)


def isolate(audio: bytes) -> bytes:
    """Voice Isolator: the voice kept, background removed -> MP3. (Measured: it hurts speech
    recognition; with talkers in the background it cannot tell whose voice to keep.)"""
    return _join(client().audio_isolation.convert(audio=("audio", audio)))


def sound_effect(prompt: str, seconds: float | None = None, loop: bool = False) -> bytes:
    kwargs = {"duration_seconds": seconds} if seconds else {}
    out = client().text_to_sound_effects.convert(
        text=prompt, loop=loop, output_format="mp3_44100_128", **kwargs)  # fmt: skip
    return _join(out)


def music(prompt: str, seconds: float = 15.0, instrumental: bool = True) -> bytes:
    out = client().music.compose(prompt=prompt, music_length_ms=int(seconds * 1000),
                                 force_instrumental=instrumental,
                                 output_format="mp3_44100_128")  # fmt: skip
    return _join(out)


def design_voice(description: str, text: str | None = None) -> dict:
    """Voice design: a voice from a description, three previews to listen to (saving one needs
    the voices_write permission). {"text", "previews": [{"id", "mp3"}]}."""
    kwargs = {"text": text} if text and len(text) >= 100 else {"auto_generate_text": True}
    r = client().text_to_voice.design(voice_description=description, model_id="eleven_ttv_v3",
                                      output_format="mp3_44100_128", **kwargs)  # fmt: skip
    previews = [{"id": p.generated_voice_id, "mp3": base64.b64decode(p.audio_base_64)}
                for p in r.previews]  # fmt: skip
    return {"text": r.text, "previews": previews}


def dub(audio: bytes, target: str, source: str | None = "zh", timeout: float = 600) -> bytes:
    """Dubbing into `target` (needs the dubbing_write permission): the dubbed audio as MP3."""
    c = client()
    job = c.dubbing.create(file=("audio.wav", audio), target_lang=target, source_lang=source)
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        status = c.dubbing.get(job.dubbing_id).status
        if status == "dubbed":
            return _join(c.dubbing.audio.get(job.dubbing_id, target))
        if status in ("failed", "error"):
            raise RuntimeError(f"ElevenLabs dubbing {status}")
        time.sleep(5)
    raise TimeoutError("ElevenLabs dubbing did not finish")
