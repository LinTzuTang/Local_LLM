"""Simultaneous interpretation: Chinese speech into English while the speaker talks.

Used by the platform (app/speech_simul.py, Compare speech models -> Simultaneous) and the
evaluation (scripts/speech_eval/run_simul.py), so both time and score the same way.

- Local cascade (`cascade`): an energy VAD (asr_streaming.OnlineVad, GAP_S) marks pauses; once
  the speech since the last cut is at least `min_span` seconds and a pause is seen, that span
  (padded) goes to a whole-clip ASR through the gateway and its text to a translator in Ollama.
  Spans are handled one at a time in order: a span's English appears at max(cut detected,
  previous done) + ASR + MT seconds (measured), as for a live speaker.
- Gemini Live Translate (`gemini`): Vertex AI Live API, 16 kHz PCM in 100 ms chunks at real
  time; the translated speech comes back with its transcript, each piece timestamped.

`score`: the speaker's progress through the text is taken as proportional to speaking time
(VAD speech, pauses not counted); an English word at position k of N is due when the speaker
had said k/N, and its lag is when it appeared minus that moment. first_s (speech start -> first
English), lag_avg_s / lag_p90_s over words, tail_s (last speech -> last English word), and
chrF / BLEU against a reference when there is one.
"""

import asyncio
import base64
import json
import statistics
import tempfile
import time
import wave
from collections.abc import Callable
from pathlib import Path

import numpy as np

from . import asr_streaming
from . import speech_translate as stx

RATE = 16000
GAP_S = 0.4  # a pause the VAD reports
MIN_SPAN_S = 4.0  # speech since the last cut before a pause may cut again
PAD_S = 0.3
GEMINI = "gemini-3.5-live-translate-preview"
TAIL_S = 8.0  # silence sent after the speaker, for Gemini's last words


def utterances(x: np.ndarray, gap: float = GAP_S) -> list[tuple[float, float]]:
    vad, events = asr_streaming.OnlineVad(RATE, min_gap=gap), []
    for i in range(0, len(x), 1600):
        events += vad.feed(x[i : i + 1600])
    segs, start = [], None
    for kind, t in events:
        if kind == "start":
            start = t
        elif start is not None:
            segs.append((start, t))
            start = None
    if start is not None:
        segs.append((start, len(x) / RATE))
    return segs


def spans(segs: list[tuple[float, float]], min_span: float = MIN_SPAN_S) -> list[dict]:
    """Utterances joined until `min_span` s of speech; a span is cut once the pause after it
    is detected (its last speech + GAP_S)."""
    out, cur = [], None
    for a, b in segs:
        cur = {"start": a, "end": b} if cur is None else {**cur, "end": b}
        if cur["end"] - cur["start"] >= min_span:
            out.append({**cur, "cut": cur["end"] + GAP_S})
            cur = None
    if cur:
        out.append({**cur, "cut": cur["end"] + GAP_S})
    return out


def score(pieces: list[dict], x: np.ndarray, reference: str = "") -> dict:
    """pieces: [{"t": seconds after the recording started, "text": English}] in order."""
    segs = utterances(x)
    total = sum(b - a for a, b in segs) or 1.0

    def said_by(share: float) -> float:
        need = share * total
        for a, b in segs:
            if need <= b - a:
                return a + need
            need -= b - a
        return segs[-1][1] if segs else 0.0

    words = [(p["t"], w) for p in pieces for w in p["text"].split()]
    n = len(words)
    lags = [t - said_by((k + 0.5) / n) for k, (t, _) in enumerate(words)]
    text = " ".join(p["text"].strip() for p in pieces if p["text"].strip())
    start = segs[0][0] if segs else 0.0
    end = segs[-1][1] if segs else len(x) / RATE
    out = {
        "first_s": round(words[0][0] - start, 2) if n else None,
        "lag_avg_s": round(statistics.mean(lags), 2) if lags else None,
        "lag_p90_s": round(float(np.percentile(lags, 90)), 2) if lags else None,
        "tail_s": round(words[-1][0] - end, 2) if n else None,
        "text": text,
        "chrf": None,
        "bleu": None,
    }
    if reference.strip() and text:
        out |= {k: v for k, v in stx.score(text, reference).items() if k in ("chrf", "bleu")}
    return out


def _wav(x: np.ndarray, path: Path) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    return path


def cascade(x: np.ndarray, transcribe: Callable[[Path], str],
            translate: Callable[[str], str], min_span: float = MIN_SPAN_S,
            progress: Callable[[int, int], None] | None = None,
            cancelled: Callable[[], bool] = lambda: False) -> list[dict]:  # fmt: skip
    """The local cascade over a recording, timed as for a live speaker: one row per span with
    its source text, English, measured ASR / MT seconds and `t`, when the English appears."""
    parts = spans(utterances(x), min_span)
    rows, busy = [], 0.0
    with tempfile.TemporaryDirectory() as tmp:
        for i, p in enumerate(parts):
            if cancelled():
                break
            if progress:
                progress(i, len(parts))
            a, b = max(0, int((p["start"] - PAD_S) * RATE)), int((p["end"] + PAD_S) * RATE)
            path = _wav(x[a:b], Path(tmp) / f"s{i}.wav")
            started = time.perf_counter()
            heard = transcribe(path).strip()
            asr_s = time.perf_counter() - started
            started = time.perf_counter()
            english = translate(heard).strip() if heard else ""
            mt_s = time.perf_counter() - started if heard else 0.0
            begin = max(p["cut"], busy)
            busy = begin + asr_s + mt_s
            rows.append({**{k: round(v, 3) for k, v in p.items()}, "source": heard,
                         "asr_s": round(asr_s, 3), "mt_s": round(mt_s, 3),
                         "t": round(busy, 3), "text": english})  # fmt: skip
    return rows


async def _gemini(x: np.ndarray, token: str, project: str, model: str, target: str,
                  cancelled: Callable[[], bool]) -> dict:  # fmt: skip
    import websockets

    url = ("wss://aiplatform.googleapis.com/ws/google.cloud.aiplatform.v1beta1."
           "LlmBidiService/BidiGenerateContent")  # fmt: skip
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes() + b"\0" * int(RATE * 2 * TAIL_S)
    chunk = RATE * 2 // 10
    pieces, heard, audio, errors = [], [], bytearray(), []
    clock = {"t0": 0.0, "first_audio": None}
    async with websockets.connect(url, additional_headers={"Authorization": f"Bearer {token}"},
                                  max_size=None) as ws:  # fmt: skip
        await ws.send(json.dumps({"setup": {
            "model": f"projects/{project}/locations/global/publishers/google/models/{model}",
            "generationConfig": {"responseModalities": ["AUDIO"],
                                 "translationConfig": {"targetLanguageCode": target}},
            "inputAudioTranscription": {}, "outputAudioTranscription": {}}}))  # fmt: skip
        reply = json.loads(await ws.recv())
        if "setupComplete" not in reply:
            raise RuntimeError(f"setup: {json.dumps(reply)[:300]}")

        async def receive():
            async for message in ws:
                t = time.monotonic() - clock["t0"]
                event = json.loads(message)
                content = event.get("serverContent") or {}
                if text := (content.get("outputTranscription") or {}).get("text"):
                    pieces.append({"t": round(t, 3), "text": text})
                if text := (content.get("inputTranscription") or {}).get("text"):
                    heard.append(text)
                for part in (content.get("modelTurn") or {}).get("parts", []):
                    if data := (part.get("inlineData") or {}).get("data"):
                        clock["first_audio"] = clock["first_audio"] or round(t, 3)
                        audio.extend(base64.b64decode(data))
                if "error" in event:
                    errors.append(event["error"])

        receiver = asyncio.create_task(receive())
        clock["t0"] = time.monotonic()
        for i, start in enumerate(range(0, len(pcm), chunk), 1):
            if cancelled():
                break
            await ws.send(json.dumps({"realtimeInput": {"audio": {
                "mimeType": f"audio/pcm;rate={RATE}",
                "data": base64.b64encode(pcm[start : start + chunk]).decode()}}}))  # fmt: skip
            await asyncio.sleep(max(0.0, clock["t0"] + i * 0.1 - time.monotonic()))
        last = len(pieces)
        while not cancelled():  # until nothing new for 5 s
            await asyncio.sleep(5)
            if len(pieces) == last:
                break
            last = len(pieces)
        receiver.cancel()
        await asyncio.gather(receiver, return_exceptions=True)
    return {"pieces": pieces, "heard": "".join(heard), "audio": bytes(audio),
            "first_audio_s": clock["first_audio"], "errors": errors}  # fmt: skip


def gemini(x: np.ndarray, model: str = GEMINI, target: str = "en",
           cancelled: Callable[[], bool] = lambda: False) -> dict:  # fmt: skip
    """Gemini Live Translate over a recording at real time: {"pieces", "heard" (its transcript
    of the speaker), "audio" (24 kHz PCM16 of the translated speech), "first_audio_s",
    "errors"}. Uses the service account of gcp_auth."""
    from . import gcp_auth

    token, project = gcp_auth.token()
    return asyncio.run(_gemini(x, token, project, model, target, cancelled))
