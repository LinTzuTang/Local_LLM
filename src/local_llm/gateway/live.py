"""Live microphone streaming: what the gateway's WebSocket endpoint does with each message.

The browser sends 16 kHz mono PCM16 (little-endian) frames, about 100 ms each. A LiveSession
feeds them to a streaming model's decoding session (stream_backends) and turns the output into
JSON events for the page:

    {"type": "partial" | "final", "text", "t_audio", "t_wall", "final", "seg"[, "display"]}
    {"type": "speech_start" | "speech_end", "t_audio"}           (online energy VAD)
    {"type": "metric", "kind": "first" | "final", "utterance", "latency"}
    {"type": "done", "text", "summary", "first_latencies", "final_latencies", ...}

`t_audio` is the audio received so far (s); `t_wall` is wall-clock time on the same axis: the
clock starts when the first frame arrives, minus that frame's length (when its audio began).
So `t_wall - t_audio` is how far the text trails the speaker, and the latencies include
buffering in the browser and the network as well as the model. With a `display` function
(e.g. OpenCC Simplified -> Traditional) each text event also carries the text to show;
`text` stays the model's raw output.

- metric "first": speech start (VAD) -> the first text after it, per utterance;
- metric "final": end of an utterance (VAD) -> its final text (asr_streaming.finalise_one),
  reported as soon as it is known: for models with endpointing at their final event, for
  append-only models (Voxtral) once the next utterance starts or the stream stops.
"""

import time
from collections.abc import Callable

from local_llm import asr_streaming

RATE = 16000


class LiveSession:
    def __init__(
        self,
        session,
        rate: int = RATE,
        clock: Callable[[], float] = time.perf_counter,
        display: Callable[[str], str] | None = None,
    ):
        self.session, self.rate, self.clock, self.display = session, rate, clock, display
        self.append_only = bool(getattr(session, "append_only", False))
        self.vad = asr_streaming.OnlineVad(rate)
        self.t0: float | None = None
        self.rest = b""
        self.received = 0  # samples
        self.audio: list = []
        self.events: list[dict] = []  # the text events, in asr_streaming's format
        self.utterances: list[list] = []  # [start, end or None] from the online VAD
        self.awaiting_first: int | None = None
        self.firsts: dict[int, float] = {}
        self.finals: dict[int, float] = {}
        self.seg = 0
        self.compute = 0.0
        self.closed = False

    # -- input ---------------------------------------------------------------------------

    def feed(self, data: bytes, arrived: float | None = None) -> list[dict]:
        """One binary message (PCM16) -> the events it causes."""
        if self.closed:
            return []
        arrived = self.clock() if arrived is None else arrived
        samples, self.rest = asr_streaming.pcm16(self.rest + data)
        if not len(samples):
            return []
        if self.t0 is None:
            self.t0 = arrived - len(samples) / self.rate
        self.received += len(samples)
        self.audio.append(samples)
        out = []
        for kind, t in self.vad.feed(samples):
            out += self._speech(kind, t)
        start = self.clock()
        outputs = self.session.feed(samples)
        self.compute += self.clock() - start
        out += self._text(outputs)
        return out

    def stop(self) -> list[dict]:
        """The stream ended: flush the model and the VAD, report what is left and a summary."""
        if self.closed:
            return []
        self.closed = True
        out = []
        for kind, t in self.vad.flush():
            out += self._speech(kind, t)
        start = self.clock()
        outputs = self.session.close()
        self.compute += self.clock() - start
        out += self._text(outputs)
        out += self._final_metrics(at_stop=True)
        out.append(self.done())
        return out

    # -- events ----------------------------------------------------------------------------

    def now(self) -> float:
        return round(self.clock() - (self.t0 or self.clock()), 3)

    def t_audio(self) -> float:
        return round(self.received / self.rate, 3)

    def _speech(self, kind: str, t: float) -> list[dict]:
        out = [{"type": f"speech_{kind}", "t_audio": t}]
        if kind == "start":
            self.utterances.append([t, None])
            self.awaiting_first = len(self.utterances) - 1
            out += self._final_metrics()  # the previous utterance's next start is known now
        elif self.utterances and self.utterances[-1][1] is None:
            self.utterances[-1][1] = t
        return out

    def _text(self, outputs) -> list[dict]:
        out = []
        t_wall = self.now()
        for text, final in outputs:
            event = {"type": "final" if final else "partial", "text": text,
                     "t_audio": self.t_audio(), "t_wall": t_wall, "final": bool(final),
                     "seg": self.seg}  # fmt: skip
            if self.display:
                event["display"] = self.display(text)
            self.events.append(event)
            out.append(event)
            if final:
                self.seg += 1
            if self.awaiting_first is not None and text.strip():
                i, self.awaiting_first = self.awaiting_first, None
                latency = round(max(t_wall - self.utterances[i][0], 0.0), 3)
                self.firsts[i] = latency
                out.append({"type": "metric", "kind": "first", "utterance": i,
                            "latency": latency})  # fmt: skip
            if final:
                out += self._final_metrics()
        return out

    def _final_metrics(self, at_stop: bool = False) -> list[dict]:
        """Finalisation latency of every ended utterance that can be decided now."""
        finals = asr_streaming.final_times(self.events)
        out = []
        for i, (_, end) in enumerate(self.utterances):
            if end is None or i in self.finals:
                continue
            nxt = self.utterances[i + 1][0] if i + 1 < len(self.utterances) else None
            if nxt is None and self.append_only and not at_stop:
                continue  # more pieces may still come for it
            row = asr_streaming.finalise_one(finals, end, nxt if nxt is not None else float("inf"))
            if row is not None:
                self.finals[i] = row["latency"]
                out.append({"type": "metric", "kind": "final", "utterance": i,
                            "latency": row["latency"]})  # fmt: skip
        return out

    def done(self) -> dict:
        import numpy as np

        duration = self.received / self.rate
        audio = np.concatenate(self.audio) if self.audio else np.zeros(0, dtype=np.float32)
        segments = asr_streaming.speech_segments(audio, self.rate)
        summary = asr_streaming.summary(self.events, segments, duration, self.compute,
                                        self.append_only)  # fmt: skip
        firsts, finals = list(self.firsts.values()), list(self.finals.values())
        return {
            "type": "done",
            "text": asr_streaming.transcript(self.events, self.append_only),
            "duration": round(duration, 3),
            "compute_s": round(self.compute, 3),
            "append_only": self.append_only,
            "summary": summary,
            "segments": segments,
            "utterances": [[s, e] for s, e in self.utterances],
            "first_latencies": firsts,
            "final_latencies": finals,
            "first_median_s": _median(firsts),
            "final_median_s": _median(finals),
        }


def _median(values: list[float]) -> float | None:
    import statistics

    return round(statistics.median(values), 3) if values else None
