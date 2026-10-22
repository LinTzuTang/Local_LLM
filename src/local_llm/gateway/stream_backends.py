"""True streaming speech recognition: models that take audio a frame at a time.

A streaming backend's `load` returns a handle like any other backend; `session(handle)` opens
one decoding session with `feed(samples) -> [(text, final)]` and `close() -> [(text, final)]`.
`text` is the current utterance's hypothesis (sherpa-onnx) or a newly appended piece
(`append_only` sessions such as Voxtral, whose pieces are all final). `drive` feeds a whole
recording through a session frame by frame and turns the outputs into timed events
(asr_streaming documents the format and how simulated time is computed).

- sherpa-online: sherpa-onnx OnlineRecognizer (streaming Zipformer transducer or Paraformer)
  on the CPU, with endpoint detection committing an utterance after trailing silence.
- mlx-audio-realtime: Voxtral Mini 4B Realtime through mlx-audio's VoxtralStreamingSession
  (causal audio encoder, decoder running a fixed delay behind the audio) on the Apple GPU.
"""

import os
import re
import threading
import time
from collections.abc import Callable
from pathlib import Path

from .backends import SAMPLE_RATE, Backend, mlx_turn

# Silence appended at the end so the last words leave the model's lookahead window.
TAIL_PADDING_S = 0.66


class StreamBackend(Backend):
    task = "stream"
    append_only = False

    def session(self, handle):
        raise NotImplementedError

    def run(self, handle, samples, frame_s: float = 0.16, pace: str = "fast", cancel=None,
            emit: Callable[[dict], None] | None = None) -> dict:  # fmt: skip
        return drive(self.session(handle), samples, frame_s, pace, cancel, emit)


def drive(
    session,
    samples,
    frame_s: float = 0.16,
    pace: str = "fast",
    cancel: threading.Event | None = None,
    emit: Callable[[dict], None] | None = None,
) -> dict:
    """Feed `samples` (16 kHz mono) to `session` in frames of `frame_s` seconds.

    pace="fast" feeds as fast as the model goes and reports simulated time (t_wall = when the
    output would be ready live, from the measured compute per frame); pace="realtime" waits
    for each frame's arrival time, as a microphone would, and reports the measured time.
    Each event is passed to `emit` as it happens; returns {"events", "duration", "compute_s",
    "frames", "frame_s", "append_only", "stopped"}."""
    import numpy as np

    from local_llm import asr_streaming

    samples = np.ascontiguousarray(samples, dtype=np.float32)
    duration = len(samples) / SAMPLE_RATE
    events: list[dict] = []
    seg, busy, compute_total, frames = 0, 0.0, 0.0, 0
    start = time.perf_counter()

    def publish(outputs, arrival: float, compute: float) -> None:
        nonlocal seg, busy
        busy = max(arrival, busy) + compute  # asr_streaming.schedule, one frame at a time
        t_wall = busy if pace == "fast" else time.perf_counter() - start
        for text, final in outputs:
            event = {"t_audio": round(arrival, 3), "t_wall": round(t_wall, 3), "text": text,
                     "final": bool(final), "seg": seg}  # fmt: skip
            events.append(event)
            if emit:
                emit(event)
            if final:
                seg += 1

    stopped = False
    for a, b in asr_streaming.frame_bounds(len(samples), frame_s):
        if cancel is not None and cancel.is_set():
            stopped = True
            break
        arrival = b / SAMPLE_RATE
        if pace == "realtime" and (wait := start + arrival - time.perf_counter()) > 0:
            time.sleep(wait)
        t0 = time.perf_counter()
        outputs = session.feed(samples[a:b])
        compute = time.perf_counter() - t0
        compute_total += compute
        frames += 1
        publish(outputs, arrival, compute)
    t0 = time.perf_counter()
    outputs = session.close()
    compute = time.perf_counter() - t0
    compute_total += compute
    fed = frames and min(frames * round(frame_s * SAMPLE_RATE), len(samples)) / SAMPLE_RATE
    publish(outputs, fed or 0.0, compute)
    return {
        "events": events,
        "duration": round(duration, 3),
        "compute_s": round(compute_total, 3),
        "frames": frames,
        "frame_s": frame_s,
        "append_only": bool(getattr(session, "append_only", False)),
        "stopped": stopped,
    }


# -- sherpa-onnx ------------------------------------------------------------------------------


# A silence token some zipformer models write into the text at a long pause ("好 SIL謝謝").
_SIL = re.compile(r"\s*(?<![A-Za-z])SIL(?![A-Za-z])\s*")
_CJK_GAP = re.compile(r"(?<=[一-鿿])\s+(?=[一-鿿])")


def clean(text: str) -> str:
    return _CJK_GAP.sub("", _SIL.sub(" ", text)).strip()


class SherpaSession:
    append_only = False

    def __init__(self, recognizer):
        self.recognizer = recognizer
        self.stream = recognizer.create_stream()
        self.last = ""

    def _decode(self) -> list[tuple[str, bool]]:
        r, s = self.recognizer, self.stream
        while r.is_ready(s):
            r.decode_stream(s)
        text = clean(r.get_result(s))
        out = []
        if r.is_endpoint(s):
            if text:
                out.append((text, True))
            r.reset(s)
            self.last = ""
        elif text != self.last:
            out.append((text, False))
            self.last = text
        return out

    def feed(self, samples) -> list[tuple[str, bool]]:
        self.stream.accept_waveform(SAMPLE_RATE, samples)
        return self._decode()

    def close(self) -> list[tuple[str, bool]]:
        import numpy as np

        tail = np.zeros(int(TAIL_PADDING_S * SAMPLE_RATE), dtype=np.float32)
        self.stream.accept_waveform(SAMPLE_RATE, tail)
        self.stream.input_finished()
        r, s = self.recognizer, self.stream
        while r.is_ready(s):
            r.decode_stream(s)
        text = clean(r.get_result(s))
        return [(text, True)] if text else []


class SherpaOnlineBackend(StreamBackend):
    """sherpa-onnx OnlineRecognizer on the CPU. Catalog options: `kind` ("transducer" or
    "paraformer"), the model files (`encoder`, `decoder`, `joiner`, `tokens`, relative to the
    model folder) and `endpoint` {"rule1", "rule2", "rule3"}: trailing silence (s) that ends
    an utterance with nothing decoded / after some text, and the longest utterance (s)."""

    def load(self, spec, path):
        import sherpa_onnx

        o, root = spec.options, Path(path)
        rules = {"rule1": 2.4, "rule2": 1.2, "rule3": 20.0} | o.get("endpoint", {})
        common = {
            "tokens": str(root / o["tokens"]),
            "encoder": str(root / o["encoder"]),
            "decoder": str(root / o["decoder"]),
            "num_threads": o.get("num_threads", max(1, min(4, (os.cpu_count() or 4) // 4))),
            "sample_rate": SAMPLE_RATE,
            "feature_dim": 80,
            "enable_endpoint_detection": True,
            "rule1_min_trailing_silence": rules["rule1"],
            "rule2_min_trailing_silence": rules["rule2"],
            "rule3_min_utterance_length": rules["rule3"],
            "decoding_method": "greedy_search",
        }
        missing = [f for k in ("tokens", "encoder", "decoder", "joiner") if (f := o.get(k))
                   and not (root / f).exists()]  # fmt: skip
        if missing:
            raise RuntimeError(f"model files missing: {missing}")
        if o.get("kind") == "paraformer":
            recognizer = sherpa_onnx.OnlineRecognizer.from_paraformer(**common)
        else:
            recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                joiner=str(root / o["joiner"]), **common
            )
        return {"recognizer": recognizer}, "cpu"

    def session(self, handle):
        return SherpaSession(handle["recognizer"])


# -- Voxtral Realtime (mlx-audio) --------------------------------------------------------------


class VoxtralSession:
    """mlx-audio's VoxtralStreamingSession: feed() queues audio, step() encodes what arrived
    and decodes up to the audio available; text is only ever appended.

    The text is decoded here from the generated token ids with an incremental UTF-8 decoder:
    mlx-audio's own deltas slice successive decodes, so a Chinese character split across two
    byte-level tokens leaves a replacement character behind and loses the real one."""

    append_only = True

    def __init__(self, model, delay_ms: int | None, cache_limit_gb: float | None = None):
        import codecs

        # Decoding frame by frame churns through MLX buffers; with MLX's default (unlimited)
        # cache the process footprint grew to 40-48 GB on an 88 s recording, against ~5 GB
        # of arrays actually in use. A cache limit for the session keeps it near that, at no
        # measured speed cost; the previous limit is restored on close.
        self.previous_limit = None
        if cache_limit_gb:
            import mlx.core as mx

            self.previous_limit = mx.set_cache_limit(int(cache_limit_gb * 2**30))
        self.inner = model.create_streaming_session(transcription_delay_ms=delay_ms)
        self.tokenizer = model._tokenizer
        self.utf8 = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.used = 0  # generated tokens already turned into text

    def _text(self, final: bool = False) -> list[tuple[str, bool]]:
        tokens = self.inner.generated[self.used :]
        self.used += len(tokens)
        data = b"".join(self.tokenizer.token_bytes(int(t)) for t in tokens)  # specials: b""
        text = self.utf8.decode(data, final=final)
        return [(text, True)] if text else []

    def _run(self, final: bool = False) -> list[tuple[str, bool]]:
        with mlx_turn():
            while not self.inner.done:
                before = (self.inner._pos, len(self.inner.generated))
                self.inner.step(max_decode_tokens=32)
                if (self.inner._pos, len(self.inner.generated)) == before:
                    break  # waiting for more audio
        return self._text(final)

    def feed(self, samples) -> list[tuple[str, bool]]:
        self.inner.feed(samples)
        return self._run()

    def close(self) -> list[tuple[str, bool]]:
        self.inner.close()
        try:
            return self._run(final=True)
        finally:
            if self.previous_limit is not None:
                import mlx.core as mx

                mx.set_cache_limit(self.previous_limit)
                mx.clear_cache()
                self.previous_limit = None


class MlxRealtimeBackend(StreamBackend):
    """Voxtral Mini 4B Realtime via mlx-audio on the GPU. Options: `transcription_delay_ms`
    (80-2400, multiples of 80; 480 recommended) trades latency for accuracy; `cache_limit_gb`
    caps MLX's buffer cache during a session (see VoxtralSession)."""

    append_only = True

    def load(self, spec, path):
        from mlx_audio.stt.utils import load_model

        model = load_model(Path(path))
        if not hasattr(model, "create_streaming_session"):
            raise RuntimeError("this mlx-audio model has no streaming session")
        options = spec.options
        return {"model": model, "delay": options.get("transcription_delay_ms"),
                "cache_limit_gb": options.get("cache_limit_gb")}, "mlx"  # fmt: skip

    def session(self, handle):
        return VoxtralSession(handle["model"], handle["delay"], handle.get("cache_limit_gb"))


STREAM_BACKENDS = {
    "sherpa-online": SherpaOnlineBackend(),
    "mlx-audio-realtime": MlxRealtimeBackend(),
}
