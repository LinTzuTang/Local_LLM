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


# Voxtral restarts only while something is audible: the background talk it locks up on is far
# louder (babble at 10 dB SNR about -30 dBFS) than a quiet room (about -50 dBFS)
SILENCE_RMS = 10 ** (-40 / 20)


class VoxtralSession:
    """mlx-audio's VoxtralStreamingSession: feed() queues audio, step() encodes what arrived
    and decodes up to the audio available; text is only ever appended.

    The text is decoded here from the generated token ids with an incremental UTF-8 decoder:
    mlx-audio's own deltas slice successive decodes, so a Chinese character split across two
    byte-level tokens leaves a replacement character behind and loses the real one.

    `restart_quiet_s`: when the model has written nothing but padding for that long while audio
    keeps coming, the session is replaced by a fresh one that is first given those seconds of
    audio again (the old session wrote nothing for them, so nothing repeats; the replayed part
    does not count towards the next restart). With other people talking in the background
    Voxtral can settle into "nobody is speaking" at the start of a stream and stay there: on
    the 92 s noise clips the 4-bit model wrote nothing after 27 s (babble 10 dB CER 80.5%,
    office 5 dB 82.1%) or nothing at all (babble 0 dB). Whether it locks up flips with tiny
    numeric changes (8-bit locks up too, 6-bit does not; other delays differ), so tuning does
    not fix it. With 3 s and the replay the three clips came to 3.5%, 4.3% and 16.0%.
    Restarting without the replay lost the words in the quiet window (National Day 5.0%,
    AliMeeting 59.0%).

    `restart_context_s`: a fresh session does not know what was said before; restarted in a
    pause, its first words could come out wrong or in another language ("面對缺工" as
    "Mentre ciego"; National Day 3.7 -> 4.9%). So when the old session had been writing text
    in the seconds before it went quiet, the fresh one first hears that much earlier speech,
    and the text it writes for it is dropped. Not when it had written nothing: then the
    earlier audio is what it locked up on, and hearing it again locks the new one too
    (babble 10 dB 3.5 -> 15.6% when always added). With 10 s: National Day 3.0%, AliMeeting
    50.2 -> 47.7%, synthetic meeting 14.2 -> 14.5%, English and the noise clips as above."""

    append_only = True

    def __init__(self, model, delay_ms: int | None, cache_limit_gb: float | None = None,
                 restart_quiet_s: float | None = None, restart_context_s: float = 0.0):  # fmt: skip
        import codecs

        # Decoding frame by frame churns through MLX buffers; with MLX's default (unlimited)
        # cache the process footprint grew to 40-48 GB on an 88 s recording, against ~5 GB
        # of arrays actually in use. A cache limit for the session keeps it near that, at no
        # measured speed cost; the previous limit is restored on close.
        self.previous_limit = None
        if cache_limit_gb:
            import mlx.core as mx

            self.previous_limit = mx.set_cache_limit(int(cache_limit_gb * 2**30))
        self.model, self.delay_ms = model, delay_ms
        self.inner = model.create_streaming_session(transcription_delay_ms=delay_ms)
        self.tokenizer = model._tokenizer
        self.utf8 = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.used = 0  # generated tokens already turned into text
        self.pad = getattr(getattr(model, "config", None), "streaming_pad_token_id", None)
        # the model writes one token per 80 ms of audio
        self.quiet_tokens = int(restart_quiet_s / 0.08) if restart_quiet_s else 0
        self.context_tokens = int(restart_context_s / 0.08) if restart_quiet_s else 0
        self.quiet_samples = int((restart_quiet_s or 0) * SAMPLE_RATE)
        self.context_samples = int(restart_context_s * SAMPLE_RATE) if restart_quiet_s else 0
        self.keep = self.quiet_samples + self.context_samples
        self.eos = getattr(getattr(model, "config", None), "eos_token_id", None)
        self.recent: list = []  # the last `keep` samples, replayed into a fresh session
        self.skip = 0  # tokens of the replayed audio, not counted as new silence
        self.fresh = 0  # tokens before this are the replayed context's: not "heard" again
        self.restarts = 0

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
        out = self._run()
        if self.quiet_tokens and self.pad is not None:
            import numpy as np

            self.recent.append(np.asarray(samples, dtype=np.float32))
            while sum(len(r) for r in self.recent[1:]) >= self.keep:
                self.recent.pop(0)
            old = self.inner.generated
            tail = old[self.skip :][-self.quiet_tokens :]
            if len(tail) == self.quiet_tokens and all(int(t) == self.pad for t in tail):
                q, k = self.quiet_tokens, self.context_tokens
                window = np.concatenate(self.recent)[-self.quiet_samples :]
                whole = len(window) // SAMPLE_RATE * SAMPLE_RATE
                levels = np.sqrt((window[:whole].reshape(-1, SAMPLE_RATE) ** 2).mean(axis=1))
                if not len(levels) or float(levels.min()) < SILENCE_RMS:
                    # A quiet second in the window: nobody was talking all along, so writing
                    # nothing is no lock-up (that is background talk going on and on). Without
                    # this the session restarted in silences and as a line began, before its
                    # first words could come out, each time replaying seconds of audio: between
                    # the assistant's replies the recogniser fell 2-6 s behind (the end of a
                    # line, and a barge-in, then came that late). On the test script's lines
                    # with silence between them: 140 ms of work per 100 ms of speech with
                    # those restarts, 45 ms without.
                    self.skip = len(old)  # look again after another quiet window
                    return out
                # what it wrote just before going quiet, not counting the replayed context:
                # else every restart in a silence replays the context again, and the backlog
                # grows by those seconds each time (live assistant: the ASR fell 10 s further
                # behind with every line of the test script)
                before = old[self.fresh :][-(q + k) : -q]
                heard = any(int(t) not in (self.pad, self.eos) for t in before)
                audio = np.concatenate(self.recent)
                quiet = audio[-self.quiet_samples :]
                context = audio[: len(audio) - len(quiet)][-self.context_samples :]
                if not (heard and self.context_samples):
                    context = audio[:0]
                with mlx_turn():
                    self.inner = self.model.create_streaming_session(
                        transcription_delay_ms=self.delay_ms
                    )
                self.inner.feed(np.concatenate([context, quiet]))
                drop = int(len(context) / SAMPLE_RATE / 0.08)  # the context's tokens: text dropped
                self.used, self.skip, self.fresh = drop, drop + q, drop
                self.restarts += 1
        return out

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
    caps MLX's buffer cache during a session; `restart_quiet_s` starts a fresh session after
    that many seconds of padding only, given `restart_context_s` of earlier speech first (see
    VoxtralSession)."""

    append_only = True

    def load(self, spec, path):
        from mlx_audio.stt.utils import load_model

        model = load_model(Path(path))
        if not hasattr(model, "create_streaming_session"):
            raise RuntimeError("this mlx-audio model has no streaming session")
        options = spec.options
        return {"model": model, "delay": options.get("transcription_delay_ms"),
                "cache_limit_gb": options.get("cache_limit_gb"),
                "restart_quiet_s": options.get("restart_quiet_s"),
                "restart_context_s": options.get("restart_context_s", 0.0)}, "mlx"  # fmt: skip

    def session(self, handle):
        return VoxtralSession(handle["model"], handle["delay"], handle.get("cache_limit_gb"),
                              handle.get("restart_quiet_s"),
                              handle.get("restart_context_s", 0.0))  # fmt: skip


STREAM_BACKENDS = {
    "sherpa-online": SherpaOnlineBackend(),
    "mlx-audio-realtime": MlxRealtimeBackend(),
}
