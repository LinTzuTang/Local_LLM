"""Scoring true streaming speech recognition (sherpa-onnx Zipformer / Paraformer, Voxtral).

A streaming run feeds a recording to the model in short frames (100-200 ms), exactly as a
microphone would, and records an event whenever the text changes:

    {"t_audio": 3.36, "t_wall": 3.41, "text": "各位早", "final": False, "seg": 0}

`t_audio` is how much audio had arrived, `t_wall` when the text was available, both in seconds
from the start of the recording. `text` is the hypothesis for the current utterance `seg`;
`final` marks the model committing it (sherpa-onnx endpointing). Models that never revise
(Voxtral) send every piece as final; their pieces are concatenated as they are
(`append_only`), the others' utterances are joined like chunk transcripts.

Feeding faster than real time saves wall-clock time, so `t_wall` is usually *simulated*
(`schedule`): frame i arrives at its end time, is processed once it has arrived and the
previous frame is done, and takes the compute time measured for it. That is the moment a live
listener would have seen the text on a machine doing nothing else.

The metrics (`summary`):
- first partial: speech start (energy VAD on the input) -> first non-empty text;
- finalisation latency, per utterance: end of speech (VAD) -> the final text for it: the last
  final event made while the audio was in the pause after it (by t_audio), or, when the model
  only commits later, the first final event after the speech ended; measured on t_wall, so a
  model slower than real time shows its growing backlog;
- RTF: compute time / audio length (below 1 keeps up);
- partial instability: characters shown on screen and later taken back, as a share of all
  characters shown (0 for append-only models).
"""

import statistics

from local_llm.diarization import join_text

RATE = 16000


def schedule(arrivals: list[float], compute: list[float]) -> list[float]:
    """Simulated time each frame's output is ready: frame i starts once it has arrived
    (`arrivals[i]`) and frame i-1 is done, and takes `compute[i]` seconds."""
    out, busy = [], 0.0
    for arrival, seconds in zip(arrivals, compute, strict=True):
        busy = max(arrival, busy) + seconds
        out.append(busy)
    return out


def frame_bounds(n_samples: int, frame_s: float, rate: int = RATE) -> list[tuple[int, int]]:
    """(start, end) sample indices of consecutive frames of `frame_s` seconds."""
    step = max(1, int(round(frame_s * rate)))
    return [(a, min(a + step, n_samples)) for a in range(0, n_samples, step)]


# -- utterances in the input (a small energy VAD) -----------------------------------------------


def speech_segments(
    samples,
    rate: int = RATE,
    frame_s: float = 0.03,
    min_gap: float = 0.6,
    min_speech: float = 0.25,
) -> list[tuple[float, float]]:
    """(start, end) seconds of speech: 30 ms frames louder than a threshold set between the
    noise floor (10th percentile of frame energy) and the loud speech level (95th
    percentile), with pauses shorter than `min_gap` bridged and blips under `min_speech`
    dropped. Enough to find where utterances end in read speech and meetings."""
    import numpy as np

    x = np.asarray(samples, dtype=np.float32)
    size = max(1, int(frame_s * rate))
    n = len(x) // size
    if n == 0:
        return []
    energy = (x[: n * size].reshape(n, size) ** 2).mean(axis=1)
    db = 10 * np.log10(energy + 1e-10)
    floor, loud = np.percentile(db, 10), np.percentile(db, 95)
    if loud - floor < 6:  # flat: silence or constant noise
        return []
    voiced = db > floor + 0.3 * (loud - floor)
    segments: list[list[float]] = []
    for i in np.flatnonzero(voiced):
        start, end = i * frame_s, (i + 1) * frame_s
        if segments and start - segments[-1][1] < min_gap:
            segments[-1][1] = end
        else:
            segments.append([start, end])
    return [(round(float(a), 2), round(float(b), 2)) for a, b in segments if b - a >= min_speech]


# -- text over time -----------------------------------------------------------------------------


def _join(parts: list[str], append_only: bool) -> str:
    if append_only:
        return "".join(parts).strip()
    out = ""
    for p in parts:
        out = join_text(out, (p or "").strip())
    return out


def displays(events: list[dict], append_only: bool = False) -> list[str]:
    """The text on screen after each event: committed utterances plus the current partial."""
    committed: list[str] = []
    partial, out = "", []
    for e in events:
        if e.get("final"):
            committed.append(e.get("text") or "")
            partial = ""
        else:
            partial = e.get("text") or ""
        out.append(_join([*committed, partial], append_only))
    return out


def transcript(events: list[dict], append_only: bool = False) -> str:
    """The final transcript: every committed utterance (plus a trailing partial, if the run
    stopped before the model committed it)."""
    shown = displays(events, append_only)
    return shown[-1] if shown else ""


def caption_at(events: list[dict], t: float, append_only: bool = False) -> str:
    """What a viewer saw `t` seconds after the speaker started (for replaying a run)."""
    seen = [e for e in events if e.get("t_wall", 0) <= t]
    return transcript(seen, append_only)


def instability(events: list[dict], append_only: bool = False) -> dict:
    """{"shown", "revised", "rate"}: characters newly put on screen, characters later taken
    back (the screen text lost them when the next text did not start with them), and
    revised / shown. Spaces are ignored."""
    shown = revised = 0
    before = ""
    for text in displays(events, append_only):
        text = text.replace(" ", "")
        common = 0
        for a, b in zip(before, text, strict=False):
            if a != b:
                break
            common += 1
        revised += len(before) - common
        shown += len(text) - common
        before = text
    return {"shown": shown, "revised": revised,
            "rate": round(revised / shown, 4) if shown else None}  # fmt: skip


def first_partial(events: list[dict], speech_start: float | None) -> float | None:
    """Seconds from the start of speech to the first non-empty text."""
    if speech_start is None:
        return None
    for e in events:
        if (e.get("text") or "").strip():
            return round(max(e["t_wall"] - speech_start, 0.0), 3)
    return None


# A model can commit an utterance a little before the energy VAD sees its speech end (frame
# and VAD resolution, a trailing breath); such a final still counts for that utterance.
FINAL_TOLERANCE_S = 0.3


def finalisation(
    events: list[dict], segments: list[tuple[float, float]], tolerance: float = FINAL_TOLERANCE_S
) -> list[dict]:
    """Per utterance {"end", "final_at", "latency"}: from the end of speech to the final text
    for it (see the module docstring); a final up to `tolerance` s before the end counts, with
    latency 0. Utterances with no final text after them are left out."""
    # Which utterance a final belongs to goes by the audio it had heard (t_audio); when it
    # appeared, by t_wall. A model that falls behind real time emits old text late, and must
    # not look fast because that late text happens to appear during a later pause.
    finals = final_times(events)
    out = []
    for i, (_, end) in enumerate(segments):
        nxt = segments[i + 1][0] if i + 1 < len(segments) else float("inf")
        if (row := finalise_one(finals, end, nxt, tolerance)) is not None:
            out.append(row)
    return out


def final_times(events: list[dict]) -> list[tuple[float, float]]:
    """(t_audio, t_wall) of the final events that carry text."""
    return [(e["t_audio"], e["t_wall"]) for e in events
            if e.get("final") and (e.get("text") or "").strip()]  # fmt: skip


def finalise_one(
    finals: list[tuple[float, float]],
    end: float,
    next_start: float = float("inf"),
    tolerance: float = FINAL_TOLERANCE_S,
) -> dict | None:
    """One utterance's finalisation (see `finalisation`), or None while no final text for it
    has appeared yet."""
    in_gap = [w for a, w in finals if end - tolerance <= a < next_start]
    later = [w for a, w in finals if a >= end - tolerance]
    at = max(in_gap) if in_gap else (min(later) if later else None)
    if at is None:
        return None
    return {"end": end, "final_at": round(at, 3), "latency": round(max(at - end, 0.0), 3)}


def percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolated percentile (q in 0-100) of a non-empty list, else None."""
    if not values:
        return None
    xs = sorted(values)
    k = (len(xs) - 1) * q / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), 3)


def summary(
    events: list[dict],
    segments: list[tuple[float, float]],
    duration: float,
    compute_s: float,
    append_only: bool = False,
) -> dict:
    """The headline numbers of one streaming run."""
    lat = [f["latency"] for f in finalisation(events, segments)]
    unstable = instability(events, append_only)
    end_lag = round(events[-1]["t_wall"] - duration, 3) if events else None
    return {
        "first_partial_s": first_partial(events, segments[0][0] if segments else None),
        "final_median_s": round(statistics.median(lat), 3) if lat else None,
        "final_p90_s": percentile(lat, 90),
        "final_max_s": round(max(lat), 3) if lat else None,
        "utterances": len(segments),
        "finalised": len(lat),
        "rtf": round(compute_s / duration, 4) if duration > 0 else None,
        "compute_s": round(compute_s, 3),
        "instability": unstable["rate"],
        "revised_chars": unstable["revised"],
        "shown_chars": unstable["shown"],
        "lag_at_end_s": end_lag,
        "events": len(events),
    }


# -- live microphone input ----------------------------------------------------------------------


def pcm16(data: bytes) -> tuple:
    """Little-endian 16-bit PCM bytes -> (float32 samples in [-1, 1), leftover odd byte)."""
    import numpy as np

    whole = len(data) - len(data) % 2
    samples = np.frombuffer(data[:whole], dtype="<i2").astype(np.float32) / 32768.0
    return samples, data[whole:]


class OnlineVad:
    """Utterance starts and ends while audio streams in, for live latency readouts.

    30 ms frames; a frame is speech when it is `margin_db` above a noise floor that follows
    quiet frames down at once and creeps up slowly (so it cannot climb into long speech), and
    above `min_db` dBFS. Speech starts after three speech frames in a row (reported at the
    first of them) and ends after `min_gap` s without speech (reported at the last speech
    frame), as speech_segments does for a whole recording."""

    def __init__(self, rate: int = RATE, frame_s: float = 0.03, min_gap: float = 0.6,
                 margin_db: float = 12.0, min_db: float = -55.0):  # fmt: skip
        self.rate, self.size = rate, int(frame_s * rate)
        self.frame_s, self.min_gap, self.margin, self.min_db = frame_s, min_gap, margin_db, min_db
        self.rest = None
        self.frames = 0
        self.floor: float | None = None
        self.run = 0  # consecutive speech frames while not in speech
        self.in_speech = False
        self.last_voiced = 0.0

    def feed(self, samples) -> list[tuple[str, float]]:
        """[("start" | "end", seconds)] for the frames completed by these samples."""
        import numpy as np

        x = np.asarray(samples, dtype=np.float32)
        if self.rest is not None and len(self.rest):
            x = np.concatenate([self.rest, x])
        n = len(x) // self.size
        self.rest = x[n * self.size :]
        out = []
        for frame in x[: n * self.size].reshape(n, self.size) if n else []:
            db = float(10 * np.log10(float((frame**2).mean()) + 1e-10))
            end = (self.frames + 1) * self.frame_s
            self.frames += 1
            self.floor = db if self.floor is None or db < self.floor else self.floor + 0.01
            voiced = db > max(self.floor + self.margin, self.min_db)
            if voiced:
                self.last_voiced = end
                if not self.in_speech:
                    self.run += 1
                    if self.run >= 3:
                        self.in_speech, self.run = True, 0
                        out.append(("start", round(end - 3 * self.frame_s, 3)))
            else:
                self.run = 0
                if self.in_speech and end - self.last_voiced >= self.min_gap:
                    self.in_speech = False
                    out.append(("end", round(self.last_voiced, 3)))
        return out

    def flush(self) -> list[tuple[str, float]]:
        """End the utterance in progress when the stream stops."""
        if self.in_speech:
            self.in_speech = False
            return [("end", round(self.last_voiced, 3))]
        return []
