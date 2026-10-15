"""Speaker diarization helpers: speaker-labelled transcripts and the diarization error rate.

The gateway's /v1/audio/diarize returns who spoke when, `[{"start", "end", "speaker"}]`
(gateway_client.diarize); a speech model returns what was said, with timed segments. This
module joins the two and scores diarization against a reference:

    turns = diarization.label_segments(diar["segments"], asr["segments"])
    diarization.transcript(turns)            # "Speaker 1: ...\nSpeaker 2: ..."
    diarization.der(reference, diar["segments"], collar=0.25)["der"]

References are RTTM (`SPEAKER <file> 1 <start> <duration> <NA> <NA> <name> <NA> <NA>`) or
plain lines `<start> <end> <speaker>` (seconds or m:ss), parsed by `parse_reference`.
"""

import re
from collections.abc import Iterable

FRAME_S = 0.01


def overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def speaker_of(start: float, end: float, turns: list[dict]):
    """The speaker who talks most during [start, end]; the nearest turn when none overlaps."""
    if not turns:
        return None
    talk: dict = {}
    for t in turns:
        talk[t["speaker"]] = talk.get(t["speaker"], 0.0) + overlap(start, end, t["start"], t["end"])
    best = max(talk, key=talk.get)
    if talk[best] > 0:
        return best
    middle = (start + end) / 2
    nearest = min(turns, key=lambda t: min(abs(middle - t["start"]), abs(middle - t["end"])))
    return nearest["speaker"]


def label_segments(turns: list[dict], segments: list[dict]) -> list[dict]:
    """ASR segments with the speaker who talks most during each one; consecutive segments of
    the same speaker are joined. Returns [{"start", "end", "speaker", "text"}]."""
    out: list[dict] = []
    for s in sorted(segments, key=lambda s: s["start"]):
        text = str(s.get("text") or "").strip()
        if not text:
            continue
        who = speaker_of(s["start"], s["end"], turns)
        if out and out[-1]["speaker"] == who:
            out[-1]["end"] = s["end"]
            out[-1]["text"] = join_text(out[-1]["text"], text)
        else:
            out.append({"start": s["start"], "end": s["end"], "speaker": who, "text": text})
    return out


def join_text(a: str, b: str) -> str:
    """Join two pieces of transcript: no space between Chinese characters, one between words."""
    if not a or not b:
        return a or b
    cjk = re.compile(r"[　-〿㐀-鿿＀-￯]")
    return a + ("" if cjk.match(a[-1]) or cjk.match(b[0]) else " ") + b


def merge_turns(turns: list[dict], gap: float = 0.5) -> list[dict]:
    """Join consecutive turns of the same speaker separated by less than `gap` seconds."""
    out: list[dict] = []
    for t in sorted(turns, key=lambda t: t["start"]):
        if out and out[-1]["speaker"] == t["speaker"] and t["start"] - out[-1]["end"] < gap:
            out[-1]["end"] = max(out[-1]["end"], t["end"])
        else:
            out.append({"start": t["start"], "end": t["end"], "speaker": t["speaker"]})
    return out


def speaker_names(speakers: Iterable, template: str = "Speaker {n}") -> dict:
    """{speaker id: "Speaker 1", ...} numbered in order of first appearance."""
    names: dict = {}
    for s in speakers:
        if s not in names:
            names[s] = template.format(n=len(names) + 1)
    return names


def transcript(labelled: list[dict], template: str = "Speaker {n}", sep: str = ": ") -> str:
    """One "Speaker 1: ..." line per turn."""
    names = speaker_names((t["speaker"] for t in labelled), template)
    return "\n".join(f"{names[t['speaker']]}{sep}{t['text']}" for t in labelled)


# -- reference files ------------------------------------------------------------------------

_TIME = r"(\d+(?::\d{1,2}){0,2}(?:\.\d+)?)"
_LINE = re.compile(rf"^\s*{_TIME}\s*(?:-|–|to|\s)\s*{_TIME}\s*[\s,:|]\s*(.+?)\s*$")


def seconds(value: str) -> float:
    """Seconds from "75.5", "1:15.5" or "0:01:15.5"."""
    total = 0.0
    for part in value.split(":"):
        total = total * 60 + float(part)
    return total


def parse_reference(text: str) -> list[dict]:
    """Reference speaker turns from RTTM or `<start> <end> <speaker>` lines (seconds or m:ss;
    "0:12-0:18 Alice" also works). Blank lines and lines starting with # or ;; are skipped."""
    turns = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", ";;")):
            continue
        fields = line.split()
        if fields[0].upper() == "SPEAKER" and len(fields) >= 8:
            start, duration = float(fields[3]), float(fields[4])
            turns.append({"start": start, "end": start + duration, "speaker": fields[7]})
            continue
        match = _LINE.match(line)
        if not match:
            raise ValueError(f"cannot read reference line: {raw!r}")
        start, end, who = seconds(match[1]), seconds(match[2]), match[3]
        if end <= start:
            raise ValueError(f"end before start: {raw!r}")
        turns.append({"start": start, "end": end, "speaker": who})
    return sorted(turns, key=lambda t: t["start"])


def to_rttm(turns: list[dict], file_id: str = "audio", names: dict | None = None) -> str:
    rows = []
    for t in turns:
        who = (names or {}).get(t["speaker"], t["speaker"])
        rows.append(
            f"SPEAKER {file_id} 1 {t['start']:.2f} {t['end'] - t['start']:.2f} <NA> <NA> "
            f"{str(who).replace(' ', '_')} <NA> <NA>"
        )
    return "\n".join(rows) + ("\n" if rows else "")


# -- diarization error rate ------------------------------------------------------------------


def _frames(turns: list[dict], n: int, speakers: list):
    """Speakers x frames boolean activity matrix."""
    import numpy as np

    index = {s: i for i, s in enumerate(speakers)}
    grid = np.zeros((len(speakers), n), dtype=bool)
    for t in turns:
        a, b = max(round(t["start"] / FRAME_S), 0), min(round(t["end"] / FRAME_S), n)
        grid[index[t["speaker"]], a:b] = True
    return grid


def der(reference: list[dict], hypothesis: list[dict], collar: float = 0.0) -> dict:
    """Diarization error rate = (missed speech + false alarm + speaker confusion) / reference
    speech, counted in 10 ms frames, after mapping hypothesis speakers to reference speakers
    one-to-one so that they agree as much as possible (Hungarian algorithm). Overlapping speech
    counts once per speaker. `collar` seconds around each reference boundary are not scored
    (0.25 is the usual NIST setting). Returns {"der", "miss", "false_alarm", "confusion",
    "speech", "mapping"}, the error parts as fractions of the reference speech."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    end = max([t["end"] for t in [*reference, *hypothesis]], default=0.0)
    n = int(np.ceil(end / FRAME_S)) + 1
    ref_spk = sorted({t["speaker"] for t in reference}, key=str)
    hyp_spk = sorted({t["speaker"] for t in hypothesis}, key=str)
    ref, hyp = _frames(reference, n, ref_spk), _frames(hypothesis, n, hyp_spk)
    scored = np.ones(n, dtype=bool)
    if collar > 0:
        c = round(collar / FRAME_S)
        for t in reference:
            for edge in (round(t["start"] / FRAME_S), round(t["end"] / FRAME_S)):
                scored[max(edge - c, 0) : edge + c] = False
    ref, hyp = ref[:, scored], hyp[:, scored]
    mapping: dict = {}
    if ref_spk and hyp_spk:
        agree = ref.astype(np.int64) @ hyp.T.astype(np.int64)  # frames both speak
        rows, cols = linear_sum_assignment(-agree)
        mapping = {hyp_spk[c]: ref_spk[r] for r, c in zip(rows, cols, strict=True)}
        correct = int(agree[rows, cols].sum())
    else:
        correct = 0
    n_ref, n_hyp = ref.sum(axis=0), hyp.sum(axis=0)
    speech = int(n_ref.sum())
    miss = int(np.maximum(n_ref - n_hyp, 0).sum())
    false_alarm = int(np.maximum(n_hyp - n_ref, 0).sum())
    confusion = int(np.minimum(n_ref, n_hyp).sum()) - correct
    total = max(speech, 1)
    return {
        "der": (miss + false_alarm + confusion) / total,
        "miss": miss / total,
        "false_alarm": false_alarm / total,
        "confusion": confusion / total,
        "speech": round(speech * FRAME_S, 2),
        "mapping": mapping,
    }
