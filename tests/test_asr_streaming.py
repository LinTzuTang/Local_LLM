"""True streaming ASR metrics: simulated time, VAD utterances, latency, instability."""

import numpy as np

from local_llm import asr_streaming as ast


def ev(t, text, final=False, seg=0):
    return {"t_audio": t, "t_wall": t, "text": text, "final": final, "seg": seg}


def test_schedule_waits_for_audio_and_for_the_previous_frame():
    # Fast frames finish right after they arrive; a slow one pushes the next ones back.
    assert ast.schedule([0.2, 0.4, 0.6], [0.05, 0.3, 0.05]) == [0.25, 0.7, 0.75]
    assert ast.frame_bounds(10, 0.25, rate=16) == [(0, 4), (4, 8), (8, 10)]


def test_speech_segments_find_utterances_and_bridge_short_pauses():
    rate = 1000
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.001, 6 * rate).astype(np.float32)
    voice = rng.normal(0, 0.3, 6 * rate).astype(np.float32)
    x = noise.copy()
    for a, b in ((0.5, 1.5), (1.8, 2.5), (4.0, 5.0)):  # 0.3 s pause is bridged, 1.5 s is not
        x[int(a * rate) : int(b * rate)] = voice[int(a * rate) : int(b * rate)]
    segments = ast.speech_segments(x, rate=rate)
    assert len(segments) == 2
    (s1, e1), (s2, e2) = segments
    assert abs(s1 - 0.5) < 0.05 and abs(e1 - 2.5) < 0.05
    assert abs(s2 - 4.0) < 0.05 and abs(e2 - 5.0) < 0.05
    assert all(type(t) is float for seg in segments for t in seg)
    assert ast.speech_segments(np.zeros(3 * rate, dtype=np.float32), rate=rate) == []


def test_displays_join_utterances_and_append_only_pieces():
    events = [ev(1, "各位"), ev(2, "各位早安", True), ev(3, "HELLO"), ev(4, "HELLO THERE", True)]
    assert ast.displays(events) == ["各位", "各位早安", "各位早安HELLO", "各位早安HELLO THERE"]
    pieces = [ev(1, " Good", True), ev(2, " morning", True), ev(3, ".", True)]
    assert ast.transcript(pieces, append_only=True) == "Good morning."
    assert ast.caption_at(events, 2.5) == "各位早安"
    assert ast.transcript([]) == ""


def test_instability_counts_characters_taken_back():
    # 今天 -> 今天天氣 (+2) -> 今天天汽很 (氣 taken back, 汽很 added) -> final
    events = [ev(1, "今天"), ev(2, "今天天氣"), ev(3, "今天天汽很"), ev(4, "今天天氣很好", True)]
    out = ast.instability(events)
    assert out["revised"] == 3  # 氣, then 汽很
    assert out["shown"] == 2 + 2 + 2 + 3
    assert out["rate"] == round(3 / 9, 4)
    stable = [ev(1, "a", True), ev(2, "b", True)]
    assert ast.instability(stable, append_only=True)["rate"] == 0
    assert ast.instability([])["rate"] is None


def test_first_partial_and_finalisation_latency():
    segments = [(1.0, 3.0), (5.0, 7.0), (7.5, 9.0)]
    events = [
        ev(1.6, ""),
        ev(1.9, "各位"),
        ev(3.9, "各位早安", True),  # 0.9 s after the first utterance ended
        ev(6.0, "今天"),
        ev(9.8, "今天開會", True),  # utterances 2 and 3 committed together at 9.8
    ]
    # A final just before the VAD's end of speech still finalises that utterance (latency 0).
    early = [ev(2.9, "各位早安", True), ev(9.0, "今天開會", True)]
    assert [r["latency"] for r in ast.finalisation(early, segments[:2])][0] == 0.0
    # Slower than real time: text for the first utterance (heard by t_audio 3.2) only appears
    # at t_wall 8, during the second utterance's pause; it must count as 5 s late, not early.
    lagging = [{"t_audio": 3.2, "t_wall": 8.0, "text": "各位", "final": True},
               {"t_audio": 7.2, "t_wall": 15.0, "text": "今天", "final": True}]  # fmt: skip
    assert [r["latency"] for r in ast.finalisation(lagging, segments[:2])] == [5.0, 8.0]
    assert ast.first_partial(events, 1.0) == 0.9
    assert ast.first_partial([], 1.0) is None
    rows = ast.finalisation(events, segments)
    assert [r["latency"] for r in rows] == [0.9, 2.8, 0.8]
    out = ast.summary(events, segments, duration=9.5, compute_s=0.95)
    assert out["first_partial_s"] == 0.9 and out["final_median_s"] == 0.9
    assert out["rtf"] == 0.1 and out["utterances"] == 3 and out["finalised"] == 3
    assert out["lag_at_end_s"] == 0.3
    assert ast.percentile([1, 2, 3, 4], 50) == 2.5 and ast.percentile([], 90) is None
