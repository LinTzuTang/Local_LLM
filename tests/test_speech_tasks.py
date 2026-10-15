"""Pure logic behind the speech page: speaker-labelled transcripts, DER, translation prompts,
scoring, chunking and the live-input schedule."""

import pytest

from local_llm import diarization as dz
from local_llm import speech_translate as stx

TURNS = [
    {"start": 0.0, "end": 5.0, "speaker": 0},
    {"start": 5.5, "end": 9.0, "speaker": 1},
    {"start": 9.2, "end": 12.0, "speaker": 0},
]


def test_segments_get_the_speaker_who_talks_most():
    segments = [
        {"start": 0.0, "end": 2.4, "text": "各位早安，"},
        {"start": 2.4, "end": 5.2, "text": "開始開會。"},  # same speaker: joined
        {"start": 5.3, "end": 8.8, "text": "好的"},
        {"start": 8.9, "end": 11.9, "text": "OK"},
        {"start": 12.0, "end": 12.5, "text": " "},  # empty text: dropped
    ]
    out = dz.label_segments(TURNS, segments)
    assert [t["speaker"] for t in out] == [0, 1, 0]
    assert out[0]["text"] == "各位早安，開始開會。" and out[0]["end"] == 5.2
    assert dz.transcript(out) == "Speaker 1: 各位早安，開始開會。\nSpeaker 2: 好的\nSpeaker 1: OK"
    assert dz.transcript(out, "講者 {n}", "：").startswith("講者 1：")


def test_segment_without_overlap_takes_the_nearest_turn():
    assert dz.speaker_of(5.1, 5.4, TURNS) in (0, 1)
    assert dz.speaker_of(20.0, 21.0, TURNS) == 0  # after the last turn (speaker 0)
    assert dz.speaker_of(1.0, 2.0, []) is None


def test_join_text_spaces_only_between_words():
    assert dz.join_text("你好", "世界") == "你好世界"
    assert dz.join_text("hello", "world") == "hello world"
    assert dz.join_text("", "x") == "x"


def test_merge_turns_joins_short_gaps_of_one_speaker():
    turns = [*TURNS, {"start": 12.2, "end": 13.0, "speaker": 0}]
    merged = dz.merge_turns(turns, gap=0.5)
    assert len(merged) == 3 and merged[-1] == {"start": 9.2, "end": 13.0, "speaker": 0}


def test_speaker_names_follow_first_appearance():
    assert dz.speaker_names([3, 1, 3, 0]) == {3: "Speaker 1", 1: "Speaker 2", 0: "Speaker 3"}


def test_parse_reference_formats():
    rttm = (
        "SPEAKER m 1 0.00 5.00 <NA> <NA> 林課長 <NA> <NA>\n"
        "SPEAKER m 1 5.50 3.50 <NA> <NA> 陳經理 <NA> <NA>"
    )
    assert dz.parse_reference(rttm) == [
        {"start": 0.0, "end": 5.0, "speaker": "林課長"},
        {"start": 5.5, "end": 9.0, "speaker": "陳經理"},
    ]
    plain = "# start end speaker\n0 5 Alice\n0:05.5-0:09 Bob\n1:00 1:02.5 Alice Smith"
    turns = dz.parse_reference(plain)
    assert [t["speaker"] for t in turns] == ["Alice", "Bob", "Alice Smith"]
    assert turns[1]["start"] == 5.5 and turns[2]["end"] == 62.5
    with pytest.raises(ValueError):
        dz.parse_reference("nonsense")
    with pytest.raises(ValueError):
        dz.parse_reference("5 3 Alice")
    back = dz.parse_reference(dz.to_rttm(turns, "x"))
    assert [t["speaker"] for t in back] == ["Alice", "Bob", "Alice_Smith"]


def test_der_is_zero_for_a_perfect_relabelled_hypothesis():
    ref = [{**t, "speaker": "AB"[t["speaker"]]} for t in TURNS]
    out = dz.der(ref, TURNS)
    assert out["der"] == pytest.approx(0.0, abs=1e-9)
    assert out["mapping"] == {0: "A", 1: "B"}
    assert out["speech"] == pytest.approx(11.3, abs=0.02)


def test_der_counts_miss_false_alarm_and_confusion():
    ref = [{"start": 0, "end": 10, "speaker": "A"}, {"start": 10, "end": 20, "speaker": "B"}]
    # 2 s of A missed, 1 s of extra speech, and 4 s of B attributed to A's cluster.
    hyp = [
        {"start": 2, "end": 14, "speaker": 0},
        {"start": 14, "end": 21, "speaker": 1},
    ]
    out = dz.der(ref, hyp)
    assert out["miss"] == pytest.approx(2 / 20, abs=0.002)
    assert out["false_alarm"] == pytest.approx(1 / 20, abs=0.002)
    assert out["confusion"] == pytest.approx(4 / 20, abs=0.002)
    assert out["der"] == pytest.approx(7 / 20, abs=0.003)


def test_der_overlap_and_collar():
    ref = [{"start": 0, "end": 10, "speaker": "A"}, {"start": 5, "end": 10, "speaker": "B"}]
    hyp = [{"start": 0, "end": 10, "speaker": 0}]  # overlap: B is missed entirely
    assert dz.der(ref, hyp)["miss"] == pytest.approx(5 / 15, abs=0.002)
    shifted = [{"start": 0.2, "end": 10, "speaker": 0}, {"start": 5.2, "end": 10, "speaker": 1}]
    assert dz.der(ref, shifted)["der"] > 0
    assert dz.der(ref, shifted, collar=0.25)["der"] == pytest.approx(0, abs=1e-9)
    assert dz.der([], [])["der"] == 0


def test_translategemma_prompt_follows_the_model_card():
    prompt = stx.translation_prompt("各位早安", "translategemma:4b")
    assert prompt.startswith("You are a professional Chinese (zh-TW) to English (en) translator.")
    assert prompt.endswith("into English:\n\n\n各位早安")
    simplified = stx.translation_prompt("各位早安", "translategemma:12b", script="simplified")
    assert "(zh-Hans)" in simplified
    other = stx.translation_prompt("各位早安", "gemma4:12b")
    assert "Output only the English translation" in other and other.endswith("各位早安")


def test_clean_output():
    assert stx.clean_output('Translation: "Good morning."') == "Good morning."
    assert stx.clean_output("  Good morning  ") == "Good morning"


def test_score_with_and_without_reference():
    assert stx.score("Good morning everyone", "") is None
    same = stx.score("Good morning everyone.", "Good morning everyone.")
    assert same["chrf"] == 100.0 and same["bleu"] == 100.0
    worse = stx.score("Hello all.", "Good morning everyone.")
    assert worse["chrf"] < 50


def test_chunk_bounds_cover_the_audio():
    rate = 16000
    bounds = stx.chunk_bounds(rate * 12, 5)
    assert bounds == [(0, 80000), (80000, 160000), (160000, 192000)]
    # A 0.5 s tail joins the chunk before it.
    assert stx.chunk_bounds(int(rate * 10.5), 5)[-1] == (80000, 168000)
    assert stx.chunk_bounds(rate * 3, 5) == [(0, 48000)]
    with pytest.raises(ValueError):
        stx.chunk_bounds(100, 0)


def test_live_schedule_lag_grows_when_too_slow():
    fast = stx.live_schedule([5, 10, 15], [1, 1, 1])
    assert [r["lag"] for r in fast] == [1, 1, 1]
    slow = stx.live_schedule([5, 10, 15], [7, 7, 7])
    assert [r["start"] for r in slow] == [5, 12, 19]
    assert [r["lag"] for r in slow] == [7, 9, 11]


class FakeStep:
    def __init__(self):
        self.inputs, self.detail, self.stream, self.ollama_url = {}, {}, "", ""

    def cancelled(self):
        return False


def test_diarize_and_speaker_labelled_transcribe_nodes(monkeypatch):
    from local_llm import gateway_client
    from local_llm.lab import engine, nodes, store

    calls = {}

    def fake_diarize(path, num_speakers=None, **kw):
        calls["diarize"] = (num_speakers, kw.get("model"))
        return {"segments": TURNS, "speakers": 2, "duration": 12.0, "seconds": 0.5}

    def fake_transcribe(path, model, language=None, **kw):
        calls["transcribe"] = model
        segments = [{"start": 0.1, "end": 4.8, "text": "各位早安"},
                    {"start": 5.6, "end": 8.9, "text": "業務這邊"}]  # fmt: skip
        return {"text": "各位早安業務這邊", "segments": segments, "duration": 12.0}

    def fake_translate(path, model, language=None, **kw):
        calls["translate"] = model
        return {"text": "Good morning", "segments": [], "duration": 12.0}

    monkeypatch.setattr(gateway_client, "diarize", fake_diarize)
    monkeypatch.setattr(gateway_client, "transcribe", fake_transcribe)
    monkeypatch.setattr(gateway_client, "translate", fake_translate)
    data = {"audio_path": "meeting.wav"}
    step = FakeStep()
    out = nodes.run_diarize({"input": "audio_path", "num_speakers": "2"}, data, step)
    assert out["speaker_turns"] == TURNS and calls["diarize"] == (2, "speaker-diarization")
    assert step.detail["speakers"] == 2 and "S1 0–5s" in step.stream
    data |= out
    p = {"input": "audio_path", "speakers": "speaker_turns", "output": "transcript"}
    text = nodes.run_transcribe(p, data, FakeStep())["transcript"]
    assert text == "講者 1：各位早安\n講者 2：業務這邊"
    p |= {"speaker_label": "Speaker {n}"}
    assert nodes.run_transcribe(p, data, FakeStep())["transcript"].startswith("Speaker 1: ")
    p = {"input": "audio_path", "model": "whisper-large-v3", "task": "translate"}
    assert nodes.run_transcribe(p, data, FakeStep())["transcript"] == "Good morning"
    assert calls["translate"] == "whisper-large-v3"
    meeting = store.scenario("meeting_minutes")
    assert not store.validate_scenario(meeting).errors
    assert "speaker-diarization" in engine.requirements(meeting)["gateway"]
