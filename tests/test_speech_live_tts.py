"""Live captions, voice chat and text to speech: pure helpers, plus macOS `say` when present."""

import shutil
import wave

import numpy as np
import pytest

from local_llm import asr_metrics, audio_aug, tts
from local_llm import speech_live as sl
from local_llm import speech_translate as stx

LISTING = """\
Bubbles             en_US    # Hello! My name is Bubbles.
Eddy (中文（台灣）)       zh_TW    # 你好，我叫Eddy。
Eddy (英文（美國）)       en_US    # Hello! My name is Eddy.
Meijia (中文（台灣）)     zh_TW    # 你好，我叫美佳。
Meijia (中文（台灣）)     zh_TW    # 你好，我叫美佳。
Samantha            en_US    # Hello! My name is Samantha.
Thomas              fr_FR    # Bonjour, je m’appelle Thomas.
Tingting            zh_CN    # 你好！我叫婷婷。
"""


def test_join_inserts_silence_between_clips_only():
    a, b = np.ones(100, dtype=np.float32), np.full(50, 0.5, dtype=np.float32)
    out = audio_aug.join([a, b], gap_s=0.01, sr=1000)
    assert len(out) == 100 + 10 + 50
    assert not out[100:110].any() and out[-1] == 0.5
    assert len(audio_aug.join([], 0.8)) == 0


def test_sample_reference_reads_both_sample_folders(tmp_path):
    bundled, local = tmp_path / "samples", tmp_path / "local"
    for d in (bundled, local):
        d.mkdir()
    (local / "x_nan.txt").write_text("# header\n人客遐濟\n", encoding="utf-8")
    dirs = (bundled, local)
    assert asr_metrics.sample_reference(local / "x_nan.flac", dirs) == "人客遐濟"
    assert asr_metrics.sample_reference(tmp_path / "x_nan.flac", dirs) is None
    # A second reference of the same audio: the Mandarin translation (x_nan.zh.txt).
    assert asr_metrics.sample_reference(local / "x_nan.flac", dirs, variant="zh") is None
    (local / "x_nan.zh.txt").write_text("# header\n客人這麼多\n", encoding="utf-8")
    assert asr_metrics.sample_reference(local / "x_nan.flac", dirs, variant="zh") == "客人這麼多"


def test_parse_voices_dedupes_and_keeps_short_names():
    rows = tts.parse_voices(LISTING)
    names = [r["name"] for r in rows]
    assert names.count("Meijia (中文（台灣）)") == 1
    meijia = next(r for r in rows if r["short"] == "Meijia")
    assert meijia["locale"] == "zh_TW" and meijia["sample"] == "你好，我叫美佳。"
    assert tts.find("Meijia", rows) is meijia
    assert tts.find("Eddy (英文（美國）)", rows)["locale"] == "en_US"


def test_voices_filters_locales_and_novelty(monkeypatch):
    monkeypatch.setattr(tts, "available", lambda: True)
    monkeypatch.setattr(tts, "_listing", lambda: LISTING)
    rows = tts.voices()
    assert [r["short"] for r in rows] == ["Eddy", "Meijia", "Tingting", "Eddy", "Samantha"]
    assert "Bubbles" not in {r["short"] for r in rows}  # novelty
    assert "Thomas" not in {r["short"] for r in rows}  # French: not a default locale
    assert "Bubbles" in {r["short"] for r in tts.voices(None, novelty=True)}
    assert tts.language(rows[-1]) == "en" and tts.language(rows[0]) == "zh"


@pytest.mark.skipif(shutil.which("say") is None, reason="macOS `say` not available")
def test_say_renders_a_wav(tmp_path):
    out = tts.synthesize("Hello there.", "Samantha", rate=200, out=tmp_path / "a.wav")
    with wave.open(str(out)) as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2
        assert w.getframerate() == tts.SAMPLE_RATE and w.getnframes() > 0
    timed = tts.timed_synthesize("Hello again.", "Samantha")
    assert timed["duration"] > 0.2 and timed["rtf"] > 0 and timed["path"].exists()
    timed["path"].unlink()
    with pytest.raises(ValueError):
        tts.synthesize("  ", "Samantha")


def test_join_texts_spaces_only_between_words():
    assert sl.join_texts(["各位早安，", "今天開會", ""]) == "各位早安，今天開會"
    assert sl.join_texts(["Good morning", "everyone."]) == "Good morning everyone."


def test_latency_stats_and_caption_replay():
    ends, compute = [2.0, 4.0, 6.0], [0.5, 2.5, 0.5]
    chunks = [
        {"start": e - 2, "end": e, "compute": c, "text": t}
        for e, c, t in zip(ends, compute, ["今天", "天氣", "很好"], strict=True)
    ]
    for c, s in zip(chunks, stx.live_schedule(ends, compute), strict=True):
        c.update(finish=s["finish"], lag=s["lag"])
    stats = sl.latency_stats(chunks)
    assert stats["first_s"] == 2.5 and stats["max_lag"] == 2.5 and stats["lag_end"] == 1.0
    assert stats["mean_lag"] == pytest.approx((0.5 + 2.5 + 1.0) / 3, abs=0.01)
    assert stats["keeps_up"] and stats["rtf"] == pytest.approx(3.5 / 6, abs=1e-3)
    assert sl.caption_at(chunks, 2.4) == ""
    assert sl.caption_at(chunks, 6.6) == "今天天氣"  # the second chunk finished at 6.5
    assert sl.caption_at(chunks, 7.0) == "今天天氣很好"
    assert sl.latency_stats([]) == {}


def test_voice_chat_messages_keep_the_last_turns():
    history = [{"question": f"q{i}", "answer": f"a{i}"} for i in range(6)]
    history.append({"question": "failed", "answer": None})
    msgs = sl.messages(history, "new", "sys", max_turns=2)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[1]["content"] == "q5" and msgs[-1]["content"] == "new"
    assert len(sl.messages(history, "new", "sys", max_turns=0)) == 2
    assert "Traditional Chinese" in sl.system_prompt("zh")
    assert "reply in English" in sl.system_prompt("en") and "speaking" not in sl.system_prompt(None)


def test_speakable_drops_markup_thinking_and_emoji():
    text = "<think>plan</think>## 建議\n- **大安森林公園** 😀\n- 看 [官網](https://x.tw)\n"
    assert sl.speakable(text) == "建議 大安森林公園 看 官網"
    assert sl.speakable("1. First\n2. Second") == "First Second"
