"""Neural voices (neural_tts.py, mlx_tts.py): the text they are given, their loudness, and how
tts.py lists, labels and routes them (no model is run)."""

from pathlib import Path

import numpy as np

from local_llm import mlx_tts, neural_tts, tts
from local_llm.gateway.stream_backends import clean


def test_numbers_degrees_and_ranges_are_spelled_out():
    assert neural_tts.speech_text("內湖現在 26°C，今天 21–27°C，降雨機率 33%。") == (
        "內湖現在二十六度，今天二十一到二十七度，降雨機率百分之三十三。"
    )
    assert neural_tts.speech_text("在 350 公尺外") == "在三百五十公尺外"
    assert neural_tts.english("Cafe ACME is close") and not neural_tts.english("營養早午餐 ACME")


def test_neural_voices_are_listed_and_routed(monkeypatch, tmp_path):
    monkeypatch.setattr(neural_tts, "available", lambda name: True)
    monkeypatch.setattr(tts, "available", lambda: False)  # no macOS voices needed
    fast = [v["name"] for v in tts.voices(("zh_TW",))]
    every = [v["name"] for v in tts.voices(("zh_TW",), neural="all")]
    azure = list(neural_tts.AZURE_TTS)  # cloud, but they stream: live
    google = [n for n, (_, loc) in neural_tts.GOOGLE_TTS.items() if loc == "zh_TW"]
    taiwan = [neural_tts.VITS, *azure, *google, "Qwen3-TTS 台灣腔（複製）",
              "VoxCPM2 台灣腔（複製）",
              "MOSS-TTS-Nano 台灣腔（複製）", "OmniVoice 台灣腔（複製）",
              "Breeze-TTS-2 台灣腔（複製）"]  # fmt: skip
    # live: the fast VITS, the streaming Azure voices (first piece in about 0.3 s) and the
    # streaming Qwen3-TTS clone (first chunk in about 0.2 s)
    assert fast == [neural_tts.VITS, *azure, *google, "Qwen3-TTS 台灣腔（複製）"]
    assert every == taiwan
    assert tts.find("Azure 曉臻（雲端）")["engine"] == "azure"
    mainland = [v["name"] for v in tts.voices(("zh_CN",))]
    assert mainland == ["Google Chirp 3 HD Aoede（雲端）", "Kokoro 曉北", "Kokoro 雲希",
                        "Qwen3-TTS Vivian", "Qwen3-TTS Uncle Fu"]  # fmt: skip
    assert tts.find("Google 台灣 WaveNet A（雲端）")["engine"] == "google"
    assert tts.source(tts.find("Kokoro 曉北"))[0] == "Kokoro-82M・MLX"
    assert tts.source({"name": "Meijia"}) == tts.MACOS_SOURCE
    gemini = tts.find("Gemini 2.5 Flash TTS（雲端）")
    assert gemini and gemini["engine"] == "gemini" and not gemini["fast"]  # cloud, too slow live
    assert tts.voices(("en_US",)) == [] and tts.voices(neural="none") == []
    rendered = []

    def render(text, voice, out, sample_rate):
        rendered.append((text, voice, sample_rate))
        Path(out).write_bytes(tts.wav_bytes(np.zeros(1600, dtype="<i2"), sample_rate))
        return Path(out)

    monkeypatch.setattr(neural_tts, "render", render)
    monkeypatch.setattr(tts, "CACHE_DIR", tmp_path)
    clone = "Qwen3-TTS 台灣腔（複製）"
    path = tts.synthesize("你好", clone, sample_rate=16000)
    again = tts.synthesize("你好", clone, sample_rate=16000)  # from the cache
    assert path == again and rendered == [("你好", clone, 16000)]
    data, seconds = tts.speak_wav("你好", neural_tts.VITS)
    assert data[:4] == b"RIFF" and seconds > 0


def test_loudness_reaches_the_target_without_clipping():
    rng = np.random.default_rng(0)
    quiet = rng.normal(0, 0.01, 24000).astype(np.float32)
    quiet[1000] = 0.2  # one loud click must not cap the gain
    out = neural_tts.loudness(quiet)
    frames = out[: len(out) // 320 * 320].reshape(-1, 320)
    level = 20 * np.log10(np.sqrt((frames**2).mean()))
    assert abs(level - neural_tts.TARGET_RMS_DB) < 1.5
    assert np.abs(out).max() <= neural_tts.PEAK + 1e-6


def test_mlx_voices_read_simplified_and_clone_the_taiwanese_prompt():
    assert neural_tts.PROMPT.exists()  # data/lab/voices/taiwan_speaker.wav ships with the repo
    qwen = mlx_tts.VOICES["Qwen3-TTS 台灣腔（複製）"]
    options = mlx_tts._options(qwen)
    assert options["ref_audio"] == str(neural_tts.PROMPT)
    assert options["ref_text"] == neural_tts.PROMPT_TEXT and "clone" not in options
    assert qwen["simplified"] and not mlx_tts.VOICES["Kokoro 曉北"]["simplified"]
    assert mlx_tts._to_simplified("客戶希望交貨") == "客户希望交货"


def test_sentences_split_like_the_assistant_speaks():
    assert tts.sentences("第一句。第二句！\n\n第三句？還有") == [
        "第一句。",
        "第二句！",
        "第三句？",
        "還有",
    ]


def test_streaming_text_drops_the_silence_token():
    assert clean("好 SIL謝謝") == "好謝謝" and clean("等一下 SIL先幫我") == "等一下先幫我"
    assert clean("SILVER SIL") == "SILVER" and clean("我在 CNC 產線") == "我在 CNC 產線"
