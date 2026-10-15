"""Speech models: mlx-audio and Ollama-audio backends, Ollama-backed specs, CER scoring.

No model is loaded: decoding and the Ollama HTTP calls are replaced with fakes.
"""

import asyncio
from pathlib import Path

import numpy as np
import pytest

from local_llm import asr_metrics, catalog
from local_llm.gateway import backends, download, ollama
from local_llm.gateway.registry import Store, load_specs

RATE = backends.SAMPLE_RATE


def test_ollama_specs_have_a_tag_and_no_weights():
    specs = load_specs()
    gemma = specs["gemma4:e4b"]
    assert gemma.backend == "ollama-audio" and gemma.ollama == "gemma4:e4b"
    assert gemma.weights == () and gemma.task == "asr"
    # Only Gemma 4 sizes whose audio input was verified are offered as speech models.
    assert {s.id for s in specs.values() if s.ollama} == {"gemma4:e2b", "gemma4:e4b"}
    for tag in ("gemma4:26b", "gemma4:31b"):
        assert "audio" not in catalog.find(tag)["caps"]
    assert specs["qwen3-asr-1.7b"].options["language_names"] is True
    assert "FunASR" in specs["sensevoice-small"].license  # custom licence, flagged


def test_store_asks_ollama_for_ollama_models(tmp_path, monkeypatch):
    spec = load_specs()["gemma4:e2b"]
    store = Store(tmp_path)
    monkeypatch.setattr(ollama, "installed", lambda: {"gemma4:e2b": 7_000_000_000})
    assert store.installed(spec) and store.size_bytes(spec) == 7_000_000_000
    monkeypatch.setattr(ollama, "installed", lambda: {})
    assert not store.installed(spec) and store.size_bytes(spec) is None
    deleted = []
    monkeypatch.setattr(ollama, "delete", deleted.append)
    store.remove(spec)
    assert deleted == ["gemma4:e2b"]


def test_pull_of_an_ollama_model_streams_ollamas_events(tmp_path, monkeypatch):
    spec = load_specs()["gemma4:e4b"]
    monkeypatch.setattr(ollama, "installed", lambda: {})

    async def fake_pull(tag):
        yield {"status": "pulling manifest"}
        yield {"status": f"pulling {tag}", "total": 10, "completed": 10}
        yield {"status": "success"}

    monkeypatch.setattr(ollama, "pull", fake_pull)

    async def collect():
        return [e async for e in download.pull(spec, Store(tmp_path))]

    events = asyncio.run(collect())
    assert events[1]["total"] == 10 and events[-1] == {"status": "success"}


def test_mlx_audio_kwargs_map_language_and_prompt():
    qwen = {"language_names": True, "prompt_arg": "system_prompt", "generate": {"max_tokens": 9}}
    assert backends.mlx_audio_kwargs(qwen, "zh", "人名：王建國") == {
        "max_tokens": 9,
        "language": "Chinese",
        "system_prompt": "人名：王建國",
    }
    sense = {"generate": {"use_itn": True}}
    assert backends.mlx_audio_kwargs(sense, "zh", "ignored") == {"use_itn": True, "language": "zh"}
    assert backends.mlx_audio_kwargs({}, None, None) == {}


def test_split_points_cut_at_the_quietest_frame():
    samples = np.ones(RATE * 25, dtype=np.float32)
    samples[RATE * 9 : RATE * 9 + RATE // 10] = 0.0  # silence at 9.0-9.1 s
    cuts = backends.split_points(samples, max_seconds=10)
    assert cuts[-1] == len(samples)
    assert abs(cuts[0] - RATE * 9.05) < RATE * 0.05  # inside the silence, not at 10 s
    assert all(b - a <= RATE * 10 for a, b in zip([0, *cuts], cuts, strict=False))
    assert backends.split_points(samples[: RATE * 5], max_seconds=10) == [RATE * 5]


def test_ollama_audio_backend_chunks_and_flags_truncation(monkeypatch):
    samples = np.zeros(RATE * 30, dtype=np.float32)
    monkeypatch.setattr(backends, "decode_audio", lambda path: samples)
    calls = []

    def fake_chat(tag, instruction, audio, max_tokens):
        calls.append((tag, instruction, len(audio), max_tokens))
        reason = "length" if len(calls) == 2 else "stop"
        return {"message": {"content": f" 第{len(calls)}段 "}, "done_reason": reason,
                "load_duration": 1_000_000_000}  # fmt: skip

    monkeypatch.setattr(ollama, "chat_audio", fake_chat)
    handle = {"tag": "gemma4:e2b", "options": {"chunk_seconds": 20}}
    out = backends.OllamaAudioBackend().run(handle, Path("x.wav"), language="zh", prompt="品保課")
    assert out["text"] == "第1段第2段" and out["duration"] == 30.0
    assert len(calls) == 2 and out["truncated"] and out["segments"][1]["truncated"]
    assert "Traditional Chinese" in calls[0][1] and "品保課" in calls[0][1]
    assert calls[0][3] < 64 + 12 * 21  # output capped by the chunk length
    assert out["ollama_load_seconds"] == 2.0


def test_wav_bytes_round_trip():
    import io
    import wave

    data = ollama.wav_bytes(np.array([0.0, 0.5, -1.0, 2.0], dtype=np.float32))
    with wave.open(io.BytesIO(data)) as w:
        assert (w.getnchannels(), w.getframerate(), w.getnframes()) == (1, RATE, 4)


def test_spell_numbers():
    assert asr_metrics.spell_number(15) == "十五"
    assert asr_metrics.spell_number(105) == "一百零五"
    assert asr_metrics.spell_number(250) == "二百五十"
    assert asr_metrics.spell_number(10005) == "一萬零五"
    assert asr_metrics.spell_numbers("87% 2.3% 2026年") == "百分之八十七 百分之二點三 二零二六年"


def test_cer_ignores_script_punctuation_and_number_format():
    ref = "本週稼動率百分之八十七，兩百五十組。"
    assert asr_metrics.cer(ref, "本周稼动率87%，250组")["cer"] == 0.0
    raw = asr_metrics.cer(ref, "本周稼动率87%，250组", canonical=False)
    assert raw["cer"] > 0.3
    one = asr_metrics.cer("各位早安", "各位早上好")
    assert (one["sub"], one["ins"], one["del"], one["errors"]) == (1, 1, 0, 2)
    assert one["cer"] == 0.5 and asr_metrics.cer("", "x") is None


def test_sample_reference_reads_the_bundled_script():
    ref = asr_metrics.sample_reference(asr_metrics.SAMPLES_DIR / "meeting_zh.wav")
    assert ref.startswith("各位早安") and "[" not in ref and "#" not in ref
    assert asr_metrics.sample_reference(Path("/tmp/meeting_zh.wav")) is None
    assert asr_metrics.reference_text("# note\n[Meijia] 你好\n\n[Eddy] 再見") == "你好\n再見"


@pytest.mark.parametrize("model", ["sensevoice-small", "qwen3-asr-0.6b", "gemma4:e2b"])
def test_transcribe_node_lists_new_models(model):
    assert model in {m["id"] for m in catalog.gateway_models("asr")}


def test_management_calls_accept_curl_style_bodies(tmp_path):
    from fastapi.testclient import TestClient

    from local_llm.gateway.manager import Manager
    from local_llm.gateway.server import create_app

    async def fake_pull(spec, store):
        yield {"status": "success"}

    manager = Manager(load_specs(), {}, Store(tmp_path), free=lambda: None)
    app = create_app(manager, puller=fake_pull)
    with TestClient(app) as client:
        form = {"content-type": "application/x-www-form-urlencoded"}  # what `curl -d` sends
        r = client.post("/api/pull", content=b'{"model": "sensevoice-small"}', headers=form)
        assert r.status_code == 200 and r.text.strip().endswith('{"status": "success"}')
        r = client.post("/api/unload", content=b'{"model": "sensevoice-small"}', headers=form)
        assert r.json() == {"status": "not loaded", "model": "sensevoice-small"}
        r = client.post("/api/unload", content=b"not json", headers=form)
        assert r.status_code == 400 and "invalid request" in r.json()["error"]


def test_bundled_samples_have_references_and_the_scenario_validates():
    import json

    from local_llm.lab import store

    speech = asr_metrics.sample_reference(asr_metrics.SAMPLES_DIR / "national_day_2026_zh.wav")
    assert speech.startswith("各位國人同胞") and speech.rstrip().endswith("也願意留下來。")
    assert "不能夠只看產值與出口" in speech  # spoken wording, not the prepared text
    doc = json.loads(store.SCENARIOS_FILE.read_text())
    scenario = next(s for s in doc["scenarios"] if s["id"] == "asr_model_comparison")
    check = store.validate_scenario(scenario)
    assert check.ok and not check.warnings
    audio = next(i for i in scenario["inputs"] if i["type"] == "audio")
    for sample in audio["samples"]:
        path = asr_metrics.PROJECT_DIR / sample["path"]
        assert path.exists() and asr_metrics.sample_reference(path)
    reference = next(i for i in scenario["inputs"] if i["key"] == "reference")
    assert reference["from_sample"] == audio["key"]


def test_taiwan_tongues_is_converted_like_breeze():
    spec = load_specs()["taiwan-tongues-asr"]
    assert spec.backend == "mlx-whisper" and spec.convert == "hf-whisper-to-mlx"
    assert spec.weights[0].revision
    assert spec.weights[0].files == ("config.json", "model.safetensors")
    entry = next(m for m in catalog.gateway_models("asr") if m["id"] == "taiwan-tongues-asr")
    assert "TRAIL" in entry["license"] and entry["license_url"].endswith("/LICENSE")
    # Real recordings: Whisper loops less without conditioning; Fun-ASR-Nano needs chunks.
    specs = load_specs()
    for model in ("whisper-large-v3", "whisper-large-v3-turbo", "taiwan-tongues-asr"):
        assert specs[model].options["decode"] == {"condition_on_previous_text": False}
    assert "decode" not in specs["breeze-asr-25"].options
    assert specs["fun-asr-nano"].options["generate"]["chunk_duration"] == 60
