import json

from local_llm import cloud_llm, streaming
from local_llm.streaming import Backend, collect


def test_cloud_ids_never_look_like_ollama_tags():
    assert all(cloud_llm.is_cloud(m) for m in cloud_llm.MODELS)
    assert not cloud_llm.is_cloud("gemma4:31b")
    assert cloud_llm.label("cloud/claude-haiku-5-5").startswith("☁ Claude Haiku 5.5")


def test_no_keys_no_models(monkeypatch):
    monkeypatch.setattr(cloud_llm, "_env", lambda: {})
    monkeypatch.setattr("local_llm.gcp_auth.available", lambda path=None: False)
    assert cloud_llm.models() == []


def test_keys_pick_the_models(monkeypatch):
    monkeypatch.setattr(
        cloud_llm, "_env", lambda: {"AWS_ACCESS_KEY_ID": "a", "AWS_SECRET_ACCESS_KEY": "b"}
    )
    monkeypatch.setattr("local_llm.gcp_auth.available", lambda path=None: False)
    assert cloud_llm.models() == ["cloud/claude-sonnet-5-5", "cloud/claude-haiku-5-5"]


def test_stream_chat_sends_cloud_models_to_the_cloud_and_logs_use(monkeypatch, tmp_path):
    monkeypatch.setattr(cloud_llm, "USAGE_DIR", tmp_path)
    monkeypatch.setattr(cloud_llm, "available", lambda m: True)

    def fake(info, messages, think, temperature, max_tokens, images, timeout, stats):
        assert info["provider"] == "Azure"
        stats.prompt_tokens, stats.output_tokens = 5, 2
        yield "content", "你好"

    monkeypatch.setattr(cloud_llm, "_openai_compatible", fake)
    monkeypatch.setattr(streaming, "_ollama", lambda *a: (_ for _ in ()).throw(AssertionError))
    backend = Backend("local", "ollama", "http://127.0.0.1:11434")
    _, answer, stats = collect(
        streaming.stream_chat(backend, "cloud/gpt-5.5", [{"role": "user", "content": "hi"}])
    )
    assert answer == "你好" and stats.output_tokens == 2
    rows = [
        json.loads(line) for f in tmp_path.glob("*.jsonl") for line in f.read_text().splitlines()
    ]
    assert (
        rows[0]["model"] == "cloud/gpt-5.5"
        and rows[0]["output_tokens"] == 2
        and "failed" not in rows[0]
    )
