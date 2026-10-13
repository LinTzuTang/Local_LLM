import pytest

from local_llm.config import Settings, settings


def test_vllm_requires_versioned_endpoint():
    with pytest.raises(ValueError):
        Settings("vllm", "test", "http://127.0.0.1:8000")


def test_override_provider_does_not_inherit_other_backend(monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:11434")
    monkeypatch.setenv("LOCAL_LLM_MODEL", "qwen3:1.7b")
    config = settings(provider="vllm")
    assert config.base_url == "http://127.0.0.1:8000/v1"
    assert config.model == "Qwen/Qwen3-8B"
