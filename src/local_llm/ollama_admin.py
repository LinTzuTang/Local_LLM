"""Model management through the Ollama HTTP API (list, pull, delete, loaded models)."""

import json
from collections.abc import Iterator
from functools import lru_cache

import httpx

from .catalog import normalize

DEFAULT_URL = "http://127.0.0.1:11434"


def _client(base_url: str, timeout: float | None = 10) -> httpx.Client:
    return httpx.Client(base_url=base_url, timeout=timeout, trust_env=False)


def version(base_url: str = DEFAULT_URL) -> str | None:
    try:
        with _client(base_url, 3) as client:
            return client.get("/api/version").json()["version"]
    except httpx.HTTPError:
        return None


def installed(base_url: str = DEFAULT_URL) -> list[dict]:
    with _client(base_url) as client:
        response = client.get("/api/tags")
        response.raise_for_status()
        return response.json().get("models", [])


def installed_names(base_url: str = DEFAULT_URL) -> set[str]:
    return {normalize(m["name"]) for m in installed(base_url)}


def loaded(base_url: str = DEFAULT_URL) -> list[dict]:
    """Models currently held in memory (/api/ps)."""
    with _client(base_url) as client:
        response = client.get("/api/ps")
        response.raise_for_status()
        return response.json().get("models", [])


def unload(model: str, base_url: str = DEFAULT_URL) -> None:
    with _client(base_url, 60) as client:
        client.post("/api/generate", json={"model": model, "keep_alive": 0}).raise_for_status()


def delete(model: str, base_url: str = DEFAULT_URL) -> None:
    with _client(base_url, 60) as client:
        client.request("DELETE", "/api/delete", json={"model": model}).raise_for_status()
    thinking_mode.cache_clear()
    supports_tools.cache_clear()


def pull(model: str, base_url: str = DEFAULT_URL) -> Iterator[dict]:
    """Stream pull progress: dicts with status and optional completed/total bytes."""
    with _client(base_url, None) as client:
        with client.stream("POST", "/api/pull", json={"model": model, "stream": True}) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    event = json.loads(line)
                    if "error" in event:
                        raise RuntimeError(event["error"])
                    yield event
    thinking_mode.cache_clear()
    supports_tools.cache_clear()


def show(model: str, base_url: str = DEFAULT_URL) -> dict:
    with _client(base_url, 30) as client:
        response = client.post("/api/show", json={"model": model})
        response.raise_for_status()
        return response.json()


@lru_cache(maxsize=64)
def supports_tools(model: str, base_url: str = DEFAULT_URL) -> bool:
    """Whether Ollama will pass tool definitions to this model (from its chat template)."""
    try:
        return "tools" in show(model, base_url).get("capabilities", [])
    except httpx.HTTPError:
        return False


@lru_cache(maxsize=64)
def thinking_mode(model: str, base_url: str = DEFAULT_URL) -> str:
    """'none', 'optional' (think can be switched off) or 'always'.

    Some tags (e.g. qwen3:30b-a3b = Thinking-2507) always open a <think> block; sending
    think=false then leaks the reasoning into the answer, so callers must send think=true.
    """
    try:
        info = show(model, base_url)
    except httpx.HTTPError:
        return "none"
    if "thinking" not in info.get("capabilities", []):
        return "none"
    return "optional" if "not $.Think" in info.get("template", "") else "always"


def think_value(model: str, want_thinking: bool, base_url: str = DEFAULT_URL) -> bool | None:
    """The Ollama `think` value that produces correct output for this model."""
    mode = thinking_mode(model, base_url)
    if mode == "none":
        return None
    if mode == "always":
        return True
    return want_thinking
