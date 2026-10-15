"""The project's own Ollama (127.0.0.1:11434) as a gateway runtime: audio-capable LLMs.

Gemma 4 accepts audio in Ollama (0.40+): the native /api/chat takes the WAV bytes in a
message's `images` list (Ollama sniffs the media type). The gateway serves such a model as
an ASR model (`"ollama": "<tag>"` in its catalog gateway block): it is "installed" when
Ollama has the tag, a pull streams Ollama's own /api/pull, and Ollama itself loads and
unloads the weights (its keep_alive), so the gateway never holds them in memory.
"""

import base64
import io
import json
import os
import time
import wave
from collections.abc import AsyncIterator

import httpx
import numpy as np

OLLAMA_URL = os.getenv("LOCAL_LLM_OLLAMA_URL", "http://127.0.0.1:11434")
TAGS_TTL_S = 3.0
_tags: dict = {"at": 0.0, "models": {}}


def _client(timeout: float | None = 10) -> httpx.Client:
    return httpx.Client(base_url=OLLAMA_URL, timeout=timeout, trust_env=False)


def installed() -> dict[str, int]:
    """{tag: size in bytes} of the models Ollama has, cached for a few seconds."""
    if time.monotonic() - _tags["at"] > TAGS_TTL_S:
        try:
            with _client(3) as client:
                rows = client.get("/api/tags").json()["models"]
            _tags["models"] = {r["name"]: r.get("size", 0) for r in rows}
        except (httpx.HTTPError, ValueError, KeyError):
            _tags["models"] = {}  # Ollama not running: nothing counts as installed
        _tags["at"] = time.monotonic()
    return _tags["models"]


def forget() -> None:
    """Drop the cached tag list (after a pull or delete)."""
    _tags["at"] = 0.0


def capabilities(tag: str) -> list[str]:
    with _client() as client:
        response = client.post("/api/show", json={"model": tag})
        response.raise_for_status()
        return response.json().get("capabilities") or []


def delete(tag: str) -> None:
    with _client(60) as client:
        response = client.request("DELETE", "/api/delete", json={"model": tag})
        if response.status_code != 404:
            response.raise_for_status()
    forget()


async def pull(tag: str) -> AsyncIterator[dict]:
    """Ollama's own progress events ({"status", "total", "completed"}), then a check that the
    model really takes audio."""
    async with httpx.AsyncClient(base_url=OLLAMA_URL, timeout=None, trust_env=False) as client:
        async with client.stream("POST", "/api/pull", json={"model": tag}) as response:
            if response.is_error:
                await response.aread()
                raise RuntimeError(f"Ollama pull failed: {response.text}")
            async for line in response.aiter_lines():
                if not line:
                    continue
                event = json.loads(line)
                if "error" in event:
                    raise RuntimeError(event["error"])
                if event.get("status") != "success":
                    yield event
    forget()
    if "audio" not in capabilities(tag):
        raise RuntimeError(f"{tag} was downloaded, but Ollama does not accept audio for it")
    yield {"status": "success"}


def wav_bytes(samples: np.ndarray, rate: int = 16000) -> bytes:
    """16-bit mono WAV, the format every Ollama audio model reads."""
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return out.getvalue()


def chat_audio(tag: str, instruction: str, audio: bytes, max_tokens: int) -> dict:
    """One non-streaming /api/chat call with the audio attached; returns Ollama's reply.

    max_tokens matters: an LLM that starts repeating itself would otherwise write until its
    context (256K tokens for Gemma 4) is full."""
    body = {
        "model": tag,
        "messages": [
            {"role": "user", "content": instruction, "images": [base64.b64encode(audio).decode()]}
        ],
        "stream": False,
        "think": False,  # a transcript, not a reasoning task (and much faster)
        "options": {"temperature": 0, "num_predict": max_tokens},
    }
    with _client(900) as client:
        response = client.post("/api/chat", json=body)
        if response.is_error:
            try:
                message = response.json().get("error") or response.text
            except ValueError:
                message = response.text
            raise RuntimeError(f"Ollama: {message} (HTTP {response.status_code})")
        return response.json()


def unload(tag: str) -> None:
    with _client(60) as client:
        client.post("/api/generate", json={"model": tag, "keep_alive": 0})
