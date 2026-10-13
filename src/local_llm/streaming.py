"""Streaming chat with timing metrics, over Ollama's native API or any OpenAI-compatible server.

The same measurements (TTFT, decode tokens/s, total time) are taken for both so Ollama and
vLLM can be compared side by side. Server-reported counters are preferred when available.
"""

import json
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass

import httpx

from . import ollama_admin


@dataclass(frozen=True)
class Backend:
    label: str
    # "ollama": native /api/chat. "ollama_v1" and "vllm": OpenAI-compatible
    # /v1/chat/completions; they differ only in how thinking is switched off.
    kind: str
    base_url: str
    api_key: str = "local"

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {sorted(KINDS)}")
        if self.kind != "ollama" and not self.base_url.rstrip("/").endswith("/v1"):
            raise ValueError("OpenAI-compatible base_url must end in /v1")


KINDS = {"ollama", "ollama_v1", "vllm"}


@dataclass
class Stats:
    ttft_s: float | None = None  # first token of any kind (thinking or answer)
    first_answer_s: float | None = None
    total_s: float = 0.0
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    decode_tps: float | None = None
    prompt_tps: float | None = None
    load_s: float | None = None
    source: str = "client"  # where decode_tps came from: "server" or "client"

    def to_dict(self) -> dict:
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in asdict(self).items()}


@dataclass
class Event:
    kind: str  # "thinking", "content" or "done"
    text: str = ""
    stats: Stats | None = None


def list_models(backend: Backend) -> list[str]:
    with httpx.Client(timeout=10, trust_env=False) as client:
        if backend.kind == "ollama":
            response = client.get(backend.base_url + "/api/tags")
            response.raise_for_status()
            return [m["name"] for m in response.json().get("models", [])]
        response = client.get(backend.base_url + "/models", headers=_auth(backend))
        response.raise_for_status()
        return [m["id"] for m in response.json().get("data", [])]


def stream_chat(
    backend: Backend,
    model: str,
    messages: list[dict],
    *,
    think: bool = False,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    num_ctx: int = 8192,
    images: list[str] | None = None,
    timeout: float = 600,
) -> Iterator[Event]:
    """Yield thinking/content chunks, then one "done" event with Stats.

    `images` are base64 strings attached to the last user message.
    """
    stream = _ollama if backend.kind == "ollama" else _openai
    yield from stream(
        backend, model, messages, think, temperature, max_tokens, num_ctx, images or [], timeout
    )


def _ollama(backend, model, messages, think, temperature, max_tokens, num_ctx, images, timeout):
    msgs = [dict(m) for m in messages]
    if images:
        msgs[-1]["images"] = images
    body = {
        "model": model,
        "messages": msgs,
        "stream": True,
        "options": {"temperature": temperature, "num_predict": max_tokens, "num_ctx": num_ctx},
    }
    think_value = ollama_admin.think_value(model, think, backend.base_url)
    if think_value is not None:
        body["think"] = think_value
    stats, started = Stats(), time.perf_counter()
    with httpx.Client(timeout=timeout, trust_env=False) as client:
        with client.stream("POST", backend.base_url + "/api/chat", json=body) as response:
            _raise_with_body(response)
            for line in response.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if "error" in chunk:
                    raise RuntimeError(chunk["error"])
                message = chunk.get("message", {})
                for kind in ("thinking", "content"):
                    if text := message.get(kind):
                        _mark_first(stats, kind, started)
                        yield Event(kind, text)
                if chunk.get("done"):
                    stats.prompt_tokens = chunk.get("prompt_eval_count")
                    stats.output_tokens = chunk.get("eval_count")
                    stats.decode_tps = _rate(chunk.get("eval_count"), chunk.get("eval_duration"))
                    stats.prompt_tps = _rate(
                        chunk.get("prompt_eval_count"), chunk.get("prompt_eval_duration")
                    )
                    stats.load_s = (chunk.get("load_duration") or 0) / 1e9
                    stats.source = "server"
    stats.total_s = time.perf_counter() - started
    yield Event("done", stats=stats)


def _openai(backend, model, messages, think, temperature, max_tokens, num_ctx, images, timeout):
    msgs = [dict(m) for m in messages]
    if images:
        parts = [{"type": "text", "text": msgs[-1]["content"]}]
        parts += [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
            for b64 in images
        ]
        msgs[-1]["content"] = parts
    body = {
        "model": model,
        "messages": msgs,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if backend.kind == "vllm":
        # Qwen3-style chat templates read this switch.
        body["chat_template_kwargs"] = {"enable_thinking": think}
    elif not think and ollama_admin.thinking_mode(model, _ollama_root(backend)) == "optional":
        body["reasoning_effort"] = "none"
    stats, started = Stats(), time.perf_counter()
    with httpx.Client(timeout=timeout, trust_env=False) as client:
        with client.stream(
            "POST", backend.base_url + "/chat/completions", json=body, headers=_auth(backend)
        ) as response:
            _raise_with_body(response)
            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta", {})
                    # Ollama uses "reasoning"; vLLM uses "reasoning_content" (or "reasoning").
                    reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                    if reasoning:
                        _mark_first(stats, "thinking", started)
                        yield Event("thinking", reasoning)
                    if text := delta.get("content"):
                        _mark_first(stats, "content", started)
                        yield Event("content", text)
                if usage := chunk.get("usage"):
                    stats.prompt_tokens = usage.get("prompt_tokens")
                    stats.output_tokens = usage.get("completion_tokens")
                if timings := chunk.get("timings"):
                    stats.decode_tps = timings.get("predicted_per_second")
                    stats.prompt_tps = timings.get("prompt_per_second")
                    stats.source = "server"
    stats.total_s = time.perf_counter() - started
    if stats.decode_tps is None and stats.output_tokens and stats.ttft_s is not None:
        decode_time = stats.total_s - stats.ttft_s
        stats.decode_tps = stats.output_tokens / decode_time if decode_time > 0 else None
    yield Event("done", stats=stats)


def collect(events: Iterator[Event]) -> tuple[str, str, Stats]:
    """Consume a stream; return (thinking, answer, stats)."""
    thinking, answer, stats = [], [], Stats()
    for event in events:
        if event.kind == "thinking":
            thinking.append(event.text)
        elif event.kind == "content":
            answer.append(event.text)
        else:
            stats = event.stats
    return "".join(thinking), "".join(answer), stats


def _mark_first(stats: Stats, kind: str, started: float) -> None:
    now = time.perf_counter() - started
    if stats.ttft_s is None:
        stats.ttft_s = now
    if kind == "content" and stats.first_answer_s is None:
        stats.first_answer_s = now


def _rate(count, duration_ns):
    return count / (duration_ns / 1e9) if count and duration_ns else None


def _ollama_root(backend: Backend) -> str:
    return backend.base_url.rstrip("/").removesuffix("/v1")


def _auth(backend: Backend) -> dict:
    return {"Authorization": f"Bearer {backend.api_key}"}


def _raise_with_body(response: httpx.Response) -> None:
    if response.is_error:
        response.read()
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")
