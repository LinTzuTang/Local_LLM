"""Cloud language models for the Language models page (compare and chat), behind the cloud switch.

The platform otherwise runs only local models; these are here as a reference to compare against.
A model id starts with "cloud/", so it can never clash with an Ollama tag; streaming.stream_chat
sends those here instead of to Ollama. The models are every one the keys could call when
scripts/cloud_llm_probe.py last ran (data/lab/cloud_llms.json: Claude and the other Bedrock
vendors, Azure's deployments, Vertex's Gemini). What each provider needs in .env (a server without
the key does not list its models, see `available`):

    Claude (Bedrock)   AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (AWS_REGION). The Bedrock API key
                       (AWS_BEARER_TOKEN_BEDROCK) is not used: 11-26 it had expired while the IAM
                       keys still worked, and the SDK prefers it when it is set.
    GPT (Azure)        AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY, OpenAI-compatible /openai/v1
    Gemini (Vertex)    GOOGLE_APPLICATION_CREDENTIALS (service-account key), Vertex's
                       OpenAI-compatible endpoint

Every call adds a line to storage/cloud_usage/YYYY-MM-DD.jsonl: time, account, model, tokens,
seconds (never the text), so the cost can be told per account.
"""

import base64
import datetime as dt
import json
import os
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import httpx

from . import request_user

ROOT = Path(__file__).resolve().parents[2]
USAGE_DIR = Path(os.getenv("LOCAL_LLM_CLOUD_USAGE_DIR", ROOT / "storage/cloud_usage"))
PREFIX = "cloud/"
LIST = ROOT / "data" / "lab" / "cloud_llms.json"
# listed first: the cloud references the evaluations use; the rest follow by provider and vendor
FEATURED = [
    "cloud/bedrock/global.anthropic.claude-sonnet-5-5",
    "cloud/bedrock/global.anthropic.claude-haiku-5-5",
    "cloud/bedrock/global.anthropic.claude-opus-5-5",
    "cloud/azure/gpt-5.5",
    "cloud/vertex/gemini-3.8-flash",
]


def _load() -> dict[str, dict]:
    try:
        rows = json.loads(LIST.read_text(encoding="utf-8"))["models"]
    except (OSError, ValueError, KeyError):
        return {}
    by_id = {r["id"]: r for r in rows}
    return {m: by_id[m] for m in FEATURED if m in by_id} | by_id


MODELS = _load()
_bedrock_lock = threading.Lock()
# GPT-5.x and Gemini spend part of the output budget on reasoning: a small cap left answers empty
MIN_REASONING_BUDGET = 4096


def _env() -> dict[str, str]:
    from dotenv import dotenv_values

    return {**{k: v for k, v in dotenv_values(ROOT / ".env").items() if v}, **os.environ}


def is_cloud(model: str) -> bool:
    return model.startswith(PREFIX)


def available(model: str) -> bool:
    provider = (MODELS.get(model) or {}).get("provider")
    env = _env()
    if provider == "AWS":
        return bool(env.get("AWS_ACCESS_KEY_ID") and env.get("AWS_SECRET_ACCESS_KEY"))
    if provider == "Azure":
        return bool(env.get("AZURE_OPENAI_ENDPOINT") and env.get("AZURE_OPENAI_API_KEY"))
    if provider == "Google":
        from . import gcp_auth

        return gcp_auth.available()
    return False


def models() -> list[str]:
    """The cloud models this server has keys for."""
    return [m for m in MODELS if available(m)]


def label(model: str) -> str:
    info = MODELS.get(model)
    if not info:
        return model
    return f"☁ {info['label']}（{info['provider']}）"


def stream(
    model: str,
    messages: list[dict],
    *,
    think: bool = False,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    images: list[str] | None = None,
    timeout: float = 600,
) -> Iterator:
    """Yield streaming.Event's like stream_chat: thinking/content chunks, then "done" with Stats."""
    from .streaming import Event, Stats

    info = MODELS.get(model)
    if not info:
        raise ValueError(f"unknown cloud model {model}")
    if not available(model):
        raise RuntimeError(f"this server has no {info['provider']} key for {info['label']}")
    stats, started = Stats(), time.perf_counter()

    def mark(kind: str) -> None:
        now = time.perf_counter() - started
        stats.ttft_s = stats.ttft_s if stats.ttft_s is not None else now
        if kind == "content" and stats.first_answer_s is None:
            stats.first_answer_s = now

    run = {"anthropic": _claude, "converse": _converse}.get(info["api"], _openai_compatible)
    failed = True
    try:
        for kind, text in run(
            info, messages, think, temperature, max_tokens, images or [], timeout, stats
        ):
            mark(kind)
            yield Event(kind, text)
        failed = False
    finally:
        # no tokens/s: over the network it is the connection's pace, not the model's
        stats.total_s = time.perf_counter() - started
        _log_usage(model, stats, failed)
    yield Event("done", stats=stats)


def _split_system(messages: list[dict]) -> tuple[str, list[dict]]:
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system" and m["content"])
    return system, [dict(m) for m in messages if m["role"] != "system"]


def _claude(info, messages, think, temperature, max_tokens, images, timeout, stats):
    from anthropic import AnthropicBedrock

    env = _env()
    with _bedrock_lock:  # the SDK reads the (expired) API key from the environment and refuses both
        saved = os.environ.pop("AWS_BEARER_TOKEN_BEDROCK", None)
        try:
            client = AnthropicBedrock(
                aws_access_key=env["AWS_ACCESS_KEY_ID"],
                aws_secret_key=env["AWS_SECRET_ACCESS_KEY"],
                aws_region=env.get("AWS_REGION") or "us-east-1",
                timeout=timeout,
                max_retries=2,
            )
        finally:
            if saved is not None:
                os.environ["AWS_BEARER_TOKEN_BEDROCK"] = saved
    system, msgs = _split_system(messages)
    if images:
        src = [{"type": "base64", "media_type": "image/png", "data": b} for b in images]
        parts = [{"type": "image", "source": x} for x in src]
        parts.append({"type": "text", "text": msgs[-1]["content"]})
        msgs[-1]["content"] = parts
    kw = {"system": system} if system else {}
    if think:  # no temperature for the 5.5 models: their defaults, as in the evaluations
        kw["thinking"] = {"type": "adaptive"}
    from anthropic import BadRequestError

    try:
        with client.messages.stream(
            model=info["call"], max_tokens=max(max_tokens, 1024), messages=msgs, **kw
        ) as s:
            for event in s:
                if event.type == "content_block_delta":
                    if event.delta.type == "thinking_delta":
                        yield "thinking", event.delta.thinking
                    elif event.delta.type == "text_delta":
                        yield "content", event.delta.text
            final = s.get_final_message()
        stats.prompt_tokens, stats.output_tokens = (
            final.usage.input_tokens,
            final.usage.output_tokens,
        )
        stats.source = "client"
    except BadRequestError as exc:  # an older Claude (Opus 4.1) refuses the SDK's API version
        if "API version" not in str(exc):
            raise
        yield from _converse(info, messages, think, temperature, max_tokens, images, timeout, stats)


def _converse(info, messages, think, temperature, max_tokens, images, timeout, stats):
    """Bedrock's Converse API: every other vendor on Bedrock (Amazon, Meta, Mistral, Qwen, ...)."""
    import boto3
    from botocore.config import Config

    env = _env()
    client = boto3.client(
        "bedrock-runtime",
        region_name=env.get("AWS_REGION") or "us-east-1",
        aws_access_key_id=env["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=env["AWS_SECRET_ACCESS_KEY"],
        config=Config(read_timeout=timeout, retries={"max_attempts": 2}),
    )
    system, msgs = _split_system(messages)
    convo = []
    for m in msgs:
        content = [{"text": m["content"]}] if isinstance(m["content"], str) else m["content"]
        convo.append({"role": m["role"], "content": content})
    if images:
        convo[-1]["content"] = [
            {"image": {"format": "png", "source": {"bytes": base64.b64decode(b)}}} for b in images
        ] + convo[-1]["content"]
    kw = {"system": [{"text": system}]} if system else {}
    config = {"maxTokens": max(max_tokens, 1024), "temperature": temperature}
    for _ in range(3):  # what some older models refuse: a temperature, a system prompt
        try:
            r = client.converse_stream(
                modelId=info["call"], messages=convo, inferenceConfig=config, **kw
            )
            break
        except client.exceptions.ValidationException as exc:
            why = str(exc)
            if "temperature" in why and "temperature" in config:
                config.pop("temperature")
            elif kw and ("system" in why or "alternate" in why):
                text = kw.pop("system")[0]["text"]  # into the first user turn instead
                first = convo[0]["content"]
                convo[0]["content"] = [{"text": f"{text}\n\n{first[-1]['text']}"}, *first[:-1]]
            else:
                raise
    else:
        raise RuntimeError(f"{info['label']} refused the request")
    for event in r["stream"]:
        if delta := event.get("contentBlockDelta", {}).get("delta"):
            if text := delta.get("text"):
                yield "content", text
            elif reasoning := (delta.get("reasoningContent") or {}).get("text"):
                yield "thinking", reasoning
        elif usage := event.get("metadata", {}).get("usage"):
            stats.prompt_tokens, stats.output_tokens = (
                usage.get("inputTokens"),
                usage.get("outputTokens"),
            )


def _openai_compatible(info, messages, think, temperature, max_tokens, images, timeout, stats):
    env = _env()
    msgs = [dict(m) for m in messages]
    if images:
        msgs[-1]["content"] = [{"type": "text", "text": msgs[-1]["content"]}] + [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b}"}}
            for b in images
        ]
    budget = max(max_tokens, MIN_REASONING_BUDGET)
    body = {
        "model": info["call"],
        "messages": msgs,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if info["provider"] == "Azure":
        endpoint = env["AZURE_OPENAI_ENDPOINT"].rstrip("/").split("/openai")[0]
        url = f"{endpoint}/openai/v1/chat/completions"
        headers = {"api-key": env["AZURE_OPENAI_API_KEY"]}
        body["max_completion_tokens"] = budget  # GPT-5.x: only the default temperature
        body["reasoning_effort"] = "medium" if think else "low"
    else:
        from . import gcp_auth

        token, project = gcp_auth.token()
        url = (
            f"https://aiplatform.googleapis.com/v1/projects/{project}/locations/global/endpoints/"
            "openapi/chat/completions"
        )
        headers = {"Authorization": f"Bearer {token}"}
        body |= {
            "max_tokens": budget,
            "temperature": temperature,
            "reasoning_effort": "medium" if think else "low",
        }
    with httpx.Client(timeout=timeout) as client:
        for attempt in range(
            3
        ):  # older models refuse reasoning_effort, some a temperature: drop it
            with client.stream("POST", url, json=body, headers=headers) as r:
                if r.is_error:
                    r.read()
                    bad = next(
                        (
                            k
                            for k in ("reasoning_effort", "temperature")
                            if k in body and k.split("_")[0] in r.text
                        ),
                        None,
                    )
                    if r.status_code == 400 and bad and attempt < 2:
                        body.pop(bad)
                        continue
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
                yield from _sse(r, stats)
                return


def _sse(r: httpx.Response, stats):
    for line in r.iter_lines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if reasoning := delta.get("reasoning_content") or delta.get("reasoning"):
                yield "thinking", reasoning
            if text := delta.get("content"):
                yield "content", text
        if usage := chunk.get("usage"):
            stats.prompt_tokens = usage.get("prompt_tokens")
            stats.output_tokens = usage.get("completion_tokens")


def _log_usage(model: str, stats, failed: bool = False) -> None:
    from urllib.parse import unquote

    user = request_user.headers().get("X-Lab-User")
    row = {
        "t": dt.datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "user": unquote(user) if user else None,
        "input_tokens": stats.prompt_tokens,
        "output_tokens": stats.output_tokens,
        "seconds": round(stats.total_s, 1),
    }
    if failed:
        row["failed"] = True
    try:
        USAGE_DIR.mkdir(parents=True, exist_ok=True)
        with open(USAGE_DIR / f"{row['t'][:10]}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError:
        pass
