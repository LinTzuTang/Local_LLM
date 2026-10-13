"""Typed decisions with a decision model (Ollama `/v1/systemone`, e.g. nimble:9b).

A decision model does not write text: it reads `state` and answers predefined questions, each
one of `choice` (pick a key), `score` (pick a level) or `noul` (a yes/no probability), with
probabilities. That makes it a fast, inspectable router for multi-agent scenarios.
"""

import json
import re

import httpx

from ..streaming import Backend, collect, stream_chat


def system_one(
    state: dict, questions: dict, model: str = "nimble:9b", base_url: str = "http://127.0.0.1:11434"
) -> dict:
    """Raw call; returns {"answers": {...}, "usage": {...}}."""
    with httpx.Client(timeout=300, trust_env=False) as client:
        r = client.post(
            base_url + "/v1/systemone",
            json={"model": model, "state": state, "questions": questions},
        )
    if r.is_error:
        raise RuntimeError(f"/v1/systemone failed ({r.status_code}): {r.text[:300]}")
    return r.json()


def summarize(answer: dict) -> dict:
    """One answer as {"value", "probabilities", "confidence"} whatever its type."""
    kind = answer.get("type")
    if kind == "choice":
        return {
            "value": answer.get("choice"),
            "probabilities": answer.get("probabilities", {}),
            "confidence": answer.get("confidence"),
        }
    if kind == "score":
        legend = answer.get("legend", {})
        probs = {legend.get(k, k): v for k, v in (answer.get("probabilities") or {}).items()}
        score = answer.get("score")
        label = legend.get(str(round(score)), score) if score is not None else None
        return {"value": label, "probabilities": probs, "confidence": answer.get("confidence")}
    if kind == "noul":
        p = answer.get("noul")
        return {"value": p is not None and p >= 0.5, "probabilities": {"yes": p}, "confidence": p}
    return {"value": answer, "probabilities": {}, "confidence": None}


def choose_with_llm(text: str, instructions: str, options: dict, model: str, base_url: str) -> dict:
    """Fallback router for when no decision model is available: ask a chat model for JSON."""
    listing = "\n".join(f"- {key}: {desc}" for key, desc in options.items())
    prompt = (
        f"{instructions}\n\nOptions:\n{listing}\n\nInput:\n{text}\n\n"
        'Reply with JSON only, like {"choice": "<option key>"}.'
    )
    _, answer, stats = collect(
        stream_chat(
            Backend("Ollama", "ollama", base_url),
            model,
            [{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=2048,
        )
    )
    choice = None
    if m := re.search(r"\{.*?\}", answer, re.S):
        try:
            choice = json.loads(m.group(0)).get("choice")
        except json.JSONDecodeError:
            pass
    if choice not in options:
        choice = next((k for k in options if k in answer), next(iter(options)))
    return {
        "value": choice,
        "probabilities": {},
        "confidence": None,
        "tokens": stats.output_tokens,
    }
