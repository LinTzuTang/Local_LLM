"""Which models fit a scenario node, and which of them are already downloaded.

Each node type needs a different kind of model: chat LLMs, tool-calling LLMs for agents,
decision models for nimble routers, speech/document/reranker models from the gateway.
The scenario page lists only compatible models, marks the downloaded ones and picks a
downloaded one by default so a demo works without downloading anything.
"""

import streamlit as st

from local_llm import catalog, gateway_client, ollama_admin
from local_llm.lab.nodes import NODE_TYPES

GATEWAY_TASK = {
    "transcribe": "asr",
    "diarize": "diarize",
    "parse_document": "doc",
    "rerank": "rerank",
}


@st.cache_data(ttl=15, show_spinner=False)
def _ollama_installed(url: str) -> list[str]:
    try:
        return [m["name"] for m in ollama_admin.installed(url)]
    except Exception:
        return []


@st.cache_data(ttl=15, show_spinner=False)
def _gateway_status() -> dict[str, bool]:
    try:
        return {m["id"]: bool(m.get("installed")) for m in gateway_client.models()}
    except Exception:
        return {}


def _caps(tag: str, url: str) -> set[str]:
    info = catalog.find(tag)
    if info:
        return set(info.get("caps") or [])
    try:
        return set(ollama_admin.show(tag, url).get("capabilities", []))
    except Exception:
        return set()


def selectable(node: dict) -> bool:
    """Nodes whose model the page lets you choose (embedding is fixed by the index)."""
    return node["type"] in ("llm", "agent", "router", "classify", *GATEWAY_TASK)


def configured_model(node: dict) -> str | None:
    ntype = NODE_TYPES[node["type"]]
    params = {k: p.default for k, p in ntype.params.items()} | node.get("params", {})
    models = ntype.models(params)
    return models[0] if models else None


def choices(node: dict, url: str) -> tuple[list[str], dict[str, bool]]:
    """(options, installed?) for a node, downloaded models first."""
    ntype, params = node["type"], node.get("params", {})
    if ntype in GATEWAY_TASK:
        status = _gateway_status()
        options = [m["id"] for m in catalog.gateway_models(GATEWAY_TASK[ntype])]
        installed = {m: status.get(m, False) for m in options}
    else:
        local = _ollama_installed(url)
        if ntype in ("router", "classify") and params.get("method", "nimble") == "nimble":
            want = "decision"
        elif ntype == "agent":
            want = "tools"
        else:
            want = "chat"

        def fits(tag: str) -> bool:
            caps = _caps(tag, url)
            if want == "decision":
                return "decision" in caps
            if want == "tools":
                return "tools" in caps and catalog.is_chat(tag)
            return catalog.is_chat(tag) and "embedding" not in caps

        have = [m for m in local if fits(m)]
        more = [
            m["tag"]
            for m in catalog.models()
            if m.get("fit") != "no"
            and (m.get("recommended") or want == "decision")
            and not any(catalog.same_model(m["tag"], h) for h in have)
            and fits(m["tag"])
        ]
        options = have + more
        installed = {m: m in have for m in options}
    current = configured_model(node)
    if current and current not in options:
        options.append(current)
        installed[current] = False
    options.sort(key=lambda m: not installed[m])  # stable: downloaded first
    return options, installed


def default_choice(node: dict, url: str) -> str | None:
    """The configured model if downloaded, else the first downloaded compatible one."""
    options, installed = choices(node, url)
    current = configured_model(node)
    if current and installed.get(current):
        return current
    return next((m for m in options if installed[m]), current)
