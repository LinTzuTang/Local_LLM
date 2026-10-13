import json
from functools import cache
from pathlib import Path

CATALOG_FILE = Path(__file__).resolve().parents[2] / "data" / "catalog.json"
FIT_LABELS = {"ok": "✅ 可跑", "tight": "⚠️ 吃緊", "no": "❌ 放不下"}
FIT_LABELS_EN = {"ok": "✅ fits", "tight": "⚠️ tight", "no": "❌ too big"}
# A model's `caps` are Ollama's capability badges; `tags` are labels derived by us (from the
# tasks, context length, parameter count and licence). Both are keys of ui.CAPS in the app.
CAPABILITY_KEYS = ("tools", "thinking", "vision", "audio", "embedding", "decision")
TAG_KEYS = ("code", "translate", "ocr", "taiwan", "multilingual", "long_context", "moe",
            "commercial_ok", "non_commercial")  # fmt: skip
LONG_CONTEXT = 128_000  # tokens; Ollama shows 128000 as "125K"


@cache
def load_catalog() -> dict:
    return json.loads(CATALOG_FILE.read_text())


def localized(obj: dict, field: str, lang: str):
    """`field` in the UI language: English uses the `<field>_en` sibling when it exists."""
    if lang == "en" and obj.get(f"{field}_en") not in (None, "", []):
        return obj[f"{field}_en"]
    return obj.get(field)


def task_labels() -> dict[str, str]:
    return load_catalog()["tasks"]


def models(task: str | None = None) -> list[dict]:
    rows = load_catalog()["models"]
    return [m for m in rows if task is None or task in m["tasks"]]


def find(tag: str) -> dict | None:
    return next((m for m in load_catalog()["models"] if same_model(m["tag"], tag)), None)


def default_model() -> str:
    return next(m["tag"] for m in load_catalog()["models"] if m.get("default"))


def same_model(a: str, b: str) -> bool:
    """Ollama treats a tag without ':' as ':latest'."""
    return normalize(a) == normalize(b)


def normalize(tag: str) -> str:
    return tag if ":" in tag else f"{tag}:latest"


def tiers() -> list[dict]:
    return load_catalog()["tiers"]


def tier_of(model: dict) -> dict:
    """Size tier by Q4 download size; the last tier has no upper bound."""
    return next(t for t in tiers() if t["max_gb"] is None or model["size_gb"] <= t["max_gb"])


def is_chat(tag: str) -> bool:
    """Usable on the chat/agent/compare pages (not an embedding or decision-only model)."""
    caps = (find(tag) or {}).get("caps") or []
    return not {"embedding", "decision"} & set(caps)


def other_models(category: str | None = None) -> list[dict]:
    """Models Ollama cannot run (speech, document parsing, rerankers)."""
    rows = load_catalog()["other_models"]
    return [m for m in rows if category is None or m["category"] == category]


def gateway_models(task: str | None = None) -> list[dict]:
    """Non-Ollama models served by the model gateway: the entry's `gateway` block plus
    display fields (name, license, recommended) from the entry itself."""
    rows = []
    for m in other_models():
        if gw := m.get("gateway"):
            if task is None or gw["task"] == task:
                keys = ("name", "name_en", "license", "license_en", "repo")
                base = {k: m[k] for k in keys if k in m}
                rows.append({**base, "recommended": m.get("recommended", False), **gw})
    return rows


def is_huggingface(tag: str) -> bool:
    """Pulled straight from a Hugging Face GGUF repo (`hf.co/...`), not Ollama's library."""
    return tag.startswith("hf.co/")


def all_capabilities(model: dict) -> list[str]:
    """Ollama's capability badges (`caps`) followed by our derived labels (`tags`), no repeats.

    Both use the capability keys of app/ui.py (tools, vision, code, long_context, ...).
    """
    return list(dict.fromkeys([*(model.get("caps") or []), *(model.get("tags") or [])]))


def families() -> list[str]:
    return sorted({m["family"] for m in models()}, key=str.casefold)


def context_label(model: dict) -> str:
    """Context window the way Ollama prints it: 262144 -> "256K", 1048576 -> "1M"."""
    tokens = model.get("context")
    if not tokens:
        return ""
    if tokens >= 1 << 20 and tokens % (1 << 20) == 0:
        return f"{tokens >> 20}M"
    if tokens >= 1024:
        return f"{round(tokens / 1024)}K"
    return str(tokens)


def page_url(model: dict) -> str:
    """Model page: the entry's own `url` (Hugging Face, Cohere) or its Ollama library page."""
    return model.get("url") or f"https://ollama.com/library/{model['tag'].split(':')[0]}"


def pull_command(model: dict) -> str:
    return f"ollama pull {model['tag']}"
