import json
import sys
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
def _read_catalog(mtime_ns: int) -> dict:
    return json.loads(CATALOG_FILE.read_text())


def load_catalog() -> dict:
    """data/catalog.json, read again whenever the file changes (a running platform picks up an
    edited catalog without a restart)."""
    return _read_catalog(CATALOG_FILE.stat().st_mtime_ns)


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
    """Usable on the chat/agent/compare pages: not an embedding or decision-only model, and not an
    OCR-only one (glm-ocr, minicpm-ocr, deepseek-ocr read text off images and do not converse;
    vision models that also chat, such as granite3.2-vision, list more tasks than OCR)."""
    entry = find(tag) or {}
    if {"embedding", "decision"} & set(entry.get("caps") or []):
        return False
    tasks = set(entry.get("tasks") or [])
    return not tasks or not tasks <= {"ocr"}


def other_models(category: str | None = None) -> list[dict]:
    """Models Ollama cannot run (speech, document parsing, rerankers)."""
    rows = load_catalog()["other_models"]
    return [m for m in rows if category is None or m["category"] == category]


def is_catalog_only(model: dict) -> bool:
    """Listed for reference only: no `gateway` block, so the platform cannot download or run it
    yet (the entry says `"status": "catalog"`)."""
    return not model.get("gateway")


def cloud_models(category: str | None = None) -> list[dict]:
    """Cloud speech models (Azure OpenAI deployments): audio is uploaded, nothing to download."""
    rows = load_catalog().get("cloud_models", [])
    return [m for m in rows if category is None or m["category"] == category]


def other_url(model: dict) -> str | None:
    """Official page of a non-Ollama entry: its own `url`, else the Hugging Face repo."""
    repo = model.get("repo") or ""
    if model.get("url"):
        return model["url"]
    if repo.count("/") == 1 and " " not in repo and "（" not in repo:
        return f"https://huggingface.co/{repo}"
    return None


COMMERCIAL_PREFIXES = ("apache", "mit", "bsd", "cc-by-4.0", "cc-by-sa", "cc0")


def licence_keys(model: dict) -> list[str]:
    """Licence badges (keys of ui.CAPS) of a non-Ollama entry: its `license_flags`
    (non_commercial / gated / custom) plus what the licence text itself says."""
    flags = set(model.get("license_flags") or [])
    low = (model.get("license_en") or model.get("license") or "").lower()
    words = low.replace("-", " ").replace("（", " ").replace("(", " ").split()
    if "nc" in words or "非商用" in low or "non-commercial" in low or "noncommercial" in low:
        flags.add("non_commercial")
    keys = []
    if "non_commercial" in flags:
        keys.append("non_commercial")
    elif "custom" not in flags and low.startswith(COMMERCIAL_PREFIXES):
        keys.append("commercial_ok")
    keys += [k for k, f in (("custom_license", "custom"), ("gated", "gated")) if f in flags]
    return keys


# Gateway backends that need Apple's MLX: on Linux a model on one of these is served only when
# its `linux` block names another backend (and usually other weights), else it is left out.
MLX_BACKENDS = {"mlx-whisper", "mlx-audio", "mlx-audio-realtime", "mlx-diarization"}
PLATFORM = sys.platform  # tests pin it to "darwin" so the catalog's shape is the same everywhere


def for_platform(gw: dict, platform: str | None = None) -> dict | None:
    """A catalog `gateway` block as this machine runs it: on Linux its `linux` keys replace
    the macOS ones (backend, weights, path, convert, options); None when it needs MLX."""
    linux = gw.get("linux")
    gw = {k: v for k, v in gw.items() if k != "linux"}
    if (platform or PLATFORM) == "darwin":
        return gw
    gw.update(linux or {})
    return None if gw["backend"] in MLX_BACKENDS else gw


def gateway_models(task: str | None = None) -> list[dict]:
    """Non-Ollama models served by the model gateway on this machine: the entry's `gateway`
    block (for_platform) plus display fields (name, license, recommended) from the entry."""
    rows = []
    for m in other_models():
        if gw := m.get("gateway"):
            gw = for_platform(gw)
            if gw and (task is None or gw["task"] == task):
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
