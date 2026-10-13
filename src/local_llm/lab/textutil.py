"""Small text helpers shared by the scenario nodes."""

import json
import re

PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def format_value(value) -> str:
    """Turn a state value into prompt text; retrieval hits become numbered passages."""
    if isinstance(value, list) and value and isinstance(value[0], dict) and "text" in value[0]:
        return "\n\n".join(
            f"[{i}] ({hit.get('title') or hit.get('doc', '')}) {hit['text']}"
            for i, hit in enumerate(value, start=1)
        )
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return "" if value is None else str(value)


def render(template: str, data: dict) -> str:
    """Fill `{key}` placeholders from the scenario state. Other braces (e.g. JSON examples in a
    prompt) are left alone, and missing keys become empty strings."""
    return PLACEHOLDER.sub(lambda m: format_value(data.get(m.group(1))), template)


def lookup(spec: str, data: dict):
    """An input parameter is either a state key (`question`) or a template (`{a}\\n{b}`)."""
    return render(spec, data) if "{" in spec else data.get(spec)


def placeholders(template: str) -> list[str]:
    return PLACEHOLDER.findall(template or "")


def preview(value, limit: int = 400) -> str:
    text = format_value(value)
    return text if len(text) <= limit else text[:limit] + f"…（+{len(text) - limit}）"
