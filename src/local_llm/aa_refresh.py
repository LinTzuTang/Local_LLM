"""Keep the Artificial Analysis copies fresh: language models (aa_scores) and the other kinds of
model (aa_media, speech / image / video). The platform's weekly background loop (app/model_news.py)
calls `refresh_if_due()`; each copy is fetched again once it is a week old. A failure keeps the old
file and is only logged (AA may be down or may have changed its pages).

    uv run python -m local_llm.aa_refresh          # refresh now if due
    uv run python -m local_llm.aa_refresh --force  # refresh now
"""

import datetime
import importlib
import logging
import sys

log = logging.getLogger(__name__)
EVERY_DAYS = 7
MODULES = ("local_llm.aa_scores", "local_llm.aa_media")


def _age_days(data: dict | None) -> float:
    try:
        fetched = datetime.date.fromisoformat(data["fetched"])
    except (TypeError, KeyError, ValueError):
        return float("inf")
    return (datetime.date.today() - fetched).days


def refresh_if_due(force: bool = False) -> dict[str, str]:
    """Fetch each copy that is missing or a week old.
    Returns {module: "fresh" | "updated" | the error}."""
    out = {}
    for name in MODULES:
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        if not force and _age_days(mod.load()) < EVERY_DAYS:
            out[name] = "fresh"
            continue
        try:
            mod.fetch_all()
            out[name] = "updated"
        except Exception as exc:  # noqa: BLE001  (network, or AA changed its pages)
            log.warning("AA refresh failed for %s: %s", name, exc)
            out[name] = f"failed: {exc}"[:200]
    return out


if __name__ == "__main__":
    for module, result in refresh_if_due(force="--force" in sys.argv).items():
        print(f"{module}: {result}")
