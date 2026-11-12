"""Weekly look for new models that the catalog (data/catalog.json) does not list yet.

Two sources: Ollama's library (newest first) and Hugging Face's trending models for the speech,
document and rerank tasks the catalog covers. It only reports: what it finds is kept in
storage/model_watch.json and shown on the home page; the catalog itself is curated by hand.

    uv run local-agent model-watch          check now (whatever the date of the last check)
"""

import json
import re
import sys
import threading
from datetime import UTC, datetime, timedelta
from html import unescape
from pathlib import Path

import httpx

from . import catalog

PROJECT_DIR = Path(__file__).resolve().parents[2]
STATE_FILE = PROJECT_DIR / "storage" / "model_watch.json"

INTERVAL = timedelta(days=7)
RETRY_AFTER_ERROR = timedelta(days=1)
FIRST_LOOKBACK = timedelta(days=30)  # Ollama models updated this recently count on the first check
OLLAMA_PAGES = 3  # 20 models a page, newest first
# Hugging Face: a model counts when it is this recent and already has this many likes (trending
# lists are full of small fine-tunes; the likes keep only the ones people noticed).
HF_MAX_AGE = timedelta(days=60)
HF_MIN_LIKES = 30
HF_PER_TASK = 50
# catalog category -> how to ask Hugging Face for it
HF_QUERIES = {
    "asr": {"pipeline_tag": "automatic-speech-recognition"},
    "tts": {"pipeline_tag": "text-to-speech"},
    "voice_chat": {"pipeline_tag": "audio-text-to-text"},
    "doc": {"search": "ocr"},
    "rerank": {"pipeline_tag": "text-ranking"},
}
# Speech models that name their languages but not Chinese are left out.
SPEECH = {"asr", "tts", "voice_chat"}
LANGUAGE = re.compile(r"^[a-z]{2,3}$")

_lock = threading.Lock()


def _now() -> datetime:
    return datetime.now(UTC)


def _iso(when: datetime) -> str:
    return when.astimezone(UTC).isoformat(timespec="seconds")


def _when(text: str | None) -> datetime | None:
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


# --- sources ------------------------------------------------------------------------------------
_ITEM = re.compile(r'<li[^>]*>\s*<a href="/library/([^"]+)"(.*?)</a>\s*</li>', re.S)
_SPAN_TEXT = r"<span[^>]*>([^<]+)</span>"


def parse_ollama(page: str) -> list[dict]:
    """The models on one page of ollama.com/search."""
    found = []
    for name, body in _ITEM.findall(page):
        updated = re.search(r'title="Updated ([^"]+)"', body)
        when = None
        if updated:
            try:
                when = datetime.strptime(updated.group(1), "%B %d, %Y").replace(tzinfo=UTC)
            except ValueError:
                pass
        desc = re.search(r"<p[^>]*>(.*?)</p>", body, re.S)
        downloads = re.search(r'title="([\d,]+) downloads"', body)
        found.append(
            {
                "source": "ollama",
                "id": unescape(name),
                "name": unescape(name),
                "url": f"https://ollama.com/library/{name}",
                "date": _iso(when) if when else None,
                "summary": unescape(re.sub(r"<[^>]+>", "", desc.group(1))).strip() if desc else "",
                "caps": re.findall(r"bg-indigo-50[^>]*>([^<]+)</span>", body),
                "sizes": re.findall(r"bg-\[#ddf4ff\][^>]*>([^<]+)</span>", body),
                "downloads": int(downloads.group(1).replace(",", "")) if downloads else None,
            }
        )
    return found


def fetch_ollama(client: httpx.Client) -> list[dict]:
    found = []
    for page in range(1, OLLAMA_PAGES + 1):
        response = client.get(
            "https://ollama.com/search", params={"c": "local", "o": "newest", "p": page}
        )
        response.raise_for_status()
        rows = parse_ollama(response.text)
        if not rows:
            break
        found += rows
    return found


def _chinese_or_unknown(tags: list[str]) -> bool:
    languages = [t for t in tags if LANGUAGE.match(t)]
    return not languages or "zh" in languages


def fetch_hf(client: httpx.Client, now: datetime) -> list[dict]:
    found = []
    for category, query in HF_QUERIES.items():
        response = client.get(
            "https://huggingface.co/api/models",
            params={
                **query,
                "sort": "trendingScore",
                "direction": -1,
                "limit": HF_PER_TASK,
                "full": "true",
            },  # fmt: skip
        )
        response.raise_for_status()
        for m in response.json():
            created = _when(m.get("createdAt"))
            tags = m.get("tags") or []
            if not created or now - created > HF_MAX_AGE or (m.get("likes") or 0) < HF_MIN_LIKES:
                continue
            if category in SPEECH and not _chinese_or_unknown(tags):
                continue
            if any(t.startswith("base_model:quantized:") for t in tags):  # a copy in another format
                continue
            found.append(
                {
                    "source": "huggingface",
                    "id": m["id"],
                    "name": m["id"],
                    "url": f"https://huggingface.co/{m['id']}",
                    "date": _iso(created),
                    "category": category,
                    "summary": "",
                    "likes": m.get("likes"),
                    "downloads": m.get("downloads"),
                    "chinese": "zh" in tags,
                }
            )
    return found


# --- what is already known ----------------------------------------------------------------------
def known() -> tuple[set[str], set[str]]:
    """Ollama model names and Hugging Face repos (lower case) the catalog already lists."""
    data = catalog.load_catalog()
    names = {m["tag"].split(":")[0].lower() for m in data["models"]}
    repos = {m["repo"].lower().rstrip("/") for m in data["other_models"] if m.get("repo")}
    repos |= {r.split("huggingface.co/")[-1] for r in repos}
    return names, repos


# --- state --------------------------------------------------------------------------------------
def load(path: Path = STATE_FILE) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _save(state: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1))
    tmp.replace(path)


def unread(state: dict) -> list[dict]:
    """Models found since the home page notice was last dismissed, newest first; ones that have
    been added to the catalog since are left out."""
    seen_at = state.get("acknowledged_at") or ""
    names, repos = known()
    rows = [
        m
        for m in state.get("found", [])
        if m.get("found_at", "") > seen_at
        and m["id"].lower() not in (names if m["source"] == "ollama" else repos)
    ]
    return sorted(rows, key=lambda m: m.get("date") or "", reverse=True)


def acknowledge(path: Path = STATE_FILE) -> None:
    with _lock:
        state = load(path)
        state["acknowledged_at"] = _iso(_now())
        _save(state, path)


def check(
    path: Path = STATE_FILE, client: httpx.Client | None = None, now: datetime | None = None
) -> list[dict]:
    """Look now; returns the models that are new this time (also saved to the state file)."""
    now = now or _now()
    own = client is None
    client = client or httpx.Client(
        timeout=30, follow_redirects=True, headers={"User-Agent": "local-agent-lab"}
    )
    try:
        state = load(path)
        last = _when(state.get("last_check"))
        since = last or now - FIRST_LOOKBACK
        state["last_attempt"] = _iso(now)
        try:
            ollama, hf = fetch_ollama(client), fetch_hf(client, now)
        except (httpx.HTTPError, ValueError) as e:
            state["last_error"] = f"{type(e).__name__}: {e}"[:300]
            with _lock:
                _save(state, path)
            raise
        names, repos = known()
        seen = set(state.get("seen", []))
        new = []
        for m in ollama:
            key = f"ollama:{m['id']}"
            date = _when(m["date"])
            if (
                key in seen
                or m["id"].lower() in names
                or not date
                or date < since - timedelta(days=1)
            ):
                continue
            new.append(m)
        for m in hf:
            key = f"huggingface:{m['id']}"
            if key in seen or m["id"].lower() in repos:
                continue
            new.append(m)
        # the same Hugging Face repo can trend under two tasks
        unique = {f"{m['source']}:{m['id']}": m for m in new}
        for m in unique.values():
            m["found_at"] = _iso(now)
        state["found"] = (state.get("found", []) + list(unique.values()))[-300:]
        state["seen"] = sorted(seen | set(unique))
        state["last_check"] = _iso(now)
        state.pop("last_error", None)
        with _lock:
            _save(state, path)
        return list(unique.values())
    finally:
        if own:
            client.close()


def due(state: dict, now: datetime | None = None) -> bool:
    now = now or _now()
    last_ok = _when(state.get("last_check"))
    last_try = _when(state.get("last_attempt"))
    if last_try and state.get("last_error") and now - last_try < RETRY_AFTER_ERROR:
        return False
    return last_ok is None or now - last_ok >= INTERVAL


def check_if_due(path: Path = STATE_FILE) -> list[dict] | None:
    """Run check() when the last one is a week old (None when it was not due or failed)."""
    if not due(load(path)):
        return None
    try:
        return check(path)
    except (httpx.HTTPError, ValueError):
        return None


def main(argv: list[str]) -> int:
    try:
        new = check()
    except (httpx.HTTPError, ValueError) as e:
        print(f"Check failed: {e}", file=sys.stderr)
        return 1
    print(f"{len(new)} new model(s) not in data/catalog.json:")
    for m in new:
        print(f"  [{m['source']}] {m['name']}  {(m.get('date') or '')[:10]}  {m['url']}")
    return 0
