"""Artificial Analysis (artificialanalysis.ai) scores for the models we test.

AA has no free bulk download, but each model page carries the records of that model's variants
(reasoning levels) in its page data. `fetch_all` reads those pages (one a second, about 20) and
keeps the fields we show; `load` returns the saved copy. The copy lives in storage/ (not in git:
the numbers are AA's, shown with a link to the source), so a new server fetches its own:

    uv run python -m local_llm.aa_scores            # download and write the file
    uv run python -m local_llm.aa_scores --cache D  # reuse / fill a folder of saved pages

File: storage/llm_eval/aa/aa_models.json  {"fetched": date, "models": {our name: {...}}}
Fractions are stored as AA gives them (0..1); Elo scores and the index are plain numbers.
AA tests through cloud APIs at full precision, in English; our own runs are 4-bit local models.
"""

import datetime
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "storage" / "llm_eval" / "aa" / "aa_models.json"
URL = "https://artificialanalysis.ai/models/"
# our name -> (AA page, variant slug). The variant is the one closest to how we run the model:
# thinking on for local models; the highest effort AA lists for cloud models.
MODELS = {
    "Gemma 4 31B": ("gemma-4-31b", "gemma-4-31b"),
    "Gemma 4 31B（不思考）": ("gemma-4-31b", "gemma-4-31b-non-reasoning"),
    "Gemma 4 26B": ("gemma-4-26b-a4b", "gemma-4-26b-a4b"),
    "Gemma 4 12B": ("gemma-4-12b", "gemma-4-12b"),
    "Gemma 4 E4B": ("gemma-4-e4b", "gemma-4-e4b"),
    "Qwen3.8 27B": ("qwen3-8-27b", "qwen3-8-27b"),
    "Qwen3.8 27B（不思考）": ("qwen3-8-27b", "qwen3-8-27b-non-reasoning"),
    "Qwen3.6 35B-A3B": ("qwen3-6-35b-a3b", "qwen3-6-35b-a3b"),
    "Qwen3.6 35B-A3B（不思考）": ("qwen3-6-35b-a3b", "qwen3-6-35b-a3b-non-reasoning"),
    "Qwen3.5 9B": ("qwen3-5-9b", "qwen3-5-9b"),
    "Qwen3 30B-A3B": ("qwen3-30b-a3b-instruct-reasoning", "qwen3-30b-a3b-instruct-reasoning"),
    "Qwen3-Coder 30B-A3B": ("qwen3-coder-30b-a3b-instruct", "qwen3-coder-30b-a3b-instruct"),
    "GLM-4.7-Flash": ("glm-4-7-flash", "glm-4-7-flash"),
    "gpt-oss 20B": ("gpt-oss-20b", "gpt-oss-20b"),
    "Nemotron 3.5 Bolt E4B": ("nemotron-3-5-bolt-e4b-4gb", "nemotron-3-5-bolt-e4b-4gb"),
    "Claude Fable 5.2": ("claude-fable-5-2", "claude-fable-5-2"),
    "GPT-6.2 Astra": ("gpt-6-2-astra", "gpt-6-2-astra"),
    "Gemini 4 Carbon": ("gemini-4-carbon", "gemini-4-carbon"),
    "Claude Opus 5.5": ("claude-opus-5-5", "claude-opus-5-5"),
    "Claude Sonnet 5.5": ("claude-sonnet-5-5", "claude-sonnet-5-5"),
    "Claude Haiku 5.5": ("claude-haiku-5-5", "claude-haiku-5-5"),
    "GPT-5.5": ("gpt-5-5", "gpt-5-5"),
    "Gemini 3.8 Flash": ("gemini-3-8-flash", "gemini-3-8-flash"),
}
FIELDS = [
    "name", "releaseDate", "isOpenWeights", "parameters", "contextWindowTokens", "licenseName",
    "intelligenceIndex", "omniscienceAccuracy", "omniscienceHallucinationRate", "gdpval",
    "gdpPdfAllPass", "automationBenchPartialScore", "enterpriseOpsGym", "itBenchSre", "tau2",
    "tauBanking", "terminalBench21", "terminalBench40", "frontierCode20", "scicode",
    "livecodebench", "lcr", "ifbench", "hle", "hleDiamond", "gpqa", "mmmuPro",
    "price1mInputTokens", "price1mOutputTokens",
]  # fmt: skip


def page_data(html: str) -> str:
    """The page's data stream: the strings of every self.__next_f.push([1, "..."]) joined."""
    parts = re.findall(r'self\.__next_f\.push\(\[1,("(?:[^"\\]|\\.)*")\]\)', html)
    return "".join(json.loads(p) for p in parts)


def variants(html: str) -> dict:
    data = page_data(html)
    start = data.index("[", data.index('"initialModels":'))
    return {m["slug"]: m for m in json.JSONDecoder().raw_decode(data[start:])[0]}


def fetch(page: str, cache: Path | None = None) -> str:
    saved = cache / f"{page}.html" if cache else None
    if saved and saved.exists():
        return saved.read_text(encoding="utf-8", errors="ignore")
    req = urllib.request.Request(URL + page, headers={"User-Agent": "Mozilla/5.0 research-script"})
    html = urllib.request.urlopen(req, timeout=60).read().decode("utf-8", errors="ignore")
    if saved:
        saved.write_text(html, encoding="utf-8")
    time.sleep(1)  # one page a second: be gentle
    return html


def row(m: dict, page: str) -> dict:
    out = {k: m.get(k) for k in FIELDS}
    out["outputSpeed"] = (m.get("headline") or {}).get("outputSpeed")
    out["briefcaseElo"] = ((m.get("briefcaseBreakdown") or {}).get("overall") or {}).get("elo")
    out["url"] = URL + page
    return out


def fetch_all(cache: Path | None = None, progress=lambda done, total: None) -> dict:
    """Download every model's page and write OUT. Models AA does not list are left out (and
    named under "missing"); a page that fails keeps the whole old file untouched (raises)."""
    pages, out, missing = {}, {}, []
    for n, (ours, (page, slug)) in enumerate(MODELS.items()):
        if page not in pages:
            pages[page] = variants(fetch(page, cache))
        m = pages[page].get(slug)
        if m is None:
            missing.append(ours)
        else:
            out[ours] = row(m, page)
        progress(n + 1, len(MODELS))
    data = {"fetched": datetime.date.today().isoformat(), "models": out, "missing": missing}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return data


def load() -> dict | None:
    """The saved scores, or None when this server has not fetched them yet."""
    try:
        data = json.loads(OUT.read_text(encoding="utf-8"))
        return data if isinstance(data.get("models"), dict) else None
    except (OSError, ValueError):
        return None


def is_local(m: dict) -> bool:
    """Open weights we can run ourselves (AA's flag); cloud-only models are the reference."""
    return bool(m.get("isOpenWeights"))


def main() -> None:
    cache = Path(sys.argv[sys.argv.index("--cache") + 1]) if "--cache" in sys.argv else None
    data = fetch_all(cache)
    for name in data["missing"]:
        print(f"missing: {name}", file=sys.stderr)
    print(f"wrote {OUT} ({len(data['models'])} models)")


if __name__ == "__main__":
    main()
