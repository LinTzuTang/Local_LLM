"""Multi-step tool-calling tests on the platform: BFCL multi-turn, BFCL web search, BFCL memory and
the company-tools selection test, on a few questions picked by the user.

The platform does not reimplement them: it runs the same batch scripts as the report
(scripts/llm_eval/run_bfcl.py, run_bfcl_agentic.py and the company test's script) in the BFCL venv
(storage/llm_eval/bfcl_venv), each run in its own folder under storage/platform_steps/, and reads
their result and score files back. A server without those (the venv, the scripts, BFCL's data,
the company tool definitions) simply does not offer the test: `missing()` says why. The company
test is internal: where it lives comes from storage/llm_eval/company_test.json (not in git):
{"script": ..., "questions": ..., "tools": ..., "results_env": ...}, paths relative to the project.

Local models run with their own sampling (model-card defaults); the report's batch runs used
BFCL's 0.001 for most local models, so a live result can differ from the report.
"""

import ast
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "storage" / "llm_eval"
BFCL_PY = EVAL / "bfcl_venv" / "bin" / "python"
SCRIPTS = ROOT / "scripts" / "llm_eval"
COMPANY_CONF = EVAL / "company_test.json"  # local only, see the module docstring
WORK_ROOT = ROOT / "storage" / "platform_steps"

KINDS = ("multi_turn", "search", "memory", "company")
# cloud models the batch scripts know, and the .env key each needs (the company test has no Gemini)
CLOUD = {
    "claude-sonnet-5-5": ("Claude Sonnet 5.5", "AWS_ACCESS_KEY_ID"),
    "claude-haiku-5-5": ("Claude Haiku 5.5", "AWS_ACCESS_KEY_ID"),
    "claude-opus-5-5": ("Claude Opus 5.5", "AWS_ACCESS_KEY_ID"),
    "gpt-5.5": ("GPT-5.5", "AZURE_OPENAI_API_KEY"),
    "gemini-3.8-flash": ("Gemini 3.8 Flash", "GOOGLE_APPLICATION_CREDENTIALS"),
}
# rough seconds per question for one local model (Gemma 4 31B on the Mac in the batch runs)
# (11-17, qwen3.5:4b on the Mac: 5 memory prerequisites + 1 question 643 s; 1 search 191 s)
SECONDS = {"multi_turn": 120, "search": 200, "memory_prereq": 120, "memory": 40, "company": 15}
MEMORY_BACKENDS = ("kv", "vector", "rec_sum")


def _jsonl(path: Path) -> list[dict]:
    return (
        [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []
    )


def bfcl_data() -> Path | None:
    found = sorted((EVAL / "bfcl_venv" / "lib").glob("python*/site-packages/bfcl_eval/data"))
    return found[0] if found else None


def env_keys() -> set[str]:
    """Names (not values) of the keys set in .env or the environment."""
    names = {k for k, v in os.environ.items() if v}
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(errors="replace").splitlines():
            m = re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)", line)
            if m and m[2].strip().strip("'\""):
                names.add(m[1])
    return names


def missing(kind: str) -> str | None:
    """Why `kind` cannot run on this server (None: it can). Cloud needs are checked by the page."""
    if not BFCL_PY.exists():
        return "no BFCL venv (storage/llm_eval/bfcl_venv)"
    script = {
        "multi_turn": "run_bfcl.py",
        "search": "run_bfcl_agentic.py",
        "memory": "run_bfcl_agentic.py",
    }.get(kind)
    if kind == "company":
        c = company()
        if c is None:
            return "no company test configured on this server (storage/llm_eval/company_test.json)"
        if not c["script"].exists():
            return "the company test's script is missing"
        if not c["tools"].exists():
            return "no company tool definitions on this server"
        return None
    if not (SCRIPTS / script).exists():
        return f"no scripts/llm_eval/{script}"
    if bfcl_data() is None:
        return "no BFCL data in the venv"
    ids = (
        EVAL
        / ("bfcl" if kind == "multi_turn" else "bfcl_agentic")
        / "test_case_ids_to_generate.json"
    )
    if not ids.exists():
        return f"no question list ({ids.relative_to(ROOT)})"
    return None


def _first_user(turn) -> str:
    for m in turn if isinstance(turn, list) else []:
        if isinstance(m, dict) and m.get("role") == "user":
            return str(m.get("content", ""))
    return ""


def multi_turn_questions() -> list[dict]:
    """The 100 multi-turn questions of our English BFCL sample: id, category, the user's turns."""
    data, ids = (
        bfcl_data(),
        json.loads((EVAL / "bfcl" / "test_case_ids_to_generate.json").read_text()),
    )
    out = []
    for cat, chosen in ids.items():
        if not cat.startswith("multi_turn"):
            continue
        rows = {r["id"]: r for r in _jsonl(data / f"BFCL_v4_{cat}.json")}
        for i in chosen:
            if i in rows:
                turns = [_first_user(t) for t in rows[i]["question"]]
                out.append({"id": i, "category": cat, "turns": turns})
    return out


def _agentic_ids() -> dict[str, list[str]]:
    return json.loads((EVAL / "bfcl_agentic" / "test_case_ids_to_generate.json").read_text())


def search_questions() -> list[dict]:
    """Our 20 web-search questions (data ids web_search_N, run as web_search_base_N)."""
    rows = {r["id"]: r for r in _jsonl(bfcl_data() / "BFCL_v4_web_search.json")}
    out = []
    for i in _agentic_ids().get("web_search_base", []):
        r = rows.get("web_search_" + i.rsplit("_", 1)[1])
        if r:
            out.append(
                {
                    "id": i,
                    "category": "web_search_base",
                    "turns": [_first_user(t) for t in r["question"]],
                }
            )
    return out


def memory_sets(backend: str) -> dict[str, dict]:
    """scenario -> {"prereqs": [ids], "questions": [{"id", "turns"}]} for one memory backend."""
    rows = {r["id"]: r for r in _jsonl(bfcl_data() / "BFCL_v4_memory.json")}
    out: dict[str, dict] = {}
    for i in _agentic_ids().get(f"memory_{backend}", []):
        scenario = i.split("-")[1]
        entry = out.setdefault(scenario, {"prereqs": [], "questions": []})
        if "_prereq_" in i:
            entry["prereqs"].append(i)
        else:
            r = rows.get("memory_" + i.removeprefix(f"memory_{backend}_"))
            turns = [_first_user(t) for t in r["question"]] if r else []
            entry["questions"].append({"id": i, "category": f"memory_{backend}", "turns": turns})
    return out


def company() -> dict | None:
    """Where the company-tools test lives (storage/llm_eval/company_test.json), or None."""
    try:
        c = json.loads(COMPANY_CONF.read_text(encoding="utf-8"))
        return {
            "script": ROOT / c["script"],
            "questions": ROOT / c["questions"],
            "tools": ROOT / c["tools"],
            "results_env": c.get("results_env", "COMPANY_RESULTS"),
        }
    except (OSError, ValueError, KeyError, TypeError):
        return None


def company_questions() -> list[dict]:
    """The 109 company-tool questions: id, group letter, the request, what is expected."""
    spec = importlib.util.spec_from_file_location("_company_q", company()["questions"])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return [{"id": q[0], "category": q[0][0], "turns": [q[1]], "expect": q[2]} for q in mod.Q]


def estimate_seconds(kind: str, n: int, prereqs: int = 0) -> int:
    if kind == "memory":
        return prereqs * SECONDS["memory_prereq"] + n * SECONDS["memory"]
    return n * SECONDS[kind]


def model_key(tag: str) -> str:
    """BFCL registry name for a downloaded Ollama model (run_bfcl.py's BFCL_EXTRA_OLLAMA)."""
    return "plat-" + re.sub(r"[^a-z0-9.]+", "-", tag.lower()).strip("-")


def _literal(v):
    if isinstance(v, str):
        try:
            return ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return v
    return v


def _steps(raw) -> list[list[dict]]:
    """BFCL's per-turn result -> [[{"calls": [...]} | {"text": ...}, ...] per turn]."""
    turns = []
    for turn in _literal(raw) if isinstance(raw, (list, str)) else []:
        steps = []
        for step in turn if isinstance(turn, list) else [turn]:
            if isinstance(step, list):
                steps.append(
                    {"calls": [c if isinstance(c, dict) else {"raw": str(c)} for c in step]}
                )
            else:
                steps.append({"text": str(step)})
        turns.append(steps)
    return turns


def _bfcl_rows(work: Path, key: str, ids: list[str]) -> list[dict]:
    results = {}
    for f in (work / "result" / key).rglob("*_result.json"):
        for r in _jsonl(f):
            results[r["id"]] = r
    failed, scored = {}, set()
    for f in (work / "score" / key).rglob("*_score.json"):
        cat = f.name.removeprefix("BFCL_v4_").removesuffix("_score.json")
        scored.add(cat)
        for r in _jsonl(f)[1:]:
            err = r.get("error")
            failed[r["id"]] = (
                (err.get("error_message") or err.get("error_type"))
                if isinstance(err, dict)
                else " ".join(map(str, err))
                if isinstance(err, list)
                else str(err or "")
            )[:400]
    rows = []
    for i in ids:
        if "_prereq_" in i:
            continue
        r = results.get(i)
        lat = r.get("latency") if r else None
        flat = [
            x
            for t in (lat if isinstance(lat, list) else [lat])
            for x in (t if isinstance(t, list) else [t])
            if x
        ]
        rows.append(
            {
                "id": i,
                "answered": r is not None,
                "ok": r is not None and i not in failed,
                "why": failed.get(i, "" if r else "no answer (stopped or failed)"),
                "turns": _steps(r.get("result")) if r else [],
                "seconds": round(sum(flat), 1) if flat else None,
            }
        )
    return rows


def _run(
    cmd: list[str],
    env: dict,
    log: Path,
    done: Callable[[], int],
    total: int,
    progress: Callable[[int, int], None],
    cancel,
) -> int:
    with log.open("a") as out:
        proc = subprocess.Popen(
            cmd, cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL
        )
        while proc.poll() is None:
            if cancel is not None and cancel.is_set():
                proc.terminate()
                try:
                    proc.wait(10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                break
            progress(done(), total)
            time.sleep(2)
        progress(done(), total)
        return proc.wait()


def run(
    kind: str,
    model: str,
    ids: list[str],
    *,
    cond: str = "A",
    run_dir: Path,
    ollama_url: str = "http://127.0.0.1:11434",
    progress=lambda d, t: None,
    cancel=None,
) -> dict:
    """Run one model on `ids` (memory: prerequisite ids included) and return
    {"rows": [{"id", "ok", "why", "turns", "seconds"}], "log": path}. `model` is an Ollama tag or a
    CLOUD key."""
    run_dir.mkdir(parents=True, exist_ok=True)
    log = run_dir / "run.log"
    env = {**os.environ, "PYTHONWARNINGS": "ignore", "OLLAMA_URL": ollama_url}
    local = model not in CLOUD
    if kind == "company":
        c = company()
        env[c["results_env"]] = str(run_dir)
        tag = model.replace("/", "_").replace(":", "-")
        out = run_dir / f"{tag}__{cond}.jsonl"
        cmd = [
            str(BFCL_PY),
            str(c["script"]),
            "run",
            "--model",
            model,
            "--cond",
            cond,
            "--ids",
            ",".join(ids),
            "--threads",
            "1",
        ]
        _run(cmd, env, log, lambda: len(_jsonl(out)), len(ids), progress, cancel)
        got = {r["id"]: r for r in _jsonl(out)}
        rows = []
        for i in ids:
            r = got.get(i)
            g = (r or {}).get("grade") or {}
            calls = [{c.get("name"): c.get("args")} for c in (r or {}).get("calls") or []]
            steps = [{"calls": calls}] + ([{"text": r["final"]}] if r and r.get("final") else [])
            secs = [t.get("seconds") or 0 for t in (r or {}).get("turns") or []]
            rows.append(
                {
                    "id": i,
                    "answered": r is not None,
                    "ok": bool(g.get("success")),
                    "why": g.get("why", "" if r else "no answer (stopped or failed)"),
                    "turns": [steps] if r else [],
                    "seconds": round(sum(secs), 1) if secs else None,
                }
            )
        return {"rows": rows, "log": str(log)}

    key = model_key(model) if local else model
    if local:
        env["BFCL_EXTRA_OLLAMA"] = f"{key}={model}"
    by_cat: dict[str, list[str]] = {}
    for i in ids:
        cat = re.sub(r"_(prereq_)?\d+.*$", "", i) if kind != "multi_turn" else i.rsplit("_", 1)[0]
        by_cat.setdefault(cat, []).append(i)
    (run_dir / "test_case_ids_to_generate.json").write_text(json.dumps(by_cat, indent=1))
    if kind == "multi_turn":
        env["BFCL_WORK"] = str(run_dir)
        script = SCRIPTS / "run_bfcl.py"
    else:
        env["BFCL_AGENTIC_WORK"] = str(run_dir)
        script = SCRIPTS / "run_bfcl_agentic.py"
        cache = EVAL / "bfcl_agentic" / "search_cache.json"
        if cache.exists() and not (run_dir / "search_cache.json").exists():
            shutil.copy(
                cache, run_dir / "search_cache.json"
            )  # same answers for queries already made

    def answered() -> int:
        return sum(len(_jsonl(f)) for f in (run_dir / "result" / key).rglob("*_result.json"))

    base = [str(BFCL_PY), str(script)]
    code = _run(
        base + ["run", "--models", key, "--threads", "1"],
        env,
        log,
        answered,
        len(ids),
        progress,
        cancel,
    )
    if not (cancel is not None and cancel.is_set()) and (run_dir / "result" / key).exists():
        subprocess.run(
            base + ["score", "--models", key],
            cwd=ROOT,
            env=env,
            stdout=log.open("a"),
            stderr=subprocess.STDOUT,
            timeout=900,
        )
    rows = _bfcl_rows(run_dir, key, ids)
    return {"rows": rows, "log": str(log), "exit": code}


def new_run_dir(job_id: str, model: str) -> Path:
    return (
        WORK_ROOT
        / time.strftime("%Y%m%d-%H%M%S")
        / f"{job_id}-{re.sub(r'[^A-Za-z0-9.]+', '-', model)}"
    )


def python_ok() -> bool:
    return sys.version_info >= (3, 10)
