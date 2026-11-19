"""Coding test on the platform: LiveCodeBench v6 on a few questions picked by the user.

Like step_bench.py, the platform does not reimplement the test: it runs the report's batch script
(scripts/llm_eval/run_lcb.py: LiveCodeBench's prompt, one answer per question, thinking on unless
switched off, a 32,768-token cap) and LiveCodeBench's own tester, each run in its own folder
under storage/platform_code/, and reads the score file back. The questions offered are the 60
every model answered in the batch runs (20 easy, 20 medium, 20 hard), so a live result can be
read next to them. A server without the files (the script, the question file, LiveCodeBench and
its venv under storage/coding_eval/) does not offer the test: `missing()` says why.
"""

import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
CE = ROOT / "storage" / "coding_eval"
SCRIPT = ROOT / "scripts" / "llm_eval" / "run_lcb.py"
DATA = CE / "lcb_data" / "test6.jsonl"
LCB = CE / "LiveCodeBench"
SCORE_PY = CE / "lcb_venv" / "bin" / "python"
RESULTS = CE / "results"
WORK_ROOT = ROOT / "storage" / "platform_code"
PICK = 20  # per difficulty: the batch runs' 60-question sample
# rough minutes per question for one model, thinking on (batch runs: Gemma 4 31B 1.8 h and
# Qwen3.8 27B 5.2 h for 60 questions on the GPU machine; hard questions take several times longer)
MINUTES = {"easy": 1, "medium": 3, "hard": 6}


def missing() -> str | None:
    """Why the test cannot run on this server (None: it can)."""
    if not SCRIPT.exists():
        return "no scripts/llm_eval/run_lcb.py"
    if not DATA.exists():
        return "no question file (storage/coding_eval/lcb_data/test6.jsonl)"
    if not (LCB / "lcb_runner").exists():
        return "no LiveCodeBench checkout (storage/coding_eval/LiveCodeBench)"
    if not SCORE_PY.exists():
        return "no scoring venv (storage/coding_eval/lcb_venv)"
    return None


def _script():
    spec = importlib.util.spec_from_file_location("_run_lcb", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def questions() -> list[dict]:
    """The 60 sampled questions: id, title, difficulty, platform, date, the statement, starter
    code (LeetCode) or "" (AtCoder: read stdin)."""
    rows = _script().chosen(SimpleNamespace(pick=PICK, limit=None, ids=""))
    return [
        {
            "id": r["question_id"],
            "title": r["question_title"],
            "difficulty": r["difficulty"],
            "platform": r["platform"],
            "date": str(r.get("contest_date", ""))[:10],
            "content": r["question_content"],
            "starter": r.get("starter_code") or "",
        }
        for r in rows
    ]


def batch_results() -> dict[str, dict[str, bool]]:
    """question id -> {model key: passed} from the batch runs' score files, where present."""
    out: dict[str, dict[str, bool]] = {}
    for f in sorted(RESULTS.glob(f"lcb_*-p{PICK}.score.json")):
        try:
            s = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        for r in s.get("rows", []):
            out.setdefault(r["question_id"], {})[s["model"]] = bool(r["pass"])
    return out


def estimate_minutes(difficulties: list[str]) -> int:
    return sum(MINUTES.get(d, 3) for d in difficulties)


def key(model: str, think: bool) -> str:
    """run_lcb.py's result key for a model."""
    return model.replace("/", "_").replace(":", "_") + ("" if think else "-nothink")


def _jsonl(path: Path) -> list[dict]:
    return (
        [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []
    )


def rows_from(run_dir: Path, model: str, think: bool, ids: list[str]) -> list[dict]:
    """One row per asked question from the run's answer and score files."""
    answer_file = run_dir / "result" / key(model, think) / "lcb_result.json"
    answers = {r["question_id"]: r for r in _jsonl(answer_file)}
    try:
        score = json.loads((run_dir / "score.json").read_text())
        scored = {r["question_id"]: r for r in score["rows"]}
    except (OSError, ValueError, KeyError):
        scored = {}
    rows = []
    for i in ids:
        a, s = answers.get(i), scored.get(i, {})
        why = ""
        if a is None:
            why = "no answer (stopped or failed)"
        elif a.get("error"):
            why = a["error"]
        elif a.get("done_reason") == "length":
            why = "hit the 32,768-token cap before finishing"
        elif s.get("no_code"):
            why = "no code block in the answer"
        elif not s:
            why = "not scored"
        elif not s.get("pass"):
            why = "failed at least one test"
        rows.append(
            {
                "id": i,
                "answered": a is not None,
                "ok": bool(s.get("pass")),
                "why": why,
                "capped": bool(a and a.get("done_reason") == "length"),
                "code": s.get("code") or "",
                "answer": (a or {}).get("content", "")[-4000:],
                "thinking_chars": (a or {}).get("thinking_chars"),
                "seconds": (a or {}).get("seconds"),
                "tokens": (a or {}).get("eval_count"),
            }
        )
    return rows


def run(
    model: str,
    ids: list[str],
    *,
    think: bool = True,
    run_dir: Path,
    ollama_url: str = "http://127.0.0.1:11434",
    progress: Callable[[int, int], None] = lambda d, t: None,
    cancel=None,
) -> dict:
    """Generate and score `ids` with one Ollama model. Returns {"rows": [...], "log": path}."""
    run_dir.mkdir(parents=True, exist_ok=True)
    log = run_dir / "run.log"
    env = {**os.environ, "OLLAMA_URL": ollama_url, "LCB_RESULTS": str(run_dir)}
    common = ["--model", model, "--pick", str(PICK), "--ids", ",".join(ids)]
    if not think:
        common.append("--no-think")
    answers = run_dir / "result" / key(model, think) / "lcb_result.json"
    with log.open("a") as out:
        proc = subprocess.Popen(
            [sys.executable, str(SCRIPT), "gen", *common],
            cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        )  # fmt: skip
        while proc.poll() is None:
            if cancel is not None and cancel.is_set():
                proc.terminate()
                try:
                    proc.wait(10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                break
            progress(len(_jsonl(answers)), len(ids))
            time.sleep(2)
        progress(len(_jsonl(answers)), len(ids))
        if answers.exists():  # also after Stop: what was answered is still scored
            subprocess.run(
                [str(SCORE_PY), str(SCRIPT), "score", *common, "--out", str(run_dir / "score.json")],  # noqa: E501
                cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT, timeout=1800,
            )  # fmt: skip
    return {"rows": rows_from(run_dir, model, think, ids), "log": str(log)}


def new_run_dir(job_id: str, model: str) -> Path:
    return (
        WORK_ROOT
        / time.strftime("%Y%m%d-%H%M%S")
        / f"{job_id}-{re.sub(r'[^A-Za-z0-9.]+', '-', model)}"
    )
