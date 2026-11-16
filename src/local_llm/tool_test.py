"""Live tool-calling test: send one question with tools to a model, then check the calls.

Used by the Language models page (app/llm_tools_page.py). The questions are every single-turn
question of our batch runs, one file per suite in data/lab/tool_sets/ (written by
scripts/llm_eval/export_tool_samples.py): BFCL English, MediaTek's zh-TW BFCL and ACEBench zh. The
same files are offered for download on the Test sets page and can be uploaded back (parse_set).
The checker follows BFCL's AST checker in a simplified form (same string normalisation,
optional parameters, any order for parallel calls), so a live result can differ from the
official score in edge cases. ACEBench's "cannot solve" cases pass here whenever no tool is
called (ACEBench also wants an exact English phrase).

    from local_llm import tool_test
    out = tool_test.call_ollama("http://127.0.0.1:11434", "gemma4:26b", messages, tools)
    verdict = tool_test.check(sample["expect"], out["calls"], out["content"])
"""

import json
import re
import time
from itertools import permutations
from pathlib import Path

import httpx

from . import ollama_admin

SETS = Path(__file__).resolve().parents[2] / "data" / "lab" / "tool_sets"
SET_FORMAT = "local-agent-lab/tool-set/1"


def load_samples(folder: Path = SETS) -> list[dict]:
    """Every question of every set in `folder` (bfcl_en, bfcl_zhtw, acebench), in file order."""
    out = []
    for name in ("bfcl_en", "bfcl_zhtw", "acebench"):
        p = folder / f"{name}.json"
        if p.exists():
            out += json.loads(p.read_text())["samples"]
    return out


def parse_set(raw: bytes) -> dict:
    """An uploaded test set: the downloaded file as it is, or edited (questions removed, changed or
    added). Raises ValueError with a short reason when it is not usable."""
    try:
        data = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as e:
        raise ValueError(f"not JSON ({e})") from None
    rows = data.get("samples") if isinstance(data, dict) else data
    if not isinstance(rows, list) or not rows:
        raise ValueError('no questions: expected {"samples": [...]} or a list of questions')
    out = []
    for i, r in enumerate(rows, 1):
        if not isinstance(r, dict) or not isinstance(r.get("messages"), list) or not r["messages"]:
            raise ValueError(f"question {i}: needs a non-empty messages list")
        if not isinstance(r.get("tools", []), list):
            raise ValueError(f"question {i}: tools must be a list")
        expect = r.get("expect") or {"kind": "any_call"}
        if expect.get("kind") not in ("calls", "no_call", "any_call", "mention"):
            raise ValueError(f"question {i}: unknown expect.kind {expect.get('kind')!r}")
        sid = str(r.get("id") or i)
        suite = str(
            r.get("suite") or (data.get("suite") if isinstance(data, dict) else None) or "uploaded"
        )
        out.append(
            {
                **r,
                "id": sid,
                "suite": suite,
                "category": str(r.get("category") or "uploaded"),
                "key": str(r.get("key") or f"{suite}/{sid}"),
                "tools": r.get("tools") or [],
                "expect": expect,
                "batch": r.get("batch") or {},
            }
        )
    name = data.get("name") if isinstance(data, dict) else None
    return {"name": name or "uploaded", "samples": out}


def call_ollama(
    base_url: str,
    model: str,
    messages: list[dict],
    tools: list[dict],
    *,
    think: bool = True,
    temperature: float | None = 0.0,
    max_tokens: int = 4096,
    timeout: float = 600,
) -> dict:
    """One /api/chat request with tools. temperature=None keeps the model's own default."""
    options: dict = {"num_predict": max_tokens}
    if temperature is not None:
        options["temperature"] = temperature
    body = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "stream": False,
        "options": options,
    }
    think_value = ollama_admin.think_value(model, think, base_url)
    if think_value is not None:
        body["think"] = think_value
    started = time.perf_counter()
    r = httpx.post(f"{base_url.rstrip('/')}/api/chat", json=body, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    msg = data.get("message") or {}
    calls = []
    for c in msg.get("tool_calls") or []:
        fn = c.get("function") or {}
        args = fn.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"_raw": args}
        calls.append({"name": fn.get("name", ""), "arguments": args})
    return {
        "calls": calls,
        "content": msg.get("content") or "",
        "thinking": msg.get("thinking") or "",
        "total_s": round(time.perf_counter() - started, 2),
        "input_tokens": data.get("prompt_eval_count"),
        "output_tokens": data.get("eval_count"),
        "hit_limit": data.get("done_reason") == "length",
    }


# ------------------------------------------------------------------ checking


def norm(s: str) -> str:
    """BFCL's string standardisation: drop spaces and ,./-_*^, lower-case, ' -> "."""
    return re.sub(r"[ ,./\-_*^]", "", s).lower().replace("'", '"')


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def same_value(actual, expected) -> bool:
    """One concrete expected value (no alternatives) against the model's value."""
    if _num(actual) and _num(expected):
        return float(actual) == float(expected)
    if (
        isinstance(expected, str)
        and isinstance(actual, (str, int, float))
        and not isinstance(actual, bool)
    ):
        return norm(str(actual)) == norm(expected)
    if isinstance(expected, list) and isinstance(actual, list):
        return len(actual) == len(expected) and all(map(same_value, actual, expected))
    if isinstance(expected, dict) and isinstance(actual, dict):
        return set(actual) == set(expected) and all(
            same_value(actual[k], expected[k]) for k in expected
        )
    return actual == expected


def bfcl_value(actual, choices: list) -> bool:
    """BFCL answers list the acceptable values; a dict value lists choices per key."""
    for choice in choices:
        if isinstance(choice, dict) and isinstance(actual, dict):
            keys_ok = all(
                k in actual or "" in (v if isinstance(v, list) else [v]) for k, v in choice.items()
            )
            if (
                keys_ok
                and set(actual) <= set(choice)
                and all(
                    bfcl_value(actual[k], choice[k] if isinstance(choice[k], list) else [choice[k]])
                    for k in actual
                )
            ):
                return True
        elif (
            isinstance(choice, list)
            and isinstance(actual, list)
            and choice
            and all(isinstance(c, dict) for c in choice)
        ):
            if len(actual) == len(choice) and all(
                bfcl_value(a, [c]) for a, c in zip(actual, choice, strict=True)
            ):
                return True
        elif choice != "" and same_value(actual, choice):
            return True
    return False


def ace_value(actual, expected) -> bool:
    """ACEBench is looser than BFCL: a string passes when one contains the other."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        return all(k in actual and ace_value(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list) and isinstance(actual, list):
        return len(actual) == len(expected) and all(map(ace_value, actual, expected))
    if isinstance(actual, str) and not isinstance(expected, (dict, list, bool)):
        a, e = norm(actual), norm(str(expected))
        return bool(e) and (e in a or a in e)
    return same_value(actual, expected)


def match_call(call: dict, expected: dict, style: str, alias: dict) -> str | None:
    """None when the call matches; otherwise why not (a short English reason)."""
    ((want_name, want_args),) = expected.items()
    name = alias.get(call["name"], call["name"])
    if style == "ace":  # ACEBench numbers repeated calls of one API: name_1, name_2
        want_name = re.sub(r"_\d+$", "", want_name) if want_name != name else want_name
    if name.replace(".", "_") != want_name.replace(".", "_"):
        return f"function {call['name']} (expected {want_name})"
    args = call["arguments"]
    for k in args:
        if k not in want_args:
            return f"unexpected parameter {k}"
    for k, v in want_args.items():
        choices = v if style == "bfcl" else [v]
        optional = style == "bfcl" and "" in choices
        if k not in args:
            if optional:
                continue
            return f"missing parameter {k}"
        ok = bfcl_value(args[k], choices) if style == "bfcl" else ace_value(args[k], v)
        if not ok:
            want = [c for c in choices if c != ""] if style == "bfcl" else v
            got = json.dumps(args[k], ensure_ascii=False)
            return f"{k}={got} (expected {json.dumps(want, ensure_ascii=False)})"
    return None


def check(expect: dict, calls: list[dict], content: str = "") -> dict:
    """{"ok": bool, "code": short reason code, "detail": text}. Codes: ok, called (should not
    have), no_call, count, mismatch, not_mentioned."""
    kind = expect["kind"]
    if kind == "no_call":
        return {
            "ok": not calls,
            "code": "ok" if not calls else "called",
            "detail": ", ".join(c["name"] for c in calls),
        }
    if kind == "any_call":
        return {"ok": bool(calls), "code": "ok" if calls else "no_call", "detail": ""}
    if kind == "mention":
        if calls:
            return {"ok": False, "code": "called", "detail": ", ".join(c["name"] for c in calls)}
        missing = [w for w in expect["words"] if norm(w) not in norm(content)]
        return {
            "ok": not missing,
            "code": "ok" if not missing else "not_mentioned",
            "detail": ", ".join(missing),
        }
    want = expect["calls"]
    if not calls:
        return {"ok": False, "code": "no_call", "detail": ""}
    if len(calls) != len(want):
        return {"ok": False, "code": "count", "detail": f"{len(calls)} / {len(want)}"}
    style, alias = expect.get("style", "bfcl"), expect.get("alias") or {}
    first_reason = None
    for order in permutations(range(len(calls))) if len(calls) <= 6 else [range(len(calls))]:
        reasons = [match_call(calls[i], w, style, alias) for i, w in zip(order, want, strict=True)]
        if not any(reasons):
            return {"ok": True, "code": "ok", "detail": ""}
        first_reason = first_reason or next(r for r in reasons if r)
    return {"ok": False, "code": "mismatch", "detail": first_reason or ""}
