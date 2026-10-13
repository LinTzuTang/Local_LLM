"""Scenario and agent definitions on disk, with validation and version history.

`data/lab/scenarios.json` and `data/lab/agents.json` are the single source of truth. Every
save (from the UI or the control-center coding agent) first validates, then keeps the previous
version in `storage/lab/history/`, so any change can be inspected and rolled back.
"""

import copy
import difflib
import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

from .nodes import NODE_TYPES
from .textutil import placeholders

ROOT = Path(__file__).resolve().parents[3]
LAB_DIR = ROOT / "data" / "lab"
SCENARIOS_FILE = LAB_DIR / "scenarios.json"
AGENTS_FILE = LAB_DIR / "agents.json"
HISTORY_DIR = ROOT / "storage" / "lab" / "history"
START, END = "START", "END"
INPUT_TYPES = {"text", "audio", "document", "choice"}


# --- loading -------------------------------------------------------------------------------------


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def scenarios() -> list[dict]:
    return _read(SCENARIOS_FILE)["scenarios"]


def scenario(scenario_id: str) -> dict:
    for s in scenarios():
        if s["id"] == scenario_id:
            return s
    raise KeyError(f"no scenario '{scenario_id}'; known: {[s['id'] for s in scenarios()]}")


def agents() -> list[dict]:
    return _read(AGENTS_FILE)["agents"]


def agent(agent_id: str) -> dict:
    for a in agents():
        if a["id"] == agent_id:
            return a
    raise KeyError(f"no agent '{agent_id}'; known: {[a['id'] for a in agents()]}")


# --- validation ------------------------------------------------------------------------------


@dataclass
class Check:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_scenario(s: dict, agent_ids: set[str] | None = None) -> Check:
    """Structural checks that keep a scenario runnable."""
    check = Check()
    err, warn = check.errors.append, check.warnings.append
    for key in ("id", "name", "nodes", "edges"):
        if key not in s:
            err(f"missing field '{key}'")
    if check.errors:
        return check
    agent_ids = agent_ids if agent_ids is not None else {a["id"] for a in agents()}
    nodes = {}
    for node in s["nodes"]:
        nid, ntype = node.get("id"), node.get("type")
        if not nid or nid in (START, END):
            err(f"node id {nid!r} is not allowed")
            continue
        if nid in nodes:
            err(f"duplicate node id '{nid}'")
        nodes[nid] = node
        if ntype not in NODE_TYPES:
            err(f"node '{nid}': unknown type '{ntype}' (known: {sorted(NODE_TYPES)})")
            continue
        spec, params = NODE_TYPES[ntype].params, node.get("params", {})
        for name in params:
            if name not in spec:
                err(f"node '{nid}': unknown param '{name}' for type {ntype} ({sorted(spec)})")
        for name, p in spec.items():
            if p.required and params.get(name) in (None, "", [], {}):
                err(f"node '{nid}': param '{name}' is required")
        if ntype == "agent" and params.get("agent") and params["agent"] not in agent_ids:
            err(f"node '{nid}': agent '{params['agent']}' is not defined in agents.json")
        if ntype == "router" and not isinstance(params.get("options"), dict):
            err(f"node '{nid}': options must be an object {{key: description}}")
        if ntype == "rrf" and not isinstance(params.get("inputs"), list):
            err(f"node '{nid}': inputs must be a list of state keys")
        if node.get("join") not in (None, "any", "all"):
            err(f"node '{nid}': join must be 'any' or 'all'")

    outgoing, incoming = defaultdict(list), defaultdict(list)
    for e in s["edges"]:
        src, dst = e.get("from"), e.get("to")
        if src != START and src not in nodes:
            err(f"edge from unknown node '{src}'")
            continue
        if dst != END and dst not in nodes:
            err(f"edge to unknown node '{dst}'")
            continue
        if src == END or dst == START:
            err(f"edge {src} → {dst} goes the wrong way")
            continue
        if e.get("when"):
            router = nodes.get(src, {})
            if router.get("type") != "router":
                err(f"edge {src} → {dst}: only router nodes can have conditional ('when') edges")
            elif e["when"] != "*" and e["when"] not in router.get("params", {}).get("options", {}):
                err(f"edge {src} → {dst}: '{e['when']}' is not an option of router '{src}'")
        outgoing[src].append(e)
        incoming[dst].append(e)
    if not outgoing[START]:
        err("nothing starts: add an edge from START")
    if check.errors:
        return check

    def reach(start: str, graph: dict, key: str) -> set:
        seen, todo = {start}, deque([start])
        while todo:
            for e in graph[todo.popleft()]:
                if e[key] not in seen:
                    seen.add(e[key])
                    todo.append(e[key])
        return seen

    from_start = reach(START, outgoing, "to")
    to_end = reach(END, incoming, "from")
    for nid, node in nodes.items():
        if nid not in from_start:
            err(f"node '{nid}' is not reachable from START")
        elif nid not in to_end:
            err(f"node '{nid}' has no path to END")
        plain_in = [e for e in incoming[nid] if not e.get("when")]
        if node.get("join") == "all" and len(plain_in) < 2:
            warn(f"node '{nid}': join 'all' only matters with two or more incoming edges")
        router = node.get("type") == "router"
        conditional = [e for e in outgoing[nid] if e.get("when")]
        if router and conditional and len(conditional) != len(outgoing[nid]):
            err(f"router '{nid}': mix of conditional and unconditional outgoing edges")
        if router and conditional:
            missing = set(node["params"]["options"]) - {e["when"] for e in conditional}
            if missing and not any(e["when"] == "*" for e in conditional):
                warn(f"router '{nid}': options {sorted(missing)} have no edge (they end the run)")

    produced = {i["key"] for i in s.get("inputs", [])}
    for node in nodes.values():
        p = node.get("params", {})
        produced.add(p.get("output") or NODE_TYPES[node["type"]].params.get("output").default)
        if node["type"] == "classify":
            produced.update(p.get("questions", {}))
    for nid, node in nodes.items():
        p = node.get("params", {})
        for name in ("prompt", "system", "template", "input", "query", "role"):
            for key in placeholders(str(p.get(name) or "")):
                if key not in produced:
                    warn(f"node '{nid}': '{{{key}}}' in {name} is not produced by any node/input")
    for i in s.get("inputs", []):
        if i.get("type", "text") not in INPUT_TYPES:
            err(f"input '{i.get('key')}': type must be one of {sorted(INPUT_TYPES)}")
    if s.get("output") and s["output"] not in produced:
        warn(f"scenario output '{s['output']}' is not produced by any node")
    return check


def validate_agent(a: dict) -> Check:
    from ..agents import resolve_tools

    check = Check()
    for key in ("id", "name", "system_prompt"):
        if not a.get(key):
            check.errors.append(f"missing field '{key}'")
    try:
        resolve_tools(a.get("tools", []), a.get("db_role"))
    except ValueError as exc:
        check.errors.append(str(exc))
    from .. import company_db, skills

    roles = [i["role"] for i in company_db.identities()]
    db = {"db_list_tables", "db_describe_table", "db_query"} & set(a.get("tools", []))
    if db and not a.get("db_role"):
        check.errors.append(f"database tools need a db_role, one of {roles}")
    if a.get("db_role") and a["db_role"] not in roles:
        check.errors.append(f"unknown db_role '{a['db_role']}'; identities: {roles}")
    installed = {s["name"] for s in skills.list_skills()}
    for name in a.get("skills") or []:
        if name not in installed:
            check.errors.append(f"unknown skill '{name}'; installed: {sorted(installed)}")
    return check


# --- saving with history -----------------------------------------------------------------------


def _save(path: Path, key: str, items: list[dict]) -> None:
    doc = _read(path)
    doc[key] = items
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def _record(kind: str, item_id: str, before: dict | None, after: dict | None, source: str) -> Path:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{time.time_ns() % 1_000_000:06d}"
    path = HISTORY_DIR / f"{stamp}_{kind}_{item_id}.json"
    path.write_text(
        json.dumps(
            {
                "kind": kind,
                "id": item_id,
                "source": source,
                "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "before": before,
                "after": after,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return path


def save_scenario(s: dict, source: str = "ui") -> Check:
    check = validate_scenario(s)
    if not check.ok:
        return check
    items = scenarios()
    before = next((x for x in items if x["id"] == s["id"]), None)
    items = [s if x["id"] == s["id"] else x for x in items] if before else items + [s]
    _save(SCENARIOS_FILE, "scenarios", items)
    _record("scenario", s["id"], before, s, source)
    return check


def save_agent(a: dict, source: str = "ui") -> Check:
    check = validate_agent(a)
    if not check.ok:
        return check
    items = agents()
    before = next((x for x in items if x["id"] == a["id"]), None)
    items = [a if x["id"] == a["id"] else x for x in items] if before else items + [a]
    _save(AGENTS_FILE, "agents", items)
    _record("agent", a["id"], before, a, source)
    return check


def history(limit: int = 50) -> list[dict]:
    if not HISTORY_DIR.exists():
        return []
    files = sorted(HISTORY_DIR.glob("*.json"), reverse=True)[:limit]
    return [{**json.loads(f.read_text()), "file": f.name} for f in files]


def rollback(file: str) -> Check:
    """Restore the version from before a recorded change (a new history entry is written)."""
    entry = json.loads((HISTORY_DIR / Path(file).name).read_text())
    kind, before = entry["kind"], entry["before"]
    path, key = (SCENARIOS_FILE, "scenarios") if kind == "scenario" else (AGENTS_FILE, "agents")
    if before is None:  # the change created the item: rolling back removes it
        items = [x for x in _read(path)[key] if x["id"] != entry["id"]]
        _save(path, key, items)
        _record(kind, entry["id"], entry["after"], None, "rollback")
        return Check()
    return (save_scenario if kind == "scenario" else save_agent)(before, source="rollback")


def diff(before: dict | None, after: dict | None) -> str:
    a = json.dumps(before, ensure_ascii=False, indent=2).splitlines() if before else []
    b = json.dumps(after, ensure_ascii=False, indent=2).splitlines() if after else []
    return "\n".join(difflib.unified_diff(a, b, "before", "after", lineterm="", n=2))


def deep_merge(base: dict, changes: dict) -> dict:
    """Merge `changes` into a copy of `base`; nested objects merge, `null` deletes a key."""
    out = copy.deepcopy(base)
    for key, value in changes.items():
        if value is None:
            out.pop(key, None)
        elif isinstance(value, dict) and isinstance(out.get(key), dict) and key != "options":
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out
