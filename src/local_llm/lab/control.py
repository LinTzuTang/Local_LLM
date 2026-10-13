"""The control-center coding agent: reads and edits agent and scenario definitions.

Read tools are free; every write tool pauses for human approval (the UI shows the diff and the
validation result computed by `plan_change`). Writes are validated again when they run and are
versioned by `store`, so each change can be rolled back.

`config_tools(root, fresh)` builds the same tools for any copy of the project. When the target
is another copy, or when Python code was changed after this process started (the local coding
agent in `local_llm.coder` can add tools or node types), previews, writes and rollbacks run in
a new Python process that imports that copy's current code, so validation knows about them.
"""

import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from langchain.agents import create_agent as new_agent_graph  # `create_agent` is a tool below
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from .. import ollama_admin
from ..config import Settings
from ..providers import build_model
from . import store
from .engine import requirements
from .nodes import NODE_TYPES, describe_node_types

WRITE_TOOLS = (
    "update_agent",
    "create_agent",
    "update_node",
    "add_node",
    "remove_node",
    "set_edges",
    "save_scenario",
)


def _dump(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=1)


# --- planning a change (shared by the approval preview and the tool itself) -----------------


def plan_change(name: str, args: dict) -> dict:
    """What a write tool would change: {"kind", "id", "before", "after", "check"}."""
    if name in ("update_agent", "create_agent"):
        if name == "update_agent":
            before = store.agent(args["agent_id"])
            after = store.deep_merge(before, args.get("changes") or {})
            after["id"] = before["id"]
        else:
            after = args["definition"]
            before = next((a for a in store.agents() if a["id"] == after.get("id")), None)
        return {
            "kind": "agent",
            "id": after.get("id"),
            "before": before,
            "after": after,
            "check": store.validate_agent(after),
        }
    if name == "save_scenario":
        after = args["scenario"]
        before = next((s for s in store.scenarios() if s["id"] == after.get("id")), None)
    else:
        before = store.scenario(args["scenario_id"])
        after = json.loads(json.dumps(before))
        nodes = {n["id"]: i for i, n in enumerate(after["nodes"])}
        if name == "update_node":
            if args["node_id"] not in nodes:
                raise KeyError(f"no node '{args['node_id']}' in {before['id']}: {list(nodes)}")
            i = nodes[args["node_id"]]
            after["nodes"][i] = store.deep_merge(after["nodes"][i], args.get("changes") or {})
            after["nodes"][i]["id"] = args["node_id"]
        elif name == "add_node":
            node = args["node"]
            if node.get("id") in nodes:
                raise ValueError(f"node id '{node.get('id')}' already exists")
            after["nodes"].append(node)
            drop = {(e["from"], e["to"]) for e in args.get("remove_edges") or []}
            after["edges"] = [e for e in after["edges"] if (e["from"], e["to"]) not in drop]
            after["edges"] += args.get("edges") or []
        elif name == "remove_node":
            nid = args["node_id"]
            after["nodes"] = [n for n in after["nodes"] if n["id"] != nid]
            after["edges"] = [e for e in after["edges"] if nid not in (e["from"], e["to"])]
            after["edges"] += args.get("edges") or []
        elif name == "set_edges":
            after["edges"] = args["edges"]
        for key in ("output",):
            if args.get(key):
                after[key] = args[key]
    return {
        "kind": "scenario",
        "id": after.get("id"),
        "before": before,
        "after": after,
        "check": store.validate_scenario(after),
    }


def _apply(name: str, args: dict) -> str:
    try:
        plan = plan_change(name, args)
    except (KeyError, ValueError, TypeError, AttributeError) as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"
    check = plan["check"]
    if not check.ok:
        return "NOT SAVED, validation failed:\n- " + "\n- ".join(check.errors)
    save = store.save_agent if plan["kind"] == "agent" else store.save_scenario
    save(plan["after"], source="control-center")
    warnings = ("\nWarnings:\n- " + "\n- ".join(check.warnings)) if check.warnings else ""
    return f"Saved {plan['kind']} '{plan['id']}' (previous version kept in history).{warnings}"


def _failed(name: str, exc: Exception) -> dict:
    kind = "agent" if name in ("update_agent", "create_agent") else "scenario"
    error = f"{type(exc).__name__}: {exc}"
    return {"kind": kind, "id": None, "diff": "", "errors": [error], "warnings": [], "ok": False}


def _preview(name: str, args: dict) -> dict:
    """JSON-safe plan for an approval card: kind, id, diff, errors, warnings, ok."""
    try:
        plan = plan_change(name, args)
    except Exception as exc:
        return _failed(name, exc)
    check = plan["check"]
    return {
        "kind": plan["kind"],
        "id": plan["id"],
        "diff": store.diff(plan["before"], plan["after"]),
        "errors": check.errors,
        "warnings": check.warnings,
        "ok": check.ok,
    }


# --- the same operations in a fresh process (another copy of the project, or changed code) -----

_MARK = "@@CONTROL_RESULT@@"
_FRESH_SCRIPT = f"""
import json, sys
sys.path.insert(0, sys.argv[1])
from local_llm.lab import control
print({_MARK!r} + json.dumps(control.handle(json.load(sys.stdin)), ensure_ascii=False))
"""


def handle(request: dict) -> dict:
    """Run one request in this process: {"op": "preview" | "apply" | "rollback", ...}."""
    if request["op"] == "preview":
        return _preview(request["name"], request["args"])
    if request["op"] == "apply":
        return {"result": _apply(request["name"], request["args"])}
    if request["op"] == "rollback":
        check = store.rollback(request["file"])
        return {"ok": check.ok, "errors": check.errors}
    raise ValueError(f"unknown op {request['op']!r}")


def _is_here(root: Path | str | None) -> bool:
    return root is None or Path(root).resolve() == store.ROOT.resolve()


def _run(request: dict, root: Path | str | None, fresh: bool) -> dict:
    if _is_here(root) and not fresh:
        return handle(request)
    root = Path(root or store.ROOT).resolve()
    proc = subprocess.run(
        [sys.executable, "-c", _FRESH_SCRIPT, str(root / "src")],
        input=json.dumps(request, ensure_ascii=False),
        capture_output=True,
        text=True,
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
        timeout=180,
        check=False,
    )
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(_MARK):
            return json.loads(line[len(_MARK) :])
    raise RuntimeError(
        "the validation process failed (does the changed code import?):\n"
        + (proc.stderr or proc.stdout)[-2000:]
    )


def preview(name: str, args: dict, root: Path | str | None = None, fresh: bool = False) -> dict:
    """What a write tool would change, for the approval card (diff + validation)."""
    try:
        return _run({"op": "preview", "name": name, "args": args}, root, fresh)
    except Exception as exc:
        return _failed(name, exc)


def apply(name: str, args: dict, root: Path | str | None = None, fresh: bool = False) -> str:
    try:
        return _run({"op": "apply", "name": name, "args": args}, root, fresh)["result"]
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


def rollback(file: str, root: Path | str | None = None, fresh: bool = False) -> dict:
    """Restore the version from before a recorded change: {"ok", "errors"}."""
    try:
        return _run({"op": "rollback", "file": file}, root, fresh)
    except Exception as exc:
        return {"ok": False, "errors": [f"{type(exc).__name__}: {exc}"]}


def history(root: Path | str | None = None, limit: int = 30) -> list[dict]:
    """Version history of agents.json / scenarios.json in the project at `root`."""
    if _is_here(root):
        return store.history(limit)
    folder = Path(root) / "storage" / "lab" / "history"
    files = sorted(folder.glob("*.json"), reverse=True)[:limit] if folder.exists() else []
    return [{**json.loads(f.read_text()), "file": f.name} for f in files]


# --- tools ---------------------------------------------------------------------------------


def config_tools(
    root: Path | str | None = None, fresh: Callable[[], bool] | None = None
) -> tuple[list, list]:
    """(read tools, write tools) for the agents and scenarios of the project at `root`.

    `fresh()` returning True sends previews and writes to a new process (see module doc).
    """
    here = _is_here(root)
    root = Path(root or store.ROOT).resolve()

    def use_fresh() -> bool:
        return not here or bool(fresh and fresh())

    def items(kind: str) -> list[dict]:
        if here:
            return store.agents() if kind == "agents" else store.scenarios()
        return json.loads((root / "data" / "lab" / f"{kind}.json").read_text())[kind]

    def one(kind: str, item_id: str) -> dict:
        found = next((x for x in items(kind) if x["id"] == item_id), None)
        if found is None:
            known = [x["id"] for x in items(kind)]
            raise KeyError(f"no {kind[:-1]} '{item_id}'; known: {known}")
        return found

    def safe(fn, *args) -> str:
        try:
            return fn(*args)
        except Exception as exc:
            return f"ERROR: {type(exc).__name__}: {exc}"

    def write(name: str, args: dict) -> str:
        return apply(name, args, root, use_fresh())

    @tool
    def list_agents() -> str:
        """List agent definitions: id, name, tools, db_role, model."""
        keys = ("id", "name", "tools", "db_role", "model")
        return safe(lambda: _dump([{k: a.get(k) for k in keys} for a in items("agents")]))

    @tool
    def show_agent(agent_id: str) -> str:
        """Full JSON definition of one agent (system prompt, tools, examples)."""
        return safe(lambda: _dump(one("agents", agent_id)))

    @tool
    def list_scenarios() -> str:
        """List scenarios: id, name, node ids/types and the models they use."""

        def rows():
            out = []
            for s in items("scenarios"):
                need = requirements(s)
                out.append(
                    {
                        "id": s["id"],
                        "name": s["name"],
                        "nodes": [f"{n['id']}:{n['type']}" for n in s["nodes"]],
                        "models": sorted(need["ollama"] | need["gateway"]),
                    }
                )
            return _dump(out)

        return safe(rows)

    @tool
    def show_scenario(scenario_id: str) -> str:
        """Full JSON definition of one scenario (nodes with params, edges, inputs, output)."""
        return safe(lambda: _dump(one("scenarios", scenario_id)))

    @tool
    def list_node_types() -> str:
        """Node types usable in scenarios, with every parameter, its default and whether
        required."""
        return safe(lambda: _dump(describe_node_types()))

    @tool
    def list_models() -> str:
        """Models installed in Ollama (with capabilities) and in the model gateway."""
        out = {"ollama": [], "gateway": []}
        try:
            names = [m["name"] for m in ollama_admin.installed()]
        except Exception as exc:
            names, out["ollama"] = [], f"ollama unavailable: {exc}"
        for name in names:
            try:
                caps = ollama_admin.show(name).get("capabilities", [])
            except Exception:
                caps = []
            out["ollama"].append({"name": name, "capabilities": caps})
        try:
            from .. import gateway_client

            out["gateway"] = [
                {k: m.get(k) for k in ("id", "task", "installed")} for m in gateway_client.models()
            ]
        except Exception as exc:
            out["gateway"] = f"gateway unavailable: {exc}"
        return _dump(out)

    @tool
    def list_agent_tools() -> str:
        """Tools an agent definition may list (name, what it does, whether it calls the
        network). Tools added to the code in this session appear after the app restarts."""
        from ..agents import TOOL_INFO, all_tools

        return safe(
            lambda: _dump(
                [
                    {"name": n, "description": t.description, "network": TOOL_INFO[n].network}
                    for n, t in all_tools().items()
                    if n in TOOL_INFO
                ]
            )
        )

    @tool
    def validate_scenario(scenario: dict) -> str:
        """Check a full scenario JSON without saving it; returns errors and warnings."""

        def check():
            c = store.validate_scenario(scenario, {a["id"] for a in items("agents")})
            return _dump({"ok": c.ok, "errors": c.errors, "warnings": c.warnings})

        return safe(check)

    # --- write tools (each pauses for human approval) ---------------------------------------

    @tool
    def update_agent(agent_id: str, changes: dict) -> str:
        """Change fields of an agent, e.g. {"system_prompt": "...", "tools": [...], "model":
        "..."}. Nested objects merge; a null value deletes a field. `tools` is replaced whole,
        so pass the full list."""
        return write("update_agent", {"agent_id": agent_id, "changes": changes})

    @tool
    def create_agent(definition: dict) -> str:
        """Create an agent: {"id", "name": {"zh","en"}, "description": {"zh","en"},
        "system_prompt", "tools": [...], "examples": {"zh": [...], "en": [...]}} (+ "db_role"
        for database tools)."""
        return write("create_agent", {"definition": definition})

    @tool
    def update_node(scenario_id: str, node_id: str, changes: dict) -> str:
        """Change one node, e.g. {"params": {"model": "qwen3:1.7b"}} or {"label": {"zh","en"}}.
        Nested objects merge (params keep other keys); router `options` are replaced whole."""
        args = {"scenario_id": scenario_id, "node_id": node_id, "changes": changes}
        return write("update_node", args)

    @tool
    def add_node(
        scenario_id: str, node: dict, edges: list[dict], remove_edges: list[dict] | None = None
    ) -> str:
        """Add a node and its edges. To insert B between A and C: node=B,
        edges=[{"from":"A","to":"B"}, {"from":"B","to":"C"}], remove_edges=[{"from":"A",
        "to":"C"}]. Node = {"id", "type", "label": {"zh","en"}, "params": {...}} (see
        list_node_types)."""
        args = {
            "scenario_id": scenario_id,
            "node": node,
            "edges": edges,
            "remove_edges": remove_edges,
        }
        return write("add_node", args)

    @tool
    def remove_node(scenario_id: str, node_id: str, edges: list[dict] | None = None) -> str:
        """Remove a node and all its edges; pass `edges` to reconnect its neighbours."""
        args = {"scenario_id": scenario_id, "node_id": node_id, "edges": edges}
        return write("remove_node", args)

    @tool
    def set_edges(scenario_id: str, edges: list[dict], output: str | None = None) -> str:
        """Replace all edges of a scenario: [{"from","to","when"?}], START/END allowed.
        Optionally change the scenario's `output` state key."""
        return write("set_edges", {"scenario_id": scenario_id, "edges": edges, "output": output})

    @tool
    def save_scenario(scenario: dict) -> str:
        """Create a scenario or replace one completely (full JSON with id, name {"zh","en"},
        description, category, inputs, examples, nodes, edges, output)."""
        return write("save_scenario", {"scenario": scenario})

    read = [
        list_agents,
        show_agent,
        list_scenarios,
        show_scenario,
        list_node_types,
        list_models,
        list_agent_tools,
        validate_scenario,
    ]
    writes = [update_agent, create_agent, update_node, add_node, remove_node, set_edges]
    return read, writes + [save_scenario]


READ_TOOLS, _WRITE_TOOL_OBJECTS = config_tools()
TOOLS = READ_TOOLS + _WRITE_TOOL_OBJECTS

SCENARIO_GUIDE = f"""How scenarios work: a scenario is a graph. Nodes have an id, a type, a
bilingual label {{"zh","en"}} and params. Each node reads values from a shared state and writes
its result under params.output; prompts and templates insert state values with {{key}}. Edges go
from START, between nodes, to END. A router node's outgoing edges carry "when": "<option key>".
Several edges from one node run in parallel; a node with "join": "all" waits for all incoming
branches. Node types: {", ".join(NODE_TYPES)}."""

SYSTEM_PROMPT = f"""You are the control-center coding agent of Local Agent Lab, a local
multi-agent platform. You change agent definitions (data/lab/agents.json) and multi-agent
scenarios (data/lab/scenarios.json) when the user asks.

{SCENARIO_GUIDE}

Rules:
1. Inspect before you change: call show_scenario / show_agent (and list_node_types or list_models
   when you need a param or a model name). Only use models that list_models reports as installed;
   agent and llm nodes need models with the "tools"/"completion" capability.
2. Make the smallest change with the most specific write tool (update_node for a param,
   add_node to insert a step, set_edges to rewire, update_agent for prompts/tools). One change per
   call; when the user asks for the same change in several places, call the tool once per place.
3. Keep labels bilingual {{"zh": ..., "en": ...}} and ids in snake_case.
4. Every write waits for the user's approval; if it is rejected, ask what to do instead.
5. When done, summarise what changed (scenario/agent, node, before → after) and suggest running it
   on the Scenarios page. Answer in the user's language; for Chinese use Traditional Chinese.
"""


def build_control_agent(model: str, base_url: str):
    return new_agent_graph(
        model=build_model(
            Settings("ollama", model, base_url, max_tokens=8192, context=32768), temperature=0
        ),
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    name: {"allowed_decisions": ["approve", "reject"]} for name in WRITE_TOOLS
                },
                description_prefix="Configuration change needs approval",
            )
        ],
        checkpointer=InMemorySaver(),
    )
