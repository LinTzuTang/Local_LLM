"""Compile a scenario (JSON nodes + edges) into a LangGraph graph and run it with a trace.

State is a single `data` dict merged by a reducer, so parallel branches can each add their
own keys. Every node execution is recorded as a step (model, timing, inputs, outputs, details)
in `Run.trace`; the UI draws the flow diagram and the timeline from it while the run is going.
"""

import operator
import threading
import time
from collections import defaultdict
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from . import store
from .nodes import NODE_TYPES
from .textutil import preview


def _merge(left: dict | None, right: dict | None) -> dict:
    return {**(left or {}), **(right or {})}


class LabState(TypedDict):
    data: Annotated[dict, _merge]
    visited: Annotated[list, operator.add]


class Cancelled(Exception):
    pass


class Step:
    """What one node execution exposes to its implementation; backed by a trace entry."""

    def __init__(self, run: "Run", entry: dict):
        self._run, self._entry = run, entry
        self.inputs, self.detail = entry["inputs"], entry["detail"]
        self.ollama_url = run.ollama_url

    @property
    def stream(self) -> str:
        return self._entry["stream"]

    @stream.setter
    def stream(self, text: str) -> None:
        self._entry["stream"] = text[-2000:]

    def cancelled(self) -> bool:
        return self._run.cancel.is_set()


class Run:
    def __init__(self, scenario: dict, inputs: dict, ollama_url: str, cancel=None):
        self.scenario, self.inputs, self.ollama_url = scenario, inputs, ollama_url
        self.cancel = cancel or threading.Event()
        self.trace: list[dict] = []
        self.data: dict = dict(inputs)
        self.status, self.error = "pending", None
        self.started = self.finished = None
        self._lock = threading.Lock()

    def elapsed(self) -> float:
        if self.started is None:
            return 0.0
        return (self.finished or time.perf_counter()) - self.started

    def node_status(self) -> dict[str, str]:
        """Latest status per node id (a node can run more than once)."""
        status = {}
        for entry in self.trace:
            status[entry["node"]] = entry["status"]
        return status

    def _wrap(self, node: dict):
        ntype = NODE_TYPES[node["type"]]
        params = {
            name: p.default for name, p in ntype.params.items() if p.default is not None
        } | node.get("params", {})

        def execute(state: LabState) -> dict:
            if self.cancel.is_set():
                raise Cancelled()
            entry = {
                "node": node["id"],
                "type": node["type"],
                "label": node.get("label") or node["id"],
                "models": ntype.models(params),
                "status": "running",
                "start": time.perf_counter() - self.started,
                "end": None,
                "inputs": {},
                "outputs": {},
                "detail": {},
                "stream": "",
                "error": None,
            }
            with self._lock:
                self.trace.append(entry)
            try:
                updates = ntype.run(params, state["data"], Step(self, entry))
            except Exception as exc:
                entry.update(status="error", error=f"{type(exc).__name__}: {exc}")
                entry["end"] = time.perf_counter() - self.started
                raise
            entry["outputs"] = {k: preview(v, 3000) for k, v in updates.items()}
            entry["status"] = "cancelled" if self.cancel.is_set() else "done"
            entry["end"] = time.perf_counter() - self.started
            with self._lock:
                self.data.update(updates)
            return {"data": updates, "visited": [node["id"]]}

        return execute

    def compile(self):
        s = self.scenario
        check = store.validate_scenario(s)
        if not check.ok:
            raise ValueError("; ".join(check.errors))
        nodes = {n["id"]: n for n in s["nodes"]}
        graph = StateGraph(LabState)
        for node in s["nodes"]:
            graph.add_node(node["id"], self._wrap(node))

        def target(name: str) -> str:
            return {store.START: START, store.END: END}.get(name, name)

        by_source, incoming = defaultdict(list), defaultdict(list)
        for e in s["edges"]:
            by_source[e["from"]].append(e)
        for src, edges in by_source.items():
            conditional = [e for e in edges if e.get("when")]
            if conditional:
                key = nodes[src].get("params", {}).get("output", "route")
                mapping = {e["when"]: target(e["to"]) for e in conditional}
                fallback = mapping.pop("*", END)

                def path(state, key=key, mapping=mapping, fallback=fallback):
                    return mapping.get(str(state["data"].get(key)), fallback)

                graph.add_conditional_edges(
                    target(src), path, sorted({*mapping.values(), fallback})
                )
            for e in edges:
                if not e.get("when"):
                    incoming[e["to"]].append(e["from"])
        for dst, sources in incoming.items():
            node = nodes.get(dst, {})
            if node.get("join") == "all" and len(sources) > 1:
                graph.add_edge([target(x) for x in sources], target(dst))  # wait for all
            else:
                for src in sources:
                    graph.add_edge(target(src), target(dst))
        return graph.compile()

    def execute(self) -> dict:
        """Run to completion (blocking). Returns the final state data."""
        self.started, self.status = time.perf_counter(), "running"
        try:
            graph = self.compile()
            final = graph.invoke(
                {"data": dict(self.inputs), "visited": []}, config={"recursion_limit": 60}
            )
            self.data = final["data"]
            self.status = "cancelled" if self.cancel.is_set() else "done"
        except Cancelled:
            self.status = "cancelled"
        except Exception as exc:
            self.status, self.error = "error", f"{type(exc).__name__}: {exc}"
        finally:
            self.finished = time.perf_counter()
            for entry in self.trace:
                if entry["status"] == "running":
                    entry["status"] = "cancelled"
                    entry["end"] = self.finished - self.started
        return self.data

    def output(self):
        return self.data.get(self.scenario.get("output", "answer"))

    def summary(self) -> dict:
        """JSON-friendly record of the run for the history page."""
        return {
            "scenario": self.scenario["id"],
            "status": self.status,
            "error": self.error,
            "seconds": round(self.elapsed(), 2),
            "inputs": {k: preview(v, 2000) for k, v in self.inputs.items()},
            "output": preview(self.output(), 8000),
            "steps": [{k: v for k, v in entry.items() if k != "stream"} for entry in self.trace],
            "models": sorted({m for e in self.trace for m in e["models"]}),
        }


GATEWAY_NODES = {"rerank", "transcribe", "diarize", "parse_document"}
RETRIEVAL_NODES = {"retrieve_bm25", "retrieve_vector"}


def requirements(scenario: dict) -> dict[str, set]:
    """Models (Ollama / gateway) and knowledge-base collections a scenario needs."""
    need = {"ollama": set(), "gateway": set(), "collections": set()}
    for node in scenario["nodes"]:
        ntype = NODE_TYPES.get(node["type"])
        if not ntype:
            continue
        params = {k: p.default for k, p in ntype.params.items()} | node.get("params", {})
        kind = "gateway" if node["type"] in GATEWAY_NODES else "ollama"
        need[kind].update(m for m in ntype.models(params) if m)
        if node["type"] in RETRIEVAL_NODES:
            need["collections"].add(params.get("collection", "company_kb"))
    return need
