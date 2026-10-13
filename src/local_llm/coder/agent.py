"""The local coding agent: configuration tools + code tools, every write approved by a human.

`CoderSession` is shared by the terminal REPL (`local-agent code`) and the Streamlit control
center. It keeps the conversation, streams the model's progress, pauses on every write with a
preview (unified diff + validation) and records the change log used for undo.
"""

import json
import threading
import time
import uuid
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Literal

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ClearToolUsesEdit,
    ContextEditingMiddleware,
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
)
from langchain.tools import tool
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from .. import ollama_admin
from ..config import Settings
from ..lab import control
from ..providers import build_model
from .workspace import CHECKS, PROJECT_ROOT, EditError, PathError, Workspace

# Local coding models, best first (measured on this Mac, see docs/LOCAL_CODING_AGENT.md).
# The first installed one is the default; general models with tool calling come last.
CODING_MODELS = (
    "qwen3.6:35b-a3b-coding",
    "qwen3.8:27b",
    "qwen3-coder-next:latest",
    "qwen3-coder:30b",
    "north-mini-code-1.0:latest",
    "devstral-small-2:24b",
    "qwen3:30b-a3b",
)
RECOMMENDED_MODEL = CODING_MODELS[0]
MODES = ("settings", "code")  # settings only / settings + code
CODE_WRITE_TOOLS = ("edit_file", "write_file")
DEFAULT_CONTEXT = 65536  # coding agents need room for files and tool results
MAX_MODEL_CALLS = 80  # per user turn (approvals do not reset it)


def best_coding_model(installed: Iterable[str]) -> str | None:
    """The preferred coding model among the installed Ollama models."""
    names = {n if ":" in n else f"{n}:latest" for n in installed}
    return next((m for m in CODING_MODELS if m in names), None)


def thinking_options(model: str, base_url: str = ollama_admin.DEFAULT_URL) -> tuple[str, bool]:
    """('none' | 'optional' | 'always', default) for a model's thinking.

    Newer Ollama models (e.g. the MLX builds of Qwen3.6/3.8) declare their thinking values in
    /api/show instead of in the chat template, so this checks that first.
    """
    try:
        info = ollama_admin.show(model, base_url)
    except Exception:
        return "none", False
    if "thinking" not in info.get("capabilities", []):
        return "none", False
    declared = info.get("thinking") or {}
    if "values" in declared:
        default = declared.get("default") not in (False, None)
        return ("optional" if False in declared["values"] else "always"), default
    mode = ollama_admin.thinking_mode(model, base_url)
    return mode, mode == "always"


def ordered_models(installed: Iterable[str]) -> list[str]:
    """Installed models with the coding models first, in preference order."""
    names = sorted(installed)
    first = [m for m in CODING_MODELS if m in names]
    return first + [n for n in names if n not in first]


# --- code tools ----------------------------------------------------------------------------


def code_tools(ws: Workspace) -> tuple[list, list]:
    """(read tools, write tools) over the project files of `ws`."""

    def safe(fn, *args) -> str:
        try:
            return fn(*args)
        except (PathError, EditError, ValueError, OSError) as exc:
            return f"ERROR: {exc}"

    @tool
    def list_files(glob: str = "**/*") -> str:
        """List project files matching a glob, e.g. "src/**/*.py", "*.md", "tests/*",
        "app/views". Secrets, model weights, the database and .venv are never shown."""
        return safe(ws.list_files, glob)

    @tool
    def read_file(path: str, start: int = 1, end: int | None = None) -> str:
        """Read a text file with line numbers ("   12<TAB>code"), at most 400 lines per call.
        The numbers are not part of the file: leave them out of edit_file's `old`."""
        return safe(ws.read_file, path, start, end)

    @tool
    def search(pattern: str, glob: str | None = None) -> str:
        """Regex search in project files (like ripgrep); returns "path:line: text". `glob`
        narrows the files, e.g. "src/**/*.py" or "*.json"."""
        return safe(ws.search, pattern, glob)

    @tool
    def edit_file(path: str, old: str, new: str) -> str:
        """Replace one exact occurrence of `old` with `new` in an existing file. `old` must
        match exactly one place: copy it from read_file without the line numbers and include
        2-3 surrounding lines. Needs the user's approval; a backup is kept for undo."""

        def run():
            entry = ws.apply(ws.plan_edit(path, old, new))
            return f"Edited {entry['path']} (+{entry['added']} -{entry['removed']} lines)."

        return safe(run)

    @tool
    def write_file(path: str, content: str) -> str:
        """Create a NEW file with this content (refused if it exists; use edit_file then).
        Needs the user's approval; a backup entry is kept for undo."""

        def run():
            entry = ws.apply(ws.plan_write(path, content))
            return f"Created {entry['path']} ({entry['added']} lines)."

        return safe(run)

    @tool
    def run_check(
        name: Literal["pytest", "ruff", "ruff_format", "compileall", "config"],
        path: str | None = None,
    ) -> str:
        """Run a project check: "pytest" (pytest -q), "ruff" (ruff check), "ruff_format"
        (ruff format --check), "compileall", "config" (validate agents.json + scenarios.json
        with the current code). `path` limits it to one file or folder, e.g.
        "tests/test_tools.py". No other commands can be run."""
        return safe(ws.run_check, name, path)

    return [list_files, read_file, search, run_check], [edit_file, write_file]


# --- prompts -------------------------------------------------------------------------------

CODE_PROMPT = """You are the local coding agent of Local Agent Lab, running fully on this
machine (no cloud). You change two things when the user asks:
(1) agent and scenario configuration, with the configuration tools, and
(2) the project's code, with the file tools.

Project map (Python 3.12, LangChain 1.x / LangGraph, Streamlit):
- src/local_llm/agents.py: the agent tool registry. TOOLS maps tool names to LangChain tools,
  TOOL_INFO describes each tool in Chinese and English with ToolInfo(runs, reads, network,
  runs_en, reads_en), and register_tool(tool, info) registers tools defined in other modules
  (media_tools.py is an example of that pattern). Tools are functions decorated with @tool from
  langchain.tools (see tools.py: calculator), with a docstring the model reads.
- src/local_llm/lab/: multi-agent scenario engine (nodes.py node types, store.py validation and
  version history, control.py configuration tools). src/local_llm/coder/: this agent.
- app/: Streamlit pages (app/views/*.py); visible text uses tr("中文", "English").
- data/lab/agents.json and data/lab/scenarios.json: agent and scenario definitions. Change them
  ONLY with the configuration tools (update_agent, update_node, ...), never with edit_file.
- tests/: pytest tests that need no models or network.
- Off limits: .env, models/, storage/, .venv.

How to work:
1. Explore first: search, list_files and read_file to find the code involved and a similar
   existing example; follow its pattern. Read a file before editing it.
2. Edit with edit_file: copy `old` exactly from read_file (without the line-number prefix) so it
   matches exactly one place. Create new files with write_file. Keep changes small and in the
   project's style (type hints, docstrings, line length 100).
3. Every write waits for the user's approval. If one is rejected, read the reason and adapt.
4. Verify: run_check("ruff") and run_check("pytest"); after changing tools or agents also
   run_check("config"). If a check fails, read the error, fix the cause, run it again. Do not
   weaken or delete tests to make them pass.
5. Finish with a short summary: files changed, configuration changed, checks and results. If
   Python code under src/ changed, tell the user to restart the app (./scripts/restart_app.sh).
   Answer in the user's language; for Chinese use Traditional Chinese.

Adding an agent tool: write the @tool function in a module under src/local_llm/, register it
with its ToolInfo (zh and en) in agents.py, add a test under tests/, run the checks, then add
the tool name to the agent with update_agent (pass the agent's full tools list).

"""


def system_prompt(mode: str, root: Path) -> str:
    if mode != "code":
        return control.SYSTEM_PROMPT
    prompt = CODE_PROMPT + control.SCENARIO_GUIDE
    notes = root / "AGENTS.md"  # project instructions, the convention other agents read too
    if notes.is_file():
        prompt += "\n\nProject instructions (AGENTS.md):\n" + notes.read_text()[:6000]
    return prompt


# --- the session ---------------------------------------------------------------------------


def run_stats(messages: list[BaseMessage]) -> dict:
    """LLM calls, tool calls, tokens and decode speed of a run (from Ollama's metadata)."""
    ai = [m for m in messages if isinstance(m, AIMessage)]
    meta = [m.response_metadata or {} for m in ai]
    out_tokens = sum(x.get("eval_count") or 0 for x in meta)
    out_ns = sum(x.get("eval_duration") or 0 for x in meta)
    prompts = [x.get("prompt_eval_count") or 0 for x in meta]
    return {
        "llm_calls": len(ai),
        "tool_calls": sum(len(m.tool_calls) for m in ai),
        "input_tokens": sum(prompts),
        "max_prompt_tokens": max(prompts, default=0),
        "output_tokens": out_tokens,
        "tokens_per_s": round(out_tokens / (out_ns / 1e9), 1) if out_ns else None,
        "model_seconds": round(sum(x.get("total_duration") or 0 for x in meta) / 1e9, 1),
    }


def _answered(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Drop tool calls that never got a result (a run stopped while waiting for approval)."""
    done = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    return [
        m
        for m in messages
        if not (isinstance(m, AIMessage) and any(c["id"] not in done for c in m.tool_calls))
    ]


class CoderSession:
    """One conversation with the local coding agent over one project folder."""

    def __init__(
        self,
        model: str,
        base_url: str = ollama_admin.DEFAULT_URL,
        mode: str = "code",
        root: Path | str = PROJECT_ROOT,
        context: int = DEFAULT_CONTEXT,
        think: bool | None = None,
        temperature: float | None = None,
    ):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.ws = Workspace(root)
        self.base_url, self.context, self.think, self.temperature = (
            base_url,
            context,
            think,
            temperature,
        )
        self.history: list[BaseMessage] = []
        self.started = time.time()
        self.undone_config: set[str] = set()
        self._previews: dict[str, dict] = {}
        self._lock = threading.Lock()
        self.model, self.mode = model, mode
        self.graph = self._build()
        self.config: dict | None = None

    # -- building ------------------------------------------------------------------------------

    def fresh(self) -> bool:
        """Validate configuration in a new process (another copy, or code changed)."""
        return self.ws.code_changed_since()

    def write_tools(self) -> tuple[str, ...]:
        return control.WRITE_TOOLS + (CODE_WRITE_TOOLS if self.mode == "code" else ())

    def _build(self):
        read, write = control.config_tools(self.ws.root, fresh=self.fresh)
        tools = read + write
        if self.mode == "code":
            code_read, code_write = code_tools(self.ws)
            tools = code_read + tools + code_write
        settings = Settings(
            "ollama", self.model, self.base_url, max_tokens=8192, context=self.context, timeout=600
        )
        approve = {
            name: {
                "allowed_decisions": ["approve", "reject"],
                # Edits that cannot apply (text not found, invalid config) skip the human and
                # return their error to the model right away; nothing is written.
                "when": lambda req: self.preview(req.tool_call["name"], req.tool_call["args"])[
                    "ok"
                ],
            }
            for name in self.write_tools()
        }
        middleware = [
            HumanInTheLoopMiddleware(approve, description_prefix="Change needs approval"),
            ModelCallLimitMiddleware(thread_limit=MAX_MODEL_CALLS, exit_behavior="end"),
            ContextEditingMiddleware(
                edits=[ClearToolUsesEdit(trigger=int(self.context * 0.6), keep=6)]
            ),
        ]
        llm = build_model(settings, think=self.think, temperature=self.temperature)
        mode, default = thinking_options(self.model, self.base_url)
        if mode == "optional":  # None = the model's own default
            llm.reasoning = default if self.think is None else self.think
        elif mode == "always":
            llm.reasoning = True
        return create_agent(
            model=llm,
            tools=tools,
            system_prompt=system_prompt(self.mode, self.ws.root),
            middleware=middleware,
            checkpointer=InMemorySaver(),
        )

    def rebuild(self, model: str | None = None, mode: str | None = None) -> None:
        """Switch model or mode; the conversation so far is kept."""
        if mode and mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.model, self.mode = model or self.model, mode or self.mode
        self.graph = self._build()

    # -- previews ------------------------------------------------------------------------------

    def preview(self, name: str, args: dict) -> dict:
        """What a write would change: {"kind", "id", "diff", "errors", "warnings", "ok"}."""
        key = name + json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
        with self._lock:
            if key in self._previews:
                return self._previews[key]
        if name in CODE_WRITE_TOOLS:
            try:
                if name == "edit_file":
                    change = self.ws.plan_edit(args["path"], args["old"], args["new"])
                else:
                    change = self.ws.plan_write(args["path"], args["content"])
                added, removed = change.counts()
                result = {
                    "kind": "code",
                    "id": change.path,
                    "action": change.action,
                    "diff": change.diff(),
                    "errors": [],
                    "warnings": [],
                    "ok": True,
                    "added": added,
                    "removed": removed,
                }
            except (PathError, EditError, KeyError, TypeError, OSError) as exc:
                error = f"{type(exc).__name__}: {exc}"
                path = args.get("path") if isinstance(args, dict) else None
                result = {"kind": "code", "id": path, "diff": "", "errors": [error]}
                result |= {"warnings": [], "ok": False}
        else:
            result = control.preview(name, args, self.ws.root, self.fresh())
        with self._lock:
            self._previews[key] = result
        return result

    # -- running -------------------------------------------------------------------------------

    def start(self, prompt: str) -> dict:
        """Payload for a new user turn (the whole conversation, on a new thread)."""
        self.config = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 250}
        self._previews.clear()
        self.history = _answered(self.history)
        return {"messages": [*self.history, HumanMessage(prompt, id=str(uuid.uuid4()))]}

    def run(
        self,
        payload,
        on_event: Callable[[str, object], None] | None = None,
        cancel: threading.Event | None = None,
    ) -> dict | None:
        """Stream until the agent answers (returns None) or pauses for approval (returns the
        interrupt: {"action_requests": [...], "previews": [...]})."""
        emit = on_event or (lambda kind, data: None)
        seen = {m.id for m in self.history if m.id}
        pending = None
        stream = self.graph.stream(payload, self.config, stream_mode=["messages", "updates"])
        for mode, data in stream:
            if cancel is not None and cancel.is_set():
                break
            if mode == "messages":
                chunk = data[0]
                if isinstance(chunk, AIMessageChunk):
                    if reasoning := chunk.additional_kwargs.get("reasoning_content"):
                        emit("reasoning", reasoning)
                    if isinstance(chunk.content, str) and chunk.content:
                        emit("text", chunk.content)
                continue
            for node, value in data.items():
                if node == "__interrupt__":
                    pending = dict(value[0].value)
                    continue
                for message in (value or {}).get("messages", []) if isinstance(value, dict) else []:
                    key = message.id or f"obj-{id(message)}"
                    if key in seen:
                        continue
                    seen.add(key)
                    emit("message", message)
        state = self.graph.get_state(self.config).values
        self.history = list(state.get("messages", self.history))
        if pending is None or (cancel is not None and cancel.is_set()):
            return None
        pending["previews"] = [
            self.preview(a["name"], a["args"]) for a in pending["action_requests"]
        ]
        return pending

    def send(
        self,
        prompt: str,
        decide: Callable[[list[dict], list[dict]], list[dict]],
        on_event: Callable[[str, object], None] | None = None,
    ) -> list[BaseMessage]:
        """Run one user turn to the end, asking `decide(requests, previews)` at every pause.

        Returns the messages added in this turn.
        """
        before = len(self.history)
        payload = self.start(prompt)
        while (pending := self.run(payload, on_event)) is not None:
            decisions = decide(pending["action_requests"], pending["previews"])
            payload = Command(resume={"decisions": decisions})
        return self.history[before:]

    # -- change log and undo ------------------------------------------------------------------

    def changes(self, limit: int = 30) -> list[dict]:
        """Code and configuration changes, newest first: {"type", "id", "label", "at", ...}."""
        rows = [
            {"type": "code", "id": e["id"], "label": f"{e['action']} {e['path']}", **e}
            for e in self.ws.changes(limit)
        ]
        for e in control.history(self.ws.root, limit):
            label = f"{e['kind']} {e['id']} ({e['source']})"
            rows.append({**e, "type": "config", "id": e["file"], "label": label})
        return sorted(rows, key=lambda r: r["at"], reverse=True)[:limit]

    def undoable(self) -> list[dict]:
        """Changes made since this session started that can still be undone, newest first."""
        since = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.started))
        return [
            r
            for r in self.changes(100)
            if r["at"] >= since
            and not r.get("undone_at")
            and r["id"] not in self.undone_config
            and (r["type"] == "code" or r.get("source") == "control-center")
        ]

    def undo(self, change: dict) -> str:
        if change["type"] == "code":
            meta = self.ws.undo(change["id"])
            return f"restored {meta['path']} ({meta['action']} undone)"
        result = control.rollback(change["id"], self.ws.root, self.fresh())
        if not result["ok"]:
            raise EditError("; ".join(result["errors"]))
        self.undone_config.add(change["id"])
        return f"restored the previous version of {change['label']}"


CHECK_NAMES = tuple(CHECKS)
