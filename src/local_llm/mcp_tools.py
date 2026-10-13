"""Ready-made agent tools from Model Context Protocol (MCP) servers, all running on this Mac.

MCP is an open protocol for offering tools to a model: a *server* (here: a local child process
talking JSON-RPC over stdin/stdout) lists its tools with a JSON schema and runs them on request.
This module starts a curated set of servers on first use and exposes some of their tools as
ordinary LangChain tools, so agents call them like any other tool:

- ``time``  - official reference server ``mcp-server-time`` (Python, offline: tzdata).
- ``git``   - official ``mcp-server-git``, read-only subset, fixed to one repository.
- ``fetch`` - official ``mcp-server-fetch``: a *network* tool (hidden unless allowed, approval).
- ``files`` - this project's own FastMCP server for ``storage/inbox`` (read-only) and the
  agent sandbox ``storage/lab/workspace`` (``python -m local_llm.mcp_tools serve-files``).

Agents run tools synchronously (LangGraph ``.invoke``/``.stream`` in worker threads), while the
MCP SDK is async. One background thread owns an event loop that keeps a live session per server
(started lazily, restarted if the process dies, closed at exit); tool functions submit calls to
that loop and wait. Tool schemas are declared below (a curated, reviewed subset of what each
server offers), so importing this module never starts a process; ``discover()`` asks a running
server what it really offers and the tests compare the two.
"""

import asyncio
import atexit
import fnmatch
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from importlib import metadata
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
INBOX_DIR = PROJECT_DIR / "storage" / "inbox"
WORKSPACE_DIR = PROJECT_DIR / "storage" / "lab" / "workspace"
LOG_FILE = PROJECT_DIR / "storage" / "lab" / "mcp.log"
# Repository the git tools read; the model cannot pick another one (repo_path is fixed below).
GIT_REPO = Path(os.getenv("LOCAL_LLM_GIT_REPO") or PROJECT_DIR).expanduser().resolve()
LOCAL_TIMEZONE = "Asia/Taipei"
USER_AGENT = "Local_LLM-demo/0.1 (on-prem agent platform demo; MCP fetch)"
MAX_TOOL_CHARS = 12_000  # same budget as the inbox tools: leaves room for the answer
CALL_TIMEOUT = 30  # seconds per tool call
START_TIMEOUT = 20  # seconds for a server to start and finish the MCP handshake
# Soft guard for the offline servers: point every proxy variable at a closed local port, so an
# accidental HTTP request from them fails instead of leaving the machine.
OFFLINE_ENV = {
    key: "http://127.0.0.1:9"
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
}


@dataclass(frozen=True)
class Server:
    """One MCP server: how to start it and what to say about it in the UI."""

    key: str
    package: str  # Python distribution that provides it (its version is shown in the UI)
    module: str  # started as `python -m <module> <args>` from this virtualenv (no download)
    args: tuple[str, ...]
    network: bool
    origin: str  # "official" (modelcontextprotocol/servers) or "project" (this repository)
    source: str
    license: str
    title: dict
    about: dict

    def command(self) -> list[str]:
        return [sys.executable, "-m", self.module, *self.args]

    def environment(self) -> dict[str, str]:
        from mcp.client.stdio import get_default_environment

        env = {**get_default_environment(), "PYTHONIOENCODING": "utf-8", "GIT_TERMINAL_PROMPT": "0"}
        return env if self.network else {**env, **OFFLINE_ENV}

    def version(self) -> str | None:
        try:
            return metadata.version(self.package)
        except metadata.PackageNotFoundError:
            return None


OFFICIAL = "https://github.com/modelcontextprotocol/servers/tree/main/src/"
SERVERS = {
    s.key: s
    for s in [
        Server(
            "time", "mcp-server-time", "mcp_server_time", ("--local-timezone", LOCAL_TIMEZONE),
            network=False, origin="official", source=OFFICIAL + "time", license="MIT",
            title={"zh": "時間與時區", "en": "Time and time zones"},
            about={
                "zh": "官方參考伺服器。查現在時間、換算時區；用 Python 內建 zoneinfo 與 tzdata，完全離線。",
                "en": "Official reference server. Current time and time-zone conversion with "
                "Python's zoneinfo and tzdata; fully offline.",
            },
        ),
        Server(
            "git", "mcp-server-git", "mcp_server_git", ("--repository", str(GIT_REPO)),
            network=False, origin="official", source=OFFICIAL + "git", license="MIT",
            title={"zh": "Git 版本紀錄（唯讀）", "en": "Git history (read-only)"},
            about={
                "zh": "官方參考伺服器。只開放唯讀指令（status、log、show、diff），"
                "commit／add／reset／checkout 等寫入指令不交給模型；repo 路徑由程式固定。",
                "en": "Official reference server. Only its read-only commands (status, log, "
                "show, diff) are given to models; commit / add / reset / checkout are not. "
                "The repository path is fixed in code.",
            },
        ),
        Server(
            "fetch", "mcp-server-fetch", "mcp_server_fetch", ("--user-agent", USER_AGENT),
            network=True, origin="official", source=OFFICIAL + "fetch", license="MIT",
            title={"zh": "網頁擷取（連網）", "en": "Web fetch (network)"},
            about={
                "zh": "官方參考伺服器。下載網頁並轉成 Markdown，遵守 robots.txt。屬於外部工具："
                "預設不交給模型，開放後每次呼叫要人工核准；程式先擋內網與 localhost 網址。",
                "en": "Official reference server. Downloads a page as Markdown and honours "
                "robots.txt. An external tool: hidden unless network tools are allowed, every "
                "call needs approval, and intranet / localhost URLs are refused in code first.",
            },
        ),
        Server(
            "files", "local-llm-agent", "local_llm.mcp_tools",
            # Roots passed explicitly: the child may import another copy of this package.
            ("serve-files", "--inbox", str(INBOX_DIR), "--workspace", str(WORKSPACE_DIR)),
            network=False, origin="project", source="src/local_llm/mcp_tools.py",
            license="internal",
            title={"zh": "檔案沙盒", "en": "File sandbox"},
            about={
                "zh": "本專案用 MCP Python SDK（FastMCP）寫的小型檔案伺服器，取代需要 Node.js 的官方 "
                "filesystem 伺服器：inbox/ 只能讀，workspace/ 是 agent 可寫入草稿的沙盒，其他路徑一律拒絕。",
                "en": "A small file server built for this project with the MCP Python SDK "
                "(FastMCP), replacing the official filesystem server that needs Node.js: "
                "inbox/ is read-only, workspace/ is a sandbox where agents may save drafts, and "
                "every other path is refused.",
            },
        ),
    ]
}  # fmt: skip


@dataclass(frozen=True)
class McpTool:
    """A server tool offered to agents under `name`, with the schema the model sees."""

    name: str
    server: str
    remote: str  # the tool's name on the MCP server
    description: str
    schema: dict
    info: dict  # ToolInfo fields (runs, reads, runs_en, reads_en, network)
    fixed: dict = field(default_factory=dict)  # arguments set by code, hidden from the model


def _schema(properties: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": properties, "required": required or []}


TZ = "IANA time zone name, e.g. 'Asia/Taipei', 'Asia/Tokyo', 'Europe/Berlin', 'America/New_York'."
GIT_RUNS = "MCP 伺服器 mcp-server-git（官方，本機 Python 子行程，stdio）："
GIT_RUNS_EN = "MCP server mcp-server-git (official; local Python child process, stdio): "
GIT_READS = f"git repo {GIT_REPO.name}（唯讀；repo_path 由程式固定，模型改不了）"
GIT_READS_EN = f"git repo {GIT_REPO.name} (read-only; repo_path fixed in code)"
FILES_RUNS = "MCP 伺服器 local-files（本專案 FastMCP，本機子行程）："
FILES_RUNS_EN = "MCP server local-files (this project, FastMCP; local child process): "
FIXED_REPO = {"repo_path": str(GIT_REPO)}

TOOLS_SPEC = [
    McpTool(
        "mcp_time_get_current_time", "time", "get_current_time",
        "Get the current date, time and weekday in a time zone (via the MCP time server). "
        "Use 'Asia/Taipei' for Taiwan when the user names no other place.",
        _schema({"timezone": {"type": "string", "description": TZ}}, ["timezone"]),
        {"runs": "MCP 伺服器 mcp-server-time（官方，本機 Python 子行程，stdio）：查指定時區現在時間",
         "reads": "系統時鐘與 tzdata 時區資料（不連網）",
         "runs_en": "MCP server mcp-server-time (official; local Python child process, stdio): "
         "current time in a time zone",
         "reads_en": "System clock and tzdata (no network)"},
    ),
    McpTool(
        "mcp_time_convert_time", "time", "convert_time",
        "Convert a clock time (24-hour HH:MM) from one time zone to another, e.g. a meeting "
        "time in Taipei for a customer in Munich (via the MCP time server).",
        _schema(
            {
                "source_timezone": {"type": "string", "description": "Source " + TZ},
                "time": {"type": "string", "description": "Time to convert, 24-hour HH:MM"},
                "target_timezone": {"type": "string", "description": "Target " + TZ},
            },
            ["source_timezone", "time", "target_timezone"],
        ),
        {"runs": "MCP 伺服器 mcp-server-time（官方，本機子行程）：時區換算",
         "reads": "tzdata 時區資料（不連網）",
         "runs_en": "MCP server mcp-server-time (official; local child process): time-zone "
         "conversion",
         "reads_en": "tzdata (no network)"},
    ),
    McpTool(
        "mcp_git_status", "git", "git_status",
        "Show the working-tree status of the project's git repository (changed, staged and "
        "untracked files). Read-only.",
        _schema({}),
        {"runs": GIT_RUNS + "git status", "reads": GIT_READS,
         "runs_en": GIT_RUNS_EN + "git status", "reads_en": GIT_READS_EN},
        FIXED_REPO,
    ),
    McpTool(
        "mcp_git_log", "git", "git_log",
        "Show recent commits (hash, author, date, message) of the project's git repository. "
        "Optionally filter by time, e.g. start_timestamp '2 weeks ago' or '2026-10-01'.",
        _schema(
            {
                "max_count": {"type": "integer", "default": 10, "description": "Max commits"},
                "start_timestamp": {
                    "type": "string",
                    "description": "Only commits after this, e.g. '2026-10-01' or 'yesterday'",
                },
                "end_timestamp": {"type": "string", "description": "Only commits before this"},
            }
        ),
        {"runs": GIT_RUNS + "git log", "reads": GIT_READS,
         "runs_en": GIT_RUNS_EN + "git log", "reads_en": GIT_READS_EN},
        FIXED_REPO,
    ),
    McpTool(
        "mcp_git_show", "git", "git_show",
        "Show one commit (message and diff) of the project's git repository, or a file at a "
        "revision given as '<revision>:<path>', e.g. 'HEAD:README.md'.",
        _schema(
            {"revision": {"type": "string", "description": "Commit hash, 'HEAD', or 'rev:path'"}},
            ["revision"],
        ),
        {"runs": GIT_RUNS + "git show", "reads": GIT_READS,
         "runs_en": GIT_RUNS_EN + "git show", "reads_en": GIT_READS_EN},
        FIXED_REPO,
    ),
    McpTool(
        "mcp_git_diff_unstaged", "git", "git_diff_unstaged",
        "Show uncommitted changes in the working directory of the project's git repository.",
        _schema({"context_lines": {"type": "integer", "default": 3}}),
        {"runs": GIT_RUNS + "git diff（未暫存的修改）", "reads": GIT_READS,
         "runs_en": GIT_RUNS_EN + "git diff (unstaged changes)", "reads_en": GIT_READS_EN},
        FIXED_REPO,
    ),
    McpTool(
        "mcp_git_diff", "git", "git_diff",
        "Show the differences between the working tree and a branch or commit "
        "(e.g. 'HEAD~3' or 'main') in the project's git repository.",
        _schema(
            {
                "target": {"type": "string", "description": "Branch or commit to compare with"},
                "context_lines": {"type": "integer", "default": 3},
            },
            ["target"],
        ),
        {"runs": GIT_RUNS + "git diff <target>", "reads": GIT_READS,
         "runs_en": GIT_RUNS_EN + "git diff <target>", "reads_en": GIT_READS_EN},
        FIXED_REPO,
    ),
    McpTool(
        "mcp_fetch", "fetch", "fetch",
        "Download a public web page (external internet request) and return it as Markdown. "
        "Long pages are cut at max_length characters; call again with start_index to continue.",
        _schema(
            {
                "url": {"type": "string", "description": "http(s) URL of a public page"},
                "max_length": {"type": "integer", "default": 5000, "description": "Max chars"},
                "start_index": {
                    "type": "integer",
                    "default": 0,
                    "description": "Start at this character (to read on after a cut)",
                },
            },
            ["url"],
        ),
        {"runs": "MCP 伺服器 mcp-server-fetch（官方，本機子行程）：下載網頁轉 Markdown；"
         "程式先擋內網與 localhost，遵守 robots.txt",
         "reads": "送出：網址（到該網站）",
         "runs_en": "MCP server mcp-server-fetch (official; local child process): downloads a "
         "page as Markdown; intranet / localhost refused in code, robots.txt honoured",
         "reads_en": "Sends: the URL (to that website)",
         "network": True},
    ),
    McpTool(
        "mcp_files_list_directory", "files", "list_directory",
        "List files in the file sandbox. Paths start with 'inbox/' (uploaded files, read-only) "
        "or 'workspace/' (your drafts); an empty path lists both.",
        _schema({"path": {"type": "string", "default": "", "description": "e.g. 'workspace/'"}}),
        {"runs": FILES_RUNS + "列出資料夾", "reads": "storage/inbox（唯讀）、storage/lab/workspace",
         "runs_en": FILES_RUNS_EN + "lists a folder",
         "reads_en": "storage/inbox (read-only), storage/lab/workspace"},
    ),
    McpTool(
        "mcp_files_read_text_file", "files", "read_text_file",
        "Read a text file (.txt, .md, .csv, .json ...) from 'inbox/' or 'workspace/'. For PDF, "
        "Office files or audio use parse_document / transcribe_audio instead.",
        _schema(
            {
                "path": {"type": "string", "description": "e.g. 'workspace/minutes.md'"},
                "head": {"type": "integer", "description": "Only the first N lines (optional)"},
            },
            ["path"],
        ),
        {"runs": FILES_RUNS + "讀取文字檔", "reads": "storage/inbox、storage/lab/workspace（唯讀）",
         "runs_en": FILES_RUNS_EN + "reads a text file",
         "reads_en": "storage/inbox, storage/lab/workspace (read-only)"},
    ),
    McpTool(
        "mcp_files_search_files", "files", "search_files",
        "Find files by name in the sandbox: a glob such as '*.md' or part of a name.",
        _schema(
            {
                "pattern": {"type": "string", "description": "e.g. '*.csv' or 'contract'"},
                "path": {"type": "string", "default": "", "description": "Folder to search"},
            },
            ["pattern"],
        ),
        {"runs": FILES_RUNS + "依檔名搜尋", "reads": "storage/inbox、storage/lab/workspace（只看檔名）",
         "runs_en": FILES_RUNS_EN + "finds files by name",
         "reads_en": "storage/inbox, storage/lab/workspace (names only)"},
    ),
    McpTool(
        "mcp_files_write_file", "files", "write_file",
        "Save a text draft (e.g. meeting minutes or a memo as .md) under 'workspace/'. "
        "Overwrites a file of the same name. 'inbox/' is read-only.",
        _schema(
            {
                "path": {"type": "string", "description": "e.g. 'workspace/2026-10-13-minutes.md'"},
                "content": {"type": "string", "description": "Full text to save"},
            },
            ["path", "content"],
        ),
        {"runs": FILES_RUNS + "寫入文字檔（.md/.txt/.csv/.json，200 KB 內）",
         "reads": "只能寫 storage/lab/workspace（沙盒；其他位置一律拒絕）",
         "runs_en": FILES_RUNS_EN + "saves a text file (.md/.txt/.csv/.json, max 200 KB)",
         "reads_en": "Writes only to storage/lab/workspace (sandbox; anything else is refused)"},
    ),
]  # fmt: skip
TOOL_SPECS = {t.name: t for t in TOOLS_SPEC}


# --- Connection hub ---------------------------------------------------------------------


class _Connection:
    """A live stdio session; created and closed on the hub's event loop."""

    def __init__(self) -> None:
        self.session = None
        self.ready = asyncio.Event()
        self.stop = asyncio.Event()
        self.task: asyncio.Task | None = None
        self.error: str | None = None
        self.started = time.time()


class McpHub:
    """One background event loop holding a live session per MCP server (started on demand)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._conns: dict[str, _Connection] = {}
        self._start_locks: dict[str, asyncio.Lock] = {}
        self._errlog = None
        self.calls: dict[str, int] = {}
        self.errors: dict[str, str] = {}

    # -- thread side ------------------------------------------------------------------
    def _event_loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is None:
                loop = asyncio.new_event_loop()
                threading.Thread(target=loop.run_forever, name="mcp-hub", daemon=True).start()
                self._loop = loop
                atexit.register(self.close)
            return self._loop

    def run(self, coro, timeout: float):
        """Run a coroutine on the hub loop from any (non-hub) thread and wait for it."""
        future = asyncio.run_coroutine_threadsafe(coro, self._event_loop())
        try:
            return future.result(timeout)
        except TimeoutError:
            future.cancel()
            raise

    def call(self, server: str, tool: str, args: dict, timeout: float = CALL_TIMEOUT):
        self.calls[server] = self.calls.get(server, 0) + 1
        return self.run(self._call(server, tool, args, timeout), timeout + START_TIMEOUT + 5)

    def list_tools(self, server: str) -> list:
        return self.run(self._list_tools(server), START_TIMEOUT + 10)

    def running(self, server: str) -> bool:
        conn = self._conns.get(server)
        return bool(conn and conn.session and conn.task and not conn.task.done())

    def stop(self, server: str) -> None:
        if self._loop and server in self._conns:
            self.run(self._stop([server]), 10)

    def close(self) -> None:
        """Stop every server process (registered with atexit)."""
        if self._loop and self._loop.is_running() and self._conns:
            try:
                self.run(self._stop(list(self._conns)), 10)
            except Exception:  # noqa: BLE001 - shutting down anyway
                pass

    # -- loop side --------------------------------------------------------------------
    def _log(self):
        if self._errlog is None:
            LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            self._errlog = open(LOG_FILE, "a", encoding="utf-8")  # noqa: SIM115 - kept open
        return self._errlog

    async def _serve(self, key: str, conn: _Connection) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        server = SERVERS[key]
        command = server.command()
        params = StdioServerParameters(
            command=command[0], args=command[1:], env=server.environment(), cwd=str(PROJECT_DIR)
        )
        try:
            async with stdio_client(params, errlog=self._log()) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    conn.session = session
                    conn.ready.set()
                    await conn.stop.wait()
        except Exception as exc:  # noqa: BLE001 - reported to the caller and the UI
            conn.error = f"{type(exc).__name__}: {exc}"
            self.errors[key] = conn.error
        finally:
            conn.session = None
            conn.ready.set()

    async def _session(self, key: str):
        lock = self._start_locks.setdefault(key, asyncio.Lock())
        async with lock:
            conn = self._conns.get(key)
            if conn and conn.session and conn.task and not conn.task.done():
                return conn.session
            conn = _Connection()
            self._conns[key] = conn
            conn.task = asyncio.create_task(self._serve(key, conn))
            try:
                await asyncio.wait_for(conn.ready.wait(), START_TIMEOUT)
            except TimeoutError:
                conn.stop.set()
                message = f"MCP server '{key}' did not start in {START_TIMEOUT}s"
                raise RuntimeError(message) from None
            if conn.session is None:
                self._conns.pop(key, None)
                raise RuntimeError(f"MCP server '{key}' could not start: {conn.error}")
            self.errors.pop(key, None)
            return conn.session

    async def _call(self, key: str, tool: str, args: dict, timeout: float):
        from mcp.shared.exceptions import McpError

        for attempt in (1, 2):
            session = await self._session(key)
            try:
                return await session.call_tool(
                    tool, args, read_timeout_seconds=timedelta(seconds=timeout)
                )
            except McpError as exc:
                # A dead process shows up as "Connection closed": restart once and retry.
                # Anything else (timeout, bad arguments) also drops the session, then reports.
                await self._stop([key])
                if attempt == 2 or "closed" not in str(exc).lower():
                    raise
            except Exception:  # noqa: BLE001 - broken pipe / closed stream: restart once
                await self._stop([key])
                if attempt == 2:
                    raise

    async def _list_tools(self, key: str) -> list:
        session = await self._session(key)
        return (await session.list_tools()).tools

    async def _stop(self, keys: list[str]) -> None:
        conns = [c for k in keys if (c := self._conns.pop(k, None))]
        for conn in conns:
            conn.stop.set()
        tasks = [c.task for c in conns if c.task]
        if tasks:
            await asyncio.wait(tasks, timeout=8)


HUB = McpHub()


# --- LangChain tools ----------------------------------------------------------------------


def _git_problem() -> str | None:
    if not (GIT_REPO / ".git").exists():
        return (
            f"The git tools are not available: {GIT_REPO} is not a git repository. Tell the "
            "user an administrator must run `git init` there (or set LOCAL_LLM_GIT_REPO)."
        )
    return None


def _guard(spec: McpTool, args: dict) -> str | None:
    """Checks done in code before a call reaches the server (the model cannot skip them)."""
    if spec.server == "git":
        return _git_problem()
    if spec.server == "fetch":
        from .web_tools import check_public_url  # lazy: trafilatura is a heavy import

        return check_public_url(str(args.get("url", "")))
    return None


def result_text(result) -> str:
    """Text of an MCP CallToolResult, cut to MAX_TOOL_CHARS."""
    parts = []
    for block in result.content:
        if getattr(block, "type", "") == "text":
            parts.append(block.text)
        elif getattr(block, "type", "") == "resource" and hasattr(block.resource, "text"):
            parts.append(block.resource.text)
        else:
            parts.append(f"[{getattr(block, 'type', 'binary')} content omitted]")
    text = "\n".join(parts)
    if not text and result.structuredContent:
        text = json.dumps(result.structuredContent, ensure_ascii=False)
    if result.isError:
        text = "Error from the MCP server: " + text
    if len(text) > MAX_TOOL_CHARS:
        text = text[:MAX_TOOL_CHARS] + f"\n[cut at {MAX_TOOL_CHARS} characters]"
    return text


def call(name: str, args: dict | None = None) -> str:
    """Run one exposed MCP tool synchronously (what the agent's tool function does)."""
    spec = TOOL_SPECS[name]
    required = set(spec.schema.get("required", []))
    args = {
        k: v
        for k, v in (args or {}).items()
        if k in spec.schema["properties"] and v is not None and (v != "" or k in required)
    }
    if problem := _guard(spec, args):
        return problem
    try:
        return result_text(HUB.call(spec.server, spec.remote, {**args, **spec.fixed}))
    except Exception as exc:  # noqa: BLE001 - the model gets a readable error instead of a crash
        return f"MCP tool {name} failed: {type(exc).__name__}: {exc}".strip()


def _make_tool(spec: McpTool):
    from langchain_core.tools import StructuredTool

    def run(**kwargs) -> str:
        return call(spec.name, kwargs)

    async def arun(**kwargs) -> str:
        return await asyncio.to_thread(call, spec.name, kwargs)

    return StructuredTool(
        name=spec.name,
        description=spec.description,
        args_schema=spec.schema,
        func=run,
        coroutine=arun,
        metadata={"mcp_server": spec.server, "mcp_tool": spec.remote},
    )


def registrations() -> list[tuple]:
    """(LangChain tool, ToolInfo fields) for agents.register_tool."""
    return [(_make_tool(spec), spec.info) for spec in TOOLS_SPEC]


def discover(server: str) -> list[dict]:
    """Ask the running server which tools it offers (starts it if needed)."""
    if server == "git" and (problem := _git_problem()):
        raise RuntimeError(problem)
    exposed = {t.remote for t in TOOLS_SPEC if t.server == server}
    return [
        {
            "name": t.name,
            "description": t.description or "",
            "schema": t.inputSchema,
            "read_only": bool(t.annotations and t.annotations.readOnlyHint),
            "open_world": bool(t.annotations and t.annotations.openWorldHint),
            "exposed": t.name in exposed,
        }
        for t in HUB.list_tools(server)
    ]


def status() -> dict[str, dict]:
    """Per server: installed version, running, calls made in this process, last error."""
    result = {}
    for key, server in SERVERS.items():
        problem = _git_problem() if key == "git" else None
        result[key] = {
            "version": server.version(),
            "running": HUB.running(key),
            "calls": HUB.calls.get(key, 0),
            "error": problem or HUB.errors.get(key),
            "tools": [t.name for t in TOOLS_SPEC if t.server == key],
        }
    return result


# --- The project's own MCP server: a file sandbox -----------------------------------------

ROOTS = {"inbox": (INBOX_DIR, False), "workspace": (WORKSPACE_DIR, True)}  # name -> (dir, write)
WRITE_TYPES = {".md", ".txt", ".csv", ".json", ".yaml", ".yml", ".html"}
BINARY_TYPES = {".pdf", ".docx", ".pptx", ".xlsx", ".png", ".jpg", ".jpeg", ".wav", ".mp3",
                ".m4a", ".mp4", ".mov", ".zip"}  # fmt: skip
MAX_READ_BYTES = 2_000_000
MAX_WRITE_CHARS = 200_000


def sandbox_path(path: str, write: bool = False) -> Path:
    """Resolve 'inbox/...' or 'workspace/...' to a real path inside that root, or raise."""
    parts = [p for p in path.replace("\\", "/").strip().split("/") if p not in ("", ".")]
    if not parts or parts[0] not in ROOTS:
        raise ValueError("paths must start with 'inbox/' or 'workspace/'")
    if any(p.startswith(".") for p in parts[1:]):
        raise ValueError("hidden files and '..' are not allowed")
    root, writable = ROOTS[parts[0]]
    if write and not writable:
        raise PermissionError(f"'{parts[0]}/' is read-only; save drafts under 'workspace/'")
    root.mkdir(parents=True, exist_ok=True)
    target = root.joinpath(*parts[1:]).resolve()
    if not target.is_relative_to(root.resolve()):
        raise PermissionError("path leaves the sandbox")
    return target


def _virtual(path: Path) -> str:
    for name, (root, _) in ROOTS.items():
        if path.is_relative_to(root.resolve()):
            rel = path.relative_to(root.resolve()).as_posix()
            return f"{name}/{rel}" if rel != "." else f"{name}/"
    return path.name


def _visible(paths) -> list[Path]:
    return sorted(p for p in paths if not p.name.startswith(".") and not p.is_symlink())


def serve_files(inbox: Path = INBOX_DIR, workspace: Path = WORKSPACE_DIR) -> None:
    """Run the sandbox file server on stdio (python -m local_llm.mcp_tools serve-files)."""
    ROOTS.update({"inbox": (inbox, False), "workspace": (workspace, True)})
    from mcp.server.fastmcp import FastMCP
    from mcp.types import ToolAnnotations

    server = FastMCP(
        "local-files",
        instructions="Files of the Local Agent Lab: inbox/ (uploads, read-only) and "
        "workspace/ (agent drafts). Paths always start with one of these two folders.",
        log_level="WARNING",
    )
    read_only = ToolAnnotations(readOnlyHint=True, openWorldHint=False)

    @server.tool(annotations=read_only)
    def list_allowed_directories() -> str:
        """List the folders this server may access."""
        return "\n".join(
            f"{name}/ ({'read-write' if writable else 'read-only'})"
            for name, (_, writable) in ROOTS.items()
        )

    @server.tool(annotations=read_only)
    def list_directory(path: str = "") -> str:
        """List a folder: [DIR] and [FILE] entries with sizes."""
        if not path.strip("/ ."):
            return list_allowed_directories()
        folder = sandbox_path(path)
        if not folder.is_dir():
            return f"Not a folder: {path}"
        lines = []
        for p in _visible(folder.iterdir()):
            if p.is_dir():
                lines.append(f"[DIR] {_virtual(p)}/")
            else:
                stat = p.stat()
                modified = datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="minutes")
                lines.append(f"[FILE] {_virtual(p)} ({stat.st_size:,} bytes, {modified})")
        return "\n".join(lines) or f"{path} is empty"

    @server.tool(annotations=read_only)
    def read_text_file(path: str, head: int | None = None) -> str:
        """Read a UTF-8 text file; `head` limits the result to the first N lines."""
        target = sandbox_path(path)
        if not target.is_file():
            return f"No such file: {path} (use list_directory)"
        if target.suffix.lower() in BINARY_TYPES:
            return f"{path} is not a text file; use parse_document or transcribe_audio for it."
        if target.stat().st_size > MAX_READ_BYTES:
            return f"{path} is larger than {MAX_READ_BYTES:,} bytes; ask for a smaller file."
        try:
            text = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return f"{path} is not UTF-8 text."
        if head:
            text = "\n".join(text.splitlines()[: max(head, 1)])
        return text

    @server.tool(annotations=read_only)
    def search_files(pattern: str, path: str = "") -> str:
        """Find files whose name matches a glob ('*.md') or contains the text."""
        if path.strip("/ ."):
            folders = [sandbox_path(path)]
        else:
            folders = [sandbox_path(f"{name}/") for name in ROOTS]
        needle = pattern.strip().lower()
        hits = []
        for folder in folders:
            for p in _visible(folder.rglob("*")):
                name = p.name.lower()
                if p.is_file() and (fnmatch.fnmatch(name, needle) or needle in name):
                    hits.append(_virtual(p))
        return "\n".join(hits[:100]) or f"No file matches {pattern!r}"

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True,
                                             openWorldHint=False))  # fmt: skip
    def write_file(path: str, content: str) -> str:
        """Save a text file under workspace/ (overwrites)."""
        target = sandbox_path(path, write=True)
        if target.suffix.lower() not in WRITE_TYPES:
            return f"Only {', '.join(sorted(WRITE_TYPES))} files can be written."
        if len(content) > MAX_WRITE_CHARS:
            return f"Content is longer than {MAX_WRITE_CHARS:,} characters."
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Saved {len(content):,} characters to {_virtual(target)}"

    server.run("stdio")


if __name__ == "__main__":
    command, options = sys.argv[1:2], dict(zip(sys.argv[2::2], sys.argv[3::2], strict=False))
    if command != ["serve-files"] or set(options) - {"--inbox", "--workspace"}:
        print(
            "usage: python -m local_llm.mcp_tools serve-files [--inbox DIR] [--workspace DIR]",
            file=sys.stderr,
        )
        sys.exit(2)
    serve_files(
        Path(options.get("--inbox", INBOX_DIR)), Path(options.get("--workspace", WORKSPACE_DIR))
    )
