"""Agent types offered by the platform, defined in data/lab/agents.json.

Each definition fixes the agent's system prompt and tools (by name). The control-center coding
agent edits that file; the registry below re-reads it whenever it changes.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, ToolCallLimitMiddleware
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from . import mcp_tools, media_tools, skills
from .company_db import make_tools
from .config import Settings
from .providers import build_model
from .tools import calculator, search_models
from .web_tools import fetch_webpage, get_weather, wikipedia_search


@tool
def current_time() -> str:
    """Return the current local date and time."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass(frozen=True)
class ToolInfo:
    """What a tool really does when the agent calls it (the model only sees name + schema)."""

    runs: str  # how it executes
    reads: str  # what it touches
    network: bool = False
    runs_en: str = ""
    reads_en: str = ""

    def text(self, lang: str) -> tuple[str, str]:
        """(runs, reads) in the UI language."""
        if lang == "en" and self.runs_en:
            return self.runs_en, self.reads_en
        return self.runs, self.reads


TOOL_INFO = {
    "calculator": ToolInfo(
        "本機 Python 函式：用 AST 解析並計算四則運算",
        "無（不讀檔）",
        runs_en="Local Python: parses and evaluates arithmetic with the AST",
        reads_en="Nothing (no file access)",
    ),
    "current_time": ToolInfo(
        "本機 Python 函式：讀取系統時鐘",
        "系統時間",
        runs_en="Local Python: reads the system clock",
        reads_en="System time",
    ),
    "search_models": ToolInfo(
        "本機 Python 函式：關鍵字比對模型清單",
        "data/models.json（唯讀）",
        runs_en="Local Python: keyword search over the model list",
        reads_en="data/models.json (read-only)",
    ),
    "db_list_tables": ToolInfo(
        "本機 PostgreSQL：列出這個角色可讀的資料表",
        "democo 資料庫（依部門角色，唯讀）",
        runs_en="Local PostgreSQL: lists the tables this role may read",
        reads_en="democo database (per-department role, read-only)",
    ),
    "db_describe_table": ToolInfo(
        "本機 PostgreSQL：列出資料表欄位與中文說明",
        "democo 資料庫（只顯示有權限的欄位）",
        runs_en="Local PostgreSQL: lists a table's columns with descriptions",
        reads_en="democo database (only permitted columns)",
    ),
    "db_query": ToolInfo(
        "本機 PostgreSQL：執行 SELECT，最多 100 列、5 秒逾時",
        "democo 資料庫（權限由資料庫角色強制，無寫入權限）",
        runs_en="Local PostgreSQL: runs a SELECT, max 100 rows, 5 s timeout",
        reads_en="democo database (permissions enforced by the DB role, no writes)",
    ),
    "get_weather": ToolInfo(
        "外部 API：Open-Meteo 地名查詢＋天氣預報",
        "送出：城市名稱",
        network=True,
        runs_en="External API: Open-Meteo geocoding + forecast",
        reads_en="Sends: city name",
    ),
    "wikipedia_search": ToolInfo(
        "外部 API：中文維基百科搜尋＋摘要",
        "送出：搜尋關鍵字",
        network=True,
        runs_en="External API: Chinese Wikipedia search + summary",
        reads_en="Sends: search keywords",
    ),
    "fetch_webpage": ToolInfo(
        "外部請求：下載網頁並擷取正文（擋掉內網與 localhost）",
        "送出：網址",
        network=True,
        runs_en="External request: downloads a page and extracts its text (no intranet hosts)",
        reads_en="Sends: the URL",
    ),
}
TOOL_INFO["load_skill"] = ToolInfo(
    "本機：讀取 agent 技能（data/lab/skills/*/SKILL.md）的完整說明",
    "data/lab/skills（唯讀）",
    runs_en="Local: reads the full instructions of one of the agent's skills (SKILL.md)",
    reads_en="data/lab/skills (read-only)",
)
# Default per-run cap on each external tool, enforced in code (the model cannot exceed it).
EXTERNAL_CALL_LIMIT = 3

# Tools an agent definition can list by name. Database tools are created per department role
# (`db_role` in the agent definition), so the same names exist once per role.
TOOLS = {t.name: t for t in [calculator, current_time, search_models, get_weather,
                              wikipedia_search, fetch_webpage]}  # fmt: skip
DB_TOOLS = ("db_list_tables", "db_describe_table", "db_query")
DB_ROLES = ("agent_sales", "agent_hr", "agent_support")


def register_tool(tool_obj, info: ToolInfo) -> None:
    """Let other modules (e.g. the model gateway tools) offer tools to agent definitions."""
    TOOLS[tool_obj.name] = tool_obj
    TOOL_INFO[tool_obj.name] = info


# Inbox tools backed by the local model gateway (speech-to-text, document parsing).
for _tool in media_tools.TOOLS:
    register_tool(_tool, ToolInfo(**media_tools.TOOL_TEXT[_tool.name]))
# Ready-made tools from local MCP servers (time, read-only git, fetch, file sandbox); the
# server processes start on first use. See mcp_tools.py.
for _tool, _info in mcp_tools.registrations():
    register_tool(_tool, ToolInfo(**_info))


def is_external(tool_name: str) -> bool:
    return TOOL_INFO[tool_name].network


AGENTS_FILE = Path(__file__).resolve().parents[2] / "data" / "lab" / "agents.json"


@dataclass(frozen=True)
class AgentSpec:
    key: str
    name: str
    description: str
    system_prompt: str
    tools: list = field(default_factory=list)
    suggested_task: str = "chat"  # catalog task whose models suit this agent
    examples: list[str] = field(default_factory=list)
    name_en: str = ""
    description_en: str = ""
    examples_en: list[str] = field(default_factory=list)
    db_role: str | None = None  # database identity (PostgreSQL role) for the db_* tools
    model: str | None = None  # preferred model, if the definition names one
    skills: list[str] = field(default_factory=list)  # names under data/lab/skills

    def label(self, lang: str) -> str:
        return self.name_en if lang == "en" and self.name_en else self.name

    def about(self, lang: str) -> str:
        return self.description_en if lang == "en" and self.description_en else self.description

    def example_prompts(self, lang: str) -> list[str]:
        return self.examples_en if lang == "en" and self.examples_en else self.examples


def resolve_tools(names: list[str], db_role: str | None) -> list:
    db_tools = {t.name: t for t in make_tools(db_role)} if db_role else {}
    missing = [n for n in names if n not in TOOLS and n not in db_tools]
    if missing:
        raise ValueError(f"unknown tools {missing}; available: {sorted(TOOLS) + list(DB_TOOLS)}")
    return [db_tools.get(n) or TOOLS[n] for n in names]


def spec_from_dict(d: dict) -> AgentSpec:
    def both(value) -> tuple:
        return (
            (value.get("zh", ""), value.get("en", "")) if isinstance(value, dict) else (value, "")
        )

    name, name_en = both(d["name"])
    description, description_en = both(d.get("description", ""))
    examples = d.get("examples") or {}
    examples = examples if isinstance(examples, dict) else {"zh": examples}
    return AgentSpec(
        key=d["id"],
        name=name,
        name_en=name_en,
        description=description,
        description_en=description_en,
        system_prompt=d["system_prompt"],
        tools=resolve_tools(d.get("tools", []), d.get("db_role")),
        suggested_task=d.get("suggested_task", "chat"),
        examples=examples.get("zh", []),
        examples_en=examples.get("en", []),
        db_role=d.get("db_role"),
        model=d.get("model"),
        skills=list(d.get("skills") or []),
    )


def load_agent_definitions(path: Path = AGENTS_FILE) -> list[dict]:
    return json.loads(path.read_text())["agents"]


class _AgentRegistry(Mapping):
    """`AGENTS[key]` -> AgentSpec, re-read whenever data/lab/agents.json changes."""

    def __init__(self, path: Path):
        self._path, self._stamp, self._specs = path, None, {}

    def _load(self) -> dict[str, AgentSpec]:
        stamp = self._path.stat().st_mtime_ns
        if stamp != self._stamp:
            self._specs = {d["id"]: spec_from_dict(d) for d in load_agent_definitions(self._path)}
            self._stamp = stamp
        return self._specs

    def __getitem__(self, key: str) -> AgentSpec:
        return self._load()[key]

    def __iter__(self):
        return iter(self._load())

    def __len__(self) -> int:
        return len(self._load())


AGENTS = _AgentRegistry(AGENTS_FILE)


def all_tools() -> dict:
    """Every tool an agent can use, by name (database tools shown with the Sales role)."""
    return {**TOOLS, **{t.name: t for t in make_tools(DB_ROLES[0])}}


def tool_users(name: str) -> list[str]:
    return [spec.name for spec in AGENTS.values() if any(t.name == name for t in spec.tools)]


@dataclass(frozen=True)
class Policy:
    """Code-level controls on tool use; the model cannot override any of these."""

    allow_network: bool = False  # bind external tools at all
    approve_external: bool = True  # pause before every external call for a human decision
    external_limit: int = EXTERNAL_CALL_LIMIT  # max calls per external tool per run


def available_tools(spec: AgentSpec, policy: Policy) -> list:
    """Tools the model will actually see: external ones are dropped unless allowed."""
    return [t for t in spec.tools if policy.allow_network or not is_external(t.name)]


def build(
    spec: AgentSpec,
    config: Settings,
    think: bool | None = None,
    temperature: float = 0,
    policy: Policy | None = None,
):
    policy = policy or Policy()
    tools = available_tools(spec, policy)
    if spec.skills:
        tools = [*tools, skills.skill_tool(spec.skills)]
    external = [t.name for t in tools if is_external(t.name)]
    middleware = [
        ToolCallLimitMiddleware(tool_name=name, run_limit=policy.external_limit)
        for name in external
    ]
    if external and policy.approve_external:
        middleware.append(
            HumanInTheLoopMiddleware(
                interrupt_on={
                    name: {"allowed_decisions": ["approve", "edit", "reject"]} for name in external
                },
                description_prefix="外部 API 呼叫需要核准",
            )
        )
    return create_agent(
        model=build_model(config, think=think, temperature=temperature),
        tools=tools,
        system_prompt=spec.system_prompt + skills.skill_prompt(spec.skills),
        middleware=middleware,
        # Approval pauses the run; the checkpointer keeps its state until a human decides.
        checkpointer=InMemorySaver() if middleware else None,
    )
