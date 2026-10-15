import html
import json

import pandas as pd
import streamlit as st
import ui
from i18n import current, pick, tr
from langchain_core.utils.function_calling import convert_to_openai_tool

from local_llm import company_db, mcp_tools, media_tools, skills
from local_llm.agents import AGENTS, DB_TOOLS, TOOL_INFO, all_tools

lang = current()
ui.page_header(
    "build",
    tr("工具與技能", "Tools & skills"),
    tr(
        "Agent 能呼叫的<b>工具</b>（本機 Python、公司資料庫、模型閘道、MCP 伺服器、外部 API）與能載入的"
        "<b>技能</b>（Agent Skills 格式的作業說明），以及外部生態系中哪些已整合、哪些沒有、為什麼。",
        "The <b>tools</b> agents can call (local Python, the company database, the model "
        "gateway, MCP servers, external APIs), the <b>skills</b> they can load (instructions "
        "in the Agent Skills format), and which parts of the wider ecosystem are integrated, "
        "which are not, and why.",
    ),
    eyebrow=tr("全部在這台 Mac 上執行", "Everything runs on this Mac"),
)

tools = dict(all_tools())
installed_skills = skills.list_skills()
if installed_skills:  # load_skill is created per agent; show it with every installed skill
    tools["load_skill"] = skills.skill_tool([s["name"] for s in installed_skills])
MEDIA = {t.name for t in media_tools.TOOLS}
GROUPS = {
    "local": ("build", tr("本機 Python", "Local Python"), "#15803D"),
    "db": ("database", tr("公司資料庫", "Company database"), "#0369A1"),
    "media": ("mic", tr("模型閘道／語音文件", "Model gateway / media"), "#BE185D"),
    "mcp": ("power", tr("MCP 伺服器", "MCP servers"), "#6D28D9"),
    "external": ("public", tr("外部 API", "External APIs"), "#C2410C"),
}
GROUP_TEXT = {
    "local": tr(
        "寫在本專案裡的 Python 函式，資料不離開這台電腦。",
        "Python functions in this project; data never leaves this computer.",
    ),
    "db": tr(
        "每個資料庫身分各有一組；agent 定義的 `db_role` 決定用哪個 PostgreSQL 角色連線，權限由資料庫強制。",
        "One set per database identity: the agent's `db_role` picks the PostgreSQL role it "
        "connects as, and the database enforces the permissions.",
    ),
    "media": tr(
        "讀 inbox 裡的檔案，交給本機模型閘道（127.0.0.1:11500）轉錄語音、解析文件。",
        "Read files in the inbox and hand them to the local model gateway (127.0.0.1:11500) "
        "for transcription and document parsing.",
    ),
    "mcp": tr(
        "現成的 MCP 伺服器，在本機以子行程（stdio）執行，第一次呼叫時啟動。",
        "Ready-made MCP servers running as local child processes (stdio), started on first use.",
    ),
    "external": tr(
        "會把參數送到網路上的服務：預設不交給模型，開放後每次呼叫都要人工核准，且有次數上限。",
        "Send their arguments to an internet service: hidden by default; once allowed, every "
        "call needs human approval and is capped per run.",
    ),
}


def group_of(name: str) -> str:
    if name in DB_TOOLS:
        return "db"
    if name in mcp_tools.TOOL_SPECS:
        return "mcp"
    if name in MEDIA:
        return "media"
    if TOOL_INFO.get(name) and TOOL_INFO[name].network:
        return "external"
    return "local"


def pill(text: str, color: str, icon: str = "") -> str:
    """Coloured chip; `icon` is an optional Material Symbols name shown before the text."""
    mark = ui.icon(icon) + " " if icon else ""
    return (
        f'<span class="lab-badge" style="color:{color};border-color:{color}33;'
        f'background:{color}0F">{mark}{html.escape(text)}</span>'
    )


def users(name: str) -> list[str]:
    """Agents (in the current language) that use a tool."""
    if name == "load_skill":
        return [s.label(lang) for s in AGENTS.values() if s.skills]
    return [s.label(lang) for s in AGENTS.values() if any(t.name == name for t in s.tools)]


def md_table(headers: list[str], rows) -> str:
    """A Markdown table (wraps long text, unlike a dataframe)."""
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(c).replace("|", "\\|") for c in row) + " |" for row in rows]
    return "\n".join(lines)


def schema_of(tool) -> dict:
    return convert_to_openai_tool(tool)["function"]["parameters"]


def arg_type(spec: dict) -> str:
    if "type" in spec:
        return spec["type"]
    types = [s.get("type") for s in spec.get("anyOf", []) if s.get("type") != "null"]
    return types[0] if types else "string"


def try_form(name: str, tool) -> None:
    schema = schema_of(tool)
    props, required = schema.get("properties", {}), set(schema.get("required", []))
    network = bool(TOOL_INFO.get(name) and TOOL_INFO[name].network)
    with st.form(f"try_{name}", border=False):
        role = None
        if name in DB_TOOLS:
            idents = company_db.identities()
            role = st.selectbox(
                tr("以哪個資料庫身分執行", "Run as database identity"),
                [i["role"] for i in idents],
                format_func=lambda r: (
                    next(pick(i["label"]) for i in idents if i["role"] == r) + f" · {r}"
                ),
            )
        args = {}
        for arg, spec in props.items():
            label = arg + (" *" if arg in required else "")
            help_text = spec.get("description")
            kind = arg_type(spec)
            key = f"arg_{name}_{arg}"
            if kind == "integer":
                args[arg] = st.number_input(
                    label, value=spec.get("default"), step=1, help=help_text, key=key
                )
            elif kind == "boolean":
                args[arg] = st.checkbox(label, value=bool(spec.get("default")), help=help_text)
            elif arg in {"query", "content", "expression"}:
                args[arg] = st.text_area(label, help=help_text, key=key, height=90)
            else:
                args[arg] = st.text_input(
                    label, value=spec.get("default") or "", help=help_text, key=key
                )
        agreed = (
            st.checkbox(
                tr(
                    "我了解執行後會把參數送出這台電腦",
                    "I understand that running it sends the arguments off this computer",
                ),
                key=f"agree_{name}",
            )
            if network
            else True
        )
        submitted = st.form_submit_button(tr("執行", "Run"), type="primary")
    if not submitted:
        return
    if not agreed:
        st.warning(tr("請先勾選確認。", "Tick the confirmation first."))
        return
    clean = {
        k: (int(v) if isinstance(v, float) and v.is_integer() else v)
        for k, v in args.items()
        if k in required or v not in (None, "")
    }
    runner = tool
    if role:
        runner = {t.name: t for t in company_db.make_tools(role)}[name]
    with st.spinner(tr("執行中…", "Running…")):
        try:
            result = str(runner.invoke(clean))
        except Exception as exc:  # noqa: BLE001 - show any tool failure to the user
            st.error(f"{type(exc).__name__}: {exc}")
            return
    language = "json" if result.lstrip()[:1] in "[{" else None
    st.code(result[:6000], language=language, wrap_lines=True)


def tool_card(name: str, tool) -> None:
    info = TOOL_INFO.get(name)
    group = group_of(name)
    icon, group_label, color = GROUPS[group]
    network = bool(info and info.network)
    with st.container(border=True):
        st.markdown(f"**`{name}`**")
        bits = [pill(group_label, color, icon)]
        if network:
            bits.append(pill(tr("連網", "Network"), "#C2410C", "public"))
        elif group != "local":
            bits.append(pill(tr("本機", "Local"), "#15803D", "computer"))
        if network:
            bits.append(pill(tr("需人工核准", "Needs approval"), "#B91C1C", "front_hand"))
            bits.append(pill(tr("預設隱藏", "Hidden by default"), "#64748B", "visibility_off"))
        if spec := mcp_tools.TOOL_SPECS.get(name):
            bits.append(pill(f"MCP · {spec.server} → {spec.remote}", "#6D28D9"))
        if name in DB_TOOLS:
            bits.append(pill(tr("唯讀 SELECT", "read-only SELECT"), "#0369A1"))
        st.markdown(f'<div class="lab-badges">{"".join(bits)}</div>', unsafe_allow_html=True)
        if info:
            runs, reads = info.text(lang)
            st.markdown(
                f"**{tr('實際執行', 'What runs')}**{tr('：', ': ')}{runs}  \n"
                f"**{tr('存取／送出', 'Reads / sends')}**{tr('：', ': ')}{reads}"
            )
        st.caption(tr("模型看到的說明：", "Description the model sees: ") + tool.description)
        used = users(name)
        st.markdown(
            f"<span class='lab-kv'>{tr('使用的 agent', 'Used by')}{tr('：', ': ')}"
            f"{html.escape(tr('、', ', ').join(used)) if used else '—'}</span>",
            unsafe_allow_html=True,
        )
        with st.expander(tr("模型看到的工具定義（JSON）", "Tool definition the model sees (JSON)")):
            st.code(json.dumps(convert_to_openai_tool(tool), ensure_ascii=False, indent=2), "json")
            if spec and spec.fixed:
                st.caption(
                    tr(
                        f"由程式固定、模型看不到的參數：{', '.join(spec.fixed)}",
                        f"Arguments fixed in code and hidden from the model: "
                        f"{', '.join(spec.fixed)}",
                    )
                )
        with st.expander(
            tr("試跑（不經過模型）", "Try it (no model involved)"), icon=":material/play_arrow:"
        ):
            try_form(name, tool)


tools_tab, skills_tab, eco_tab = st.tabs(
    [
        ":material/build: " + tr(f"工具（{len(tools)}）", f"Tools ({len(tools)})"),
        ":material/menu_book: "
        + tr(f"技能（{len(installed_skills)}）", f"Skills ({len(installed_skills)})"),
        ":material/extension: " + tr("生態系", "Ecosystem"),
    ]
)

# --- Tools ------------------------------------------------------------------------------
with tools_tab:
    by_group = {g: [n for n in tools if group_of(n) == g] for g in GROUPS}
    n_network = sum(1 for n in tools if TOOL_INFO.get(n) and TOOL_INFO[n].network)
    m = st.columns(5)
    m[0].metric(tr("工具總數", "Tools"), len(tools))
    m[1].metric(tr("本機", "Local"), len(tools) - n_network)
    m[2].metric(tr("連網（需核准）", "Network (approval)"), n_network)
    m[3].metric(tr("MCP 工具", "MCP tools"), len(by_group["mcp"]))
    m[4].metric(tr("技能", "Skills"), len(installed_skills))

    with st.expander(tr("工具怎麼運作？", "How do tools work?")):
        st.markdown(
            tr(
                """
**工具不是 API。** 工具是一個在本機執行的函式，加上名稱、說明和參數格式（下方每張卡片的 JSON）。整個 agent 只呼叫一個模型 API：**Ollama 的 `POST /api/chat`**。

1. LangChain 呼叫 `/api/chat`，送出對話與**工具定義**。
2. 模型判斷需要工具時，**不會自己執行任何東西**，只回傳一段 JSON，例如 `{"name": "db_query", "arguments": {"query": "SELECT …"}}`。
3. LangChain 在本機執行對應的函式（Python 函式、資料庫查詢，或轉給 MCP 伺服器），把結果當成 `tool` 訊息加進對話。
4. 再呼叫一次 `/api/chat`，模型看到結果後決定繼續呼叫工具或直接回答。

**MCP 工具**也是一樣：模型只看到 JSON 定義；呼叫時由平台把參數轉給在本機跑的 MCP 伺服器子行程。**連網工具**預設不交給模型；在 Agent 頁開啟「允許連網工具」後，每次呼叫前都要人工核准，並有次數上限——這些都由程式強制，模型無法繞過。
""",
                """
**A tool is not an API.** A tool is a function that runs on this machine, plus a name, a description and a parameter schema (the JSON on each card below). The agent calls a single model API: **Ollama's `POST /api/chat`**.

1. LangChain calls `/api/chat` with the conversation and the **tool definitions**.
2. When the model wants a tool, it **runs nothing itself**; it only returns JSON such as `{"name": "db_query", "arguments": {"query": "SELECT …"}}`.
3. LangChain runs the matching function locally (Python, a database query, or a call forwarded to an MCP server) and adds the result as a `tool` message.
4. It calls `/api/chat` again; with the result in view the model calls another tool or answers.

**MCP tools** work the same way: the model sees only the JSON definition, and the platform forwards the call to an MCP server running as a local child process. **Network tools** are not given to the model by default; after “Allow network tools” is switched on in the Agent page, every call still needs human approval and is capped per run — all enforced in code, which the model cannot bypass.
""",
            )
        )

    shown_groups = st.segmented_control(
        tr("類型", "Kind"),
        list(GROUPS),
        format_func=lambda g: f":material/{GROUPS[g][0]}: {GROUPS[g][1]} ({len(by_group[g])})",
        selection_mode="multi",
        default=list(GROUPS),
        key="tool_groups",
    )
    needle = (
        st.columns([2, 3])[0]
        .text_input(
            tr("搜尋", "Search"),
            placeholder=tr("名稱或說明，例如 time、資料庫", "name or description, e.g. time, git"),
            key="tool_search",
        )
        .lower()
    )
    for group in GROUPS:
        if group not in (shown_groups or []):
            continue
        names = [
            n
            for n in by_group[group]
            if not needle
            or needle in n.lower()
            or needle in tools[n].description.lower()
            or (TOOL_INFO.get(n) and needle in " ".join(TOOL_INFO[n].text(lang)).lower())
        ]
        if not names:
            continue
        icon, label, _ = GROUPS[group]
        st.markdown(f"### :material/{icon}: {label}")
        st.caption(GROUP_TEXT[group])
        for row in range(0, len(names), 2):
            for col, name in zip(st.columns(2, gap="medium"), names[row : row + 2], strict=False):
                with col:
                    tool_card(name, tools[name])

    st.caption(
        tr(
            "新增工具：在 `src/local_llm/` 寫一個 `@tool` 函式，用 `agents.register_tool()` 註冊（附 `ToolInfo` 說明），"
            "再把名稱加進 `data/lab/agents.json` 某個 agent 的 `tools`。MCP 工具在 `src/local_llm/mcp_tools.py` 宣告。",
            "To add a tool: write an `@tool` function in `src/local_llm/`, register it with "
            "`agents.register_tool()` (with a `ToolInfo`) and list its name in an agent's "
            "`tools` in `data/lab/agents.json`. MCP tools are declared in "
            "`src/local_llm/mcp_tools.py`.",
        )
    )

# --- Skills -----------------------------------------------------------------------------
with skills_tab:
    st.markdown(
        tr(
            """
**技能（Skill）** 是一個資料夾 `data/lab/skills/<名稱>/`，裡面的 `SKILL.md` 用開放的 [Agent Skills](https://agentskills.io) 格式寫成：YAML 開頭（`name`、`description`）加上 Markdown 作業說明，可附範例或參考檔。技能只是文字，**不會執行任何程式**。

**漸進揭露**：agent 的系統提示只列出它技能的名稱與一句說明；任務需要時模型呼叫 `load_skill(name)` 讀完整說明，說明提到附檔時再 `load_skill(name, file)`。這樣小型本機模型的 prompt 不會一開始就塞滿。在 Agent 工作室替 agent 勾選技能。
""",
            """
A **skill** is a folder `data/lab/skills/<name>/` whose `SKILL.md` follows the open [Agent Skills](https://agentskills.io) format: YAML front matter (`name`, `description`) and Markdown instructions, optionally with example or reference files. Skills are text only — **nothing in them is executed**.

**Progressive disclosure**: an agent's system prompt lists only its skills' names and one-line descriptions; when a task needs one, the model calls `load_skill(name)` for the full instructions, then `load_skill(name, file)` for a file they mention. Small local models start with a short prompt. Pick an agent's skills in the Agent studio.
""",
        )
    )

    def is_imported(s: dict) -> bool:
        return s["source"].startswith("http")

    own = [s for s in installed_skills if not is_imported(s)]
    imported = [s for s in installed_skills if is_imported(s)]
    k1, k2, k3 = st.columns(3)
    k1.metric(tr("技能", "Skills"), len(installed_skills))
    k2.metric(tr("自製", "In-house"), len(own))
    k3.metric(tr("外部匯入（開源授權）", "Imported (open licence)"), len(imported))
    which = st.segmented_control(
        tr("來源", "Source"),
        ["all", "own", "imported"],
        format_func={
            "all": tr("全部", "All"),
            "own": tr("自製", "In-house"),
            "imported": tr("外部匯入", "Imported"),
        }.get,
        default="all",
        key="skill_filter",
    )
    shown = {"own": own, "imported": imported}.get(which or "all", installed_skills)
    skill_users = {
        s["name"]: [a.label(lang) for a in AGENTS.values() if s["name"] in a.skills]
        for s in installed_skills
    }
    for row in range(0, len(shown), 2):
        for col, s in zip(st.columns(2, gap="medium"), shown[row : row + 2], strict=False):
            with col.container(border=True):
                st.markdown(f":material/menu_book: **`{s['name']}`**")
                bits = [
                    pill(tr("外部匯入", "Imported"), "#6D28D9", "download")
                    if is_imported(s)
                    else pill(tr("自製", "In-house"), "#15803D", "factory"),
                    pill(
                        tr("授權：", "Licence: ")
                        + (
                            tr("公司內部", "internal")
                            if s["license"] == "internal"
                            else s["license"] or "?"
                        ),
                        "#0369A1",
                    ),
                    pill(tr(f"{len(s['files'])} 個附檔", f"{len(s['files'])} files"), "#64748B"),
                ]
                st.markdown(
                    f'<div class="lab-badges">{"".join(bits)}</div>', unsafe_allow_html=True
                )
                description = (
                    s["description_zh"]
                    if lang == "zh" and s["description_zh"]
                    else s["description"]
                )
                st.markdown(description)
                source = (
                    f"[{s['source'].split('github.com/')[-1].split('/tree/')[0]}]({s['source']})"
                    if is_imported(s)
                    else tr("本專案自製", "written for this project")
                )
                st.markdown(
                    f"<span class='lab-kv'>{tr('來源', 'Source')}{tr('：', ': ')}</span>"
                    + source
                    + f"  \n<span class='lab-kv'>{tr('使用的 agent', 'Used by')}{tr('：', ': ')}"
                    + (html.escape(tr("、", ", ").join(skill_users[s["name"]])) or "—")
                    + "</span>",
                    unsafe_allow_html=True,
                )
                if lang == "zh":  # the model always sees the English description
                    english = s["description"]
                    english = english if len(english) < 240 else english[:240] + "…"
                    st.caption("模型看到的說明（英文）：" + english)
                with st.expander(tr("SKILL.md 內容", "SKILL.md")):
                    with st.container(height=420, border=False):
                        st.markdown(skills.get_skill(s["name"])["body"])
                if s["files"]:
                    with st.expander(tr("附檔", "Files")):
                        pick_file = st.selectbox(
                            tr("檔案", "File"), s["files"], key=f"skill_file_{s['name']}"
                        )
                        try:
                            text = skills.read_skill_file(s["name"], pick_file)
                        except (FileNotFoundError, PermissionError) as exc:
                            st.caption(str(exc))
                        else:
                            st.code(text, language="markdown", wrap_lines=True, height=300)

    st.caption(
        tr(
            "新增技能：建立 `data/lab/skills/<名稱>/SKILL.md`（YAML 開頭寫 name、description、description_zh、source、license），"
            "頁面會自動讀到。匯入外部技能前先確認授權允許再散布（例如 Apache-2.0、MIT），並保留 LICENSE 檔。",
            "To add a skill: create `data/lab/skills/<name>/SKILL.md` (front matter: name, "
            "description, description_zh, source, license); the page picks it up "
            "automatically. Before importing an external skill, check that its licence allows "
            "redistribution (e.g. Apache-2.0, MIT) and keep its LICENSE file.",
        )
    )

# --- Ecosystem ----------------------------------------------------------------------------
with eco_tab:
    st.markdown(
        tr(
            "**原則**：整個平台必須能在公司內部離線運作。現成的工具與技能只有在「能在本機跑、授權允許、"
            "不需要雲端帳號」時才整合；需要網路的只能當**連網工具**（預設隱藏、每次人工核准、次數上限）。",
            "**Rule**: the whole platform must work on-prem without the internet. Ready-made "
            "tools and skills are integrated only when they run locally, their licence allows "
            "it and they need no cloud account; anything that needs the network becomes a "
            "**network tool** (hidden by default, human approval per call, capped).",
        )
    )

    st.markdown("### :material/power: " + tr("已整合的 MCP 伺服器", "Integrated MCP servers"))
    st.caption(
        tr(
            "[Model Context Protocol](https://modelcontextprotocol.io) 是開放的工具協定：伺服器列出工具與 JSON schema，"
            "客戶端（這個平台）代模型呼叫。這裡的伺服器都是安裝在本專案虛擬環境裡的 Python 套件，"
            "以 `python -m …` 子行程（stdio）啟動，不需要 Node.js、不會在執行時下載任何東西；"
            "第一次呼叫時啟動、平台結束時關閉。",
            "The [Model Context Protocol](https://modelcontextprotocol.io) is an open tool "
            "protocol: a server lists tools with JSON schemas and the client (this platform) "
            "calls them for the model. These servers are Python packages installed in this "
            "project's virtualenv, started as `python -m …` child processes (stdio): no "
            "Node.js and no downloads at run time. They start on first use and stop when the "
            "platform exits.",
        )
    )
    server_status = mcp_tools.status()
    for row_keys in (list(mcp_tools.SERVERS)[:2], list(mcp_tools.SERVERS)[2:]):
        for col, key in zip(st.columns(2, gap="medium"), row_keys, strict=False):
            server, state = mcp_tools.SERVERS[key], server_status[key]
            with col.container(border=True):
                st.markdown(f"**{pick(server.title)}**　`{key}`")
                bits = [
                    pill(
                        tr("官方參考伺服器", "official reference")
                        if server.origin == "official"
                        else tr("本專案", "this project"),
                        "#6D28D9",
                    ),
                    pill(
                        tr("連網", "network") if server.network else tr("離線", "offline"),
                        "#C2410C" if server.network else "#15803D",
                        "public" if server.network else "wifi_off",
                    ),
                    pill(
                        tr("授權：", "licence: ")
                        + (
                            tr("內部", "internal")
                            if server.license == "internal"
                            else server.license
                        ),
                        "#0369A1",
                    ),
                    pill(
                        f"{server.package} {state['version'] or tr('未安裝', 'not installed')}",
                        "#64748B",
                    ),
                    pill(
                        tr("執行中", "running") if state["running"] else tr("未啟動", "idle"),
                        "#15803D" if state["running"] else "#94A3B8",
                        "radio_button_checked" if state["running"] else "radio_button_unchecked",
                    ),
                ]
                st.markdown(
                    f'<div class="lab-badges">{"".join(bits)}</div>', unsafe_allow_html=True
                )
                st.markdown(pick(server.about))
                st.code(
                    " ".join(["python", *server.command()[1:]]), language="bash", wrap_lines=True
                )
                st.markdown(
                    f"<span class='lab-kv'>{tr('交給 agent 的工具', 'Tools given to agents')}"
                    f"{tr('：', ': ')}</span>"
                    + " ".join(f"`{t}`" for t in state["tools"])
                    + f"  \n<span class='lab-kv'>{tr('來源', 'Source')}{tr('：', ': ')}</span>"
                    + (
                        f"[modelcontextprotocol/servers · {key}]({server.source})"
                        if server.origin == "official"
                        else f"`{server.source}`"
                    ),
                    unsafe_allow_html=True,
                )
                if state["calls"]:
                    st.caption(
                        tr(
                            f"這次平台啟動後已呼叫 {state['calls']} 次。",
                            f"{state['calls']} call(s) since the platform started.",
                        )
                    )
                if key == "git" and not (mcp_tools.GIT_REPO / ".git").exists():
                    st.warning(
                        tr(
                            f"`{mcp_tools.GIT_REPO}` 還不是 git repo，git 工具會回覆「無法使用」。"
                            "在該資料夾執行 `git init`（或設定 `LOCAL_LLM_GIT_REPO` 指向其他 repo）後重新啟動平台即可。",
                            f"`{mcp_tools.GIT_REPO}` is not a git repository yet, so the git "
                            "tools answer “not available”. Run `git init` there (or point "
                            "`LOCAL_LLM_GIT_REPO` at another repository) and restart the "
                            "platform.",
                        )
                    )
                elif state["error"]:
                    st.warning(state["error"])
                b1, b2 = st.columns(2)
                if b1.button(
                    tr("啟動並列出伺服器的工具", "Start and list the server's tools"),
                    key=f"discover_{key}",
                    disabled=bool(key == "git" and state["error"]),
                ):
                    try:
                        with st.spinner(tr("啟動 MCP 伺服器…", "Starting the MCP server…")):
                            st.session_state[f"mcp_tools_{key}"] = mcp_tools.discover(key)
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"{type(exc).__name__}: {exc}")
                if state["running"] and b2.button(tr("停止", "Stop"), key=f"stop_{key}"):
                    mcp_tools.HUB.stop(key)
                    st.rerun()
                if found := st.session_state.get(f"mcp_tools_{key}"):
                    st.caption(
                        tr(
                            "伺服器自己回報的工具（MCP `tools/list`）。只有唯讀、且經過審查的才交給 agent。",
                            "What the server itself reports (MCP `tools/list`). Only reviewed, "
                            "read-only ones are given to agents.",
                        )
                    )
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    tr("工具", "Tool"): t["name"],
                                    # table cells cannot show icons: plain-text marks
                                    tr("唯讀", "Read-only"): "✓" if t["read_only"] else "✗",
                                    tr("連外", "Open world"): "✓" if t["open_world"] else "—",
                                    tr("交給 agent", "Given to agents"): (
                                        "✓" if t["exposed"] else "—"
                                    ),
                                    tr("說明", "Description"): t["description"].split("\n")[0],
                                }
                                for t in found
                            ]
                        ),
                        hide_index=True,
                        width="stretch",
                    )
    st.caption(
        tr(
            f"Git 工具讀的是 `{mcp_tools.GIT_REPO}`（環境變數 `LOCAL_LLM_GIT_REPO` 可改）；它必須是 git repo。"
            "檔案沙盒：`inbox/` = `storage/inbox`（唯讀），`workspace/` = `storage/lab/workspace`。"
            "伺服器的錯誤訊息記在 `storage/lab/mcp.log`。",
            f"The git tools read `{mcp_tools.GIT_REPO}` (change it with `LOCAL_LLM_GIT_REPO`); "
            "it must be a git repository. File sandbox: `inbox/` = `storage/inbox` (read-only), "
            "`workspace/` = `storage/lab/workspace`. Server errors are logged to "
            "`storage/lab/mcp.log`.",
        )
    )

    NOT_INTEGRATED = [
        (
            "MCP · filesystem",
            tr("官方參考伺服器（TypeScript）", "Official reference server (TypeScript)"),
            tr(
                "需要 Node.js／npx，這台 Mac 沒有安裝 → 改用本專案的 Python 檔案沙盒伺服器（同名工具的子集）。",
                "Needs Node.js / npx, not installed here → replaced by this project's Python "
                "file-sandbox server (a subset with the same tool names).",
            ),
        ),
        (
            "MCP · memory",
            tr("官方知識圖譜記憶（TypeScript）", "Official knowledge-graph memory (TypeScript)"),
            tr(
                "需要 Node.js；長期記憶另由「知識庫 RAG」頁處理。",
                "Needs Node.js; long-term knowledge is handled by the Knowledge base (RAG) page.",
            ),
        ),
        (
            "MCP · sequential-thinking / everything",
            tr("官方（TypeScript）", "Official (TypeScript)"),
            tr(
                "需要 Node.js；qwen3 本身就會先思考，everything 只是協定測試用伺服器。",
                "Need Node.js; qwen3 already reasons on its own and “everything” is only a "
                "protocol test server.",
            ),
        ),
        (
            "MCP · postgres / sqlite",
            tr("已封存的參考伺服器", "Archived reference servers"),
            tr(
                "本平台的資料庫工具用「每個身分一個 PostgreSQL 角色」強制權限，比通用 SQL 伺服器更嚴格。",
                "The platform's database tools enforce permissions with one PostgreSQL role per "
                "identity, stricter than a generic SQL server.",
            ),
        ),
        (
            "MCP · GitHub, GitLab, Slack, Google Drive, Google Maps, Brave Search, Sentry",
            tr("已封存或廠商維護", "Archived or vendor-maintained"),
            tr(
                "需要雲端服務與帳號金鑰，不符合地端要求。",
                "Need cloud services and account keys: not on-prem.",
            ),
        ),
        (
            "MCP · Puppeteer / Playwright",
            tr("瀏覽器自動化", "Browser automation"),
            tr(
                "需要 Node.js 與瀏覽器，且會連網；目前不提供給 agent。",
                "Need Node.js and a browser, and reach the network; not offered to agents.",
            ),
        ),
        (
            "MCP · markitdown-mcp (Microsoft, MIT)",
            tr("Python 文件轉 Markdown", "Python document-to-Markdown"),
            tr(
                "需要 mcp 2.x，與官方伺服器要求的 mcp 1.x 衝突；文件解析已由模型閘道的 Docling 處理。",
                "Requires mcp 2.x, which conflicts with the mcp 1.x the official servers need; "
                "document parsing is already done by Docling in the model gateway.",
            ),
        ),
        (
            "LangChain · FileManagementToolkit",
            tr("本機檔案工具", "Local file tools"),
            tr(
                "可離線，但功能與 MCP 檔案沙盒重複；不另外安裝 langchain-community（依賴很多）。",
                "Works offline but duplicates the MCP file sandbox; langchain-community (many "
                "dependencies) is not installed for it.",
            ),
        ),
        (
            "LangChain · ShellTool / PythonREPLTool",
            tr("執行任意指令／程式碼", "Run arbitrary commands / code"),
            tr(
                "可離線，但等於把整台電腦交給模型；只有「總控 Coding Agent」在受控流程中改檔。",
                "Offline, but hands the whole computer to the model; only the control-center "
                "coding agent edits files, in its own supervised flow.",
            ),
        ),
        (
            "LangChain · SQLDatabaseToolkit",
            tr("通用 SQL 工具", "Generic SQL tools"),
            tr(
                "由本平台的 db_* 工具取代（身分權限、唯讀、逾時、列數上限）。",
                "Replaced by the platform's db_* tools (identity permissions, read-only, "
                "timeout, row cap).",
            ),
        ),
        (
            "LangChain · DuckDuckGo / Tavily / SerpAPI / Brave / OpenWeatherMap",
            tr("網路搜尋與天氣", "Web search and weather"),
            tr(
                "需要網路，Tavily／SerpAPI／Brave／OpenWeatherMap 還要雲端金鑰；已有不需金鑰的 wikipedia_search、"
                "get_weather（Open-Meteo）與 fetch。",
                "Need the internet, and Tavily / SerpAPI / Brave / OpenWeatherMap need cloud "
                "keys; key-free wikipedia_search, get_weather (Open-Meteo) and fetch exist.",
            ),
        ),
        (
            "LangChain · RequestsToolkit",
            tr("通用 HTTP 請求", "Generic HTTP requests"),
            tr(
                "可以打內網（SSRF 風險）；fetch_webpage 與 mcp_fetch 都會先擋內網與 localhost。",
                "Can reach intranet hosts (SSRF risk); fetch_webpage and mcp_fetch refuse "
                "intranet and localhost first.",
            ),
        ),
    ]
    st.markdown(
        "### :material/block: " + tr("沒有整合的工具與原因", "Tools not integrated, and why")
    )
    st.markdown(
        md_table(
            [tr("項目", "Item"), tr("是什麼", "What"), tr("為什麼沒整合", "Why not")],
            NOT_INTEGRATED,
        )  # fmt: skip
    )

    st.markdown("### :material/menu_book: " + tr("技能來源", "Skill sources"))
    SKILL_SOURCES = [
        (
            "[anthropics/skills](https://github.com/anthropics/skills)",
            "Apache-2.0 " + tr("（範例技能）", "(example skills)"),
            "internal-comms, discernment-nudge",
            tr(
                "docx／pdf／pptx／xlsx 是 source-available（專有授權）不匯入；doc-coauthoring 資料夾沒有授權檔不匯入；"
                "mcp-builder、skill-creator、claude-api 依賴線上文件或雲端 API；其餘多為設計／網頁產出，需要執行程式。",
                "docx / pdf / pptx / xlsx are source-available (proprietary licence): not "
                "imported; doc-coauthoring has no licence file in its folder; mcp-builder, "
                "skill-creator and claude-api depend on online docs or a cloud API; most "
                "others produce designs or web pages and need code execution.",
            ),
        ),
        (
            "[obra/superpowers](https://github.com/obra/superpowers)",
            "MIT",
            "systematic-debugging",
            tr(
                "其他技能假設 agent 有 shell、git worktree 與子 agent，適合 Coding Agent 而不是一般業務 agent。",
                "The other skills assume an agent with a shell, git worktrees and sub-agents: "
                "suited to coding agents, not business agents.",
            ),
        ),
        (
            "[addyosmani/agent-skills](https://github.com/addyosmani/agent-skills)",
            "MIT",
            "—",
            tr(
                "軟體工程流程技能（規格、測試、程式審查），之後可給總控 Coding Agent 使用。",
                "Software-engineering workflow skills (specs, tests, code review); candidates "
                "for the control-center coding agent later.",
            ),
        ),
        (
            "[agentskills.io](https://agentskills.io)",
            tr("開放規格", "Open specification"),
            "—",
            tr(
                "SKILL.md 格式的規格；本平台多讀 description_zh、source、license 三個欄位給介面用。",
                "The SKILL.md format; this platform also reads description_zh, source and "
                "license for the UI.",
            ),
        ),
    ]
    st.markdown(
        md_table(
            [
                tr("來源", "Source"),
                tr("授權", "Licence"),
                tr("已匯入", "Imported"),
                tr("說明", "Notes"),
            ],
            SKILL_SOURCES,
        )  # fmt: skip
    )
    st.caption(
        tr(
            "匯入的技能保留原始內容與 LICENSE.txt，只在 YAML 開頭加上 description_zh、source（固定到 commit）與 license。",
            "Imported skills keep their original instructions and LICENSE.txt; only "
            "description_zh, source (pinned to a commit) and license were added to the front "
            "matter.",
        )
    )
