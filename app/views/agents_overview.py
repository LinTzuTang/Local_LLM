import streamlit as st
import ui
from i18n import current, pick, tr

from local_llm import catalog, company_db, skills
from local_llm.agents import AGENTS, TOOL_INFO, load_agent_definitions
from local_llm.lab import store

lang = current()
ui.page_header(
    "groups",
    tr("Agents 一覽", "All agents"),
    tr(
        "每個 agent 的模型、工具、技能與資料庫身分。在「Agent 工作室」修改或新增；"
        "設定存在 `data/lab/agents.json`，也可以請總控 Coding Agent 修改。",
        "Model, tools, skills and database identity of every agent. Edit or add them in the "
        "Agent studio; definitions live in `data/lab/agents.json`, and the coding agent can "
        "change them too.",
    ),
)

definitions = load_agent_definitions()
identity = {i["role"]: i for i in company_db.identities()}
skill_info = {s["name"]: s for s in skills.list_skills()}
used_in = {}
for s in store.scenarios():
    for node in s["nodes"]:
        if node["type"] == "agent":
            used_in.setdefault(node["params"].get("agent"), []).append(pick(s["name"]))

GROUPS = {
    "all": tr("全部", "All"),
    "db": ":material/database: " + tr("資料庫（依身分權限）", "Database (identity-scoped)"),
    "tools": ":material/build: " + tr("其他工具型", "Other tool users"),
    "plain": ":material/chat_bubble: " + tr("純對話", "Chat only"),
}
group = st.segmented_control(
    tr("分類", "Group"), list(GROUPS), format_func=GROUPS.get, default="all"
)


def kind(d: dict) -> str:
    if d.get("db_role"):
        return "db"
    return "tools" if d.get("tools") or d.get("skills") else "plain"


def open_in_studio(agent_id: str) -> None:
    st.session_state["agent_type"] = agent_id
    st.session_state["_open_studio"] = True


C = tr("：", ": ")
shown = [d for d in definitions if group in (None, "all") or kind(d) == group]
for row in range(0, len(shown), 2):
    for col, d in zip(st.columns(2, gap="medium"), shown[row : row + 2], strict=False):
        spec = AGENTS[d["id"]]
        with col.container(border=True):
            top, btn = st.columns([4, 1.3], vertical_alignment="center")
            top.markdown(f"#### {spec.label(lang)}")
            btn.button(
                tr("在工作室開啟", "Open in studio"),
                key=f"open_{d['id']}",
                on_click=open_in_studio,
                args=(d["id"],),
                use_container_width=True,
            )
            st.caption(spec.about(lang))
            caps = ["chat"] + (["tools"] if d.get("tools") else [])
            network = any(TOOL_INFO.get(t) and TOOL_INFO[t].network for t in d.get("tools", []))
            st.markdown(
                ui.badges(
                    caps,
                    [f"id: {d['id']}"]
                    + (
                        [":material/public: " + tr("需連網核准", "network, approval")]
                        if network
                        else []
                    )
                    + ([f"model: {d['model']}"] if d.get("model") else [])
                    + (
                        [f"{tr('預設模型', 'default model')}: {catalog.default_model()}"]
                        if not d.get("model")
                        else []
                    ),  # fmt: skip
                ),
                unsafe_allow_html=True,
            )
            rows = []
            if d.get("db_role"):
                ident = identity.get(d["db_role"])
                label = pick(ident["label"]) if ident else d["db_role"]
                desc = pick(ident["description"]) if ident else ""
                rows.append(
                    f":material/database: **{tr('資料庫身分', 'Database identity')}**{C}{label}（`{d['db_role']}`）· {desc}"
                )
            if d.get("tools"):
                tool_bits = []
                for t in d["tools"]:
                    info = TOOL_INFO.get(t)
                    icon = (
                        "public"
                        if info and info.network
                        else ("database" if t.startswith("db_") else "build")
                    )
                    tool_bits.append(f":material/{icon}: `{t}`")
                rows.append(f"**{tr('工具', 'Tools')}**{C}" + " ".join(tool_bits))
            else:
                rows.append(
                    f"**{tr('工具', 'Tools')}**{C}" + tr("無（只呼叫 LLM）", "none (LLM only)")
                )
            if d.get("skills"):
                rows.append(
                    f"**{tr('技能', 'Skills')}**{C}"
                    + " ".join(
                        f":material/menu_book: `{k}`"
                        + ("" if k in skill_info else tr("（未安裝）", " (missing)"))
                        for k in d["skills"]
                    )
                )
            if used_in.get(d["id"]):
                rows.append(
                    f"**{tr('用在場景', 'Used in scenarios')}**{C}" + "、".join(used_in[d["id"]])
                )
            st.markdown("  \n".join(rows))
            with st.expander("System prompt"):
                st.code(d["system_prompt"], language=None, wrap_lines=True)
            examples = spec.example_prompts(lang)
            if examples:
                st.caption(tr("範例：", "Examples: ") + " ／ ".join(examples[:3]))

if st.session_state.pop("_open_studio", False):
    st.switch_page("views/agent.py")

c1, c2 = st.columns(2)
c1.page_link(
    "views/agent.py",
    label=tr("在工作室建立新 agent", "Create an agent in the studio"),
    icon=":material/add:",
)
c2.page_link(
    "views/agent_compare.py",
    label=tr("同一題問多個 agent", "Ask several agents the same question"),
    icon=":material/compare:",
)
