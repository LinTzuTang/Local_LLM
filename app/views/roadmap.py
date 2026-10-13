"""Adoption roadmap: the eight platform steps as a journey (prepare → pilot → govern → operate),
a typical 12-week plan for the fictional 300-person company, and the full write-up."""

import html

import altair as alt
import pandas as pd
import streamlit as st
import ui
from common import PROJECT_DIR
from i18n import is_en, tr

# Phase colours: the first four slots of the validated categorical palette (in fixed order).
PHASES = {
    "prepare": ("準備", "Prepare", "#2A78D6", "W1–4"),
    "pilot": ("試點", "Pilot", "#EB6834", "W4–9"),
    "govern": ("治理", "Govern", "#1BAF7A", "W7–10"),
    "operate": ("營運", "Operate", "#EDA100", "W9–12"),
}
PHASE_GOALS = {
    "prepare": ("挑模型、整理資料與權限", "Pick models, sort out data and access"),
    "pilot": ("在受控範圍內讓部門真的用", "Real use by departments, in a controlled scope"),
    "govern": ("資料出公司前一定有人把關", "A human gate before data leaves the company"),
    "operate": ("持續調整，每次都可稽核", "Keep improving, every run auditable"),
}


def step(icon, phase, title, page, page_name, short, do, see, stage):
    return {
        "icon": icon,
        "phase": phase,
        "title": title,
        "page": page,
        "page_name": page_name,
        "short": short,
        "do": do,
        "see": see,
        "stage": stage,
    }


STEPS = [
    step(
        "📦",
        "prepare",
        tr("選模型", "Pick models"),
        "views/models.py",
        tr("模型庫", "Model library"),
        tr("這台跑得動哪些模型", "What this machine can run"),
        tr(
            "依「級距」看這台跑得動哪些模型；用「系列」篩選 TAIDE、Breeze 等台灣主權 LLM，按「下載」。"
            "語音、OCR、reranker 在「非 Ollama」分頁。",
            "Use the size tiers to see what this machine can run; filter by family for Taiwan's "
            "sovereign LLMs (TAIDE, Breeze) and download. Speech, OCR and rerankers are under "
            "'Non-Ollama'.",
        ),
        tr(
            "有哪些開源模型可選、大小與授權（注意 Cohere 多為非商用）。",
            "Which open models exist, their size and licence (most Cohere models are "
            "non-commercial).",
        ),
        tr("平台建置", "Platform setup"),
    ),
    step(
        "⚖️",
        "prepare",
        tr("比較模型", "Compare models"),
        "views/compare.py",
        tr("模型比較", "Compare models"),
        tr("選出公司預設模型", "Choose the default model"),
        tr(
            "按「快速組合 → 台灣主權 LLM」，用「繁中寫作」題目跑一次，開啟裁判評分。",
            "Pick the 'Taiwan sovereign LLMs' preset, run the Traditional-Chinese writing prompt "
            "and turn on the judge.",
        ),
        tr(
            "同一題各模型的答案、速度、分數並排，決定公司預設模型。",
            "Answers, speed and scores side by side — choose the company's default model.",
        ),
        tr("模型選型", "Model selection"),
    ),
    step(
        "🔐",
        "prepare",
        tr("準備資料與權限", "Data & permissions"),
        "views/company.py",
        tr("公司資料庫", "Company database"),
        tr("權限由資料庫強制", "Access enforced by the DB"),
        tr(
            "在「⬆️ 上傳資料」上傳一份 CSV，再到「🔐 權限設定」只開放給業務部。",
            "Upload a CSV under 'Upload', then open it to Sales only under 'Permissions'.",
        ),
        tr(
            "權限由資料庫強制：沒開放的部門 agent 根本查不到。",
            "The database enforces permissions: other departments' agents cannot read it.",
        ),
        tr("資料盤點與分級", "Data inventory"),
    ),
    step(
        "📚",
        "pilot",
        tr("文件知識庫", "Document knowledge base"),
        "views/knowledge.py",
        tr("知識庫 RAG", "Knowledge base"),
        tr("混合檢索＋權限過濾", "Hybrid, permission-aware search"),
        tr(
            "在「🔎 檢索比較」用英文問中文文件，比較 BM25、向量、RRF 與 Rerank 的排名；"
            "再以業務部身分查「P3 月薪範圍」。",
            "In 'Compare retrieval' ask an English question over Chinese documents and compare "
            "BM25, vector, RRF and rerank; then search 'P3 salary range' as Sales.",
        ),
        tr(
            "混合檢索為什麼比單一方法穩；機密文件在檢索時就被過濾。",
            "Why hybrid retrieval is more robust, and how restricted documents are filtered "
            "before they reach a prompt.",
        ),
        tr("知識管理", "Knowledge management"),
    ),
    step(
        "🆚",
        "pilot",
        tr("部門試點", "Department pilot"),
        "views/agent_compare.py",
        tr("多 Agent 比較", "Compare agents"),
        tr("同一題，不同權限", "Same question, different access"),
        tr(
            "選業務部、人資部、客服部三個 agent，問「各部門的在職人數與平均月薪」，三個同時回答。",
            "Pick the Sales, HR and Support agents and ask 'Headcount and average salary per "
            "department'; all three answer at once.",
        ),
        tr(
            "同模型同問題，人資部答得出來，業務部與客服部被權限擋下；步驟與時間並排比較。",
            "Same model, same question: HR answers, Sales and Support are blocked — steps and "
            "time side by side.",
        ),
        tr("受控試點", "Controlled pilot"),
    ),
    step(
        "🕸️",
        "pilot",
        tr("多 Agent 流程", "Multi-agent workflows"),
        "views/scenarios.py",
        tr("場景展示", "Scenario gallery"),
        tr("串成流程、找出瓶頸", "Chain steps, find bottlenecks"),
        tr(
            "跑「公司問答路由」與「客服工單分流」，看 nimble 怎麼分派、哪些步驟平行執行。",
            "Run the company router and support triage; watch nimble dispatch and which steps "
            "run in parallel.",
        ),
        tr(
            "把多個模型與 agent 串成流程，並用時間軸找出瓶頸。",
            "Chaining models and agents into workflows, and finding bottlenecks on the timeline.",
        ),
        tr("流程自動化", "Workflow automation"),
    ),
    step(
        "✋",
        "govern",
        tr("外部資料要核准", "Approve external calls"),
        "views/agent.py",
        tr("Agent 工作室", "Agent studio"),
        tr("人工核准才送出", "A human approves egress"),
        tr(
            "選「研究助理」，開啟「允許連網工具」，問天氣；跳出核准卡片時檢查參數再核准。",
            "Pick the research assistant, allow network tools and ask about the weather; check "
            "the arguments on the approval card.",
        ),
        tr(
            "資料送出公司前一定經過人工確認，也能拒絕或修改。",
            "Nothing leaves the company without a human approving, editing or rejecting it.",
        ),
        tr("治理與資安", "Governance & security"),
    ),
    step(
        "🔁",
        "operate",
        tr("持續調整與稽核", "Iterate & audit"),
        "views/control.py",
        tr("總控 Coding Agent", "Coding agent"),
        tr("diff、版本、可稽核", "Diffs, versions, audits"),
        tr(
            "請總控 Coding Agent「把混合檢索 RAG 的回答模型改成 qwen3:1.7b」，核准 diff 後再到"
            "「執行紀錄」對照前後兩次執行。",
            "Ask the coding agent to 'switch the hybrid RAG answer model to qwen3:1.7b', approve "
            "the diff, then compare the runs before and after in the run history.",
        ),
        tr(
            "設定變更有 diff、核准與版本，每次執行都存成 JSON 可稽核。",
            "Configuration changes are diffed, approved and versioned; every run is saved as "
            "JSON for audits.",
        ),
        tr("維運與持續改善", "Operations"),
    ),
]

CSS = """
<style>
.rm-scroll { overflow-x: auto; margin: .3rem 0 1.4rem 0; }
.rm-journey { min-width: 900px; display: grid; grid-template-columns: repeat(8, minmax(0, 1fr));
  column-gap: 10px; row-gap: 14px; }
.rm-phase { border-radius: 12px; padding: 8px 12px; display: flex; align-items: baseline;
  justify-content: space-between; gap: 8px; background: color-mix(in srgb, var(--c) 10%, white);
  border: 1px solid color-mix(in srgb, var(--c) 30%, white); }
.rm-phase b { color: color-mix(in srgb, var(--c) 75%, black); font-size: .92rem; }
.rm-phase span { color: #64748B; font-size: .76rem; white-space: nowrap; }
.rm-phase small { display: block; color: #475569; font-size: .76rem; font-weight: 500;
  margin-top: 1px; }
.rm-track { grid-column: 1 / -1; display: grid; grid-template-columns: repeat(8, minmax(0, 1fr));
  column-gap: 10px; position: relative; }
.rm-track::before { content: ""; position: absolute; top: 25px; left: 6%; right: 6%; height: 3px;
  border-radius: 3px; background: linear-gradient(90deg, #2A78D6 0 37.5%, #EB6834 37.5% 75%,
  #1BAF7A 75% 87.5%, #EDA100 87.5% 100%); opacity: .55; }
a.rm-node, a.rm-node:hover { text-decoration: none; color: inherit; position: relative;
  display: flex; flex-direction: column; align-items: center; text-align: center; }
.rm-dot { width: 52px; height: 52px; border-radius: 50%; display: flex; align-items: center;
  justify-content: center; font-size: 1.4rem; background: #FFFFFF; border: 3px solid var(--c);
  box-shadow: 0 2px 8px rgba(15, 23, 42, .08); transition: transform .15s ease; }
a.rm-node:hover .rm-dot { transform: translateY(-2px) scale(1.05); }
.rm-num { margin-top: 8px; font-size: .7rem; font-weight: 750; letter-spacing: .07em;
  color: color-mix(in srgb, var(--c) 80%, black); }
.rm-t { font-weight: 700; font-size: .92rem; color: #0F172A; line-height: 1.3; margin-top: 1px; }
.rm-s { font-size: .78rem; color: #64748B; line-height: 1.35; margin-top: 2px; padding: 0 4px; }
.rm-sechead { display: flex; align-items: baseline; gap: 10px; margin: 1.2rem 0 .5rem 0; }
.rm-sechead .tag { font-size: .74rem; font-weight: 750; letter-spacing: .06em; padding: 2px 10px;
  border-radius: 999px; color: #FFFFFF; background: var(--c); }
.rm-sechead .goal { color: #475569; font-size: .92rem; }
.rm-sechead .weeks { color: #94A3B8; font-size: .8rem; margin-left: auto; }
.rm-card { min-height: 236px; scroll-margin-top: 120px; }
.rm-card-head { display: flex; gap: 12px; align-items: center; margin-bottom: 10px; }
.rm-ic { flex: 0 0 44px; height: 44px; border-radius: 12px; display: flex; align-items: center;
  justify-content: center; font-size: 1.35rem;
  background: color-mix(in srgb, var(--c) 12%, white); }
.rm-eyebrow { font-size: .72rem; font-weight: 750; letter-spacing: .06em; text-transform: uppercase;
  color: color-mix(in srgb, var(--c) 80%, black); }
.rm-title { font-size: 1.08rem; font-weight: 750; color: #0F172A; line-height: 1.3; }
.rm-lbl { font-size: .74rem; font-weight: 700; color: #64748B; letter-spacing: .04em;
  margin-top: 8px; }
.rm-txt { font-size: .9rem; color: #1E293B; line-height: 1.5; }
.rm-need { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
.rm-need div { border-radius: 12px; padding: 12px 14px; background: #FFFBEB;
  border: 1px solid #FDE68A; }
.rm-need b { display: block; color: #78350F; margin-bottom: 2px; }
.rm-need span { color: #92400E; font-size: .86rem; line-height: 1.45; }
@media (max-width: 760px) { .rm-need { grid-template-columns: 1fr; } }
</style>
"""


def esc(text: str) -> str:
    return html.escape(str(text))


def phase_label(key: str) -> str:
    zh, en, _, _ = PHASES[key]
    return tr(zh, en)


def journey() -> str:
    """Horizontal stepper: phase bands over the eight numbered steps."""
    bands = []
    for key, (_, _, color, weeks) in PHASES.items():
        span = sum(s["phase"] == key for s in STEPS)
        goal = tr(*PHASE_GOALS[key]) if span > 1 else ""
        bands.append(
            f'<div class="rm-phase" style="grid-column:span {span};--c:{color}">'
            f"<div><b>{esc(phase_label(key))}</b>"
            + (f"<small>{esc(goal)}</small>" if goal else "")
            + f"</div><span>{weeks}</span></div>"
        )
    nodes = []
    for i, s in enumerate(STEPS, start=1):
        color = PHASES[s["phase"]][2]
        nodes.append(
            f'<a class="rm-node" href="#rm-step-{i}" target="_self" style="--c:{color}">'
            f'<div class="rm-dot">{s["icon"]}</div>'
            f'<div class="rm-num">{tr("步驟", "STEP")} {i}</div>'
            f'<div class="rm-t">{esc(s["title"])}</div>'
            f'<div class="rm-s">{esc(s["short"])}</div></a>'
        )
    return (
        f'<div class="rm-scroll"><div class="rm-journey">{"".join(bands)}'
        f'<div class="rm-track">{"".join(nodes)}</div></div></div>'
    )


def step_card(i: int, s: dict) -> None:
    color = PHASES[s["phase"]][2]
    with st.container(border=True):
        st.markdown(
            f'<div class="rm-card" id="rm-step-{i}" style="--c:{color}">'
            f'<div class="rm-card-head"><div class="rm-ic">{s["icon"]}</div><div>'
            f'<div class="rm-eyebrow">{tr("步驟", "Step")} {i} · {esc(s["stage"])}</div>'
            f'<div class="rm-title">{esc(s["title"])}</div></div></div>'
            f'<div class="rm-lbl">{tr("怎麼做", "WHAT YOU DO")}</div>'
            f'<div class="rm-txt">{esc(s["do"])}</div>'
            f'<div class="rm-lbl">{tr("會看到", "WHAT YOU SEE")}</div>'
            f'<div class="rm-txt">{esc(s["see"])}</div></div>',
            unsafe_allow_html=True,
        )
        st.page_link(
            s["page"],
            label=tr(f"開啟「{s['page_name']}」", f"Open {s['page_name']}"),
            icon=":material/arrow_forward:",
        )


def phase_heading(key: str, container=st) -> None:
    zh, en, color, weeks = PHASES[key]
    container.markdown(
        f'<div class="rm-sechead" style="--c:{color}"><span class="tag">{esc(tr(zh, en))}</span>'
        f'<span class="goal">{esc(tr(*PHASE_GOALS[key]))}</span>'
        f'<span class="weeks">{tr("第", "Weeks ")}{weeks.removeprefix("W").replace("W", "")}'
        f"{tr(' 週', '')}</span></div>",
        unsafe_allow_html=True,
    )


# --- 12-week plan ------------------------------------------------------------------------------
# (phase, zh, en, first week, last week, step numbers)
PLAN = [
    ("prepare", "成立 AI 小組、訂定使用政策", "AI steering group & usage policy", 1, 2, ""),
    ("prepare", "選模型與比較", "Pick and compare models", 1, 3, "1–2"),
    ("prepare", "資料盤點、權限矩陣", "Data inventory & permission matrix", 2, 4, "3"),
    ("pilot", "知識庫建置與檢索評測", "Knowledge base & retrieval tests", 4, 6, "4"),
    ("pilot", "三部門試點（約 30 人）", "Three-department pilot (~30 people)", 5, 9, "5"),
    ("pilot", "多 Agent 流程自動化", "Multi-agent workflow automation", 6, 9, "6"),
    ("govern", "外部呼叫核准、紅隊測試", "External-call approval & red-team tests", 7, 10, "7"),
    ("operate", "稽核、回歸評測、用量報表", "Audits, regression evals, usage report", 9, 12, "8"),
    ("operate", "全公司推廣準備", "Company-wide rollout preparation", 11, 12, ""),
]
# (after plan row, zh, en, week)
MILESTONES = [
    (1, "選定預設模型", "Default model chosen", 3),
    (2, "資料分級完成", "Data classified", 4),
    (4, "試點期中檢討", "Pilot mid-point review", 7),
    (6, "資安簽核", "Security sign-off", 10),
    (8, "上線決策（Go / No-go）", "Go / no-go decision", 12),
]


def plan_chart() -> alt.LayerChart:
    rows, order = [], []
    for i, (phase, zh, en, first, last, steps) in enumerate(PLAN):
        label = tr(zh, en)
        order.append(label)
        rows.append(
            {
                "row": label,
                "phase": phase_label(phase),
                "start": first - 1,
                "end": last,
                "weeks": f"W{first}–W{last}" if first != last else f"W{first}",
                "steps": steps or "—",
                "kind": "task",
            }
        )
        for after, mzh, men, week in MILESTONES:
            if after == i:
                label = "◆ " + tr(mzh, men)
                order.append(label)
                rows.append(
                    {
                        "row": label,
                        "phase": phase_label(phase),
                        "start": week,
                        "end": week,
                        "weeks": f"W{week}",
                        "steps": "—",
                        "kind": "milestone",
                    }
                )
    df = pd.DataFrame(rows)
    domain = [phase_label(k) for k in PHASES]
    colors = alt.Scale(domain=domain, range=[c for _, _, c, _ in PHASES.values()])
    y = alt.Y(
        "row:N",
        sort=order,
        title=None,
        axis=alt.Axis(
            labelLimit=340,
            labelFontSize=12,
            labelPadding=8,
            ticks=False,
            domain=False,
            # milestones (◆ rows) read darker than the work items
            labelColor={
                "condition": {"test": "indexof(datum.label, '◆') === 0", "value": "#0F172A"},
                "value": "#475569",
            },
            labelFontWeight={
                "condition": {"test": "indexof(datum.label, '◆') === 0", "value": "bold"},
                "value": "normal",
            },
        ),
    )
    tooltip = [
        alt.Tooltip("row:N", title=tr("項目", "Item")),
        alt.Tooltip("phase:N", title=tr("階段", "Phase")),
        alt.Tooltip("weeks:N", title=tr("週次", "Weeks")),
        alt.Tooltip("steps:N", title=tr("對應步驟", "Platform step")),
    ]
    x = alt.X(
        "start:Q",
        scale=alt.Scale(domain=[0, 12.35], nice=False),
        title=None,
        axis=alt.Axis(
            values=[w - 0.5 for w in range(1, 13)],
            labelExpr="'W' + (datum.value + 0.5)",
            grid=False,
            ticks=False,
            domain=False,
            orient="top",
            labelFontSize=11,
            labelColor="#64748B",
        ),
    )
    grid = (
        alt.Chart(pd.DataFrame({"w": list(range(13))}))
        .mark_rule(color="#E2E8F0", strokeWidth=1)
        .encode(x=alt.X("w:Q", scale=alt.Scale(domain=[0, 12.35], nice=False)))
    )
    bars = (
        alt.Chart(df[df["kind"] == "task"])
        .mark_bar(cornerRadius=4, height=16)
        .encode(
            x=x,
            x2="end:Q",
            y=y,
            color=alt.Color(
                "phase:N",
                scale=colors,
                title=None,
                legend=alt.Legend(orient="bottom", direction="horizontal", symbolType="square"),
            ),
            tooltip=tooltip,
        )
    )
    marks = (
        alt.Chart(df[df["kind"] == "milestone"])
        .mark_point(
            shape="diamond", size=260, filled=True, opacity=1, stroke="#FFFFFF", strokeWidth=1.5
        )
        .encode(x=x, y=y, color=alt.Color("phase:N", scale=colors, legend=None), tooltip=tooltip)
    )
    return (
        (grid + bars + marks)
        .properties(height=alt.Step(30))
        .configure_view(stroke=None)
        .configure_axis(labelFont="-apple-system, BlinkMacSystemFont, PingFang TC, sans-serif")
        .configure_legend(labelFontSize=12)
    )


# --- page ----------------------------------------------------------------------------------------

st.markdown(CSS, unsafe_allow_html=True)
ui.page_header(
    "🗺️",
    tr("導入路線圖", "Adoption roadmap"),
    tr(
        "情境：示範精密工業（虛構，約 300 人）想導入地端 AI，資料不能離開公司。"
        "照下面八步在平台上操作一遍，就是一次縮小版的導入專案：準備 → 試點 → 治理 → 營運。",
        "Scenario: 示範精密工業, a fictional 300-person Taiwanese manufacturer, wants on-prem AI "
        "and its data must not leave the company. Walking through the eight steps below on this "
        "platform is a miniature adoption project: prepare → pilot → govern → operate.",
    ),
)

st.markdown(journey(), unsafe_allow_html=True)

# --- step cards, grouped by phase ---------------------------------------------------------------
numbered = list(enumerate(STEPS, start=1))
for key in ("prepare", "pilot"):
    phase_heading(key)
    group = [(i, s) for i, s in numbered if s["phase"] == key]
    for col, (i, s) in zip(st.columns(len(group), gap="medium"), group, strict=True):
        with col:
            step_card(i, s)
tail = st.columns(2, gap="medium")
for col, key in zip(tail, ("govern", "operate"), strict=True):
    with col:
        phase_heading(key)
        for i, s in numbered:
            if s["phase"] == key:
                step_card(i, s)

# --- 12-week plan --------------------------------------------------------------------------------
st.markdown(f"### {tr('典型的 12 週導入計畫', 'A typical 12-week adoption plan')}")
st.caption(
    tr(
        "以示範精密工業為例：先用 4 週把模型、資料與權限準備好，再讓約 30 人的三個部門試點，"
        "治理與稽核和試點並行，第 12 週決定是否全公司上線。◆ 是里程碑；滑過長條可看對應的平台步驟。",
        "For 示範精密工業: four weeks to get models, data and permissions ready, then a pilot "
        "with about 30 people in three departments, with governance and audits running alongside; "
        "week 12 decides on a company-wide rollout. ◆ marks a milestone; hover a bar for its "
        "platform step.",
    )
)
with st.container(border=True):
    st.altair_chart(plan_chart(), use_container_width=True)

# --- before production -----------------------------------------------------------------------------
st.markdown(f"### {tr('正式導入前還要補的', 'Still needed before production')}")
needs = [
    (
        tr("🔑 使用者登入（SSO）", "🔑 User login (SSO)"),
        tr(
            "目前權限依「選哪個 agent」決定，而不是「誰登入」；正式環境要接 AD／SSO。",
            "Permissions follow the chosen agent, not the signed-in user; production needs "
            "AD / SSO.",
        ),
    ),
    (
        tr("🧪 評測題庫", "🧪 Evaluation set"),
        tr(
            "每個用途 100–300 題繁中標準答案與攻擊題，換模型前先跑回歸評測。",
            "100–300 Traditional-Chinese reference questions and attack prompts per use case, "
            "run before every model change.",
        ),
    ),
    (
        tr("📊 用量上限", "📊 Usage quotas"),
        tr(
            "依部門與使用者設定用量上限與報表，避免單一部門佔滿 GPU。",
            "Per-department and per-user limits and reports, so no team can hog the GPU.",
        ),
    ),
]
st.markdown(
    '<div class="rm-need">'
    + "".join(f"<div><b>{esc(t)}</b><span>{esc(d)}</span></div>" for t, d in needs)
    + "</div>",
    unsafe_allow_html=True,
)
st.write("")

with st.expander(
    tr(
        "完整版：八個導入階段、常見陷阱、可借鏡的平台與治理框架",
        "Full version: eight adoption phases, pitfalls, reference platforms and governance",
    ),
    icon=":material/description:",
):
    en_doc = PROJECT_DIR / "docs" / "ADOPTION_ROADMAP.en.md"
    path = en_doc if is_en() and en_doc.exists() else PROJECT_DIR / "docs" / "ADOPTION_ROADMAP.md"
    doc = path.read_text()
    st.markdown(doc.split("\n", 1)[1] if doc.startswith("# ") else doc)
