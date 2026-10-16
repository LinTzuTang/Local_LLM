"""Drawing scenario runs: a Graphviz flow diagram with live node status, an Altair timeline
(waterfall, like a tracing UI) and per-step details."""

import html

import altair as alt
import pandas as pd
import streamlit as st
import ui
from i18n import pick, tr

from local_llm.lab.nodes import NODE_TYPES

STATUS = {  # border colour, fill override, pen width
    "running": ("#2563EB", "#DBEAFE", 3.0),
    "done": ("#16A34A", None, 2.2),
    "error": ("#DC2626", "#FEE2E2", 2.5),
    "cancelled": ("#D97706", "#FEF3C7", 2.0),
}
# Graphviz cannot draw icon fonts, so diagrams use plain text marks and the node colours.
ICON = {"running": "●", "done": "✓", "error": "✗", "cancelled": "■"}
# Step status and node type as Material Symbols, for expanders and other Streamlit labels.
STATUS_ICON = {
    "running": ":blue[:material/progress_activity:]",
    "done": ":green[:material/check_circle:]",
    "error": ":red[:material/error:]",
    "cancelled": ":orange[:material/stop_circle:]",
}
NODE_ICON = {
    "llm": "psychology",
    "agent": "smart_toy",
    "router": "alt_route",
    "classify": "label",
    "retrieve_bm25": "match_case",
    "retrieve_vector": "scatter_plot",
    "rrf": "merge",
    "rerank": "sort",
    "transcribe": "mic",
    "diarize": "record_voice_over",
    "parse_document": "description",
    "template": "extension",
}
# Scenario categories (scenarios.json "category"); unknown ones fall back to a label icon.
CATEGORY_ICONS = {
    "routing": "alt_route",
    "rag": "travel_explore",
    "speech": "graphic_eq",
    "documents": "description",
    "multi_model": "hub",
    "pipeline": "account_tree",
    "comparison": "compare_arrows",
}
PRIMARY = "#4F46E5"


def category_icon(category: str | None) -> str:
    """Material Symbols name for a scenario category."""
    return CATEGORY_ICONS.get(category or "", "label")


def node_icon(node_type: str) -> str:
    """':material/x:' for a scenario node type (NODE_TYPES[...].icon is an emoji)."""
    return f":material/{NODE_ICON.get(node_type, 'circle')}:"


def short_model(name: str) -> str:
    """hf.co/org/Repo-GGUF:Q4_K_M → Repo; long names stay readable in the diagram."""
    if name.startswith("hf.co/"):
        name = name.split("/")[-1].split(":")[0].removesuffix("-GGUF")
    return name if len(name) <= 28 else name[:26] + "…"


def _esc(text: str) -> str:
    return html.escape(str(text), quote=True)


def flow_dot(
    scenario: dict,
    trace: list[dict] | None = None,
    finished: bool = False,
    rankdir: str = "LR",
    compact: bool = False,
    models_override: dict | None = None,
) -> str:
    """Graphviz DOT for a scenario; with a trace, nodes show status, time and the path taken.

    compact: thumbnail style (label only, larger text). models_override: {node_id: model}
    chosen on the scenario page for this run.
    """
    trace = trace or []
    latest = {e["node"]: e for e in trace}
    ran = set(latest)
    lines = [
        "digraph {",
        f'graph [rankdir={rankdir}, bgcolor="transparent", nodesep={0.15 if compact else 0.3}, '
        f"ranksep={0.22 if compact else 0.35}, pad=0.05]",
        f'node [shape=box, style="rounded,filled", fontname="Helvetica", '
        f'fontsize={15 if compact else 11}, color="#CBD5E1", penwidth=1.2, '
        f'margin="{"0.08,0.04" if compact else "0.14,0.06"}"]',
        'edge [color="#94A3B8", arrowsize=0.7, fontname="Helvetica", fontsize=9, '
        'fontcolor="#64748B"]',
        f'START [shape=circle, label="{"" if compact else "▶"}", '
        f"width={0.14 if compact else 0.32}, fixedsize=true, style=filled, "
        f'fillcolor="{PRIMARY}", color="{PRIMARY}", fontcolor="white", fontsize=10]',
        f'END [shape=doublecircle, label="", width={0.12 if compact else 0.22}, fixedsize=true, '
        f'style=filled, fillcolor="{PRIMARY if finished else "#CBD5E1"}", color="#94A3B8"]',
    ]
    for node in scenario["nodes"]:
        ntype = NODE_TYPES.get(node["type"])
        params = node.get("params", {})
        if models_override and node["id"] in models_override:
            params = {**params, "model": models_override[node["id"]]}
        models = ntype.models({**{k: p.default for k, p in ntype.params.items()}, **params})
        entry = latest.get(node["id"])
        status = entry["status"] if entry else ("skipped" if finished and trace else "pending")
        title = pick(node.get("label")) or node["id"]
        kind = node_kind({**node, "params": params})
        kind_label = tr(*ui.CAPS[kind][1:3]) if kind else pick(ntype.label)
        sub = " · ".join([kind_label, *(short_model(m) for m in models)])
        rows = [f"<b>{_esc(title)}</b>"]
        if not compact:
            rows.append(f'<font point-size="9" color="#475569">{_esc(sub)}</font>')
        if entry and entry.get("end") is not None:
            secs = entry["end"] - entry["start"]
            rows.append(
                f'<font point-size="9" color="#16A34A">{ICON.get(status, "")} {secs:.1f}s</font>'
            )
        elif status == "running":
            running = _esc(tr("執行中", "running"))
            rows.append(f'<font point-size="9" color="#2563EB">● {running}</font>')
        fill, border, width, style = ntype.color, "#CBD5E1", 1.2, "rounded,filled"
        if status in STATUS:
            border, fill_override, width = STATUS[status]
            fill = fill_override or fill
        elif status == "skipped":
            fill, style = "#F8FAFC", "rounded,filled,dashed"
            rows[0] = f'<font color="#94A3B8">{rows[0]}</font>'
        label = "<" + "<br/>".join(rows) + ">"
        lines.append(
            f'"{node["id"]}" [label={label}, fillcolor="{fill}", color="{border}", '
            f'penwidth={width}, style="{style}"]'
        )
    for e in scenario["edges"]:
        src, dst = e["from"], e["to"]
        taken = (src == "START" or src in ran) and (dst in ran or (dst == "END" and finished))
        if e.get("when") and src in latest:  # a router only takes the edge it chose
            chosen = latest[src]["outputs"].get(
                next(
                    (n.get("params", {}).get("output", "route") for n in scenario["nodes"]
                     if n["id"] == src),
                    "route",
                )
            )  # fmt: skip
            taken = taken and (chosen == e["when"] or e["when"] == "*")
        attrs = [f'color="{PRIMARY}", penwidth=2.2'] if taken and trace else []
        if e.get("when"):
            label = e["when"]
            probs = (latest.get(src) or {}).get("detail", {}).get("probabilities") or {}
            if e["when"] in probs:
                label += f" {probs[e['when']]:.0%}"
            attrs.append(f'label="{_esc(label)}", style="dashed"')
        lines.append(f'"{src}" -> "{dst}" [{", ".join(attrs)}]')
    lines.append("}")
    return "\n".join(lines)


def timeline(trace: list[dict], running: bool, elapsed: float) -> None:
    """Waterfall of node executions: which model ran when, and what ran in parallel."""
    if not trace:
        return
    rows = []
    for e in trace:
        end = e["end"] if e["end"] is not None else elapsed
        ntype = NODE_TYPES[e["type"]]
        rows.append(
            {
                "step": pick(e["label"]),
                "start": round(e["start"], 2),
                "end": round(max(end, e["start"] + 0.05), 2),
                "seconds": round(end - e["start"], 2),
                "type": pick(ntype.label),
                "model": ", ".join(short_model(m) for m in e["models"]) or "—",
                "status": e["status"],
            }
        )
    df = pd.DataFrame(rows)
    order = list(dict.fromkeys(df.sort_values("start")["step"]))
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadius=4, height=16)
        .encode(
            x=alt.X("start:Q", title=tr("秒", "seconds")),
            x2="end:Q",
            y=alt.Y("step:N", sort=order, title=None, scale=alt.Scale(paddingInner=0.35)),
            color=alt.Color("type:N", title=tr("節點類型", "Node type")),
            opacity=alt.condition(alt.datum.status == "running", alt.value(0.55), alt.value(1)),
            tooltip=["step", "model", "start", "end", "seconds", "status"],
        )
        .properties(height=alt.Step(36))
    )
    text = (
        alt.Chart(df)
        .mark_text(align="left", dx=4, fontSize=10, color="#475569")
        .encode(x="end:Q", y=alt.Y("step:N", sort=order, title=None), text="model:N")
    )
    st.altair_chart(chart + text, use_container_width=True)


def step_details(trace: list[dict], elapsed: float) -> None:
    for e in trace:
        ntype = NODE_TYPES[e["type"]]
        secs = (e["end"] if e["end"] is not None else elapsed) - e["start"]
        models = ", ".join(e["models"]) or pick(ntype.label)
        title = f"{STATUS_ICON.get(e['status'], '·')} {pick(e['label'])} · {models}"
        if e["end"] is not None:
            title += f" · {secs:.1f}s"
        with st.expander(title, expanded=e["status"] == "error"):
            if e.get("error"):
                st.error(e["error"])
            detail = e.get("detail") or {}
            if e["inputs"]:
                st.markdown(f"**{tr('輸入', 'Input')}**")
                for key, value in e["inputs"].items():
                    st.caption(key)
                    st.code(value, language=None, wrap_lines=True)
            if probs := detail.get("probabilities"):
                st.markdown(f"**{tr('各選項機率', 'Probability per option')}**")
                st.bar_chart(pd.Series(probs, name="p"), horizontal=True, height=40 * len(probs))
            for name, answer in detail.items():
                if isinstance(answer, dict) and "value" in answer and "probabilities" in answer:
                    st.markdown(f"**{name}** → `{answer['value']}`")
                    if answer["probabilities"]:
                        st.bar_chart(
                            pd.Series(answer["probabilities"], name="p"),
                            horizontal=True,
                            height=40 * len(answer["probabilities"]),
                        )
            if hits := detail.get("hits"):
                st.markdown(f"**{tr('檢索結果', 'Retrieved passages')}**")
                st.dataframe(pd.DataFrame(hits), hide_index=True, width="stretch")
            if trail := detail.get("tool_trail"):
                st.markdown(f"**{tr('工具呼叫', 'Tool calls')}**")
                for t in trail:
                    if "call" in t:
                        st.markdown(f":material/build: `{t['call']}` `{t['args']}`")
                    else:
                        st.code(t["content"], language="json", wrap_lines=True)
            stats = {
                k: v
                for k, v in detail.items()
                if k in ("output_tokens", "prompt_tokens", "tokens_per_s", "first_token_s",
                         "audio_seconds", "segments", "model_seconds", "rtf", "queue_seconds",
                         "load_seconds", "truncated", "pages", "tables", "usage", "speakers",
                         "turns")
                and v is not None
            }  # fmt: skip
            if (cer := detail.get("cer")) is not None:
                st.markdown(
                    f":material/spellcheck: **{detail.get('metric', 'CER')}** {cer * 100:.1f}%"
                )
            if stats:
                st.caption(" · ".join(f"{k}: {v}" for k, v in stats.items()))
            if thinking := detail.get("thinking"):
                st.markdown(
                    f":material/psychology: **{tr('思考過程', 'Thinking')}** ({len(thinking)})"
                )
                st.code(thinking[-4000:], language=None, wrap_lines=True, height=200)
            if e["status"] == "running" and e.get("stream"):
                st.markdown(e["stream"] + " ▌")
            elif e["outputs"]:
                st.markdown(f"**{tr('輸出', 'Output')}**")
                for key, value in e["outputs"].items():
                    st.caption(key)
                    if isinstance(value, str) and len(value) < 4000 and not value.startswith("[{"):
                        st.markdown(value)
                    else:
                        st.code(value, language=None, wrap_lines=True)


def node_kind(node: dict) -> str | None:
    """Model type of a scenario node as a ui.CAPS key (None for algorithm-only nodes)."""
    ntype, params = node.get("type"), node.get("params", {})
    if ntype in ("router", "classify"):
        return "decision" if params.get("method", "nimble") == "nimble" else "chat"
    return {
        "llm": "chat",
        "agent": "tools",
        "retrieve_vector": "embedding",
        "rerank": "rerank",
        "transcribe": "asr",
        "diarize": "diarize",
        "parse_document": "doc",
    }.get(ntype)


def architecture_dot() -> str:
    def box(title: str, sub: str, extra: str = "") -> str:
        return f'[label=<<b>{_esc(title)}</b><br/><font point-size="9">{_esc(sub)}</font>>{extra}]'

    nodes = {
        "ui": box("Streamlit UI", tr("中英雙語・背景工作", "bilingual · background jobs"),
                  ', fillcolor="#EEF2FF", color="#4F46E5"'),
        "engine": box(tr("場景引擎", "Scenario engine"), "scenarios.json → StateGraph"),
        "agents": box("Agents", "agents.json · tools · skills · HITL"),
        "coder": box(tr("總控 Coding Agent", "Coding agent"),
                     tr("diff 核准・版本", "approved diffs · versions")),
        "ollama": box("Ollama :11434", "LLM · VLM · embedding · nimble", ', fillcolor="#ECFDF5"'),
        "gateway": box(tr("模型閘道", "Model gateway") + " :11500",
                       "ASR · Docling · reranker (MLX/MPS)", ', fillcolor="#FCE7F3"'),
        "pg": box("PostgreSQL :5433", tr("角色權限", "role permissions"), ', fillcolor="#FEF3C7"'),
        "kb": box(tr("知識庫", "Knowledge base"), "BM25 + vectors", ', fillcolor="#ECFDF5"'),
        "mcp": box("MCP / " + tr("技能", "skills"), tr("本機工具伺服器", "local tool servers"),
                   ', fillcolor="#F1F5F9"'),
        "web": box(tr("外部 API", "External APIs"), tr("需人工核准", "human approval"),
                   ', style="rounded,dashed", fillcolor="white"'),
    }  # fmt: skip
    core = " ".join(f"{n} {nodes.pop(n)}" for n in ("engine", "agents", "coder"))
    return "\n".join(
        [
            "digraph {",
            'graph [rankdir=LR, bgcolor="transparent", nodesep=0.25, ranksep=0.6]',
            'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, '
            'color="#CBD5E1", fillcolor="#F8FAFC", margin="0.15,0.07"]',
            'edge [color="#94A3B8", arrowsize=0.7]',
            *(f"{name} {attrs}" for name, attrs in nodes.items()),
            'subgraph cluster_core { label="LangGraph / LangChain"; fontname="Helvetica"; '
            f'fontsize=10; color="#E2E8F0"; style="rounded"; {core} }}',
            "ui -> engine; ui -> agents; ui -> coder",
            'coder -> engine [style=dashed, label=" edit", fontsize=9, fontcolor="#64748B"]',
            "coder -> agents [style=dashed]",
            "engine -> agents; engine -> ollama; engine -> gateway; engine -> kb",
            "agents -> ollama; agents -> pg; agents -> mcp; agents -> web [style=dashed]",
            "kb -> ollama [style=dotted]",
            "}",
        ]
    )
