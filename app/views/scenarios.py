import copy
import html
import json
import time
import uuid
from pathlib import Path

import flow_ui
import jobs
import model_choice
import pandas as pd
import streamlit as st
import ui
import voice_assistant
import voice_panel
from common import PROJECT_DIR, installed_names, kept, ollama_url, save_run
from i18n import pick, tr

from local_llm import asr_metrics, catalog
from local_llm.lab import engine, rag, store
from local_llm.lab.nodes import NODE_TYPES
from local_llm.lab.textutil import placeholders

ui.page_header(
    "hub",
    tr("場景展示", "Scenario gallery"),
    tr(
        "每個場景是一張由節點（模型、agent、檢索、路由）組成的流程圖，用 LangGraph 執行。"
        "選一個場景、確認每一步要用的模型，按執行就能即時看到哪個節點在跑、用了哪個模型、花了多久。",
        "Each scenario is a graph of nodes (models, agents, retrieval, routers) run with "
        "LangGraph. Pick one, check the model for each step and run it to watch which node is "
        "active, which model it uses and how long it takes.",
    ),
    eyebrow=tr("多 Agent 場景", "Scenarios"),
)

st.markdown(
    """<style>
.sc-title {display: flex; align-items: center; gap: 10px; font-size: 1.3rem; font-weight: 750;
  line-height: 1.3; margin: .1rem 0 0 0; color: #0F172A; flex-wrap: wrap;}
.sc-title .lab-badges {margin: 0 0 0 4px;}
.st-key-sc_picker [data-testid="stButtonGroup"] button {min-height: 2rem; padding: 0 9px; gap: 4px;}
.st-key-sc_picker [data-testid="stButtonGroup"] button p {font-size: .8rem;}
.st-key-sc_picker [data-testid="stButtonGroup"] [data-testid="stIconMaterial"] {font-size: 1rem;}
.sc-title > .ms {font-size: 22px; color: var(--lab-primary); background: #EEF2FF; border-radius: 9px;
  padding: 5px; vertical-align: 0;}
</style>""",
    unsafe_allow_html=True,
)
UPLOAD_DIR = PROJECT_DIR / "storage" / "lab" / "uploads"
doc = json.loads(store.SCENARIOS_FILE.read_text())
scenarios = {s["id"]: s for s in doc["scenarios"]}
# The local GPT-Live is a live session rather than a graph, but is picked like any scenario.
scenarios.setdefault(voice_assistant.SCENARIO["id"], voice_assistant.SCENARIO)
url = ollama_url()

# Categories come from the data; one only used by a scenario still gets a readable name.
FALLBACK_NAMES = {
    "comparison": {"zh": "模型比較", "en": "Comparison"},
    "other": {"zh": "其他", "en": "Other"},
}
categories = dict(doc.get("categories", {}))
for s in scenarios.values():
    c = s.get("category") or "other"
    title = c.replace("_", " ").title()
    categories.setdefault(c, FALLBACK_NAMES.get(c, {"zh": title, "en": title}))
used = {s.get("category") or "other" for s in scenarios.values()}
categories = {c: v for c, v in categories.items() if c in used}


def category_of(s: dict) -> str:
    return s.get("category") or "other"


def category_label(c: str) -> str:
    return f":material/{flow_ui.category_icon(c)}: {pick(categories[c])}"


# --- compact picker: category filter + scenario select box ---------------------------------
st.session_state.setdefault("scenario_id", next(iter(scenarios)))
if st.session_state.scenario_id not in scenarios:
    st.session_state.scenario_id = next(iter(scenarios))
# Arriving from the home page with another category filtered: show all categories.
chosen_cat = st.session_state.get("scenario_category")
if chosen_cat and category_of(scenarios[st.session_state.scenario_id]) != chosen_cat:
    st.session_state.scenario_category = None


def on_category() -> None:
    """Picking a category jumps to its first scenario (the select box lists only that one)."""
    cat_ = st.session_state.get("_kept_scenario_category")
    st.session_state.scenario_category = cat_
    current_ = scenarios[st.session_state.scenario_id]
    if cat_ and category_of(current_) != cat_:
        st.session_state.scenario_id = next(
            s["id"] for s in scenarios.values() if category_of(s) == cat_
        )


with st.container(border=True, key="sc_picker"):
    pick_col, cat_col = st.columns([1.12, 2.2], vertical_alignment="center", gap="small")
    cat_kw = kept("scenario_category")
    cat_kw["on_change"], cat_kw["args"] = on_category, ()
    cat = cat_col.pills(
        tr("類別", "Category"),
        list(categories),
        format_func=category_label,
        label_visibility="collapsed",
        **cat_kw,
    )
    order = sorted(
        (s for s in scenarios.values() if not cat or category_of(s) == cat),
        key=lambda s: list(categories).index(category_of(s)),
    )
    sid = pick_col.selectbox(
        tr("場景", "Scenario"),
        [s["id"] for s in order],
        format_func=lambda i: (
            f"{pick(scenarios[i]['name'])}  ·  {pick(categories[category_of(scenarios[i])])}"
        ),
        label_visibility="collapsed",
        **kept("scenario_id"),
    )

scenario = scenarios[sid]
kinds = sorted({flow_ui.node_kind(n) for n in scenario["nodes"]} - {None}) or scenario.get(
    "kinds", []
)
st.markdown(
    f'<div class="sc-title">{ui.icon(flow_ui.category_icon(category_of(scenario)))}'
    f"<span>{html.escape(pick(scenario['name']))}</span>"
    f"{ui.badges(kinds, small=True)}</div>",
    unsafe_allow_html=True,
)
info_col, hl_col = st.columns([1, 1], gap="large")
info_col.caption(pick(scenario.get("description")))
if highlights := pick(scenario.get("highlights")) or []:
    hl_col.caption("\n".join(f"- {line}" for line in highlights))
if sid == voice_assistant.SCENARIO["id"]:
    voice_panel.panel(page="views/scenarios.py")
    st.stop()

# --- models for this run ------------------------------------------------------------------------
overrides: dict[str, str] = {}
model_nodes = [n for n in scenario["nodes"] if model_choice.selectable(n)]
with st.container(border=True):
    st.markdown(
        f":material/tune: **{tr('這次使用的模型', 'Models for this run')}**",
        help=tr(
            "每一步只列出能用的模型類型：✓ 已下載、↓ 需要先下載。預設選已下載的模型，方便直接 demo。",
            "Each step lists only compatible models: ✓ downloaded, ↓ needs a download. "
            "Downloaded models are picked by default so the demo just runs.",
        ),
    )
    cols = st.columns(4)
    for i, node in enumerate(model_nodes):
        options, installed = model_choice.choices(node, url)
        key = f"mdl_{sid}_{node['id']}"
        if st.session_state.get(key) not in options:
            st.session_state[key] = model_choice.default_choice(node, url)
        kind = flow_ui.node_kind(node)
        label = f"{flow_ui.node_icon(node['type'])} {pick(node.get('label')) or node['id']}"
        with cols[i % 4]:
            overrides[node["id"]] = st.selectbox(
                label,
                options,
                key=key,
                format_func=lambda m, inst=installed: (
                    ("✓ " if inst.get(m) else "↓ ") + flow_ui.short_model(m)
                ),
                help=ui.cap_label(kind) if kind else None,
            )
    for node in scenario["nodes"]:
        if node["type"] == "retrieve_vector":
            index = rag.manifest(node.get("params", {}).get("collection", "company_kb")) or {}
            st.caption(
                tr(
                    f":material/scatter_plot: 向量檢索用 `{index.get('embed_model', rag.DEFAULT_EMBED)}`（由知識庫索引決定，換模型要重建索引）",
                    f":material/scatter_plot: Vector search uses `{index.get('embed_model', rag.DEFAULT_EMBED)}` (set by the "
                    "knowledge-base index; rebuild it to change the model)",
                )
            )
            break
    changed = {
        n["id"]: overrides[n["id"]]
        for n in model_nodes
        if overrides.get(n["id"]) and overrides[n["id"]] != model_choice.configured_model(n)
    }
    if changed and st.button(
        tr("把這些模型設為場景預設", "Make these the scenario defaults"), icon=":material/save:"
    ):
        updated = copy.deepcopy(scenario)
        for n in updated["nodes"]:
            if n["id"] in changed:
                n.setdefault("params", {})["model"] = changed[n["id"]]
        check = store.save_scenario(updated, source="ui")
        if check.ok:
            st.toast(tr("已儲存（可在總控頁復原）", "Saved (undo from the control center)"))
            st.rerun()
        for e in check.errors:
            st.error(e)


def with_models(s: dict, chosen: dict[str, str]) -> dict:
    out = copy.deepcopy(s)
    for n in out["nodes"]:
        if chosen.get(n["id"]):
            n.setdefault("params", {})["model"] = chosen[n["id"]]
    return out


effective = with_models(scenario, overrides)

# --- readiness ----------------------------------------------------------------------------------
need = engine.requirements(effective)
have = {catalog.normalize(n) for n in installed_names()}
problems = [
    tr(f"Ollama 模型 `{m}` 尚未下載", f"Ollama model `{m}` is not downloaded")
    for m in sorted(need["ollama"])
    if catalog.normalize(m) not in have
]
if need["gateway"]:
    try:
        from local_llm import gateway_client

        gw = {m["id"]: m for m in gateway_client.models()}
        problems += [
            tr(f"模型服務的 `{m}` 尚未下載", f"Gateway model `{m}` is not downloaded")
            for m in sorted(need["gateway"])
            if not gw.get(m, {}).get("installed")
        ]
    except Exception:
        problems.append(
            tr(
                "模型服務（gateway，port 11500）沒有回應：`./scripts/start_gateway.sh`",
                "The model gateway (port 11500) is not answering: `./scripts/start_gateway.sh`",
            )
        )
problems += [
    tr(f"知識庫 `{c}` 尚未建立索引", f"Knowledge base `{c}` has no index yet")
    for c in sorted(need["collections"])
    if not rag.manifest(c)
]
if problems:
    st.warning(
        tr("執行前需要：", "Before running:") + "\n\n" + "\n".join(f"- {p}" for p in problems)
    )
    a, b = st.columns(2)
    a.page_link("views/models.py", label=tr("到模型庫下載 →", "Download in the model library →"))
    b.page_link("views/knowledge.py", label=tr("到知識庫建立索引 →", "Build the index →"))

# --- run state ------------------------------------------------------------------------------------
job_ids = st.session_state.setdefault("scenario_jobs", {})
# The latest run of this scenario, also when it was started in another tab.
job = jobs.get(job_ids.get(sid)) or jobs.find("scenario", scenario=sid)
busy = bool(job and job.running)


def scenario_work(j: jobs.Job, run: engine.Run) -> None:
    """Worker thread: run the graph, then record it (no Streamlit calls)."""
    j.state["run"] = run
    run.execute()
    path = save_run(f"scenario-{run.scenario['id']}", run.summary())
    j.state["saved"] = path.name
    if run.status == "error":
        raise RuntimeError(run.error)


def save_upload(file) -> str:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in file.name)
    path = UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{safe}"
    path.write_bytes(file.getvalue())
    return str(path.relative_to(PROJECT_DIR))


tab_run, tab_def, tab_runs = st.tabs(
    [
        ":material/play_arrow: " + tr("執行", "Run"),
        ":material/schema: " + tr("定義與編輯", "Definition"),
        ":material/history: " + tr("最近執行", "Recent runs"),
    ]
)

with tab_run:
    left, right = st.columns([1, 1.35], gap="large")
    with left:
        inputs = {}
        chosen_samples = {}  # audio/document input key -> bundled sample in use (None = upload)
        for spec in scenario.get("inputs", []):
            key, kind, label = spec["key"], spec.get("type", "text"), pick(spec.get("label"))
            wkey = f"in_{sid}_{key}"
            if kind == "text":
                examples = pick(scenario.get("examples")) or []
                if source_key := spec.get("from_sample"):
                    # Pre-filled with the reference script of the chosen sample (e.g. a transcript
                    # for scoring speech recognition); an upload falls back to the first example.
                    sample = chosen_samples.get(source_key)
                    if st.session_state.get(f"{wkey}_for") != sample:
                        script = sample and asr_metrics.sample_reference(PROJECT_DIR / sample)
                        st.session_state[wkey] = script or (examples[0] if examples else "")
                        st.session_state[f"{wkey}_for"] = sample
                st.session_state.setdefault(wkey, examples[0] if examples else "")
                inputs[key] = st.text_area(label, key=wkey, height=110)
                if examples:
                    st.caption(tr("範例：", "Examples:"))
                    for i, ex in enumerate(examples):
                        # A callback runs before the text area is drawn, so it may set its value.
                        st.button(
                            ex,
                            key=f"ex_{sid}_{i}",
                            disabled=busy,
                            on_click=st.session_state.__setitem__,
                            args=(wkey, ex),
                        )
            elif kind == "choice":
                options = spec["options"]
                inputs[key] = st.segmented_control(
                    label,
                    list(options),
                    default=spec.get("default"),
                    format_func=lambda o, options=options: pick(options[o]),
                    key=wkey,
                )
            else:  # audio / document: bundled sample or an upload
                default = spec.get("default")
                samples = {
                    (x if isinstance(x, str) else x["path"]): (
                        Path(x).name if isinstance(x, str) else pick(x.get("label")) or x["path"]
                    )
                    for x in spec.get("samples") or []
                }
                if len(samples) > 1:
                    st.session_state.setdefault(
                        f"{wkey}_sample", default if default in samples else next(iter(samples))
                    )
                    default = st.selectbox(
                        tr("範例檔", "Sample"),
                        list(samples),
                        format_func=samples.get,
                        key=f"{wkey}_sample",
                        disabled=busy,
                    )
                types = (
                    ["wav", "mp3", "m4a", "flac", "ogg", "webm"]
                    if kind == "audio"
                    else ["pdf", "docx", "md"]
                )
                upload = st.file_uploader(label, type=types, key=wkey)
                if upload:
                    inputs[key] = save_upload(upload) if upload.size else default
                    st.caption(tr(f"使用上傳的 {upload.name}", f"Using upload {upload.name}"))
                    chosen_samples[key] = None
                else:
                    inputs[key] = default
                    chosen_samples[key] = default
                    exists = default and (PROJECT_DIR / default).exists()
                    st.caption(
                        tr("使用範例檔：", "Using the sample: ")
                        + f"`{default}`"
                        + ("" if exists else tr("（尚未建立）", " (missing)"))
                    )
                path = PROJECT_DIR / (inputs[key] or "")
                if kind == "audio" and inputs[key] and path.exists():
                    st.audio(str(path))
        ready = all(v for v in inputs.values())
        c1, c2 = st.columns([2, 1])
        if c1.button(
            tr("執行場景", "Run scenario"),
            icon=":material/play_arrow:",
            type="primary",
            disabled=busy or not ready,
            use_container_width=True,
        ):
            resolved = {
                k: (str(PROJECT_DIR / v) if isinstance(v, str) and v.startswith(("data/", "storage/"))
                    else v)
                for k, v in inputs.items()
            }  # fmt: skip
            run = engine.Run(effective, resolved, url)
            job = jobs.start(
                jobs.Job(
                    "scenario",
                    tr("場景：", "Scenario: ") + pick(scenario["name"])[:14],
                    "views/scenarios.py",
                    state={"run": run, "scenario": sid},
                    cancel=run.cancel,  # the stop button cancels the graph
                ),
                lambda j, run=run: scenario_work(j, run),
            )
            job_ids[sid] = job.id
            st.rerun()
        if busy and c2.button(tr("停止", "Stop"), use_container_width=True):
            job.cancel.set()
    with right:
        flow_box = st.empty()

    def draw(j) -> None:
        run = j.state["run"] if j else None
        trace = run.trace if run else []
        finished = bool(run and run.status in ("done", "error", "cancelled"))
        with flow_box.container():
            shown_s = run.scenario if run else effective
            st.graphviz_chart(flow_ui.flow_dot(shown_s, trace, finished), use_container_width=True)
            if run and run.status != "pending":
                state = {
                    "running": tr("執行中", "Running"),
                    "done": tr("完成", "Done"),
                    "error": tr("失敗", "Failed"),
                    "cancelled": tr("已停止", "Stopped"),
                }.get(run.status, run.status)
                state = f"{flow_ui.STATUS_ICON.get(run.status, '')} {state}"
                models = sorted({m for e in trace for m in e["models"]})
                st.caption(
                    f"{state} · {run.elapsed():.1f}s · "
                    + tr(f"{len(trace)} 個步驟", f"{len(trace)} steps")
                    + " · "
                    + ", ".join(flow_ui.short_model(m) for m in models)
                )

    @st.fragment(run_every=1)
    def live(j: jobs.Job) -> None:
        if not j.running:
            st.rerun()
        draw(j)
        run = j.state["run"]
        st.markdown(f"#### {tr('時間軸', 'Timeline')}")
        flow_ui.timeline(run.trace, True, run.elapsed())
        st.markdown(f"#### {tr('每一步', 'Steps')}")
        flow_ui.step_details(run.trace, run.elapsed())

    if job and job.running:
        live(job)
    else:
        draw(job)
        if job:
            run = job.state["run"]
            if run.error:
                st.error(run.error)
            output = run.output()
            if output:
                st.markdown(f"#### {tr('結果', 'Result')}")
                with st.container(border=True):
                    st.markdown(output)
            st.markdown(f"#### {tr('時間軸', 'Timeline')}")
            flow_ui.timeline(run.trace, False, run.elapsed())
            st.markdown(f"#### {tr('每一步', 'Steps')}")
            flow_ui.step_details(run.trace, run.elapsed())
            if saved := job.state.get("saved"):
                st.caption(tr(f"已存到 `runs/{saved}`", f"Saved to `runs/{saved}`"))

with tab_def:
    nodes = {n["id"]: n for n in scenario["nodes"]}
    incoming: dict[str, list[str]] = {}
    for e in scenario["edges"]:
        src = (
            tr("開始", "start")
            if e["from"] == "START"
            else (pick(nodes[e["from"]].get("label")) or e["from"])
        )
        if e.get("when"):
            src += tr(f"（{e['when']}）", f" (if {e['when']})")
        incoming.setdefault(e["to"], []).append(src)

    def reads(node: dict) -> list[str]:
        p_ = node.get("params", {})
        keys = []
        for name in ("prompt", "system", "template", "input", "query", "role"):
            value = str(p_.get(name) or "")
            keys += placeholders(value) or ([value] if name in ("input", "query") and value else [])
        keys += p_.get("inputs", []) if isinstance(p_.get("inputs"), list) else []
        return list(dict.fromkeys(keys))

    st.markdown(
        tr(
            "**流程怎麼走**：每一步從共用狀態讀資料、呼叫一個模型或演算法，再把結果寫回一個欄位給下一步用。",
            "**How the flow works**: each step reads values from a shared state, calls one model "
            "or algorithm and writes its result to a field the next steps can use.",
        )
    )
    st.graphviz_chart(flow_ui.flow_dot(effective), use_container_width=True)
    rows = []
    for i, node in enumerate(scenario["nodes"], start=1):
        ntype = NODE_TYPES[node["type"]]
        kind = flow_ui.node_kind(node)
        p_ = node.get("params", {})
        writes = [p_.get("output") or ntype.params.get("output").default]
        if node["type"] == "classify":
            writes += list(p_.get("questions", {}))
        rows.append(
            {
                "#": i,
                tr("步驟", "Step"): pick(node.get("label")) or node["id"],
                tr("類型", "Kind"): ui.cap_text(kind) if kind else pick(ntype.label),
                tr("模型", "Model"): flow_ui.short_model(
                    overrides.get(node["id"]) or model_choice.configured_model(node) or "—"
                ),
                tr("讀取", "Reads"): ", ".join(reads(node)) or "—",
                tr("寫入", "Writes"): ", ".join(w for w in writes if w),
                tr("接在後面", "After"): tr("、", ", ").join(incoming.get(node["id"], [])) or "—",
                tr("等待", "Waits"): tr("全部上游", "all inputs")
                if node.get("join") == "all"
                else "",
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    inputs_text = ", ".join(
        f"{i_['key']} ({i_.get('type', 'text')})" for i_ in scenario.get("inputs", [])
    )
    st.caption(
        tr("輸入：", "Inputs: ")
        + (inputs_text or "—")
        + tr("　·　結果欄位：", " · Result field: ")
        + str(scenario.get("output", "answer"))
    )

    st.markdown(f"#### {tr('編輯一個步驟', 'Edit a step')}")
    node_id = st.selectbox(
        tr("步驟", "Step"),
        list(nodes),
        format_func=lambda k: (
            f"{pick(nodes[k].get('label')) or k}  ·  {pick(NODE_TYPES[nodes[k]['type']].label)}"
        ),
        key=f"edit_node_{sid}",
    )
    node = nodes[node_id]
    ntype = NODE_TYPES[node["type"]]
    params = node.get("params", {})
    st.caption(f"{pick(ntype.label)} — {ntype.summary}")
    with st.form(f"node_form_{sid}_{node_id}"):
        c1, c2 = st.columns(2)
        label = node.get("label") or {}
        label_zh = c1.text_input("名稱（中文）", label.get("zh", node_id))
        label_en = c2.text_input("Name (English)", label.get("en", node_id))
        new_params = {}
        for name, spec in ntype.params.items():
            value = params.get(name, spec.default)
            help_text = spec.help
            if name == "model":
                options, installed = model_choice.choices(node, url)
                current = params.get("model") or model_choice.configured_model(node)
                new_params[name] = st.selectbox(
                    "model",
                    options,
                    index=options.index(current) if current in options else 0,
                    format_func=lambda m, inst=installed: ("✓ " if inst.get(m) else "↓ ") + m,
                    help=help_text,
                )
            elif name in ("prompt", "system", "template", "instructions"):
                new_params[name] = st.text_area(
                    name, value or "", height=140 if name != "instructions" else 70, help=help_text
                )
            elif name == "options" and isinstance(value, dict):
                edited = st.data_editor(
                    pd.DataFrame({"option": list(value), "description": list(value.values())}),
                    num_rows="dynamic",
                    hide_index=True,
                    width="stretch",
                    key=f"opts_{sid}_{node_id}",
                )
                new_params[name] = {
                    str(r["option"]): str(r["description"])
                    for _, r in edited.iterrows()
                    if str(r["option"]).strip() and str(r["option"]) != "nan"
                }
            elif isinstance(value, bool) or isinstance(spec.default, bool):
                new_params[name] = st.toggle(name, bool(value), help=help_text)
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                new_params[name] = st.number_input(name, value=value, help=help_text)
            elif isinstance(value, (dict, list)):
                raw = st.text_area(
                    f"{name} (JSON)",
                    json.dumps(value, ensure_ascii=False, indent=2),
                    height=120,
                    help=help_text,
                )
                try:
                    new_params[name] = json.loads(raw)
                except json.JSONDecodeError:
                    new_params[name] = value
                    st.warning(
                        tr(
                            f"{name} 不是合法 JSON，維持原值。",
                            f"{name} is not valid JSON; kept as is.",
                        )
                    )
            else:
                new_params[name] = st.text_input(
                    name, "" if value is None else str(value), help=help_text
                )
        saved = st.form_submit_button(
            tr("驗證並儲存這一步", "Validate & save this step"), type="primary"
        )
    if saved:
        # Keep params that were set before; drop empty optional ones.
        cleaned = {
            k: v
            for k, v in new_params.items()
            if k in params or v not in ("", None, [], {}) and v != ntype.params[k].default
        }
        updated = copy.deepcopy(scenario)
        for n in updated["nodes"]:
            if n["id"] == node_id:
                n["label"] = {"zh": label_zh, "en": label_en}
                n["params"] = cleaned
        if updated == scenario:
            st.info(tr("沒有變更。", "No changes."))
        else:
            check = store.save_scenario(updated, source="ui")
            for e in check.errors:
                st.error(e)
            if check.ok:
                for w in check.warnings:
                    st.warning(w)
                st.toast(tr("已儲存（可在總控頁復原）", "Saved (undo from the control center)"))
                time.sleep(0.6)
                st.rerun()

    st.markdown(
        tr(
            "要新增或刪除步驟、改變串接方式，可以用自然語言請總控 Coding Agent 修改（會顯示 diff 讓你核准）。",
            "To add or remove steps or rewire the flow, ask the coding agent in plain language "
            "(it shows a diff for your approval).",
        )
    )
    st.page_link("views/control.py", label=tr("開啟總控 Coding Agent →", "Open the coding agent →"))
    with st.expander(tr("進階：直接編輯 JSON", "Advanced: edit the JSON")):
        text = st.text_area(
            tr("場景定義（JSON）", "Scenario definition (JSON)"),
            json.dumps(scenario, ensure_ascii=False, indent=2),
            height=380,
            key=f"def_{sid}",
        )
        try:
            edited_json = json.loads(text)
        except json.JSONDecodeError as exc:
            edited_json = None
            st.error(tr(f"JSON 格式錯誤：{exc}", f"Invalid JSON: {exc}"))
        if edited_json is not None and edited_json != scenario:
            check = store.validate_scenario(edited_json)
            for e in check.errors:
                st.error(e)
            for w in check.warnings:
                st.warning(w)
            st.code(store.diff(scenario, edited_json), language="diff")
            if st.button(
                tr("驗證並儲存", "Validate & save"), type="primary", disabled=not check.ok
            ):
                store.save_scenario(edited_json, source="ui")
                st.toast(tr("已儲存", "Saved"))
                time.sleep(0.5)
                st.rerun()
        from local_llm.lab.nodes import describe_node_types

        st.caption(tr("節點類型與參數", "Node types and parameters"))
        st.json(describe_node_types(), expanded=False)

with tab_runs:
    runs_dir = PROJECT_DIR / "runs"
    files = sorted(runs_dir.glob(f"scenario-{sid}-*.json"), reverse=True)[:10]
    if not files:
        st.info(tr("還沒有執行紀錄。", "No runs yet."))
    for f in files:
        data = json.loads(f.read_text())
        icon = flow_ui.STATUS_ICON.get(data.get("status"), ":material/radio_button_unchecked:")
        with st.expander(
            f"{icon} {data.get('recorded_at', '')[:19]} · {data.get('seconds')}s · "
            + ", ".join(flow_ui.short_model(m) for m in data.get("models", []))
        ):
            st.markdown(
                f"**{tr('輸入', 'Input')}**: {json.dumps(data.get('inputs'), ensure_ascii=False)}"
            )
            st.markdown(data.get("output") or "")
            st.caption(Path(f).name)
