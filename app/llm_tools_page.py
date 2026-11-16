"""Tool-calling mode of the Language models page (views/llm.py): send one question with tools to
local models and see which tools they call and with what arguments.

The question bank is every single-turn question of our batch runs (English BFCL, MediaTek's zh-TW
BFCL, ACEBench zh; data/lab/tool_sets/, 1,198 questions, about 4 MB, loaded once per server
process), each with the expected answer and how every model did in the batch run. The pickers
narrow it to one suite and one category (at most 80 questions in a list, searchable), so the size
does not slow the page. A set downloaded from the Test sets page (views/datasets.py) can be
uploaded back, edited or not. Or write your own question with the platform's tools or your own
tool definitions.
Run as a script by views/llm.py (runpy), like a page of its own.
"""

import json

import i18n
import jobs
import streamlit as st
from common import installed_names, kept, memory_guard, model_label, ollama_url, save_run
from i18n import tr
from langchain_core.utils.function_calling import convert_to_openai_tool

from local_llm import catalog, ollama_admin, tool_test

MAX_MODELS = 4
lang = i18n.current()

SUITES = {
    "bfcl_en": (
        tr("英文 BFCL", "English BFCL"),
        tr(
            "Berkeley Function Calling Leaderboard v4，英文；live 開頭的是使用者實際送來的題目。",
            "Berkeley Function Calling Leaderboard v4 (English); live categories are real user "
            "questions.",
        ),
    ),
    "bfcl_zhtw": (
        tr("繁中 BFCL", "zh-TW BFCL"),
        tr(
            "聯發科翻譯的繁中版 BFCL，和英文是同一批題目，可以切換對照；工具說明仍是英文。",
            "MediaTek's Traditional Chinese BFCL: the same questions as English, so you can switch "
            "between them; tool descriptions stay in English.",
        ),
    ),
    "acebench": (
        tr("ACEBench 中文", "ACEBench (Chinese)"),
        tr(
            "ACEBench 簡體中文版；special 類測缺參數、參數錯誤、"
            "工具做不到時會不會說出來而不是亂呼叫。",
            "ACEBench in simplified Chinese; the special categories test whether the model "
            "points out "
            "missing or wrong parameters, or that no tool fits, instead of calling one anyway.",
        ),
    ),
}
CATEGORIES = {
    "simple_python": tr("單一工具（1 個工具、呼叫 1 次）", "One tool, one call"),
    "multiple": tr("從 2～4 個工具挑一個", "Pick one of 2–4 tools"),
    "parallel": tr("同一工具一次呼叫多次", "Same tool, several calls at once"),
    "parallel_multiple": tr("多個工具一次呼叫", "Several tools at once"),
    "irrelevance": tr("不該呼叫（工具都不相關）", "Should not call (no tool fits)"),
    "live_simple": tr("真實提問：單一工具", "Real question: one tool"),
    "live_multiple": tr("真實提問：從多個工具挑", "Real question: pick among tools"),
    "live_parallel": tr("真實提問：一次呼叫多次", "Real question: several calls"),
    "live_parallel_multiple": tr("真實提問：多個工具一次呼叫", "Real question: several tools"),
    "live_irrelevance": tr("真實提問：不該呼叫", "Real question: should not call"),
    "live_relevance": tr("真實提問：該呼叫時要呼叫", "Real question: should call"),
    "normal_single_turn_single_function": tr("單一工具", "One tool"),
    "normal_single_turn_parallel_function": tr("一次呼叫多個", "Several calls at once"),
    "normal_similar_api": tr("相似工具二選一", "Two similar tools"),
    "normal_atom_enum": tr("參數要從選項中挑", "Parameter from a fixed list"),
    "normal_preference": tr("依使用者資料填參數", "Fill parameters from the user profile"),
    "special_incomplete": tr("缺參數：要追問", "Missing parameter: ask"),
    "special_error_param": tr("參數值有錯：要指出", "Wrong value: point it out"),
    "special_irrelevant": tr("工具做不到：要說明", "No tool fits: say so"),
    "normal_atom_bool": tr("參數是是非值", "Yes/no parameter"),
    "normal_atom_number": tr("參數是數字", "Number parameter"),
    "normal_atom_list": tr("參數是清單", "List parameter"),
    "normal_atom_object_short": tr("參數是物件（淺）", "Object parameter (shallow)"),
    "normal_atom_object_deep": tr("參數是物件（多層）", "Object parameter (nested)"),
    "normal_multi_turn_user_adjust": tr(
        "對話中使用者改需求", "User changes the request mid-dialogue"
    ),
    "normal_multi_turn_user_switch": tr("對話中使用者換話題", "User switches topic mid-dialogue"),
    "uploaded": tr("上傳的題目", "Uploaded questions"),
}
REASONS = {
    "called": tr("不該呼叫卻呼叫了：", "Called a tool but should not have: "),
    "no_call": tr("沒有呼叫工具", "No tool call"),
    "count": tr("呼叫數量不對（實際／應有）：", "Wrong number of calls (made / expected): "),
    "mismatch": tr("呼叫內容不對：", "Wrong call: "),
    "not_mentioned": tr("回覆沒有指出：", "The reply does not mention: "),
}
CUSTOM_TOOLS_EXAMPLE = json.dumps(
    [
        {
            "type": "function",
            "function": {
                "name": "book_meeting_room",
                "description": "預約會議室",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "room": {"type": "string", "description": "會議室名稱，例如 101、大會議室"},
                        "date": {"type": "string", "description": "日期，格式 YYYY-MM-DD"},
                        "start": {"type": "string", "description": "開始時間，格式 HH:MM"},
                        "hours": {"type": "number", "description": "使用幾小時"},
                    },
                    "required": ["room", "date", "start"],
                },
            },
        }
    ],
    ensure_ascii=False,
    indent=2,
)

st.caption(
    tr(
        "把一個問題和一組工具交給地端模型，看它呼叫哪個工具、填了什麼參數（工具不會真的執行）。"
        "題庫是我們實測三組題目的全部單輪題（1,198 題），附標準答案和每個模型在批次測試裡的對錯；"
        "★ 是各類型挑出來、模型表現分歧的題目。也可以上傳從「測試資料集」下載的題組，或自己出題。"
        "多輪題要模擬環境執行工具，這裡沒有收；"
        "完整結果見 HackMD 報告〈地端語言模型工具呼叫實測〉。",
        "Give local models one question and a set of tools and see which tool they call with which "
        "arguments (tools are not executed). The bank is every single-turn question of the three "
        "suites we ran (1,198), with the expected answer and how each model did in the batch run; "
        "★ marks questions picked per category where the models disagree. You can also upload a "
        "set downloaded from Test sets, or write your own. Multi-turn cases need a simulated "
        "environment and are not included here.",
    )
)


@st.cache_data(ttl=300)
def tool_models(url: str, names: tuple[str, ...]) -> list[str]:
    return [n for n in names if catalog.is_chat(n) and ollama_admin.supports_tools(n, url)]


@st.cache_data
def samples() -> list[dict]:
    return tool_test.load_samples()


def pick_question(pool: list[dict], suites: dict, prefix: str) -> dict:
    """Suite -> category -> question pickers over `pool`; returns the chosen question."""
    names = list(dict.fromkeys(s["suite"] for s in pool))
    if st.session_state.get(f"{prefix}_suite") not in names:
        st.session_state[f"{prefix}_suite"] = "bfcl_zhtw" if "bfcl_zhtw" in names else names[0]
    c1, c2 = st.columns([2, 3])
    suite = (
        c1.segmented_control(
            tr("題庫", "Suite"),
            names,
            format_func=lambda s: suites.get(s, (s,))[0],
            **kept(f"{prefix}_suite"),
        )
        or names[0]
    )
    in_suite = [s for s in pool if s["suite"] == suite]
    cats = list(dict.fromkeys(s["category"] for s in in_suite))
    count = {c: sum(s["category"] == c for s in in_suite) for c in cats}
    if st.session_state.get(f"{prefix}_cat") not in cats:
        st.session_state[f"{prefix}_cat"] = cats[0]
    cat = c2.selectbox(
        tr("類型", "Category"),
        cats,
        format_func=lambda c: f"{CATEGORIES.get(c, c)}（{count[c]}）",
        **kept(f"{prefix}_cat"),
    )
    if suite in suites:
        st.caption(suites[suite][1])
    in_cat = [s for s in in_suite if s["category"] == cat]
    in_cat.sort(key=lambda s: not s.get("featured"))  # stable: featured first, then file order
    by_key = {s["key"]: s for s in in_cat}
    keys = list(by_key)
    if st.session_state.get(f"{prefix}_key") not in keys:
        st.session_state[f"{prefix}_key"] = keys[0]
    key = st.selectbox(
        tr(f"題目（{len(keys)} 題，可打字搜尋）", f"Question ({len(keys)}, type to search)"),
        keys,
        format_func=lambda k: (
            ("★ " if by_key[k].get("featured") else "")
            + f"{keys.index(k) + 1}. "
            + last_user(by_key[k]["messages"]).replace("\n", " ")[:70]
        ),
        **kept(f"{prefix}_key"),
    )
    return by_key[key]


def platform_tools() -> dict:
    from local_llm.agents import all_tools

    return all_tools()


def last_user(messages: list[dict]) -> str:
    return next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")


def expected_text(expect: dict) -> str:
    kind = expect["kind"]
    if kind == "no_call":
        return tr(
            "不呼叫任何工具，直接回答或說明做不到。",
            "Call no tool; answer or say it cannot be done.",
        )
    if kind == "any_call":
        return tr("要呼叫工具（呼叫哪一個都算）。", "Call a tool (any tool counts).")
    if kind == "mention":
        return tr(
            "不呼叫工具，回覆裡要指出：", "Call no tool; the reply must point out: "
        ) + "、".join(expect["words"])
    lines = []
    for call in expect["calls"]:
        ((name, args),) = call.items()
        if expect.get("style") == "bfcl":
            shown = {
                k: [x for x in v if x != ""] or tr("（可省略）", "(optional)")
                for k, v in args.items()
            }
            optional = [k for k, v in args.items() if "" in v]
        else:
            shown, optional = args, []
        lines.append(
            f"{name}({json.dumps(shown, ensure_ascii=False)})"
            + (tr("　可省略：", "  optional: ") + ", ".join(optional) if optional else "")
        )
    note = (
        tr("（列表是可接受的值，任一個都算對）", "(lists are accepted alternatives)")
        if expect.get("style") == "bfcl"
        else ""
    )
    return "\n".join(lines) + ("\n" + note if note else "")


# ------------------------------------------------------------------ choose the question

mode = (
    st.segmented_control(
        tr("出題方式", "Question"),
        ["sample", "upload", "custom"],
        label_visibility="collapsed",
        format_func=lambda m: {
            "sample": ":material/quiz: " + tr("題庫", "Question bank"),
            "upload": ":material/upload_file: " + tr("上傳題組", "Upload a set"),
            "custom": ":material/edit: " + tr("自己出題", "Your own question"),
        }[m],
        **kept("tools_mode"),
    )
    or "sample"
)

sample = None
if mode == "upload":
    up = st.file_uploader(
        tr(
            "題組檔（JSON，從「測試資料集」下載的格式；可以刪改題目再上傳）",
            "Test set (JSON in the format downloaded from Test sets; edit it if you like)",
        ),
        type=["json"],
        key="tools_upload",
    )
    if up is not None and st.session_state.get("tools_upload_id") != up.file_id:
        try:
            st.session_state.tools_uploaded = tool_test.parse_set(up.getvalue())
            st.session_state.tools_upload_id = up.file_id
        except ValueError as e:
            st.session_state.pop("tools_uploaded", None)
            st.error(tr(f"無法讀取這個檔案：{e}", f"Cannot read this file: {e}"))
    got = st.session_state.get("tools_uploaded")
    if up is None or not got:
        st.info(
            tr(
                "到左側選單「測試資料集」下載題組，或照同樣格式自己寫："
                '`{"samples": [{"messages": [...], "tools": [...], "expect": {...}}]}`，'
                "expect 可省略（省略時只看有沒有呼叫工具）。",
                "Download a set from Test sets in the menu, or write one in the same format: "
                '`{"samples": [{"messages": [...], "tools": [...], "expect": {...}}]}`; '
                "expect is optional (without it, any tool call counts).",
            ),
            icon=":material/info:",
        )
        st.stop()
    st.caption(
        tr(
            f"已載入：{got['name']}，{len(got['samples'])} 題",
            f"Loaded: {got['name']}, {len(got['samples'])} questions",
        )
    )
    sample = pick_question(got["samples"], SUITES, "tools_up")
elif mode == "sample":
    all_samples = samples()
    if not all_samples:
        st.warning(
            tr(
                "找不到 data/lab/tool_sets/ 的題組檔。",
                "The question files in data/lab/tool_sets/ are missing.",
            )
        )
        st.stop()
    sample = pick_question(all_samples, SUITES, "tools")
if sample is not None:
    suite, cat = sample["suite"], sample["category"]
    messages, tools = sample["messages"], sample["tools"]

    with st.container(border=True):
        for m in messages:
            if m["role"] == "system":
                with st.expander(
                    tr("系統提示（批次測試用的原文）", "System prompt (as used in the batch run)")
                ):
                    st.text(m["content"])
            else:
                st.markdown(f"**{m['role']}**：{m['content']}")
        if sample.get("pair") and mode == "sample":
            other = "bfcl_en" if suite == "bfcl_zhtw" else "bfcl_zhtw"
            if st.button(
                tr(f"看同一題的{SUITES[other][0]}版", f"Same question in {SUITES[other][0]}"),
                icon=":material/translate:",
            ):
                st.session_state.tools_suite, st.session_state.tools_cat = other, cat
                st.session_state.tools_key = sample["pair"]
                st.rerun()
    a, b = st.columns(2)
    with a.expander(
        tr(f"提供給模型的工具（{len(tools)} 個）", f"Tools given to the model ({len(tools)})")
    ):
        st.code(json.dumps(tools, ensure_ascii=False, indent=2), "json")
    with b.expander(tr("標準答案", "Expected answer"), expanded=True):
        st.text(expected_text(sample["expect"]))
    if sample["batch"]:
        with st.expander(tr("各模型在批次測試裡的結果", "How each model did in the batch run")):
            st.caption(
                tr(
                    "批次測試：溫度 0.001、輸出上限 8192 token，地端開啟思考；雲端是各家預設設定。",
                    "Batch run: temperature 0.001, 8,192-token cap, thinking on for local models; "
                    "cloud "
                    "models at their defaults.",
                )
            )
            st.dataframe(
                [
                    {
                        tr("模型", "Model"): m,
                        tr("結果", "Result"): "✅" if r["ok"] else "❌",
                        tr("錯在哪", "Why"): r.get("why", ""),
                    }
                    for m, r in sample["batch"].items()
                ],
                hide_index=True,
                width="stretch",
            )
else:
    question = st.text_area(
        tr("問題", "Question"),
        tr(
            "幫我訂明天下午兩點的 101 會議室，用兩個小時。",
            "Book room 101 for two hours from 2 pm tomorrow.",
        ),
        height=80,
        **kept("tools_question"),
    )
    source = st.radio(
        tr("工具", "Tools"),
        ["platform", "json"],
        horizontal=True,
        format_func=lambda s: {
            "platform": tr("平台上的工具", "Platform tools"),
            "json": tr("自訂工具（JSON）", "Your own tools (JSON)"),
        }[s],
        **kept("tools_source"),
    )
    tools = []
    if source == "platform":
        available = platform_tools()
        st.session_state.setdefault(
            "tools_pick",
            [n for n in ("calculator", "current_time", "get_weather") if n in available],
        )
        picked = st.multiselect(
            tr("要給模型的工具（越多越難挑）", "Tools to offer (more tools, harder choice)"),
            list(available),
            **kept("tools_pick"),
        )
        tools = [convert_to_openai_tool(available[n]) for n in picked]
    else:
        raw = st.text_area(
            tr("工具定義（OpenAI 格式的 JSON 陣列）", "Tool definitions (OpenAI-style JSON array)"),
            CUSTOM_TOOLS_EXAMPLE,
            height=260,
            **kept("tools_json"),
        )
        try:
            tools = json.loads(raw)
            assert isinstance(tools, list)
        except (json.JSONDecodeError, AssertionError):
            st.error(tr("JSON 格式不對，要是一個陣列。", "Not valid JSON: it must be an array."))
            tools = []
    system = st.text_input(
        "System prompt",
        tr(
            "今天是 {today}。需要時呼叫工具；資訊不足時先問使用者。",
            "Today is {today}. Call tools when needed; ask the user when information is missing.",
        ),
        **kept("tools_system"),
    )
    from datetime import date

    messages = (
        [{"role": "system", "content": system.replace("{today}", date.today().isoformat())}]
        if system
        else []
    ) + [{"role": "user", "content": question}]

# ------------------------------------------------------------------ models and settings

url = ollama_url()
names = installed_names()
options = tool_models(url, tuple(names))
installed = {catalog.normalize(m) for m in names}
default = [m for m in [catalog.default_model()] if m in options]
st.session_state.setdefault("tools_models", default)
st.session_state.tools_models = [m for m in st.session_state.tools_models if m in options]
selected = st.multiselect(
    tr(
        f"模型（最多 {MAX_MODELS} 個；只列出支援工具呼叫、已下載的模型）",
        f"Models (up to {MAX_MODELS}; downloaded models that support tool calling)",
    ),
    options,
    format_func=lambda t: model_label(t, installed),
    max_selections=MAX_MODELS,
    **kept("tools_models"),
)
c1, c2, c3 = st.columns(3)
think = c1.toggle(
    tr("允許思考", "Allow thinking"),
    value=True,
    help=tr(
        "批次測試是開著思考跑的；一定會思考的模型關不掉。",
        "The batch run had thinking on; some models always think.",
    ),
)
temp_mode = c2.selectbox(
    tr("溫度", "Temperature"),
    ["zero", "default"],
    format_func=lambda t: {
        "zero": tr("0（接近批次測試）", "0 (like the batch run)"),
        "default": tr("模型預設（原廠建議）", "Model default (vendor's)"),
    }[t],
    help=tr(
        "Qwen 官方說思考模式不要用接近 0 的溫度，否則容易想不停。",
        "Qwen advises against near-zero temperature with thinking: it can loop.",
    ),
)
max_tokens = c3.select_slider(
    tr("輸出上限 tokens", "Max output tokens"), [2048, 4096, 8192], value=8192
)
memory_guard(selected)


def work(job: jobs.Job, base_url: str) -> None:
    s = job.state
    for i, r in enumerate(s["results"]):
        if job.cancel.is_set():
            break
        r["status"] = "running"
        job.detail, job.progress = (
            f"{i + 1}/{len(s['results'])} {r['model']}",
            i / len(s["results"]),
        )
        try:
            out = tool_test.call_ollama(
                base_url,
                r["model"],
                s["messages"],
                s["tools"],
                think=s["think"],
                temperature=0.0 if s["temp_mode"] == "zero" else None,
                max_tokens=s["max_tokens"],
            )
            r.update(out)
            if s["expect"]:
                r["verdict"] = tool_test.check(s["expect"], out["calls"], out["content"])
        except Exception as exc:  # noqa: BLE001 - shown to the user
            r["error"] = str(exc)
        r["status"] = "done"
        if s["unload"]:
            ollama_admin.unload(r["model"], base_url)
    path = save_run(
        "toolcall",
        {k: s[k] for k in ("key", "messages", "tools", "think", "temp_mode", "max_tokens")}
        | {"results": s["results"]},
    )
    s["path"] = str(path.relative_to(path.parents[1]))


def render(job: jobs.Job) -> None:
    s = job.state
    cols = st.columns(len(s["results"]))
    for col, r in zip(cols, s["results"], strict=True):
        with col.container(border=True):
            st.markdown(f"**{r['model']}**")
            if r["status"] != "done":
                st.caption(
                    tr("等待中…", "Waiting…")
                    if r["status"] == "waiting"
                    else tr("執行中…", "Running…")
                )
                continue
            if r.get("error"):
                st.error(r["error"])
                continue
            v = r.get("verdict")
            if v:
                if v["ok"]:
                    st.success(tr("答對", "Correct"), icon=":material/check_circle:")
                else:
                    st.error(
                        REASONS.get(v["code"], v["code"]) + v["detail"], icon=":material/cancel:"
                    )
            if r["calls"]:
                st.code(json.dumps(r["calls"], ensure_ascii=False, indent=1), "json")
            else:
                st.caption(tr("沒有呼叫工具", "No tool call"))
            if r["content"]:
                st.markdown(r["content"])
            if r.get("hit_limit"):
                st.warning(tr("達到輸出上限，沒有想完。", "Hit the output limit before finishing."))
            if r["thinking"]:
                with st.expander(
                    tr(
                        f"思考過程（{len(r['thinking'])} 字）",
                        f"Thinking ({len(r['thinking'])} chars)",
                    )
                ):
                    st.text(r["thinking"])
            st.caption(
                tr(
                    f"{r['total_s']} 秒 · 輸入 {r['input_tokens']} · "
                    f"輸出 {r['output_tokens']} tokens",
                    f"{r['total_s']} s · {r['input_tokens']} in · {r['output_tokens']} out tokens",
                )
            )
    if job.running:
        a, b = st.columns([5, 1])
        a.info(
            f"{job.detail or tr('準備中', 'Preparing')} · {job.elapsed():.0f}s",
            icon=":material/progress_activity:",
        )
        if b.button(tr("停止", "Stop"), key=f"stop_{job.id}"):
            job.cancel.set()
    elif job.error:
        st.error(job.error)
    elif s.get("path"):
        st.caption(tr(f"已儲存：`{s['path']}`", f"Saved: `{s['path']}`"))


@st.fragment(run_every=1)
def live(job: jobs.Job) -> None:
    if not job.running:
        st.rerun()
    render(job)


job = jobs.get(st.session_state.get("tools_job"))
busy = bool(job and job.running)
if st.button(
    tr("開始測試", "Run"),
    type="primary",
    icon=":material/play_arrow:",
    disabled=busy or not selected or not tools or not last_user(messages).strip(),
):
    job = jobs.Job(
        "toolcall",
        tr(f"工具呼叫測試 {len(selected)} 個模型", f"Tool-call test, {len(selected)} model(s)"),
        "views/llm.py",
        state={
            "key": sample["key"] if sample else None,
            "messages": messages,
            "tools": tools,
            "expect": sample["expect"] if sample else None,
            "think": think,
            "temp_mode": temp_mode,
            "max_tokens": max_tokens,
            "unload": sum((catalog.find(m) or {}).get("size_gb", 0) for m in selected) > 40,
            "results": [{"model": m, "status": "waiting"} for m in selected],
        },
    )
    jobs.start(job, lambda j: work(j, url))
    st.session_state.tools_job = job.id
    st.rerun()

if job and job.running:
    live(job)
elif job:
    render(job)
