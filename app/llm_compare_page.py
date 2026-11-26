"""Compare mode of the Language models page (views/llm.py): one prompt to several models,
side by side. Run as a script by views/llm.py (runpy), like a page of its own.
"""

import json
import re

import access
import i18n
import jobs
import pandas as pd
import streamlit as st
from common import (
    cloud_enabled,
    cloud_hint,
    installed_names,
    kept,
    memory_guard,
    model_label,
    ollama_backend,
    ollama_url,
    save_run,
    stats_caption,
)
from i18n import tr

from local_llm import catalog, cloud_llm, ollama_admin
from local_llm.streaming import collect, stream_chat

st.caption(tr(
    "同一個問題依序丟給多個語言模型（一次跑一個，速度數字才公平），並排比較答案與速度。"
    "比較在背景執行，可以切換到其他頁面，回來再看結果。",
    "One prompt goes to several language models in turn (one at a time, so the speed numbers "
    "are fair); compare answers and speed side by side. It runs in the background, so you can "
    "switch pages and come back for the results.",
))  # fmt: skip

# Prompt templates: id -> (label, prompt) per language.
PRESETS = {
    "custom": {"zh": ("自訂", ""), "en": ("Custom", "")},
    "writing": {
        "zh": (
            "繁中寫作",
            "請用繁體中文寫一段 150 字左右的公告，通知同仁下週一起改用地端 AI 助理。",
        ),
        "en": (
            "Writing",
            "Write a short announcement (about 100 words) telling colleagues that from next "
            "Monday they will use the on-premises AI assistant.",
        ),
    },
    "reasoning": {
        "zh": (
            "推理",
            "一個水池有甲乙兩管，甲管單獨注滿要 6 小時，乙管單獨排空要 9 小時。同時打開，幾小時注滿？請說明過程。",
        ),
        "en": (
            "Reasoning",
            "A pool has two pipes. Pipe A alone fills it in 6 hours; pipe B alone drains it in "
            "9 hours. With both open, how many hours does it take to fill? Show your working.",
        ),
    },
    "code": {
        "zh": (
            "程式碼",
            "用 Python 寫一個函式，找出字串中最長的不重複字元子字串，並附上 3 個測試。",
        ),
        "en": (
            "Code",
            "Write a Python function that finds the longest substring without repeating "
            "characters, with 3 tests.",
        ),
    },
    "summary": {
        "zh": (
            "摘要",
            "請用三點條列摘要：地端部署 LLM 可以保護資料隱私、降低長期 API 成本、離線可用，但需要自行維護硬體、模型更新與資安。",
        ),
        "en": (
            "Summary",
            "Summarise in three bullet points: running LLMs on-premises protects data privacy, "
            "lowers long-term API costs and works offline, but you must maintain the hardware, "
            "model updates and security yourself.",
        ),
    },
    "knowledge": {
        "zh": ("知識", "台灣的國定假日有哪些？請列出日期。"),
        "en": ("Knowledge", "What are Taiwan's national holidays? List their dates."),
    },
}

MAX_MODELS = 6


models = installed_names()
installed = {catalog.normalize(m) for m in models}
chat_models = [m for m in models if catalog.is_chat(m)]
# Installed models first, then every catalog chat model that fits this machine.
options = chat_models + [
    m["tag"]
    for m in catalog.models()
    if catalog.is_chat(m["tag"])
    and catalog.fit(m) != "no"
    and catalog.normalize(m["tag"]) not in installed
]
cloud_models = cloud_llm.models() if cloud_enabled() else []  # the top bar's cloud switch
options += cloud_models
st.session_state.setdefault(
    "compare_models", [m for m in [catalog.default_model()] if m in options]
)
st.session_state.compare_models = [m for m in st.session_state.compare_models if m in options]
selected = st.multiselect(
    tr(
        f"要比較的模型（最多 {MAX_MODELS} 個；⬇️ 表示尚未下載，開始比較前會自動下載）",
        f"Models to compare (up to {MAX_MODELS}; ⬇️ = not downloaded yet, fetched before the "
        "comparison starts)",
    ),
    options,
    format_func=lambda t: (
        cloud_llm.label(t) if cloud_llm.is_cloud(t) else model_label(t, installed)
    ),
    max_selections=MAX_MODELS,
    **kept("compare_models"),
)
cloud_hint("語言模型（Claude、GPT、Gemini）", "language models (Claude, GPT, Gemini)")
if picked_cloud := [m for m in selected if cloud_llm.is_cloud(m)]:
    st.warning(
        tr(
            f"{'、'.join(cloud_llm.label(m) for m in picked_cloud)} 是雲端模型：問題與系統提示會送到"
            "該服務（AWS、Azure、Google），請不要貼公司機密或個資。用量會依帳號記錄。",
            f"{', '.join(cloud_llm.label(m) for m in picked_cloud)}: cloud models; the prompt and "
            "system prompt go to that service (AWS, Azure, Google), so no confidential data or "
            "personal data. Use is logged per account.",
        ),
        icon=":material/cloud_upload:",
    )
local_selected = [m for m in selected if not cloud_llm.is_cloud(m)]
missing = [m for m in local_selected if catalog.normalize(m) not in installed]
if missing:
    size = sum((catalog.find(m) or {}).get("size_gb", 0) for m in missing)
    st.info(
        tr(
            f"需要先下載 {len(missing)} 個模型（約 {size:.0f}GB）：{'、'.join(missing)}",
            f"{len(missing)} model(s) to download first (about {size:.0f}GB): {', '.join(missing)}",
        )
    )
nc = [m for m in selected if "非商用" in ((catalog.find(m) or {}).get("license") or "")]
if nc:
    st.warning(
        tr(
            f"{'、'.join(nc)} 是 CC-BY-NC（非商用）授權，只能用於評估與研究，不能用在商業服務。",
            f"{', '.join(nc)} {'is' if len(nc) == 1 else 'are'} licensed CC-BY-NC "
            "(non-commercial): evaluation and research only, not for commercial services.",
        )
    )

lang = i18n.current()
preset = st.segmented_control(
    tr("題目範本", "Prompt templates"),
    list(PRESETS),
    default="writing",
    format_func=lambda k: PRESETS[k][lang][0],
)
prompt = st.text_area(tr("問題", "Prompt"), PRESETS[preset or "custom"][lang][1], height=100)
c1, c2, c3 = st.columns(3)
system = c1.text_input("System prompt", tr("請用繁體中文回答。", "Answer in English."))
think = c2.toggle(
    tr("允許思考（可關閉的模型才有效）", "Allow thinking (for models where it can be turned off)"),
    value=False,
)
max_tokens = c3.select_slider(
    tr("最大輸出 tokens", "Max output tokens"), [512, 1024, 2048, 4096, 8192], value=4096
)
unload_between = st.toggle(
    tr(
        "每個模型跑完就卸載（大模型比較時避免記憶體不足）",
        "Unload each model when it finishes (avoids running out of memory with large models)",
    ),
    value=sum((catalog.find(m) or {}).get("size_gb", 0) for m in local_selected) > 40,
)
judge = st.selectbox(
    tr(
        "裁判模型（選填，替答案評 1–10 分；需已下載）",
        "Judge model (optional; scores each answer 1–10; must be downloaded)",
    ),
    ["", *chat_models, *cloud_models],
    format_func=lambda m: (
        (cloud_llm.label(m) if cloud_llm.is_cloud(m) else m) or tr("不評分", "No judge")
    ),
)

memory_guard(local_selected)


def compare_work(job: jobs.Job, backend, base_url: str, text: dict) -> None:
    """Worker thread: download, run each model in turn, then judge (no Streamlit calls).

    `text` holds the status strings and the judge prompt, translated on the page.
    """
    state = job.state
    results = state["results"]
    for r in results:
        if not r["missing"] or job.cancel.is_set():
            continue
        r["status"] = "pulling"
        job.detail = text["pulling"].format(model=r["model"])
        try:
            for event in ollama_admin.pull(r["model"], base_url):
                if job.cancel.is_set():
                    break
                if event.get("total") and event.get("completed"):
                    r["pull"] = f"{event['completed'] / 1e9:.1f} / {event['total'] / 1e9:.1f} GB"
        except Exception as exc:
            r["answer"], r["status"] = text["pull_failed"].format(error=exc), "done"
    for i, r in enumerate(results, start=1):
        if job.cancel.is_set():
            break
        if r["status"] == "done":
            continue
        r["status"] = "running"
        job.detail = f"{i}/{len(results)} {r['model']}"
        job.progress = (i - 1) / len(results)
        messages = conversation(state, r["model"])
        try:
            for event in stream_chat(
                backend, r["model"], messages, think=state["think"], temperature=0.2,
                max_tokens=state["max_tokens"],
            ):  # fmt: skip
                if job.cancel.is_set():
                    break
                if event.kind == "thinking":
                    r["thinking"] += event.text
                elif event.kind == "content":
                    r["answer"] += event.text
                else:
                    r["stats"] = event.stats
        except Exception as exc:
            r["answer"] = text["failed"].format(error=exc)
        r["status"] = "done"
        # Not a model someone else (an evaluation, another account) is also using.
        if (state["unload_between"] and not cloud_llm.is_cloud(r["model"])
                and access.may_unload(r["model"], state.get("who"))):  # fmt: skip
            ollama_admin.unload(r["model"], base_url)

    judge = state["judge"]
    if judge and not job.cancel.is_set():
        for r in results:
            job.detail = text["judging"].format(judge=judge, model=r["model"])
            judge_prompt = text["judge_prompt"] + text["judge_input"].format(
                prompt=state["prompt"], answer=r["answer"]
            )
            _, verdict, _ = collect(
                stream_chat(
                    backend, judge, [{"role": "user", "content": judge_prompt}],
                    temperature=0, max_tokens=4096,
                )
            )  # fmt: skip
            match = re.search(r"\{.*\}", verdict, re.S)
            try:
                r["judge"] = json.loads(match.group(0)) if match else {"reason": verdict}
            except json.JSONDecodeError:
                r["judge"] = {"reason": verdict}

    turns = [*state.get("turns", []), {"prompt": state["prompt"], "results": results}]
    saved = [{"prompt": t["prompt"], "results": [record(r) for r in t["results"]]} for t in turns]
    path = save_run(
        "compare",
        {
            # the first question and its answers, as before conversations (the list reads these)
            "prompt": saved[0]["prompt"],
            "system": state["system"],
            "think": state["think"],
            "judge": judge or None,
            "stopped": job.cancel.is_set(),
            "results": saved[0]["results"],
            "turns": saved,  # every question and what each model answered, in order
        },
        path=state.get("file"),
    )
    state["file"] = path
    state["path"] = path.relative_to(path.parents[1])


def record(r: dict) -> dict:
    return {k: r.get(k) for k in ("model", "answer", "thinking", "judge")} | {
        "stats": r["stats"] and r["stats"].to_dict()
    }


def conversation(state: dict, model: str) -> list[dict]:
    """This model's own conversation: the system prompt, each earlier question with the answer this
    model gave (not another's), then the new question."""
    messages = [{"role": "system", "content": state["system"]}] if state["system"] else []
    for t in state.get("turns", []):
        mine = next((r for r in t["results"] if r["model"] == model), {})
        messages += [{"role": "user", "content": t["prompt"]},
                     {"role": "assistant", "content": mine.get("answer") or "（沒有回答）"}]  # fmt: skip
    return messages + [{"role": "user", "content": state["prompt"]}]


def answer_columns(results: list[dict]):
    """Three columns per row so up to six models stay readable."""
    return [c for i in range(0, len(results), 3) for c in st.columns(3)][: len(results)]


def earlier_turn(n: int, turn: dict) -> None:
    """A finished question of this conversation, with every model's answer."""
    st.markdown(f"#### :material/help: {tr(f'第 {n} 問', f'Q{n}')}：{turn['prompt']}")
    for col, r in zip(answer_columns(turn["results"]), turn["results"], strict=True):
        with col.container(border=True):
            st.markdown(f"**{cloud_llm.label(r['model'])}**")
            st.markdown(r["answer"] or tr("（沒有回答）", "(no answer)"))
            if r["stats"]:
                st.caption(stats_caption(r["stats"]))


def render(job: jobs.Job) -> None:
    state = job.state
    results = state["results"]
    turns = state.get("turns", [])
    for n, turn in enumerate(turns, start=1):
        earlier_turn(n, turn)
    head = f"{tr(f'第 {len(turns) + 1} 問', f'Q{len(turns) + 1}')}：" if turns else ""
    st.markdown(f"#### :material/help: {head}{state['prompt']}")
    for col, r in zip(answer_columns(results), results, strict=True):
        with col.container(border=True):
            st.markdown(f"**{cloud_llm.label(r['model'])}**")
            thought = len(r["thinking"])
            if r["status"] == "waiting":
                st.info(tr("等待中…", "Waiting…"))
            elif r["status"] == "pulling":
                st.info(tr("⬇️ 下載中… ", "⬇️ Downloading… ") + r.get("pull", ""))
            elif r["status"] == "running" and not r["answer"]:
                st.caption(tr(f"思考中…（{thought} 字）", f"Thinking… ({thought} chars)"))
            else:
                running = r["status"] == "running"
                st.markdown(
                    r["answer"] + ("▌" if running else "")
                    or tr(
                        "（達到輸出上限，沒有產生答案）",
                        "(Hit the output limit without producing an answer)",
                    )
                )
            if r["status"] == "done":
                if r["thinking"]:
                    with st.expander(
                        tr(f"思考過程（{thought} 字）", f"Thinking ({thought} chars)")
                    ):
                        st.markdown(r["thinking"])
                st.caption(stats_caption(r["stats"]))
    if job.running:
        a, b = st.columns([5, 1])
        a.info(
            f"{job.detail or tr('準備中', 'Preparing')} · {job.elapsed():.0f}s",
            icon=":material/progress_activity:",
        )
        if b.button(tr("停止", "Stop"), key=f"stop_{job.id}"):
            job.cancel.set()
        return
    if job.error:
        st.error(job.error)
    done = [r for r in results if r["stats"]]
    if not done:
        return
    model_col, total_col = tr("模型", "Model"), tr("總時間 (s)", "Total (s)")
    table = pd.DataFrame(
        [
            {
                model_col: cloud_llm.label(r["model"]),
                tr("首字 (s)", "First token (s)"): r["stats"].first_answer_s,
                "tokens/s": r["stats"].decode_tps,
                tr("輸出 tokens", "Output tokens"): r["stats"].output_tokens,
                total_col: r["stats"].total_s,
                tr("思考字數", "Thinking chars"): len(r["thinking"]),
                **(
                    {
                        tr("評分", "Score"): r["judge"].get("score"),
                        tr("理由", "Reason"): r["judge"].get("reason"),
                    }
                    if r.get("judge")
                    else {}
                ),
            }
            for r in done
        ]
    )
    st.subheader(tr("比較結果", "Results"))
    if table["tokens/s"].isna().all():  # cloud models only: no generation speed to show
        table = table.drop(columns="tokens/s")
    st.dataframe(table.round(2), hide_index=True, width="stretch")
    if "tokens/s" in table:
        chart_cols = st.columns(2)
        chart_cols[0].bar_chart(table, x=model_col, y="tokens/s", horizontal=True)
        chart_cols[1].bar_chart(table, x=model_col, y=total_col, horizontal=True)
    else:
        st.bar_chart(table, x=model_col, y=total_col, horizontal=True)
    if state.get("path"):
        st.caption(tr(f"已儲存：`{state['path']}`", f"Saved: `{state['path']}`"))


@st.fragment(run_every=1)
def live(job: jobs.Job) -> None:
    if not job.running:
        st.rerun()
    render(job)


def launch(job: jobs.Job) -> None:
    """Start a comparison (a new conversation or the next question of one) in the background."""
    text = {
        "pulling": tr("下載 {model}", "downloading {model}"),
        "pull_failed": tr("下載失敗：{error}", "Download failed: {error}"),
        "failed": tr("錯誤：{error}", "Error: {error}"),
        "judging": tr("{judge} 評分 {model}", "{judge} scoring {model}"),
        "judge_prompt": tr(
            "你是嚴格的評審。依據正確性、完整性、繁體中文表達，替下面的回答打 1 到 10 分。"
            '只輸出 JSON：{"score": 分數, "reason": "一句理由"}\n\n',
            "You are a strict judge. Score the answer below from 1 to 10 for correctness, "
            'completeness and clarity. Output only JSON: {"score": <score>, "reason": '
            '"<one-sentence reason>"}\n\n',
        ),
        "judge_input": tr(
            "問題：{prompt}\n\n回答：{answer}", "Question: {prompt}\n\nAnswer: {answer}"
        ),
    }
    backend, url = ollama_backend(), ollama_url()
    jobs.start(job, lambda j: compare_work(j, backend, url, text))
    st.session_state.compare_job = job.id
    st.rerun()


def follow_up(job: jobs.Job) -> None:
    """Under a finished comparison: ask every model the next question, each going on from its own
    earlier answers (same system prompt and settings)."""
    state = job.state
    if not any(r["stats"] for r in state["results"]):
        return
    with st.container(border=True):
        st.markdown(":material/forum: **" + tr("接著問", "Ask a follow-up") + "**")
        st.caption(tr(
            "每個模型接著自己前面的回答繼續對話，系統提示與設定不變；整段對話存成同一筆執行紀錄。"
            "要換模型或設定，請改上面的選項後按「開始比較」，會開新的對話。",
            "Each model goes on from its own earlier answers, with the same system prompt and "
            "settings; the whole conversation is one record in Run history. To change models or "
            "settings, change them above and press Compare: that starts a new conversation.",
        ))  # fmt: skip
        question = st.text_area(
            tr("追問", "Follow-up question"), key=f"follow_{job.id}", height=80,
            label_visibility="collapsed",
            placeholder=tr("例如：請改寫成更正式的語氣", "e.g. Make it more formal"),
        )  # fmt: skip
        if st.button(tr("問所有模型", "Ask every model"), type="primary", icon=":material/send:",
                     disabled=not question.strip(), key=f"ask_{job.id}"):  # fmt: skip
            turns = [
                *state.get("turns", []),
                {"prompt": state["prompt"], "results": state["results"]},
            ]
            n = len(state["results"])
            follow = jobs.Job(
                "compare",
                tr(f"追問 {n} 個模型（第 {len(turns) + 1} 問）",
                   f"Follow-up to {n} model{'' if n == 1 else 's'} (Q{len(turns) + 1})"),
                "views/llm.py",
                state={k: state[k] for k in ("system", "think", "max_tokens", "unload_between", "judge")}
                | {"who": access.viewer(), "turns": turns, "prompt": question.strip(),
                   "file": state.get("file"),
                   "results": [{"model": r["model"], "missing": False, "status": "waiting",
                                "answer": "", "thinking": "", "stats": None}
                               for r in state["results"]]},
            )  # fmt: skip
            launch(follow)


job = jobs.get(st.session_state.get("compare_job"))
busy = bool(job and job.running)
label = (
    tr("下載缺少的模型並開始比較", "Download missing models and compare")
    if missing
    else tr("開始比較", "Compare")
)
if st.button(label, type="primary", disabled=busy or not (selected and prompt)):
    job = jobs.Job(
        "compare",
        tr(
            f"比較 {len(selected)} 個模型",
            f"Compare {len(selected)} model{'' if len(selected) == 1 else 's'}",
        ),
        "views/llm.py",
        state={
            "prompt": prompt,
            "system": system,
            "think": think,
            "max_tokens": max_tokens,
            "unload_between": unload_between,
            "who": access.viewer(),  # the job thread cannot read the page's headers
            "judge": judge,
            "results": [
                {
                    "model": m,
                    "missing": m in missing,
                    "status": "waiting",
                    "answer": "",
                    "thinking": "",
                    "stats": None,
                }
                for m in selected
            ],
        },
    )
    launch(job)

if job and job.running:
    live(job)
elif job:
    render(job)
    follow_up(job)
