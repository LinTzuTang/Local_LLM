import base64

import jobs
import streamlit as st
import ui
from common import (
    kept,
    model_picker,
    ollama_backend,
    require_ollama,
    stats_caption,
    think_control,
)
from i18n import current, tr

from local_llm import catalog
from local_llm.streaming import stream_chat

ui.page_header(
    "chat",
    tr("對話測試", "Chat"),
    tr(
        "直接和一個地端模型對話：看它的回答品質、思考過程與速度（首字時間、tokens/s）。",
        "Talk to one local model directly: answer quality, reasoning and speed "
        "(time to first token, tokens/s).",
    ),
    eyebrow=tr("模型", "Models"),
)
require_ollama()
lang = current()


def setting(key: str, default) -> dict:
    """kept() with a default, so the setting survives page switches without a value= clash."""
    st.session_state.setdefault(key, default)
    return kept(key)


# --- settings bar (main area, no sidebar) ---------------------------------------------------
with st.container(border=True):
    bar = st.columns([3.2, 1.3, 1.2, 1.2, 1.1], vertical_alignment="bottom", gap="small")
    model = model_picker(tr("模型", "Model"), key="chat_model", container=bar[0])
    with bar[1]:
        think = think_control(model, key="chat_think", compact=True)
    is_vision = bool(model and "vision" in ((catalog.find(model) or {}).get("caps") or []))
    with bar[2].popover(tr("進階設定", "Settings"), icon=":material/tune:", width="stretch"):
        system = st.text_area(
            "System prompt",
            height=90,
            **setting(
                f"chat_system_{lang}",
                tr("你是地端助理，請用繁體中文回答。",
                   "You are an on-premises assistant. Answer in English."),
            ),
        )  # fmt: skip
        temperature = st.slider("Temperature", 0.0, 1.5, step=0.1, **setting("chat_temp", 0.7))
        max_tokens = st.select_slider(
            tr("最大輸出 tokens", "Max output tokens"),
            [512, 1024, 2048, 4096, 8192],
            **setting("chat_max_tokens", 4096),
        )
        num_ctx = st.select_slider(
            tr("Context 長度", "Context length"),
            [4096, 8192, 16384, 32768],
            **setting("chat_num_ctx", 16384),
        )
    with bar[3].popover(
        tr("附加圖片", "Image"),
        icon=":material/image:",
        width="stretch",
        disabled=not is_vision,
        help=None if is_vision else tr("所選模型不支援看圖", "The selected model has no vision"),
    ):
        upload = st.file_uploader(
            tr("附加圖片（視覺模型）", "Attach an image (vision models)"),
            type=["png", "jpg", "jpeg"],
            key="chat_image",
        )
    image = upload if is_vision else None
    if bar[4].button(tr("清除", "Clear"), icon=":material/delete_sweep:", width="stretch"):
        st.session_state.chat_history = []
        st.session_state.pop("chat_job", None)  # a running reply finishes in the background
    if image:
        st.caption(
            tr(f"下一則訊息會附上圖片：{image.name}", f"The next message includes {image.name}")
        )

if not model:
    st.info(
        tr(
            "請先在上方選擇已下載的模型，或下載所選模型。",
            "Pick a downloaded model above, or download the selected one.",
        ),
        icon=":material/info:",
    )
    st.stop()


def reply_stats(msg: dict) -> str:
    """Caption under a reply, built on the page so it follows the current language."""
    stats = msg.get("stats")
    if isinstance(stats, str):  # replies kept from before this format
        return stats
    return f"{msg.get('model', '')} · {stats_caption(stats)}" if msg.get("model") else ""


history = st.session_state.setdefault("chat_history", [])
job = jobs.get(st.session_state.get("chat_job"))
if job and not job.running:
    # The reply finished, possibly while the user was on another page.
    history.append(job.state["reply"])
    del st.session_state.chat_job
    job = None
for msg in history:
    with st.chat_message(msg["role"]):
        if msg.get("thinking"):
            with st.expander(tr("思考過程", "Thinking")):
                st.markdown(msg["thinking"])
        st.markdown(msg["content"])
        if caption := reply_stats(msg):
            st.caption(caption)


def reply_work(
    job: jobs.Job, backend, model: str, messages: list[dict], options: dict, text: dict
) -> None:
    """Worker thread: streams the reply into job.state (no Streamlit calls).

    `text` holds the status strings, translated on the page.
    """
    state, stats = job.state, None
    try:
        for event in stream_chat(backend, model, messages, **options):
            if job.cancel.is_set():
                break  # closing the stream makes Ollama stop generating
            if event.kind == "thinking":
                state["thinking"] += event.text
                job.detail = text["thinking"].format(n=len(state["thinking"]))
            elif event.kind == "content":
                state["answer"] += event.text
                job.detail = text["answering"].format(n=len(state["answer"]))
            else:
                stats = event.stats
    except Exception as exc:
        state["answer"] += f"\n\n**{type(exc).__name__}**: {exc}"
    answer, thinking = state["answer"], state["thinking"]
    if job.cancel.is_set():
        answer += text["stopped"]
    elif not answer and thinking:
        answer = text["cut_off"]
    state["reply"] = {
        "role": "assistant",
        "content": answer,
        "thinking": thinking,
        "model": model,
        "stats": stats,
    }


@st.fragment(run_every=0.5)
def live_reply(job: jobs.Job) -> None:
    if not job.running:
        st.rerun()
    with st.chat_message("assistant"):
        thinking, answer = job.state["thinking"], job.state["answer"]
        if thinking:
            with st.expander(tr("思考中…", "Thinking…"), expanded=not answer):
                st.markdown(thinking[-3000:])
        st.markdown(answer + "▌" if answer else "")
        a, b = st.columns([5, 1])
        a.caption(
            f"{job.state['model']} · {job.elapsed():.0f}s · "
            + tr(
                "可切換到其他頁面，回答會在背景繼續",
                "you can switch pages; the reply continues in the background",
            )
        )
        if b.button(tr("停止", "Stop"), key=f"stop_{job.id}", icon=":material/stop_circle:"):
            job.cancel.set()


if job:
    live_reply(job)

if prompt := st.chat_input(tr("輸入訊息…", "Type a message…"), disabled=job is not None):
    history.append({"role": "user", "content": prompt})
    messages = [{"role": "system", "content": system}] + [
        {"role": m["role"], "content": m["content"]} for m in history
    ]
    images = [base64.b64encode(image.getvalue()).decode()] if image else None
    options = dict(
        think=think,
        temperature=temperature,
        max_tokens=max_tokens,
        num_ctx=num_ctx,
        images=images,
    )
    text = {
        "thinking": tr("思考中（{n} 字）", "thinking ({n} chars)"),
        "answering": tr("回答中（{n} 字）", "answering ({n} chars)"),
        "stopped": tr("\n\n（已停止）", "\n\n(stopped)"),
        "cut_off": tr(
            "（思考尚未結束就達到輸出上限；請調高「最大輸出 tokens」。）",
            "(Hit the output limit before thinking finished; raise “Max output tokens”.)",
        ),
    }
    job = jobs.Job(
        "chat",
        tr(f"對話：{prompt[:12]}", f"Chat: {prompt[:16]}"),
        "views/chat.py",
        state={"model": model, "thinking": "", "answer": ""},
    )
    backend = ollama_backend()
    jobs.start(job, lambda j: reply_work(j, backend, model, messages, options, text))
    st.session_state.chat_job = job.id
    st.rerun()
