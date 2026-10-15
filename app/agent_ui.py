"""Rendering of agent steps (LLM calls and tool executions), shared by the agent pages."""

import i18n
import streamlit as st
from common import ollama_url
from i18n import tr
from langchain_core.messages import AIMessage, ToolMessage

from local_llm.agents import TOOL_INFO


def was_rejected(message: ToolMessage) -> bool:
    """A human declined the call, so the tool never ran (the middleware writes this text)."""
    return message.status == "error" or str(message.content).startswith("User rejected")


def render_step(message):
    if isinstance(message, AIMessage):
        meta, usage = message.response_metadata, message.usage_metadata or {}
        seconds = (meta.get("total_duration") or 0) / 1e9
        if message.tool_calls:
            names = [f"`{c['name']}`" for c in message.tool_calls]
            decision = tr("要求執行 " + "、".join(names), "calls " + ", ".join(names))
        else:
            decision = tr("直接回答", "answers directly")
        tokens_in, tokens_out = usage.get("input_tokens", "?"), usage.get("output_tokens", "?")
        st.markdown(
            ":material/psychology: "
            + tr("**LLM 呼叫**：", "**LLM call**: ")
            + f"`POST {ollama_url()}/api/chat` → {decision}  \n"
            f"<small>{meta.get('model', '')} · "
            + tr(
                f"輸入 {tokens_in} tokens（含工具定義）· 輸出 {tokens_out} tokens",
                f"{tokens_in} tokens in (incl. tool definitions) · {tokens_out} tokens out",
            )
            + f" · {seconds:.1f}s</small>",
            unsafe_allow_html=True,
        )
        if reasoning := message.additional_kwargs.get("reasoning_content"):
            with st.expander(tr("思考", "Thinking"), icon=":material/psychology:"):
                st.markdown(reasoning)
        for call in message.tool_calls:
            label = tr(
                f"模型回傳的工具呼叫 JSON：`{call['name']}`",
                f"Tool call JSON returned by the model: `{call['name']}`",
            )
            with st.status(label, state="complete"):
                st.json({"name": call["name"], "arguments": call["args"]})
        if message.content:
            st.markdown(message.content)
    elif isinstance(message, ToolMessage):
        info = TOOL_INFO.get(message.name)
        if info and info.network:
            icon = ":material/public: " + tr("**外部呼叫**", "**External call**")
        else:
            icon = ":material/build: " + tr("**本機執行工具**", "**Local tool run**")
        runs = info.text(i18n.current())[0] if info else ""
        where = tr(f"（{runs}）", f" ({runs})") if info else ""
        rejected = was_rejected(message)
        label = f"{icon} `{message.name}`{where}" + (
            tr(" — 已拒絕／未執行", " — rejected / not run") if rejected else ""
        )
        with st.status(label, state="error" if rejected else "complete"):
            st.code(str(message.content)[:3000], language="json")
