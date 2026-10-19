"""The live voice assistant page's pieces: the browser component, demo lines, saving runs and
the History view of a saved session (views/voice_assistant.py draws the page).

The conversation itself runs in the model gateway (WebSocket /v1/voice/agent/ws,
local_llm/voice_agent.py); the component (app/static/voice_agent) streams the microphone or
the scripted demo to it, plays the assistant's speech and draws everything live. When a
session ends its summary comes back here and is saved as runs/voice_agent-*.json.
"""

import base64
import html
import wave
from functools import cache
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import ui
from common import save_run
from i18n import tr

from local_llm import tts, voice_agent

STATIC = Path(__file__).parent / "static" / "voice_agent"
DEFAULT_ASR = "zipformer-zh-en-stream"
STREAM_MODELS = ("zipformer-zh-en-stream", "zipformer-zh-2025-stream", "paraformer-zh-en-stream",
                 "voxtral-mini-4b-realtime-4bit")  # fmt: skip
ASSISTANT_VOICE = voice_agent.DEFAULT_VOICE
# The demo's user. macOS's male zh voices (Eddy, Reed, Rocko, Grandpa) are compact
# formant voices that the streaming models mostly mishear (measured: 「今天天氣如何」 came out
# as 「先先先如何」), so the user is Tingting, a natural Mandarin voice unlike Meijia's.
USER_VOICE = "Tingting"
DEMO_LINES = [
    ("今天天氣如何？", False),
    ("我在內湖。", False),
    ("謝謝，這附近有什麼好吃的早午餐店嗎？", False),
    ("好，那就第一家吧，謝謝！", True),  # spoken while the assistant reads the list
]
BARGE_AFTER_MS = 2200
GPT_LIVE_URL = "https://openai.com/index/continuous-voice-interaction-with-gpt-live/"


@cache
def component():
    return st.components.v2.component(
        "lal_voice_agent",
        html=(STATIC / "voice_agent.html").read_text(encoding="utf-8"),
        css=(STATIC / "voice_agent.css").read_text(encoding="utf-8"),
        js=(STATIC / "voice_agent.js").read_text(encoding="utf-8"),
    )


def ws_url(url: str) -> str:
    """The gateway's voice endpoint for the browser: http://127.0.0.1:11500 -> ws://..."""
    base = url.rstrip("/")
    base = "wss://" + base[8:] if base.startswith("https://") else "ws://" + base.split("://")[-1]
    return base + "/v1/voice/agent/ws"


@st.cache_data(show_spinner=False)
def demo_pcm(text: str, voice: str) -> str:
    """A demo line spoken by a macOS voice, as base64 16 kHz mono PCM16."""
    path = tts.synthesize(text, voice, None, sample_rate=16000)
    with wave.open(str(path), "rb") as w:
        frames = w.readframes(w.getnframes())
    return base64.b64encode(frames).decode()


def demo_data(voice: str) -> dict:
    return {
        "lines": [{"text": t, "barge": b, "pcm": demo_pcm(t, voice)} for t, b in DEMO_LINES],
        "barge_after_ms": BARGE_AFTER_MS,
    }


def labels() -> dict:
    return {
        "mic": tr("開始對話（麥克風）", "Talk (microphone)"),
        "demo": tr("播放示範對話", "Play the demo conversation"),
        "stop": tr("結束", "End"),
        "chat": tr("對話", "Conversation"),
        "board": tr("背景工作", "Background work"),
        "timeline": tr("時間軸：說話與工具同時進行", "Timeline: talking and tools overlap"),
        "metrics": tr("每一輪的延遲", "Latency per turn"),
        "idle": tr(
            "按「開始對話」用麥克風，或播放示範對話。",
            "Press Talk to use the microphone, or play the demo.",
        ),  # fmt: skip
        "connecting": tr("連線模型閘道…", "Connecting to the model gateway…"),
        "loading_asr": tr("載入串流辨識模型…", "Loading the streaming ASR model…"),
        "loading_llm": tr(
            "載入 LLM 並預熱（第一次較久）…", "Loading and warming the LLM (slower the first time)…"
        ),  # fmt: skip
        "listening": tr("聆聽中，直接說話。", "Listening: just talk."),
        "speaking": tr("助理說話中（可以直接插話）", "Assistant speaking (you can interrupt)"),
        "finishing": tr("收尾中…", "Finishing…"),
        "done": tr("已結束，紀錄已儲存。", "Ended; the session was saved."),
        "failed": tr("已中斷。", "Stopped."),
        "closed": tr("連線中斷。", "The connection closed."),
        "empty": tr(
            "說「今天天氣如何？」試試看，或按「播放示範對話」看完整流程。",
            "Try asking 「今天天氣如何？」 (what's the weather today), or play the demo.",
        ),
        "you": tr("你", "You"),
        "hearing": tr("辨識中…", "hearing…"),
        "assistant": tr("助理", "Assistant"),
        "speaking_tag": tr("說話中", "speaking"),
        "filler_tag": tr("先回應", "instant reply"),
        "answer_tag": tr("查詢結果", "from the lookup"),
        "cut_tag": tr("被打斷", "interrupted"),
        "cached": tr("快取", "cached"),
        "user_talking": tr("聽到你在說話…", "Hearing you…"),
        "tools_running": tr(
            "背景查詢中，可以繼續說話。", "Looking it up in the background; keep talking."
        ),  # fmt: skip
        "no_tools": tr(
            "工具呼叫會出現在這裡：執行中、花了幾秒、參數與結果。",
            "Tool calls appear here: running or done, seconds, arguments, result.",
        ),
        "tool_running": tr("查詢中…對話照常進行。", "Running… the conversation goes on."),
        "no_timeline": tr(
            "開始後，使用者、LLM、助理說話與工具會畫在同一條時間軸上。",
            "Once started, the user, the LLM, the assistant's speech and the "
            "tools are drawn on one timeline.",
        ),  # fmt: skip
        "lane_user": tr("使用者", "User"),
        "lane_llm": "LLM",
        "lane_asst": tr("助理說話", "Assistant"),
        "lane_tool": tr("工具", "Tools"),
        "metrics_note": tr(
            "回應＝使用者說完（VAD）到助理開始出聲；答案＝說完到念出查詢結果。辨識＝說完到定稿文字。",
            "Response = user stopped speaking (VAD) to the assistant's first sound; answer = to "
            "the spoken lookup result; ASR = to the final transcript.",
        ),
        "m_turn": tr("使用者", "User"),
        "m_asr": tr("辨識", "ASR"),
        "m_llm": tr("LLM 首字", "LLM first"),
        "m_response": tr("回應", "Response"),
        "m_tool": tr("工具", "Tool"),
        "m_answer": tr("答案", "Answer"),
        "demo_step": tr("示範：使用者第", "Demo: user line"),
        "demo_barge": tr("示範插話", "Demo barge-in"),
        "denied": tr(
            "麥克風權限被拒絕。請在瀏覽器網址列左側允許麥克風，再按一次。",
            "Microphone permission was denied. Allow it in the browser's address bar and try "
            "again.",
        ),
        "no_mic": tr("無法開啟麥克風", "Could not open the microphone"),
        "no_gateway": tr(
            "連不到模型閘道的語音助理端點，請重新啟動閘道（./scripts/restart_gateway.sh）。",
            "Cannot reach the gateway's voice endpoint; restart the gateway "
            "(./scripts/restart_gateway.sh).",
        ),
    }


# -- saving and replaying -----------------------------------------------------------------------


def save(payload: dict) -> dict:
    """A finished session -> runs/voice_agent-*.json; returns the saved data."""
    summary = payload.get("summary") or {}
    data = {
        "mode": payload.get("mode"),
        "settings": payload.get("params") or {},
        "started_at": payload.get("started_at"),
        "asr_load_s": payload.get("asr_load_s"),
        "llm_load_s": payload.get("llm_load_s"),
        "barge_ins": payload.get("barge_ins"),
        "messages": payload.get("messages") or [],
        **{k: summary.get(k) for k in ("llm", "voice", "allow_network", "barge_in", "turns",
                                       "tool_calls", "speeches", "user_speech", "memory")},
    }  # fmt: skip
    path = save_run("voice_agent", data)
    data["path"] = str(path.relative_to(path.parents[1]))
    return data


def history_summary(data: dict) -> dict:
    """What a History card shows."""
    turns = data.get("turns") or []
    first = next((t.get("user") for t in turns if t.get("user")), "")
    last = next((t.get("answer") or t.get("assistant") for t in reversed(turns)
                 if t.get("answer") or t.get("assistant")), "")  # fmt: skip
    speeches = [s for s in data.get("speeches") or [] if s.get("end") is not None]
    seconds = max((s["end"] for s in speeches), default=None)
    models = [m for m in ((data.get("settings") or {}).get("asr"), data.get("llm")) if m]
    return {"title": first, "snippet": last, "models": models, "seconds": seconds,
            "turns": len(turns), "tools": len(data.get("tool_calls") or [])}  # fmt: skip


def _fmt(x) -> str:
    return "—" if x is None else f"{x:.2f}"


DECISIONS = {
    "ask": ("反問", "asked back"),
    "answer": ("直接回答", "answered"),
    "tool": ("背景查詢", "looked up"),
    "interrupted": ("被打斷", "interrupted"),
}


def metrics_frame(data: dict) -> pd.DataFrame:
    rows = []
    for t in data.get("turns") or []:
        decision = DECISIONS.get(t.get("decision") or "")
        rows.append({
            tr("使用者說", "User said"): t.get("user"),
            tr("決定", "Decision"): tr(*decision) if decision else t.get("decision"),
            tr("辨識", "ASR"): _secs(t.get("asr_s")),
            tr("LLM 首字", "LLM first"): _secs(t.get("llm_first_s")),
            tr("回應", "Response"): _secs(t.get("response_s")),
            tr("工具", "Tool"): _secs(t.get("tool_s")),
            tr("答案", "Answer"): _secs(t.get("answer_s")),
        })  # fmt: skip
    return pd.DataFrame(rows)


def _secs(x) -> str:
    return "—" if x is None else f"{x:.2f} s"


def timeline_chart(data: dict) -> alt.Chart | None:
    """User speech, LLM work, assistant speech and tools on one time axis."""
    lanes = [tr("使用者", "User"), "LLM", tr("助理說話", "Assistant"), tr("工具", "Tools")]
    rows = []
    for start, end in data.get("user_speech") or []:
        if start is not None and end is not None:
            rows.append({"lane": lanes[0], "start": start, "end": end,
                         "what": tr("使用者說話", "user speech")})  # fmt: skip
    for t in data.get("turns") or []:
        if t.get("llm_start") is not None and t.get("llm_end") is not None:
            rows.append({"lane": lanes[1], "start": t["llm_start"], "end": t["llm_end"],
                         "what": tr("決定", "decide")})  # fmt: skip
    for c in data.get("tool_calls") or []:
        if c.get("start") is not None and c.get("end") is not None:
            rows.append({"lane": lanes[3], "start": c["start"], "end": c["end"],
                         "what": c.get("name")})  # fmt: skip
        if c.get("compose_start") is not None and c.get("compose_s") is not None:
            rows.append({"lane": lanes[1], "start": c["compose_start"],
                         "end": c["compose_start"] + c["compose_s"],
                         "what": tr("整理答案", "compose")})  # fmt: skip
    kinds = {"filler": tr("先回應", "instant reply"), "answer": tr("查詢結果", "lookup answer"),
             "reply": tr("回答", "reply")}  # fmt: skip
    for s in data.get("speeches") or []:
        if s.get("start") is not None and s.get("end") is not None:
            what = tr("被打斷", "interrupted") if s.get("interrupted") else kinds.get(
                s.get("kind"), s.get("kind"))  # fmt: skip
            rows.append({"lane": lanes[2], "start": s["start"], "end": s["end"], "what": what,
                         "text": s.get("text")})  # fmt: skip
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["text"] = df.get("text", pd.Series(dtype=str)).fillna("")
    order = [
        tr("使用者說話", "user speech"),
        tr("決定", "decide"),
        tr("整理答案", "compose"),
        kinds["reply"],
        kinds["filler"],
        kinds["answer"],
        tr("被打斷", "interrupted"),
    ]
    tools = sorted({r["what"] for r in rows if r["lane"] == lanes[3]})
    colors = alt.Scale(
        domain=order + tools,
        range=["#4F46E5", "#7C3AED", "#A78BFA", "#0F766E", "#34D399", "#0D9488", "#DC2626"]
        + ["#D97706", "#F59E0B"][: len(tools)]
        + ["#B45309"] * max(len(tools) - 2, 0),
    )
    return (
        alt.Chart(df)
        .mark_bar(cornerRadius=3, height=14)
        .encode(
            x=alt.X("start:Q", title=tr("秒（工作階段時間）", "seconds (session clock)")),
            x2="end:Q",
            y=alt.Y("lane:N", sort=lanes, title=None),
            color=alt.Color("what:N", scale=colors, title=None, legend=alt.Legend(orient="right")),
            tooltip=[
                "lane",
                "what",
                "text",
                alt.Tooltip("start:Q", format=".2f"),
                alt.Tooltip("end:Q", format=".2f"),
            ],
        )  # fmt: skip
        .properties(height=190)
    )


def history_view(data: dict) -> None:
    """A saved session: settings, the conversation, latencies, tool calls and the timeline."""
    settings = data.get("settings") or {}
    st.markdown(
        ui.badges(["asr", "tools", "tts"], [
            f":material/subtitles: {settings.get('asr', '?')}",
            f":material/smart_toy: {data.get('llm', '?')}",
            f":material/record_voice_over: {data.get('voice', '?')}",
            tr("示範", "demo") if data.get("mode") == "demo" else tr("麥克風", "microphone"),
        ]),
        unsafe_allow_html=True,
    )  # fmt: skip
    chart = timeline_chart(data)
    if chart is not None:
        st.altair_chart(chart, use_container_width=True)
    st.dataframe(metrics_frame(data), hide_index=True, use_container_width=True)
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.markdown(f"**{tr('對話', 'Conversation')}**")
        for m in data.get("messages") or []:
            role = "user" if m.get("role") == "user" else "assistant"
            with st.chat_message(role):
                text = m.get("text") or ""
                unsaid = m.get("unsaid") or ""
                if unsaid and text.endswith(unsaid):
                    text = f"{text[: -len(unsaid)]}~~{unsaid}~~"
                text = html.escape(text)
                tags = []
                if m.get("kind") == "filler":
                    tags.append(tr("先回應", "instant reply"))
                if m.get("kind") == "answer":
                    tags.append(tr("查詢結果", "from the lookup"))
                if m.get("cut"):
                    tags.append(tr("被打斷", "interrupted"))
                st.markdown(text + (f"  \n:gray[{' · '.join(tags)}]" if tags else ""))
    with right:
        st.markdown(f"**{tr('工具呼叫', 'Tool calls')}**")
        calls = data.get("tool_calls") or []
        if not calls:
            st.caption(tr("這次沒有呼叫工具。", "No tools were called."))
        for c in calls:
            with st.container(border=True):
                ok = c.get("ok")
                mark = ":green[:material/check_circle:]" if ok else ":red[:material/error:]"
                st.markdown(f"{mark} **{c.get('name')}** · {_fmt(c.get('seconds'))} s · "
                            f"`{c.get('args')}`")  # fmt: skip
                st.caption(c.get("summary") or "")
                result = c.get("result") or {}
                if result.get("attribution"):
                    st.caption(result["attribution"])


def flow_dot(compact: bool = False) -> str:
    """The cascade as a small Graphviz diagram (plain text labels, no emoji); `compact` is
    the home page card's shorter version."""
    labels = {
        "mic": tr("麥克風\\n16 kHz PCM", "Microphone\\n16 kHz PCM"),
        "asr": tr("串流辨識\\nVAD + ASR", "Streaming ASR\\nVAD + ASR"),
        "llm": tr("前台 LLM\\n回答／反問／查詢", "Front LLM\\nanswer / ask / look up"),
        "tool": tr("背景工具\\n天氣・店家", "Background tools\\nweather, places"),
        "tts": tr("語音合成\\nmacOS", "Speech\\nmacOS voices"),
        "out": tr("瀏覽器播放", "Browser plays"),
        "filler": tr("先說一句", "say a line first"),
        "result": tr("結果", "result"),
        "barge": tr("插話：停止播放", "barge-in: stop"),
    }
    node_size, edge_size = (13, 11) if compact else (11, 10)
    head = f"""digraph {{
  rankdir=LR; bgcolor="transparent"; pad=0.1; nodesep=0.25; ranksep=0.3;
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize={node_size},
        fillcolor="#EEF2FF", color="#C7D2FE", fontcolor="#1E293B"];
  edge [fontname="Helvetica", fontsize={edge_size}, color="#94A3B8", fontcolor="#64748B"];
"""
    nodes = f"""  asr [label="{labels["asr"]}"]; llm [label="{labels["llm"]}"];
  tool [label="{labels["tool"]}", fillcolor="#FFFBEB", color="#FCD34D"];
  tts [label="{labels["tts"]}", fillcolor="#ECFDF5", color="#A7F3D0"];
"""
    if compact:
        return (
            head
            + nodes
            + """  asr -> llm -> tts; llm -> tool [style=dashed];
  tool -> tts [style=dashed]; {rank=same; llm; tool}
}"""
        )
    return (
        head
        + nodes
        + f"""  mic [label="{labels["mic"]}"];
  out [label="{labels["out"]}", fillcolor="#ECFDF5", color="#A7F3D0"];
  mic -> asr -> llm; llm -> tts [label="{labels["filler"]}"]; tts -> out;
  llm -> tool [style=dashed]; tool -> llm [label="{labels["result"]}", style=dashed];
  asr -> out [label="{labels["barge"]}", style=dotted, color="#DC2626", fontcolor="#DC2626"];
}}"""
    )


PAGE = "views/voice_assistant.py"


def blurb() -> str:
    return tr(
        "GPT-Live 風格的即時語音對話：串流辨識 → 前台 LLM 決定回答、反問或查詢 → 工具在背景跑、"
        "助理先說一句話撐住對話 → 查到後念出結果；可以隨時插話打斷。",
        "GPT-Live-style live voice conversation: streaming ASR → a front LLM answers, asks "
        "back or looks something up → tools run in the background while the assistant says a "
        "line first → the result is spoken; interrupt at any time.",
    )


def home_card(col) -> None:
    """The live voice assistant as a card among the home page's scenarios."""
    with col.container(border=True, height=450):
        st.markdown(f":material/support_agent: **{tr('即時語音助理', 'Live voice assistant')}**")
        st.graphviz_chart(flow_dot(compact=True), use_container_width=True)
        st.markdown(
            ui.badges(["stream", "voice_chat", "tools", "tts"], [voice_agent.DEFAULT_LLM]),
            unsafe_allow_html=True,
        )
        st.caption(blurb())
        if st.button(tr("開啟", "Open"), icon=":material/arrow_forward:", type="primary",
                     key="open_voice_assistant"):  # fmt: skip
            st.switch_page(PAGE)


def gallery_link() -> None:
    """A pointer from the scenario gallery: this scenario is a live page, not a graph."""
    with st.container(border=True):
        text, link = st.columns([4, 1.2], vertical_alignment="center")
        text.markdown(
            f":material/support_agent: **{tr('即時語音助理', 'Live voice assistant')}** · "
            + tr("另一個頁面的即時場景", "a live scenario on its own page")
        )
        text.caption(blurb())
        link.page_link(PAGE, label=tr("開啟即時語音助理", "Open the live assistant"),
                       icon=":material/arrow_forward:")  # fmt: skip
