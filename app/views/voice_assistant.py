"""Live voice assistant: a GPT-Live-style conversation, run as a local cascade.

Settings and the explanation are drawn here; the conversation runs in the model gateway
(/v1/voice/agent/ws) and is drawn live by the voice_agent component. A finished session is
saved as runs/voice_agent-*.json and shown below (and in History).
"""

import speech_common as sc
import speech_stream
import streamlit as st
import ui
import voice_assistant as va
from common import download_control, gateway_url, installed_names, kept
from i18n import tr

from local_llm import tts, voice_agent

ui.page_header(
    "support_agent",
    tr("即時語音助理", "Live voice assistant"),
    tr(
        "像跟人講電話一樣跟助理對話：它邊聽邊辨識，你一說完就先用一句話回應（例如「好，我幫你查一下內湖的天氣」），"
        "需要查資料時把工具放到背景執行、同時繼續聽你說話，查到後再把結果念給你聽；你隨時可以插話打斷它。"
        f'這個流程參考 OpenAI 的 <a href="{va.GPT_LIVE_URL}" target="_blank">GPT-Live</a>'
        "（快速的語音層維持對話、較重的工作在背景進行）。",
        "Talk to the assistant as you would on the phone: it transcribes while you speak, "
        'answers with a short line the moment you finish ("OK, let me check the weather in '
        'Neihu"), runs lookups in the background while it keeps listening, and speaks the '
        "result when it arrives; you can interrupt it at any time. The flow follows OpenAI's "
        f'<a href="{va.GPT_LIVE_URL}" target="_blank">GPT-Live</a> (a fast voice layer keeps '
        "the conversation going while heavier work runs in the background).",
    ),
    eyebrow=tr("多 Agent 場景", "Scenarios"),
)
with st.expander(tr("在本機做得到什麼、和 GPT-Live 有什麼不同", "What is different locally"),
                 icon=":material/info:"):  # fmt: skip
    st.markdown(
        tr(
            "- **串接，不是全雙工模型**：GPT-Live 是一個同時聽和說的語音模型；這裡是三個本機元件串起來——"
            "串流語音辨識（sherpa-onnx / Voxtral）→ 前台 LLM（Ollama，關閉思考，帶工具定義）→ macOS `say` 語音合成。\n"
            "- **延遲約 1–2 秒**：要等辨識判定一句話結束（句尾靜音約 0.8 秒），再加 LLM 決定與合成；"
            "實際每一輪的數字列在下方。\n"
            "- **背景工具**：前台 LLM 決定要查時，先念一句短回應，工具在背景執行，對話照常進行；"
            "查到後由同一個 LLM 整理成兩三句口語再念出來。\n"
            "- **插話**：助理說話時偵測到你開口（音量 VAD），瀏覽器立刻停止播放、改聽你說。"
            "用喇叭時麥克風可能收到助理自己的聲音，建議戴耳機（瀏覽器已開回音消除）。\n"
            "- **資料來源**：天氣來自 Open-Meteo，店家來自 OpenStreetMap（Nominatim、Overpass），"
            "只會說查到的店名，查不到就說查不到。",
            "- **A cascade, not a full-duplex model**: GPT-Live is one speech model that listens "
            "and talks at once; here three local parts are chained: streaming ASR (sherpa-onnx / "
            "Voxtral) → a front LLM (Ollama, thinking off, with tool definitions) → macOS `say`.\n"
            "- **About 1–2 s of latency**: the recogniser must decide the sentence has ended "
            "(about 0.8 s of trailing silence), then the LLM decides and speech is synthesised; "
            "the measured numbers of every turn are listed below.\n"
            "- **Background tools**: when the front LLM decides to look something up it says a "
            "short line first, the tool runs in the background while the conversation goes on, "
            "and the same LLM turns the result into two or three spoken sentences.\n"
            "- **Barge-in**: when you start talking while the assistant speaks (energy VAD), "
            "the browser stops playback at once and listens. With speakers the microphone can "
            "hear the assistant itself; headphones are best (echo cancellation is on).\n"
            "- **Data**: weather from Open-Meteo, places from OpenStreetMap (Nominatim, "
            "Overpass); it only names places it found, and says so when it finds none.",
        )
    )
    st.graphviz_chart(va.flow_dot(), use_container_width=True)

url = gateway_url()
status = sc.gateway_status(url)
if status is None:
    st.stop()
names = sc.names("stream")
asr_models = [m for m in va.STREAM_MODELS if status.get(m, {}).get("installed")
              and speech_stream.realtime(m)]  # fmt: skip
if not asr_models:
    st.info(
        tr("還沒有下載串流辨識模型；到「語音模型比較 → 即時」或模型庫下載 sherpa-onnx Zipformer。",
           "No streaming ASR model is downloaded; download sherpa-onnx Zipformer under Compare "
           "speech models → Live or in the model library."),
        icon=":material/download:",
    )  # fmt: skip
    st.stop()
installed = set(installed_names())
llms = [m for m in voice_agent.CANDIDATE_LLMS if m in installed]
voices = tts.voices(("zh_TW", "zh_CN", "en_US"))
voice_names = [v["name"] for v in voices]


def short(name: str) -> str:
    v = tts.find(name, voices)
    return f"{v['short']} ({v['locale']})" if v else name


with st.container(border=True):
    st.markdown(f":material/tune: **{tr('設定', 'Settings')}**")
    c1, c2, c3, c4 = st.columns(4)
    st.session_state.setdefault("va_asr", va.DEFAULT_ASR if va.DEFAULT_ASR in asr_models
                                else asr_models[0])  # fmt: skip
    if st.session_state.va_asr not in asr_models:
        st.session_state.va_asr = asr_models[0]
    asr = c1.selectbox(
        tr("串流辨識模型", "Streaming ASR"),
        asr_models,
        format_func=lambda m: names.get(m, m),
        help=tr(
            "只列出跟得上即時的串流模型（4-bit Voxtral 可用，全精度版太慢）。",
            "Only streaming models that keep up with real time (4-bit Voxtral; the "
            "full-precision build is too slow).",
        ),  # fmt: skip
        **kept("va_asr"),
    )
    llm_choices = llms or [voice_agent.DEFAULT_LLM]
    st.session_state.setdefault("va_llm", llm_choices[0])
    if st.session_state.va_llm not in llm_choices:
        st.session_state.va_llm = llm_choices[0]
    llm = c2.selectbox(
        tr("前台 LLM", "Front LLM"),
        llm_choices,
        help=tr(
            "量過三輪示範對話（每個模型 16 次、關閉思考）：Gemma 4 E4B、Qwen3.6 35B、Gemma 4 12B、"
            "Qwen3.8 27B 的工具判斷全對，E4B 最快（決定中位數約 0.1 秒）也最省記憶體，所以是預設；"
            "qwen3:1.7b 有 5／18 次判斷錯。",
            "Measured on the three demo turns (16 runs per model, thinking off): Gemma 4 E4B, "
            "Qwen3.6 35B, Gemma 4 12B and Qwen3.8 27B made every tool decision right; E4B was "
            "fastest (median decision about 0.1 s) and smallest, so it is the default; "
            "qwen3:1.7b got 5 of 18 wrong.",
        ),
        **kept("va_llm"),
    )
    default_voice = next((n for n in voice_names if n.startswith(va.ASSISTANT_VOICE)), None)
    st.session_state.setdefault("va_voice", default_voice or (voice_names or [""])[0])
    voice = c3.selectbox(tr("助理的聲音", "Assistant voice"), voice_names or [""],
                         format_func=short, **kept("va_voice"))  # fmt: skip
    user_default = next((n for n in voice_names if n.startswith(va.USER_VOICE)), None)
    st.session_state.setdefault("va_user_voice", user_default or (voice_names or [""])[0])
    user_voice = c4.selectbox(
        tr("示範中使用者的聲音", "Demo user's voice"),
        voice_names or [""],
        format_func=short,
        help=tr(
            "示範時三句問題用這個聲音合成，再即時送進同一個串流辨識。macOS 的男聲（Eddy、Reed…）"
            "音質較機械，辨識常出錯，所以預設用婷婷。",
            "The demo's questions are spoken by this voice and streamed through the same ASR. "
            "macOS's male zh voices (Eddy, Reed…) are robotic and often misheard, so the "
            "default is Tingting.",
        ),
        **kept("va_user_voice"),
    )
    n1, n2 = st.columns([2.2, 1])
    st.session_state.setdefault("va_net", False)
    net = n1.toggle(
        tr(
            "允許這個場景連網查詢天氣與店家",
            "Allow this scenario to look up weather and places online",
        ),
        help=tr(
            "外部工具平常預設關閉、每次呼叫都要人工核准；語音對話沒辦法停下來等核准，所以這個場景用這個開關"
            "一次授權：開啟後，get_weather（Open-Meteo）與 search_places（OpenStreetMap）兩個唯讀查詢"
            f"不需逐次核准，每個工具每次對話最多 {voice_agent.CALL_LIMIT} 次。",
            "External tools are normally off and need a human approval per call; a voice "
            "conversation cannot stop for that, so this switch grants it once for this "
            "scenario: when on, the two read-only lookups get_weather (Open-Meteo) and "
            f"search_places (OpenStreetMap) run without per-call approval, at most "
            f"{voice_agent.CALL_LIMIT} calls per tool per session.",
        ),  # fmt: skip
        **kept("va_net"),
    )
    st.session_state.setdefault("va_barge", True)
    barge = n2.toggle(tr("允許插話（barge-in）", "Allow barge-in"), **kept("va_barge"),
                      help=tr("助理說話時你一開口就停下來聽你說。",
                              "Stop talking as soon as you start speaking."))  # fmt: skip
    if net:
        st.caption(
            tr(
                f":material/public: **已允許連網**：只會送出地名與店家類型到 Open-Meteo 與 OpenStreetMap，"
                f"不需逐次核准，每個工具每次對話最多 {voice_agent.CALL_LIMIT} 次；其他外部工具仍然關閉。",
                ":material/public: **Online lookups allowed**: only the place name and the kind "
                "of shop go to Open-Meteo and OpenStreetMap, without per-call approval, at most "
                f"{voice_agent.CALL_LIMIT} calls per tool per session; every other external "
                "tool stays off.",
            )
        )
    else:
        st.caption(
            tr(
                ":material/public_off: **未允許連網**：助理不會查天氣或店家，被問到時會說需要先打開上面的開關。"
                "示範對話要看到工具在背景執行，請先開啟。",
                ":material/public_off: **No online lookups**: the assistant will not look up "
                "weather or places and says the switch above must be on. Turn it on to see the "
                "demo's background tools.",
            )
        )

if voice_agent.DEFAULT_LLM not in installed:
    download_control(voice_agent.DEFAULT_LLM, key="va_pull_llm",
                     page="views/voice_assistant.py")  # fmt: skip
if llm not in installed:
    st.stop()

result = va.component()(
    key="lal_voice_agent",
    data={
        "key": "lal_voice_agent",
        "ws_url": va.ws_url(url),
        "params": {
            "asr": asr,
            "llm": llm,
            "voice": voice,
            "rate": 0,
            "net": net,
            "barge": barge,
        },  # fmt: skip
        "labels": va.labels(),
        "demo": va.demo_data(user_voice) if user_voice else None,
    },
    on_session_change=lambda: None,
)
payload = getattr(result, "session", None) if result is not None else None
if payload and payload.get("id") != st.session_state.get("va_saved_id"):
    st.session_state.va_saved_id = payload.get("id")
    st.session_state.va_last = va.save(payload)
st.caption(
    tr(
        "示範對話：用另一個聲音合成三句問題（今天天氣如何？／我在內湖。／謝謝，這附近有什麼好吃的早午餐店嗎？），"
        "每句等助理講完才說下一句，並在助理念店家清單時插一句話，示範打斷。"
        "天氣：Open-Meteo（CC BY 4.0）；店家：© OpenStreetMap contributors（ODbL）。",
        "The demo speaks three questions in another voice (what's the weather today / I'm in "
        "Neihu / thanks, any good brunch places nearby?), each after the assistant has "
        "finished, and cuts in while the assistant reads the list of places to show barge-in. "
        "Weather: Open-Meteo (CC BY 4.0); places: © OpenStreetMap contributors (ODbL).",
    )
)
last = st.session_state.get("va_last")
if last:
    st.subheader(tr("上一段對話", "Last session"))
    va.history_view(last)
    sc.saved_caption(last)
