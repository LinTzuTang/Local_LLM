"""The live voice assistant's settings, test scripts and live session, drawn by its own page
(views/voice_assistant.py) and inside the scenario gallery (views/scenarios.py).

The conversation runs in the model gateway (/v1/voice/agent/ws); the component streams the
microphone or a test script to it. A test script needs no microphone: its lines are spoken by a
macOS voice (or come from uploaded recordings) and fed to the gateway as a virtual microphone,
one 100 ms frame per 100 ms, each line after the assistant has finished; a line marked as a
barge-in is spoken while the assistant reads out a lookup result.
"""

import speech_common as sc
import speech_stream
import streamlit as st
import voice_assistant as va
from common import download_control, gateway_url, installed_names, kept
from i18n import tr

from local_llm import tts, voice_agent


def panel(page: str = va.PAGE) -> None:
    """Everything below the page header; `page` is where a model download returns to."""
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
        c1, c2, c3 = st.columns(3)
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
                "用 7 段測試對話比過 10 個模型（關閉思考；中英文、反問地點、追問、閒聊、插話）：Gemma 4 12B 判斷全對、"
                "不編造、最自然，記憶體 7.7 GB，所以是預設；說完約 0.5 秒開口。E4B 最快（約 0.3 秒）但 46 輪錯 4 輪"
                "（英文找夜市不查、說要查卻沒查）；Gemma 4 26B 也全對，但會把數字念錯；Qwen3.6 35B 全對但要 23.6 GB。",
                "Compared on 7 test conversations across 10 models (thinking off; Chinese and "
                "English, asking where, follow-ups, small talk, barge-in): Gemma 4 12B got every "
                "decision right, made nothing up and sounded the most natural in 7.7 GB, so it is "
                "the default; it starts talking about 0.5 s after you stop. E4B is the fastest "
                "(about 0.3 s) but missed 4 of 46 turns; Gemma 4 26B is right too but misreads "
                "numbers; Qwen3.6 35B is right but needs 23.6 GB.",
            ),
            **kept("va_llm"),
        )
        default_voice = next((n for n in voice_names if n.startswith(va.ASSISTANT_VOICE)), None)
        st.session_state.setdefault("va_voice", default_voice or (voice_names or [""])[0])
        voice = c3.selectbox(tr("助理的聲音", "Assistant voice"), voice_names or [""],
                             format_func=short, **kept("va_voice"))  # fmt: skip
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
                    "測試腳本要看到工具在背景執行，請先開啟。",
                    ":material/public_off: **No online lookups**: the assistant will not look up "
                    "weather or places and says the switch above must be on. Turn it on to see the "
                    "test scripts' background tools.",
                )
            )

    demo, script_note = test_script(voice_names, short)

    if voice_agent.DEFAULT_LLM not in installed:
        download_control(voice_agent.DEFAULT_LLM, key="va_pull_llm", page=page)
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
            "demo": demo,
        },
        on_session_change=lambda: None,
    )
    payload = getattr(result, "session", None) if result is not None else None
    if payload and payload.get("id") != st.session_state.get("va_saved_id"):
        st.session_state.va_saved_id = payload.get("id")
        st.session_state.va_last = va.save(payload)
    st.caption(
        script_note
        + tr(
            "　天氣：Open-Meteo（CC BY 4.0）；店家：© OpenStreetMap contributors（ODbL）。",
            " Weather: Open-Meteo (CC BY 4.0); places: © OpenStreetMap contributors (ODbL).",
        )
    )
    last = st.session_state.get("va_last")
    if last:
        st.subheader(tr("上一段對話", "Last session"))
        va.history_view(last)
        sc.saved_caption(last)


def test_script(voice_names: list[str], short) -> tuple[dict | None, str]:
    """The test-script picker (a conversation without a microphone): the component's demo data
    ({"lines": [{text, barge, pcm}], "barge_after_ms"}) and a caption describing it."""
    with st.container(border=True):
        st.markdown(
            f":material/play_circle: **{tr('測試腳本（不用麥克風）', 'Test script (no microphone)')}**",
            help=tr(
                "按下方的「播放測試腳本」，每句台詞依序當成麥克風輸入送進同一條串流辨識（每 0.1 秒一小段，"
                "跟真的講話一樣），每句等助理講完才說下一句；標成插話的那句在助理念查詢結果時插進去。",
                "Press Play the test script below: each line goes into the same streaming ASR as "
                "microphone input (100 ms at a time, as if spoken), each after the assistant has "
                "finished; the line marked as a barge-in cuts in while the assistant reads a "
                "lookup result.",
            ),
        )
        s1, s2 = st.columns([2, 1])
        options = list(va.TEST_SCRIPTS)
        st.session_state.setdefault("va_script", options[0])
        if st.session_state.va_script not in options:
            st.session_state.va_script = options[0]
        key = s1.selectbox(
            tr("腳本", "Script"),
            options,
            format_func=lambda k: tr(va.TEST_SCRIPTS[k]["zh"], va.TEST_SCRIPTS[k]["en"]),
            **kept("va_script"),
        )
        script = va.TEST_SCRIPTS[key]
        if key == "upload":
            files = st.file_uploader(
                tr(
                    "錄音（每個檔案一句，依檔名順序）",
                    "Recordings (one line per file, by file name)",
                ),
                type=["wav", "mp3", "m4a", "flac", "ogg", "aiff"],
                accept_multiple_files=True,
                key="va_script_files",
            )
            files = sorted(files or [], key=lambda f: f.name)
            lines = [
                {"text": f.name, "barge": False, "pcm": va.upload_pcm(f.getvalue(), f.name)}
                for f in files
            ]
            s2.caption(tr("上傳的錄音直接送進辨識，不經語音合成。",
                          "Uploaded recordings go straight to the ASR, no synthesis."))  # fmt: skip
            note = tr(
                f"測試腳本：{len(lines)} 段上傳的錄音，每段等助理講完才播下一段。",
                f"Test script: {len(lines)} uploaded recordings, each after the assistant has "
                "finished.",
            )
            demo = va.demo_payload(lines) if lines else None
            return demo, note
        if key == "custom":
            st.session_state.setdefault("va_script_text", "\n".join(
                ("插話：" if b else "") + t for t, b in va.DEMO_LINES))  # fmt: skip
            text = st.text_area(
                tr("台詞（一行一句）", "Lines (one per line)"),
                height=130,
                help=tr(
                    "以「插話：」（或 barge:）開頭的那一行，會在助理念查詢結果時插進去；只用最後一個。",
                    "A line starting with 插話： (or barge:) cuts in while the assistant reads a "
                    "lookup result; only the last one is used.",
                ),
                **kept("va_script_text"),
            )
            pairs = va.parse_lines(text)
        else:
            pairs = script["lines"]
        english = bool(pairs) and all(voice_agent.is_english(t) for t, _ in pairs)
        default = va.USER_VOICE_EN if english else va.USER_VOICE
        voice_key = "va_user_voice_en" if english else "va_user_voice"
        user_default = next((n for n in voice_names if n.startswith(default)), None)
        st.session_state.setdefault(voice_key, user_default or (voice_names or [""])[0])
        user_voice = s2.selectbox(
            tr("使用者的聲音", "User's voice"),
            voice_names or [""],
            format_func=short,
            help=tr(
                "台詞用這個聲音合成，再即時送進串流辨識。Flo 代表 macOS 的 Eloquence 合成器（Eddy、Reed 等 8 個聲音"
                "只差音高），音質較機械、辨識較常出錯，所以中文預設婷婷、英文預設 Samantha。",
                "The lines are spoken by this voice and streamed through the ASR. Flo stands for "
                "macOS's Eloquence synthesiser (Eddy, Reed and six more differ only in pitch), "
                "which is robotic and more often misheard, so the defaults are Tingting for "
                "Chinese and Samantha for English.",
            ),
            **kept(voice_key),
        )
        if key != "custom":
            st.caption("　→　".join(("⤷ " + tr("插話：", "barge-in: ") if b else "") + t
                                   for t, b in pairs))  # fmt: skip
    if not pairs or not user_voice:
        return None, tr("測試腳本：沒有台詞。", "Test script: no lines.")
    normal = sum(1 for _, b in pairs if not b)
    has_barge = any(b for _, b in pairs)
    note = tr(
        f"測試腳本：用 {short(user_voice)} 合成 {normal} 句台詞，每句等助理講完才說下一句"
        + ("，並在助理念查詢結果時插一句話，測試打斷。" if has_barge else "。"),
        f"Test script: {normal} lines spoken by {short(user_voice)}, each after the assistant "
        "has finished"
        + (
            ", plus one cutting in while it reads a lookup result (barge-in)." if has_barge else "."
        ),
    )
    return va.demo_data(pairs, user_voice), note
