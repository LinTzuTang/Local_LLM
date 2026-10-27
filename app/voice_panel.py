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
import voice_bench
from common import download_control, gateway_url, installed_names, kept
from i18n import tr

from local_llm import tts, voice_agent, voice_cloud, voice_omni

ENGINE_LABELS = {
    "qwen3-omni": ("Qwen3-Omni 30B-A3B（本機，約 20 GB）", "Qwen3-Omni 30B-A3B (local, about 20 GB)"),
    "minicpm-o": ("MiniCPM-o 4.5（本機，8.6 GB）", "MiniCPM-o 4.5 (local, 8.6 GB)"),
    "minicpm-o-duplex": ("MiniCPM-o 4.5（本機，8.6 GB）", "MiniCPM-o 4.5 (local, 8.6 GB)"),
    "gpt-realtime": ("Azure gpt-realtime（雲端，錄音會上傳）", "Azure gpt-realtime (cloud, audio is uploaded)"),
    "gpt-live-1": ("Azure gpt-live-1（雲端，錄音會上傳）", "Azure gpt-live-1 (cloud, audio is uploaded)"),
}  # fmt: skip


def kind_of(engine: str) -> str:
    """ "cascade", "turn" or "duplex": the way of talking of an engine."""
    info = voice_omni.ENGINES.get(engine) or voice_cloud.ENGINES.get(engine)
    return info["kind"] if info else "cascade"


def voice_source(v: dict | None) -> str:
    """Where a voice comes from, for the voice pickers ("macOS 內建", "Kokoro-82M（MLX）")."""
    return tr(*tts.source(v))


def panel(page: str = va.PAGE) -> None:
    """Everything below the page header; `page` is where a model download returns to."""
    with st.expander(tr("在本機做得到什麼、和 GPT-Live 有什麼不同", "What is different locally"),
                     icon=":material/info:"):  # fmt: skip
        st.markdown(
            tr(
                "- **串接，不是全雙工模型**：GPT-Live 是一個同時聽和說的語音模型；這裡是三個本機元件串起來——"
                "串流語音辨識（sherpa-onnx / Voxtral）→ 前台 LLM（Ollama，關閉思考，帶工具定義）→ 語音合成（macOS 內建或本機神經網路聲音）。\n"
                "- **延遲約 1–2 秒**：要等辨識判定一句話結束（句尾靜音約 0.8 秒），再加 LLM 決定與合成；"
                "實際每一輪的數字列在下方。\n"
                "- **背景工具**：前台 LLM 決定要查時，先念一句短回應，工具在背景執行，對話照常進行；"
                "查到後由同一個 LLM 整理成兩三句口語再念出來。\n"
                "- **插話**：助理說話時偵測到你開口（音量 VAD），瀏覽器立刻停止播放、改聽你說。"
                "用喇叭時麥克風可能收到助理自己的聲音，建議戴耳機（瀏覽器已開回音消除）。\n"
                "- **資料來源**：天氣來自 Open-Meteo，店家來自 OpenStreetMap（Nominatim、Overpass），"
                "只會說查到的店名，查不到就說查不到。\n"
                "- **也可以換成單一語音模型**：設定的「對話方式」可選 ② 回合式（Qwen3-Omni、MiniCPM-o：你講完，"
                "一個模型直接聽你的聲音、自己說話）或 ③ 全雙工（MiniCPM-o：每秒都在聽，自己決定開口、附和或停下）。"
                "它們沒有工具，問天氣、店家會說查不到；串流辨識只用來顯示字幕和判斷你講完。",
                "- **A cascade, not a full-duplex model**: GPT-Live is one speech model that listens "
                "and talks at once; here three local parts are chained: streaming ASR (sherpa-onnx / "
                "Voxtral) → a front LLM (Ollama, thinking off, with tool definitions) → speech synthesis "
                "(macOS or a local neural voice).\n"
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
                "Overpass); it only names places it found, and says so when it finds none.\n"
                "- **Or one speech model**: under Settings → How it talks pick ② turn-based "
                "(Qwen3-Omni, MiniCPM-o: when you finish, one model hears your voice and speaks "
                "itself) or ③ full-duplex (MiniCPM-o: it hears every second and decides itself "
                "to talk, back-channel or stop). They have no tools, so they say they cannot look "
                "up weather or shops; the streaming ASR only draws the captions and finds the turns.",
            )
        )

    url = gateway_url()
    status = sc.gateway_status(url)
    if status is None:
        st.stop()
    names = sc.names("stream") | sc.names("asr")
    asr_models = [m for m in va.STREAM_MODELS if status.get(m, {}).get("installed")
                  and speech_stream.realtime(m)]  # fmt: skip
    pause_models = [m for m in va.PAUSE_MODELS if status.get(m, {}).get("installed")]

    def asr_label(m: str) -> str:
        kind = (tr("串流", "streaming") if m in va.STREAM_MODELS
                else tr("非串流・停頓後整句辨識", "whole-clip, after each pause"))  # fmt: skip
        return f"{names.get(m, m)} · {kind}"

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
    # the test lines are rendered once and cached, so the slow voices are offered there too
    user_voices = tts.voices(("zh_TW", "zh_CN", "en_US"), neural="all")
    user_names = [v["name"] for v in user_voices]

    def short(name: str) -> str:
        v = tts.find(name, user_voices)
        return f"{v['short']} ({v['locale']}) · {voice_source(v)}" if v else name

    engines = ["cascade"] + [e for e in voice_omni.ENGINES if voice_omni.available(e)]
    engines += [e for e in voice_cloud.ENGINES if voice_cloud.available(e)]
    with st.container(border=True):
        st.markdown(f":material/tune: **{tr('設定', 'Settings')}**")
        st.session_state.setdefault("va_engine", "cascade")
        if st.session_state.va_engine not in engines:
            st.session_state.va_engine = "cascade"
        by_kind = {k: [e for e in engines if kind_of(e) == k] for k in va.KINDS}
        kinds = [k for k in va.KINDS if by_kind[k]]
        st.session_state.setdefault("va_kind", kind_of(st.session_state.va_engine))
        if st.session_state.va_kind not in kinds:
            st.session_state.va_kind = "cascade"
        kind = (
            st.segmented_control(
                tr("對話方式", "How it talks"),
                kinds,
                format_func=lambda k: tr(*va.KINDS[k]),
                required=True,
                help=tr(
                    "① 串接式＝三個模型接力，可接工具查資料；② 回合式＝一個語音模型，等你講完才回（像對講機）；"
                    "③ 全雙工＝一個語音模型邊聽邊說（像打電話）。測試結果：串接式最可靠；單一模型裡 Qwen3-Omni "
                    "全程繁體、不編造，約 1.3 秒開口；MiniCPM-o 回合式約 0.9 秒開口但多用簡體，全雙工常沒回應。",
                    "① Cascade = three models in a relay, with tools; ② turn-based = one speech "
                    "model that answers once you finish (like a walkie-talkie); ③ full-duplex = "
                    "one speech model that listens while it talks (like a phone call). In the "
                    "tests the cascade was the most reliable; among single models Qwen3-Omni kept "
                    "to Traditional Chinese, made nothing up and started in about 1.3 s; "
                    "MiniCPM-o turn-based starts in about 0.9 s but mostly writes Simplified, and "
                    "its full-duplex mode often stays silent.",
                ),
                **kept("va_kind"),
            )
            or "cascade"
        )
        st.caption(tr(*va.KIND_NOTES[kind]))
        if kind == "cascade":
            engine = "cascade"
        else:
            options = by_kind[kind]
            if st.session_state.va_engine not in options:
                st.session_state.va_engine = options[0]
            engine = st.selectbox(
                tr("語音模型", "Speech model"),
                options,
                format_func=lambda e: tr(*ENGINE_LABELS.get(e, (e, e))),
                help=tr(
                    "本機模型第一次要載入：Qwen3-Omni 約 20 GB、MiniCPM-o 約 8.6 GB；雲端模型會把錄音上傳。",
                    "A local model loads on first use (Qwen3-Omni about 20 GB, MiniCPM-o 8.6 GB); "
                    "a cloud model uploads the audio.",
                ),
                **kept("va_engine"),
            )
        single = engine != "cascade"
        hear = False  # the cascade's audio LLM hears the recording (set below)
        local_single = single and engine not in voice_cloud.ENGINES
        captions = True
        if local_single:
            st.session_state.setdefault("va_captions", True)
            captions = st.toggle(
                tr(
                    "顯示使用者字幕（另外用串流辨識）",
                    "Show the user's captions (a separate streaming ASR)",
                ),
                help=tr(
                    "Qwen3-Omni、MiniCPM-o 只輸出自己的回答，不會回傳它聽到了什麼，所以畫面上的使用者字幕要另外用"
                    "串流辨識產生（雲端的 gpt-realtime、gpt-live-1 會回傳，所以不用）。關掉就不跑辨識，只用音量偵測"
                    "判斷你講完：回合式照樣在你講完時把那段聲音交給模型，只是看不到你說了什麼。",
                    "Qwen3-Omni and MiniCPM-o only output their own replies and never say what they "
                    "heard, so the user's captions need a separate streaming ASR (the cloud "
                    "gpt-realtime and gpt-live-1 send their transcript, so they need none). Off: no "
                    "ASR, only an energy VAD finds the end of your turn; turn-based still gets your "
                    "audio when you finish, you just do not see what you said.",
                ),
                **kept("va_captions"),
            )
        c1, c2, c3 = st.columns(3)
        # the cascade can also cut at pauses and use a whole-clip model (gateway/voice.py)
        asr_choices = asr_models + (pause_models if engine == "cascade" else [])
        st.session_state.setdefault("va_asr", va.DEFAULT_ASR if va.DEFAULT_ASR in asr_models
                                    else asr_models[0])  # fmt: skip
        if st.session_state.va_asr not in asr_choices:
            st.session_state.va_asr = asr_models[0]
        if engine in voice_cloud.ENGINES or not captions:  # no ASR here
            asr = ""
            note = (
                tr(
                    "模型自己的轉寫（不另外接辨識模型）",
                    "the model's own transcript (no separate ASR)",
                )
                if engine in voice_cloud.ENGINES
                else tr(
                    "關閉（只用音量偵測判斷你講完）",
                    "off (an energy VAD finds the end of your turn)",
                )
            )
            c1.markdown(f"**{tr('字幕', 'Captions')}**  \n{note}")
        else:
            asr = c1.selectbox(
                tr("語音辨識模型", "Speech recognition")
                if not single
                else tr("串流辨識模型", "Streaming ASR"),
                asr_choices,
                format_func=asr_label,
                help=tr(
                    "**串流**：邊講邊出字，講完約 0.3 秒內定稿（sherpa-onnx、X-ASR、Voxtral 4-bit）。"
                    "**非串流**：偵測到你停頓（約 0.8 秒）後把整句交給模型辨識，講話時看不到字，但通常更準"
                    "（國慶演說評測：Qwen3-ASR-1.7B 2.9%、Voxtral 4-bit 3.7%）；每輪延遲表的「辨識」欄可以直接比較兩種。"
                    if not single
                    else "只列出跟得上即時的串流模型（4-bit Voxtral 可用，全精度版太慢）。"
                    + (
                        "這個模型不會回傳它聽到的字，所以用串流辨識顯示你的字幕；它直接聽你的聲音，"
                        "辨識結果不會送給它。"
                        if single
                        else ""
                    ),
                    "**Streaming**: words appear while you talk, final about 0.3 s after you "
                    "stop (sherpa-onnx, X-ASR, Voxtral 4-bit). **Whole-clip**: once you pause "
                    "(about 0.8 s) the whole utterance goes to the model; nothing shows while you "
                    "talk, but it is usually more accurate (National Day speech: Qwen3-ASR-1.7B "
                    "2.9%, Voxtral 4-bit 3.7%); the latency table's ASR column compares the two."
                    if not single
                    else "Only streaming models that keep up with real time (4-bit Voxtral; the "
                    "full-precision build is too slow)."
                    + (
                        " This model never says what it heard, so the ASR draws your captions; "
                        "the model hears your audio itself, the ASR text is not given to it."
                        if single
                        else ""
                    ),
                ),  # fmt: skip
                **kept("va_asr"),
            )
        if engine in voice_cloud.ENGINES:
            info = voice_cloud.ENGINES[engine]
            llm, voice = info["name"], info["voice"]
            c2.markdown(f"**{tr('語音模型', 'Speech model')}**  \n{info['name']}")
            c3.markdown(f"**{tr('助理的聲音', 'Assistant voice')}**  \n"
                        + tr(f"模型自己的聲音（{info['voice']}）", f"the model's own ({info['voice']})"))  # fmt: skip
            n1, n2 = st.columns([2.2, 1])
            st.session_state.setdefault("va_net", False)
            net = n1.toggle(tr("允許查詢天氣與店家（在本機執行）", "Allow weather and place lookups (run here)"),
                            **kept("va_net"))  # fmt: skip
            st.session_state.setdefault("va_barge", True)
            if info["kind"] == "duplex":
                barge = False
            else:
                barge = n2.toggle(tr("允許插話（barge-in）", "Allow barge-in"), **kept("va_barge"))
            st.warning(
                tr(
                    "**錄音會上傳到雲端**：麥克風或測試腳本的聲音會即時送到 Azure OpenAI（East US 2），"
                    "只供比較測試用。查詢工具在本機執行，只把結果回傳給模型。"
                    + (
                        "gpt-live-1 由背景的 gpt-5.5 呼叫工具；這個部署長對話會變慢、偶爾斷線。"
                        if info["kind"] == "duplex"
                        else "gpt-realtime 由伺服器判斷你講完（靜音 0.5 秒）。"
                    ),
                    "**The audio goes to the cloud**: the microphone or test script is streamed to "
                    "Azure OpenAI (East US 2), for comparison only. The lookups run here; only their "
                    "results go to the model. "
                    + (
                        "gpt-live-1 delegates the tools to gpt-5.5 in the background; this deployment "
                        "slows down in long sessions and sometimes disconnects."
                        if info["kind"] == "duplex"
                        else "gpt-realtime's server decides when you have "
                        "finished (0.5 s of silence)."
                    ),
                ),
                icon=":material/cloud_upload:",
            )
        elif single:
            info = voice_omni.ENGINES[engine]
            llm, voice, net = info["name"], info["voice"], False
            c2.markdown(
                f"**{tr('語音模型', 'Speech model')}**  \n{info['name']}（{info['size_gb']:.1f} GB）"
            )
            c3.markdown(f"**{tr('助理的聲音', 'Assistant voice')}**  \n"
                        + tr(f"模型自己的聲音（{info['voice']}）", f"the model's own ({info['voice']})"))  # fmt: skip
            duplex = info["kind"] == "duplex"
            st.session_state.setdefault("va_barge", True)
            if duplex:
                barge = False
            else:
                barge = st.toggle(tr("允許插話（barge-in）", "Allow barge-in"), **kept("va_barge"),
                                  help=tr("助理說話時你一開口就停下來聽你說。",
                                          "Stop talking as soon as you start speaking."))  # fmt: skip
            st.caption(
                tr(
                    ":material/info: **單一語音模型沒有工具**：問天氣、店家會說查不到。"
                    + (
                        "全雙工：每秒把你的聲音（包括安靜）送給模型，由它決定開口、附和或停下；插話由模型自己處理。"
                        if duplex
                        else "回合式：你講完（停頓偵測）才把這段聲音交給模型。"
                    )
                    + "第一次使用要先載入模型（Qwen3-Omni 約 1 分鐘）。",
                    ":material/info: **A single speech model has no tools**: asked about weather or "
                    "shops, it says it cannot look them up. "
                    + (
                        "Full-duplex: every second of your audio (silence too) goes to the model, "
                        "which decides to talk, back-channel or stop; it handles interruptions itself. "
                        if duplex
                        else "Turn-based: once you finish (pause detection) that stretch "
                        "of audio goes to the model. "
                    )
                    + "The model loads on first use (Qwen3-Omni about a minute).",
                )
            )
        else:
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
            if llm in voice_agent.AUDIO_LLMS:
                st.session_state.setdefault("va_hear", False)
                hear = st.toggle(
                    tr(
                        "讓 LLM 直接聽語音（不用辨識結果）",
                        "Let the LLM hear the audio (not the ASR text)",
                    ),
                    help=tr(
                        "Gemma 4 E4B／E2B 能直接聽聲音：打開後，你講完的那段錄音直接交給 LLM，辨識模型只用來顯示字幕，"
                        "辨識結果不會給它。查資料、先回一句、插話照常。實測（中文測試腳本、E4B）：直接聽 1.1–1.6 秒開口、"
                        "辨識後再回答 1.5–1.6 秒，速度差不多；但直接聽理解較淺，4 句有 1 句聽不懂"
                        "（「我在內湖」回「你在哪湖？」），還會編出天氣。",
                        "Gemma 4 E4B / E2B hear audio themselves: when on, your recording goes to the "
                        "LLM and the ASR only draws the captions (its text is not given to the LLM). "
                        "Lookups, the first line and barge-in work as usual. Measured (Chinese test "
                        "script, E4B): it starts in 1.1-1.6 s hearing the audio and 1.5-1.6 s "
                        "through the ASR, about the same; but hearing it understands less, missing 1 "
                        "of 4 lines and inventing weather.",
                    ),
                    **kept("va_hear"),
                )
            default_voice = next((n for n in voice_names if n.startswith(va.ASSISTANT_VOICE)), None)
            default_voice = next(
                (n for n in voice_names if n == va.ASSISTANT_VOICE_TW), default_voice
            )
            st.session_state.setdefault("va_voice", default_voice or (voice_names or [""])[0])
            voice = c3.selectbox(tr("助理的聲音", "Assistant voice"), voice_names or [""],
                                 format_func=short, **kept("va_voice"))  # fmt: skip
            voice_limits = tr(*tts.limits(tts.find(voice, voices)))
            if voice_limits:
                st.caption(f":material/record_voice_over: {tr('助理的聲音', 'Assistant voice')}："
                           f"{voice_limits}")  # fmt: skip
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

        st.markdown(f"**{tr('架構', 'How it works')}**")
        tts_label = voice_source(tts.find(voice, voices)) if kind == "cascade" else None
        st.graphviz_chart(va.flow_dot(kind, engine, captions=captions, tts_label=tts_label,
                                      pause=asr in va.PAUSE_MODELS, hear=hear),
                          width="content")  # fmt: skip

    demo, script_note = test_script(user_names, short)

    if not single and voice_agent.DEFAULT_LLM not in installed:
        download_control(voice_agent.DEFAULT_LLM, key="va_pull_llm", page=page)
    if not single and llm not in installed:
        st.stop()
    labels = va.labels()
    if single:
        labels |= {"loading_llm": tr("載入語音模型並預熱（第一次較久）…",
                                     "Loading and warming the speech model (slower the first time)…"),
                   "lane_llm": tr("語音模型", "Speech model")}  # fmt: skip

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
                "engine": engine,
                "captions": captions,
                "hear": hear,
            },  # fmt: skip
            "labels": labels,
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
    voice_bench.section()


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
        every = va.raw_scripts()
        with st.expander(tr(f"看所有測試腳本的內容（{len(every)} 個）",
                            f"See every test script ({len(every)})"), icon=":material/list_alt:"):  # fmt: skip
            st.caption(
                tr(
                    "每句台詞依序念給助理聽；「⤷ 插話」那句在助理念查詢結果時插進去。右欄是這句話助理該做的事，"
                    "評測時用來判斷對錯。",
                    "Each line is spoken in turn; the ⤷ barge-in line cuts in while the assistant "
                    "reads a lookup result. The right column is what the assistant should do, "
                    "which the evaluation scores.",
                )
            )
            for item in every:
                st.markdown(f"**{tr(item['name']['zh'], item['name']['en'])}**")
                st.markdown(va.script_table(item))
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
        if not english:  # the Qwen3-TTS Taiwanese clone when it is installed
            user_default = next((n for n in voice_names if n == va.USER_VOICE_TW), user_default)
        st.session_state.setdefault(voice_key, user_default or (voice_names or [""])[0])

        def user_label(name: str) -> str:
            v = tts.find(name)
            weak = v is not None and v.get("test_speaker") is False
            return short(name) + (
                tr("（不建議當測試講者）", " (not a good test speaker)") if weak else ""
            )

        user_voice = s2.selectbox(
            tr("使用者的聲音", "User's voice"),
            voice_names or [""],
            format_func=user_label,
            help=tr(
                "台詞用這個聲音合成，再即時送進串流辨識。中文預設 Qwen3-TTS 台灣腔（複製台灣講者的聲音）：實測串流辨識"
                "錯字率 0.7%，跟 macOS 婷婷一樣清楚。"
                "台詞第一次合成後會存起來，自訂台詞第一次每句要等幾秒。英文預設 Samantha。"
                "Flo 代表 macOS 的 Eloquence 合成器（Eddy、Reed 等 8 個聲音只差音高），音質較機械、辨識較常出錯。"
                "評測數字是用 macOS 婷婷念的台詞測的。",
                "The lines are spoken by this voice and streamed through the ASR. Chinese "
                "defaults to the Qwen3-TTS clone of a Taiwanese speaker: 0.7% errors through the "
                "streaming ASR, as clear as macOS Tingting. "
                "Lines are cached after the first rendering; your own lines take a few seconds "
                "each the first time. English defaults to Samantha. Flo stands for macOS's "
                "Eloquence synthesiser (Eddy, Reed and six more differ only in pitch), robotic "
                "and more often misheard. The evaluation numbers used lines spoken by macOS "
                "Tingting.",
            ),
            **kept(voice_key),
        )
        user_limits = tr(*tts.limits(tts.find(user_voice))) if user_voice else ""
        if user_limits:
            st.caption(f":material/record_voice_over: {tr('使用者的聲音', 'User voice')}："
                       f"{user_limits}")  # fmt: skip
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
