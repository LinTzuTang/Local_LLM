"""ElevenLabs (cloud): its speech features to try here, one tab each (local_llm/elevenlabs.py).

Everything on this page sends text or audio to ElevenLabs, so it works only with the top bar's
cloud switch on. The measured comparison with the local models (2026-11-05,
scripts/speech_eval/run_elevenlabs.py) is summarised at the top; ElevenLabs voices are also
selectable in Text to speech and the live voice assistant, and Scribe v2 in the meeting-speaker
comparison.
"""

import re
import time

import pandas as pd
import streamlit as st
import ui
from common import cloud_enabled
from i18n import tr

from local_llm import elevenlabs as el
from local_llm import media_tools

SAMPLES = media_tools.SAMPLES_DIR
SAMPLE_AUDIO = ["meeting_zh.wav", "national_day_2026_zh.wav", "alimeeting_r8009_zh.flac"]
AUDIO_TYPES = ["wav", "mp3", "m4a", "flac", "ogg", "webm"]
TTS_SAMPLE = "各位早安，現在開始第四十二週的產銷會議。本週 CNC 產線稼動率百分之八十七，比上週提高三個百分點。"
TAG_SAMPLE = ("[calm] 聽說她是在一個星期二失蹤的。[building tension] 沒有留下紙條，沒有掙扎，"
              "沒有任何痕跡。[whispers] 但大家都知道。")  # fmt: skip


def voice_label(name: str) -> str:
    _, zh, en = el.VOICES[name]
    return f"{name}（{tr(zh, en)}）"


def voice_pick(key: str, label: str | None = None) -> str:
    """A Taiwanese-accent library voice, or any voice ID typed in; returns the voice ID."""
    names = list(el.VOICES) + ["__id__"]
    pick = st.selectbox(label or tr("聲音（聲音庫的台灣口音聲音）", "Voice (Taiwanese-accent library voices)"),
                        names, key=key,
                        format_func=lambda n: tr("自己輸入聲音 ID…", "Type a voice ID…") if n == "__id__"
                        else voice_label(n))  # fmt: skip
    if pick == "__id__":
        return st.text_input(tr("聲音 ID", "Voice ID"), key=key + "_id").strip()
    return el.VOICES[pick][0]


def audio_source(key: str) -> tuple[bytes | None, str]:
    """A sample recording, an upload or the microphone: (bytes, name)."""
    kinds = {"sample": tr("範例錄音", "Sample"), "upload": tr("上傳", "Upload"),
             "mic": tr("麥克風錄音", "Record")}  # fmt: skip
    kind = st.segmented_control(tr("聲音來源", "Audio"), list(kinds), format_func=kinds.get,
                                default="sample", key=key + "_kind")  # fmt: skip
    if kind == "upload":
        f = st.file_uploader(tr("上傳音檔", "Audio file"), type=AUDIO_TYPES, key=key + "_file")
        return (f.getvalue(), f.name) if f else (None, "")
    if kind == "mic":
        f = st.audio_input(tr("按下開始錄音", "Record"), key=key + "_mic")
        return (f.getvalue(), "mic.wav") if f else (None, "")
    name = st.selectbox(tr("範例", "Sample"), SAMPLE_AUDIO, key=key + "_sample")
    data = (SAMPLES / name).read_bytes()
    st.audio(data)
    return data, name


def run(key: str, fn, *args, **kwargs):
    """Call ElevenLabs, keep the result and how long it took in the session, show errors."""
    started = time.perf_counter()
    try:
        out = fn(*args, **kwargs)
    except Exception as exc:  # shown, not raised: a missing permission is a normal answer here
        st.session_state[key] = {"error": str(getattr(exc, "body", None) or exc)}
        return
    st.session_state[key] = {"out": out, "seconds": round(time.perf_counter() - started, 2)}


def result(key: str):
    """The kept result of `key` (None when there is none or it failed; the error is shown)."""
    r = st.session_state.get(key)
    if not r:
        return None
    if "error" in r:
        st.error(r["error"][:600], icon=":material/error:")
        return None
    st.caption(tr(f"花了 {r['seconds']} 秒（含上傳與下載）", f"{r['seconds']} s including transfer"))
    return r["out"]


def api(text: str) -> None:
    st.caption(f":material/api: `{text}`")


def summary() -> None:
    with st.expander(tr("實測結果摘要（11-05，和地端同一套資料比較）", "Measured (2026-11-05, same data as the local models)"),
                     icon=":material/insights:"):  # fmt: skip
        rows = [
            (tr("文字轉語音 v4 Turbo", "TTS v4 Turbo"), "1.6–1.9%，0.12 s", tr("地端 Qwen3-TTS 台灣腔 1.6%、0.16 秒", "local Qwen3-TTS Taiwanese 1.6%, 0.16 s")),
            (tr("文字轉語音 v4", "TTS v4"), "1.4–1.9%，0.70–0.81 s", ""),
            (tr("文字轉語音 Flash v2.5", "TTS Flash v2.5"), tr("10–19%，3 個聲音有句子失控", "10-19%, 3 voices ran away"), ""),
            (tr("整段辨識 Scribe v2", "Scribe v2"), tr("演說 0.8%、真實會議 21.5%", "speech 0.8%, real meeting 21.5%"), tr("地端 Qwen3-ASR-1.7B 0.4%、18.7%", "local Qwen3-ASR-1.7B 0.4%, 18.7%")),
            (tr("即時辨識 Scribe v2 Realtime", "Scribe v2 Realtime"), tr("乾淨 1.6%，但約 13 秒才定稿", "1.6% clean, but final after about 13 s"), tr("地端 Voxtral 3.9%、0.65 秒", "local Voxtral 3.9%, 0.65 s")),
            (tr("講者辨識", "Diarization"), tr("2 人 4.5–6.4%、4 人 34–43%", "2 speakers 4.5-6.4%, 4 speakers 34-43%"), tr("地端 Sortformer v2 4.5–9.6%", "local Sortformer v2 4.5-9.6%")),
            (tr("去背景音再辨識", "Isolator before ASR"), tr("人聲噪音 0 dB 9.3% → 76.7%（變差）", "talkers at 0 dB 9.3% -> 76.7% (worse)"), ""),
        ]  # fmt: skip
        st.dataframe(pd.DataFrame(rows, columns=[tr("功能", "Feature"), "ElevenLabs", tr("對照", "Compared with")]),
                     hide_index=True, width="stretch")  # fmt: skip
        st.caption(tr("錯字率越低越好；腳本 scripts/speech_eval/run_elevenlabs.py。",
                      "Lower error rates are better; script scripts/speech_eval/run_elevenlabs.py."))  # fmt: skip


def tab_tts() -> None:
    model = st.selectbox(tr("模型", "Model"), list(el.TTS_MODELS), key="el_tts_model",
                         format_func=lambda m: tr(*el.TTS_MODELS[m]))  # fmt: skip
    voice = voice_pick("el_tts_voice")
    tags = model in ("eleven_v4", "eleven_v3")
    if tags:
        st.caption(tr("這個模型可以在句子裡插語氣標籤，例如 [whispers]、[laughs]、[calm]、[excited]。",
                      "This model takes inline audio tags such as [whispers], [laughs], [calm], [excited]."))  # fmt: skip
    text = st.text_area(tr("要念的文字", "Text"), TAG_SAMPLE if tags else TTS_SAMPLE, key=f"el_tts_text_{tags}")
    c1, c2 = st.columns(2)
    language = c1.selectbox(tr("語言", "Language"), ["zh", "en", None], key="el_tts_lang",
                            format_func=lambda x: {"zh": tr("中文", "Chinese"), "en": tr("英文", "English"),
                                                   None: tr("自動", "Auto")}[x])  # fmt: skip
    stability = None
    if model != "eleven_v4_turbo":
        stability = c2.slider(tr("穩定度（低＝起伏大）", "Stability (low = more expressive)"), 0.0, 1.0, 0.5, 0.05,
                              key="el_tts_stab")  # fmt: skip
    api("WebSocket /v1/text-to-dialogue/stream-input" if model == "eleven_v4_turbo"
        else f"POST /v1/text-to-speech/{{voice}}/stream  model_id={model}")  # fmt: skip
    if st.button(tr("合成", "Synthesize"), type="primary", icon=":material/play_arrow:", key="el_tts_go",
                 disabled=not text.strip() or not voice):  # fmt: skip
        run("el_tts", el.speak, text, voice, model, language, stability)
    out = result("el_tts")
    if out:
        st.audio(out["wav"], format="audio/wav")
        m1, m2, m3 = st.columns(3)
        m1.metric(tr("第一段聲音", "First audio"), f"{out['first_s']} s")
        m2.metric(tr("整段合成", "Whole text"), f"{out['total_s']} s")
        m3.metric(tr("聲音長度", "Audio"), f"{out['seconds']} s")


def tab_dialogue() -> None:
    st.caption(tr("一次產生多人對話，語氣與停頓會接在一起（Text to Dialogue）。", "Several voices in one take (Text to Dialogue)."))
    model = st.selectbox(tr("模型", "Model"), el.DIALOGUE_MODELS, key="el_dlg_model")
    default = pd.DataFrame({"voice": ["Anna Su", "Yu", "Anna Su"],
                            "text": ["[cheerful] 早安！今天的產銷會議要開始了嗎？",
                                     "對，先報告生產進度，本週稼動率百分之八十七。",
                                     "[impressed] 比上週還高耶，太好了。"]})  # fmt: skip
    table = st.data_editor(default, num_rows="dynamic", width="stretch", key="el_dlg_lines",
                           column_config={"voice": st.column_config.SelectboxColumn(tr("聲音", "Voice"),
                                                                                    options=list(el.VOICES)),
                                          "text": st.column_config.TextColumn(tr("台詞", "Line"), width="large")})  # fmt: skip
    lines = [(el.VOICES[r.voice][0], str(r.text)) for r in table.itertuples() if r.voice in el.VOICES and r.text]
    api(f"POST /v1/text-to-dialogue  model_id={model}")
    if st.button(tr("產生對話", "Make the dialogue"), type="primary", icon=":material/forum:", key="el_dlg_go",
                 disabled=not lines):  # fmt: skip
        run("el_dlg", el.dialogue, lines, model)
    out = result("el_dlg")
    if out:
        st.audio(out, format="audio/mpeg")


def tab_stt() -> None:
    data, name = audio_source("el_stt")
    c1, c2, c3 = st.columns(3)
    diar = c1.toggle(tr("分講者", "Diarize"), True, key="el_stt_diar")
    count = c2.number_input(tr("人數（0＝自己判斷）", "Speakers (0 = estimate)"), 0, 32, 0, key="el_stt_n")
    events = c3.toggle(tr("標記笑聲等聲音", "Tag audio events"), False, key="el_stt_ev")
    api(f"POST /v1/speech-to-text  model_id={el.STT_MODEL} diarize={str(diar).lower()}")
    if st.button(tr("轉文字", "Transcribe"), type="primary", icon=":material/subtitles:", key="el_stt_go",
                 disabled=not data):  # fmt: skip
        run("el_stt", el.transcribe, data, "zho", diar, count or None, events)
    out = result("el_stt")
    if out:
        st.text_area(tr("文字（Scribe 輸出簡體）", "Text (Scribe writes Simplified)"), out["text"], height=160)
        if out["segments"]:
            segs = pd.DataFrame(out["segments"])
            st.caption(tr(f"{segs['speaker'].nunique()} 位講者、{len(segs)} 段", f"{segs['speaker'].nunique()} speakers, {len(segs)} turns"))
            st.dataframe(segs, hide_index=True, width="stretch", height=260)


def tab_change() -> None:
    st.caption(tr("把錄音裡的話換成另一個聲音說（語氣、停頓保留）。", "The same words, said in another voice."))
    data, _ = audio_source("el_sts")
    voice = voice_pick("el_sts_voice", tr("換成這個聲音", "Target voice"))
    clean = st.toggle(tr("先去背景音", "Remove background noise first"), False, key="el_sts_clean")
    api(f"POST /v1/speech-to-speech/{{voice}}  model_id={el.STS_MODEL}")
    if st.button(tr("換聲", "Change voice"), type="primary", icon=":material/swap_horiz:", key="el_sts_go",
                 disabled=not data or not voice):  # fmt: skip
        run("el_sts", el.change_voice, data, voice, clean)
    out = result("el_sts")
    if out:
        st.audio(out, format="audio/mpeg")


def tab_isolate() -> None:
    st.warning(tr("實測：放在語音辨識前面反而變差（人聲噪音 0 dB：Qwen3-ASR 9.3% → 76.7%；辦公室噪音 0.8% → 8.6%）。"
                  "背景有人講話時它分不出誰是主角。適合給人聽，不適合給辨識模型。",
                  "Measured: it makes speech recognition worse (talkers at 0 dB: Qwen3-ASR 9.3% -> 76.7%; office "
                  "noise 0.8% -> 8.6%); with talkers in the background it cannot tell whose voice to keep. "
                  "For people to listen to, not for recognisers."), icon=":material/info:")  # fmt: skip
    data, _ = audio_source("el_iso")
    api("POST /v1/audio-isolation")
    if st.button(tr("去背景音", "Isolate the voice"), type="primary", icon=":material/noise_control_off:",
                 key="el_iso_go", disabled=not data):  # fmt: skip
        run("el_iso", el.isolate, data)
    out = result("el_iso")
    if out:
        st.audio(out, format="audio/mpeg")


def tab_sfx() -> None:
    prompt = st.text_input(tr("描述音效（英文較準）", "Describe the sound"), "office door opening, footsteps, keyboard typing",
                           key="el_sfx_prompt")  # fmt: skip
    c1, c2 = st.columns(2)
    seconds = c1.slider(tr("長度（0＝自動）", "Length (0 = auto)"), 0.0, 22.0, 0.0, 0.5, key="el_sfx_len")
    loop = c2.toggle(tr("可無縫循環", "Seamless loop"), False, key="el_sfx_loop")
    api("POST /v1/sound-generation")
    if st.button(tr("產生音效", "Make the sound"), type="primary", icon=":material/graphic_eq:", key="el_sfx_go",
                 disabled=not prompt.strip()):  # fmt: skip
        run("el_sfx", el.sound_effect, prompt, seconds or None, loop)
    out = result("el_sfx")
    if out:
        st.audio(out, format="audio/mpeg")


def tab_music() -> None:
    prompt = st.text_area(tr("描述音樂", "Describe the music"),
                          "Calm corporate background music, soft piano and strings, 90 BPM", key="el_music_prompt")  # fmt: skip
    c1, c2 = st.columns(2)
    seconds = c1.slider(tr("長度（秒）", "Length (s)"), 10, 120, 20, 5, key="el_music_len")
    instrumental = c2.toggle(tr("純音樂（不要人聲）", "Instrumental only"), True, key="el_music_inst")
    st.caption(tr("音樂用額度較多：每次大約和念同樣長度的文字相當或更多。", "Music uses more credits than speech of the same length."))
    api("POST /v1/music")
    if st.button(tr("產生音樂", "Compose"), type="primary", icon=":material/music_note:", key="el_music_go",
                 disabled=not prompt.strip()):  # fmt: skip
        run("el_music", el.music, prompt, seconds, instrumental)
    out = result("el_music")
    if out:
        st.audio(out, format="audio/mpeg")


def tab_design() -> None:
    st.caption(tr("用文字描述產生新聲音，一次給 3 個候選試聽。要存成可用的聲音需要 voices_write 權限（這把金鑰沒有）。",
                  "A new voice from a description, three previews. Saving one needs the voices_write permission, "
                  "which this key lacks."))  # fmt: skip
    desc = st.text_area(tr("描述聲音", "Describe the voice"), "一位語氣溫和、咬字清楚的年輕台灣女性，說台灣華語，適合當客服助理",
                        key="el_design_desc")  # fmt: skip
    text = st.text_area(tr("試聽文字（至少 100 字；留空就自動產生）", "Preview text (100+ characters; empty = generated)"), "",
                        key="el_design_text")  # fmt: skip
    api("POST /v1/text-to-voice/design  model_id=eleven_ttv_v3")
    if st.button(tr("產生候選聲音", "Design"), type="primary", icon=":material/auto_awesome:", key="el_design_go",
                 disabled=not desc.strip()):  # fmt: skip
        run("el_design", el.design_voice, desc, text or None)
    out = result("el_design")
    if out:
        st.caption(out["text"])
        for k, p in enumerate(out["previews"], 1):
            st.markdown(f"**{tr('候選', 'Preview')} {k}** · `{p['id']}`")
            st.audio(p["mp3"], format="audio/mpeg")


def sample_script(name: str) -> str:
    path = SAMPLES / (name.rsplit(".", 1)[0] + ".txt")
    if not path.exists():
        return ""
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln and not ln.startswith("#")]
    return "\n".join(re.sub(r"^\[[^\]]+\]\s*", "", ln) for ln in lines)


def tab_align() -> None:
    st.caption(tr("給錄音和它的文字稿，算出每個字在第幾秒說（做字幕、對嘴用）。", "When each word of a script is said in the audio."))
    data, name = audio_source("el_align")
    text = st.text_area(tr("文字稿", "Script"), sample_script(name) if name in SAMPLE_AUDIO else "",
                        height=140, key=f"el_align_text_{name}")  # fmt: skip
    api("POST /v1/forced-alignment")
    if st.button(tr("對齊", "Align"), type="primary", icon=":material/timeline:", key="el_align_go",
                 disabled=not data or not text.strip()):  # fmt: skip
        run("el_align", el.align, data, text)
    out = result("el_align")
    if out:
        st.dataframe(pd.DataFrame(out), hide_index=True, width="stretch", height=300)


def tab_dub() -> None:
    st.warning(tr("這把金鑰沒有 dubbing_write 權限，配音會被拒絕；要在 ElevenLabs 網站把金鑰的配音權限打開。",
                  "This key lacks the dubbing_write permission; turn it on for the key on the ElevenLabs website."),
               icon=":material/lock:")  # fmt: skip
    data, _ = audio_source("el_dub")
    target = st.selectbox(tr("配成", "Dub into"), ["en", "ja", "ko", "es", "fr", "de"], key="el_dub_to")
    api("POST /v1/dubbing  (→ GET /v1/dubbing/{id} → GET /v1/dubbing/{id}/audio/{lang})")
    if st.button(tr("配音", "Dub"), icon=":material/translate:", key="el_dub_go", disabled=not data):
        run("el_dub", el.dub, data, target)
    out = result("el_dub")
    if out:
        st.audio(out, format="audio/mpeg")


ui.page_header("record_voice_over", tr("ElevenLabs（雲端）", "ElevenLabs (cloud)"),
               tr("直接試 ElevenLabs 的語音功能：文字轉語音、多人對話、轉文字、換聲、去背景音、音效、音樂、聲音設計、文字對齊、配音。",
                  "Try ElevenLabs' speech features: text to speech, dialogue, speech to text, voice changer, "
                  "voice isolator, sound effects, music, voice design, forced alignment, dubbing."))  # fmt: skip
if not el.available():
    st.error(tr("沒有 ElevenLabs 金鑰：在專案資料夾執行 `npx -y @elevenlabs/cli@latest onboard init`，它會把 "
                "ELEVENLABS_API_KEY 寫進 .env。", "No ElevenLabs key: run `npx -y @elevenlabs/cli@latest onboard init` "
                "in the project folder; it writes ELEVENLABS_API_KEY into .env."), icon=":material/key_off:")  # fmt: skip
    st.stop()
summary()
if not cloud_enabled():
    st.info(tr("這頁每個功能都會把文字或錄音上傳到 ElevenLabs。要使用請打開右上角的「雲端模型」。",
               "Every feature here uploads text or audio to ElevenLabs. Turn on Cloud at the top right to use them."),
            icon=":material/cloud_off:")  # fmt: skip
    st.stop()
try:
    used, limit = el.usage()
    st.caption(tr(f":material/data_usage: 這個月已用 {used:,}／{limit:,} 額度", f":material/data_usage: {used:,} of {limit:,} credits used this month"))
except Exception:  # the page still works without the usage line
    pass
st.caption(tr("即時辨識（Scribe v2 Realtime）與語音助理（Agents）還沒接上這頁；即時辨識的實測見上方摘要。",
              "Realtime transcription and Agents are not on this page yet; see the summary for realtime results."))  # fmt: skip

TABS = [(":material/record_voice_over:", "文字轉語音", "Text to speech", tab_tts),
        (":material/forum:", "多人對話", "Dialogue", tab_dialogue),
        (":material/subtitles:", "語音轉文字", "Speech to text", tab_stt),
        (":material/swap_horiz:", "換聲", "Voice changer", tab_change),
        (":material/noise_control_off:", "去背景音", "Voice isolator", tab_isolate),
        (":material/graphic_eq:", "音效", "Sound effects", tab_sfx),
        (":material/music_note:", "音樂", "Music", tab_music),
        (":material/auto_awesome:", "聲音設計", "Voice design", tab_design),
        (":material/timeline:", "文字對齊", "Forced alignment", tab_align),
        (":material/translate:", "配音", "Dubbing", tab_dub)]  # fmt: skip
for tab, (_, _, _, fn) in zip(st.tabs([f"{i} {tr(zh, en)}" for i, zh, en, _ in TABS]), TABS, strict=True):
    with tab:
        fn()
