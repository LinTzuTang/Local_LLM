"""Compare speech models: transcription (with noise / telephone conditions), speech translation,
meeting diarization, simulated live captions, text to speech and a voice-chat assistant. The
modes live in speech_compare, speech_translate, speech_meeting, speech_live, speech_tts and
speech_voice; the language-model comparison is views/compare.py."""

import speech_common as sc
import speech_compare
import speech_live
import speech_meeting
import speech_translate
import speech_tts
import speech_voice
import streamlit as st
import ui
from common import gateway_url, kept
from i18n import tr

ui.page_header(
    "graphic_eq",
    tr("語音模型比較", "Compare speech models"),
    tr(
        "同一段錄音交給多個語音模型，並排比較逐字稿、速度與錯誤率；也能測噪音與電話音質、中翻英、"
        "會議裡誰在什麼時候說話、即時字幕的延遲、文字轉語音的聲音，以及用說的跟本機助理對話。"
        "全部在本機執行，比較在背景進行，可以切換頁面。",
        "Give the same recording to several speech models and compare transcripts, speed and "
        "error rates side by side; also test noise and telephone audio, Chinese-to-English "
        "translation, who spoke when in a meeting, live-caption latency, text-to-speech voices "
        "and a spoken conversation with a local assistant. Everything runs on this machine, in "
        "the background, so you can switch pages.",
    ),
)

MODES = {
    "transcribe": (tr("逐字稿", "Transcribe"), "subtitles"),
    "translate": (tr("翻成英文", "Translate"), "translate"),
    "meeting": (tr("會議講者", "Meeting"), "record_voice_over"),
    "live": (tr("串流（即時）", "Live"), "sensors"),
    "tts": (tr("文字轉語音", "Text to speech"), "text_to_speech"),
    "voice_chat": (tr("語音對話", "Voice chat"), "forum"),
}
left, right = st.columns([3, 1], vertical_alignment="bottom")
st.session_state.setdefault("speech_mode", "transcribe")
mode = (
    left.segmented_control(
        tr("模式", "Mode"),
        list(MODES),
        format_func=lambda m: f":material/{MODES[m][1]}: {MODES[m][0]}",
        label_visibility="collapsed",
        **kept("speech_mode"),
    )
    or "transcribe"
)
with right:
    sc.s2tw_toggle()

blurbs = {
    "transcribe": tr(
        "同一段錄音依序交給每個語音辨識模型（一次一個，時間才公平）；填入原稿時計算字錯率（CER）。"
        "打開「噪音與電話音質」可以看模型在吵雜或電話線上的表現。",
        "The same recording goes to each speech-recognition model in turn (one at a time, so "
        "timings are fair); with a reference transcript you also get the character error rate "
        "(CER). Turn on noise and telephone conditions to see how models cope with a noisy room "
        "or a phone line.",
    ),
    "translate": tr(
        "中文語音翻成英文的三種做法：先辨識再用 LLM 翻譯（串接）、音訊 LLM 直接聽音訊寫英文、"
        "Whisper 內建翻譯。比較譯文、等多久出現第一段文字，有參考譯文時計算 chrF／BLEU；"
        "也能模擬即時口譯，一段一段送出。",
        "Three ways to turn Chinese speech into English: transcribe then translate with an LLM "
        "(cascade), an audio LLM that hears the audio and writes English (direct), and "
        "Whisper's built-in translation. Compare the English, how soon text appears and, with "
        "a reference, chrF / BLEU; or simulate live interpreting, chunk by chunk.",
    ),
    "meeting": tr(
        "說話者分段模型找出誰在什麼時候說話，再和語音辨識合併成「講者 1：…」的逐字稿；"
        "有參考答案時計算說話者分段錯誤率（DER）。",
        "A diarization model finds who spoke when; merged with speech recognition it gives a "
        "'Speaker 1: …' transcript, and with a reference the diarization error rate (DER).",
    ),
    "live": tr(
        "即時字幕三種做法：切段模擬（錄音切成 1–5 秒依序送給整段辨識模型）、真串流（串流模型每 "
        "0.1–0.3 秒收一小段、邊收邊出字），以及麥克風即時（對著麥克風講，字幕當場出現）。看文字"
        "晚多久出現、定稿要等多久、部分結果改了多少字，以及最後的錯誤率；「評測總表」列出本機與雲端串流模型的實測結果。",
        "Live captions three ways: chunked simulation (the recording is cut into 1–5 s "
        "chunks for whole-clip models), true streaming (streaming models get 0.1–0.3 s at a "
        "time and write text as audio arrives) and live microphone (speak and watch the "
        "captions appear). See how soon text appears, how long until it is final, how much "
        "partial text gets revised, and the final error rate; Benchmark lists the measured "
        "results of local and cloud streaming models.",
    ),
    "tts": tr(
        "同一段文字用不同聲音念出來（macOS 內建語音，本機免費），比較合成速度，並用語音辨識"
        "「回聽」合成結果、和原文比對錯誤率，看機器聽不聽得懂。",
        "Speak the same text with different voices (macOS's built-in voices, local and free), "
        "compare synthesis speed, and transcribe the result back with a speech model "
        "(round trip) to score how intelligible it is against the input text.",
    ),
    "voice_chat": tr(
        "用說的問問題：語音辨識 → LLM 回答 → 語音合成念出來，全部在本機；或讓 Gemma 4 直接聽錄音回答。"
        "每一步花多少時間都列出來，可以連續對話幾輪。",
        "Ask a question out loud: speech recognition → an LLM answers → speech synthesis reads "
        "it, all on this machine; or let Gemma 4 hear the recording and answer. Each stage's "
        "time is shown, and the conversation can go on for a few turns.",
    ),
}
st.caption(blurbs[mode])

url = gateway_url()
status = sc.gateway_status(url)
if status is None:
    st.stop()
pages = {
    "transcribe": speech_compare.transcribe_page,
    "translate": speech_translate.translate_page,
    "meeting": speech_meeting.meeting_page,
    "live": speech_live.live_page,
    "tts": speech_tts.tts_page,
    "voice_chat": speech_voice.voice_page,
}
pages.get(mode, speech_compare.transcribe_page)(url, status)
