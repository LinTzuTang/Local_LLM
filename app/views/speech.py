"""Compare speech models: transcription (with noise / telephone conditions), speech translation
and meeting diarization. The modes live in speech_compare, speech_translate and speech_meeting;
the language-model comparison is views/compare.py."""

import speech_common as sc
import speech_compare
import speech_meeting
import speech_translate
import streamlit as st
import ui
from common import gateway_url, kept
from i18n import tr

ui.page_header(
    "graphic_eq",
    tr("語音模型比較", "Compare speech models"),
    tr(
        "同一段錄音交給多個語音模型，並排比較逐字稿、速度與錯誤率；也能測噪音與電話音質、"
        "中翻英，以及會議裡誰在什麼時候說話。全部在本機執行，比較在背景進行，可以切換頁面。",
        "Give the same recording to several speech models and compare transcripts, speed and "
        "error rates side by side; also test noise and telephone audio, Chinese-to-English "
        "translation, and who spoke when in a meeting. Everything runs on this machine, in the "
        "background, so you can switch pages.",
    ),
)

MODES = {
    "transcribe": (tr("逐字稿", "Transcribe"), "subtitles"),
    "translate": (tr("翻成英文", "Translate"), "translate"),
    "meeting": (tr("會議講者", "Meeting"), "record_voice_over"),
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
}
st.caption(blurbs[mode])

url = gateway_url()
status = sc.gateway_status(url)
if status is None:
    st.stop()
if mode == "transcribe":
    speech_compare.transcribe_page(url, status)
elif mode == "translate":
    speech_translate.translate_page(url, status)
else:
    speech_meeting.meeting_page(url, status)
