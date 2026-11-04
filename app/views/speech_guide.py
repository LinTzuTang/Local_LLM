"""Pick a speech model: which one to use in which situation, local and cloud side by side, from
the measured results (speech_pick, data/lab/recommendations.json). Its own page so it does not
sit among the test modes of views/speech.py; nothing runs here, so no gateway is needed."""

import speech_pick
import streamlit as st
import ui
from i18n import tr

ui.page_header(
    "recommend",
    tr("依情境選語音模型", "Pick a speech model"),
    tr(
        "各種語音任務（逐字稿、即時字幕、翻譯、同步口譯、講者辨識、文字轉語音、語音助理）該用哪個模型："
        "本機與雲端並排，依實測結果整理。要自己測，請到「語音模型比較」。",
        "Which model for each speech task (transcription, live captions, translation, "
        "simultaneous interpreting, diarization, text to speech, voice assistant), local and "
        "cloud side by side, from the measured results. To run the tests yourself, open "
        "Compare speech models.",
    ),
)
st.page_link("views/speech.py", label=tr("到語音模型比較自己測", "Run the tests: Compare speech models"),
             icon=":material/graphic_eq:")  # fmt: skip
speech_pick.pick_page()
