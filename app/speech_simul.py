"""Simultaneous mode of the speech-model page: interpreting while the speaker talks.

Nothing is run here: the page reads data/lab/simul_benchmark.json, exported by
scripts/speech_eval/run_simul.py. Unlike Translate (a recording translated whole, or cut into
fixed chunks), the systems here get the audio at the speaker's pace and are scored on how far
the English trails the speaker as well as on chrF / BLEU. The models not run yet are listed
with the reason (a key to apply for, an NVIDIA GPU).
"""

import json

import altair as alt
import pandas as pd
import streamlit as st
from common import PROJECT_DIR
from i18n import tr

DATA = PROJECT_DIR / "data" / "lab" / "simul_benchmark.json"


def load() -> dict | None:
    try:
        return json.loads(DATA.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def frame(data: dict) -> pd.DataFrame:
    rows = []
    for r in data["rows"]:
        rows.append({
            "system": tr(*r["label"]),
            "where": tr("雲端", "Cloud") if r["where"] == "cloud" else tr("本機", "Local"),
            "out": tr("英文語音＋字幕", "Speech + captions") if r.get("speech_out")
                   else tr("英文字幕", "Captions"),
            "first_s": r["first_s"], "lag_avg_s": r["lag_avg_s"], "lag_p90_s": r["lag_p90_s"],
            "tail_s": r["tail_s"], "chrf": r["chrf"], "bleu": r["bleu"],
            "note": tr(*r["note"]),
        })  # fmt: skip
    return pd.DataFrame(rows)


def chart(df: pd.DataFrame) -> alt.Chart:
    lag = tr("平均落後講者（秒，越左越即時）", "Average lag (s, left = sooner)")
    base = alt.Chart(df).encode(
        x=alt.X("lag_avg_s:Q", title=lag, scale=alt.Scale(zero=True)),
        y=alt.Y("chrf:Q", title=tr("chrF（越上越準）", "chrF (up = better)"),
                scale=alt.Scale(zero=False)),
        color=alt.Color("where:N", title=None,
                        scale=alt.Scale(range=["#2a78d6", "#ea8a3b"])),
        tooltip=["system", "first_s", "lag_avg_s", "tail_s", "chrf", "bleu"],
    )  # fmt: skip
    points = base.mark_circle(size=110)
    labels = base.mark_text(align="left", dx=8, fontSize=12).encode(text="system")
    return (points + labels).properties(height=320)


def simul_page(url: str | None = None, status: dict | None = None) -> None:
    data = load()
    if not data or not data.get("rows"):
        st.info(tr("找不到評測結果（data/lab/simul_benchmark.json）。",
                   "No benchmark results found (data/lab/simul_benchmark.json)."))  # fmt: skip
        return
    st.markdown(
        tr(
            f"**國慶演說前 {data['clip_s']:.0f} 秒（4 段）照講者的速度送進去，邊聽邊翻成英文**"
            f"（{data['date']}）。對照總統府官方英譯算 chrF／BLEU，並看英文落後講者幾秒。",
            f"**The first {data['clip_s']:.0f} s of the National Day address (4 paragraphs), fed "
            f"at the speaker's pace and interpreted into English as it goes** ({data['date']}). "
            "Scored with chrF / BLEU against the official English, and by how far the English "
            "trails the speaker.",
        )
    )
    df = frame(data)
    st.altair_chart(chart(df), width="stretch")
    st.dataframe(
        df.rename(columns={
            "system": tr("做法", "System"), "where": tr("位置", "Where"),
            "out": tr("輸出", "Output"), "first_s": tr("第一句英文（秒）", "First English (s)"),
            "lag_avg_s": tr("平均落後（秒）", "Avg lag (s)"),
            "lag_p90_s": tr("落後 P90（秒）", "Lag P90 (s)"),
            "tail_s": tr("講完再等（秒）", "After the end (s)"), "chrf": "chrF", "bleu": "BLEU",
            "note": tr("說明", "Notes"),
        }),
        hide_index=True,
        width="stretch",
    )  # fmt: skip
    st.caption(
        tr(
            "落後怎麼算：假設講者說到全文的幾成，對應譯文也該出到幾成（依實際說話時間、不算停頓）；"
            "每個英文字出現的時間減去講者說到那裡的時間，取平均與 P90。"
            "第一句英文：從開口到第一個英文字；"
            "講完再等：講者最後一個字到最後一個英文字。地端做法照真實速度逐段計時（辨識＋翻譯的實測秒數）。"
            "官方英譯是書面稿、會重組句子，chrF 只適合互相比較。"
            "腳本：scripts/speech_eval/run_simul.py。",
            "Lag: the English is assumed to follow the speaker in proportion (by speaking time, "
            "pauses not counted); each English word's arrival minus when the speaker reached that "
            "point, averaged and at P90. First English: speech start to the first English word; "
            "after the end: speaker's last word to the last English word. Local systems are "
            "timed span by span with measured ASR + MT seconds. The official English is a "
            "written text that restructures sentences, so chrF is for comparison only. "
            "Script: scripts/speech_eval/run_simul.py.",
        )
    )
    with st.expander(tr("還沒測的同步口譯模型", "Simultaneous models not run yet")):
        st.dataframe(
            pd.DataFrame([
                {tr("模型", "Model"): tr(zh, en), tr("原因", "Why"): tr(wz, we)}
                for zh, en, wz, we in data.get("not_run", [])
            ]),
            hide_index=True,
            width="stretch",
        )  # fmt: skip
