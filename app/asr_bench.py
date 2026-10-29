"""Transcribe mode, Benchmark: every whole-clip ASR model of the evaluation, local and cloud.

Nothing is run here: data/lab/asr_benchmark.json holds the report's whole-clip table
(docs/SPEECH_EVALUATION.md section 2.4). Rows are tinted by where the model runs (ui palette).
"""

import json

import altair as alt
import pandas as pd
import streamlit as st
import ui
from common import PROJECT_DIR
from i18n import tr

DATA = PROJECT_DIR / "data" / "lab" / "asr_benchmark.json"
COLUMNS = {
    "national_day": ("國慶演說", "National Day speech"),
    "meeting": ("合成會議", "Synthetic meeting"),
    "real_meeting": ("真實會議", "Real meeting"),
    "accent_en": ("台灣口音英語", "Accented English"),
    "librispeech": ("英語有聲書", "Audiobook English"),
    "taigi": ("台語", "Taiwanese"),
}


def load() -> dict | None:
    try:
        return json.loads(DATA.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:.1f}%"


def frame(data: dict) -> pd.DataFrame:
    rows = []
    for r in data["rows"]:
        row = {tr("模型", "Model"): r["model"],
               tr("位置", "Where"): ui.where_names()[r["where"] == "cloud"]}  # fmt: skip
        for key, names in COLUMNS.items():
            value = _pct(r[key])
            if key == "real_meeting" and r.get("invalid_real_meeting"):
                value = tr("無效", "invalid")
            row[tr(*names)] = value
        row[tr("處理秒數", "Seconds")] = r["seconds"]
        row[tr("記憶體", "Memory")] = r["memory"] or "—"
        rows.append(row)
    return pd.DataFrame(rows)


def chart(data: dict, key: str) -> alt.Chart:
    df = pd.DataFrame([{"model": r["model"], "where": ui.where_names()[r["where"] == "cloud"],
                        "err": r[key] * 100} for r in data["rows"] if r[key] is not None])  # fmt: skip
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            y=alt.Y("model:N", sort="x", title=None, axis=alt.Axis(labelLimit=260)),
            x=alt.X("err:Q", title=tr("錯字率 %（越短越準）", "Error rate % (shorter = better)")),
            color=alt.Color(
                "where:N", scale=ui.where_scale(), title=None, legend=alt.Legend(orient="bottom")
            ),  # fmt: skip
            tooltip=["model", "where", alt.Tooltip("err:Q", format=".1f")],
        )
        .properties(height=alt.Step(18))
    )


def benchmark_view() -> None:
    data = load()
    if not data:
        st.info(tr("找不到評測結果（data/lab/asr_benchmark.json）。",
                   "No benchmark results found (data/lab/asr_benchmark.json)."))  # fmt: skip
        return
    st.markdown(
        tr(
            f"**{len(data['rows'])} 個整段辨識模型、同一批錄音**（{data['date']}）：中文算字錯率（CER）、"
            "英文算詞錯率（WER），越低越好。藍色＝本機，橘色＝雲端（錄音要上傳）。",
            f"**{len(data['rows'])} whole-clip ASR models on the same recordings** "
            f"({data['date']}): CER for Chinese, WER for English, lower is better. Blue = this "
            "machine, orange = cloud (the audio is uploaded).",
        )
    )
    pick = st.segmented_control(
        tr("圖表", "Chart"), ["national_day", "real_meeting", "accent_en"], required=True,
        format_func=lambda k: tr(*COLUMNS[k]), key="asr_bench_chart",
    ) or "national_day"  # fmt: skip
    st.altair_chart(chart(data, pick), width="stretch")
    st.dataframe(ui.where_rows(frame(data), tr("位置", "Where")), hide_index=True,
                 width="stretch")  # fmt: skip
    st.caption(
        tr(
            "國慶演說 4 分 29 秒、合成會議 88 秒、真實會議 AliMeeting 4 人 5 分鐘、台灣口音英語是副總統帛琉致詞、"
            "有聲書是 LibriSpeech 20 句、台語是教育部辭典 20 句。處理秒數＝國慶演說（雲端含上傳）。"
            "—＝沒測；真實會議「無效」：gpt-4o-transcribe-diarize 把中文轉成英文。",
            "National Day speech 4 min 29 s, synthetic meeting 88 s, real meeting AliMeeting "
            "(4 people, 5 min), accented English = the Vice President's Palau speech, audiobook = "
            "20 LibriSpeech clips, Taiwanese = 20 dictionary sentences. Seconds = the speech "
            "(cloud: upload included). — = not run; invalid: gpt-4o-transcribe-diarize turned the "
            "Chinese meeting into English.",
        )
    )
