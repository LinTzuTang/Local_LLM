"""Live mode, results: the streaming benchmark, local and cloud models side by side.

Nothing is run here: the page reads data/lab/streaming_benchmark.json, exported from the
speech evaluation (local streaming models through the gateway, timing simulated from measured
compute; Azure OpenAI Realtime models streamed at real time, so their latency includes the
network). One row per model with the error rate on each recording and the latency on the
national-day speech; the detail table has every run.
"""

import json
import statistics

import pandas as pd
import speech_common as sc
import streamlit as st
from common import PROJECT_DIR
from i18n import tr

DATA = PROJECT_DIR / "data" / "lab" / "streaming_benchmark.json"
LATENCY_SAMPLE = "national_day_2026_zh.wav"
NAMES = {  # models outside the gateway catalog
    "kyutai-stt-1b-en_fr": ("Kyutai STT 1B（英文、法文）", "Kyutai STT 1B (English, French)"),
}


def CHUNK_NOTES() -> str:  # noqa: N802 - text, filled in from the chunked evaluation
    return ""


def load() -> dict | None:
    try:
        return json.loads(DATA.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def sample_names() -> dict:
    return {
        "national_day_2026_zh.wav": tr("國慶演說", "National Day speech"),
        "meeting_zh.wav": tr("合成會議", "Synthetic meeting"),
        "alimeeting_r8001_zh.flac": tr("真實會議", "Real meeting"),
        "palau_hsiao_2026_en.wav": tr("英語致詞", "English speech"),
    }


def model_name(model: str, info: dict, labels: dict) -> str:
    if info.get("location") == "cloud":
        mode = tr("（伺服器切句）", " (server VAD)") if info.get("mode") == "server_vad" else ""
        return f"{model}{mode}"
    if model in NAMES:
        return tr(*NAMES[model])
    return labels.get(model, model)


def summary_frame(data: dict, labels: dict) -> pd.DataFrame:
    samples = sample_names()
    by_key = {(r["model"], r["sample"]): r for r in data["runs"]}
    rows = []
    for model, info in data["models"].items():
        runs = [r for r in data["runs"] if r["model"] == model]
        if not runs:
            continue
        cloud = info.get("location") == "cloud"
        row = {
            tr("模型", "Model"): model_name(model, info, labels),
            tr("位置", "Where"): (
                tr("雲端（Azure）", "Cloud (Azure)") if cloud else tr("本機", "Local")
            ),
        }
        for sample, title in samples.items():
            row[title] = rate_cell(by_key.get((model, sample)))
        lat = by_key.get((model, LATENCY_SAMPLE))
        if lat is None or not lat["valid"]:  # English-only models: their English run
            lat = next((r for r in runs if r["valid"]), None)
        row[tr("第一個字（秒）", "First text (s)")] = lat and lat["first_partial_s"]
        row[tr("定稿中位數（秒）", "Final, median (s)")] = lat and lat["final_median_s"]
        rtfs = [r["rtf"] for r in runs if r.get("rtf")]
        row[tr("處理速度", "Speed")] = (
            tr("雲端", "cloud")
            if cloud
            else (speed_label(statistics.median(rtfs)) if rtfs else None)
        )
        memory = info.get("memory_gb")
        row[tr("記憶體", "Memory")] = f"{memory:.1f} GB" if memory else "—"
        row[tr("備註", "Notes")] = notes(model, info, runs, samples)
        rows.append(row)
    return pd.DataFrame(rows)


def rate_cell(r: dict | None) -> str:
    """An error rate as table text: 「1.2%」, 「—」 not run, 「限流」 rate-limited (no score)."""
    if r is None or r["error_rate"] is None:
        return "—"
    if not r["valid"]:
        return tr("限流", "rate-limited")
    return f"{r['error_rate'] * 100:.1f}%"


def speed_label(rtf: float) -> str:
    if rtf >= 1:
        return tr(f"跟不上（慢 {rtf:.1f} 倍）", f"too slow ({rtf:.1f}×)")
    return tr(f"{1 / rtf:.0f} 倍快", f"{1 / rtf:.0f}× real time")


def notes(model: str, info: dict, runs: list[dict], samples: dict) -> str:
    out = []
    limited = [samples.get(r["sample"], r["sample"]) for r in runs if not r["valid"]]
    if limited:
        out.append(tr("限流，結果無效：", "rate-limited, invalid: ") + "、".join(limited))
    if info.get("mode") == "server_vad":
        out.append(tr("一句說完才出字", "text only after each utterance"))
    if info.get("delay"):
        out.append(f"delay={info['delay']}")
    if model == "voxtral-mini-4b-realtime":
        out.append(tr("這台 Mac 跟不上即時", "too slow for real time here"))
    return tr("；", "; ").join(out)


def method_name(method: str) -> str:
    if method.startswith("chunk"):
        return tr(f"切段 {method[5:]} 秒", f"{method[5:]} s chunks")
    return {"vad": tr("偵測停頓再轉", "Cut at pauses"),
            "whole": tr("整段（對照）", "Whole file (reference)")}.get(method, method)  # fmt: skip


def chunked_frame(data: dict, labels: dict) -> pd.DataFrame:
    """Whole-clip models used live: one row per model and method, error rate per recording and
    the latency on the National Day speech (none for the whole file, which is not live)."""
    samples = sample_names()
    runs = data.get("chunked_runs") or []
    by_key = {(r["model"], r["method"], r["sample"]): r for r in runs}
    rows = []
    for model in data.get("chunked_models") or []:
        for method in data.get("chunked_methods") or []:
            if not any(k[0] == model and k[1] == method for k in by_key):
                continue
            row = {
                tr("模型", "Model"): labels.get(model, model),
                tr("做法", "Method"): method_name(method),
            }
            for sample, title in samples.items():
                row[title] = rate_cell(by_key.get((model, method, sample)))
            lat = by_key.get((model, method, LATENCY_SAMPLE)) or {}
            row[tr("第一個字（秒）", "First text (s)")] = lat.get("first_partial_s")
            row[tr("定稿中位數（秒）", "Final, median (s)")] = lat.get("final_median_s")
            rows.append(row)
    return pd.DataFrame(rows)


def detail_frame(data: dict, labels: dict) -> pd.DataFrame:
    samples = sample_names()
    rows = []
    for r in data["runs"]:
        info = data["models"].get(r["model"], {})
        rows.append({
            tr("模型", "Model"): model_name(r["model"], info, labels),
            tr("錄音", "Recording"): samples.get(r["sample"], r["sample"]),
            tr("指標", "Metric"): r["metric"],
            tr("錯誤率 %", "Error %"): round(r["error_rate"] * 100, 1)
            if r["error_rate"] is not None else None,
            tr("漏掉（字／詞）", "Deleted (chars / words)"): r.get("deletions"),
            tr("參考稿長度", "Reference length"): r.get("ref_units"),
            tr("第一個字（秒）", "First text (s)"): r["first_partial_s"],
            tr("定稿中位數（秒）", "Final, median (s)"): r["final_median_s"],
            tr("定稿 p90（秒）", "Final, p90 (s)"): r["final_p90_s"],
            "RTF": r.get("rtf"),
            tr("有效", "Valid"): r["valid"],
        })  # fmt: skip
    return pd.DataFrame(rows)


def benchmark_page() -> None:
    data = load()
    if not data:
        st.info(tr("找不到評測結果（data/lab/streaming_benchmark.json）。",
                   "No benchmark results found (data/lab/streaming_benchmark.json)."),
                icon=":material/info:")  # fmt: skip
        return
    labels = sc.names("stream")
    st.caption(
        tr(
            f"同四段錄音的串流辨識結果：本機模型在 {data['hardware']} 上經模型閘道執行"
            f"（每 {data['local_frame_ms']} ms 一段，延遲依實測運算時間換算），雲端模型經 "
            f"{data['cloud']} 以說話速度每 {data['cloud_chunk_ms']} ms 送一段"
            "（延遲是實際量到的，含網路）。錯誤率：中文字錯率（CER）、英文詞錯率（WER），"
            "評分前統一繁簡與數字寫法。本頁只顯示結果，不會執行。",
            f"Streaming recognition of the same four recordings: local models run through the "
            f"model gateway on {data['hardware']} ({data['local_frame_ms']} ms frames, latency "
            f"simulated from the measured compute); cloud models through {data['cloud']}, sent "
            f"at speaking pace in {data['cloud_chunk_ms']} ms pieces (latency measured, network "
            "included). Error rate: CER for Chinese, WER for English, after unifying script and "
            "number forms. This page only shows results; it runs nothing.",
        )
    )
    frame = summary_frame(data, labels)
    rate_help = tr(
        "中文 CER、英文 WER；—＝沒有測（不支援該語言）。",
        "CER for Chinese, WER for English; — = not run (language not supported).",
    )
    titles = list(sample_names().values())
    st.dataframe(
        frame,
        hide_index=True,
        use_container_width=True,
        column_config={
            **{t: st.column_config.TextColumn(help=rate_help, width="small") for t in titles},
            tr("第一個字（秒）", "First text (s)"): st.column_config.NumberColumn(
                format="%.2f",
                width="small",
                help=tr(
                    "國慶演說：開始說話到第一個字出現。",
                    "National Day speech: speech start to the first text.",
                ),
            ),
            tr("定稿中位數（秒）", "Final, median (s)"): st.column_config.NumberColumn(
                format="%.2f",
                width="small",
                help=tr(
                    "國慶演說：一句說完到這句文字定稿。",
                    "National Day speech: end of an utterance to its final text.",
                ),
            ),
            tr("模型", "Model"): st.column_config.TextColumn(width="medium"),
            tr("位置", "Where"): st.column_config.TextColumn(width="small"),
            tr("處理速度", "Speed"): st.column_config.TextColumn(width="small"),
            tr("記憶體", "Memory"): st.column_config.TextColumn(width="small"),
        },
    )  # fmt: skip
    st.markdown(
        tr(
            "- **雲端 `gpt-live-transcribe` 最準也最快**：國慶演說 1.2%、一句說完約 0.3 秒定稿；"
            "代價是錄音要上傳到 Azure。\n"
            "- **本機首選 Voxtral 4B Realtime（4-bit）**：3.7%、0.7 秒定稿，不連網；"
            "sherpa-onnx 最輕（CPU、記憶體 0.3 GB）。\n"
            "- **真實會議（遠距、多人）**：雲端串流模型漏掉大半內容（展開下方明細看「漏掉」），"
            "放大音量也一樣；本機 Paraformer 在這段最好。\n"
            "- **`gpt-4o-transcribe`、`gpt-transcribe` 不是真串流**：靠伺服器偵測停頓切句，"
            "一句說完才出字。`gpt-transcribe` 的配額撐不住每句一次呼叫，被限流的錄音不列分數。",
            "- **Cloud `gpt-live-transcribe` is the most accurate and the fastest**: 1.2% on the "
            "National Day speech, final text about 0.3 s after each utterance; the audio is "
            "uploaded to Azure.\n"
            "- **Best local: Voxtral 4B Realtime (4-bit)**: 3.7%, final in 0.7 s, offline; "
            "sherpa-onnx is the lightest (CPU, 0.3 GB).\n"
            "- **Real meeting (far-field, several speakers)**: the cloud streaming models drop "
            "most of the speech (see Deleted in the details), louder audio does not help; "
            "the local Paraformer does best here.\n"
            "- **`gpt-4o-transcribe` and `gpt-transcribe` do not really stream**: the server "
            "cuts utterances at pauses and text comes after each one. `gpt-transcribe`'s quota "
            "cannot take one call per utterance; rate-limited recordings get no score.",
        )
    )
    if data.get("chunked_runs"):
        title = tr("一般模型用在即時字幕：切段與偵測停頓再轉",
                   "Whole-clip models live: chunks and cutting at pauses")  # fmt: skip
        st.markdown(f"**{title}**")
        st.caption(
            tr(
                "整段辨識模型（Qwen3-ASR、Breeze、Whisper…）不能邊聽邊出字，要做即時字幕只能"
                "把聲音切成小段送：固定每 N 秒切一段，或偵測到停頓（0.6 秒）再把一整句送去轉。"
                "時間的算法和上表的本機串流相同（依實測運算時間換算）。",
                "Whole-clip models (Qwen3-ASR, Breeze, Whisper…) cannot write while they "
                "listen; for live captions the audio goes in pieces: every N seconds, or a "
                "whole sentence once a 0.6 s pause is heard. Timing is simulated as for the "
                "local streaming models above.",
            )
        )
        st.dataframe(
            chunked_frame(data, labels),
            hide_index=True,
            use_container_width=True,
            column_config={
                **{t: st.column_config.TextColumn(help=rate_help, width="small") for t in titles},
                tr("第一個字（秒）", "First text (s)"): st.column_config.NumberColumn(
                    format="%.2f", width="small"
                ),
                tr("定稿中位數（秒）", "Final, median (s)"): st.column_config.NumberColumn(
                    format="%.2f", width="small"
                ),
            },
        )
        st.markdown(CHUNK_NOTES())
    with st.expander(tr("每段錄音的明細", "Every run"), icon=":material/table_rows:"):
        st.dataframe(detail_frame(data, labels), hide_index=True, use_container_width=True)
        local_at, cloud_at = str(data.get("local_at", ""))[:10], str(data.get("cloud_at", ""))[:10]
        st.caption(
            tr(
                f"本機結果 {local_at}、雲端結果 {cloud_at}；"
                "完整報告見〈本機語音模型評測〉「二、8. 串流」。",
                f"Local results {local_at}, cloud results {cloud_at}; the full report is "
                "section 2.8 of the speech evaluation.",
            )
        )
