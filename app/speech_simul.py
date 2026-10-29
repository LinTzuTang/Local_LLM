"""Simultaneous mode of the speech-model page: interpreting while the speaker talks.

Try it: a recording goes to the chosen systems at the speaker's pace (local_llm/simul.py): the
local cascade (cut at pauses -> a whole-clip ASR through the gateway -> a translator in Ollama,
timed span by span) and, with the Cloud switch on, Gemini Live Translate (real time, it speaks
the English too). Each is scored on how far the English trails the speaker and, with a
reference, chrF / BLEU. Unlike Translate (a recording translated whole, or cut into fixed
chunks), the audio arrives as a speaker would say it.

Benchmark: data/lab/simul_benchmark.json, exported by scripts/speech_eval/run_simul.py, and
the models not run yet with the reason (a key to apply for, an NVIDIA GPU).
"""

import io
import json
import tempfile
import wave
from pathlib import Path

import altair as alt
import jobs
import pandas as pd
import speech_common as sc
import speech_translate
import streamlit as st
import ui
from common import PROJECT_DIR, cloud_enabled, cloud_hint, kept, ollama_url
from i18n import tr

from local_llm import asr_metrics, audio_aug, gateway_client, simul

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
        color=alt.Color("where:N", title=None, scale=ui.where_scale()),
        tooltip=["system", "first_s", "lag_avg_s", "tail_s", "chrf", "bleu"],
    )  # fmt: skip
    points = base.mark_circle(size=110)
    labels = base.mark_text(align="left", dx=8, fontSize=12).encode(text="system")
    return (points + labels).properties(height=320)


def bench_view() -> None:
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
    shown = df.rename(columns={
            "system": tr("做法", "System"), "where": tr("位置", "Where"),
            "out": tr("輸出", "Output"), "first_s": tr("第一句英文（秒）", "First English (s)"),
            "lag_avg_s": tr("平均落後（秒）", "Avg lag (s)"),
            "lag_p90_s": tr("落後 P90（秒）", "Lag P90 (s)"),
            "tail_s": tr("講完再等（秒）", "After the end (s)"), "chrf": "chrF", "bleu": "BLEU",
            "note": tr("說明", "Notes"),
        })  # fmt: skip
    st.dataframe(ui.where_rows(shown, tr("位置", "Where")), hide_index=True, width="stretch")
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


# -- try it ---------------------------------------------------------------------------------------

CLIPS = [92, 60, 30, 0]  # seconds; 0 = the whole recording
SPANS = [2.0, 3.0, 4.0, 6.0]
TRANSLATORS = ("translategemma:12b", "translategemma:4b")
ND_SAMPLE, ND_PARAGRAPHS, ND_END = "national_day_2026_zh.wav", 4, 92.25


def reference_for(sample: str | None, clip: int) -> str:
    """The bundled English of a sample: the whole text, or the National Day speech's first four
    paragraphs for its first 92 s (the clip of the evaluation); else nothing."""
    if not sample:
        return ""
    text = sc.ref_file(sample, ".en.txt")
    if sample == ND_SAMPLE and clip == 92:
        return "\n".join(asr_metrics.reference_text(text).splitlines()[:ND_PARAGRAPHS])
    return text if not clip else ""


def simul_work(job: jobs.Job, url: str, base: str) -> None:
    state = job.state
    x = audio_aug.load(state["path"])
    if state["clip"]:
        end = ND_END if state["clip"] == 92 and state["sample"] == ND_SAMPLE else state["clip"]
        x = x[: int(end * simul.RATE)]
    state["audio_s"] = round(len(x) / simul.RATE, 1)
    for r in state["rows"]:
        if job.cancel.is_set():
            break
        r["status"] = "running"
        try:
            if r["system"] == "cascade":
                asr, mt = r["asr"], r["mt"]

                def transcribe(path, asr=asr):
                    return gateway_client.transcribe(path, asr, "zh", base_url=url)["text"]

                def translate(text, mt=mt):
                    return speech_translate._llm(text, mt, base)[0]

                job.detail = tr("載入模型…", "Loading models…")
                with tempfile.NamedTemporaryFile(suffix=".wav") as f:
                    simul._wav(x[: simul.RATE * 2], Path(f.name))
                    transcribe(f.name)  # warm both up: loading is not part of the timing
                translate("各位國人同胞")

                def progress(i, n, r=r):
                    job.detail = f"{r['label']}: {i + 1}/{n}"
                    job.progress = i / max(n, 1)

                rows = simul.cascade(x, transcribe, translate, r["min_span"], progress,
                                     job.cancel.is_set)  # fmt: skip
                r["spans"] = rows
                r.update(simul.score(rows, x, state["reference"]))
            else:
                job.detail = tr(f"Gemini 即時口譯中（約 {len(x) / simul.RATE + 15:.0f} 秒）",
                                f"Gemini interpreting (about {len(x) / simul.RATE + 15:.0f} s)")  # fmt: skip
                out = simul.gemini(x, cancelled=job.cancel.is_set)
                r["spans"] = out["pieces"]
                r["heard"], r["first_audio_s"] = out["heard"], out["first_audio_s"]
                r["errors"] = out["errors"]
                r["audio"] = out["audio"]
                r.update(simul.score(out["pieces"], x, state["reference"]))
        except Exception as exc:
            r["error"] = f"{type(exc).__name__}: {exc}"
        r["status"] = "done"
    job.progress = 1.0


def wav24(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(pcm)
    return buf.getvalue()


def result_view(job: jobs.Job) -> None:
    state = job.state
    rows = [r for r in state["rows"] if r.get("status") == "done"]
    if not rows:
        return
    table = pd.DataFrame([{
        tr("做法", "System"): r["label"],
        tr("位置", "Where"): tr("雲端", "Cloud") if r["system"] == "gemini" else tr("本機", "Local"),
        tr("第一句英文（秒）", "First English (s)"): r.get("first_s"),
        tr("平均落後（秒）", "Avg lag (s)"): r.get("lag_avg_s"),
        tr("落後 P90（秒）", "Lag P90 (s)"): r.get("lag_p90_s"),
        tr("講完再等（秒）", "After the end (s)"): r.get("tail_s"),
        "chrF": r.get("chrf"), "BLEU": r.get("bleu"),
    } for r in rows])  # fmt: skip
    st.markdown(tr(f"**結果**（錄音 {state.get('audio_s', 0):.0f} 秒）",
                   f"**Results** ({state.get('audio_s', 0):.0f} s of audio)"))  # fmt: skip
    st.dataframe(ui.where_rows(table, tr("位置", "Where")), hide_index=True, width="stretch")
    for r in rows:
        with st.container(border=True):
            st.markdown(f"**{r['label']}**")
            if r.get("error"):
                st.error(r["error"])
                continue
            if r.get("audio"):
                st.caption(tr("Gemini 說出的英文", "The English Gemini spoke"))
                st.audio(wav24(r["audio"]), format="audio/wav")
            st.write(r.get("text") or "")
            if r["system"] == "cascade" and r.get("spans"):
                with st.expander(tr("逐段：講者說到哪、英文何時出現", "Span by span")):
                    st.dataframe(pd.DataFrame([{
                        tr("講者（秒）", "Speaker (s)"): f"{p['start']:.1f}–{p['end']:.1f}",
                        tr("英文出現（秒）", "English at (s)"): p["t"],
                        tr("辨識", "Heard"): p["source"], tr("英文", "English"): p["text"],
                    } for p in r["spans"]]), hide_index=True, width="stretch")  # fmt: skip
            elif r.get("heard"):
                st.caption(tr("Gemini 聽到的：", "What Gemini heard: ") + r["heard"])


@st.fragment(run_every=2)
def _progress(job_id: str) -> None:
    job = jobs.get(job_id)
    if job is None:
        return
    if job.running:
        st.progress(job.progress or 0.0, text=job.detail)
    else:
        st.rerun()


def try_view(url: str, status: dict) -> None:
    labels = sc.names()
    chosen = sc.audio_choice("si", default=ND_SAMPLE)
    a, b = st.columns(2)
    st.session_state.setdefault("si_clip", 92)
    clip = a.select_slider(
        tr("只用開頭幾秒", "Use the first"), CLIPS,
        format_func=lambda s: tr(f"{s} 秒", f"{s} s") if s else tr("整段", "whole recording"),
        **kept("si_clip"),
    )  # fmt: skip
    sample = chosen[3] if chosen and chosen[2] == "sample" else None
    wanted = (sample, clip)
    if st.session_state.get("si_reference_for") != wanted:
        st.session_state.si_reference = reference_for(sample, clip)
        st.session_state.si_reference_for = wanted
    reference = st.text_area(
        tr("參考英文譯文（選填，用來算 chrF／BLEU）", "Reference English (optional, for chrF / BLEU)"),
        height=90,
        help=tr("國慶演說的前 92 秒會自動帶入官方英譯的前 4 段（和評測同一段）。",
                "The National Day speech's first 92 s fill in the first four paragraphs of the "
                "official English (the evaluation's clip)."),
        **kept("si_reference"),
    )  # fmt: skip
    installed = [m for m in labels if status.get(m, {}).get("installed")]
    asr_options = [m for m in ("qwen3-asr-1.7b", "qwen3-asr-0.6b", "breeze-asr-25",
                               "whisper-large-v3-turbo") if m in installed] or installed  # fmt: skip
    ollama = sc.ollama_models(ollama_url())
    mt_options = [m for m in TRANSLATORS if m in ollama] + [m for m in ollama
                                                            if m not in TRANSLATORS]  # fmt: skip
    systems = {"cascade": tr("地端：停頓切段 → 辨識 → 翻譯", "Local: cut at pauses → ASR → MT")}
    if cloud_enabled():
        systems["gemini"] = tr("雲端：Gemini 3.5 Live Translate（錄音會上傳）",
                               "Cloud: Gemini 3.5 Live Translate (audio is uploaded)")  # fmt: skip
    st.session_state.setdefault("si_systems", ["cascade"])
    picked = st.pills(tr("做法", "Systems"), list(systems), format_func=systems.get,
                      selection_mode="multi", **kept("si_systems")) or []  # fmt: skip
    cloud_hint("同步口譯（Gemini 3.5 Live Translate）", "interpreting (Gemini Live Translate)")
    rows = []
    if "cascade" in picked:
        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            st.session_state.setdefault("si_asr", asr_options[0] if asr_options else "")
            asr = c1.selectbox(tr("辨識模型", "Speech model"), asr_options or [""],
                               format_func=lambda m: labels.get(m, m), **kept("si_asr"))  # fmt: skip
            st.session_state.setdefault("si_mt", mt_options[0] if mt_options else "")
            mt = c2.selectbox(tr("翻譯模型（Ollama）", "Translator (Ollama)"), mt_options or [""],
                              **kept("si_mt"))  # fmt: skip
            st.session_state.setdefault("si_span", simul.MIN_SPAN_S)
            span = c3.select_slider(
                tr("累積多久才送出翻譯", "Speech before a cut"), SPANS,
                format_func=lambda s: tr(f"{s:.0f} 秒", f"{s:.0f} s"),
                help=tr("停頓 0.4 秒以上、而且已累積這麼多話才送出；越短越即時，但每段太短翻不好。",
                        "Sent at a pause of 0.4 s once this much speech has built up; shorter is "
                        "sooner but short spans translate worse."),
                **kept("si_span"),
            )  # fmt: skip
            if asr and mt:
                rows.append({"system": "cascade", "asr": asr, "mt": mt, "min_span": span,
                             "label": f"{labels.get(asr, asr)} → {mt}（{span:.0f} s）",
                             "status": "waiting"})  # fmt: skip
    if "gemini" in picked:
        rows.append({"system": "gemini", "label": "Gemini 3.5 Live Translate", "status": "waiting"})
    job = jobs.get(st.session_state.get("si_job"))
    busy = bool(job and job.running)
    if st.button(tr("開始同步口譯", "Interpret"), type="primary", icon=":material/play_arrow:",
                 disabled=busy or not rows or chosen is None):  # fmt: skip
        path = chosen[0] if not isinstance(chosen[0], bytes) else sc.cached_audio(*chosen[:2])
        job = jobs.Job("speech_simul", tr("同步口譯", "Simultaneous interpreting"), sc.PAGE,
                       state={"path": str(path), "sample": sample, "clip": clip,
                              "reference": reference.strip(), "rows": rows})  # fmt: skip
        jobs.start(job, lambda j: simul_work(j, url, ollama_url()))
        st.session_state.si_job = job.id
        st.rerun()
    if job and job.running:
        _progress(job.id)
    elif job:
        result_view(job)


def simul_page(url: str | None = None, status: dict | None = None) -> None:
    st.session_state.setdefault("si_view", "try")
    view = st.segmented_control(
        tr("檢視", "View"), ["try", "bench"], required=True,
        format_func=lambda v: {"try": f":material/play_arrow: {tr('自己試', 'Try it')}",
                               "bench": f":material/leaderboard: {tr('評測總表', 'Benchmark')}"}[v],
        **kept("si_view"),
    ) or "try"  # fmt: skip
    if view == "bench":
        bench_view()
    else:
        try_view(url, status or {})
