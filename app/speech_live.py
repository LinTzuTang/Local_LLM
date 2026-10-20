"""Live mode of the speech-model page: live captions, simulated from a recording.

The recording (a sample, an upload or a microphone take) is fed to each chosen speech model in
short chunks (1-5 s), as a live speaker's audio would arrive. Each chunk is transcribed on its
own and the measured processing time gives, through speech_translate.live_schedule, when its
text would appear: lag = text returned - chunk spoken. The page shows the captions growing
chunk by chunk, a lag-per-chunk chart, the lag at the end and the final CER / WER of the joined
captions; optionally the whole recording is also transcribed at once, to show what chunking
costs in accuracy.

This is not true streaming: these models take a whole clip at a time. The page's other method,
"true streaming" (speech_stream), runs real streaming models (sherpa-onnx Zipformer /
Paraformer, Voxtral Realtime) frame by frame; cloud realtime APIs are mentioned as text only.
"""

from pathlib import Path

import altair as alt
import jobs
import pandas as pd
import speech_bench
import speech_common as sc
import speech_mic
import speech_stream
import streamlit as st
from common import kept, save_run
from i18n import pick, tr

from local_llm import asr_metrics, audio_aug, catalog, gateway_client
from local_llm import speech_live as sl
from local_llm import speech_translate as stx

CLIPS = [30, 60, 120, 0]  # seconds; 0 = the whole recording
CHUNKS = [1.0, 1.5, 2.0, 3.0, 5.0]
MAX_MODELS = 3
DEFAULT_MODELS = ["qwen3-asr-0.6b", "sensevoice-small", "whisper-large-v3-turbo"]
SAVED = ("model", "text", "chunks", "stats", "score", "offline", "error")
METHODS = {  # key -> (zh, en, icon); tr() runs on the page, not at import
    "chunk": ("切段模擬", "Chunked simulation", "content_cut"),
    "stream": ("真串流", "True streaming", "stream"),
    "mic": ("麥克風即時", "Live microphone", "mic"),
    "results": ("評測總表", "Benchmark", "leaderboard"),
}


# -- worker (no Streamlit calls) ------------------------------------------------------------


def _compute(out: dict) -> float:
    load = (out.get("load_seconds") or 0) + (out.get("ollama_load_seconds") or 0)
    return max(out["seconds"] - load - (out.get("queue_seconds") or 0), 0.01)


def live_work(job: jobs.Job, url: str) -> None:
    state = job.state
    rows, reference, language = state["rows"], state["reference"], state["language"] or None
    source = Path(state["audio"])
    samples = audio_aug.load(source)
    if state["clip_s"]:
        samples = samples[: int(state["clip_s"] * audio_aug.SR)]
    folder = sc.CACHE_DIR / sc.cache_key(source)
    tag = f"clip{state['clip_s'] or 'all'}"
    clip = folder / f"{tag}.wav"
    if not clip.exists():
        audio_aug.save(clip, samples)
    state["duration"] = round(len(samples) / audio_aug.SR, 2)
    state["clip_file"] = str(clip)
    chunks = []
    for i, (a, b) in enumerate(stx.chunk_bounds(len(samples), state["chunk_s"], min_s=0.5)):
        f = folder / f"{tag}_live{state['chunk_s']}s_{i:03d}.wav"
        if not f.exists():
            audio_aug.save(f, samples[a:b])
        chunks.append({"file": f, "start": a / audio_aug.SR, "end": b / audio_aug.SR})
    state["n_chunks"] = len(chunks)

    def transcribe(path: Path, model: str) -> dict:
        return gateway_client.transcribe(path, model, language, base_url=url)

    for n, r in enumerate(rows, start=1):
        if job.cancel.is_set():
            break
        r["status"], r["chunks"] = "running", []
        job.detail = f"{n}/{len(rows)} {r['model']} · warm-up"
        try:
            transcribe(chunks[0]["file"], r["model"])  # load the model; not timed
            texts = []
            for i, c in enumerate(chunks):
                if job.cancel.is_set():
                    break
                job.detail = f"{n}/{len(rows)} {r['model']} · {i + 1}/{len(chunks)}"
                job.progress = ((n - 1) + i / len(chunks)) / len(rows)
                out = transcribe(c["file"], r["model"])
                texts.append(out["text"].strip())
                r["chunks"].append(
                    {"start": round(c["start"], 2), "end": round(c["end"], 2),
                     "text": texts[-1], "compute": round(_compute(out), 3)}
                )  # fmt: skip
                schedule = stx.live_schedule(
                    [x["end"] for x in r["chunks"]], [x["compute"] for x in r["chunks"]]
                )
                for x, s in zip(r["chunks"], schedule, strict=True):
                    x.update(finish=s["finish"], lag=s["lag"])
                r["text"] = sl.join_texts(texts)
            r["stats"] = sl.latency_stats(r["chunks"])
            if reference and r["text"]:
                r["score"] = asr_metrics.score(reference, r["text"], language)
            if state["offline"] and not job.cancel.is_set():
                job.detail = f"{n}/{len(rows)} {r['model']} · whole recording"
                out = transcribe(clip, r["model"])
                r["offline"] = {
                    "text": out["text"],
                    "compute": round(_compute(out), 2),
                    "score": asr_metrics.score(reference, out["text"], language)
                    if reference
                    else None,
                }
        except Exception as exc:
            r["error"] = str(exc)
        r["status"] = "done"
    for r in rows:
        r["status"] = "done"
    job.progress = 1.0
    path = save_run(
        "asr_compare",
        {
            "mode": "live",
            "audio": state["audio_name"],
            "source": state["source"],
            "clip_s": state["clip_s"] or None,
            "duration": state["duration"],
            "chunk_s": state["chunk_s"],
            "language": state["language"] or None,
            "reference": reference or None,
            "stopped": job.cancel.is_set(),
            "results": [{k: r.get(k) for k in SAVED} for r in rows],
        },
    )
    state["path"] = path.relative_to(path.parents[1])


# -- drawing --------------------------------------------------------------------------------


def facts(r: dict) -> str:
    s, parts = r.get("stats") or {}, []
    if not s:
        return ""
    parts.append(tr(f"第一段文字 {s['first_s']:.1f}s", f"first text at {s['first_s']:.1f}s"))
    parts.append(tr(f"平均落後 {s['mean_lag']:.2f}s", f"mean lag {s['mean_lag']:.2f}s"))
    parts.append(tr(f"結束時落後 {s['lag_end']:.2f}s", f"{s['lag_end']:.2f}s behind at end"))
    if s.get("rtf"):
        parts.append(f"RTF {s['rtf']:.3f}")
    if r.get("score"):
        parts.append(f"{r['score'].get('metric', 'CER')} {sc.pct(r['score']['cer'])}")
    return " · ".join(parts)


def card(r: dict, names: dict, caption: str | None = None) -> None:
    with st.container(border=True):
        st.markdown(f"**{names.get(r['model'], r['model'])}**")
        st.caption(f"`{r['model']}`")
        if r["status"] == "waiting":
            st.info(tr("等待中…", "Waiting…"))
            return
        if r.get("error"):
            st.error(r["error"])
            return
        if r["status"] == "running" and not r.get("chunks"):
            st.info(tr("載入模型中…", "Loading the model…"), icon=":material/progress_activity:")
            return
        if r.get("stats"):
            st.caption(facts(r))
        text = (r.get("text") or "") if caption is None else caption
        with st.container(height=200, border=False):
            st.write(sc.show(text) + (" ▌" if r["status"] == "running" else ""))
        off = r.get("offline")
        if off:
            score = off.get("score")
            with st.expander(
                tr("整段一次辨識（對照）", "Whole recording at once (for comparison)")
                + (f" · {score.get('metric', 'CER')} {sc.pct(score['cer'])}" if score else "")
            ):
                st.caption(tr(f"處理 {off['compute']:.2f}s", f"processing {off['compute']:.2f}s"))
                st.write(sc.show(off["text"]))
        if r.get("chunks"):
            with st.expander(tr("逐段結果", "Chunk by chunk")):
                st.dataframe(chunk_table(r), hide_index=True, width="stretch")


def chunk_table(r: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                tr("音訊", "Audio"): f"{c['start']:.1f}–{c['end']:.1f}s",
                tr("處理（秒）", "Compute (s)"): c["compute"],
                tr("落後（秒）", "Lag (s)"): c.get("lag"),
                tr("文字", "Text"): sc.show(c["text"]),
            }
            for c in r.get("chunks") or []
        ]
    )


def summary(rows: list[dict], names: dict, reference: str) -> None:
    done = [r for r in rows if r.get("stats") and not r.get("error")]
    if not done:
        return
    st.subheader(tr("比較結果", "Results"))
    model = tr("模型", "Model")
    metric = next((r["score"].get("metric") for r in done if r.get("score")), "CER")
    live_col = tr(f"逐段 {metric}", f"Live {metric}")
    whole_col = tr(f"整段 {metric}", f"Whole-file {metric}")
    table = pd.DataFrame(
        [
            {
                model: names.get(r["model"], r["model"]),
                tr("第一段文字（秒）", "First text (s)"): r["stats"]["first_s"],
                tr("平均落後（秒）", "Mean lag (s)"): r["stats"]["mean_lag"],
                tr("最大落後（秒）", "Max lag (s)"): r["stats"]["max_lag"],
                tr("結束時落後（秒）", "Lag at end (s)"): r["stats"]["lag_end"],
                "RTF": r["stats"].get("rtf"),
                live_col: (r.get("score") or {}).get("cer"),
                whole_col: ((r.get("offline") or {}).get("score") or {}).get("cer"),
            }
            for r in done
        ]
    )
    if not reference:
        table = table.drop(columns=[live_col, whole_col])
    elif table[whole_col].isna().all():
        table = table.drop(columns=[whole_col])
    pct = st.column_config.NumberColumn(format="percent")
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            live_col: pct,
            whole_col: pct,
            "RTF": st.column_config.NumberColumn(
                help=tr(
                    "處理時間 ÷ 音訊長度；小於 1 才跟得上即時",
                    "Processing time ÷ audio length; below 1 keeps up with real time",
                ),
                format="%.3f",
            ),  # fmt: skip
        },
    )
    long = pd.DataFrame(
        [
            {
                "model": names.get(r["model"], r["model"]),
                "t": c["end"],
                "lag": c["lag"],
                "compute": c["compute"],
            }
            for r in done
            for c in r.get("chunks") or []
        ]  # fmt: skip
    )
    if long.empty:
        return
    st.caption(
        tr(
            "每段講完後，文字晚多久出現（秒）＝等前一段＋這段的處理時間。線一直往上表示跟不上即時。",
            "How long after each chunk was spoken its text appears (s): waiting for the previous "
            "chunk plus this chunk's processing. A line that keeps rising means the model cannot "
            "keep up with real time.",
        )
    )
    chart = (
        alt.Chart(long)
        .mark_line(point=True)
        .encode(
            x=alt.X("t:Q", title=tr("講者說到第幾秒", "Speaker time (s)")),
            y=alt.Y("lag:Q", title=tr("落後（秒）", "Lag (s)")),
            color=alt.Color("model:N", title=model),
            tooltip=[
                alt.Tooltip("model:N", title=model),
                alt.Tooltip("t:Q", title=tr("秒", "s")),
                alt.Tooltip("lag:Q", title=tr("落後", "Lag"), format=".2f"),
                alt.Tooltip("compute:Q", title=tr("處理", "Compute"), format=".2f"),
            ],
        )
        .properties(height=260)
    )
    st.altair_chart(chart, width="stretch")


def notes() -> None:
    st.caption(
        tr(
            "**這是模擬**：錄音切成固定長度的片段，假設講者每講完一段才送出，每段等前一段處理完才開始；"
            "落後＝這段文字出現的時間－這段講完的時間（不含載入模型，模型先用第一段暖機）。"
            "每段各自辨識、不帶前文，切點可能切斷字詞，所以逐段的錯誤率通常比整段一次辨識高。",
            "**This is a simulation**: the recording is cut into fixed-length chunks as if each "
            "were sent when the speaker finished it, and each chunk waits for the previous one. "
            "Lag = when the chunk's text appears minus when it was spoken (model loading "
            "excluded; each model is warmed up on the first chunk). Chunks are transcribed on "
            "their own without earlier context and cuts can split words, so the live error rate "
            "is usually higher than transcribing the whole recording at once.",
        )
    )


def replay(job_id: str, rows: list[dict], duration: float, names: dict) -> dict:
    """A slider over the talk: each model's caption as a viewer would have seen it then."""
    end = max(
        [duration] + [c["finish"] for r in rows for c in r.get("chunks") or []], default=duration
    )
    t = st.slider(
        tr("重播：講者說到第幾秒時，畫面上的字幕", "Replay: the captions on screen at second"),
        0.0, float(round(end + 0.5, 1)), float(round(end + 0.5, 1)), 0.5,
        key=f"lv_replay_{job_id}",
        help=tr("拖動看每個模型的字幕在當時出現到哪裡。",
                "Drag to see how far each model's captions had got at that moment."),
    )  # fmt: skip
    return {r["model"]: sl.caption_at(r.get("chunks") or [], t) for r in rows}


def cards(rows: list[dict], names: dict, captions: dict | None = None) -> None:
    cols = st.columns(max(len(rows), 1), gap="small")
    for col, r in zip(cols, rows, strict=False):
        with col:
            card(r, names, (captions or {}).get(r["model"]))


def render(job: jobs.Job) -> None:
    state, names = job.state, sc.names()
    rows = state["rows"]
    title = state["audio_name"]
    if state["clip_s"]:
        title += tr(f"（前 {state['clip_s']} 秒）", f" (first {state['clip_s']} s)")
    st.markdown(f"#### {title}")
    st.info(
        tr(
            f"模擬即時字幕：每 {state['chunk_s']:g} 秒送出一段"
            + (f"，共 {state['n_chunks']} 段" if state.get("n_chunks") else ""),
            f"Simulated live captions: a chunk every {state['chunk_s']:g} s"
            + (f", {state['n_chunks']} chunks" if state.get("n_chunks") else ""),
        ),
        icon=":material/sensors:",
    )
    if job.running:
        cards(rows, names)
        sc.running_bar(job)
        return
    if state.get("clip_file") and Path(state["clip_file"]).exists():
        st.audio(state["clip_file"])
    captions = replay(job.id, rows, state.get("duration") or 0, names)
    cards(rows, names, captions)
    if job.error:
        st.error(job.error)
    summary(rows, names, state["reference"])
    notes()
    sc.saved_caption(state)


@st.fragment(run_every=1)
def live_view(job: jobs.Job) -> None:
    if not job.running:
        st.rerun()
    render(job)


def streaming_info() -> None:
    """Where true streaming is, the streaming models not in the gateway, and cloud options."""
    library = [m for m in catalog.other_models("stream") if not m.get("gateway")]
    names = [pick({"zh": m["name"], "en": m.get("name_en") or m["name"]}) for m in library]
    st.info(
        tr(
            "**切段模擬**把整段辨識模型當成串流用。要看真正邊收音邊出字的模型（sherpa-onnx "
            "Zipformer／Paraformer、Voxtral Realtime），選上面的「真串流」。",
            "**Chunked simulation** uses whole-clip models as if they streamed. For models that "
            "really decode while audio arrives (sherpa-onnx Zipformer / Paraformer, Voxtral "
            "Realtime), choose **True streaming** above.",
        )
        + (
            tr(
                "\n\n模型庫另列出這些串流模型（尚未接進平台）：" + "、".join(names) + "。",
                "\n\nThe model library also lists these streaming models (not in the platform "
                "yet): " + ", ".join(names) + ".",
            )
            if names
            else ""
        )
        + tr(
            "\n\n雲端選項（只列出，本頁不連線）：Azure OpenAI `gpt-realtime-whisper`、"
            "`gpt-live-transcribe`（Realtime API，WebSocket 串流轉錄）。",
            "\n\nCloud options (listed only; this page does not connect to them): Azure OpenAI "
            "`gpt-realtime-whisper` and `gpt-live-transcribe` (Realtime API, streaming "
            "transcription over WebSocket).",
        ),
        icon=":material/stream:",
    )
    st.page_link(
        "views/models.py",
        label=tr("到模型庫看串流模型", "See streaming models in the model library"),
        icon=":material/library_books:",
    )


# -- page -----------------------------------------------------------------------------------


def live_page(url: str, status: dict) -> None:
    st.session_state.setdefault("lv_method", "chunk")
    method = (
        st.segmented_control(
            tr("做法", "Method"),
            list(METHODS),
            format_func=lambda m: f":material/{METHODS[m][2]}: {tr(*METHODS[m][:2])}",
            help=tr(
                "切段模擬：整段辨識模型每次收 1–5 秒。真串流：串流模型每 0.1–0.3 秒收一小段，"
                "邊收邊出字。麥克風即時：對著麥克風講，串流模型即時出字幕。"
                "評測總表：本機與雲端串流模型的實測結果（不會執行）。",
                "Chunked simulation: whole-clip models get 1–5 s at a time. True streaming: "
                "streaming models get 0.1–0.3 s at a time and write text as audio arrives. "
                "Live microphone: speak and a streaming model captions you live. Benchmark: the "
                "measured results of local and cloud streaming models (nothing runs).",
            ),
            **kept("lv_method"),
        )
        or "chunk"
    )
    if method == "mic":
        speech_mic.mic_page(url, status)
        return
    if method == "results":
        speech_bench.benchmark_page()
        return
    chosen = sc.audio_choice("lv", default="meeting_zh.wav")
    if st.session_state.get("lv_source") == "record":
        st.caption(
            tr(
                "錄完後會把這段錄音當成串流重播（依序送出）；不是邊講邊送。",
                "After you stop recording, the take is replayed as a stream (sent piece by "
                "piece); it is not sent while you speak.",
            )
        )
    if method == "stream":
        speech_stream.stream_page(url, status, chosen)
        return
    labels = sc.names()
    installed = [m for m in labels if status.get(m, {}).get("installed")]
    a, b, c = st.columns(3)
    st.session_state.setdefault("lv_chunk", 2.0)
    chunk = a.select_slider(
        tr("每段長度", "Chunk length"), CHUNKS, format_func=lambda s: f"{s:g} s",
        **kept("lv_chunk"),
    )  # fmt: skip
    st.session_state.setdefault("lv_clip", 0)
    clip = b.select_slider(
        tr("只用開頭幾秒", "Use the first"),
        CLIPS,
        format_func=lambda s: tr(f"{s} 秒", f"{s} s") if s else tr("整段", "whole recording"),
        **kept("lv_clip"),
    )
    languages = {"zh": tr("中文", "Chinese"), "en": tr("英文", "English"),
                 "": tr("自動偵測", "Detect")}  # fmt: skip
    picked = (chosen[2], chosen[3]) if chosen else None
    if chosen and chosen[2] == "sample" and st.session_state.get("lv_language_for") != picked:
        st.session_state.lv_language = sc.sample_language(chosen[3])
    st.session_state.lv_language_for = picked
    st.session_state.setdefault("lv_language", "zh")
    language = c.selectbox(
        tr("語言", "Language"), list(languages), format_func=languages.get, **kept("lv_language")
    )

    options = [m for m in labels if m in status and sc.backend(m) != "ollama-audio"]
    st.session_state.setdefault(
        "lv_models", [m for m in DEFAULT_MODELS if m in installed] or installed[:2]
    )
    st.session_state.lv_models = [m for m in st.session_state.lv_models if m in options]
    selected = st.multiselect(
        tr(f"要比較的語音模型（最多 {MAX_MODELS} 個）", f"Speech models (up to {MAX_MODELS})"),
        options,
        format_func=lambda m: sc.model_label(m, status, labels),
        max_selections=MAX_MODELS,
        help=tr(
            "音訊 LLM（Gemma 4）每段要好幾秒，不適合即時字幕，所以不列出。",
            "Audio LLMs (Gemma 4) take seconds per chunk, too slow for live captions, so they "
            "are not listed.",
        ),
        **kept("lv_models"),
    )
    sc.download_missing(selected, status, "lv")

    # The bundled reference covers the whole sample, so only then is it filled in.
    sample = chosen[3] if chosen and chosen[2] == "sample" else None
    wanted = (chosen[2] if chosen else None, sample, clip)
    if st.session_state.get("lv_reference_for") != wanted:
        st.session_state.lv_reference = sc.sample_reference(sample) if sample and not clip else ""
        st.session_state.lv_reference_for = wanted
    reference = st.text_area(
        tr("參考原稿（選填，用來計算最後的錯誤率）",
           "Reference transcript (optional, for the final error rate)"),
        height=90,
        help=tr("要和實際使用的音訊範圍一致；範例錄音用整段時自動帶入。",
                "Must cover the audio actually used; filled in for a whole sample."),
        **kept("lv_reference"),
    )  # fmt: skip
    st.session_state.setdefault("lv_offline", True)
    offline = st.toggle(
        tr("同時整段一次辨識，對照切段的準確度損失",
           "Also transcribe the whole recording at once, to see what chunking costs"),
        **kept("lv_offline"),
    )  # fmt: skip
    streaming_info()

    job = jobs.get(st.session_state.get("lv_job"))
    busy = bool(job and job.running)
    ready = [m for m in selected if status[m]["installed"]]
    if st.button(
        tr(f"開始模擬即時字幕（{len(ready)} 個模型）",
           f"Start live captions ({len(ready)} models)"),
        type="primary", icon=":material/play_arrow:",
        disabled=busy or not ready or chosen is None,
    ):  # fmt: skip
        audio, filename, source, audio_name = chosen
        job = jobs.Job(
            "speech_live",
            tr(f"即時字幕模擬：{len(ready)} 個模型", f"Live captions: {len(ready)} models"),
            sc.PAGE,
            state={
                "audio": str(sc.cached_audio(audio, filename)),
                "audio_name": audio_name,
                "source": source,
                "clip_s": clip,
                "chunk_s": float(chunk),
                "language": language,
                "reference": reference.strip(),
                "offline": offline,
                "rows": [{"model": m, "status": "waiting", "text": ""} for m in ready],
            },
        )
        jobs.start(job, lambda j: live_work(j, url))
        st.session_state.lv_job = job.id
        st.rerun()
    if chosen is None:
        st.caption(tr("先選擇或錄一段音訊。", "Choose or record some audio first."))
    if job and job.running:
        live_view(job)
    elif job:
        render(job)


# -- history --------------------------------------------------------------------------------


def history_view(data: dict) -> None:
    if data.get("method") == "stream":
        speech_stream.history_view(data)
        return
    names = sc.names()
    rows = [r | {"status": "done"} for r in data.get("results") or [] if isinstance(r, dict)]
    bits = [tr("錄音：", "Audio: ") + f"`{data.get('audio', '?')}`"]
    if data.get("clip_s"):
        bits.append(tr(f"前 {data['clip_s']} 秒", f"first {data['clip_s']} s"))
    elif data.get("duration"):
        bits.append(tr(f"{data['duration']:.0f} 秒", f"{data['duration']:.0f} s"))
    bits.append(tr(f"模擬即時字幕，每段 {data.get('chunk_s'):g} 秒",
                   f"simulated live captions, {data.get('chunk_s'):g} s chunks"))  # fmt: skip
    if data.get("stopped"):
        bits.append(tr("中途停止", "stopped early"))
    st.caption(" · ".join(bits))
    summary(rows, names, data.get("reference") or "")
    cards(rows, names)
    notes()
    if data.get("reference"):
        with st.expander(tr("參考原稿", "Reference transcript")):
            st.write(data["reference"])


def history_summary(data: dict) -> dict:
    # Cached across languages (history.summarize): ids only, no tr().
    if data.get("method") == "stream":
        return speech_stream.history_summary(data)
    rows = [r for r in data.get("results") or [] if isinstance(r, dict)]
    scored = [(r["model"], r["score"]["cer"]) for r in rows if r.get("score")]
    metric = next((r["score"].get("metric") for r in rows if r.get("score")), None) or "CER"
    return {
        "title": f"{data.get('audio') or '?'} · live {data.get('chunk_s') or '?'}s",
        "models": [r["model"] for r in rows if r.get("model")],
        "seconds": round(sum((r.get("stats") or {}).get("compute_s") or 0 for r in rows), 2)
        or None,
        "snippet": next((r.get("text", "")[:220] for r in rows if r.get("text")), ""),
        "best": (min(scored, key=lambda x: x[1]), metric) if scored else None,
        "error": bool(rows) and all(r.get("error") for r in rows),
    }
