"""Live mode, true streaming: models that take audio a frame at a time (sherpa-onnx Zipformer
and Paraformer on the CPU, Voxtral Realtime on the GPU), served by the gateway's
/v1/audio/stream.

The recording is fed to each model in 100-320 ms frames, as a microphone would. By default the
gateway feeds as fast as the model goes and reports *simulated* time (when each text would have
appeared live, from the measured compute per frame); "real-time pace" makes it wait for each
frame instead. The page shows the text growing, with the not yet committed (partial) part in
grey, then per model: first partial latency, per-utterance finalisation latency (median / p90),
real-time factor, how much partial text was revised, and the final CER / WER
(local_llm.asr_streaming explains each measure).
"""

from pathlib import Path

import altair as alt
import jobs
import pandas as pd
import speech_common as sc
import streamlit as st
from common import PROJECT_DIR, kept, save_run
from i18n import pick, tr

from local_llm import asr_metrics, asr_streaming, audio_aug, catalog, gateway_client

FRAMES = [100, 160, 200, 320]  # ms
CLIPS = [30, 60, 120, 0]
MAX_MODELS = 4
DEFAULT_MODELS = ["zipformer-zh-en-stream", "paraformer-zh-en-stream"]
APPEND_ONLY_BACKENDS = {"mlx-audio-realtime"}
SAVED = ("model", "text", "events", "stats", "score", "error", "append_only", "load_seconds")
# Earlier chunked simulation on meeting_zh (2 s chunks): median lag and live CER, for context.
CHUNKED_BASELINE = [
    ("qwen3-asr-0.6b", 0.12, 0.113),
    ("sensevoice-small", 0.06, 0.126),
    ("whisper-large-v3-turbo", 0.21, 0.204),
]


def append_only(model: str) -> bool:
    """Models that never rewrite text they emitted (Voxtral): pieces are joined as they are."""
    return sc.backend(model) in APPEND_ONLY_BACKENDS


def spec(model: str) -> dict:
    return next((m for m in catalog.gateway_models("stream") if m["id"] == model), {})


def languages(model: str) -> list[str]:
    return spec(model).get("options", {}).get("languages", [])


def realtime(model: str) -> bool:
    """False for models measured slower than real time on this Mac (catalog
    `options.realtime`, e.g. full-precision Voxtral): fine for files, useless live."""
    return spec(model).get("options", {}).get("realtime", True)


def label(model: str, status: dict, labels: dict) -> str:
    text = sc.model_label(model, status, labels)
    return text if realtime(model) else text + tr("（跟不上即時）", " (slower than real time)")


def realtime_warning(models: list[str], labels: dict) -> None:
    for m in models:
        if not realtime(m):
            st.warning(
                f"**{labels.get(m, m)}**"
                + tr("：", ": ")
                + pick(
                    {
                        "zh": spec(m).get("realtime_note", ""),
                        "en": spec(m).get("realtime_note_en", ""),
                    }
                )  # fmt: skip
            )


# -- worker (no Streamlit calls) ------------------------------------------------------------


def stream_work(job: jobs.Job, url: str) -> None:
    state = job.state
    rows, reference, language = state["rows"], state["reference"], state["language"] or None
    source = Path(state["audio"])
    samples = audio_aug.load(source)
    if state["clip_s"]:
        samples = samples[: int(state["clip_s"] * audio_aug.SR)]
    folder = sc.CACHE_DIR / sc.cache_key(source)
    clip = folder / f"clip{state['clip_s'] or 'all'}.wav"
    if not clip.exists():
        audio_aug.save(clip, samples)
    warm = folder / "warmup_1s.wav"
    if not warm.exists():
        audio_aug.save(warm, samples[: audio_aug.SR])
    state["duration"] = round(len(samples) / audio_aug.SR, 2)
    state["clip_file"] = str(clip)
    state["segments"] = asr_streaming.speech_segments(samples)

    for n, r in enumerate(rows, start=1):
        if job.cancel.is_set():
            break
        r["status"], r["events"], r["append_only"] = "running", [], append_only(r["model"])
        job.detail = f"{n}/{len(rows)} {r['model']} · warm-up"
        try:
            # Loads the model and warms up the runtime; not timed.
            warmup = gateway_client.stream_transcribe(warm, r["model"], base_url=url)
            r["load_seconds"] = warmup.get("load_seconds")
            stream = gateway_client.stream(
                clip, r["model"], frame_ms=state["frame_ms"], pace=state["pace"], base_url=url
            )
            for item in stream:
                if job.cancel.is_set():
                    stream.close()  # closing the connection stops the decode
                    break
                if item.get("done"):
                    r["done"] = item
                    continue
                r["events"].append(item)
                r["now"] = item["t_audio"]
                job.detail = (
                    f"{n}/{len(rows)} {r['model']} · {item['t_audio']:.0f}/{state['duration']:.0f}s"
                )
                job.progress = ((n - 1) + item["t_audio"] / max(state["duration"], 0.1)) / len(rows)
            done = r.get("done") or {}
            r["text"] = asr_streaming.transcript(r["events"], r["append_only"])
            r["stats"] = asr_streaming.summary(
                r["events"],
                state["segments"],
                state["duration"],
                done.get("compute_s") or 0.0,
                r["append_only"],
            )
            if reference and r["text"]:
                r["score"] = asr_metrics.score(reference, r["text"], language)
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
            "method": "stream",
            "audio": state["audio_name"],
            "source": state["source"],
            "clip_s": state["clip_s"] or None,
            "duration": state["duration"],
            "frame_ms": state["frame_ms"],
            "pace": state["pace"],
            "language": state["language"] or None,
            "reference": reference or None,
            "segments": state["segments"],
            "stopped": job.cancel.is_set(),
            "results": [{k: r.get(k) for k in SAVED} for r in rows],
        },
    )
    state["path"] = path.relative_to(path.parents[1])


# -- drawing --------------------------------------------------------------------------------


def screen_text(r: dict, t: float | None = None) -> tuple[str, str]:
    """(committed, partial) text on screen, at the end or at second `t` of the replay."""
    events = r.get("events") or []
    if t is not None:
        events = [e for e in events if e.get("t_wall", 0) <= t]
    joined = asr_streaming.transcript(events, bool(r.get("append_only")))
    last = events[-1] if events else None
    if last and not last.get("final") and last.get("text"):
        committed = asr_streaming.transcript(
            [e for e in events if e.get("final")], bool(r.get("append_only"))
        )
        return committed, joined[len(committed) :].strip()
    return joined, ""


def facts(r: dict) -> str:
    s, parts = r.get("stats") or {}, []
    if not s:
        return ""
    if s.get("first_partial_s") is not None:
        parts.append(
            tr(f"首字 {s['first_partial_s']:.2f}s", f"first text {s['first_partial_s']:.2f}s")
        )
    if s.get("final_median_s") is not None:
        parts.append(
            tr(
                f"定稿 中位數 {s['final_median_s']:.2f}s／p90 {s['final_p90_s']:.2f}s",
                f"final after {s['final_median_s']:.2f}s median / {s['final_p90_s']:.2f}s p90",
            )
        )
    if s.get("rtf") is not None:
        parts.append(f"RTF {s['rtf']:.3f}")
    if s.get("instability") is not None:
        parts.append(tr(f"改字 {sc.pct(s['instability'])}", f"revised {sc.pct(s['instability'])}"))
    if r.get("score"):
        parts.append(f"{r['score'].get('metric', 'CER')} {sc.pct(r['score']['cer'])}")
    return " · ".join(parts)


def card(r: dict, names: dict, t: float | None = None) -> None:
    with st.container(border=True):
        st.markdown(f"**{names.get(r['model'], r['model'])}**")
        tag = (
            tr("只往後加字、出現即定稿", "append-only, final as it appears")
            if r.get("append_only")
            else tr("部分結果，停頓後定稿", "partial text, final after a pause")
        )
        st.caption(f"`{r['model']}` · {tag}")
        if r["status"] == "waiting":
            st.info(tr("等待中…", "Waiting…"))
            return
        if r.get("error"):
            st.error(r["error"])
            return
        if r["status"] == "running" and not r.get("events"):
            st.info(tr("載入模型中…", "Loading the model…"), icon=":material/progress_activity:")
            return
        if r.get("stats"):
            st.caption(facts(r))
        elif r.get("now") is not None:
            st.caption(tr(f"已送出 {r['now']:.1f} 秒音訊", f"{r['now']:.1f} s of audio sent"))
        committed, partial = screen_text(r, t)
        shown = sc.show(committed)
        if partial:
            shown += f" :gray[*{sc.show(partial)}*]"
        with st.container(height=220, border=False):
            st.markdown((shown or "…") + (" ▌" if r["status"] == "running" else ""))
        if r.get("events") and r["status"] == "done":
            with st.expander(tr("事件紀錄", "Event log")):
                st.dataframe(event_table(r), hide_index=True, width="stretch")


def event_table(r: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                tr("音訊（秒）", "Audio (s)"): e["t_audio"],
                tr("出現（秒）", "Shown at (s)"): e["t_wall"],
                tr("定稿", "Final"): bool(e.get("final")),
                tr("文字", "Text"): sc.show(e.get("text") or ""),
            }
            for e in r.get("events") or []
        ]
    )


def cards(rows: list[dict], names: dict, t: float | None = None) -> None:
    cols = st.columns(max(len(rows), 1), gap="small")
    for col, r in zip(cols, rows, strict=False):
        with col:
            card(r, names, t)


def summary(rows: list[dict], names: dict, reference: str, segments: list) -> None:
    done = [r for r in rows if r.get("stats") and not r.get("error")]
    if not done:
        return
    st.subheader(tr("比較結果", "Results"))
    model = tr("模型", "Model")
    metric = next((r["score"].get("metric") for r in done if r.get("score")), "CER")
    table = pd.DataFrame(
        [
            {
                model: names.get(r["model"], r["model"]),
                tr("首字延遲（秒）", "First text (s)"): r["stats"]["first_partial_s"],
                tr("定稿延遲中位數（秒）", "Final, median (s)"): r["stats"]["final_median_s"],
                tr("定稿延遲 p90（秒）", "Final, p90 (s)"): r["stats"]["final_p90_s"],
                "RTF": r["stats"]["rtf"],
                tr("改字比例", "Revised"): r["stats"]["instability"],
                metric: (r.get("score") or {}).get("cer"),
            }
            for r in done
        ]
    )
    if not reference:
        table = table.drop(columns=[metric])
    pct = st.column_config.NumberColumn(format="percent")
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            metric: pct,
            tr("改字比例", "Revised"): st.column_config.NumberColumn(
                format="percent",
                help=tr(
                    "部分結果（灰字）顯示後又被改掉的字，占所有顯示過的字的比例",
                    "Share of the characters shown as partial text that were later changed",
                ),
            ),
            "RTF": st.column_config.NumberColumn(
                format="%.3f",
                help=tr(
                    "處理時間 ÷ 音訊長度；小於 1 才跟得上即時",
                    "Processing time ÷ audio length; below 1 keeps up with real time",
                ),
            ),  # fmt: skip
        },
    )
    long = pd.DataFrame(
        [
            {"model": names.get(r["model"], r["model"]), "t": f["end"], "latency": f["latency"]}
            for r in done
            for f in asr_streaming.finalisation(r.get("events") or [], segments)
        ]
    )
    if long.empty:
        return
    st.caption(
        tr(
            "每句話講完後，多久出現定稿文字（秒）。sherpa-onnx 模型要等句尾靜音 0.8 秒才定稿；"
            "Voxtral 不改字，每段文字一出現就是定稿。",
            "How long after each utterance ended its final text appeared (s). sherpa-onnx "
            "models commit after 0.8 s of trailing silence; Voxtral never revises, so its text "
            "is final as soon as it appears.",
        )
    )
    chart = (
        alt.Chart(long)
        .mark_line(point=True)
        .encode(
            x=alt.X("t:Q", title=tr("句子結束於第幾秒", "Utterance ended at (s)")),
            y=alt.Y("latency:Q", title=tr("定稿延遲（秒）", "Finalisation latency (s)")),
            color=alt.Color("model:N", title=model),
            tooltip=[
                alt.Tooltip("model:N", title=model),
                alt.Tooltip("t:Q", title=tr("秒", "s"), format=".1f"),
                alt.Tooltip("latency:Q", title=tr("延遲", "Latency"), format=".2f"),
            ],
        )
        .properties(height=240)
    )
    st.altair_chart(chart, width="stretch")


def notes(pace: str) -> None:
    timing = (
        tr(
            "**時間是模擬的**：音訊以最快速度送進模型，量每一小段的處理時間，再推算真人說話時"
            "文字何時出現（每段音訊到齊才開始處理、前一段處理完才輪到下一段）。",
            "**Times are simulated**: the audio is fed as fast as the model goes, the compute "
            "for each frame is measured, and the time a live listener would have seen the text "
            "is worked out from it (a frame starts once it has arrived and the previous one is "
            "done).",
        )
        if pace == "fast"
        else tr(
            "**麥克風即時**：瀏覽器每 100 ms 送一段麥克風音訊，時間是閘道實際量到的（含瀏覽器"
            "緩衝與網路）。",
            "**Live microphone**: the browser sent 100 ms of microphone audio at a time, and the "
            "times were measured by the gateway (including browser buffering and the network).",
        )
        if pace == "live"
        else tr(
            "**以真實速度送出**：每一小段音訊等到講者真的說到那裡才送出，時間是實際量到的。",
            "**Real-time pace**: each frame was sent only when the speaker would have reached "
            "it, and the times are measured.",
        )
    )
    st.caption(
        timing
        + tr(
            " 句子的起訖由音量偵測（VAD）從錄音找出：首字延遲＝開始說話到第一個字；定稿延遲＝"
            "一句講完到這句的最終文字。改字比例＝灰色部分結果顯示後又被改掉的字。",
            " Utterances are found in the recording by an energy VAD: first text = speech "
            "start to the first character; finalisation = end of an utterance to its final "
            "text. Revised = partial (grey) characters later changed.",
        )
    )


def baseline_note() -> None:
    """The earlier chunked simulation on meeting_zh, for comparison."""
    rows = " · ".join(f"{m} {lag:.2f}s／CER {cer * 100:.1f}%" for m, lag, cer in CHUNKED_BASELINE)
    st.caption(
        tr(
            f"對照：切段模擬（2 秒一段，合成會議）中位落後與逐段 CER：{rows}。切段模擬的落後"
            "是從每段結束算起，加上最長 2 秒的切段等待才是講完一個字到看到它的時間。",
            f"For comparison, chunked simulation (2 s chunks, synthetic meeting), median lag "
            f"and live CER: {rows}. Its lag counts from the end of each chunk; add up to 2 s "
            "of waiting for the chunk to fill for the time from a word to its text.",
        )
    )


def render(job: jobs.Job) -> None:
    state, names = job.state, sc.names("stream")
    rows = state["rows"]
    title = state["audio_name"]
    if state["clip_s"]:
        title += tr(f"（前 {state['clip_s']} 秒）", f" (first {state['clip_s']} s)")
    st.markdown(f"#### {title}")
    st.info(
        tr(
            f"真串流：每 {state['frame_ms']} ms 送一小段音訊"
            + ("（真實速度）" if state["pace"] == "realtime" else "（模擬即時）"),
            f"True streaming: {state['frame_ms']} ms frames"
            + (" (real-time pace)" if state["pace"] == "realtime" else " (simulated real time)"),
        ),
        icon=":material/stream:",
    )
    if job.running:
        cards(rows, names)
        sc.running_bar(job)
        return
    if state.get("clip_file") and Path(state["clip_file"]).exists():
        st.audio(state["clip_file"])
    t = replay(job.id, rows, state.get("duration") or 0)
    cards(rows, names, t)
    if job.error:
        st.error(job.error)
    summary(rows, names, state["reference"], state.get("segments") or [])
    notes(state["pace"])
    baseline_note()
    sc.saved_caption(state)


def replay(key: str, rows: list[dict], duration: float) -> float:
    end = max(
        [duration] + [e["t_wall"] for r in rows for e in r.get("events") or []], default=duration
    )
    top = float(round(end + 0.5, 1))
    return st.slider(
        tr("重播：講者說到第幾秒時，畫面上的字幕", "Replay: the captions on screen at second"),
        0.0, top, top, 0.5, key=f"ls_replay_{key}",
        help=tr("拖動看每個模型當時的字幕；灰色斜體是還沒定稿的部分結果。",
                "Drag to see each model's captions at that moment; grey italics are partial "
                "text not yet final."),
    )  # fmt: skip


@st.fragment(run_every=1)
def live_view(job: jobs.Job) -> None:
    if not job.running:
        st.rerun()
    render(job)


# -- page -----------------------------------------------------------------------------------


def stream_page(url: str, status: dict, chosen: tuple | None) -> None:
    labels = sc.names("stream")
    options = [m for m in labels if m in status]
    installed = [m for m in options if status[m].get("installed")]
    a, b, c = st.columns(3)
    st.session_state.setdefault("ls_frame", 160)
    frame = a.select_slider(
        tr("每次送出", "Frame"), FRAMES, format_func=lambda ms: f"{ms} ms", **kept("ls_frame"),
        help=tr("像麥克風一樣，一次送這麼長的音訊給模型。",
                "Audio is sent in pieces this long, as a microphone would."),
    )  # fmt: skip
    st.session_state.setdefault("ls_clip", 0)
    clip = b.select_slider(
        tr("只用開頭幾秒", "Use the first"),
        CLIPS,
        format_func=lambda s: tr(f"{s} 秒", f"{s} s") if s else tr("整段", "whole recording"),
        **kept("ls_clip"),
    )
    langs = {"zh": tr("中文（CER）", "Chinese (CER)"), "en": tr("英文（WER）", "English (WER)")}
    picked = (chosen[2], chosen[3]) if chosen else None
    if chosen and chosen[2] == "sample" and st.session_state.get("ls_language_for") != picked:
        st.session_state.ls_language = sc.sample_language(chosen[3])
    st.session_state.ls_language_for = picked
    st.session_state.setdefault("ls_language", "zh")
    language = c.selectbox(
        tr("評分方式", "Scoring"), list(langs), format_func=langs.get, **kept("ls_language"),
        help=tr("串流模型不用指定語言；這裡只決定錯誤率怎麼算。",
                "Streaming models take no language setting; this only picks the error rate."),
    )  # fmt: skip

    st.session_state.setdefault(
        "ls_models", [m for m in DEFAULT_MODELS if m in installed] or installed[:2]
    )
    st.session_state.ls_models = [m for m in st.session_state.ls_models if m in options]
    selected = st.multiselect(
        tr(f"真串流模型（最多 {MAX_MODELS} 個）", f"Streaming models (up to {MAX_MODELS})"),
        options,
        format_func=lambda m: label(m, status, labels),
        max_selections=MAX_MODELS,
        **kept("ls_models"),
    )
    sc.download_missing(selected, status, "ls")
    sc.licence_warning(selected, labels)
    realtime_warning(selected, labels)
    if language == "en":
        no_en = [labels.get(m, m) for m in selected if "en" not in languages(m)]
        if no_en:
            st.warning(
                tr("這些模型只認中文，英文錄音會辨識失敗：", "These models only know Chinese: ")
                + "、".join(no_en)
            )

    sample = chosen[3] if chosen and chosen[2] == "sample" else None
    wanted = (chosen[2] if chosen else None, sample, clip)
    if st.session_state.get("ls_reference_for") != wanted:
        st.session_state.ls_reference = sc.sample_reference(sample) if sample and not clip else ""
        st.session_state.ls_reference_for = wanted
    reference = st.text_area(
        tr("參考原稿（選填，用來計算最後的錯誤率）",
           "Reference transcript (optional, for the final error rate)"),
        height=90, **kept("ls_reference"),
        help=tr("要和實際使用的音訊範圍一致；範例錄音用整段時自動帶入。",
                "Must cover the audio actually used; filled in for a whole sample."),
    )  # fmt: skip
    st.session_state.setdefault("ls_realtime", False)
    realtime = st.toggle(
        tr("以真實速度送出（實際等待：1 分鐘的錄音要跑 1 分鐘）",
           "Feed at real-time pace (really waits: a 1-minute recording takes a minute)"),
        **kept("ls_realtime"),
        help=tr("關閉時以最快速度送出，再用量到的處理時間推算即時情況下的時間，結果相同但快很多。",
                "When off, audio is fed as fast as possible and live timings are computed from "
                "the measured processing time: the same result, much sooner."),
    )  # fmt: skip

    job = jobs.get(st.session_state.get("ls_job"))
    busy = bool(job and job.running)
    ready = [m for m in selected if status[m]["installed"]]
    if st.button(
        tr(f"開始真串流（{len(ready)} 個模型）", f"Start streaming ({len(ready)} models)"),
        type="primary", icon=":material/play_arrow:",
        disabled=busy or not ready or chosen is None,
    ):  # fmt: skip
        audio, filename, source, audio_name = chosen
        job = jobs.Job(
            "speech_stream",
            tr(f"真串流：{len(ready)} 個模型", f"True streaming: {len(ready)} models"),
            sc.PAGE,
            state={
                "audio": str(sc.cached_audio(audio, filename)),
                "audio_name": audio_name,
                "source": source,
                "clip_s": clip,
                "frame_ms": int(frame),
                "pace": "realtime" if realtime else "fast",
                "language": language,
                "reference": reference.strip(),
                "rows": [{"model": m, "status": "waiting", "text": ""} for m in ready],
            },
        )
        jobs.start(job, lambda j: stream_work(j, url))
        st.session_state.ls_job = job.id
        st.rerun()
    if chosen is None:
        st.caption(tr("先選擇或錄一段音訊。", "Choose or record some audio first."))
    if job and job.running:
        live_view(job)
    elif job:
        render(job)


# -- history --------------------------------------------------------------------------------


def fmt_s(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f} s"


def history_view(data: dict) -> None:
    names = sc.names("stream")
    rows = [r | {"status": "done"} for r in data.get("results") or [] if isinstance(r, dict)]
    bits = [tr("錄音：", "Audio: ") + f"`{data.get('audio', '?')}`"]
    if data.get("clip_s"):
        bits.append(tr(f"前 {data['clip_s']} 秒", f"first {data['clip_s']} s"))
    elif data.get("duration"):
        bits.append(tr(f"{data['duration']:.0f} 秒", f"{data['duration']:.0f} s"))
    if data.get("source") == "mic":
        bits.append(tr("麥克風即時串流", "live microphone stream"))
    else:
        bits.append(tr(f"真串流，每 {data.get('frame_ms')} ms 一段",
                       f"true streaming, {data.get('frame_ms')} ms frames"))  # fmt: skip
    if data.get("stopped"):
        bits.append(tr("中途停止", "stopped early"))
    st.caption(" · ".join(bits))
    audio = data.get("audio_file")
    if audio and (PROJECT_DIR / audio).exists():
        st.audio(str(PROJECT_DIR / audio))
    live = data.get("live") or {}
    if live.get("first_latencies") is not None:
        st.caption(
            tr(
                f"即時量到：首字延遲中位數 {fmt_s(live.get('first_median_s'))}、"
                f"定稿延遲中位數 {fmt_s(live.get('final_median_s'))}"
                f"（{len(live.get('first_latencies') or [])} 句）",
                f"Measured live: median first text {fmt_s(live.get('first_median_s'))}, "
                f"median final text {fmt_s(live.get('final_median_s'))} "
                f"({len(live.get('first_latencies') or [])} utterances)",
            )
        )
    summary(rows, names, data.get("reference") or "", data.get("segments") or [])
    cards(rows, names)
    notes(data.get("pace") or "fast")
    if data.get("reference"):
        with st.expander(tr("參考原稿", "Reference transcript")):
            st.write(data["reference"])


def history_summary(data: dict) -> dict:
    # Cached across languages (history.summarize): ids only, no tr().
    rows = [r for r in data.get("results") or [] if isinstance(r, dict)]
    scored = [(r["model"], r["score"]["cer"]) for r in rows if r.get("score")]
    metric = next((r["score"].get("metric") for r in rows if r.get("score")), None) or "CER"
    compute = sum((r.get("stats") or {}).get("compute_s") or 0 for r in rows)
    return {
        "title": f"{data.get('audio') or '?'} · stream {data.get('frame_ms') or '?'}ms",
        "models": [r["model"] for r in rows if r.get("model")],
        "seconds": round(compute, 2) or None,
        "snippet": next((r.get("text", "")[:220] for r in rows if r.get("text")), ""),
        "best": (min(scored, key=lambda x: x[1]), metric) if scored else None,
        "error": bool(rows) and all(r.get("error") for r in rows),
    }
