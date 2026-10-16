"""Meeting mode of the speech-model page: who spoke when, and what they said.

The gateway's diarization model (sherpa-onnx: pyannote segmentation-3.0 + 3D-Speaker CAM++)
finds the speaker turns; a speech model writes the words. They are joined in one of two ways:

- merge timestamps: transcribe the whole recording once and give each timed ASR segment the
  speaker who talks most during it (needs a model with sentence timestamps: Whisper family);
- transcribe each turn: cut the audio at the speaker turns and transcribe every turn on its
  own (works with any speech model, slower).

With a reference (RTTM, or "start end speaker" lines) the diarization error rate (DER) is
computed with a 0.25 s collar (local_llm.diarization.der).
"""

import html
from pathlib import Path

import altair as alt
import jobs
import pandas as pd
import speech_common as sc
import streamlit as st
from common import kept, save_run
from i18n import tr

from local_llm import audio_aug, gateway_client
from local_llm import diarization as dz

DIARIZER = "speaker-diarization"
COLLAR_S = 0.25
PAD_S = 0.15  # audio kept around each turn when it is transcribed on its own
SPEAKERS = [0, 1, 2, 3, 4, 5, 6, 8]  # 0 = estimate
COLORS = ["#4F46E5", "#DB2777", "#0F766E", "#D97706", "#0284C7", "#7C3AED", "#B91C1C", "#334155"]


def speaker_template() -> str:
    return tr("講者 {n}", "Speaker {n}")


def clock(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"


# -- worker (no Streamlit calls) ------------------------------------------------------------


def _compute(out: dict) -> float:
    load = (out.get("load_seconds") or 0) + (out.get("ollama_load_seconds") or 0)
    return max(out["seconds"] - load - (out.get("queue_seconds") or 0), 0.01)


def meeting_work(job: jobs.Job, url: str) -> None:
    state = job.state
    path = Path(state["audio"])
    job.detail, job.progress = "diarization", 0.05
    diar = gateway_client.diarize(
        path, state["num_speakers"] or None, threshold=state["threshold"], base_url=url
    )
    turns = diar["segments"]
    state["diar"] = {
        "segments": turns,
        "speakers": diar["speakers"],
        "seconds": round(_compute(diar), 2),
        "duration": diar.get("duration"),
    }
    if state["reference"]:
        try:
            reference = dz.parse_reference(state["reference"])
            score = dz.der(reference, turns, collar=COLLAR_S)
            score["mapping"] = {str(k): v for k, v in score["mapping"].items()}
            state["der"], state["reference_turns"] = score, reference
        except ValueError as exc:
            state["der_error"] = str(exc)
    labelled, asr_seconds = [], 0.0
    if job.cancel.is_set():
        pass
    elif state["method"] == "merge":
        job.detail, job.progress = state["asr"], 0.3
        out = gateway_client.transcribe(path, state["asr"], state["language"], base_url=url)
        asr_seconds = _compute(out)
        labelled = dz.label_segments(turns, out.get("segments") or [])
        state["asr_segments"] = len(out.get("segments") or [])
    else:
        samples = audio_aug.load(path)
        folder = sc.CACHE_DIR / sc.cache_key(path)
        merged = dz.merge_turns(turns, gap=0.5)
        for i, t in enumerate(merged):
            if job.cancel.is_set():
                break
            job.detail = f"{state['asr']} · {i + 1}/{len(merged)}"
            job.progress = 0.1 + 0.9 * i / max(len(merged), 1)
            a = max(int((t["start"] - PAD_S) * audio_aug.SR), 0)
            b = min(int((t["end"] + PAD_S) * audio_aug.SR), len(samples))
            if b - a < audio_aug.SR // 5:  # under 0.2 s: nothing to recognise
                continue
            piece = audio_aug.save(folder / f"turn_{t['start']:.2f}_{t['end']:.2f}.wav",
                                   samples[a:b])  # fmt: skip
            out = gateway_client.transcribe(piece, state["asr"], state["language"], base_url=url)
            asr_seconds += _compute(out)
            if out["text"].strip():
                labelled.append(t | {"text": out["text"].strip()})
        state["asr_segments"] = len(merged)
    state["labelled"], state["asr_seconds"] = labelled, round(asr_seconds, 2)
    job.progress = 1.0
    path_out = save_run(
        "asr_compare",
        {
            "mode": "meeting",
            "audio": state["audio_name"],
            "source": state["source"],
            "language": state["language"],
            "diarizer": DIARIZER,
            "asr": state["asr"],
            "method": state["method"],
            "num_speakers": state["num_speakers"] or None,
            "threshold": state["threshold"],
            "diarization": state["diar"],
            "asr_seconds": state["asr_seconds"],
            "asr_segments": state.get("asr_segments"),
            "labelled": labelled,
            "reference": state["reference"] or None,
            "der": state.get("der"),
            "der_error": state.get("der_error"),
            "stopped": job.cancel.is_set(),
        },
    )
    state["path"] = path_out.relative_to(path_out.parents[1])


# -- drawing --------------------------------------------------------------------------------


def timeline(turns: list[dict], reference: list[dict], mapping: dict) -> alt.Chart:
    """Who spoke when: the model's speakers on top, the reference below (same colours for
    speakers the DER mapping matched)."""
    names = dz.speaker_names((t["speaker"] for t in turns), speaker_template())
    model, ref = tr("模型", "Model"), tr("參考", "Reference")
    back = {}  # reference name -> the matched model speaker's label (same colour)
    for k, v in mapping.items():  # keys are model speaker ids, strings after a JSON round trip
        key = int(k) if str(k).isdigit() else k
        if key in names:
            back[v] = names[key]
    rows = [
        {"lane": f"{model} · {names[t['speaker']]}", "who": names[t["speaker"]],
         "start": t["start"], "end": t["end"]}
        for t in turns
    ] + [
        {"lane": f"{ref} · {t['speaker']}", "who": back.get(t["speaker"]) or str(t["speaker"]),
         "start": t["start"], "end": t["end"]}
        for t in reference
    ]  # fmt: skip
    data = pd.DataFrame(rows)
    data["span"] = data.apply(lambda r: f"{clock(r['start'])}–{clock(r['end'])}", axis=1)
    lanes = sorted(data["lane"].unique(), key=lambda x: (not x.startswith(model), x))
    whos = list(dict.fromkeys(data["who"]))
    return (
        alt.Chart(data)
        .mark_bar(cornerRadius=3, height={"band": 0.7})
        .encode(
            x=alt.X("start:Q", title=tr("時間（秒）", "Time (s)")),
            x2="end:Q",
            y=alt.Y("lane:N", sort=lanes, title=None),
            color=alt.Color(
                "who:N",
                scale=alt.Scale(domain=whos, range=COLORS[: max(len(whos), 1)] * 3),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("lane:N", title=tr("講者", "Speaker")),
                alt.Tooltip("span:N", title=tr("時間", "Time")),
            ],
        )
        .properties(height=36 * len(lanes) + 50)
    )


def transcript_html(labelled: list[dict]) -> str:
    names = dz.speaker_names((t["speaker"] for t in labelled), speaker_template())
    colors = {s: COLORS[i % len(COLORS)] for i, s in enumerate(names)}
    rows = []
    for t in labelled:
        c = colors[t["speaker"]]
        rows.append(
            f'<div style="display:flex;gap:12px;padding:6px 0;border-bottom:1px solid #E2E8F0">'
            f'<div style="flex:0 0 118px;font-size:.8rem;color:#64748B">'
            f"{clock(t['start'])}–{clock(t['end'])}<br>"
            f'<b style="color:{c}">{html.escape(names[t["speaker"]])}</b></div>'
            f'<div style="flex:1;line-height:1.7">{html.escape(sc.show(t["text"]))}</div></div>'
        )
    return "".join(rows)


def draw(result: dict, names: dict) -> None:
    """A finished meeting run (live job state or a saved run)."""
    diar = result.get("diar") or result.get("diarization") or {}
    turns, labelled = diar.get("segments") or [], result.get("labelled") or []
    score = result.get("der")
    cols = st.columns(4)
    cols[0].metric(tr("找到的講者", "Speakers found"), diar.get("speakers", 0))
    cols[1].metric(
        "DER",
        sc.pct(score["der"]) if score else "—",
        help=tr(
            "說話者分段錯誤率＝漏掉＋多出＋認錯人的時間÷參考講話時間"
            f"（邊界前後 {COLLAR_S} 秒不計）。需要參考答案。",
            f"Diarization error rate = (missed + false alarm + wrong speaker) time ÷ reference "
            f"speech time ({COLLAR_S} s around boundaries not scored). Needs a reference.",
        ),
    )
    cols[2].metric(tr("分段耗時", "Diarization"), f"{diar.get('seconds', 0):.1f} s")
    cols[3].metric(
        tr("辨識耗時", "Transcription"),
        f"{result.get('asr_seconds') or 0:.1f} s",
        help=names.get(result.get("asr"), result.get("asr")),
    )
    if score:
        st.caption(
            tr(
                f"漏掉 {sc.pct(score['miss'])} · 多出 {sc.pct(score['false_alarm'])} · "
                f"認錯人 {sc.pct(score['confusion'])}（參考講話 {score['speech']:.0f} 秒）",
                f"Missed {sc.pct(score['miss'])} · false alarm {sc.pct(score['false_alarm'])} · "
                f"wrong speaker {sc.pct(score['confusion'])} (reference speech "
                f"{score['speech']:.0f} s)",
            )
        )
    if result.get("der_error"):
        st.warning(tr("參考答案讀不懂：", "Could not read the reference: ") + result["der_error"])
    if turns:
        reference = result.get("reference_turns") or []
        if not reference and result.get("reference") and score:
            try:
                reference = dz.parse_reference(result["reference"])
            except ValueError:
                reference = []
        st.markdown(tr("**誰在什麼時候說話**", "**Who spoke when**"))
        st.altair_chart(
            timeline(turns, reference, (score or {}).get("mapping") or {}), width="stretch"
        )
    if labelled:
        st.markdown(tr("**標上講者的逐字稿**", "**Speaker-labelled transcript**"))
        with st.container(height=420, border=True):
            st.markdown(transcript_html(labelled), unsafe_allow_html=True)
        text = dz.transcript(
            [t | {"text": sc.show(t["text"])} for t in labelled],
            speaker_template(),
            tr("：", ": "),
        )
        a, b = st.columns(2)
        a.download_button(
            tr("下載逐字稿", "Download transcript"),
            text,
            file_name="meeting_transcript.txt",
            icon=":material/download:",
        )
        b.download_button(
            tr("下載講者時間軸（RTTM）", "Download speaker turns (RTTM)"),
            dz.to_rttm(turns, "audio", dz.speaker_names((t["speaker"] for t in turns), "spk{n}")),
            file_name="diarization.rttm",
            icon=":material/download:",
        )
    elif turns:
        st.info(tr("辨識模型沒有輸出文字。", "The speech model returned no text."))
    method = result.get("method")
    st.caption(
        (
            tr(
                "合併時間軸：整段辨識一次，每個辨識片段歸給那段時間說最多話的講者。",
                "Merge timestamps: the recording is transcribed once and each ASR segment "
                "goes to the speaker who talks most during it.",
            )
            if method == "merge"
            else tr(
                "逐段辨識：依講者分段切開錄音，每段分別辨識。",
                "Transcribe each turn: the audio is cut at the speaker turns and each turn is "
                "transcribed on its own.",
            )
        )
        + " "
        + tr(
            "說話者分段：pyannote segmentation-3.0 ＋ 3D-Speaker CAM++（sherpa-onnx，CPU）。",
            "Diarization: pyannote segmentation-3.0 + 3D-Speaker CAM++ (sherpa-onnx, CPU).",
        )
    )


def render(job: jobs.Job) -> None:
    state, names = job.state, sc.names()
    st.markdown(f"#### {state['audio_name']}")
    if job.running:
        sc.running_bar(job)
        if state.get("diar"):
            st.caption(
                tr(
                    f"已找到 {state['diar']['speakers']} 位講者，正在辨識文字…",
                    f"Found {state['diar']['speakers']} speakers; transcribing…",
                )
            )
        return
    if job.error:
        st.error(job.error)
        return
    draw(state, names)
    sc.saved_caption(state)


@st.fragment(run_every=1)
def live_view(job: jobs.Job) -> None:
    if not job.running:
        st.rerun()
    render(job)


# -- page -----------------------------------------------------------------------------------


def meeting_page(url: str, status: dict) -> None:
    labels = sc.names()
    installed = [m for m in labels if status.get(m, {}).get("installed")]
    diarizer = status.get(DIARIZER)
    if not diarizer:
        st.error(tr("模型閘道沒有說話者分段模型。", "The gateway has no diarization model."))
        return
    if not diarizer["installed"]:
        st.info(
            tr(
                "會議模式需要說話者分段模型（約 34MB，CPU 執行）。",
                "Meeting mode needs the speaker diarization model (about 34 MB, runs on CPU).",
            ),
            icon=":material/record_voice_over:",
        )
        sc.download_missing([DIARIZER], status, "mt")
    chosen = sc.audio_choice("mt", default="meeting_zh.wav")
    sc.follow("mt", "reference", chosen, lambda s: sc.ref_file(s, ".rttm"))
    c1, c2, c3 = st.columns([1.3, 1, 1.4])
    options = installed or list(labels)
    st.session_state.setdefault(
        "mt_asr", "breeze-asr-25" if "breeze-asr-25" in options else options[0]
    )
    if st.session_state.mt_asr not in options:
        st.session_state.mt_asr = options[0]
    asr = c1.selectbox(
        tr("語音辨識模型", "Speech model"),
        options,
        format_func=lambda m: labels.get(m, m),
        **kept("mt_asr"),
    )
    # A bundled sample with reference turns starts at its true speaker count: estimating it
    # over-splits real far-field meetings (see docs/GATEWAY.md); your own audio starts at
    # "Estimate".
    picked = (chosen[2], chosen[3]) if chosen else None
    if st.session_state.get("mt_speakers_for") != picked:
        ref = sc.ref_file(chosen[3], ".rttm") if chosen and chosen[2] == "sample" else ""
        count = len({t["speaker"] for t in dz.parse_reference(ref)}) if ref else 0
        st.session_state.mt_speakers = count if count in SPEAKERS else 0
        st.session_state.mt_speakers_for = picked
    st.session_state.setdefault("mt_speakers", 0)
    speakers = c2.selectbox(
        tr("講者人數", "Speakers"),
        SPEAKERS,
        format_func=lambda n: str(n) if n else tr("自動估計", "Estimate"),
        **kept("mt_speakers"),
    )
    methods = {
        "merge": tr("合併時間軸", "Merge timestamps"),
        "turns": tr("逐段辨識", "Transcribe each turn"),
    }
    st.session_state.setdefault("mt_method", "merge")
    method = (
        c3.segmented_control(
            tr("怎麼對上講者", "Attribution"),
            list(methods),
            format_func=methods.get,
            help=tr(
                "合併時間軸：整段辨識一次，用辨識片段的時間對上講者"
                "（需要 Whisper 系列這種有句子時間軸的模型）。"
                "逐段辨識：依講者切開錄音再分別辨識，任何模型都能用，但比較慢。",
                "Merge timestamps: transcribe once and match the ASR segments' times to the "
                "speakers (needs sentence timestamps, as the Whisper family gives). Transcribe "
                "each turn: cut the audio at the speaker turns and transcribe each; works with "
                "any model but is slower.",
            ),
            **kept("mt_method"),
        )
        or "merge"
    )
    if method == "merge" and not sc.has_timestamps(asr):
        st.warning(
            tr(
                "這個模型不給句子時間軸，整段文字會歸給同一位講者；建議改用「逐段辨識」。",
                "This model gives no sentence timestamps, so all its text would go to one "
                "speaker; use 'Transcribe each turn' instead.",
            )
        )
    threshold = 0.7
    with st.expander(tr("進階：分群門檻", "Advanced: clustering threshold")):
        st.session_state.setdefault("mt_threshold", 0.7)
        threshold = st.slider(
            tr("講者分群門檻（越低分出越多人；講者人數設定時不用）",
               "Clustering threshold (lower = more speakers; unused when the count is set)"),
            0.3, 1.2, step=0.05, **kept("mt_threshold"),
        )  # fmt: skip
    reference = st.text_area(
        tr("參考講者時間軸（選填，用來計算 DER）", "Reference speaker turns (optional, for DER)"),
        height=110,
        placeholder="SPEAKER meeting 1 0.00 11.34 <NA> <NA> A <NA> <NA>\n0:12.1 0:17.8 B",
        help=tr(
            "RTTM，或每行「開始 結束 講者」（秒或 分:秒）。"
            "範例會議附有參考時間軸（由產生錄音的腳本推得）。",
            "RTTM, or one 'start end speaker' line per turn (seconds or m:ss). The sample "
            "meeting comes with reference turns (derived from the script that made it).",
        ),
        **kept("mt_reference"),
    )

    job = jobs.get(st.session_state.get("mt_job"))
    busy = bool(job and job.running)
    ready = bool(diarizer["installed"] and asr in installed)
    if st.button(
        tr("分析會議", "Analyse the meeting"), type="primary", icon=":material/play_arrow:",
        disabled=busy or not ready or chosen is None,
    ):  # fmt: skip
        audio, filename, source, audio_name = chosen
        job = jobs.Job(
            "speech_meeting",
            tr(f"會議講者分析：{audio_name}", f"Meeting speakers: {audio_name}"),
            sc.PAGE,
            state={
                "audio": str(sc.cached_audio(audio, filename)),
                "audio_name": audio_name,
                "source": source,
                "asr": asr,
                "language": "zh",
                "num_speakers": speakers,
                "threshold": threshold,
                "method": method,
                "reference": reference.strip(),
            },
        )
        jobs.start(job, lambda j: meeting_work(j, url))
        st.session_state.mt_job = job.id
        st.rerun()
    if chosen is None:
        st.caption(tr("先選擇或錄一段音訊。", "Choose or record some audio first."))
    if job and job.running:
        live_view(job)
    elif job:
        render(job)


# -- history --------------------------------------------------------------------------------


def history_view(data: dict) -> None:
    names = sc.names()
    bits = [tr("錄音：", "Audio: ") + f"`{data.get('audio', '?')}`"]
    bits.append(tr("辨識：", "ASR: ") + names.get(data.get("asr"), str(data.get("asr"))))
    if data.get("num_speakers"):
        bits.append(
            tr(f"指定 {data['num_speakers']} 位講者", f"{data['num_speakers']} speakers set")
        )
    if data.get("stopped"):
        bits.append(tr("中途停止", "stopped early"))
    st.caption(" · ".join(bits))
    draw(data, names)
    if data.get("reference"):
        with st.expander(tr("參考講者時間軸", "Reference speaker turns")):
            st.code(data["reference"], language=None)


def history_summary(data: dict) -> dict:
    diar = data.get("diarization") or {}
    labelled = [t for t in data.get("labelled") or [] if isinstance(t, dict)]
    der = data.get("der")
    names = dz.speaker_names((t.get("speaker") for t in labelled), "S{n}")
    snippet = " ".join(f"{names[t.get('speaker')]}: {t.get('text', '')}" for t in labelled[:3])
    # Cached across languages (history.summarize): no tr() here.
    return {
        "title": f"{data.get('audio') or '?'} · S1–S{diar.get('speakers', '?')}",
        "models": [m for m in (data.get("diarizer"), data.get("asr")) if m],
        "seconds": round((diar.get("seconds") or 0) + (data.get("asr_seconds") or 0), 2) or None,
        "snippet": snippet[:220],
        "best": ((data.get("diarizer") or DIARIZER, der["der"]), "DER") if der else None,
        "error": not diar.get("segments"),
    }
