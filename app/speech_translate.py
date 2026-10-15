"""Translate mode of the speech-model page: Chinese speech -> English text, three ways.

- Cascade: a speech model writes the Chinese transcript, an LLM in Ollama translates it
  (TranslateGemma gets its own prompt template, other chat models a plain instruction).
- Direct: an audio LLM (Gemma 4 E2B / E4B through the gateway) hears the audio and is told
  to answer in English.
- Whisper translate: Whisper large-v3's built-in task="translate".

Each path reports the English text, when the first text appeared and the total time
(model loading excluded), and chrF / BLEU against a reference translation when there is one.
The live simulation feeds the audio in N-second chunks, as a live speaker would produce it,
and shows when each chunk's English would appear (local_llm.speech_translate.live_schedule).
"""

from pathlib import Path

import altair as alt
import jobs
import pandas as pd
import speech_common as sc
import streamlit as st
from common import kept, ollama_url, save_run
from i18n import tr

from local_llm import asr_metrics, audio_aug, catalog, gateway_client
from local_llm import speech_translate as stx
from local_llm.streaming import Backend, stream_chat

PATHS = ("cascade", "direct", "whisper")
CLIPS = [30, 60, 120, 0]  # seconds; 0 = the whole recording
CHUNKS = [3, 5, 8, 10, 15]
SAVED = ("path", "asr", "llm", "model", "text", "source_text", "first_s", "total_s", "asr_s",
         "llm_s", "load_s", "score", "chunks", "error")  # fmt: skip


def path_label(path: str) -> str:
    return {
        "cascade": tr("串接：辨識 → LLM 翻譯", "Cascade: ASR → LLM"),
        "direct": tr("直接：音訊 LLM", "Direct: audio LLM"),
        "whisper": tr("Whisper 內建翻譯", "Whisper translate"),
    }[path]


def row_name(r: dict, names: dict) -> str:
    if r["path"] == "cascade":
        return f"{names.get(r['asr'], r['asr'])} → {r['llm']}"
    return names.get(r["model"], r["model"])


# -- worker (no Streamlit calls) ------------------------------------------------------------


def _transcribe(path: Path, model: str, url: str) -> tuple[str, float]:
    out = gateway_client.transcribe(path, model, "zh", base_url=url)
    return out["text"], round(_compute(out), 2)


def _compute(out: dict) -> float:
    load = (out.get("load_seconds") or 0) + (out.get("ollama_load_seconds") or 0)
    return max(out["seconds"] - load - (out.get("queue_seconds") or 0), 0.01)


def _llm(text: str, model: str, base: str, row: dict | None = None) -> tuple[str, float, float]:
    """Translate with an Ollama LLM: (English, seconds to first text, total), load excluded."""
    if not text.strip():
        return "", 0.0, 0.0
    prompt = stx.translation_prompt(text, model, script=asr_metrics.script(text))
    answer, stats = "", None
    for event in stream_chat(
        Backend("Ollama", "ollama", base), model, [{"role": "user", "content": prompt}],
        think=False, temperature=0, max_tokens=2048,
    ):  # fmt: skip
        if event.kind == "content":
            answer += event.text
            if row is not None:
                row["text"] = answer
        elif event.kind == "done":
            stats = event.stats
    load = (stats.load_s or 0) if stats else 0
    total = max((stats.total_s if stats else 0) - load, 0.01)
    first = max(((stats.first_answer_s or stats.total_s) if stats else 0) - load, 0.01)
    return stx.clean_output(answer), round(first, 2), round(total, 2)


def _direct(path: Path, model: str, url: str) -> tuple[str, float]:
    out = gateway_client.translate(path, model, "zh", base_url=url)
    return stx.clean_output(out["text"]), round(_compute(out), 2)


def translate_work(job: jobs.Job, url: str, base: str) -> None:
    state = job.state
    rows, reference = state["rows"], state["reference"]
    source = Path(state["audio"])
    samples = audio_aug.load(source)
    if state["clip_s"]:
        samples = samples[: int(state["clip_s"] * audio_aug.SR)]
    folder = sc.CACHE_DIR / sc.cache_key(source)
    clip = folder / f"clip{state['clip_s'] or 'all'}.wav"
    if not clip.exists():
        audio_aug.save(clip, samples)
    state["duration"] = round(len(samples) / audio_aug.SR, 2)
    state["clip_file"] = str(clip)
    chunks = []
    if state["live"]:
        bounds = stx.chunk_bounds(len(samples), state["chunk_s"])
        for i, (a, b) in enumerate(bounds):
            f = folder / f"clip{state['clip_s'] or 'all'}_chunk{state['chunk_s']}s_{i:03d}.wav"
            if not f.exists():
                audio_aug.save(f, samples[a:b])
            chunks.append({"file": f, "start": a / audio_aug.SR, "end": b / audio_aug.SR})
    asr_cache: dict = {}  # (asr model, chunk index or None) -> (text, seconds)

    def asr(model: str, path: Path, key) -> tuple[str, float]:
        if (model, key) not in asr_cache:
            asr_cache[(model, key)] = _transcribe(path, model, url)
        return asr_cache[(model, key)]

    for n, r in enumerate(rows, start=1):
        if job.cancel.is_set():
            break
        r["status"] = "running"
        job.progress = (n - 1) / len(rows)
        job.detail = f"{n}/{len(rows)} {r.get('llm') or r.get('model')}"
        try:
            if not state["live"]:
                if r["path"] == "cascade":
                    r["source_text"], r["asr_s"] = asr(r["asr"], clip, None)
                    r["text"], first, r["llm_s"] = _llm(r["source_text"], r["llm"], base, r)
                    r["first_s"] = round(r["asr_s"] + first, 2)
                    r["total_s"] = round(r["asr_s"] + r["llm_s"], 2)
                else:
                    r["text"], seconds = _direct(clip, r["model"], url)
                    r["first_s"] = r["total_s"] = round(seconds, 2)
            else:
                r["chunks"], texts = [], []
                for i, c in enumerate(chunks):
                    if job.cancel.is_set():
                        break
                    job.detail = f"{n}/{len(rows)} · chunk {i + 1}/{len(chunks)}"
                    if r["path"] == "cascade":
                        heard, asr_s = asr(r["asr"], c["file"], i)
                        text, _, llm_s = _llm(heard, r["llm"], base)
                        compute = asr_s + llm_s
                    else:
                        heard, (text, compute) = "", _direct(c["file"], r["model"], url)
                    texts.append(text)
                    r["chunks"].append(
                        {"start": round(c["start"], 2), "end": round(c["end"], 2),
                         "source": heard, "text": text, "compute": round(compute, 2)}
                    )  # fmt: skip
                    schedule = stx.live_schedule(
                        [x["end"] for x in r["chunks"]], [x["compute"] for x in r["chunks"]]
                    )
                    for x, s in zip(r["chunks"], schedule, strict=True):
                        x.update(finish=s["finish"], lag=s["lag"])
                    r["text"] = " ".join(t for t in texts if t)
                if r["chunks"]:
                    r["first_s"] = r["chunks"][0]["finish"]
                    r["total_s"] = round(sum(x["compute"] for x in r["chunks"]), 2)
                    r["lag_end"] = r["chunks"][-1]["lag"]
            r["score"] = stx.score(r.get("text") or "", reference) if reference else None
        except Exception as exc:
            r["error"] = str(exc)
        r["status"] = "done"
    for r in rows:
        r["status"] = "done"
    job.progress = 1.0
    path = save_run(
        "asr_compare",
        {
            "mode": "translate",
            "audio": state["audio_name"],
            "source": state["source"],
            "clip_s": state["clip_s"] or None,
            "duration": state["duration"],
            "live": state["live"],
            "chunk_s": state["chunk_s"] if state["live"] else None,
            "reference": reference or None,
            "stopped": job.cancel.is_set(),
            "results": [{k: r.get(k) for k in (*SAVED, "lag_end")} for r in rows],
        },
    )
    state["path"] = path.relative_to(path.parents[1])


# -- drawing --------------------------------------------------------------------------------


def facts(r: dict, live: bool) -> str:
    parts = []
    if live:
        parts.append(tr(f"第一段文字 {r['first_s']:.1f}s", f"first text at {r['first_s']:.1f}s"))
        parts.append(tr(f"結束時落後 {r['lag_end']:.1f}s", f"{r['lag_end']:.1f}s behind at end"))
    else:
        if r.get("path") == "cascade":
            parts.append(tr(f"首字 {r['first_s']:.1f}s", f"first text {r['first_s']:.1f}s"))
        parts.append(tr(f"總計 {r['total_s']:.1f}s", f"total {r['total_s']:.1f}s"))
        if r.get("path") == "cascade":
            parts.append(tr(f"（辨識 {r['asr_s']:.1f}s＋翻譯 {r['llm_s']:.1f}s）",
                            f"(ASR {r['asr_s']:.1f}s + LLM {r['llm_s']:.1f}s)"))  # fmt: skip
    if r.get("score"):
        parts.append(f"chrF {r['score']['chrf']:.1f} · BLEU {r['score']['bleu']:.1f}")
    return " · ".join(parts)


def chunk_table(r: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                tr("音訊", "Audio"): f"{c['start']:.0f}–{c['end']:.0f}s",
                tr("處理（秒）", "Compute (s)"): c["compute"],
                tr("落後（秒）", "Lag (s)"): c.get("lag"),
                tr("英文", "English"): c["text"],
            }
            for c in r.get("chunks") or []
        ]
    )


def card(r: dict, names: dict, live: bool) -> None:
    with st.container(border=True):
        st.markdown(f"**{row_name(r, names)}**")
        st.caption(path_label(r["path"]))
        if r["status"] == "waiting":
            st.info(tr("等待中…", "Waiting…"))
            return
        if r.get("error"):
            st.error(r["error"])
            return
        if r["status"] == "running" and not r.get("text"):
            st.info(tr("翻譯中…", "Translating…"), icon=":material/progress_activity:")
            return
        if r["status"] == "done" and r.get("total_s") is not None:
            st.caption(facts(r, live))
        with st.container(height=200, border=False):
            st.write((r.get("text") or "") + (" ▌" if r["status"] == "running" else ""))
        if live and r.get("chunks"):
            with st.expander(tr("逐段結果", "Chunk by chunk")):
                st.dataframe(chunk_table(r), hide_index=True, width="stretch")
        if r.get("source_text"):
            with st.expander(tr("第一步的中文逐字稿", "Step 1: Chinese transcript")):
                st.write(sc.show(r["source_text"]))


def summary(rows: list[dict], names: dict, live: bool, reference: str) -> None:
    done = [r for r in rows if r.get("total_s") is not None and not r.get("error")]
    if not done:
        return
    st.subheader(tr("比較結果", "Results"))
    name, path = tr("組合", "Setup"), tr("方式", "Path")
    table = pd.DataFrame(
        [
            {
                name: row_name(r, names),
                path: path_label(r["path"]),
                **(
                    {
                        tr("第一段文字（秒）", "First text (s)"): r["first_s"],
                        tr("結束時落後（秒）", "Lag at end (s)"): r.get("lag_end"),
                    }
                    if live
                    else {
                        tr("首字（秒）", "First text (s)"): r["first_s"],
                        tr("總計（秒）", "Total (s)"): r["total_s"],
                    }
                ),
                "chrF": (r.get("score") or {}).get("chrf"),
                "BLEU": (r.get("score") or {}).get("bleu"),
            }
            for r in done
        ]
    )
    if not reference:
        table = table.drop(columns=["chrF", "BLEU"])
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "chrF": st.column_config.ProgressColumn(
                help=tr(
                    "字元 n-gram F 分數（0–100，越高越接近參考譯文）；短文字比 BLEU 穩定",
                    "Character n-gram F-score (0–100, higher = closer to the reference); "
                    "steadier than BLEU on short texts",
                ),
                min_value=0.0,
                max_value=100.0,
                format="%.1f",
            ),
            "BLEU": st.column_config.NumberColumn(format="%.1f"),
        },
    )
    if live:
        long = pd.DataFrame(
            [
                {"setup": row_name(r, names), "t": c["end"], "lag": c["lag"]}
                for r in done
                for c in r.get("chunks") or []
            ]
        )
        if not long.empty:
            st.caption(
                tr(
                    "每段音訊講完後，英文晚多久出現（秒）。線一直往上表示跟不上即時。",
                    "How long after each chunk was spoken its English appears (s). A line that "
                    "keeps rising means the setup cannot keep up with real time.",
                )
            )
            chart = (
                alt.Chart(long)
                .mark_line(point=True)
                .encode(
                    x=alt.X("t:Q", title=tr("講者說到第幾秒", "Speaker time (s)")),
                    y=alt.Y("lag:Q", title=tr("落後（秒）", "Lag (s)")),
                    color=alt.Color("setup:N", title=name),
                    tooltip=["setup:N", "t:Q", "lag:Q"],
                )
                .properties(height=260)
            )
            st.altair_chart(chart, width="stretch")
    else:
        y = tr("總計（秒）", "Total (s)")
        charts = st.columns(2 if reference else 1)
        charts[0].caption(tr("總時間（不含載入模型，越低越快）", "Total time (load excluded)"))
        charts[0].bar_chart(table, x=name, y=y, horizontal=True, sort=y)
        if reference:
            charts[1].caption(tr("chrF（越高越好）", "chrF (higher is better)"))
            charts[1].bar_chart(table, x=name, y="chrF", horizontal=True, sort="-chrF")


def notes(live: bool) -> None:
    text = [
        tr(
            "時間不含載入模型；同一台機器依序執行（GPU 與其他工作共用時數字會偏慢）。"
            "直接與 Whisper 路線一次回傳整段文字，所以首字＝總計。",
            "Times exclude loading the models; everything runs one after another on this "
            "machine (slower when the GPU is shared). The direct and Whisper paths return all "
            "their text at once, so first text = total.",
        )
    ]
    if live:
        text = [
            tr(
                "**這是即時輸入的模擬**：錄音切成固定長度的片段，假設講者每講完一段才送出；"
                "每段要等前一段處理完才開始。落後＝這段英文出現的時間－這段講完的時間。"
                "沒有真的串流到模型，也沒有用前文；切點可能切斷句子。",
                "**This simulates live input**: the recording is cut into fixed-length chunks "
                "as if each were sent when the speaker finished it, and each chunk waits for "
                "the previous one. Lag = when the chunk's English appears minus when it was "
                "spoken. Nothing is actually streamed to the models, no earlier context is "
                "used, and cuts may split sentences.",
            )
        ]
    st.caption(" ".join(text))


def render(job: jobs.Job) -> None:
    state, names = job.state, sc.names()
    rows, live = state["rows"], state["live"]
    title = state["audio_name"]
    if state["clip_s"]:
        title += tr(f"（前 {state['clip_s']} 秒）", f" (first {state['clip_s']} s)")
    st.markdown(f"#### {title}")
    if live:
        st.info(
            tr(
                f"模擬即時輸入：每 {state['chunk_s']} 秒送出一段",
                f"Simulated live input: a chunk every {state['chunk_s']} s",
            ),
            icon=":material/sensors:",
        )
    per_row = 3 if len(rows) > 2 else max(len(rows), 1)
    for start in range(0, len(rows), per_row):
        cols = st.columns(per_row, gap="small")
        for col, r in zip(cols, rows[start : start + per_row], strict=False):
            with col:
                card(r, names, live)
    if job.running:
        sc.running_bar(job)
        return
    if job.error:
        st.error(job.error)
    summary(rows, names, live, state["reference"])
    notes(live)
    sc.saved_caption(state)


@st.fragment(run_every=1)
def live_view(job: jobs.Job) -> None:
    if not job.running:
        st.rerun()
    render(job)


# -- page -----------------------------------------------------------------------------------


def translate_page(url: str, status: dict) -> None:
    labels = sc.names()
    installed = [m for m in labels if status.get(m, {}).get("installed")]
    chosen = sc.audio_choice("st", default="meeting_zh.wav")
    a, b = st.columns(2)
    st.session_state.setdefault("st_clip", 0)
    clip = a.select_slider(
        tr("只用開頭幾秒", "Use the first"),
        CLIPS,
        format_func=lambda s: tr(f"{s} 秒", f"{s} s") if s else tr("整段", "whole recording"),
        **kept("st_clip"),
    )
    st.session_state.setdefault("st_live", False)
    live = b.toggle(
        tr("模擬即時輸入（逐段送出）", "Simulate live input (chunk by chunk)"),
        help=tr(
            "把錄音切成固定長度的片段依序處理，看英文落後講者多久。",
            "Cuts the recording into fixed-length chunks processed in order, to see how far "
            "the English trails the speaker.",
        ),
        **kept("st_live"),
    )
    chunk = 5
    if live:
        st.session_state.setdefault("st_chunk", 5)
        chunk = b.select_slider(
            tr("每段長度", "Chunk length"), CHUNKS, format_func=lambda s: f"{s} s",
            **kept("st_chunk"),
        )  # fmt: skip

    # The bundled reference translation covers a whole sample, so only then is it filled in.
    sample = chosen[3] if chosen and chosen[2] == "sample" else None
    wanted = (chosen[2] if chosen else None, sample, clip)
    if st.session_state.get("st_reference_for") != wanted:
        whole = sample and not clip
        st.session_state.st_reference = sc.ref_file(sample, ".en.txt") if whole else ""
        st.session_state.st_reference_for = wanted
    reference = st.text_area(
        tr("參考英文譯文（選填，用來計算 chrF／BLEU）",
           "Reference English translation (optional, for chrF / BLEU)"),
        height=100,
        help=tr(
            "要和實際翻譯的音訊範圍一致；範例會議錄音附有本專案撰寫的參考譯文（整段時自動帶入）。",
            "Must cover the same audio that is translated; the sample meeting comes with a "
            "reference written for this project (filled in when the whole recording is used).",
        ),
        **kept("st_reference"),
    )  # fmt: skip

    st.markdown(tr("**要比較的翻譯方式**", "**Translation paths to compare**"))
    st.session_state.setdefault("st_paths", list(PATHS))
    paths = st.pills(
        tr("方式", "Paths"),
        list(PATHS),
        format_func=path_label,
        selection_mode="multi",
        label_visibility="collapsed",
        **kept("st_paths"),
    )
    rows: list[dict] = []
    if "cascade" in paths:
        with st.container(border=True):
            st.markdown(f":material/linear_scale: **{path_label('cascade')}**")
            c1, c2 = st.columns([1, 2])
            asr_options = installed or list(labels)
            st.session_state.setdefault(
                "st_asr", "breeze-asr-25" if "breeze-asr-25" in asr_options else asr_options[0]
            )
            if st.session_state.st_asr not in asr_options:
                st.session_state.st_asr = asr_options[0]
            asr_model = c1.selectbox(
                tr("語音辨識模型", "Speech model"),
                asr_options,
                format_func=lambda m: labels.get(m, m),
                **kept("st_asr"),
            )
            chat = [m for m in sc.ollama_models(ollama_url()) if catalog.is_chat(m)]
            st.session_state.setdefault(
                "st_llms", [m for m in chat if m.startswith("translategemma")][:2] or chat[:1]
            )
            st.session_state.st_llms = [m for m in st.session_state.st_llms if m in chat]
            llms = c2.multiselect(
                tr("翻譯用的 LLM（Ollama 已下載）", "Translation LLMs (installed in Ollama)"),
                chat,
                max_selections=4,
                **kept("st_llms"),
            )
            if not any(m.startswith("translategemma") for m in chat):
                st.caption(
                    tr(
                        "建議到模型庫下載 translategemma:4b／12b（Google 的翻譯專用模型）。",
                        "Tip: download translategemma:4b / 12b (Google's translation models) "
                        "from the model library.",
                    )
                )
            rows += [{"path": "cascade", "asr": asr_model, "llm": m} for m in llms]
    if "direct" in paths:
        audio_llms = [m for m in labels if sc.backend(m) == "ollama-audio"]
        with st.container(border=True):
            st.markdown(f":material/hearing: **{path_label('direct')}**")
            st.session_state.setdefault(
                "st_direct", [m for m in audio_llms if m in installed] or audio_llms[:1]
            )
            direct = st.multiselect(
                tr("音訊 LLM（聽音訊、直接寫英文）", "Audio LLMs (hear the audio, write English)"),
                audio_llms,
                format_func=lambda m: sc.model_label(m, status, labels),
                **kept("st_direct"),
            )
            sc.download_missing(direct, status, "st")
            rows += [{"path": "direct", "model": m} for m in direct if m in installed]
    if "whisper" in paths:
        whispers = [m for m in labels if sc.backend(m) == "mlx-whisper" and sc.translates(m)]
        with st.container(border=True):
            st.markdown(f":material/translate: **{path_label('whisper')}**")
            st.session_state.setdefault(
                "st_whisper", [m for m in whispers if m in installed] or whispers[:1]
            )
            whisper = st.multiselect(
                tr("Whisper 模型", "Whisper models"),
                whispers,
                format_func=lambda m: sc.model_label(m, status, labels),
                **kept("st_whisper"),
            )
            st.caption(
                tr(
                    "large-v3-turbo 訓練時拿掉了翻譯；Breeze-ASR-25 與 Taiwan-Tongues "
                    "微調後只會寫中文，所以不在清單上。",
                    "large-v3-turbo was trained without translation, and Breeze-ASR-25 and "
                    "Taiwan-Tongues only write Chinese after fine-tuning, so they are not listed.",
                )
            )
            sc.download_missing(whisper, status, "st")
            rows += [{"path": "whisper", "model": m} for m in whisper if m in installed]

    job = jobs.get(st.session_state.get("st_job"))
    busy = bool(job and job.running)
    label = tr(f"開始翻譯（{len(rows)} 種組合）", f"Translate ({len(rows)} setups)")
    if st.button(
        label, type="primary", icon=":material/play_arrow:",
        disabled=busy or not rows or chosen is None,
    ):  # fmt: skip
        audio, filename, source, audio_name = chosen
        job = jobs.Job(
            "speech_translate",
            tr(f"語音翻譯：{len(rows)} 種組合", f"Speech translation: {len(rows)} setups"),
            sc.PAGE,
            state={
                "audio": str(sc.cached_audio(audio, filename)),
                "audio_name": audio_name,
                "source": source,
                "clip_s": clip,
                "live": live,
                "chunk_s": chunk,
                "reference": reference.strip(),
                "rows": [r | {"status": "waiting", "text": ""} for r in rows],
            },
        )
        base = ollama_url()
        jobs.start(job, lambda j: translate_work(j, url, base))
        st.session_state.st_job = job.id
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
    rows = [r | {"status": "done"} for r in data.get("results") or [] if isinstance(r, dict)]
    live = bool(data.get("live"))
    bits = [tr("錄音：", "Audio: ") + f"`{data.get('audio', '?')}`"]
    if data.get("clip_s"):
        bits.append(tr(f"前 {data['clip_s']} 秒", f"first {data['clip_s']} s"))
    elif data.get("duration"):
        bits.append(tr(f"{data['duration']:.0f} 秒", f"{data['duration']:.0f} s"))
    if live:
        bits.append(tr(f"模擬即時輸入，每段 {data.get('chunk_s')} 秒",
                       f"simulated live input, {data.get('chunk_s')} s chunks"))  # fmt: skip
    if data.get("stopped"):
        bits.append(tr("中途停止", "stopped early"))
    st.caption(" · ".join(bits))
    summary(rows, names, live, data.get("reference") or "")
    per_row = 3 if len(rows) > 2 else max(len(rows), 1)
    for start in range(0, len(rows), per_row):
        cols = st.columns(per_row, gap="small")
        for col, r in zip(cols, rows[start : start + per_row], strict=False):
            with col:
                card(r, names, live)
    notes(live)
    if data.get("reference"):
        with st.expander(tr("參考譯文", "Reference translation")):
            st.write(data["reference"])


def history_summary(data: dict) -> dict:
    # Cached across languages (history.summarize): ids only, no tr().
    rows = [r for r in data.get("results") or [] if isinstance(r, dict)]
    scored = [(row_name(r, {}), r["score"]["chrf"]) for r in rows if r.get("score")]
    title = str(data.get("audio") or "?") + " · zh → en"
    if data.get("live"):
        title += f" · live {data.get('chunk_s')}s"
    models = []
    for r in rows:
        for m in (r.get("asr"), r.get("llm"), r.get("model")):
            if m and m not in models:
                models.append(m)
    return {
        "title": title,
        "models": models,
        "seconds": round(sum(r.get("total_s") or 0 for r in rows), 2) or None,
        "snippet": next((r.get("text", "")[:220] for r in rows if r.get("text")), ""),
        "best": (max(scored, key=lambda x: x[1]), "chrF") if scored else None,
        "error": bool(rows) and all(r.get("error") for r in rows),
    }
