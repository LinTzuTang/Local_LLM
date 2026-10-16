"""Text-to-speech mode of the speech-model page: compare voices of macOS's built-in `say`.

No TTS model is installed, so the mode uses the voices that come with macOS (local and free,
local_llm.tts). The same text is spoken by 1-4 voices; each gets its synthesis time and
real-time factor, and an intelligibility check: the "round trip" transcribes the generated
audio with an installed speech model and scores it against the input text (CER for Chinese,
WER for English, asr_metrics.score). Neural TTS models (Kokoro, Qwen3-TTS, CosyVoice 3,
BreezyVoice) are listed in the model library but not downloaded.
"""

from pathlib import Path

import jobs
import pandas as pd
import speech_common as sc
import streamlit as st
from common import PROJECT_DIR, kept, save_run
from i18n import pick, tr

from local_llm import asr_metrics, catalog, gateway_client, tts

MAX_VOICES = 4
RATES = [0, 140, 160, 180, 200, 220, 250]  # words per minute; 0 = the voice's default
DEFAULT_ASR = "qwen3-asr-1.7b"
EXAMPLES = {
    "zh": (
        "各位同仁早安。今天的會議有三個重點：第一，第三季營收成長百分之十二；第二，新產線預計"
        "十一月試產；第三，請大家在星期五以前更新專案進度。謝謝。"
    ),
    "en": (
        "Good morning, everyone. We have three points today. First, revenue grew twelve "
        "percent in the third quarter. Second, the new production line starts trial runs in "
        "November. Third, please update your project status by Friday. Thank you."
    ),
}
# (short name, locale): voices that exist in several languages are named in the system's
# language ("Flo (中文（台灣）)"), so defaults match on the short name and locale.
DEFAULT_VOICES = {
    "zh": [("Meijia", "zh_TW"), ("Tingting", "zh_CN"), ("Flo", "zh_TW")],
    "en": [("Samantha", "en_US"), ("Daniel", "en_GB"), ("Flo", "en_US")],
}
SAVED = ("voice", "short", "locale", "file", "seconds", "duration", "rtf", "transcript",
         "asr_s", "score", "error")  # fmt: skip
LOCALES = {
    "zh_TW": ("台灣華語", "Taiwan Mandarin"),
    "zh_CN": ("中國大陸華語", "Mainland Mandarin"),
    "zh_HK": ("粵語（香港）", "Cantonese (Hong Kong)"),
    "en_US": ("美式英語", "US English"),
    "en_GB": ("英式英語", "UK English"),
}


def locale_name(locale: str) -> str:
    return tr(*LOCALES[locale]) if locale in LOCALES else locale


def voice_label(v: dict) -> str:
    return f"{v['short']} · {locale_name(v['locale'])}"


def text_language(text: str) -> str:
    return "en" if asr_metrics.is_english(text) else "zh"


# -- worker (no Streamlit calls) ------------------------------------------------------------


def _compute(out: dict) -> float:
    load = (out.get("load_seconds") or 0) + (out.get("ollama_load_seconds") or 0)
    return max(out["seconds"] - load - (out.get("queue_seconds") or 0), 0.01)


def tts_work(job: jobs.Job, url: str) -> None:
    state = job.state
    rows, text, language = state["rows"], state["text"], state["language"]
    for n, r in enumerate(rows, start=1):
        if job.cancel.is_set():
            break
        r["status"] = "running"
        job.detail, job.progress = f"{n}/{len(rows)} {r['short']}", (n - 1) / len(rows)
        try:
            out = tts.timed_synthesize(text, r["voice"], state["rate"] or None)
            r.update(file=str(out["path"]), seconds=out["seconds"], duration=out["duration"],
                     rtf=out["rtf"])  # fmt: skip
            if state["asr"]:
                job.detail = f"{n}/{len(rows)} {r['short']} · {state['asr']}"
                heard = gateway_client.transcribe(out["path"], state["asr"], language, base_url=url)
                r["transcript"], r["asr_s"] = heard["text"], round(_compute(heard), 2)
                r["score"] = asr_metrics.score(text, heard["text"], language)
        except Exception as exc:
            r["error"] = str(exc)
        r["status"] = "done"
    for r in rows:
        r["status"] = "done"
    job.progress = 1.0
    path = save_run(
        "asr_compare",
        {
            "mode": "tts",
            "engine": "macos-say",
            "text": text,
            "language": language,
            "rate": state["rate"] or None,
            "asr": state["asr"] or None,
            "stopped": job.cancel.is_set(),
            "results": [
                {k: r.get(k) for k in SAVED}
                | {"file": _relative(r.get("file")) if r.get("file") else None}
                for r in rows
            ],
        },
    )
    state["path"] = path.relative_to(path.parents[1])


def _relative(path: str) -> str:
    try:
        return str(Path(path).relative_to(PROJECT_DIR))
    except ValueError:
        return path


# -- drawing --------------------------------------------------------------------------------


def facts(r: dict) -> str:
    parts = [
        tr(f"合成 {r['seconds']:.2f}s", f"synthesis {r['seconds']:.2f}s"),
        tr(f"音訊 {r['duration']:.1f}s", f"audio {r['duration']:.1f}s"),
    ]
    if r.get("rtf"):
        parts.append(f"RTF {r['rtf']:.3f} · {1 / r['rtf']:.0f}×")
    if r.get("score"):
        parts.append(f"{r['score'].get('metric', 'CER')} {sc.pct(r['score']['cer'])}")
    return " · ".join(parts)


def card(r: dict, text: str) -> None:
    with st.container(border=True):
        st.markdown(f"**{r['short']}**")
        st.caption(f"{locale_name(r['locale'])} · `{r['voice']}`")
        if r["status"] == "waiting":
            st.info(tr("等待中…", "Waiting…"))
            return
        if r.get("error"):
            st.error(r["error"])
            return
        if r["status"] == "running" and not r.get("file"):
            st.info(tr("合成中…", "Synthesising…"), icon=":material/progress_activity:")
            return
        if not r.get("file"):
            st.caption(tr("沒有執行（中途停止）", "Not run (stopped)"))
            return
        path = Path(r["file"])
        path = path if path.is_absolute() else PROJECT_DIR / path
        if path.exists():
            st.audio(str(path))
        else:
            st.caption(tr("音檔已清除。", "The audio file has been removed."))
        if r.get("seconds") is not None:
            st.caption(facts(r))
        if r.get("transcript") is not None:
            st.markdown(tr("**語音辨識聽到的**", "**What speech recognition heard**"))
            st.write(sc.show(r["transcript"]) or tr("（沒有辨識出文字）", "(no text recognised)"))
            if text and r.get("score"):
                import speech_compare

                with st.expander(tr("對照輸入文字", "Against the input text")):
                    st.markdown(speech_compare.diff_html(text, r["transcript"]),
                                unsafe_allow_html=True)  # fmt: skip
        elif r["status"] == "running":
            st.info(tr("回聽辨識中…", "Round-trip check…"), icon=":material/progress_activity:")


def summary(rows: list[dict], asr: str | None) -> None:
    done = [r for r in rows if r.get("seconds") is not None and not r.get("error")]
    if not done:
        return
    st.subheader(tr("比較結果", "Results"))
    voice = tr("聲音", "Voice")
    metric = next((r["score"].get("metric") for r in done if r.get("score")), "CER")
    table = pd.DataFrame(
        [
            {
                voice: voice_label(r),
                tr("合成（秒）", "Synthesis (s)"): r["seconds"],
                tr("音訊（秒）", "Audio (s)"): r["duration"],
                "RTF": r.get("rtf"),
                metric: (r.get("score") or {}).get("cer"),
            }
            for r in done
        ]
    )
    if not asr:
        table = table.drop(columns=[metric])
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            metric: st.column_config.ProgressColumn(
                help=tr(
                    "回聽辨識的錯誤率：語音辨識模型聽合成語音，和輸入文字比對；越低越清楚。",
                    "Round-trip error rate: a speech model transcribes the synthetic audio and "
                    "it is compared with the input text; lower = more intelligible.",
                ),
                min_value=0.0,
                max_value=max(0.3, float(table[metric].max() or 0)) if asr else 1.0,
                format="percent",
            ),
            "RTF": st.column_config.NumberColumn(
                help=tr(
                    "合成時間 ÷ 音訊長度（越低越快）",
                    "Synthesis time ÷ audio length (lower is faster)",
                ),
                format="%.3f",
            ),  # fmt: skip
        },
    )
    asr = asr if asr and table[metric].fillna(0).max() > 0 else None  # all 0%: no bar to draw
    charts = st.columns(2 if asr else 1)
    charts[0].caption(tr("即時倍率 RTF（越低越快）", "Real-time factor (lower is faster)"))
    charts[0].bar_chart(table, x=voice, y="RTF", horizontal=True, sort="RTF")
    if asr:
        charts[1].caption(
            tr(f"回聽 {metric}（越低越清楚）", f"Round-trip {metric} (lower = clearer)")
        )
        charts[1].bar_chart(table, x=voice, y=metric, horizontal=True, sort=metric)


def notes(asr: str | None, labels: dict) -> None:
    text = tr(
        "引擎：macOS 內建 `say`（本機、免費）。合成時間是 `say` 輸出 WAV 的實際時間。",
        "Engine: macOS's built-in `say` (local and free). Synthesis time is how long `say` "
        "took to write the WAV.",
    )
    if asr:
        text += tr(
            f"回聽檢查用 {labels.get(asr, asr)} 辨識合成語音，數字與標點已統一；它量的是"
            "「機器聽不聽得懂」，不等於自然度或好不好聽。",
            f" The round trip transcribes the audio with {labels.get(asr, asr)} (numbers and "
            "punctuation unified); it measures whether a machine understands the speech, not "
            "how natural it sounds.",
        )
    st.caption(text)


def cards(rows: list[dict], text: str) -> None:
    cols = st.columns(max(len(rows), 1), gap="small")
    for col, r in zip(cols, rows, strict=False):
        with col:
            card(r, text)


def render(job: jobs.Job) -> None:
    state, labels = job.state, sc.names()
    cards(state["rows"], state["text"])
    if job.running:
        sc.running_bar(job)
        return
    if job.error:
        st.error(job.error)
    summary(state["rows"], state["asr"])
    notes(state["asr"], labels)
    sc.saved_caption(state)


@st.fragment(run_every=1)
def live_view(job: jobs.Job) -> None:
    if not job.running:
        st.rerun()
    render(job)


def better_models() -> None:
    names = [pick({"zh": m["name"], "en": m.get("name_en") or m["name"]})
             for m in catalog.other_models("tts")]  # fmt: skip
    names = names or ["Kokoro-82M", "Qwen3-TTS", "CosyVoice 3", "BreezyVoice"]
    st.info(
        tr(
            "**更自然的語音合成模型**已列在模型庫，**尚未下載**："
            + "、".join(names)
            + "（BreezyVoice 是聯發科的台灣華語 TTS）。雲端選項（只列出、不連線）：Azure OpenAI "
            "`gpt-4o-mini-tts`。",
            "**More natural TTS models** are listed in the model library, **not downloaded**: "
            + ", ".join(names)
            + " (BreezyVoice is MediaTek's Taiwan Mandarin TTS). Cloud option (listed only, not "
            "connected): Azure OpenAI `gpt-4o-mini-tts`.",
        ),
        icon=":material/record_voice_over:",
    )
    st.page_link(
        "views/models.py",
        label=tr("到模型庫看語音合成模型", "See TTS models in the model library"),
        icon=":material/library_books:",
    )


# -- page -----------------------------------------------------------------------------------


def _use_example(lang: str) -> None:
    st.session_state.tts_text = EXAMPLES[lang]
    st.session_state.tts_voices = []  # picked again for the example's language


def tts_page(url: str, status: dict) -> None:
    if not tts.available():
        st.warning(
            tr("找不到 macOS 的 `say` 指令，這個模式只能在 Mac 上使用。",
               "macOS's `say` command was not found; this mode only works on a Mac."),
            icon=":material/warning:",
        )  # fmt: skip
        better_models()
        return
    labels = sc.names()
    a, b = st.columns(2)
    a.button(tr("中文範例", "Chinese example"), icon=":material/edit_note:",
             on_click=_use_example, args=("zh",), width="stretch")  # fmt: skip
    b.button(tr("英文範例", "English example"), icon=":material/edit_note:",
             on_click=_use_example, args=("en",), width="stretch")  # fmt: skip
    st.session_state.setdefault("tts_text", EXAMPLES["zh"])
    text = st.text_area(tr("要念出來的文字", "Text to speak"), height=110, **kept("tts_text"))
    language = text_language(text)

    voices = tts.voices()
    by_name = {v["name"]: v for v in voices}
    if not st.session_state.get("tts_voices"):
        wanted = DEFAULT_VOICES[language]
        st.session_state.tts_voices = [
            v["name"] for short, loc in wanted for v in voices
            if v["short"] == short and v["locale"] == loc
        ]  # fmt: skip
    st.session_state.tts_voices = [n for n in st.session_state.tts_voices if n in by_name]
    picked = st.multiselect(
        tr(f"聲音（1–{MAX_VOICES} 個）", f"Voices (1–{MAX_VOICES})"),
        list(by_name),
        format_func=lambda n: voice_label(by_name[n]),
        max_selections=MAX_VOICES,
        help=tr(
            "只列出台灣華語、中國大陸華語、粵語與英語的內建聲音，已排除特效聲音。"
            "到「系統設定 > 輔助使用 > 朗讀內容」可下載更好的「高品質」聲音。",
            "Built-in voices for Taiwan and Mainland Mandarin, Cantonese and English, novelty "
            "voices left out. Better 'Enhanced' / 'Premium' voices can be downloaded in System "
            "Settings > Accessibility > Spoken Content.",
        ),
        **kept("tts_voices"),
    )
    c, d = st.columns(2)
    st.session_state.setdefault("tts_rate", 0)
    rate = c.select_slider(
        tr("語速（每分鐘字數）", "Rate (words per minute)"),
        RATES,
        format_func=lambda r: str(r) if r else tr("預設", "default"),
        **kept("tts_rate"),
    )
    asr_options = [m for m in labels if status.get(m, {}).get("installed")]
    st.session_state.setdefault("tts_roundtrip", True)
    roundtrip = d.toggle(
        tr("回聽檢查（用語音辨識聽合成結果）", "Round-trip check (transcribe the result)"),
        **kept("tts_roundtrip"),
    )
    asr = None
    if roundtrip and asr_options:
        st.session_state.setdefault(
            "tts_asr", DEFAULT_ASR if DEFAULT_ASR in asr_options else asr_options[0]
        )
        if st.session_state.tts_asr not in asr_options:
            st.session_state.tts_asr = asr_options[0]
        asr = d.selectbox(
            tr("辨識模型", "Speech model"), asr_options,
            format_func=lambda m: labels.get(m, m), **kept("tts_asr"),
        )  # fmt: skip
    better_models()

    job = jobs.get(st.session_state.get("tts_job"))
    busy = bool(job and job.running)
    if st.button(
        tr(f"合成並比較（{len(picked)} 個聲音）", f"Synthesise and compare ({len(picked)} voices)"),
        type="primary", icon=":material/play_arrow:",
        disabled=busy or not picked or not text.strip(),
    ):  # fmt: skip
        job = jobs.Job(
            "speech_tts",
            tr(f"語音合成：{len(picked)} 個聲音", f"Text to speech: {len(picked)} voices"),
            sc.PAGE,
            state={
                "text": text.strip(),
                "language": language,
                "rate": rate,
                "asr": asr,
                "rows": [
                    {
                        "voice": n,
                        "short": by_name[n]["short"],
                        "locale": by_name[n]["locale"],
                        "status": "waiting",
                    }
                    for n in picked
                ],  # fmt: skip
            },
        )
        jobs.start(job, lambda j: tts_work(j, url))
        st.session_state.tts_job = job.id
        st.rerun()
    if job and job.running:
        live_view(job)
    elif job:
        render(job)


# -- history --------------------------------------------------------------------------------


def history_view(data: dict) -> None:
    rows = [r | {"status": "done"} for r in data.get("results") or [] if isinstance(r, dict)]
    bits = [tr("引擎：macOS `say`", "Engine: macOS `say`")]
    if data.get("rate"):
        bits.append(tr(f"語速 {data['rate']}", f"rate {data['rate']} wpm"))
    if data.get("asr"):
        bits.append(tr("回聽：", "Round trip: ") + f"`{data['asr']}`")
    if data.get("stopped"):
        bits.append(tr("中途停止", "stopped early"))
    st.caption(" · ".join(bits))
    st.markdown(tr("**輸入文字**", "**Input text**"))
    st.write(data.get("text") or "")
    summary(rows, data.get("asr"))
    cards(rows, data.get("text") or "")
    notes(data.get("asr"), sc.names())


def history_summary(data: dict) -> dict:
    # Cached across languages (history.summarize): ids only, no tr().
    rows = [r for r in data.get("results") or [] if isinstance(r, dict)]
    scored = [(r.get("short") or r["voice"], r["score"]["cer"]) for r in rows if r.get("score")]
    metric = next((r["score"].get("metric") for r in rows if r.get("score")), None) or "CER"
    voices = ", ".join(r.get("short") or r.get("voice", "?") for r in rows)
    return {
        "title": f"TTS · macOS say · {voices}",
        "models": [m for m in [data.get("asr")] if m],
        "seconds": round(sum(r.get("seconds") or 0 for r in rows), 2) or None,
        "snippet": (data.get("text") or "")[:220],
        "best": (min(scored, key=lambda x: x[1]), metric) if scored else None,
        "error": bool(rows) and all(r.get("error") for r in rows),
    }
