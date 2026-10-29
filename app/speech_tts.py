"""Text-to-speech mode of the speech-model page: compare voices for the voice assistant.

The voices: macOS's built-in ones and the local neural models (local_llm.neural_tts: MediaTek
VITS, Kokoro, Qwen3-TTS, VoxCPM2, Breeze-TTS-2), each labelled with where it
comes from. The same text, by default the fictional production meeting, is spoken by 1-4
voices one sentence at a time, as the cascade assistant speaks (tts.timed_sentences): each
voice gets its loading time, the first sentence's delay (how long the assistant would stay
silent), the real-time factor, a verdict on whether it can talk live, and an intelligibility
check: the "round trip" transcribes the audio with an installed speech model and scores it
against the input text (CER for Chinese, WER for English, asr_metrics.score). "Benchmark"
shows the measured table (data/lab/tts_benchmark.json, scripts/speech_eval/run_tts_bench.py).
"""

import json
import re
from pathlib import Path

import altair as alt
import jobs
import pandas as pd
import speech_common as sc
import streamlit as st
from common import PROJECT_DIR, cloud_hint, kept, local_or_allowed, save_run
from i18n import tr

from local_llm import asr_metrics, gateway_client, tts

MAX_VOICES = 4
RATES = [0, 140, 160, 180, 200, 220, 250]  # words per minute; 0 = the voice's default
DEFAULT_ASR = "qwen3-asr-1.7b"
MEETING = PROJECT_DIR / "data" / "lab" / "samples" / "meeting_zh.txt"
BENCHMARK = PROJECT_DIR / "data" / "lab" / "tts_benchmark.json"
# Usable live: the first sentence renders within LIVE_FIRST_S and synthesis keeps well ahead of
# playback (RTF below LIVE_RTF), so later sentences are ready before the previous one ends.
LIVE_FIRST_S = 1.0
LIVE_RTF = 0.5


def meeting_text() -> str:
    """The fictional production meeting's lines (meeting_zh.txt without the voice tags)."""
    try:
        text = MEETING.read_text(encoding="utf-8")
    except OSError:
        return ""
    return "\n".join(re.sub(r"^\[[^\]]+\]\s*", "", ln) for ln in text.splitlines()
                     if ln.startswith("["))  # fmt: skip


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
    "zh": [
        ("Kokoro 曉北", "zh_CN"),
        ("Qwen3-TTS 台灣腔", "zh_TW"),
        ("MediaTek VITS", "zh_TW"),
        ("Meijia", "zh_TW"),
    ],  # fmt: skip
    "en": [("Samantha", "en_US"), ("Daniel", "en_GB"), ("Flo", "en_US")],
}
SAVED = ("voice", "short", "locale", "source", "limits", "file", "seconds", "duration", "rtf",
         "load_s", "first_s", "max_s", "sentences", "transcript", "asr_s", "score",
         "error")  # fmt: skip
LOCALES = {
    "zh_TW": ("台灣華語", "Taiwan Mandarin"),
    "zh_CN": ("中國大陸華語", "Mainland Mandarin"),
    "zh_HK": ("粵語（香港）", "Cantonese (Hong Kong)"),
    "en_US": ("美式英語", "US English"),
    "en_GB": ("英式英語", "UK English"),
}


def locale_name(locale: str) -> str:
    return tr(*LOCALES[locale]) if locale in LOCALES else locale


def source_name(v: dict) -> str:
    src = v.get("source")
    return tr(*src) if isinstance(src, list | tuple) else src or tr(*tts.MACOS_SOURCE)


def voice_label(v: dict) -> str:
    return f"{v['short']} · {locale_name(v['locale'])} · {source_name(v)}"


def live_verdict(first_s: float | None, rtf: float | None) -> tuple[str, str]:
    """(icon colour, text): can this voice speak for the live assistant?"""
    if first_s is None or rtf is None:
        return "gray", ""
    if first_s <= LIVE_FIRST_S and rtf < LIVE_RTF:
        return "green", tr("可即時對話", "Fast enough to talk live")
    if rtf < 1:
        return "orange", tr(
            f"勉強：每句先等 {first_s:.1f} 秒", f"Borderline: {first_s:.1f} s per sentence"
        )
    return "red", tr("太慢，只適合預先合成", "Too slow: pre-render only")


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
            out = tts.timed_sentences(text, r["voice"], state["rate"] or None)
            keys = ("seconds", "duration", "rtf", "load_s", "first_s", "max_s", "sentences")
            r.update({k: out[k] for k in keys}, file=str(out["path"]))
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
            "engine": "mixed",
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
    parts = []
    if r.get("first_s") is not None:
        parts.append(tr(f"首句 {r['first_s']:.2f}s", f"first sentence {r['first_s']:.2f}s"))
    parts += [
        tr(f"合成 {r['seconds']:.2f}s", f"synthesis {r['seconds']:.2f}s"),
        tr(f"音訊 {r['duration']:.1f}s", f"audio {r['duration']:.1f}s"),
    ]
    if r.get("load_s"):
        parts.append(tr(f"準備 {r['load_s']:.1f}s", f"warm-up {r['load_s']:.1f}s"))
    if r.get("rtf"):
        parts.append(f"RTF {r['rtf']:.3f} · {1 / r['rtf']:.0f}×")
    if r.get("score"):
        parts.append(f"{r['score'].get('metric', 'CER')} {sc.pct(r['score']['cer'])}")
    return " · ".join(parts)


def card(r: dict, text: str) -> None:
    with st.container(border=True):
        st.markdown(f"**{r['short']}**")
        st.caption(f"{locale_name(r['locale'])} · {source_name(r)}")
        if r.get("limits") and r["limits"][0]:
            st.caption(f":material/info: {tr(*r['limits'])}")
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
            colour, verdict = live_verdict(r.get("first_s"), r.get("rtf"))
            if verdict:
                st.badge(verdict, color=colour)
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
                tr("首句（秒）", "First sentence (s)"): r.get("first_s"),
                tr("合成（秒）", "Synthesis (s)"): r["seconds"],
                tr("音訊（秒）", "Audio (s)"): r["duration"],
                "RTF": r.get("rtf"),
                metric: (r.get("score") or {}).get("cer"),
                tr("即時對話", "Live use"): live_verdict(r.get("first_s"), r.get("rtf"))[1],
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
        "像助理一樣一句一句合成（先用一句短句把模型載入、預熱，不計入）。「首句」是第一句合成完的時間＝"
        f"助理要沉默多久才開口；首句 ≤ {LIVE_FIRST_S:.0f} 秒且 RTF < {LIVE_RTF} 才算「可即時對話」"
        "（後面的句子能在前一句播完前準備好）。全部在本機執行。",
        "Rendered one sentence at a time, as the assistant speaks (a short line loads and warms "
        "the model first, not counted). First sentence = how long the assistant stays silent "
        f"before it talks; ≤ {LIVE_FIRST_S:.0f} s with RTF < {LIVE_RTF} counts as fast enough "
        "to talk live (each next sentence is ready before the previous one ends). All local.",
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


def voice_sources() -> None:
    st.caption(
        tr(
            "聲音來源：**macOS 內建**（免下載）；**MediaTek VITS**（聯發科，台灣口音，sherpa-onnx"
            "）；"
            "**Kokoro-82M**、**Qwen3-TTS 0.6B**、**VoxCPM2**、**Breeze-TTS-2**（mlx-audio 在 Apple "
            "晶片上執行）。標「台灣腔（複製）」的聲音是用一段台灣講者的錄音（BreezyVoice 的範例講"
            "者）"
            "複製出來的口音。每個聲音能不能念英文、數字等限制寫在結果卡上。Qwen3-TTS、VoxCPM2 念"
            "繁體字"
            "會走音，送進模型前先轉成簡體（只影響念法）。BreezyVoice 太慢（約語音長度 12 倍）已移"
            "除。",
            "Voice sources: **macOS built-in** (nothing to download); **MediaTek VITS** (Taiwanese "
            "accent, sherpa-onnx); **Kokoro-82M**, **Qwen3-TTS 0.6B**, **VoxCPM2** and "
            "**Breeze-TTS-2** (mlx-audio on Apple silicon). Voices marked 台灣腔（複製） clone a "
            "Taiwanese speaker (BreezyVoice's example speaker). Each result card says what the "
            "voice cannot do (English, numbers, ...). Qwen3-TTS and VoxCPM2 mispronounce "
            "Traditional characters, so they read the text converted to Simplified. BreezyVoice "
            "was removed: about 12x slower than real time.",
        )
    )


# -- page -----------------------------------------------------------------------------------


def _use_example(lang: str) -> None:
    st.session_state.tts_text = meeting_text() if lang == "meeting" else EXAMPLES[lang]
    st.session_state.tts_voices = []  # picked again for the example's language


# -- quick listen -----------------------------------------------------------------------------

LISTEN = {
    "zh": "您好，台北明天晴時多雲，氣溫 21 到 27 度，降雨機率 10%。CNC 產線的報告我下午寄給你。",
    "en": "Good afternoon. Tomorrow in Taipei will be partly cloudy, 21 to 27 degrees, with a "
    "10 percent chance of rain.",
}
LISTEN_EN = ("Samantha",)  # macOS English voice next to the neural voices that speak English


def listen_voices(language: str) -> list[dict]:
    """The voices of the benchmark (fastest first), or those of them that speak English."""
    every = tts.voices(neural="all")
    rows = (load_benchmark() or {}).get("rows", [])
    found = [tts.find(r["voice"], every) for r in rows]
    voices = [v for v in found if v]
    if language == "en":
        macos = [tts.find(n, every) for n in LISTEN_EN]
        voices = [v for v in macos if v] + [
            v for v in voices if v.get("english") or v.get("engine") is None
        ]  # fmt: skip
    return list({v["name"]: v for v in voices if local_or_allowed(v)}.values())


def listen_work(job: jobs.Job) -> None:
    todo = job.state["todo"]
    for n, name in enumerate(todo, start=1):
        if job.cancel.is_set():
            break
        job.detail, job.progress = f"{n}/{len(todo)} {name}", (n - 1) / len(todo)
        try:
            tts.synthesize(job.state["text"], name)
        except Exception as exc:
            job.state.setdefault("errors", {})[name] = str(exc)[:200]
    job.progress = 1.0


@st.fragment(run_every=2)
def _listen_progress(job_id: str) -> None:
    job = jobs.get(job_id)
    if job is None:
        return
    if job.running:
        st.progress(job.progress or 0.0, text=job.detail)
    else:
        st.rerun()


def listen_view() -> None:
    st.caption(
        tr(
            "每個評測過的聲音念同一句話，直接按播放比較。已合成過的會留在快取（storage/speech_cache），"
            "下次打開不用再等；改了句子就要重新產生。句子刻意放了數字和英文縮寫（CNC），"
            "可以聽出哪些聲音會跳過或念錯。",
            "Every evaluated voice reads the same line; press play to compare. Renderings stay in "
            "the cache (storage/speech_cache), so they open at once next time; a new line needs "
            "new renderings. The line has numbers and an abbreviation (CNC) on purpose, to hear "
            "which voices skip or misread them.",
        )
    )
    language = st.segmented_control(
        tr("語言", "Language"), ["zh", "en"], required=True,
        format_func=lambda x: {"zh": tr("中文", "Chinese"), "en": tr("英文", "English")}[x],
        **kept("tts_listen_lang"),
    ) or "zh"  # fmt: skip
    st.session_state.setdefault(f"tts_listen_{language}", LISTEN[language])
    text = st.text_input(tr("句子", "Line"), **kept(f"tts_listen_{language}")).strip()
    voices = listen_voices(language)
    cloud_hint("聲音（Azure、Google、Gemini）", "voices (Azure, Google, Gemini)")
    if not text or not voices:
        return
    ready = {v["name"]: tts.cached(text, v["name"]) for v in voices}
    missing = [n for n, p in ready.items() if p is None]
    job = jobs.get(st.session_state.get("tts_listen_job"))
    if job and job.running:
        _listen_progress(job.id)
    elif missing:
        slow = tr("雲端與較慢的模型要等一下", "cloud and slower models take a while")
        if st.button(
            tr(f"產生試聽（{len(missing)} 個聲音還沒合成，{slow}）",
               f"Make the samples ({len(missing)} voices not rendered yet; {slow})"),
            icon=":material/graphic_eq:", type="primary",
        ):  # fmt: skip
            job = jobs.Job("speech_tts", tr("快速試聽", "Quick listen"), sc.PAGE,
                           state={"text": text, "todo": missing})  # fmt: skip
            jobs.start(job, listen_work)
            st.session_state.tts_listen_job = job.id
            st.rerun()
    errors = (job.state.get("errors") if job else None) or {}
    cols = st.columns(2)
    for i, v in enumerate(voices):
        with cols[i % 2].container(border=True):
            st.markdown(f"**{v['short']}**　·　{source_name(v)}")
            if ready[v["name"]]:
                st.audio(str(ready[v["name"]]))
            elif v["name"] in errors:
                st.caption(tr("合成失敗：", "Failed: ") + errors[v["name"]])
            else:
                st.caption(tr("尚未產生", "Not rendered yet"))
            if limits := tts.limits(v):
                st.caption(tr(*limits))


def tts_page(url: str, status: dict) -> None:
    st.session_state.setdefault("tts_view", "try")
    view = (
        st.segmented_control(
            tr("檢視", "View"),
            ["try", "listen", "bench"],
            format_func=lambda v: {
                "try": f":material/play_arrow: {tr('自己試', 'Try it')}",
                "listen": f":material/headphones: {tr('快速試聽', 'Quick listen')}",
                "bench": f":material/leaderboard: {tr('評測總表', 'Benchmark')}",
            }[v],
            required=True,
            **kept("tts_view"),
        )
        or "try"
    )
    if view == "bench":
        benchmark_view()
        return
    if view == "listen":
        listen_view()
        return
    if not tts.available():
        st.warning(
            tr("找不到 macOS 的 `say` 指令，這個模式只能在 Mac 上使用。",
               "macOS's `say` command was not found; this mode only works on a Mac."),
            icon=":material/warning:",
        )  # fmt: skip
        return
    labels = sc.names()
    a, b, c0 = st.columns(3)
    a.button(tr("產銷會議（虛構）", "Production meeting"), icon=":material/groups:",
             on_click=_use_example, args=("meeting",), width="stretch")  # fmt: skip
    b.button(tr("中文短句", "Short Chinese"), icon=":material/edit_note:",
             on_click=_use_example, args=("zh",), width="stretch")  # fmt: skip
    c0.button(tr("英文範例", "English example"), icon=":material/edit_note:",
              on_click=_use_example, args=("en",), width="stretch")  # fmt: skip
    st.session_state.setdefault("tts_text", meeting_text() or EXAMPLES["zh"])
    text = st.text_area(
        tr("要念出來的文字（一行一句）", "Text to speak (one sentence per line)"),
        height=180,
        help=tr(
            "預設是測試用的虛構產銷會議（9 句、約 88 秒），原稿在 data/lab/samples/meeting_zh.txt。"
            "較慢的聲音（VoxCPM2、Breeze-TTS-2）整份要 1–2 分鐘。",
            "Defaults to the fictional production meeting used in the tests (9 lines, about "
            "88 s), data/lab/samples/meeting_zh.txt. Slow voices (VoxCPM2, Breeze-TTS-2) take a m"
            "inute or two.",
        ),
        **kept("tts_text"),
    )
    language = text_language(text)

    voices = [v for v in tts.voices(neural="all") if local_or_allowed(v)]
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
            "每個聲音後面標出來源：macOS 內建，或本機的神經網路模型。macOS 只列台灣華語、中國大陸"
            "華語、粵語與英語，"
            "已排除特效聲音。",
            "Each voice says where it comes from: macOS built-in, or a local neural model. macOS "
            "voices for Taiwan and Mainland Mandarin, Cantonese and English, novelty voices "
            "left out.",
        ),
        **kept("tts_voices"),
    )
    cloud_hint("聲音（Azure、Google、Gemini）", "voices (Azure, Google, Gemini)")
    c, d = st.columns(2)
    st.session_state.setdefault("tts_rate", 0)
    rate = c.select_slider(
        tr("語速（每分鐘字數）", "Rate (words per minute)"),
        RATES,
        format_func=lambda r: str(r) if r else tr("預設", "default"),
        help=tr("只對 macOS 內建聲音有效。", "macOS voices only."),
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
    voice_sources()

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
                        "source": list(tts.source(by_name[n])),
                        "limits": list(tts.limits(by_name[n])),
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
    bits = [tr("逐句合成", "one sentence at a time")] if data.get("engine") == "mixed" else [
        tr("引擎：macOS `say`", "Engine: macOS `say`")]  # fmt: skip
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
        "title": f"TTS · {voices}",
        "models": [m for m in [data.get("asr")] if m],
        "seconds": round(sum(r.get("seconds") or 0 for r in rows), 2) or None,
        "snippet": (data.get("text") or "")[:220],
        "best": (min(scored, key=lambda x: x[1]), metric) if scored else None,
        "error": bool(rows) and all(r.get("error") for r in rows),
    }


# -- benchmark --------------------------------------------------------------------------------


def load_benchmark() -> dict | None:
    try:
        return json.loads(BENCHMARK.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def benchmark_frame(data: dict) -> pd.DataFrame:
    rows = []
    for r in data["rows"]:
        # a voice that streams starts after its first piece, not the whole sentence
        colour, verdict = live_verdict(r.get("stream_first_s") or r.get("first_s"), r.get("rtf"))
        rows.append({
            "voice": r["label"],
            "source": tr(*r["source"]),
            "first_s": r.get("first_s"),
            "stream_first_s": r.get("stream_first_s"),
            "rtf": r.get("rtf"),
            "load_s": r.get("load_s"),
            "cer_offline": r.get("cer_offline"),
            "cer_stream": r.get("cer_stream"),
            "rms_db": r.get("rms_db"),
            "size_gb": r.get("size_gb"),
            "licence": tr(*r["licence"]) if isinstance(r["licence"], list) else r["licence"],
            "verdict": verdict,
            "limits": tr(*r["limits"]) if r.get("limits") else "",
            "colour": colour,
        })  # fmt: skip
    return pd.DataFrame(rows)


def _bars(df: pd.DataFrame, field: str, title: str, *, sort: str, rule: float | None = None,
          percent: bool = False) -> alt.LayerChart | alt.Chart:  # fmt: skip
    """One horizontal bar per voice, coloured by the live-use verdict (names not cut short)."""
    colours = alt.Scale(domain=["green", "orange", "red"], range=["#16A34A", "#F59E0B", "#DC2626"])
    axis = alt.Axis(format="%") if percent else alt.Axis()
    bars = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            y=alt.Y(
                "voice:N",
                sort=list(df.sort_values(sort)["voice"]),
                title=None,
                axis=alt.Axis(labelLimit=220),
            ),  # fmt: skip
            x=alt.X(f"{field}:Q", title=title, axis=axis),
            color=alt.Color("colour:N", scale=colours, legend=None),
            tooltip=[
                "voice",
                "source",
                "first_s",
                "rtf",
                alt.Tooltip("cer_offline:Q", format=".1%"),
                alt.Tooltip("cer_stream:Q", format=".1%"),
            ],  # fmt: skip
        )
    )
    if rule is None:
        return bars
    line = alt.Chart(pd.DataFrame({"x": [rule]})).mark_rule(
        color="#64748B", strokeDash=[4, 3]).encode(x="x:Q")  # fmt: skip
    return bars + line


def benchmark_charts(df: pd.DataFrame) -> None:
    a, b = st.columns(2)
    a.markdown(tr("**首句延遲（越短越快開口）**", "**First sentence (lower = talks sooner)**"))
    a.altair_chart(_bars(df, "first_s", tr("秒", "s"), sort="first_s", rule=LIVE_FIRST_S),
                   width="stretch")  # fmt: skip
    b.markdown(
        tr("**即時倍率 RTF（合成時間 ÷ 語音長度）**", "**Real-time factor (synthesis ÷ audio)**")
    )
    b.altair_chart(_bars(df, "rtf", "RTF", sort="first_s", rule=LIVE_RTF), width="stretch")
    st.caption(
        tr(
            f"顏色＝能不能即時對話：綠色可以（首句 ≤ {LIVE_FIRST_S:.0f} 秒且 RTF < {LIVE_RTF}，虛"
            "線）；"
            "橘色勉強；紅色太慢，只適合預先合成。",
            f"Colour = live use: green yes (first sentence ≤ {LIVE_FIRST_S:.0f} s and RTF < "
            f"{LIVE_RTF}, dashed lines); orange borderline; red pre-render only.",
        )
    )
    c, d = st.columns(2)
    c.markdown(
        tr("**會議稿回聽錯字率（越低越清楚）**", "**Meeting round-trip errors (lower = clearer)**")
    )
    c.altair_chart(_bars(df, "cer_offline", "CER", sort="cer_offline", percent=True),
                   width="stretch")  # fmt: skip
    d.markdown(tr("**測試腳本串流辨識錯字率（能不能當測試講者）**",
                  "**Test script through streaming ASR (a good test speaker?)**"))  # fmt: skip
    d.altair_chart(_bars(df, "cer_stream", "CER", sort="cer_stream", percent=True),
                   width="stretch")  # fmt: skip


def benchmark_view() -> None:
    data = load_benchmark()
    if not data or not data.get("rows"):
        st.info(tr("找不到評測結果（data/lab/tts_benchmark.json）。",
                   "No benchmark results found (data/lab/tts_benchmark.json)."))  # fmt: skip
        return
    n, date = len(data["rows"]), data.get("date", "")
    st.markdown(
        tr(
            f"**{n} 個聲音念同一份虛構產銷會議（20 句、約 88 秒）**，像串接式助理一樣一句一句合成"
            f"（{date}，這台 Mac）。看三件事：多久開口、跟不跟得上、聽不聽得清楚。",
            f"**{n} voices read the same fictional production meeting (20 sentences, about 88 s)** "
            f"one sentence at a time, as the cascade assistant speaks ({date}, this Mac): how "
            "soon it talks, whether it keeps up, whether it is clear.",
        )
    )
    df = benchmark_frame(data)
    benchmark_charts(df)
    st.dataframe(
        df.drop(columns=["colour"]).rename(columns={
            "voice": tr("聲音", "Voice"), "source": tr("來源", "Source"),
            "first_s": tr("首句（秒）", "First (s)"),
            "stream_first_s": tr("邊合成邊播首段（秒）", "Streamed first piece (s)"),
            "rtf": "RTF",
            "load_s": tr("載入（秒）", "Load (s)"),
            "cer_offline": tr("會議稿 CER", "Meeting CER"),
            "cer_stream": tr("測試腳本串流 CER", "Script streaming CER"),
            "rms_db": tr("音量 dBFS", "Level dBFS"), "size_gb": tr("大小 GB", "Size GB"),
            "licence": tr("授權", "Licence"), "verdict": tr("即時對話", "Live use"),
            "limits": tr("限制", "Limits"),
        }),
        hide_index=True,
        width="stretch",
        column_config={
            tr("會議稿 CER", "Meeting CER"): st.column_config.NumberColumn(format="percent"),
            tr("測試腳本串流 CER", "Script streaming CER"): st.column_config.NumberColumn(
                format="percent"),
        },
    )  # fmt: skip
    if data.get("notes"):
        st.markdown(tr(*data["notes"]))
    st.caption(
        tr(
            "方法：每個聲音先念一句短句載入模型（「載入」），再逐句合成；「首句」是第一句的合成時間，"
            "也就是助理開口前要等多久；能串流的聲音（Qwen3-TTS、Azure）邊合成邊播，"
            "實際只要等「邊合成邊播首段」。會議稿 CER：Qwen3-ASR-1.7B 聽合成結果；測試腳本串流 "
            "CER："
            "即時語音助理的 16 句中文測試台詞送進預設串流辨識（Zipformer），看能不能當測試講者。"
            "腳本：scripts/speech_eval/run_tts_bench.py。",
            "Method: each voice first reads a short line to load the model (Load), then the "
            "meeting one sentence at a time; First is how long the first sentence took, the "
            "wait before the assistant talks; voices that stream (Qwen3-TTS, Azure) play while "
            "they are made, so the wait is the streamed first piece. Meeting CER: "
            "Qwen3-ASR-1.7B transcribes the result; script "
            "streaming CER: the voice assistant's 16 Chinese test lines through the default "
            "streaming ASR (Zipformer), i.e. can the voice act as the test speaker. Script: "
            "scripts/speech_eval/run_tts_bench.py.",
        )
    )
