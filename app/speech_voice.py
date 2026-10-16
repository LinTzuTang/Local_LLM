"""Voice-chat mode of the speech-model page: a local speech -> speech assistant.

Two ways to answer a spoken question, both on this machine:

- Cascade: a speech model writes down the question (default Qwen3-ASR 0.6B, fast), a chat LLM
  in Ollama answers it (default qwen3:1.7b, thinking off), and macOS `say` reads the answer.
- Audio LLM: Gemma 4 E4B hears the recording itself (Ollama /api/chat with the WAV in
  `images`) and answers, skipping speech recognition; `say` reads the answer.

Every stage is timed (speech recognition, LLM first token and total, speech synthesis; model
loading excluded) and the sum is the speech-to-speech latency. A short conversation is kept in
the session (the LLM sees the last turns), and saved as one `runs/asr_compare-*.json` per
conversation with mode "voice_chat".
"""

import base64
import json
import time
from datetime import datetime
from pathlib import Path

import jobs
import speech_common as sc
import streamlit as st
from common import PROJECT_DIR, kept, ollama_url, save_run
from i18n import tr

from local_llm import asr_metrics, audio_aug, catalog, gateway_client, tts
from local_llm import speech_live as sl
from local_llm.gateway import ollama as gw_ollama
from local_llm.streaming import Backend, stream_chat

PATHS = ("cascade", "audio_llm")
DEFAULT_ASR = "qwen3-asr-0.6b"
DEFAULT_LLMS = ["qwen3:1.7b", "gemma4:e2b", "gemma4:e4b"]
DEFAULT_AUDIO_LLM = "gemma4:e4b"
MAX_TURNS = 4  # earlier turns the LLM sees
MAX_TOKENS = 400
# Short questions spoken by a macOS voice, for trying the assistant without a microphone.
QUESTIONS = {
    "zh_weekend": ("台北有什麼適合週末帶小孩去的地方？", "Meijia"),
    "zh_asr": ("用一句話解釋什麼是語音辨識。", "Meijia"),
    "zh_party": ("幫我想三個團隊聚餐的點子。", "Meijia"),
    "en_speaking": ("What is a good way to practice public speaking?", "Samantha"),
    "en_asr": ("Explain in one sentence what speech recognition is.", "Samantha"),
}
AUDIO_INSTRUCTION = (
    "The attached audio is the user speaking to you. Answer what they say or ask, directly "
    "and briefly; do not transcribe or describe the audio."
)
SPOKEN = "(spoken question, heard directly by the audio model)"


def path_label(path: str) -> str:
    return {
        "cascade": tr("串接：辨識 → LLM → 合成", "Cascade: ASR → LLM → TTS"),
        "audio_llm": tr("音訊 LLM：直接聽 → 合成", "Audio LLM: hears it → TTS"),
    }[path]


# -- worker (no Streamlit calls) ------------------------------------------------------------


def _compute(out: dict) -> float:
    load = (out.get("load_seconds") or 0) + (out.get("ollama_load_seconds") or 0)
    return max(out["seconds"] - load - (out.get("queue_seconds") or 0), 0.01)


def _answer(turn: dict, base: str, model: str, messages: list[dict], images=None) -> None:
    """Stream the LLM's answer into turn["answer"]; sets the LLM timings (load excluded)."""
    turn["answer"], stats = "", None
    for event in stream_chat(
        Backend("Ollama", "ollama", base), model, messages, think=False, temperature=0.3,
        max_tokens=MAX_TOKENS, images=images,
    ):  # fmt: skip
        if event.kind == "content":
            turn["answer"] += event.text
        elif event.kind == "done":
            stats = event.stats
    load = (stats.load_s or 0) if stats else 0
    total = (stats.total_s if stats else 0) - load
    first = ((stats.first_answer_s or stats.total_s) if stats else 0) - load
    turn["llm_load_s"] = round(load, 2)
    turn["llm_first_s"] = round(max(first, 0.01), 2)
    turn["llm_s"] = round(max(total, 0.01), 2)
    turn["answer"] = turn["answer"].strip()


def voice_work(job: jobs.Job, url: str, base: str) -> None:
    state = job.state
    turn, history = state["turn"], state["history"]
    audio = Path(state["audio"])
    started = time.perf_counter()
    if turn["path"] == "cascade":
        turn["stage"] = "asr"
        job.detail, job.progress = turn["asr"], 0.1
        out = gateway_client.transcribe(audio, turn["asr"], None, base_url=url)
        turn["question"], turn["asr_s"] = out["text"].strip(), round(_compute(out), 2)
        language = "en" if asr_metrics.is_english(turn["question"]) else "zh"
        if not turn["question"]:
            raise RuntimeError("no speech was recognised in the recording")
        turn["stage"] = "llm"
        job.detail, job.progress = turn["llm"], 0.4
        messages = sl.messages(history, turn["question"], sl.system_prompt(language), MAX_TURNS)
        _answer(turn, base, turn["llm"], messages)
    else:
        turn["question"], turn["asr_s"] = None, 0.0
        turn["stage"] = "llm"
        job.detail, job.progress = turn["llm"], 0.3
        wav = gw_ollama.wav_bytes(audio_aug.load(audio))
        messages = sl.messages(history, AUDIO_INSTRUCTION, sl.system_prompt(None), MAX_TURNS)
        _answer(turn, base, turn["llm"], messages, images=[base64.b64encode(wav).decode()])
    if job.cancel.is_set():
        return
    turn["stage"] = "tts"
    job.detail, job.progress = "say", 0.8
    english = asr_metrics.is_english(turn["answer"])
    turn["voice"] = state["voice_en"] if english else state["voice_zh"]
    spoken = sl.speakable(turn["answer"])
    if spoken:
        out = tts.timed_synthesize(spoken, turn["voice"], state["rate"] or None)
        turn["answer_file"] = str(out["path"])
        turn["tts_s"], turn["answer_duration"] = out["seconds"], out["duration"]
    else:
        turn["tts_s"] = 0.0
    turn["total_s"] = round((turn["asr_s"] or 0) + turn["llm_s"] + turn["tts_s"], 2)
    turn["wall_s"] = round(time.perf_counter() - started, 2)
    turn["stage"] = "done"
    job.progress = 1.0
    state["run_path"] = str(_save(state, history + [turn]))


def _save(state: dict, turns: list[dict]) -> Path:
    """One run file per conversation: the first turn creates it, later turns rewrite it."""
    saved = [
        {k: (_relative(v) if k.endswith("_file") and v else v) for k, v in t.items()} for t in turns
    ]
    payload = {
        "mode": "voice_chat",
        "conversation": state["conversation"],
        "voices": {"zh": state["voice_zh"], "en": state["voice_en"]},
        "rate": state["rate"] or None,
        "turns": saved,
    }
    if state.get("run_path") and Path(state["run_path"]).exists():
        path = Path(state["run_path"])
        record = json.loads(path.read_text())
        record.update(payload, updated_at=datetime.now().astimezone().isoformat())
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str) + "\n")
        return path
    return save_run("asr_compare", payload)


def _relative(path: str) -> str:
    try:
        return str(Path(path).relative_to(PROJECT_DIR))
    except ValueError:
        return path


# -- drawing --------------------------------------------------------------------------------


def _file(path: str | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    p = p if p.is_absolute() else PROJECT_DIR / p
    return str(p) if p.exists() else None


def timings(turn: dict) -> str:
    parts = []
    if turn.get("path") == "cascade" and turn.get("asr_s") is not None:
        parts.append(tr(f"辨識 {turn['asr_s']:.2f}s", f"ASR {turn['asr_s']:.2f}s"))
    if turn.get("llm_s") is not None:
        parts.append(
            tr(f"LLM 首字 {turn['llm_first_s']:.2f}s／共 {turn['llm_s']:.2f}s",
               f"LLM first token {turn['llm_first_s']:.2f}s / total {turn['llm_s']:.2f}s")
        )  # fmt: skip
    if turn.get("tts_s") is not None:
        parts.append(tr(f"合成 {turn['tts_s']:.2f}s", f"TTS {turn['tts_s']:.2f}s"))
    if turn.get("total_s") is not None:
        parts.append(
            tr(
                f"**語音到語音 {turn['total_s']:.2f}s**",
                f"**speech to speech {turn['total_s']:.2f}s**",
            )
        )
    if (turn.get("llm_load_s") or 0) >= 0.5:
        parts.append(
            tr(f"另載入 {turn['llm_load_s']:.1f}s", f"plus load {turn['llm_load_s']:.1f}s")
        )
    return " · ".join(parts)


def show_turn(turn: dict, labels: dict, autoplay: bool = False, running: bool = False) -> None:
    stage = turn.get("stage")
    with st.chat_message("user", avatar=":material/person:"):
        if turn.get("question_file") and _file(turn["question_file"]):
            st.audio(_file(turn["question_file"]))
        if turn.get("path") == "audio_llm":
            st.caption(
                tr("（音訊 LLM 直接聽錄音，沒有逐字稿）",
                   "(the audio LLM hears the recording itself; there is no transcript)")
            )  # fmt: skip
        elif turn.get("question"):
            st.write(sc.show(turn["question"]))
        elif running and stage == "asr":
            st.caption(tr("辨識中…", "Transcribing…"))
    with st.chat_message("assistant", avatar=":material/smart_toy:"):
        st.caption(
            path_label(turn["path"]) + " · "
            + (f"{labels.get(turn['asr'], turn['asr'])} → " if turn["path"] == "cascade" else "")
            + (labels.get(turn["llm"], turn["llm"]))
        )  # fmt: skip
        if turn.get("error"):
            st.error(turn["error"])
            return
        if turn.get("answer"):
            st.write(sc.show(turn["answer"]) + (" ▌" if running and stage == "llm" else ""))
        elif running:
            st.caption(tr("思考中…", "Thinking…") if stage == "llm" else tr("等待中…", "Waiting…"))
        if running and stage == "tts":
            st.caption(tr("合成語音中…", "Speaking…"))
        if _file(turn.get("answer_file")):
            st.audio(_file(turn["answer_file"]), autoplay=autoplay)
        if turn.get("total_s") is not None:
            st.caption(timings(turn))


def latency_metrics(turn: dict) -> None:
    if turn.get("total_s") is None:
        return
    cols = st.columns(5)
    cols[0].metric(tr("語音辨識", "Speech recognition"),
                   f"{turn['asr_s']:.2f}s" if turn["path"] == "cascade" else "—")  # fmt: skip
    cols[1].metric(tr("LLM 首字", "LLM first token"), f"{turn['llm_first_s']:.2f}s")
    cols[2].metric(tr("LLM 完成", "LLM done"), f"{turn['llm_s']:.2f}s")
    cols[3].metric(tr("語音合成", "Speech synthesis"), f"{turn['tts_s']:.2f}s")
    cols[4].metric(tr("語音到語音", "Speech to speech"), f"{turn['total_s']:.2f}s")
    st.caption(
        tr(
            "語音到語音＝辨識＋LLM 完整回答＋合成（依序執行，不含載入模型）。"
            "逐句串流合成可以讓第一句更早播出，這裡先等整段回答完成再合成。",
            "Speech to speech = recognition + the LLM's full answer + synthesis (one after "
            "another, model loading excluded). Synthesising sentence by sentence would start "
            "the audio sooner; here the whole answer is synthesised at once.",
        )
    )


@st.fragment(run_every=1)
def live_view(job: jobs.Job, labels: dict) -> None:
    if not job.running:
        st.rerun()
    show_turn(job.state["turn"], labels, running=True)
    sc.running_bar(job)


def cloud_note() -> None:
    st.caption(
        tr(
            "雲端選項（只列出，本頁不連線）：Azure OpenAI `gpt-realtime`、`gpt-live-1` "
            "等即時語音模型直接收音、直接回語音，延遲更低，但音訊會離開本機。",
            "Cloud options (listed only; this page does not connect to them): Azure OpenAI "
            "`gpt-realtime` and `gpt-live-1` take speech in and give speech out directly with "
            "lower latency, but the audio leaves this machine.",
        )
    )


# -- page -----------------------------------------------------------------------------------


def _record_finished(job: jobs.Job | None) -> None:
    """Move a finished turn from its job into the session's conversation (once)."""
    if not job or job.running or job.id in st.session_state.setdefault("vc_recorded", []):
        return
    st.session_state.vc_recorded.append(job.id)
    turn = dict(job.state["turn"])
    if job.error:
        turn["error"] = job.error
    elif job.status == "cancelled":
        turn["error"] = tr("已停止", "Stopped")
    else:
        st.session_state.vc_run_path = job.state.get("run_path")
        st.session_state.vc_autoplay = job.id
    st.session_state.setdefault("vc_turns", []).append(turn | {"job": job.id})


def _new_conversation() -> None:
    st.session_state.vc_turns = []
    st.session_state.vc_run_path = None
    st.session_state.vc_conversation = datetime.now().strftime("%Y%m%d-%H%M%S")


def settings(labels: dict, status: dict) -> dict:
    st.session_state.setdefault("vc_path", "cascade")
    path = (
        st.segmented_control(
            tr("做法", "Pipeline"), list(PATHS), format_func=path_label, **kept("vc_path")
        )
        or "cascade"
    )
    installed = [m for m in labels if status.get(m, {}).get("installed")]
    chat = [m for m in sc.ollama_models(ollama_url()) if catalog.is_chat(m)]
    chat = [m for m in chat if not m.startswith("translategemma")] or chat
    out = {"path": path, "asr": None}
    a, b = st.columns(2)
    if path == "cascade":
        asr_options = [m for m in installed if sc.backend(m) != "ollama-audio"] or installed
        st.session_state.setdefault(
            "vc_asr", DEFAULT_ASR if DEFAULT_ASR in asr_options else (asr_options or [""])[0]
        )
        if asr_options and st.session_state.vc_asr not in asr_options:
            st.session_state.vc_asr = asr_options[0]
        out["asr"] = a.selectbox(
            tr("語音辨識模型", "Speech model"), asr_options,
            format_func=lambda m: labels.get(m, m), **kept("vc_asr"),
        )  # fmt: skip
        default = next((m for m in DEFAULT_LLMS if m in chat), chat[0] if chat else None)
        st.session_state.setdefault("vc_llm", default)
        if chat and st.session_state.vc_llm not in chat:
            st.session_state.vc_llm = default
        out["llm"] = b.selectbox(
            tr("回答用的 LLM（Ollama）", "Answering LLM (Ollama)"), chat, **kept("vc_llm"),
            help=tr("會關閉思考模式（模型支援時）以加快回答。",
                    "Thinking is switched off (when the model allows) for faster answers."),
        )  # fmt: skip
    else:
        audio_llms = [m for m in installed if sc.backend(m) == "ollama-audio"]
        st.session_state.setdefault(
            "vc_audio_llm",
            DEFAULT_AUDIO_LLM if DEFAULT_AUDIO_LLM in audio_llms else (audio_llms or [""])[0],
        )
        out["llm"] = a.selectbox(
            tr("音訊 LLM", "Audio LLM"), audio_llms,
            format_func=lambda m: labels.get(m, m), **kept("vc_audio_llm"),
        )  # fmt: skip
        b.caption(
            tr(
                "Gemma 4 直接聽錄音並回答，不經過語音辨識；回答的文字再交給 macOS 念出來。",
                "Gemma 4 hears the recording and answers, without speech recognition; the "
                "answer is then read aloud by macOS.",
            )
        )
    voices = tts.voices()
    zh = [v for v in voices if v["locale"].startswith("zh")]
    en = [v for v in voices if v["locale"].startswith("en")]
    by_name = {v["name"]: v for v in voices}
    c, d, e = st.columns(3)
    for key, options, short in (("vc_voice_zh", zh, "Meijia"), ("vc_voice_en", en, "Samantha")):
        names = [v["name"] for v in options]
        default = next(
            (v["name"] for v in options if v["short"] == short), names[0] if names else ""
        )
        st.session_state.setdefault(key, default)
        if names and st.session_state[key] not in names:
            st.session_state[key] = default
    out["voice_zh"] = c.selectbox(
        tr("中文回答的聲音", "Voice for Chinese answers"), [v["name"] for v in zh],
        format_func=lambda n: by_name[n]["short"] + f" · {by_name[n]['locale']}",
        **kept("vc_voice_zh"),
    )  # fmt: skip
    out["voice_en"] = d.selectbox(
        tr("英文回答的聲音", "Voice for English answers"), [v["name"] for v in en],
        format_func=lambda n: by_name[n]["short"] + f" · {by_name[n]['locale']}",
        **kept("vc_voice_en"),
    )  # fmt: skip
    st.session_state.setdefault("vc_rate", 0)
    out["rate"] = e.select_slider(
        tr("語速", "Rate"), [0, 160, 180, 200, 220, 250],
        format_func=lambda r: str(r) if r else tr("預設", "default"), **kept("vc_rate"),
    )  # fmt: skip
    return out


def question_audio() -> tuple[Path, str] | None:
    """(audio file, display name) of the question, or None until one is recorded."""
    sources = {"sample": tr("範例問題", "Sample question"), "record": tr("用麥克風問", "Ask aloud")}
    st.session_state.setdefault("vc_source", "sample")
    source = (
        st.segmented_control(
            tr("問題", "Question"), list(sources), format_func=sources.get, **kept("vc_source")
        )
        or "sample"
    )
    if source == "sample":
        st.session_state.setdefault("vc_question", "zh_weekend")
        key = st.selectbox(
            tr("範例問題", "Sample question"), list(QUESTIONS),
            format_func=lambda k: QUESTIONS[k][0], **kept("vc_question"),
        )  # fmt: skip
        text, voice = QUESTIONS[key]
        try:
            path = tts.synthesize(text, voice, out=None)
        except Exception as exc:
            st.error(str(exc))
            return None
        st.audio(str(path))
        st.caption(tr(f"合成的問題（macOS {voice}）", f"Synthetic question (macOS {voice})"))
        return path, f"sample:{key}"
    upload = st.audio_input(
        tr("按下開始錄音，說完再按停止", "Press to record, press again to stop"), key="vc_record"
    )
    if not upload or not upload.size:
        return None
    return sc.cached_audio(upload.getvalue(), upload.name or "question.wav"), "recording"


def voice_page(url: str, status: dict) -> None:
    labels = sc.names() | {m: m for m in sc.ollama_models(ollama_url())}
    if not tts.available():
        st.warning(
            tr(
                "找不到 macOS 的 `say` 指令，無法念出回答。",
                "macOS's `say` command was not found, so answers cannot be spoken.",
            )
        )
        return
    with st.container(border=True):
        config = settings(labels, status)
    job = jobs.get(st.session_state.get("vc_job"))
    _record_finished(job)
    st.session_state.setdefault("vc_conversation", datetime.now().strftime("%Y%m%d-%H%M%S"))
    turns = st.session_state.setdefault("vc_turns", [])

    head, clear = st.columns([4, 1], vertical_alignment="bottom")
    head.markdown(tr("#### 對話", "#### Conversation"))
    clear.button(tr("新對話", "New conversation"), icon=":material/restart_alt:",
                 on_click=_new_conversation, disabled=bool(job and job.running),
                 width="stretch")  # fmt: skip
    if not turns and not (job and job.running):
        st.caption(
            tr(
                "還沒有對話。選一個範例問題或錄音後按「發問」。",
                "No conversation yet. Pick a sample question or record one, then Ask.",
            )
        )
    autoplay = st.session_state.pop("vc_autoplay", None)
    for turn in turns:
        show_turn(turn, labels, autoplay=autoplay is not None and turn.get("job") == autoplay)
    if job and job.running:
        live_view(job, labels)
    elif turns and turns[-1].get("total_s") is not None:
        latency_metrics(turns[-1])
    if st.session_state.get("vc_run_path"):
        try:
            shown = Path(st.session_state.vc_run_path).relative_to(PROJECT_DIR)
        except ValueError:
            shown = st.session_state.vc_run_path
        st.caption(tr(f"已儲存：`{shown}`", f"Saved: `{shown}`"))

    st.divider()
    asked = question_audio()
    ready = (
        asked is not None
        and bool(config["llm"])
        and (config["path"] != "cascade" or bool(config["asr"]))
    )
    if st.button(
        tr("發問", "Ask"), type="primary", icon=":material/send:",
        disabled=bool(job and job.running) or not ready,
    ):  # fmt: skip
        audio, name = asked
        history = [
            {"question": t.get("question") or SPOKEN, "answer": t.get("answer")}
            for t in turns
            if t.get("answer") and not t.get("error")
        ]
        turn = {
            "path": config["path"], "asr": config["asr"], "llm": config["llm"],
            "question_file": str(audio), "source": name, "stage": "waiting",
            "at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }  # fmt: skip
        job = jobs.Job(
            "voice_chat",
            tr("語音對話", "Voice chat") + f" · {config['llm']}",
            sc.PAGE,
            state={
                "audio": str(audio),
                "turn": turn,
                "history": history,
                "voice_zh": config["voice_zh"],
                "voice_en": config["voice_en"],
                "rate": config["rate"],
                "conversation": st.session_state.vc_conversation,
                "run_path": st.session_state.get("vc_run_path"),
            },
        )
        base = ollama_url()
        jobs.start(job, lambda j: voice_work(j, url, base))
        st.session_state.vc_job = job.id
        st.rerun()
    cloud_note()


# -- history --------------------------------------------------------------------------------


def history_view(data: dict) -> None:
    labels = sc.names()
    turns = [t for t in data.get("turns") or [] if isinstance(t, dict)]
    voices = data.get("voices") or {}
    st.caption(
        tr(f"語音對話 · {len(turns)} 輪", f"Voice chat · {len(turns)} turns")
        + (f" · {voices.get('zh', '')} / {voices.get('en', '')}" if voices else "")
    )
    for turn in turns:
        show_turn(turn, labels)
    done = [t for t in turns if t.get("total_s") is not None]
    if done:
        mean = sum(t["total_s"] for t in done) / len(done)
        st.caption(
            tr(f"平均語音到語音延遲 {mean:.2f}s", f"Mean speech-to-speech latency {mean:.2f}s")
        )
    cloud_note()


def history_summary(data: dict) -> dict:
    # Cached across languages (history.summarize): ids only, no tr().
    turns = [t for t in data.get("turns") or [] if isinstance(t, dict)]
    models = []
    for t in turns:
        for m in (t.get("asr"), t.get("llm")):
            if m and m not in models:
                models.append(m)
    first = next((t for t in turns if t.get("answer")), {})
    snippet = (first.get("question") or "") + " → " + (first.get("answer") or "")
    return {
        "title": f"voice chat · {len(turns)} turns",
        "models": models,
        "seconds": round(sum(t.get("total_s") or 0 for t in turns), 2) or None,
        "snippet": snippet.strip(" →")[:220],
        "best": None,
        "error": bool(turns) and all(t.get("error") for t in turns),
    }
