"""Live mode, microphone: speak and watch a true streaming model caption you as you talk.

The page mounts a small component (app/static/live_mic, st.components.v2) that captures the
microphone in the browser, converts it to 16 kHz PCM16 and streams 100 ms frames over a
WebSocket straight to the model gateway (/v1/audio/stream/ws, gateway/live.py). Captions and
latency readouts update live in the component; when the person presses stop, the session
(events, per-utterance latencies, the recorded audio) comes back to Python and is saved as a
run (runs/asr_compare-*.json, mode "live", source "mic") with the audio in
storage/speech_cache/mic/, so History can replay it.
"""

import base64
from datetime import datetime
from functools import cache
from pathlib import Path

import speech_common as sc
import speech_stream
import streamlit as st
from common import PROJECT_DIR, kept, save_run
from i18n import tr

STATIC = Path(__file__).parent / "static" / "live_mic"
MIC_DIR = sc.CACHE_DIR / "mic"
DEFAULT_MODEL = "zipformer-zh-en-stream"


@cache
def component():
    return st.components.v2.component(
        "lal_live_mic",
        html=(STATIC / "live_mic.html").read_text(encoding="utf-8"),
        css=(STATIC / "live_mic.css").read_text(encoding="utf-8"),
        js=(STATIC / "live_mic.js").read_text(encoding="utf-8"),
    )


def ws_url(url: str) -> str:
    """The gateway's live endpoint for the browser: http://127.0.0.1:11500 -> ws://..."""
    base = url.rstrip("/")
    base = "wss://" + base[8:] if base.startswith("https://") else "ws://" + base.split("://")[-1]
    return base + "/v1/audio/stream/ws"


def labels() -> dict:
    return {
        "start": tr("開始說話", "Start"),
        "stop": tr("停止", "Stop"),
        "first": tr("首字延遲（最近／中位數）", "First text (last / median)"),
        "final": tr("定稿延遲（最近／中位數）", "Final text (last / median)"),
        "median": tr("中位數", "median"),
        "seconds": tr("已送出音訊", "Audio sent"),
        "utterances": tr("偵測到的語句", "Utterances"),
        "idle": tr(
            "按「開始說話」，允許使用麥克風後就可以講。",
            "Press Start and allow the microphone, then speak.",
        ),  # fmt: skip
        "connecting": tr("連線模型閘道、載入模型中…", "Connecting and loading the model…"),
        "recording": tr("收音中，講話吧。", "Listening: speak now."),
        "finishing": tr("收尾中…", "Finishing…"),
        "done": tr("已結束，結果在下方。", "Stopped; the results are below."),
        "failed": tr("已中斷。", "Stopped."),
        "denied": tr(
            "麥克風權限被拒絕。請在瀏覽器網址列左側允許麥克風，再按一次開始。",
            "Microphone permission was denied. Allow the microphone in the browser's address "
            "bar, then press Start again.",
        ),
        "no_mic": tr("無法開啟麥克風", "Could not open the microphone"),
        "no_gateway": tr(
            "連不到模型閘道的串流端點，請確認閘道正在執行（./scripts/restart_gateway.sh）。",
            "Cannot reach the model gateway's streaming endpoint; check that the gateway is "
            "running (./scripts/restart_gateway.sh).",
        ),
        "closed": tr("連線中斷。", "The connection closed."),
    }


def save(payload: dict, url: str) -> dict:
    """The finished session as a saved run (plus its audio); returns the run data."""
    done = payload.get("done") or {}
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    MIC_DIR.mkdir(parents=True, exist_ok=True)
    wav = MIC_DIR / f"mic-{stamp}.wav"
    try:
        wav.write_bytes(base64.b64decode(payload.get("wav_base64") or ""))
    except (ValueError, OSError):
        wav = None
    events = [e for e in payload.get("events") or [] if e.get("type") in ("partial", "final")]
    model = payload.get("model")
    data = {
        "mode": "live",
        "method": "stream",
        "source": "mic",
        "audio": wav.name if wav else "microphone",
        "audio_file": str(wav.relative_to(PROJECT_DIR)) if wav else None,
        "clip_s": None,
        "duration": done.get("duration"),
        "frame_ms": 100,
        "pace": "live",
        "language": None,
        "reference": None,
        "segments": done.get("segments") or [],
        "stopped": False,
        "started_at": payload.get("started_at"),
        "live": {k: done.get(k) for k in ("first_latencies", "final_latencies",
                                          "first_median_s", "final_median_s", "utterances")},
        "results": [
            {
                "model": model,
                "text": done.get("text", ""),
                "events": events,
                "stats": done.get("summary"),
                "score": None,
                "error": None,
                "append_only": done.get("append_only"),
                "load_seconds": payload.get("load_seconds"),
            }
        ],
    }  # fmt: skip
    path = save_run("asr_compare", data)
    data["path"] = str(path.relative_to(path.parents[1]))
    return data


def mic_page(url: str, status: dict) -> None:
    names = sc.names("stream")
    # Models slower than real time (full-precision Voxtral here) cannot caption a live voice.
    installed = [m for m in names if status.get(m, {}).get("installed")
                 and speech_stream.realtime(m)]  # fmt: skip
    st.caption(
        tr(
            "對著麥克風講話，真串流模型邊聽邊出字：灰色斜體是還會變的部分結果，黑字是定稿。"
            "音訊每 100 ms 從瀏覽器直接送到本機模型閘道（WebSocket），不經過其他伺服器。",
            "Speak into the microphone and a true streaming model captions you as you talk: "
            "grey italics are partial text that may still change, black text is final. Audio "
            "goes from the browser to the local model gateway every 100 ms (WebSocket) and "
            "nowhere else.",
        )
    )
    if not installed:
        st.info(
            tr("還沒有下載任何串流模型；到「真串流」或模型庫下載，例如 sherpa-onnx Zipformer。",
               "No streaming model is downloaded yet; download one (e.g. sherpa-onnx "
               "Zipformer) under True streaming or in the model library."),
            icon=":material/download:",
        )  # fmt: skip
        return
    st.session_state.setdefault(
        "lm_model", DEFAULT_MODEL if DEFAULT_MODEL in installed else installed[0]
    )
    if st.session_state.lm_model not in installed:
        st.session_state.lm_model = installed[0]
    model = st.selectbox(
        tr("串流模型（已下載）", "Streaming model (downloaded)"),
        installed,
        format_func=lambda m: names.get(m, m),
        help=tr(
            "只列出跟得上即時的模型：全精度 Voxtral 在這台 Mac 上處理比說話慢（RTF 約 2.3），"
            "所以不列出，請用 4-bit 版；sherpa-onnx 模型在 CPU 上就很快。",
            "Only models that keep up with real time are listed: full-precision Voxtral is "
            "slower than speech on this Mac (RTF about 2.3), so use the 4-bit build; the "
            "sherpa-onnx models are fast on the CPU.",
        ),
        **kept("lm_model"),
    )
    sc.licence_warning([model], names)
    result = component()(
        key="lal_live_mic",
        data={
            "key": "lal_live_mic",
            "ws_url": ws_url(url),
            "model": model,
            "model_label": names.get(model, model),
            "labels": labels(),
            "s2tw": bool(st.session_state.get("speech_s2tw", True)),
        },
        on_session_change=lambda: None,
    )
    payload = getattr(result, "session", None) if result is not None else None
    if payload and payload.get("id") != st.session_state.get("lm_saved_id"):
        st.session_state.lm_saved_id = payload.get("id")
        st.session_state.lm_last = save(payload, url)
    last = st.session_state.get("lm_last")
    if last:
        st.subheader(tr("上一段錄音", "Last session"))
        speech_stream.history_view(last)
        sc.saved_caption(last)
    st.caption(
        tr(
            "首字延遲＝偵測到開始說話到第一個字出現；定稿延遲＝一句話講完到這句的最終文字。"
            "時間從瀏覽器送出的第一段音訊算起，包含瀏覽器緩衝、網路與模型處理；語句的起訖用音量"
            "偵測（VAD）找出。停止後整段會存成執行紀錄（含錄音），可在「執行紀錄」重播。",
            "First text = from detected speech start to the first character; final text = from "
            "the end of an utterance to its final text. Times run from the first audio frame "
            "the browser sent, so they include browser buffering, the network and the model; "
            "utterances are found by an energy VAD. Stopping saves the session as a run (with "
            "the recording) that History can replay.",
        )
    )
