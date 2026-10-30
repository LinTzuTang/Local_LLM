"""Client for the local model gateway (speech-to-text, text-to-speech, documents, OCR, rerank).

The gateway (`uv run python -m local_llm.gateway`, see docs/GATEWAY.md) serves the models
Ollama cannot run with an Ollama-like API. Typical use:

    from local_llm import gateway_client as gw
    gw.transcribe(Path("meeting.wav"))["text"]
    gw.parse_document(Path("contract.pdf"))["markdown"]
    gw.rerank("逾期罰款", ["passage 1", "passage 2"])  # [{"index", "relevance_score", ...}]
    gw.translate(Path("talk.wav"), "whisper-large-v3", "zh")["text"]  # speech -> English
    gw.diarize(Path("meeting.wav"), num_speakers=2)["segments"]  # [{"start", "end", "speaker"}]
    gw.speech_file("各位早安。", "Kokoro 曉北")  # WAV path in the shared TTS cache
    gw.ocr(Path("page.png"))  # macOS Vision; gw.ocr_score(Path("block.png"), text) -> 0-1 (Jev)
    for event in gw.stream(Path("talk.wav"), "zipformer-zh-en-stream"):  # true streaming
        print(event)  # {"t_audio", "t_wall", "text", "final", "seg"}, then {"done": True, ...}

Errors from the gateway (unknown model, model not downloaded, bad input) raise GatewayError
with the server's message; a gateway that is not running raises httpx.ConnectError
(`version()` returns None instead).
"""

import json
import os
from collections.abc import Iterator
from pathlib import Path

import httpx

DEFAULT_URL = f"http://127.0.0.1:{os.getenv('LOCAL_LLM_GATEWAY_PORT', '11500')}"
INFERENCE_TIMEOUT_S = 900  # first use includes loading the model; long recordings take minutes


class GatewayError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def _client(base_url: str, timeout: float | None = 10) -> httpx.Client:
    return httpx.Client(base_url=base_url, timeout=timeout, trust_env=False)


def _check(response: httpx.Response) -> httpx.Response:
    if response.is_error:
        try:
            message = response.json().get("error") or response.text
        except ValueError:
            message = response.text
        raise GatewayError(f"{message} (HTTP {response.status_code})", response.status_code)
    return response


def version(base_url: str = DEFAULT_URL) -> str | None:
    """Gateway version, or None when it is not running."""
    try:
        with _client(base_url, 3) as client:
            return client.get("/api/version").json()["version"]
    except (httpx.HTTPError, ValueError, KeyError):
        return None


def models(base_url: str = DEFAULT_URL) -> list[dict]:
    """All gateway models: id, task, installed, size_gb, loaded, name, license."""
    with _client(base_url) as client:
        return _check(client.get("/api/tags")).json()["models"]


def loaded(base_url: str = DEFAULT_URL) -> list[dict]:
    """Models in memory: id, task, expires_at, device, loaded_at."""
    with _client(base_url) as client:
        return _check(client.get("/api/ps")).json()["models"]


def pull(model: str, base_url: str = DEFAULT_URL) -> Iterator[dict]:
    """Download a model; yields progress dicts ({"status", "total", "completed"}) like Ollama.

    Closing the iterator early cancels the download.
    """
    with _client(base_url, None) as client:
        with client.stream("POST", "/api/pull", json={"model": model}) as response:
            if response.is_error:
                response.read()
                _check(response)
            for line in response.iter_lines():
                if line:
                    event = json.loads(line)
                    if "error" in event:
                        raise GatewayError(event["error"])
                    yield event


def delete(model: str, base_url: str = DEFAULT_URL) -> None:
    with _client(base_url, 60) as client:
        _check(client.request("DELETE", "/api/delete", json={"model": model}))


def unload(model: str, base_url: str = DEFAULT_URL) -> None:
    with _client(base_url, 60) as client:
        _check(client.post("/api/unload", json={"model": model}))


def _upload(source: Path | bytes, filename: str | None, default_name: str) -> tuple:
    if isinstance(source, bytes | bytearray):
        return (filename or default_name, bytes(source))
    path = Path(source)
    return (filename or path.name, path.read_bytes())


def _form(**fields) -> dict:
    return {k: str(v) for k, v in fields.items() if v is not None}


def transcribe(
    audio: Path | bytes,
    model: str = "breeze-asr-25",
    language: str | None = None,
    filename: str | None = None,
    *,
    prompt: str | None = None,
    keep_alive: float | None = None,
    base_url: str = DEFAULT_URL,
) -> dict:
    """Speech to text. Returns {"text", "segments": [{"start", "end", "text"}], "language",
    "duration", "model", "seconds"}. language: "zh", "en", ... or None to detect; prompt: a
    Whisper initial prompt (e.g. "以下是繁體中文的會議錄音。" to steer the script)."""
    with _client(base_url, INFERENCE_TIMEOUT_S) as client:
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": _upload(audio, filename, "audio.wav")},
            data=_form(model=model, language=language, prompt=prompt, keep_alive=keep_alive),
        )
        return _check(response).json()


def rerank(
    query: str,
    documents: list[str],
    model: str = "bge-reranker-v2-m3",
    top_n: int | None = None,
    *,
    keep_alive: float | None = None,
    base_url: str = DEFAULT_URL,
) -> list[dict]:
    """[{"index", "relevance_score", "document"}], most relevant first (score 0-1)."""
    body = {"model": model, "query": query, "documents": documents, "top_n": top_n,
            "keep_alive": keep_alive}  # fmt: skip
    with _client(base_url, INFERENCE_TIMEOUT_S) as client:
        response = client.post("/v1/rerank", json={k: v for k, v in body.items() if v is not None})
        return _check(response).json()["results"]


def parse_document(
    doc: Path | bytes,
    filename: str | None = None,
    model: str = "docling",
    *,
    keep_alive: float | None = None,
    base_url: str = DEFAULT_URL,
) -> dict:
    """PDF / DOCX / PPTX / HTML / Markdown / image -> {"markdown", "pages", "tables" (count),
    "model", "seconds"}. With bytes, pass `filename` so the type is known (default .pdf)."""
    with _client(base_url, INFERENCE_TIMEOUT_S) as client:
        response = client.post(
            "/v1/documents/parse",
            files={"file": _upload(doc, filename, "document.pdf")},
            data=_form(model=model, keep_alive=keep_alive),
        )
        return _check(response).json()


def speech(
    text: str,
    voice: str,
    rate: int | None = None,
    sample_rate: int = 24000,
    *,
    base_url: str = DEFAULT_URL,
) -> bytes:
    """Text to speech with a platform voice (macOS, neural, cloud): 16-bit mono WAV bytes."""
    body = {"input": text, "voice": voice, "rate": rate, "sample_rate": sample_rate}
    with _client(base_url, INFERENCE_TIMEOUT_S) as client:
        return _check(client.post("/v1/audio/speech", json=body)).content


def speech_file(
    text: str,
    voice: str,
    rate: int | None = None,
    sample_rate: int = 24000,
    *,
    base_url: str = DEFAULT_URL,
) -> Path:
    """speech() as a file: the gateway keeps its renderings in the TTS cache on this machine
    (local_llm.tts), so the path is that cache file; written here if it is not there."""
    from . import tts

    data = speech(text, voice, rate, sample_rate, base_url=base_url)
    path = tts.cached(text, voice, rate, sample_rate)
    if path is None:
        path = tts.cache_path(text, voice + "+gateway", rate, sample_rate)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return path


def timed_speech(
    text: str, voice: str, rate: int | None = None, *, base_url: str = DEFAULT_URL
) -> dict:
    """The text sentence by sentence as the voice assistant speaks it, never from the cache:
    {"path" (WAV of all sentences, a temporary file), "seconds", "duration", "rtf", "load_s",
    "first_s", "max_s", "sentences"}."""
    import base64
    import tempfile

    body = {"input": text, "voice": voice, "rate": rate}
    with _client(base_url, INFERENCE_TIMEOUT_S) as client:
        out = _check(client.post("/v1/audio/speech/timed", json=body)).json()
    with tempfile.NamedTemporaryFile(prefix="tts-", suffix=".wav", delete=False) as f:
        f.write(base64.b64decode(out.pop("wav")))
    return {**out, "path": Path(f.name)}


def ocr(image: Path | bytes, model: str = "ocrmac", *, base_url: str = DEFAULT_URL) -> str:
    """The text of an image read on the gateway (macOS Vision)."""
    with _client(base_url, INFERENCE_TIMEOUT_S) as client:
        response = client.post("/v1/ocr", files={"file": _upload(image, None, "image.png")},
                               data=_form(model=model))  # fmt: skip
        return _check(response).json()["text"]


def ocr_score(image: Path | bytes, text: str, *, base_url: str = DEFAULT_URL) -> float:
    """MiniCPM-OCR Jev on the gateway: probability (0-1) that `text` is what the image (one
    region) shows."""
    with _client(base_url, INFERENCE_TIMEOUT_S) as client:
        response = client.post("/v1/ocr/score", files={"file": _upload(image, None, "image.png")},
                               data={"text": text})  # fmt: skip
        return float(_check(response).json()["score"])


def translate(
    audio: Path | bytes,
    model: str = "whisper-large-v3",
    language: str | None = None,
    filename: str | None = None,
    *,
    prompt: str | None = None,
    keep_alive: float | None = None,
    base_url: str = DEFAULT_URL,
) -> dict:
    """Speech -> English text in one step (Whisper task="translate", or an audio LLM such as
    gemma4:e4b told to translate). Returns the same fields as transcribe(); `language` is the
    spoken language (None to detect). Models without translation (large-v3-turbo, Qwen3-ASR,
    SenseVoice) raise GatewayError (HTTP 400)."""
    with _client(base_url, INFERENCE_TIMEOUT_S) as client:
        response = client.post(
            "/v1/audio/translations",
            files={"file": _upload(audio, filename, "audio.wav")},
            data=_form(model=model, language=language, prompt=prompt, keep_alive=keep_alive),
        )
        return _check(response).json()


def diarize(
    audio: Path | bytes,
    num_speakers: int | None = None,
    *,
    model: str = "speaker-diarization",
    threshold: float | None = None,
    filename: str | None = None,
    keep_alive: float | None = None,
    base_url: str = DEFAULT_URL,
) -> dict:
    """Who spoke when. Returns {"segments": [{"start", "end", "speaker"}], "speakers",
    "duration", "model", "seconds", "load_seconds"}; times in seconds, speakers numbered 0, 1,
    ... in order of first appearance. num_speakers: the known count (None = estimate it);
    threshold: clustering distance when the count is unknown (lower = more speakers)."""
    with _client(base_url, INFERENCE_TIMEOUT_S) as client:
        response = client.post(
            "/v1/audio/diarize",
            files={"file": _upload(audio, filename, "audio.wav")},
            data=_form(
                model=model, num_speakers=num_speakers, threshold=threshold, keep_alive=keep_alive
            ),
        )
        return _check(response).json()


def stream(
    audio: Path | bytes,
    model: str = "zipformer-zh-en-stream",
    *,
    frame_ms: int = 160,
    pace: str = "fast",
    filename: str | None = None,
    keep_alive: float | None = None,
    base_url: str = DEFAULT_URL,
) -> Iterator[dict]:
    """Run a true streaming model over a recording fed in `frame_ms` frames. Yields one dict
    per text change {"t_audio", "t_wall", "text", "final", "seg"} (see asr_streaming; with
    pace="fast" t_wall is simulated real time, with "realtime" the gateway feeds at the
    speaker's pace), then a last {"done": True, "duration", "compute_s", "append_only", ...}.
    Closing the iterator early stops the decode."""
    upload = {"file": _upload(audio, filename, "audio.wav")}
    data = _form(model=model, frame_ms=frame_ms, pace=pace, keep_alive=keep_alive)
    with _client(base_url, INFERENCE_TIMEOUT_S) as client:
        with client.stream("POST", "/v1/audio/stream", files=upload, data=data) as response:
            if response.is_error:
                response.read()
                _check(response)
            for line in response.iter_lines():
                if line:
                    event = json.loads(line)
                    if "error" in event:
                        raise GatewayError(event["error"])
                    yield event


def stream_transcribe(audio: Path | bytes, model: str = "zipformer-zh-en-stream", **kwargs) -> dict:
    """stream() collected: the closing summary plus {"events": [...]}."""
    events, done = [], {}
    for item in stream(audio, model, **kwargs):
        if item.get("done"):
            done = item
        else:
            events.append(item)
    return {**done, "events": events}
