"""HTTP API of the model gateway (FastAPI).

Ollama-style management: GET /api/version, /api/tags, /api/ps; POST /api/pull (NDJSON progress),
DELETE /api/delete, POST /api/unload. Inference, OpenAI-style where a convention exists:
POST /v1/audio/transcriptions, /v1/audio/translations (speech -> English text),
/v1/audio/diarize (who spoke when), /v1/audio/stream (true streaming models fed a recording
frame by frame; NDJSON events), /v1/audio/speech (text to speech, WAV) and /v1/audio/speech/timed
(sentence by sentence, timed), /v1/rerank, /v1/documents/parse, /v1/ocr (macOS Vision) and
/v1/ocr/score (MiniCPM-OCR Jev, its llama-server started on demand). Errors are {"error": "..."}.
WebSocket /v1/audio/stream/ws: live microphone audio (PCM16 frames) in, caption events out
(gateway/live.py); only pages served from this machine may connect. WebSocket
/v1/voice/agent/ws: the live voice assistant (gateway/voice.py, local_llm/voice_agent.py).

Endpoints are plain `def` functions, so FastAPI runs the blocking model work in its thread
pool and the event loop stays free for /api/ps, progress streams and other requests.
"""

import asyncio
import json
import queue
import re
import tempfile
import threading
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, Request, UploadFile, WebSocket
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, ValidationError
from starlette.exceptions import HTTPException

from . import MAX_UPLOAD_BYTES, VERSION, download
from .backends import BACKENDS, decode_audio
from .live import LiveSession
from .manager import GatewayError, Manager
from .registry import Store, load_specs
from .stream_backends import STREAM_BACKENDS
from .voice import voice_agent_ws

REAP_INTERVAL_S = 2.0
DOCUMENT_TYPES = {".pdf", ".docx", ".pptx", ".html", ".htm", ".md", ".png", ".jpg", ".jpeg",
                  ".tif", ".tiff", ".bmp", ".webp"}  # fmt: skip


class SpeechRequest(BaseModel):
    input: str
    voice: str
    rate: int | None = None  # words per minute (macOS voices)
    sample_rate: int = 24000


class ModelRequest(BaseModel):
    model: str | None = None
    name: str | None = None  # Ollama's older field name

    def model_id(self) -> str:
        if not (self.model or self.name):
            raise GatewayError(400, 'missing "model"')
        return self.model or self.name


async def model_request(request: Request) -> ModelRequest:
    """The JSON body of a management call, whatever its content type: `curl -d '{...}'` sends
    application/x-www-form-urlencoded, and Ollama accepts that too."""
    try:
        return ModelRequest.model_validate_json(await request.body() or b"{}")
    except ValidationError as exc:
        raise GatewayError(400, f'invalid request: expected {{"model": "<id>"}} ({exc})') from exc


Model = Annotated[ModelRequest, Depends(model_request)]


class RerankRequest(BaseModel):
    model: str = "bge-reranker-v2-m3"
    query: str = Field(min_length=1)
    documents: list[str] = Field(min_length=1, max_length=1000)
    top_n: int | None = Field(default=None, ge=1)
    keep_alive: float | str | None = None


def parse_keep_alive(value: float | str | None) -> float | None:
    """Seconds, or Ollama-style durations: "300", "5m", "1h", "30s", "-1" (forever), "0"."""
    if value is None or value == "":
        return None
    if isinstance(value, int | float):
        return float(value)
    match = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*([smh]?)\s*", value)
    if not match:
        raise GatewayError(400, f"invalid keep_alive {value!r}; use seconds or e.g. 5m")
    return float(match[1]) * {"": 1, "s": 1, "m": 60, "h": 3600}[match[2]]


LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "[::1]"}


def local_origin(origin: str | None) -> bool:
    """A browser WebSocket from a page on this machine (or no browser at all). Browsers let any
    web page open ws://127.0.0.1, so pages from elsewhere are refused."""
    if not origin:
        return True
    from urllib.parse import urlparse

    return (urlparse(origin).hostname or "") in LOCAL_HOSTS


def normalize_language(value: str | None) -> str | None:
    """Whisper language codes: "zh-TW" / "zh_Hant" -> "zh"; "auto" or empty -> detect."""
    if not value or value.lower() == "auto":
        return None
    return re.split(r"[-_]", value.strip())[0].lower()


@contextmanager
def saved_upload(upload: UploadFile):
    """Copy an upload to a temporary file whose suffix tells the backend the format."""
    suffix = Path(upload.filename or "").suffix.lower()
    suffix = suffix if re.fullmatch(r"\.[a-z0-9]{1,8}", suffix) else ""
    with tempfile.TemporaryDirectory(prefix="gateway-") as tmp:
        path = Path(tmp) / f"upload{suffix}"
        with path.open("wb") as out:
            while chunk := upload.file.read(1 << 20):
                out.write(chunk)
        if path.stat().st_size == 0:
            raise GatewayError(400, "the uploaded file is empty")
        yield path


class UploadLimit:
    """Refuse request bodies over the limit before they are read (HTTP 413)."""

    def __init__(self, app, max_bytes: int):
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            length = dict(scope["headers"]).get(b"content-length")
            if length and int(length) > self.max_bytes:
                limit = self.max_bytes / 1024**2
                response = JSONResponse(
                    {"error": f"upload too large (limit {limit:.0f} MB)"}, status_code=413
                )
                return await response(scope, receive, send)
            received = 0

            async def limited_receive():
                nonlocal received
                message = await receive()
                received += len(message.get("body", b""))
                if received > self.max_bytes:  # chunked upload without Content-Length
                    raise GatewayError(413, "upload too large")
                return message

            return await self.app(scope, limited_receive, send)
        return await self.app(scope, receive, send)


async def ndjson(events: AsyncIterator[dict]) -> AsyncIterator[str]:
    try:
        async for event in events:
            yield json.dumps(event, ensure_ascii=False) + "\n"
    except Exception as exc:  # after the stream started, errors travel in-band like Ollama's
        yield json.dumps({"error": f"{exc}"}, ensure_ascii=False) + "\n"


def create_app(
    manager: Manager | None = None,
    puller: Callable = download.pull,
    max_upload_bytes: int = MAX_UPLOAD_BYTES,
) -> FastAPI:
    manager = manager or Manager(load_specs(), BACKENDS | STREAM_BACKENDS, Store())
    pull_locks: dict[str, asyncio.Lock] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        stop = threading.Event()

        def reaper():
            while not stop.wait(REAP_INTERVAL_S):
                manager.reap()

        threading.Thread(target=reaper, name="gateway-reaper", daemon=True).start()
        yield
        stop.set()

    app = FastAPI(title="Local LLM model gateway", version=VERSION, lifespan=lifespan)
    app.state.manager = manager
    app.add_middleware(UploadLimit, max_bytes=max_upload_bytes)

    @app.exception_handler(GatewayError)
    async def gateway_error(request: Request, exc: GatewayError):
        return JSONResponse({"error": exc.message}, status_code=exc.status)

    @app.exception_handler(ValueError)
    async def bad_input(request: Request, exc: ValueError):
        return JSONResponse({"error": str(exc)}, status_code=400)

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        return JSONResponse({"error": str(exc.detail)}, status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError):
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'][1:])}: {e['msg']}" for e in exc.errors()
        )
        return JSONResponse({"error": f"invalid request: {problems}"}, status_code=422)

    @app.exception_handler(Exception)
    async def failed(request: Request, exc: Exception):
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=500)

    # -- management (Ollama-style) ---------------------------------------------------------

    @app.get("/api/version")
    def version():
        return {"version": VERSION}

    @app.get("/api/tags")
    def tags():
        return {"models": manager.tags()}

    @app.get("/api/ps")
    def ps():
        return {"models": manager.ps()}

    @app.post("/api/pull")
    async def pull(body: Model):
        spec = manager.spec(body.model_id())
        lock = pull_locks.setdefault(spec.id, asyncio.Lock())

        async def events():
            if lock.locked():
                yield {"status": f"waiting for another download of {spec.id}"}
            async with lock:  # a second pull of the same model waits, then finds it installed
                async for event in puller(spec, manager.store):
                    yield event

        return StreamingResponse(ndjson(events()), media_type="application/x-ndjson")

    @app.delete("/api/delete")
    def delete(body: Model):
        model = body.model_id()
        manager.delete(model)
        return {"status": "success", "model": model}

    @app.post("/api/unload")
    def unload(body: Model):
        model = body.model_id()
        return {"status": "unloaded" if manager.unload(model) else "not loaded", "model": model}

    # -- inference --------------------------------------------------------------------------

    @app.post("/v1/audio/transcriptions")
    def transcriptions(
        file: Annotated[UploadFile, File()],
        model: Annotated[str, Form()] = "breeze-asr-25",
        language: Annotated[str | None, Form()] = None,
        prompt: Annotated[str | None, Form()] = None,
        keep_alive: Annotated[str | None, Form()] = None,
    ):
        start = time.perf_counter()
        keep, language = parse_keep_alive(keep_alive), normalize_language(language)
        manager.require_installed(manager.spec(model, "asr"))  # fail before copying the upload
        with saved_upload(file) as path:
            out = manager.run(model, "asr", keep, audio=path, language=language, prompt=prompt)
        return {**out, "model": model, "seconds": round(time.perf_counter() - start, 2)}

    @app.post("/v1/audio/translations")
    def translations(
        file: Annotated[UploadFile, File()],
        model: Annotated[str, Form()] = "whisper-large-v3",
        language: Annotated[str | None, Form()] = None,
        prompt: Annotated[str | None, Form()] = None,
        keep_alive: Annotated[str | None, Form()] = None,
    ):
        """Speech in any language -> English text (OpenAI's translations endpoint): Whisper's
        task="translate", or an instruction for audio LLMs (Gemma 4)."""
        start = time.perf_counter()
        keep, language = parse_keep_alive(keep_alive), normalize_language(language)
        manager.require_installed(manager.spec(model, "asr"))
        with saved_upload(file) as path:
            out = manager.run(model, "asr", keep, audio=path, language=language, prompt=prompt,
                              mode="translate")  # fmt: skip
        return {**out, "model": model, "seconds": round(time.perf_counter() - start, 2)}

    @app.post("/v1/audio/diarize")
    def diarize(
        file: Annotated[UploadFile, File()],
        model: Annotated[str, Form()] = "speaker-diarization",
        num_speakers: Annotated[int | None, Form(ge=1, le=20)] = None,
        threshold: Annotated[float | None, Form(gt=0, lt=2)] = None,
        keep_alive: Annotated[str | None, Form()] = None,
    ):
        """Speaker diarization: {"segments": [{"start", "end", "speaker"}], "speakers",
        "duration"}; speakers are numbered 0, 1, ... in order of first appearance."""
        start = time.perf_counter()
        keep = parse_keep_alive(keep_alive)
        manager.require_installed(manager.spec(model, "diarize"))
        with saved_upload(file) as path:
            out = manager.run(model, "diarize", keep, audio=path, num_speakers=num_speakers,
                              threshold=threshold)  # fmt: skip
        return {**out, "model": model, "seconds": round(time.perf_counter() - start, 2)}

    @app.post("/v1/audio/stream")
    def stream(
        file: Annotated[UploadFile, File()],
        model: Annotated[str, Form()] = "zipformer-zh-en-stream",
        frame_ms: Annotated[int, Form(ge=20, le=2000)] = 160,
        pace: Annotated[str, Form(pattern="^(fast|realtime)$")] = "fast",
        keep_alive: Annotated[str | None, Form()] = None,
    ):
        """Run a streaming model over a recording fed in `frame_ms` frames, as a microphone
        would. NDJSON: one line per text change {"t_audio", "t_wall", "text", "final", "seg"}
        (asr_streaming explains the fields; with pace="fast" t_wall is simulated from the
        measured compute per frame, with "realtime" the feed waits for each frame), then
        {"done": true, "duration", "compute_s", "frames", "append_only", "load_seconds", ...}.
        Closing the connection stops the decode."""
        import numpy as np

        start = time.perf_counter()
        keep = parse_keep_alive(keep_alive)
        manager.require_installed(manager.spec(model, "stream"))
        with saved_upload(file) as path:
            # numpy now: an MLX array made in this thread cannot be used in the worker thread
            # (MLX streams are per thread).
            samples = np.array(decode_audio(path), dtype=np.float32)
        lines: queue.Queue = queue.Queue()
        cancel = threading.Event()

        def work():
            try:
                with manager.using(model, "stream", keep) as (backend, handle, load_seconds):
                    out = backend.run(handle, samples, frame_s=frame_ms / 1000, pace=pace,
                                      cancel=cancel, emit=lines.put)  # fmt: skip
                out.pop("events")
                seconds = round(time.perf_counter() - start, 2)
                lines.put({"done": True, **out, "model": model, "seconds": seconds,
                           "load_seconds": round(load_seconds, 2)})  # fmt: skip
            except Exception as exc:
                message = exc.message if isinstance(exc, GatewayError) else f"{exc}"
                lines.put({"error": message})
            finally:
                lines.put(None)

        threading.Thread(target=work, name=f"stream-{model}", daemon=True).start()

        def body():
            try:
                while (item := lines.get()) is not None:
                    yield json.dumps(item, ensure_ascii=False) + "\n"
            finally:  # the client went away: stop decoding
                cancel.set()

        return StreamingResponse(body(), media_type="application/x-ndjson")

    @app.websocket("/v1/audio/stream/ws")
    async def stream_ws(websocket: WebSocket, model: str = "zipformer-zh-en-stream",
                        keep_alive: str | None = None, s2tw: bool = False):  # fmt: skip
        """Live captions from the microphone. Query: `model` (a streaming model), `s2tw`
        (also send each text converted to Traditional Chinese as "display"). The client
        sends binary 16 kHz mono PCM16 frames (~100 ms) and the text {"type": "stop"} when
        done; the server answers {"type": "ready", ...} once the model is loaded, then the
        events of gateway/live.py, ending with {"type": "done", ...}, and closes. Closing the
        socket at any time ends the session."""
        if not local_origin(websocket.headers.get("origin")):
            await websocket.close(code=1008, reason="only pages on this machine may connect")
            return
        await websocket.accept()
        try:
            keep = parse_keep_alive(keep_alive)
            manager.require_installed(manager.spec(model, "stream"))
        except GatewayError as exc:
            await websocket.send_json({"type": "error", "error": exc.message})
            await websocket.close()
            return
        loop = asyncio.get_running_loop()
        inbox: queue.Queue = queue.Queue()
        outbox: asyncio.Queue = asyncio.Queue()

        def emit(item) -> None:
            loop.call_soon_threadsafe(outbox.put_nowait, item)

        def work() -> None:
            # One thread for the whole session: MLX arrays belong to the thread that made them.
            try:
                with manager.using(model, "stream", keep) as (backend, handle, load_seconds):
                    display = None
                    if s2tw:
                        import opencc

                        display = opencc.OpenCC("s2tw").convert
                    live = LiveSession(backend.session(handle), display=display)
                    emit({"type": "ready", "model": model, "append_only": live.append_only,
                          "load_seconds": round(load_seconds, 2)})  # fmt: skip
                    while (item := inbox.get()) is not None:
                        for event in live.feed(*item):
                            emit(event)
                    for event in live.stop():
                        emit(event)
            except Exception as exc:
                emit({"type": "error", "error": exc.message if isinstance(exc, GatewayError)
                      else f"{type(exc).__name__}: {exc}"})  # fmt: skip
            finally:
                emit(None)

        threading.Thread(target=work, name=f"live-{model}", daemon=True).start()

        async def receive() -> None:
            try:
                while True:
                    message = await websocket.receive()
                    if message["type"] == "websocket.disconnect":
                        break
                    if message.get("bytes"):
                        inbox.put((message["bytes"], time.perf_counter()))
                    elif message.get("text") and '"stop"' in message["text"]:
                        break
            except Exception:  # the socket went away
                pass
            finally:
                inbox.put(None)

        receiver = asyncio.create_task(receive())
        open_socket = True
        while (event := await outbox.get()) is not None:
            if open_socket:
                try:
                    await websocket.send_json(event)
                except Exception:
                    open_socket = False  # keep draining so the worker can finish
        receiver.cancel()
        if open_socket:
            try:
                await websocket.close()
            except Exception:
                pass

    @app.websocket("/v1/voice/agent/ws")
    async def voice_ws(websocket: WebSocket, asr: str = "zipformer-zh-en-stream",
                       llm: str = "gemma4:12b", voice: str = "Meijia",
                       rate: int = 0, net: bool = False, barge: bool = True, s2tw: bool = True,
                       keep_alive: str | None = None, engine: str = "cascade",
                       captions: bool = True, hear: bool = False):  # fmt: skip
        """Live voice assistant: microphone PCM in, captions, assistant speech (WAV), tool
        progress and timings out (gateway/voice.py); `engine` picks the cascade or a single
        speech model (local_llm/voice_omni.py)."""
        if not local_origin(websocket.headers.get("origin")):
            await websocket.close(code=1008, reason="only pages on this machine may connect")
            return
        try:
            keep = parse_keep_alive(keep_alive)
        except GatewayError:
            keep = None
        await voice_agent_ws(websocket, manager, asr, llm, voice, rate, net, barge, s2tw, keep,
                             engine=engine, captions=captions, hear=hear)  # fmt: skip

    @app.post("/v1/audio/speech")
    def speech(body: SpeechRequest):
        """Text to speech with a platform voice (macOS, neural, cloud): 16-bit mono WAV, cached
        by text, voice, rate and sample rate (local_llm/tts.py)."""
        from fastapi.responses import Response

        from local_llm import tts

        start = time.perf_counter()
        try:
            path = tts.synthesize(body.input, body.voice, body.rate, sample_rate=body.sample_rate)
        except ValueError as exc:
            raise GatewayError(400, str(exc)) from exc
        except RuntimeError as exc:
            raise GatewayError(503, str(exc)) from exc
        return Response(path.read_bytes(), media_type="audio/wav",
                        headers={"X-Seconds": f"{time.perf_counter() - start:.3f}"})  # fmt: skip

    @app.post("/v1/audio/speech/timed")
    def speech_timed(body: SpeechRequest):
        """The text sentence by sentence, as the voice assistant speaks, never from the cache:
        timings (tts.timed_sentences) and the joined audio as base64 WAV."""
        import base64

        from local_llm import tts

        try:
            out = tts.timed_sentences(body.input, body.voice, body.rate)
        except ValueError as exc:
            raise GatewayError(400, str(exc)) from exc
        path = Path(out.pop("path"))
        out["wav"] = base64.b64encode(path.read_bytes()).decode()
        path.unlink(missing_ok=True)
        return out

    @app.post("/v1/ocr")
    def ocr_read(
        file: Annotated[UploadFile, File()],
        model: Annotated[str, Form()] = "ocrmac",
    ):
        """The text of an image, read on this Mac (macOS Vision). The OCR models in Ollama
        (MiniCPM-OCR, GLM-OCR, DeepSeek-OCR) are called on Ollama itself."""
        from local_llm import ocr

        if model not in ocr.GATEWAY_ENGINES:
            known = sorted(ocr.GATEWAY_ENGINES)
            raise GatewayError(400, f"unknown OCR model {model}; served here: {known}")
        start = time.perf_counter()
        text = ocr.read_local(model, file.file.read())
        return {"model": model, "text": text, "seconds": round(time.perf_counter() - start, 3)}

    @app.post("/v1/ocr/score")
    def ocr_score(
        file: Annotated[UploadFile, File()],
        text: Annotated[str, Form()],
    ):
        """MiniCPM-OCR Jev: the probability (0-1) that `text` is exactly what the image (one
        region: a paragraph, a title) shows."""
        from local_llm import ocr

        ok, why = ocr.jev_available()
        if not ok:
            raise GatewayError(503, why)
        start = time.perf_counter()
        score = ocr.jev_score(file.file.read(), text)
        return {"model": "minicpm-ocr-jev", "score": round(score, 4),
                "seconds": round(time.perf_counter() - start, 3)}  # fmt: skip

    @app.post("/v1/rerank")
    def rerank(body: RerankRequest):
        start = time.perf_counter()
        out = manager.run(body.model, "rerank", parse_keep_alive(body.keep_alive),
                          query=body.query, documents=body.documents)  # fmt: skip
        ranked = sorted(enumerate(out["scores"]), key=lambda pair: pair[1], reverse=True)
        results = [
            {"index": i, "relevance_score": round(score, 6), "document": body.documents[i]}
            for i, score in ranked[: body.top_n]
        ]
        return {
            "model": body.model,
            "results": results,
            "seconds": round(time.perf_counter() - start, 3),
            "load_seconds": out["load_seconds"],
        }

    @app.post("/v1/documents/parse")
    def parse_document(
        file: Annotated[UploadFile, File()],
        model: Annotated[str, Form()] = "docling",
        keep_alive: Annotated[str | None, Form()] = None,
    ):
        start = time.perf_counter()
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in DOCUMENT_TYPES:
            raise GatewayError(
                415, f"unsupported file type {suffix or '(none)'}; use {sorted(DOCUMENT_TYPES)}"
            )
        keep = parse_keep_alive(keep_alive)
        manager.require_installed(manager.spec(model, "doc"))
        with saved_upload(file) as path:
            out = manager.run(model, "doc", keep, document=path)
        return {**out, "model": model, "seconds": round(time.perf_counter() - start, 2)}

    return app
