"""HTTP API of the model gateway (FastAPI).

Ollama-style management: GET /api/version, /api/tags, /api/ps; POST /api/pull (NDJSON progress),
DELETE /api/delete, POST /api/unload. Inference, OpenAI-style where a convention exists:
POST /v1/audio/transcriptions, /v1/rerank, /v1/documents/parse. Errors are {"error": "..."}.

Endpoints are plain `def` functions, so FastAPI runs the blocking model work in its thread
pool and the event loop stays free for /api/ps, progress streams and other requests.
"""

import asyncio
import json
import re
import tempfile
import threading
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException

from . import MAX_UPLOAD_BYTES, VERSION, download
from .backends import BACKENDS
from .manager import GatewayError, Manager
from .registry import Store, load_specs

REAP_INTERVAL_S = 2.0
DOCUMENT_TYPES = {".pdf", ".docx", ".pptx", ".html", ".htm", ".md", ".png", ".jpg", ".jpeg",
                  ".tif", ".tiff", ".bmp", ".webp"}  # fmt: skip


class ModelRequest(BaseModel):
    model: str | None = None
    name: str | None = None  # Ollama's older field name

    def model_id(self) -> str:
        if not (self.model or self.name):
            raise GatewayError(400, 'missing "model"')
        return self.model or self.name


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
    manager = manager or Manager(load_specs(), BACKENDS, Store())
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
    async def pull(body: ModelRequest):
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
    def delete(body: ModelRequest):
        model = body.model_id()
        manager.delete(model)
        return {"status": "success", "model": model}

    @app.post("/api/unload")
    def unload(body: ModelRequest):
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
