"""Models in memory: lazy loading, per-model locks, keep_alive and unloading idle models.

Like Ollama: a model loads on its first request and stays loaded for `keep_alive` seconds
after its last one (negative = forever, 0 = unload right after the request). A reaper thread
calls `reap()` every few seconds; a model in the middle of a request is never unloaded.
"""

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from . import DEFAULT_KEEP_ALIVE
from .backends import Backend, free_memory
from .registry import ModelSpec, Store


class GatewayError(Exception):
    """An error with an HTTP status, returned to the client as {"error": message}."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


@dataclass
class Loaded:
    spec: ModelSpec
    handle: Any
    device: str
    loaded_at: float
    load_seconds: float
    expires_at: float | None = None  # None = keep forever
    busy: int = 0  # requests using the model right now
    lock: threading.Lock = field(default_factory=threading.Lock)  # one request at a time


def _iso(ts: float | None) -> str | None:
    return datetime.fromtimestamp(ts).astimezone().isoformat(timespec="seconds") if ts else None


class Manager:
    def __init__(
        self,
        specs: dict[str, ModelSpec],
        backends: dict[str, Backend],
        store: Store,
        keep_alive: float = DEFAULT_KEEP_ALIVE,
        free: Callable[[], None] = free_memory,
    ):
        self.specs, self.backends, self.store = specs, backends, store
        self.keep_alive, self._free = keep_alive, free
        self._loaded: dict[str, Loaded] = {}
        self._guard = threading.Lock()  # protects _loaded and the busy/expiry fields
        self._load_locks = {model_id: threading.Lock() for model_id in specs}

    # -- lookup ---------------------------------------------------------------------------

    def spec(self, model: str, task: str | None = None) -> ModelSpec:
        spec = self.specs.get(model)
        if spec is None:
            available = ", ".join(self.specs)
            raise GatewayError(404, f"model '{model}' not found; available: {available}")
        if task and spec.task != task:
            raise GatewayError(400, f"model '{model}' is a {spec.task} model, not {task}")
        return spec

    def require_installed(self, spec: ModelSpec) -> None:
        if not self.store.installed(spec):
            raise GatewayError(
                404,
                f"model '{spec.id}' is not downloaded; pull it first: "
                f'POST /api/pull {{"model": "{spec.id}"}} (or the download button in the '
                "platform's Model library).",
            )

    def tags(self) -> list[dict]:
        return [
            {
                "id": s.id,
                "task": s.task,
                "installed": self.store.installed(s),
                "size_gb": s.size_gb,
                "loaded": s.id in self._loaded,
                "name": s.name,
                "license": s.license,
                "backend": s.backend,
                "size_bytes": self.store.size_bytes(s),
            }
            for s in self.specs.values()
        ]

    def ps(self) -> list[dict]:
        with self._guard:
            entries = list(self._loaded.values())
        return [
            {
                "id": e.spec.id,
                "task": e.spec.task,
                "expires_at": _iso(e.expires_at),
                "device": e.device,
                "loaded_at": _iso(e.loaded_at),
                "load_seconds": round(e.load_seconds, 2),
                "busy": e.busy > 0,
            }
            for e in entries
        ]

    # -- use ------------------------------------------------------------------------------

    def run(self, model: str, task: str, keep_alive: float | None = None, **inputs) -> dict:
        """Load the model if needed, run one request, then restart its keep_alive timer."""
        with self.using(model, task, keep_alive) as (backend, handle, load_seconds):
            output = backend.run(handle, **inputs)
        return {**output, "load_seconds": round(load_seconds, 2)}

    @contextmanager
    def using(
        self, model: str, task: str, keep_alive: float | None = None
    ) -> Iterator[tuple[Backend, Any, float]]:
        """Hold a loaded model for one request, however long (a streaming session):
        yields (backend, handle, load seconds); the keep_alive timer starts afterwards."""
        spec = self.spec(model, task)
        self.require_installed(spec)
        entry, load_seconds = self._acquire(spec)
        try:
            with entry.lock:
                yield self.backends[spec.backend], entry.handle, load_seconds
        finally:
            self._done(entry, self.keep_alive if keep_alive is None else keep_alive)

    def _acquire(self, spec: ModelSpec) -> tuple[Loaded, float]:
        with self._load_locks[spec.id]:  # concurrent first requests load the model once
            with self._guard:
                if entry := self._loaded.get(spec.id):
                    entry.busy += 1
                    return entry, 0.0
            start = time.perf_counter()
            try:
                handle, device = self.backends[spec.backend].load(spec, self.store.model_path(spec))
            except Exception as exc:
                self._free()
                raise GatewayError(500, f"loading {spec.id} failed: {exc}") from exc
            seconds = time.perf_counter() - start
            entry = Loaded(spec, handle, device, time.time(), seconds, busy=1)
            with self._guard:
                self._loaded[spec.id] = entry
            return entry, seconds

    def _done(self, entry: Loaded, keep_alive: float) -> None:
        with self._guard:
            entry.busy -= 1
            entry.expires_at = None if keep_alive < 0 else time.time() + keep_alive
        if keep_alive == 0:
            self.reap()

    # -- unload ---------------------------------------------------------------------------

    def unload(self, model: str) -> bool:
        self.spec(model)
        with self._guard:
            entry = self._loaded.get(model)
            if entry is None:
                return False
            if entry.busy:
                raise GatewayError(409, f"model '{model}' is busy with a request; try again")
            del self._loaded[model]
        self._release(entry)
        return True

    def reap(self, now: float | None = None) -> list[str]:
        """Unload every idle model whose keep_alive has run out."""
        now = time.time() if now is None else now
        with self._guard:
            expired = [
                e
                for e in self._loaded.values()
                if not e.busy and e.expires_at is not None and e.expires_at <= now
            ]
            for e in expired:
                del self._loaded[e.spec.id]
        for e in expired:
            self._release(e)
        return [e.spec.id for e in expired]

    def _release(self, entry: Loaded) -> None:
        try:
            self.backends[entry.spec.backend].release(entry.handle)
        finally:
            entry.handle = None
            self._free()

    def delete(self, model: str) -> None:
        spec = self.spec(model)
        if not self.store.installed(spec) and not self.store.manifest(spec):
            raise GatewayError(404, f"model '{model}' is not downloaded")
        self.unload(model)
        self.store.remove(spec)
