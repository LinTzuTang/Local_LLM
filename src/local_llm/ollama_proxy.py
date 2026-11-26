"""Request log in front of Ollama: how much the shared platform is really used (to size servers).

The platform reaches Ollama from many places (chat, agents, voice, OCR, ...), all at
http://127.0.0.1:11434. With LOCAL_LLM_REQUEST_LOG=1 in .env, scripts/start_ollama.sh moves Ollama
to port 11433 and runs this proxy on 11434: every request is passed through unchanged (streaming
included), and each model request (chat, generate, embeddings, the OpenAI-style /v1 routes) adds
one line to storage/request_log/YYYY-MM-DD.jsonl:

    t        start time (local, ISO)       path     e.g. /api/chat
    model    the request's "model"         stream   streamed or not
    status   HTTP status                   ttfb_s   seconds until the first byte of the answer
    seconds  seconds until the answer ended
    prompt_tokens / completion_tokens      from Ollama's own counts (eval counts, or "usage")
    user     the X-Lab-User header when the caller passes it (the signed-in account), else null
    agent    the caller's User-Agent, cut short
    unload   true for an unload request (keep_alive 0)

Never the prompt or the answer. `python -m local_llm.ollama_proxy report` sums the log per day:
requests, accounts, the most requests running at the same time, and waiting times.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import functools
import json
import os
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote

import httpx
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route

ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = Path(os.getenv("LOCAL_LLM_REQUEST_LOG_DIR", ROOT / "storage/request_log"))
LISTEN = int(os.getenv("LOCAL_LLM_OLLAMA_PROXY_PORT", "11434"))
UPSTREAM = os.getenv("LOCAL_LLM_OLLAMA_UPSTREAM", "http://127.0.0.1:11433")
LOGGED = {
    "/api/chat",
    "/api/generate",
    "/api/embed",
    "/api/embeddings",
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/embeddings",
    "/v1/responses",
}
HOP = {
    "connection",
    "keep-alive",
    "proxy-connection",
    "transfer-encoding",
    "upgrade",
    "te",
    "trailer",
    "host",
    "content-length",
    "content-encoding",
}
TAIL = 16_384  # bytes of the answer kept to read the token counts at its end


def token_counts(tail: bytes) -> tuple[int | None, int | None]:
    """Prompt and completion tokens from the end of an answer: Ollama's last NDJSON object
    (prompt_eval_count / eval_count), an SSE "usage" chunk, or a plain JSON body with either."""
    for line in reversed(tail.decode(errors="replace").splitlines()):
        line = line.strip()
        if line.startswith("data:"):
            line = line[5:].strip()
        if not line.startswith("{"):
            continue
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if "eval_count" in d or "prompt_eval_count" in d:
            return d.get("prompt_eval_count"), d.get("eval_count")
        u = d.get("usage")
        if isinstance(u, dict):
            return u.get("prompt_tokens", u.get("input_tokens")), u.get(
                "completion_tokens", u.get("output_tokens")
            )
    return None, None


def write_line(row: dict, log_dir: Path = LOG_DIR) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / f"{row['t'][:10]}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- keep the admin's models loaded
# A model that does not fit next to the loaded ones makes Ollama unload one of them, possibly an
# evaluation the admin is running. Before a signed-in user's request loads a model, the proxy
# frees room by unloading only models that just non-admin accounts are using; if that is not
# enough, the request is turned down instead. Off with LOCAL_LLM_PROTECT_MODELS=0, and when the
# GPU's memory cannot be read (the Mac: everyone is the admin there anyway).
PROTECT = os.getenv("LOCAL_LLM_PROTECT_MODELS", "1") != "0"
GIB = 1024**3


@functools.cache
def gpu_memory() -> tuple[int, int] | None:
    """(total bytes of all NVIDIA GPUs, number of GPUs), or None without nvidia-smi."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.split()  # fmt: skip
        return sum(int(x) for x in out) * 1024**2, len(out)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def role_of(user: str | None) -> str | None:
    """The account's role ("admin" / "user"); None for a request without an account."""
    if not user:
        return None
    from . import auth

    found = auth.Users().get(unquote(user))
    return found["role"] if found else "user"


def _sizes_file(log_dir: Path) -> Path:
    return log_dir / "model_sizes.json"


def _sizes(log_dir: Path) -> dict:
    try:
        return json.loads(_sizes_file(log_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


async def make_room(
    client: httpx.AsyncClient, model: str | None, user: str | None, log_dir: Path,
    gpu: tuple[int, int] | None = None, rows: list[dict] | None = None,
) -> str | None:  # fmt: skip
    """Before a non-admin account's request loads `model`: None when it fits (after unloading
    models only non-admin accounts use, if needed), else why it was turned down."""
    gpu = gpu or gpu_memory()
    if not (PROTECT and model and gpu) or role_of(user) in (None, "admin"):
        return None
    try:
        loaded = (await client.get("/api/ps")).json().get("models") or []
    except (httpx.HTTPError, ValueError):
        return None
    name = _model_name(model)
    sizes = _sizes(log_dir)
    for m in loaded:  # remember how big each model is once loaded (weights + context)
        sizes[_model_name(m["name"])] = m.get("size_vram") or m.get("size") or 0
    with contextlib.suppress(OSError):
        _sizes_file(log_dir).write_text(json.dumps(sizes), encoding="utf-8")
    if any(_model_name(m["name"]) == name for m in loaded):
        return None
    need = sizes.get(name)
    if not need:  # never loaded here: the file, plus room for the context
        try:
            tags = (await client.get("/api/tags")).json().get("models") or []
        except (httpx.HTTPError, ValueError):
            tags = []
        file = next((t.get("size", 0) for t in tags if _model_name(t.get("name")) == name), 0)
        need = int(file * 1.15) + 3 * GIB
    total, gpus = gpu
    free = int(total * 0.95) - sum(m.get("size_vram") or m.get("size") or 0 for m in loaded)
    slots = int(os.getenv("OLLAMA_MAX_LOADED_MODELS") or 3 * gpus) - len(loaded)
    if need <= free and slots > 0:
        return None
    rows = rows if rows is not None else load(log_dir, days=2)

    def users_only(m: dict) -> bool:  # only non-admin accounts have used it since it loaded
        who = users_since_load(m["name"], rows)
        return bool(who) and all(u is not None and role_of(u) != "admin" for u in who)

    spare = sorted((m for m in loaded if users_only(m)), key=lambda m: m.get("expires_at") or "")
    for m in spare:
        with contextlib.suppress(httpx.HTTPError):
            await client.post("/api/generate", json={"model": m["name"], "keep_alive": 0})
            free += m.get("size_vram") or m.get("size") or 0
            slots += 1
            if need <= free and slots > 0:
                await _wait_unloaded(client, {x["name"] for x in spare})
                return None
    busy = "、".join(m["name"] for m in loaded if m not in spare) or "—"
    return (
        f"伺服器的顯示卡記憶體不夠載入 {model}：其他模型（{busy}）正由管理者使用中，不能先卸載。"
        f"請改用已經載入的模型，或稍後再試。"
        f" Not enough GPU memory to load {model}: the loaded models ({busy}) are in use by the "
        "administrator and cannot be unloaded. Use a model that is already loaded, or try later."
    )


async def _wait_unloaded(client: httpx.AsyncClient, names: set[str], seconds: float = 15) -> None:
    """Ollama answers an unload at once and frees the memory just after."""
    import asyncio

    end = time.monotonic() + seconds
    while time.monotonic() < end:
        try:
            now = {m["name"] for m in (await client.get("/api/ps")).json().get("models") or []}
        except (httpx.HTTPError, ValueError):
            return
        if not (now & names):
            return
        await asyncio.sleep(0.5)


def refusal(path: str, message: str) -> Response:
    """Turned down in the shape the caller understands (Ollama's or OpenAI's error)."""
    if path.startswith("/v1/"):
        body = {"error": {"message": message, "type": "server_busy"}}
    else:
        body = {"error": message}
    return Response(
        json.dumps(body, ensure_ascii=False), status_code=503, media_type="application/json"
    )


def create_app(
    upstream: str = UPSTREAM,
    log_dir: Path = LOG_DIR,
    transport: httpx.AsyncBaseTransport | None = None,
):
    client = httpx.AsyncClient(
        base_url=upstream, timeout=httpx.Timeout(None, connect=10), transport=transport
    )

    async def proxy(request: Request) -> Response:
        body = await request.body()
        path = request.url.path
        logged = request.method == "POST" and path in LOGGED
        row = None
        if logged:
            try:
                req = json.loads(body or b"{}")
            except ValueError:
                req = {}
            row = {
                "t": dt.datetime.now().isoformat(timespec="milliseconds"),
                "path": path,
                "model": req.get("model") if isinstance(req, dict) else None,
                "stream": bool(req.get("stream", path.startswith("/api/")))
                if isinstance(req, dict)
                else None,
                "user": request.headers.get("x-lab-user"),
                "agent": (request.headers.get("user-agent") or "")[:40],
            }
            if isinstance(req, dict) and req.get("keep_alive") in (0, "0", "0s", "0m"):
                row["unload"] = True  # an unload request (a new load starts after it)
        if row and row["model"] and not row.get("unload"):
            why = await make_room(client, row["model"], row["user"], log_dir)
            if why:
                none = {"ttfb_s": None, "prompt_tokens": None, "completion_tokens": None}
                write_line(row | none | {"status": 503, "seconds": 0.0}, log_dir)
                return refusal(path, why)
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP}
        t0 = time.perf_counter()
        try:
            up = await client.send(
                client.build_request(
                    request.method, path, params=request.query_params, headers=headers, content=body
                ),
                stream=True,
            )
        except httpx.HTTPError as e:
            if row:
                write_line(
                    row
                    | {
                        "status": 502,
                        "ttfb_s": None,
                        "seconds": round(time.perf_counter() - t0, 3),
                        "prompt_tokens": None,
                        "completion_tokens": None,
                    },
                    log_dir,
                )
            return Response(f"Ollama not reachable at {upstream}: {e}", status_code=502)
        state = {"first": None, "tail": b""}

        async def relay():
            async for chunk in up.aiter_raw():
                if state["first"] is None:
                    state["first"] = time.perf_counter()
                if row:
                    state["tail"] = (state["tail"] + chunk)[-TAIL:]
                yield chunk

        async def done():
            await up.aclose()
            if row:
                p, c = token_counts(state["tail"])
                end = time.perf_counter()
                write_line(
                    row
                    | {
                        "status": up.status_code,
                        "ttfb_s": round(state["first"] - t0, 3) if state["first"] else None,
                        "seconds": round(end - t0, 3),
                        "prompt_tokens": p,
                        "completion_tokens": c,
                    },
                    log_dir,
                )

        out = {k: v for k, v in up.headers.items() if k.lower() not in HOP}
        return StreamingResponse(
            relay(), status_code=up.status_code, headers=out, background=BackgroundTask(done)
        )

    methods = ["GET", "POST", "PUT", "DELETE", "HEAD", "PATCH", "OPTIONS"]

    @contextlib.asynccontextmanager
    async def lifespan(app):
        yield
        await client.aclose()

    return Starlette(routes=[Route("/{path:path}", proxy, methods=methods)], lifespan=lifespan)


# ---------------------------------------------------------------- report
def load(log_dir: Path = LOG_DIR, days: int = 30) -> list[dict]:
    rows = []
    for f in sorted(log_dir.glob("*.jsonl"))[-days:]:
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    return rows


def _model_name(model: str | None) -> str:
    name = (model or "").strip()
    return name if ":" in name else name + ":latest"


def _keep_alive_s() -> float:
    """Ollama's keep-alive (OLLAMA_KEEP_ALIVE: "5m", "24h", "300", "-1" = forever)."""
    raw = os.getenv("OLLAMA_KEEP_ALIVE", "5m").strip()
    try:
        unit = {"s": 1, "m": 60, "h": 3600}.get(raw[-1])
        value = float(raw[:-1]) * unit if unit else float(raw)
    except (ValueError, IndexError, KeyError):
        return 300.0
    return float("inf") if value < 0 else value


def users_since_load(
    model: str, rows: list[dict], now: dt.datetime | None = None, gap_s: float | None = None
) -> set[str | None]:
    """Who has used `model` since it was last loaded, from the request log: the accounts, and None
    for a request without one (a script or an evaluation on the machine). Going back from now, a
    pause longer than the keep-alive (at least 30 minutes, to be safe) or an unload request means
    the model was loaded again after it, so earlier requests do not count. Empty when the log
    cannot tell (nothing logged since the model was loaded)."""
    gap = gap_s if gap_s is not None else max(1800.0, _keep_alive_s())
    name, ref = _model_name(model), now or dt.datetime.now()
    mine = sorted((r for r in rows if _model_name(r.get("model")) == name), key=lambda r: r["t"])
    users: set[str | None] = set()
    for r in reversed(mine):
        try:
            start = dt.datetime.fromisoformat(r["t"])
        except (KeyError, ValueError):
            continue
        unload = r.get("unload") or (  # rows logged before "unload" was: a bare /api/generate
            r.get("path") == "/api/generate" and r.get("prompt_tokens") is None
            and r.get("completion_tokens") is None and not r.get("user")
        )  # fmt: skip
        end = start + dt.timedelta(seconds=r.get("seconds") or 0)
        if unload or (ref - end).total_seconds() > gap:
            break
        users.add(unquote(r["user"]) if r.get("user") else None)
        ref = min(ref, start)
    return users


def peak_in_flight(rows: list[dict]) -> tuple[int, str | None]:
    """The most requests running at the same moment, and when."""
    ev = []
    for r in rows:
        if r.get("seconds") is None:
            continue
        s = dt.datetime.fromisoformat(r["t"]).timestamp()
        ev += [(s, 1), (s + r["seconds"], -1)]
    best, now, when = 0, 0, None
    for t, d in sorted(ev, key=lambda x: (x[0], x[1])):
        now += d
        if now > best:
            best, when = now, dt.datetime.fromtimestamp(t).isoformat(timespec="seconds")
    return best, when


def report(rows: list[dict], users_only: bool = False) -> list[dict]:
    """Per day: requests, accounts seen, busiest hour, most at once, waiting times.
    users_only: only requests that carry an account (the platform's pages), not scripts on the
    same machine."""
    by_day = defaultdict(list)
    for r in rows:
        if r.get("path", "").endswith(("embed", "embeddings")) or (
            users_only and not r.get("user")
        ):
            continue
        by_day[r["t"][:10]].append(r)
    out = []
    for day, rs in sorted(by_day.items()):
        hours = defaultdict(int)
        for r in rs:
            hours[r["t"][11:13]] += 1
        peak, when = peak_in_flight(rs)
        ttfb = sorted(r["ttfb_s"] for r in rs if r.get("ttfb_s") is not None)
        secs = [r["seconds"] for r in rs if r.get("seconds") is not None]
        busiest = max(hours.items(), key=lambda x: x[1]) if hours else (None, 0)
        out.append(
            {
                "day": day,
                "requests": len(rs),
                "accounts": len({r["user"] for r in rs if r.get("user")}),
                "busiest_hour": f"{busiest[0]}:00" if busiest[0] else None,
                "busiest_hour_requests": busiest[1],
                "most_at_once": peak,
                "most_at_once_at": when,
                "first_byte_p95_s": ttfb[int(0.95 * (len(ttfb) - 1))] if ttfb else None,
                "seconds_median": statistics.median(secs) if secs else None,
                "completion_tokens": sum(r.get("completion_tokens") or 0 for r in rs),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("serve", help="run the proxy (default)")
    rp = sub.add_parser("report", help="sum the log per day")
    rp.add_argument("--days", type=int, default=30)
    rp.add_argument(
        "--users-only", action="store_true", help="only requests from signed-in accounts"
    )
    a = ap.parse_args()
    if a.cmd == "report":
        for d in report(load(days=a.days), users_only=a.users_only):
            print(json.dumps(d, ensure_ascii=False))
        return
    import uvicorn

    print(
        f"Ollama request log on http://127.0.0.1:{LISTEN} -> {UPSTREAM} · log {LOG_DIR}", flush=True
    )
    uvicorn.run(create_app(), host="127.0.0.1", port=LISTEN, log_level="warning")


if __name__ == "__main__":
    main()
