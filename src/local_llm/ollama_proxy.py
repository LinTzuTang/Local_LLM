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

Never the prompt or the answer. `python -m local_llm.ollama_proxy report` sums the log per day:
requests, accounts, the most requests running at the same time, and waiting times.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path

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
