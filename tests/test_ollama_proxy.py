import json

import httpx
from starlette.applications import Starlette
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from local_llm import ollama_proxy as op


def fake_ollama():
    async def chat(request):
        body = await request.json()

        async def gen():
            yield json.dumps({"message": {"content": "你"}, "done": False}) + "\n"
            yield json.dumps({"message": {"content": "好"}, "done": False}) + "\n"
            yield (
                json.dumps(
                    {"done": True, "prompt_eval_count": 12, "eval_count": 2, "model": body["model"]}
                )
                + "\n"
            )

        return StreamingResponse(gen(), media_type="application/x-ndjson")

    async def v1(request):
        async def gen():
            yield 'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
            yield 'data: {"choices":[],"usage":{"prompt_tokens":30,"completion_tokens":5}}\n\n'
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    async def tags(request):
        return JSONResponse({"models": [{"name": "m:1"}]})

    return Starlette(
        routes=[
            Route("/api/chat", chat, methods=["POST"]),
            Route("/v1/chat/completions", v1, methods=["POST"]),
            Route("/api/tags", tags),
        ]
    )


def client(tmp_path):
    app = op.create_app(
        "http://upstream", tmp_path, transport=httpx.ASGITransport(app=fake_ollama())
    )
    return TestClient(app)


def rows(tmp_path):
    return [
        json.loads(line) for f in tmp_path.glob("*.jsonl") for line in f.read_text().splitlines()
    ]


def test_streams_through_and_logs_counts_not_content(tmp_path):
    with client(tmp_path) as c:
        r = c.post(
            "/api/chat",
            json={"model": "gemma4:31b", "messages": [{"role": "user", "content": "秘密問題"}]},
            headers={"X-Lab-User": "alice"},
        )
        assert r.status_code == 200
        assert [json.loads(x)["done"] for x in r.text.splitlines()] == [False, False, True]
    (row,) = rows(tmp_path)
    assert row["model"] == "gemma4:31b" and row["user"] == "alice" and row["stream"] is True
    assert (row["prompt_tokens"], row["completion_tokens"]) == (12, 2)
    assert row["status"] == 200 and row["seconds"] >= 0 and row["ttfb_s"] is not None
    assert "秘密" not in json.dumps(row, ensure_ascii=False) and "你" not in json.dumps(
        row, ensure_ascii=False
    )


def test_openai_route_usage_and_unlogged_routes(tmp_path):
    with client(tmp_path) as c:
        r = c.post(
            "/v1/chat/completions",
            json={"model": "qwen3.6:35b-a3b", "stream": True, "messages": []},
        )
        assert "[DONE]" in r.text
        assert c.get("/api/tags").json()["models"][0]["name"] == "m:1"
    (row,) = rows(tmp_path)
    assert row["path"] == "/v1/chat/completions" and (
        row["prompt_tokens"],
        row["completion_tokens"],
    ) == (30, 5)
    assert row["user"] is None


def test_upstream_down_is_502_and_logged(tmp_path):
    def refuse(request):
        raise httpx.ConnectError("refused")

    app = op.create_app("http://upstream", tmp_path, transport=httpx.MockTransport(refuse))
    with TestClient(app) as c:
        assert c.post("/api/chat", json={"model": "x"}).status_code == 502
    assert rows(tmp_path)[0]["status"] == 502


def test_report_peak_and_accounts():
    rs = [
        {
            "t": "2026-11-25T10:00:00.000",
            "path": "/api/chat",
            "seconds": 10,
            "ttfb_s": 0.5,
            "user": "a",
            "completion_tokens": 100,
        },
        {
            "t": "2026-11-25T10:00:05.000",
            "path": "/api/chat",
            "seconds": 10,
            "ttfb_s": 1.0,
            "user": "b",
            "completion_tokens": 50,
        },
        {
            "t": "2026-11-25T10:00:20.000",
            "path": "/api/chat",
            "seconds": 1,
            "ttfb_s": 0.2,
            "user": None,
            "completion_tokens": 5,
        },
        {
            "t": "2026-11-25T10:00:06.000",
            "path": "/api/embed",
            "seconds": 1,
            "ttfb_s": 0.1,
            "user": "a",
        },
    ]
    (d,) = op.report(rs)
    assert (
        d["requests"] == 3
        and d["accounts"] == 2
        and d["most_at_once"] == 2
        and d["completion_tokens"] == 155
    )
    assert op.report(rs, users_only=True)[0]["requests"] == 2


def test_token_counts_plain_json():
    assert op.token_counts(b'{"usage": {"prompt_tokens": 3, "completion_tokens": 4}}') == (3, 4)
    assert op.token_counts(b"not json") == (None, None)


def test_users_since_load():
    """Who has used a model since it was last loaded: a long pause or an unload starts over."""
    import datetime as dt

    from local_llm.ollama_proxy import users_since_load

    now = dt.datetime(2026, 11, 26, 12, 0, 0)

    def row(minutes_ago, user=None, model="gemma4:31b", **extra):
        t = (now - dt.timedelta(minutes=minutes_ago)).isoformat(timespec="milliseconds")
        return {"t": t, "path": "/api/chat", "model": model, "user": user, "seconds": 5,
                "prompt_tokens": 10, "completion_tokens": 10} | extra  # fmt: skip

    # Only alice since it was loaded (the evaluation's request was 3 hours ago, before a pause).
    rows = [row(180), row(20, "alice"), row(2, "alice")]
    assert users_since_load("gemma4:31b", rows, now) == {"alice"}
    # An evaluation (no account) is using it too.
    rows = [row(10), row(2, "alice")]
    assert users_since_load("gemma4:31b", rows, now) == {None, "alice"}
    # Another account, and the ":latest" tag written or not.
    rows = [row(5, "bob", model="muse"), row(1, "alice", model="muse:latest")]
    assert users_since_load("muse:latest", rows, now) == {"alice", "bob"}
    # An unload request: what came before is another load.
    unload = row(8, path="/api/generate", prompt_tokens=None, completion_tokens=None)
    assert users_since_load("gemma4:31b", [row(9), unload, row(3, "alice")], now) == {"alice"}
    # Nothing logged lately: the log cannot tell who loaded it.
    assert users_since_load("gemma4:31b", [row(120, "alice")], now) == set()
    assert users_since_load("gemma4:31b", [], now) == set()


def _fake_server(loaded: list[dict], files: dict[str, int]):
    """Ollama's /api/ps, /api/tags and unload (keep_alive 0) over a list of loaded models."""
    unloaded = []

    async def ps(request):
        return JSONResponse({"models": loaded})

    async def tags(request):
        return JSONResponse({"models": [{"name": n, "size": s} for n, s in files.items()]})

    async def generate(request):
        body = await request.json()
        if body.get("keep_alive") == 0:
            unloaded.append(body["model"])
            loaded[:] = [m for m in loaded if m["name"] != body["model"]]
        return JSONResponse({"done": True, "done_reason": "unload"})

    app = Starlette(
        routes=[Route("/api/ps", ps), Route("/api/tags", tags),
                Route("/api/generate", generate, methods=["POST"])]
    )  # fmt: skip
    return app, unloaded


def test_make_room_keeps_the_admins_models(tmp_path, monkeypatch):
    """A user's request never pushes out a model the admin (or a script) is using; it may push out
    models only non-admin accounts use; the admin and scripts are not checked at all."""
    import asyncio
    import datetime as dt

    G = op.GIB
    roles = {"alice": "user", "bob": "user", "boss": "admin"}
    monkeypatch.setattr(op, "role_of", lambda u: roles.get(u) if u else None)
    gpu = (96 * G, 1)
    now = dt.datetime.now()

    def used(model, user):
        t = (now - dt.timedelta(minutes=1)).isoformat(timespec="milliseconds")
        return {"t": t, "path": "/api/chat", "model": model, "user": user, "seconds": 3,
                "prompt_tokens": 5, "completion_tokens": 5}  # fmt: skip

    def run(loaded, files, model, user, rows):
        app, unloaded = _fake_server(loaded, files)
        client = httpx.AsyncClient(base_url="http://up", transport=httpx.ASGITransport(app=app))
        why = asyncio.run(op.make_room(client, model, user, tmp_path, gpu=gpu, rows=rows))
        return why, unloaded

    evaluation = {"name": "nanbeige:latest", "size": 66 * G, "expires_at": "2026-11-26T05:00:00Z"}
    alices = {"name": "gemma4:12b", "size": 12 * G, "expires_at": "2026-11-26T04:00:00Z"}
    files = {"gemma4:31b": 20 * G, "mid:latest": 15 * G, "small:latest": 5 * G}

    # The evaluation (no account) fills the card: alice's 31B would push it out, so no.
    why, unloaded = run([dict(evaluation)], files, "gemma4:31b", "alice",
                        [used("nanbeige", None)])  # fmt: skip
    assert why and "nanbeige:latest" in why and not unloaded
    # A small model still fits next to it.
    assert run([dict(evaluation)], files, "small", "alice", [used("nanbeige", None)]) == (None, [])
    # alice's own 12B makes way for her next model; the evaluation stays.
    why, unloaded = run([dict(evaluation), dict(alices)], files, "mid", "alice",
                        [used("nanbeige", None), used("gemma4:12b", "alice")])  # fmt: skip
    assert why is None and unloaded == ["gemma4:12b"]
    # bob's model makes way too (only the admin's are kept), but not one the admin also used.
    rows = [used("nanbeige", None), used("gemma4:12b", "bob"), used("gemma4:12b", "boss")]
    why, unloaded = run([dict(evaluation), dict(alices)], files, "mid", "alice", rows)
    assert why and not unloaded
    # The admin and scripts are not held back (Ollama decides as before).
    assert run([dict(evaluation)], files, "gemma4:31b", "boss", []) == (None, [])
    assert run([dict(evaluation)], files, "gemma4:31b", None, []) == (None, [])
    # Without a GPU to measure (the Mac), nothing is checked.
    monkeypatch.setattr(op, "gpu_memory", lambda: None)
    app, _ = _fake_server([dict(evaluation)], files)
    client = httpx.AsyncClient(base_url="http://up", transport=httpx.ASGITransport(app=app))
    assert asyncio.run(op.make_room(client, "gemma4:31b", "alice", tmp_path)) is None


def test_refusal_shapes():
    assert json.loads(op.refusal("/api/chat", "full").body) == {"error": "full"}
    body = json.loads(op.refusal("/v1/chat/completions", "full").body)
    assert body["error"]["message"] == "full" and op.refusal("/api/chat", "x").status_code == 503


def test_request_user_tags_only_ollama(monkeypatch):
    """The platform tags every request to Ollama with the account, and nothing else."""
    from local_llm import request_user

    seen = []

    def handler(request):
        seen.append((request.url.host, request.headers.get("x-lab-user")))
        return httpx.Response(200, json={})

    request_user.install("http://127.0.0.1:11434")
    request_user.set_user("alice")
    with httpx.Client(transport=httpx.MockTransport(handler)) as c:
        c.post("http://localhost:11434/api/chat", json={})
        c.post("https://api.example.com/v1/chat", json={})
    request_user.set_user(None)
    assert seen == [("localhost", "alice"), ("api.example.com", None)]
