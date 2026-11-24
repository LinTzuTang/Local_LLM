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
