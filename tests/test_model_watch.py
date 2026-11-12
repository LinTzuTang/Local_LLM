from datetime import UTC, datetime, timedelta

import httpx

from local_llm import model_watch

NOW = datetime(2026, 11, 12, tzinfo=UTC)


def item(name: str, updated: str, desc: str = "A model.") -> str:
    return (
        f'<li class="x"> <a href="/library/{name}" class="g">'
        f' <h2 title="{name}"><span>{name}</span></h2>'
        f' <span class="t" title="Updated {updated}">2w</span>'
        '<span class="rounded-md bg-indigo-50 px-2">tools</span>'
        f'<p class="mt-1">{desc}</p>'
        '<span class="i" title="75,368 downloads"><span>75.4K</span></span>'
        '<span class="rounded-md bg-[#ddf4ff] px-2">1.7b</span> </a> </li>'
    )


OLLAMA_PAGE = (
    "<ul>"
    + "".join(
        [
            item("brand-new", "November 9, 2026", "Fresh &amp; small."),
            item("qwen3", "November 8, 2026"),  # already in the catalog
            item("old-model", "August 1, 2026"),  # too old for the first check
        ]
    )
    + "</ul>"
)


def hf_model(repo: str, created: datetime, likes: int = 100, tags: list[str] | None = None) -> dict:
    return {"id": repo, "createdAt": created.isoformat().replace("+00:00", "Z"), "likes": likes,
            "downloads": 10, "tags": tags or []}  # fmt: skip


def client() -> httpx.Client:
    recent = NOW - timedelta(days=10)

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.host == "ollama.com":
            page = request.url.params.get("p")
            return httpx.Response(200, text=OLLAMA_PAGE if page == "1" else "<ul></ul>")
        if request.url.params.get("pipeline_tag") == "automatic-speech-recognition":
            return httpx.Response(200, json=[
                hf_model("acme/asr-zh", recent, tags=["zh", "en"]),
                hf_model("acme/asr-hindi", recent, tags=["hi"]),  # names its languages, no Chinese
                hf_model("acme/asr-zh-gguf", recent,
                         tags=["zh", "base_model:quantized:acme/asr-zh"]),
                hf_model("acme/asr-few-likes", recent, likes=3),
                hf_model("acme/asr-old", NOW - timedelta(days=200)),
                hf_model("MediaTek-Research/Breeze-ASR-25", recent),  # already in the catalog
            ])  # fmt: skip
        return httpx.Response(200, json=[])

    return httpx.Client(transport=httpx.MockTransport(handle))


def test_parse_ollama():
    rows = model_watch.parse_ollama(OLLAMA_PAGE)
    assert [r["id"] for r in rows] == ["brand-new", "qwen3", "old-model"]
    first = rows[0]
    assert first["summary"] == "Fresh & small."
    assert first["caps"] == ["tools"] and first["sizes"] == ["1.7b"]
    assert first["downloads"] == 75368 and first["date"].startswith("2026-11-09")


def test_check_reports_only_new_models_once(tmp_path):
    path = tmp_path / "watch.json"
    with client() as c:
        new = model_watch.check(path, client=c, now=NOW)
        assert sorted(m["id"] for m in new) == ["acme/asr-zh", "brand-new"]
        assert model_watch.check(path, client=c, now=NOW + timedelta(days=7)) == []
    state = model_watch.load(path)
    assert [m["id"] for m in model_watch.unread(state)] == ["brand-new", "acme/asr-zh"]
    model_watch.acknowledge(path)
    assert model_watch.unread(model_watch.load(path)) == []


def test_due_weekly_and_backs_off_after_errors():
    assert model_watch.due({}, NOW)
    week_ago = (NOW - timedelta(days=7)).isoformat()
    assert model_watch.due({"last_check": week_ago}, NOW)
    assert not model_watch.due({"last_check": (NOW - timedelta(days=2)).isoformat()}, NOW)
    failed = {"last_check": week_ago, "last_attempt": (NOW - timedelta(hours=2)).isoformat(),
              "last_error": "ConnectError"}  # fmt: skip
    assert not model_watch.due(failed, NOW)
