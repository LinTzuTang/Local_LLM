"""Model gateway: HTTP API with fake backends (no model downloads), client, inbox safety."""

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from local_llm import catalog, gateway_client, media_tools
from local_llm.gateway import convert
from local_llm.gateway.backends import Backend
from local_llm.gateway.manager import Manager
from local_llm.gateway.registry import Store, Weights, load_specs
from local_llm.gateway.server import create_app, normalize_language, parse_keep_alive

ASR_IDS = {
    "breeze-asr-25", "taiwan-tongues-asr", "whisper-large-v3-turbo", "whisper-large-v3",
    "qwen3-asr-1.7b", "qwen3-asr-0.6b", "sensevoice-small", "fun-asr-nano", "gemma4:e2b",
    "gemma4:e4b",
}  # fmt: skip
GATEWAY_IDS = ASR_IDS | {"docling", "bge-reranker-v2-m3", "speaker-diarization"}


class FakeBackend(Backend):
    def __init__(self, task: str):
        self.task, self.loads, self.released = task, 0, 0

    def load(self, spec, path):
        self.loads += 1
        return {"id": spec.id}, "fake"

    def run(self, handle, **inputs):
        if self.task == "asr":
            assert Path(inputs["audio"]).read_bytes() == b"RIFF fake audio"
            text = "Good morning" if inputs.get("mode") == "translate" else "各位早安"
            segments = [{"start": 0.0, "end": 1.5, "text": text}]
            return {"text": text, "segments": segments, "language": "zh", "duration": 1.5}
        if self.task == "diarize":
            self.last = inputs
            n = inputs.get("num_speakers") or 2
            segments = [{"start": i * 1.0, "end": i + 0.9, "speaker": i % n} for i in range(4)]
            return {"segments": segments, "speakers": n, "duration": 4.0}
        if self.task == "doc":
            return {"markdown": "# 合約\n\n| a | b |", "pages": 1, "tables": 1}
        # rerank: longer passages that contain the query score higher
        return {"scores": [(inputs["query"] in d) + len(d) / 100 for d in inputs["documents"]]}

    def release(self, handle):
        self.released += 1


class FakeStore(Store):
    def __init__(self, root: Path, installed: set[str]):
        super().__init__(root)
        self.ids = installed

    def installed(self, spec):
        return spec.id in self.ids

    def model_path(self, spec):
        return self.root / spec.id

    def size_bytes(self, spec):
        return 123 if spec.id in self.ids else None

    def remove(self, spec):
        self.ids.discard(spec.id)


@pytest.fixture
def gateway(tmp_path):
    specs = load_specs()
    backends = {
        "mlx-whisper": FakeBackend("asr"),
        "docling": FakeBackend("doc"),
        "cross-encoder": FakeBackend("rerank"),
        "sherpa-diarization": FakeBackend("diarize"),
    }
    store = FakeStore(
        tmp_path, {"breeze-asr-25", "docling", "bge-reranker-v2-m3", "speaker-diarization"}
    )
    manager = Manager(specs, backends, store, keep_alive=300, free=lambda: None)

    async def fake_pull(spec, store):
        yield {"status": "pulling manifest"}
        yield {"status": f"pulling {spec.id}", "total": 100, "completed": 40}
        yield {"status": f"pulling {spec.id}", "total": 100, "completed": 100}
        store.ids.add(spec.id)
        yield {"status": "success"}

    app = create_app(manager, puller=fake_pull, max_upload_bytes=1024)
    with TestClient(app) as client:
        yield client, manager, backends


def test_catalog_lists_the_gateway_models():
    models = catalog.gateway_models()
    assert {m["id"] for m in models} == GATEWAY_IDS
    for m in models:
        assert m["task"] in {"asr", "doc", "rerank", "diarize"} and m["size_gb"] > 0
        assert m.get("weights") or m.get("ollama")  # Hugging Face files, or an Ollama tag
        assert all(w.get("revision") for w in m.get("weights", []))  # pinned
    specs = load_specs()
    assert specs["breeze-asr-25"].convert == "hf-whisper-to-mlx"
    assert specs["docling"].weights[0].into.startswith("docling/")


def test_version_tags_and_ps(gateway):
    client, _, _ = gateway
    assert client.get("/api/version").json()["version"]
    tags = {m["id"]: m for m in client.get("/api/tags").json()["models"]}
    assert set(tags) == GATEWAY_IDS
    assert tags["breeze-asr-25"]["installed"] and not tags["whisper-large-v3-turbo"]["installed"]
    assert {"id", "task", "installed", "size_gb", "loaded", "name", "license"} <= set(
        tags["docling"]
    )
    assert client.get("/api/ps").json() == {"models": []}


def test_uninstalled_or_unknown_model_asks_to_pull(gateway):
    client, _, backends = gateway
    files = {"file": ("a.wav", b"RIFF fake audio")}
    r = client.post(
        "/v1/audio/transcriptions", files=files, data={"model": "whisper-large-v3-turbo"}
    )
    assert r.status_code == 404 and "pull it first" in r.json()["error"]
    r = client.post("/v1/rerank", json={"model": "nope", "query": "q", "documents": ["d"]})
    assert r.status_code == 404 and "not found" in r.json()["error"]
    r = client.post("/v1/rerank", json={"model": "docling", "query": "q", "documents": ["d"]})
    assert r.status_code == 400
    assert backends["mlx-whisper"].loads == 0  # nothing was loaded, nothing downloaded


def test_transcription_loads_lazily_and_keep_alive_unloads(gateway):
    client, manager, backends = gateway
    files = {"file": ("meeting.wav", b"RIFF fake audio")}
    r = client.post("/v1/audio/transcriptions", files=files, data={"language": "zh-TW"})
    body = r.json()
    assert r.status_code == 200, body
    assert body["text"] == "各位早安" and body["model"] == "breeze-asr-25"
    assert body["segments"][0] == {"start": 0.0, "end": 1.5, "text": "各位早安"}
    assert {"language", "duration", "seconds"} <= set(body)
    (loaded,) = client.get("/api/ps").json()["models"]
    assert loaded["id"] == "breeze-asr-25" and loaded["device"] == "fake" and loaded["expires_at"]

    # A second request reuses the loaded model; keep_alive=0 unloads right after it.
    files = {"file": ("meeting.wav", b"RIFF fake audio")}
    client.post("/v1/audio/transcriptions", files=files, data={"keep_alive": "0"})
    assert backends["mlx-whisper"].loads == 1 and backends["mlx-whisper"].released == 1
    assert client.get("/api/ps").json()["models"] == []

    # The reaper unloads idle models once keep_alive has passed.
    client.post("/v1/rerank", json={"query": "q", "documents": ["a"], "keep_alive": "1s"})
    assert manager.reap(now=time.time()) == []
    assert manager.reap(now=time.time() + 2) == ["bge-reranker-v2-m3"]


def test_rerank_sorts_and_truncates(gateway):
    client, _, _ = gateway
    docs = ["無關", "逾期罰款每日 0.2%", "逾期罰款上限為貨款 10%，另得請求損害賠償"]
    r = client.post("/v1/rerank", json={"query": "逾期罰款", "documents": docs, "top_n": 2})
    body = r.json()
    assert [x["index"] for x in body["results"]] == [2, 1]
    assert body["results"][0]["document"] == docs[2]
    scores = [x["relevance_score"] for x in body["results"]]
    assert scores == sorted(scores, reverse=True)
    bad = client.post("/v1/rerank", json={"query": "q", "documents": []})
    assert bad.status_code == 422 and "error" in bad.json()


def test_document_parse_checks_type_and_size(gateway):
    client, _, _ = gateway
    r = client.post("/v1/documents/parse", files={"file": ("c.pdf", b"%PDF-1.7 fake")})
    assert r.status_code == 200 and r.json()["tables"] == 1 and r.json()["pages"] == 1
    r = client.post("/v1/documents/parse", files={"file": ("run.exe", b"MZ")})
    assert r.status_code == 415
    r = client.post("/v1/documents/parse", files={"file": ("big.pdf", b"x" * 4096)})
    assert r.status_code == 413  # the test app allows 1 KB


def test_pull_streams_ndjson_then_model_is_installed(gateway):
    client, _, _ = gateway
    with client.stream("POST", "/api/pull", json={"model": "whisper-large-v3-turbo"}) as r:
        events = [json.loads(line) for line in r.iter_lines() if line]
    assert events[0]["status"] == "pulling manifest" and events[-1] == {"status": "success"}
    assert any(e.get("completed") == 40 and e.get("total") == 100 for e in events)
    tags = {m["id"]: m for m in client.get("/api/tags").json()["models"]}
    assert tags["whisper-large-v3-turbo"]["installed"]
    assert client.post("/api/pull", json={"model": "nope"}).status_code == 404


def test_unload_and_delete(gateway):
    client, _, _ = gateway
    client.post("/v1/rerank", json={"query": "q", "documents": ["a"]})
    assert client.post("/api/unload", json={"model": "bge-reranker-v2-m3"}).json()["status"] == (
        "unloaded"
    )
    assert client.post("/api/unload", json={"model": "docling"}).json()["status"] == "not loaded"
    assert client.request("DELETE", "/api/delete", json={"model": "docling"}).status_code == 200
    r = client.request("DELETE", "/api/delete", json={"model": "docling"})
    assert r.status_code == 404


def test_client_functions_against_the_app(gateway, monkeypatch):
    client, _, _ = gateway
    monkeypatch.setattr(gateway_client, "_client", lambda base_url, timeout=10: client)
    assert gateway_client.version() == client.get("/api/version").json()["version"]
    assert {m["id"] for m in gateway_client.models()} == GATEWAY_IDS
    result = gateway_client.transcribe(b"RIFF fake audio", language="zh", keep_alive=-1)
    assert result["text"] == "各位早安"
    assert gateway_client.loaded()[0]["expires_at"] is None  # keep_alive < 0: keep forever
    assert gateway_client.rerank("逾期", ["a", "逾期罰款"], top_n=1)[0]["index"] == 1
    assert gateway_client.parse_document(b"%PDF fake", filename="x.pdf")["markdown"]
    events = list(gateway_client.pull("whisper-large-v3-turbo"))
    assert events[-1]["status"] == "success"
    with pytest.raises(gateway_client.GatewayError, match="not found"):
        gateway_client.rerank("q", ["a"], model="nope")


def test_translation_and_diarization_endpoints(gateway):
    client, _, backends = gateway
    files = {"file": ("talk.wav", b"RIFF fake audio")}
    r = client.post("/v1/audio/translations", files=files,
                    data={"model": "breeze-asr-25", "language": "zh"})  # fmt: skip
    assert r.status_code == 200 and r.json()["text"] == "Good morning"
    r = client.post("/v1/audio/translations", files=files, data={"model": "docling"})
    assert r.status_code == 400  # not a speech model
    r = client.post("/v1/audio/diarize", files=files, data={"num_speakers": "3"})
    body = r.json()
    assert r.status_code == 200 and body["model"] == "speaker-diarization"
    assert body["speakers"] == 3 and {"start", "end", "speaker"} <= set(body["segments"][0])
    assert backends["sherpa-diarization"].last["num_speakers"] == 3
    r = client.post("/v1/audio/diarize", files=files, data={"num_speakers": "0"})
    assert r.status_code == 422
    r = client.post("/v1/audio/diarize", files=files, data={"model": "breeze-asr-25"})
    assert r.status_code == 400  # an ASR model cannot diarize


def test_diarize_and_translate_clients(gateway, monkeypatch):
    client, _, backends = gateway
    monkeypatch.setattr(gateway_client, "_client", lambda base_url, timeout=10: client)
    out = gateway_client.diarize(b"RIFF fake audio", num_speakers=2)
    assert [s["speaker"] for s in out["segments"]] == [0, 1, 0, 1]
    assert backends["sherpa-diarization"].last["threshold"] is None
    assert gateway_client.translate(b"RIFF fake audio", "breeze-asr-25")["text"] == "Good morning"


def test_audio_llm_translation_instruction():
    from local_llm.gateway.backends import audio_instruction

    text = audio_instruction({}, "zh", None, mode="translate")
    assert "into English" in text and "Traditional" not in text
    assert "verbatim" in audio_instruction({}, "zh", None)


def test_keep_alive_and_language_parsing():
    assert parse_keep_alive(None) is None and parse_keep_alive("") is None
    assert parse_keep_alive("5m") == 300 and parse_keep_alive("-1") == -1
    assert parse_keep_alive(30) == 30 and parse_keep_alive("1h") == 3600
    assert normalize_language("zh-TW") == "zh" and normalize_language("auto") is None


def test_store_manifest_marks_complete_downloads(tmp_path):
    spec = load_specs()["whisper-large-v3-turbo"]
    store = Store(tmp_path)
    assert not store.installed(spec)
    model_dir = tmp_path / "hf" / "hub" / "models--mlx-community--whisper-large-v3-turbo" / "snap"
    model_dir.mkdir(parents=True)
    files = [model_dir / "config.json", model_dir / "weights.safetensors"]
    for f in files:
        f.write_text("{}")
    store.write_manifest(spec, model_dir, files, [])
    assert store.installed(spec) and store.model_path(spec) == model_dir
    files[1].unlink()  # a missing file means not installed
    assert not store.installed(spec)
    store.remove(spec)
    assert not model_dir.exists() and store.target(Weights("a/b")).parent == store.hf_cache


def test_whisper_conversion_covers_every_tensor(tmp_path):
    mx = pytest.importorskip("mlx.core")
    from mlx_whisper.load_models import load_model

    layers, d, vocab = 1, 8, 51865
    config = {"num_mel_bins": 80, "max_source_positions": 4, "d_model": d,
              "encoder_attention_heads": 2, "encoder_layers": layers, "vocab_size": vocab,
              "max_target_positions": 6, "decoder_attention_heads": 2,
              "decoder_layers": layers}  # fmt: skip
    shapes = {
        "model.encoder.conv1.weight": (d, 80, 3), "model.encoder.conv1.bias": (d,),
        "model.encoder.conv2.weight": (d, d, 3), "model.encoder.conv2.bias": (d,),
        "model.encoder.embed_positions.weight": (4, d),
        "model.encoder.layer_norm.weight": (d,), "model.encoder.layer_norm.bias": (d,),
        "model.decoder.embed_tokens.weight": (vocab, d),
        "model.decoder.embed_positions.weight": (6, d),
        "model.decoder.layer_norm.weight": (d,), "model.decoder.layer_norm.bias": (d,),
        "proj_out.weight": (vocab, d),
    }  # fmt: skip
    attention = [("self_attn", "encoder"), ("self_attn", "decoder"), ("encoder_attn", "decoder")]
    for kind, side in attention:
        prefix = f"model.{side}.layers.0.{kind}"
        for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
            shapes[f"{prefix}.{proj}.weight"] = (d, d)
            if proj != "k_proj":
                shapes[f"{prefix}.{proj}.bias"] = (d,)
        shapes[f"model.{side}.layers.0.{kind}_layer_norm.weight"] = (d,)
        shapes[f"model.{side}.layers.0.{kind}_layer_norm.bias"] = (d,)
    for side in ("encoder", "decoder"):
        p = f"model.{side}.layers.0"
        shapes |= {f"{p}.fc1.weight": (4 * d, d), f"{p}.fc1.bias": (4 * d,),
                   f"{p}.fc2.weight": (d, 4 * d), f"{p}.fc2.bias": (d,),
                   f"{p}.final_layer_norm.weight": (d,),
                   f"{p}.final_layer_norm.bias": (d,)}  # fmt: skip
    source = tmp_path / "hf"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(str(source / "model.safetensors"),
                        {k: mx.zeros(s, dtype=mx.float16) for k, s in shapes.items()})  # fmt: skip
    out = convert.hf_whisper_to_mlx(source, tmp_path / "mlx")
    model = load_model(str(out), dtype=mx.float16)
    assert model.dims.n_text_layer == layers and model.encoder.conv1.weight.shape == (d, 3, 80)


@pytest.fixture
def inbox(tmp_path, monkeypatch):
    monkeypatch.setattr(media_tools, "INBOX_DIR", tmp_path / "inbox")
    (tmp_path / "inbox").mkdir()
    (tmp_path / "secret.txt").write_text("secret")
    return tmp_path / "inbox"


def test_inbox_names_cannot_escape(inbox):
    assert media_tools.safe_name("../../etc/passwd") == "passwd"
    assert media_tools.safe_name("C:\\Users\\me\\會議 錄音.m4a") == "會議 錄音.m4a"
    assert media_tools.safe_name("a;rm -rf *.wav") == "a_rm -rf _.wav"
    assert media_tools.safe_name(".env") == "env"  # never a hidden file
    for bad in ("..", "", "   ", "/"):
        with pytest.raises(ValueError):
            media_tools.safe_name(bad)
    name = media_tools.save("../報告.pdf", b"%PDF")
    assert name == "報告.pdf" and (inbox / name).exists()
    for bad in ("../secret.txt", "/etc/passwd", ".hidden", "sub/報告.pdf", "..\\secret.txt"):
        with pytest.raises(ValueError):
            media_tools.inbox_path(bad)
    (inbox / "link.pdf").symlink_to(inbox.parent / "secret.txt")
    with pytest.raises(FileNotFoundError):
        media_tools.inbox_path("link.pdf")  # resolves outside the inbox
    assert [r["name"] for r in media_tools.list_files()] == ["報告.pdf"]
    assert "ERROR" in media_tools.parse_document.invoke({"filename": "../secret.txt"})
    assert "ERROR" in media_tools.transcribe_audio.invoke({"filename": "報告.pdf"})
    media_tools.remove("報告.pdf")
    assert media_tools.list_files() == []


def test_inbox_tools_report_a_stopped_gateway(inbox, monkeypatch):
    (inbox / "m.wav").write_bytes(b"RIFF")
    monkeypatch.setattr(gateway_client, "DEFAULT_URL", "http://127.0.0.1:9")

    def refuse(*args, **kwargs):
        raise gateway_client.httpx.ConnectError("refused")

    monkeypatch.setattr(gateway_client, "transcribe", refuse)
    assert "start_gateway.sh" in media_tools.transcribe_audio.invoke({"filename": "m.wav"})
