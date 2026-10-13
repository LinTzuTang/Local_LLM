"""Knowledge base for the RAG scenarios: parse → chunk → BM25 + vector index → RRF.

A collection is a folder `data/lab/corpus/<name>/` with documents and a `meta.json` that tags
each file with the departments allowed to read it (`["all"]` or e.g. `["hr"]`). Indexes are
built into `storage/rag/<name>/`. Embeddings come from Ollama (`/api/embed`, default bge-m3);
PDF/DOCX parsing and reranking go through the model gateway.
"""

import json
import math
import re
import time
from collections import Counter
from collections.abc import Callable, Iterator
from functools import lru_cache
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CORPUS_DIR = ROOT / "data" / "lab" / "corpus"
INDEX_DIR = ROOT / "storage" / "rag"
DEFAULT_EMBED = "bge-m3"
TEXT_SUFFIXES = {".md", ".txt"}
PARSED_SUFFIXES = {".pdf", ".docx", ".pptx", ".html"}
CHUNK_CHARS, OVERLAP = 500, 80


# --- collections ----------------------------------------------------------------------------


def collections() -> list[str]:
    return sorted(p.name for p in CORPUS_DIR.iterdir() if p.is_dir()) if CORPUS_DIR.exists() else []


def documents(collection: str) -> list[dict]:
    """Files in a collection with their title and access tags."""
    folder = CORPUS_DIR / collection
    meta = _meta(collection)
    rows = []
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() in TEXT_SUFFIXES | PARSED_SUFFIXES:
            info = meta.get(path.name, {})
            rows.append(
                {
                    "file": path.name,
                    "title": info.get("title") or path.stem,
                    "access": info.get("access", ["all"]),
                    "bytes": path.stat().st_size,
                }
            )
    return rows


def _meta(collection: str) -> dict:
    path = CORPUS_DIR / collection / "meta.json"
    return json.loads(path.read_text()) if path.exists() else {}


def set_document_meta(collection: str, file: str, title: str, access: list[str]) -> None:
    meta = _meta(collection)
    meta[file] = {"title": title, "access": access or ["all"]}
    path = CORPUS_DIR / collection / "meta.json"
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")


# --- chunking ---------------------------------------------------------------------------------


def chunk_markdown(text: str, size: int = CHUNK_CHARS, overlap: int = OVERLAP) -> list[str]:
    """Split by headings, then pack paragraphs up to `size` characters. Each chunk starts with
    its heading path so a passage still makes sense on its own."""
    chunks, heads = [], []
    sections: list[tuple[str, list[str]]] = []
    for line in text.splitlines():
        if m := re.match(r"^(#{1,4})\s+(.*)", line):
            level = len(m.group(1))
            heads = heads[: level - 1] + [m.group(2).strip()]
            sections.append((" > ".join(heads), []))
        else:
            if not sections:
                sections.append(("", []))
            sections[-1][1].append(line)
    for head, lines in sections:
        paragraphs = [p.strip() for p in "\n".join(lines).split("\n\n") if p.strip()]
        buf = ""
        for para in paragraphs:
            while len(para) > size:  # very long paragraph: hard split with overlap
                chunks.append(_with_head(head, (buf + "\n" + para[:size]).strip()))
                para, buf = para[size - overlap :], ""
            if len(buf) + len(para) > size and buf:
                chunks.append(_with_head(head, buf))
                buf = buf[-overlap:] if overlap else ""
            buf = (buf + "\n" + para).strip()
        if buf:
            chunks.append(_with_head(head, buf))
    return chunks


def _with_head(head: str, body: str) -> str:
    return f"{head}\n{body}" if head else body


# --- BM25 -------------------------------------------------------------------------------------


def tokenize(text: str) -> list[str]:
    """Latin words plus CJK character bigrams (no extra dictionary needed for Chinese)."""
    tokens = re.findall(r"[a-z0-9]+(?:[.-][a-z0-9]+)*", text.lower())
    for run in re.findall(r"[㐀-鿿]+", text):
        tokens += [run] if len(run) == 1 else [run[i : i + 2] for i in range(len(run) - 1)]
    return tokens


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avg = sum(self.len) / max(len(docs), 1)
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for tf, length in zip(self.tf, self.len, strict=True):
            s = 0.0
            for t in query:
                if f := tf.get(t):
                    norm = f + self.k1 * (1 - self.b + self.b * length / self.avg)
                    s += self.idf[t] * f * (self.k1 + 1) / norm
            out.append(s)
        return out


# --- embeddings ---------------------------------------------------------------------------------


def embed(
    texts: list[str], model: str = DEFAULT_EMBED, base_url: str = "http://127.0.0.1:11434"
) -> np.ndarray:
    vectors = []
    with httpx.Client(base_url=base_url, timeout=300, trust_env=False) as client:
        for i in range(0, len(texts), 32):
            r = client.post("/api/embed", json={"model": model, "input": texts[i : i + 32]})
            if r.is_error:
                raise RuntimeError(f"Ollama embed failed ({r.status_code}): {r.text[:300]}")
            vectors += r.json()["embeddings"]
    matrix = np.asarray(vectors, dtype=np.float32)
    return matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9, None)


# --- index ----------------------------------------------------------------------------------------


def read_document(collection: str, file: str, parse: Callable[[Path], str] | None = None) -> str:
    path = CORPUS_DIR / collection / file
    if path.suffix.lower() in TEXT_SUFFIXES:
        return path.read_text()
    if parse is None:
        from .. import gateway_client

        return gateway_client.parse_document(path)["markdown"]
    return parse(path)


def build_index(
    collection: str,
    embed_model: str = DEFAULT_EMBED,
    base_url: str = "http://127.0.0.1:11434",
    parse: Callable[[Path], str] | None = None,
) -> Iterator[dict]:
    """Yield progress events while (re)building the index of a collection."""
    started = time.perf_counter()
    docs = documents(collection)
    chunks = []
    for n, doc in enumerate(docs, start=1):
        yield {"stage": "parse", "doc": doc["file"], "done": n - 1, "total": len(docs)}
        text = read_document(collection, doc["file"], parse)
        for i, piece in enumerate(chunk_markdown(text)):
            chunks.append(
                {
                    "id": f"{doc['file']}#{i}",
                    "doc": doc["file"],
                    "title": doc["title"],
                    "access": doc["access"],
                    "text": piece,
                }
            )
    yield {"stage": "embed", "done": 0, "total": len(chunks)}
    vectors = embed([c["text"] for c in chunks], embed_model, base_url) if chunks else None
    out = INDEX_DIR / collection
    out.mkdir(parents=True, exist_ok=True)
    (out / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False))
    if vectors is not None:
        np.save(out / "vectors.npy", vectors)
    manifest = {
        "collection": collection,
        "embed_model": embed_model,
        "documents": len(docs),
        "chunks": len(chunks),
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seconds": round(time.perf_counter() - started, 1),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    yield {"stage": "done", **manifest}


def manifest(collection: str) -> dict | None:
    path = INDEX_DIR / collection / "manifest.json"
    return json.loads(path.read_text()) if path.exists() else None


@lru_cache(maxsize=8)
def _load(collection: str, stamp: float):
    """Cached per index version (`stamp` = manifest mtime), so a rebuild is picked up."""
    folder = INDEX_DIR / collection
    chunks = json.loads((folder / "chunks.json").read_text())
    vectors = np.load(folder / "vectors.npy") if (folder / "vectors.npy").exists() else None
    return chunks, vectors, BM25([tokenize(c["text"]) for c in chunks])


def load_index(collection: str):
    info = manifest(collection)
    if not info:
        raise RuntimeError(
            f"Collection '{collection}' has no index yet; build it on the knowledge-base page."
        )
    return _load(collection, (INDEX_DIR / collection / "manifest.json").stat().st_mtime)


def _allowed(chunk: dict, role: str | None) -> bool:
    return not role or "all" in chunk["access"] or role in chunk["access"]


def _hits(chunks, order, scores, role, top_k, method) -> list[dict]:
    hits = []
    for i in order:
        if len(hits) == top_k:
            break
        if scores[i] > 0 and _allowed(chunks[i], role):
            hits.append({**chunks[i], "score": round(float(scores[i]), 4), "method": method})
    return hits


def search_bm25(collection: str, query: str, top_k: int = 8, role: str | None = None) -> list:
    chunks, _, bm25 = load_index(collection)
    scores = bm25.scores(tokenize(query))
    order = sorted(range(len(chunks)), key=lambda i: -scores[i])
    return _hits(chunks, order, scores, role, top_k, "bm25")


def search_vector(
    collection: str,
    query: str,
    top_k: int = 8,
    role: str | None = None,
    base_url: str = "http://127.0.0.1:11434",
) -> list:
    chunks, vectors, _ = load_index(collection)
    model = manifest(collection)["embed_model"]
    scores = vectors @ embed([query], model, base_url)[0]
    order = np.argsort(-scores).tolist()
    return _hits(chunks, order, scores, role, top_k, "vector")


def rrf(rankings: list[list[dict]], k: int = 60, top_k: int = 8) -> list[dict]:
    """Reciprocal Rank Fusion: score = Σ 1 / (k + rank) over the rankings a chunk appears in."""
    fused: dict[str, dict] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            entry = fused.setdefault(hit["id"], {**hit, "score": 0.0, "ranks": {}})
            entry["score"] += 1 / (k + rank)
            entry["ranks"][hit.get("method", "?")] = rank
    ordered = sorted(fused.values(), key=lambda h: -h["score"])[:top_k]
    for hit in ordered:
        hit["score"], hit["method"] = round(hit["score"], 5), "rrf"
    return ordered
