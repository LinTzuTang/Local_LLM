"""Which models the gateway offers (from data/catalog.json) and where their files live.

A model counts as installed only when its manifest (models/gateway/<id>.json, written at the
end of a successful pull) exists and every file it lists is still on disk, so an interrupted
download never looks installed. A model served through Ollama (`ollama` set: audio-capable
LLMs such as Gemma 4) is installed when Ollama has its tag; Ollama owns those files.
"""

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from .. import catalog
from . import MODELS_DIR, ollama


@dataclass(frozen=True)
class Weights:
    repo: str
    revision: str | None = None
    files: tuple[str, ...] | None = None  # allow_patterns; None = the whole repo
    into: str | None = None  # directory under models/; None = Hugging Face cache (models/hf)


@dataclass(frozen=True)
class ModelSpec:
    id: str
    task: str  # "asr", "doc" or "rerank"
    backend: str  # key in backends.BACKENDS
    weights: tuple[Weights, ...]
    size_gb: float
    name: str = ""
    license: str = ""
    convert: str | None = None  # post-download step, e.g. "hf-whisper-to-mlx"
    path: str | None = None  # model directory under models/ when it is not the download itself
    options: dict = field(default_factory=dict)
    ollama: str | None = None  # Ollama tag: the model runs in Ollama, not in the gateway


def load_specs() -> dict[str, ModelSpec]:
    specs = {}
    for m in catalog.gateway_models():
        weights = tuple(
            Weights(w["repo"], w.get("revision"), tuple(w["files"]) if "files" in w else None,
                    w.get("into"))
            for w in m.get("weights", [])
        )  # fmt: skip
        specs[m["id"]] = ModelSpec(
            id=m["id"],
            task=m["task"],
            backend=m["backend"],
            weights=weights,
            size_gb=m["size_gb"],
            name=m.get("name", m["id"]),
            license=m.get("license", ""),
            convert=m.get("convert"),
            path=m.get("path"),
            options=m.get("options", {}),
            ollama=m.get("ollama"),
        )
    return specs


class Store:
    """Model files on disk: download targets, manifests, removal."""

    def __init__(self, root: Path = MODELS_DIR):
        self.root = root
        self.hf_cache = root / "hf" / "hub"
        self.manifests = root / "gateway"

    def cache_dir(self, w: Weights) -> Path:
        """This repo's folder in the Hugging Face cache (models/hf/hub)."""
        return self.hf_cache / f"models--{w.repo.replace('/', '--')}"

    def target(self, w: Weights) -> Path:
        """Directory a download of these weights writes into."""
        return self.root / w.into if w.into else self.cache_dir(w)

    def manifest(self, spec: ModelSpec) -> dict | None:
        try:
            return json.loads((self.manifests / f"{spec.id}.json").read_text())
        except (OSError, ValueError):
            return None

    def installed(self, spec: ModelSpec) -> bool:
        if spec.ollama:
            return spec.ollama in ollama.installed()
        manifest = self.manifest(spec)
        return bool(manifest) and all((self.root / f).exists() for f in manifest["files"])

    def model_path(self, spec: ModelSpec) -> Path:
        if spec.ollama:
            return self.root / "ollama"  # not read: the backend talks to Ollama by tag
        return self.root / self.manifest(spec)["path"]

    def size_bytes(self, spec: ModelSpec) -> int | None:
        if spec.ollama:
            return ollama.installed().get(spec.ollama)
        manifest = self.manifest(spec)
        return manifest.get("size_bytes") if manifest else None

    def write_manifest(self, spec: ModelSpec, path: Path, files: list[Path], sources: list) -> None:
        self.manifests.mkdir(parents=True, exist_ok=True)
        record = {
            "id": spec.id,
            "path": str(path.relative_to(self.root)),
            "files": [str(f.relative_to(self.root)) for f in files],
            "size_bytes": sum(f.stat().st_size for f in files),
            "sources": sources,
            "pulled_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        (self.manifests / f"{spec.id}.json").write_text(json.dumps(record, indent=2) + "\n")

    def remove(self, spec: ModelSpec) -> None:
        """Delete the manifest first (the model stops counting as installed), then the files.

        A model served through Ollama is deleted in Ollama (the same weights as its chat use)."""
        if spec.ollama:
            ollama.delete(spec.ollama)
            return
        (self.manifests / f"{spec.id}.json").unlink(missing_ok=True)
        dirs = [self.root / w.into for w in spec.weights if w.into]
        if spec.path:
            dirs.append(self.root / spec.path)
        for w in spec.weights:
            if not w.into:
                self._remove_cached(w)
            # A download into a plain folder still leaves a ref-only folder in the HF cache.
            if not (self.cache_dir(w) / "snapshots").exists():
                dirs.append(self.cache_dir(w))
        root, keep = self.root.resolve(), {self.root.resolve(), self.hf_cache.resolve()}
        for d in dirs:
            d = d.resolve()
            # Never follow a bad catalog entry outside models/ (or wipe a whole store).
            if d.is_relative_to(root) and d not in keep:
                shutil.rmtree(d, ignore_errors=True)

    def _remove_cached(self, w: Weights) -> None:
        """Delete a repo from the HF cache through huggingface_hub: since 1.x the file contents
        live in a shared store (hub/blobs) that only its own cleanup frees correctly."""
        from huggingface_hub import scan_cache_dir

        try:
            cache = scan_cache_dir(self.hf_cache)
        except Exception:  # no cache yet, or nothing it recognises
            return
        for repo in cache.repos:
            if repo.repo_id == w.repo and repo.repo_type == "model":
                cache.delete_revisions(*(r.commit_hash for r in repo.revisions)).execute()
        shutil.rmtree(self.cache_dir(w), ignore_errors=True)  # leftover refs
