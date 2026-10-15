"""Model downloads with Ollama-style progress events.

The Hugging Face download runs in a child process (`python -m local_llm.gateway.download`):
the gateway itself stays offline (HF_HUB_OFFLINE=1, so inference can never download behind
the user's back), and a cancelled pull simply kills the child. The child first prints the file
list with sizes (a dry run against the Hub API), then the bytes received so far, taken from
huggingface_hub's own progress bars through `tqdm_class` (Xet downloads only appear on disk
when a file is complete, so watching the directory is not enough).

Models served through Ollama (Gemma 4 audio) are pulled by Ollama itself; its progress events
already have this shape and are passed on unchanged.
"""

import asyncio
import json
import os
import shutil
import sys
import threading
import time
from collections.abc import AsyncIterator
from pathlib import Path

from . import ollama
from .convert import CONVERTERS
from .registry import ModelSpec, Store

REPORT_S = 0.5


def _job(spec: ModelSpec, store: Store) -> dict:
    return {
        "cache_dir": str(store.hf_cache),
        "items": [
            {
                "repo": w.repo,
                "revision": w.revision,
                "files": list(w.files) if w.files else None,
                "local_dir": str(store.target(w)) if w.into else None,
            }
            for w in spec.weights
        ],
    }


async def pull(spec: ModelSpec, store: Store) -> AsyncIterator[dict]:
    """Download (and convert) one model; yields {"status", "total", "completed"} dicts."""
    if store.installed(spec):
        yield {"status": f"{spec.id} is already downloaded"}
        yield {"status": "success"}
        return
    if spec.ollama:
        async for event in ollama.pull(spec.ollama):
            yield event
        return
    yield {"status": "pulling manifest"}
    env = {**os.environ, "HF_HUB_OFFLINE": "0"}
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "local_llm.gateway.download", json.dumps(_job(spec, store)),
        stdout=asyncio.subprocess.PIPE, env=env, limit=1 << 20,
    )  # fmt: skip
    digest = " + ".join(w.repo for w in spec.weights)
    plan, paths, total = None, {}, 0
    try:
        async for line in proc.stdout:
            message = json.loads(line)
            if "error" in message:
                raise RuntimeError(message["error"])
            if "plan" in message:
                plan = message["plan"]
                total = sum(f["size"] for item in plan for f in item)
            if "done" in message:
                paths[message["done"]] = Path(message["path"])
            if "completed" in message:
                yield {"status": f"pulling {digest}", "digest": digest, "total": total,
                       "completed": min(total, message["completed"])}  # fmt: skip
        if await proc.wait() != 0 or plan is None or len(paths) != len(spec.weights):
            raise RuntimeError(f"download of {spec.id} failed (exit code {proc.returncode})")
    finally:
        if proc.returncode is None:  # the client went away: stop the download
            proc.kill()
            await proc.wait()
    yield {"status": "verifying files", "digest": digest, "total": total, "completed": total}

    files = [paths[i] / f["file"] for i, item in enumerate(plan) for f in item]
    sources = [
        {"repo": w.repo, "revision": w.revision, "commit": plan[i][0]["commit"]}
        for i, w in enumerate(spec.weights)
    ]
    model_path = store.root / spec.path if spec.path else paths[0]
    if spec.convert:
        yield {"status": f"converting to MLX ({spec.convert})"}
        await asyncio.to_thread(CONVERTERS[spec.convert], paths[0], model_path)
        # The converted copy replaces the download.
        await asyncio.to_thread(shutil.rmtree, paths[0], True)
        files = sorted(p for p in model_path.iterdir() if p.is_file())
    if missing := [f for f in files if not f.exists()]:
        raise RuntimeError(f"files missing after download: {missing[:3]}")
    store.write_manifest(spec, model_path, files, sources)
    yield {"status": "success"}


def _child(job: dict) -> None:
    """Runs in the child process: print the plan, then progress, then each download's path."""
    from huggingface_hub import snapshot_download
    from tqdm import tqdm

    lock, state = threading.Lock(), {"base": 0, "last": 0.0, "bytes": 0}

    def emit(message: dict) -> None:
        with lock:
            print(json.dumps(message), flush=True)

    class Progress(tqdm):
        """snapshot_download's byte bars: "Downloading bytes" (network, updated every second)
        and "Reconstructing" (written to disk, in bursts); the larger one is the progress."""

        def __init__(self, *args, **kwargs):
            desc = str(kwargs.get("desc", ""))
            self.counts = desc.startswith(("Downloading bytes", "Reconstructing"))
            super().__init__(*args, **{**kwargs, "file": open(os.devnull, "w"), "disable": False})

        def update(self, n=1):
            super().update(n)
            if not self.counts:
                return
            state["bytes"] = max(state["bytes"], int(self.n))
            now = time.monotonic()
            if now - state["last"] >= REPORT_S:
                state["last"] = now
                emit({"completed": state["base"] + state["bytes"]})

    def fetch(item: dict, dry_run: bool):
        return snapshot_download(
            item["repo"],
            revision=item["revision"],
            allow_patterns=item["files"],
            local_dir=item["local_dir"],
            cache_dir=None if item["local_dir"] else job["cache_dir"],
            dry_run=dry_run,
            tqdm_class=None if dry_run else Progress,
        )

    plan = [
        [
            {"file": f.filename, "size": f.file_size, "cached": f.is_cached,
             "commit": f.commit_hash}
            for f in fetch(item, dry_run=True)
        ]
        for item in job["items"]
    ]  # fmt: skip
    if not all(plan):
        raise RuntimeError("no files matched; check the repo and file patterns in the catalog")
    emit({"plan": plan})
    # Files already on disk count as done from the start.
    state["base"] = sum(f["size"] for item in plan for f in item if f["cached"])
    emit({"completed": state["base"]})
    for i, item in enumerate(job["items"]):
        path = fetch(item, dry_run=False)
        state["base"] += sum(f["size"] for f in plan[i] if not f["cached"])
        state["bytes"] = 0  # the next snapshot_download starts new bars
        emit({"done": i, "path": path, "completed": state["base"]})


if __name__ == "__main__":
    try:
        _child(json.loads(sys.argv[1]))
    except Exception as exc:  # reported to the parent, which turns it into {"error": ...}
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), flush=True)
        sys.exit(1)
