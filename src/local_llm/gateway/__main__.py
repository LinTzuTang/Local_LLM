"""uv run python -m local_llm.gateway  ->  http://127.0.0.1:11500 (localhost only)."""

import os

from . import DEFAULT_KEEP_ALIVE, HOST, MODELS_DIR, PORT


def main() -> None:
    # The serving process never downloads: inference on a missing model fails with a clear
    # "pull it first" error instead. /api/pull downloads in a child process that is online.
    os.environ["HF_HUB_OFFLINE"] = "1"
    import uvicorn

    from .server import create_app

    print(f"Model gateway on http://{HOST}:{PORT} · weights in {MODELS_DIR} · "
          f"keep_alive {DEFAULT_KEEP_ALIVE:.0f}s", flush=True)  # fmt: skip
    uvicorn.run(create_app(), host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
