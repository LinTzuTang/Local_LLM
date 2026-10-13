#!/bin/sh
set -eu
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$PROJECT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export OLLAMA_HOST=127.0.0.1:11434
curl -fsS "http://$OLLAMA_HOST/api/version" >/dev/null || { echo "Start ./scripts/start_ollama.sh in another terminal first." >&2; exit 1; }
ollama pull "${1:-qwen3:30b-a3b}"   # default platform model; pass another tag to override
uv run local-agent doctor
