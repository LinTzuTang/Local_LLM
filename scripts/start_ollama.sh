#!/bin/sh
set -eu
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
export PATH="$HOME/.local/bin:$PATH"
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_MODELS="$PROJECT_DIR/models/ollama"
export OLLAMA_NO_CLOUD=1
# Serve several requests at once (e.g. three department agents side by side).
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-3}"
if curl -fsS http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
  echo "Ollama is already running at http://127.0.0.1:11434."
  echo "This project expects its model store at: $OLLAMA_MODELS"
  echo "Verify it with: uv run local-agent doctor"
  exit 0
fi
mkdir -p "$OLLAMA_MODELS"
exec ollama serve
