#!/bin/sh
set -eu
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
export PATH="$HOME/.local/bin:$PATH"
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_MODELS="$PROJECT_DIR/models/ollama"
export OLLAMA_NO_CLOUD=1
# Serve several requests at once (e.g. three department agents side by side).
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-3}"
# On a big NVIDIA card Ollama picks the model's full context for requests that do not set one
# (qwen3:30b-a3b at 256K tokens filled all 96 GB of the GCP machine); the platform's own requests
# set LOCAL_LLM_CONTEXT, so use the same default there. The Mac keeps Ollama's own default.
if [ "$(uname -s)" = Linux ] && [ -z "${OLLAMA_CONTEXT_LENGTH:-}" ]; then
  ctx=$(sed -n 's/^LOCAL_LLM_CONTEXT=\([0-9][0-9]*\).*/\1/p' "$PROJECT_DIR/.env" 2>/dev/null | tail -1)
  export OLLAMA_CONTEXT_LENGTH="${ctx:-16384}"
fi
if curl -fsS http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
  echo "Ollama is already running at http://127.0.0.1:11434."
  echo "This project expects its model store at: $OLLAMA_MODELS"
  echo "Verify it with: uv run local-agent doctor"
  exit 0
fi
mkdir -p "$OLLAMA_MODELS"
exec ollama serve
