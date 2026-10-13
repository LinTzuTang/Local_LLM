#!/bin/sh
# Start the model gateway on http://127.0.0.1:11500 (localhost only): speech-to-text, document
# parsing and reranker models that Ollama cannot run. Weights: models/hf, models/docling, models/mlx.
set -eu
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$PROJECT_DIR"
export PATH="/opt/homebrew/bin:$HOME/.local/bin:$PATH"  # ffmpeg decodes the uploaded audio
PORT="${LOCAL_LLM_GATEWAY_PORT:-11500}"
if curl -fsS "http://127.0.0.1:$PORT/api/version" >/dev/null 2>&1; then
  echo "Model gateway is already running at http://127.0.0.1:$PORT."
  exit 0
fi
command -v ffmpeg >/dev/null || echo "提醒：找不到 ffmpeg，語音轉文字會失敗（brew install ffmpeg）" >&2
exec uv run python -m local_llm.gateway
