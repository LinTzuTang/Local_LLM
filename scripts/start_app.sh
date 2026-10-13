#!/bin/sh
# Start the Streamlit platform on http://localhost:8501 (localhost only).
set -eu
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$PROJECT_DIR"
export PATH="$HOME/.local/bin:$PATH"
curl -fsS http://127.0.0.1:11434/api/version >/dev/null 2>&1 \
  || echo "提醒：Ollama 尚未啟動，請另開 terminal 執行 ./scripts/start_ollama.sh" >&2
exec uv run streamlit run app/main.py \
  --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false "$@"
