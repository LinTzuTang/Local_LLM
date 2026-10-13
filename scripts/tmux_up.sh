#!/bin/sh
# Keep the platform running in a tmux session named "local_llm" (survives closing the terminal).
#   ./scripts/tmux_up.sh        start (or show) the session
#   tmux attach -t local_llm    look at the logs; detach again with Ctrl-b then d
#   ./scripts/tmux_down.sh      stop the platform, Ollama and gateway windows (the database keeps running)
set -eu
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
SESSION=local_llm
GATEWAY_PORT="${LOCAL_LLM_GATEWAY_PORT:-11500}"
command -v tmux >/dev/null || { echo "tmux not found: brew install tmux" >&2; exit 1; }

# Model gateway (speech, documents, reranker) in its own window, unless it already answers.
start_gateway() {
  if curl -fsS "http://127.0.0.1:$GATEWAY_PORT/api/version" >/dev/null 2>&1; then
    echo "Model gateway already running on port $GATEWAY_PORT; reusing it."
  elif ! tmux list-windows -t "$SESSION" -F '#W' | grep -qx gateway; then
    tmux new-window -d -t "$SESSION" -n gateway -c "$PROJECT_DIR"
    tmux send-keys -t "$SESSION:gateway" "./scripts/start_gateway.sh" Enter
  fi
}

if tmux has-session -t "$SESSION" 2>/dev/null; then
  start_gateway
  echo "tmux session '$SESSION' already running:"
  tmux list-windows -t "$SESSION"
  echo "Platform: http://127.0.0.1:8501   Attach: tmux attach -t $SESSION"
  exit 0
fi

"$PROJECT_DIR/scripts/start_db.sh"

tmux new-session -d -s "$SESSION" -n app -c "$PROJECT_DIR"
if curl -fsS http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
  echo "Ollama already running elsewhere; reusing it."
else
  tmux new-window -d -t "$SESSION" -n ollama -c "$PROJECT_DIR" "./scripts/start_ollama.sh"
  for _ in $(seq 1 20); do
    curl -fsS http://127.0.0.1:11434/api/version >/dev/null 2>&1 && break
    sleep 1
  done
fi
start_gateway
tmux send-keys -t "$SESSION:app" "./scripts/start_app.sh --server.headless true" Enter

for _ in $(seq 1 30); do
  curl -fsS http://127.0.0.1:8501/_stcore/health >/dev/null 2>&1 && break
  sleep 1
done
echo "Platform: http://127.0.0.1:8501 (tmux session '$SESSION'; attach with: tmux attach -t $SESSION)"
echo "Model gateway: http://127.0.0.1:$GATEWAY_PORT (window 'gateway')"
