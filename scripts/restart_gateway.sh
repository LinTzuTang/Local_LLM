#!/bin/sh
# Restart the model gateway inside the tmux session (needed after changing src/local_llm/gateway).
# Loaded models are dropped; they load again on their next request.
set -eu
SESSION=local_llm
PORT="${LOCAL_LLM_GATEWAY_PORT:-11500}"
if ! tmux has-session -t "$SESSION" 2>/dev/null; then
  exec "$(dirname -- "$0")/tmux_up.sh"
fi
if tmux list-windows -t "$SESSION" -F '#W' | grep -qx gateway; then
  tmux send-keys -t "$SESSION:gateway" C-c
  for _ in $(seq 1 20); do
    curl -fsS "http://127.0.0.1:$PORT/api/version" >/dev/null 2>&1 || break
    sleep 0.5
  done
  tmux send-keys -t "$SESSION:gateway" "./scripts/start_gateway.sh" Enter
else
  PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
  tmux new-window -d -t "$SESSION" -n gateway -c "$PROJECT_DIR"
  tmux send-keys -t "$SESSION:gateway" "./scripts/start_gateway.sh" Enter
fi
for _ in $(seq 1 30); do
  curl -fsS "http://127.0.0.1:$PORT/api/version" >/dev/null 2>&1 && break
  sleep 1
done
echo "Model gateway restarted: http://127.0.0.1:$PORT"
