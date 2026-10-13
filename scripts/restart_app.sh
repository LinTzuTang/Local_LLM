#!/bin/sh
# Restart the Streamlit platform inside the tmux session (needed after changing code in src/).
set -eu
SESSION=local_llm
if ! tmux has-session -t "$SESSION" 2>/dev/null; then
  exec "$(dirname -- "$0")/tmux_up.sh"
fi
tmux send-keys -t "$SESSION:app" C-c
sleep 2
tmux send-keys -t "$SESSION:app" "./scripts/start_app.sh --server.headless true" Enter
for _ in $(seq 1 30); do
  curl -fsS http://127.0.0.1:8501/_stcore/health >/dev/null 2>&1 && break
  sleep 1
done
echo "Platform restarted: http://127.0.0.1:8501"
