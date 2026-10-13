#!/bin/sh
# Stop the tmux session: platform, model gateway, and Ollama if it was started there.
# Database: ./scripts/stop_db.sh
set -eu
SESSION=local_llm
if tmux has-session -t "$SESSION" 2>/dev/null; then
  # Let the gateway shut down cleanly (it may be in the middle of writing a download).
  if tmux list-windows -t "$SESSION" -F '#W' | grep -qx gateway; then
    tmux send-keys -t "$SESSION:gateway" C-c
    sleep 1
  fi
  tmux kill-session -t "$SESSION" && echo "Stopped tmux session $SESSION"
else
  echo "No tmux session $SESSION"
fi
