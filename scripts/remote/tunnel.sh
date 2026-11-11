#!/bin/sh
# Forward the remote platform to this Mac: http://127.0.0.1:8501 (Streamlit) and 127.0.0.1:11500
# (model gateway; the microphone pages' browser code connects to it directly, so it must keep the
# same port). Localhost counts as a secure origin, so the browser allows the microphone.
# Stop the tunnel with Ctrl-C; the remote platform keeps running.
set -eu
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$HERE/config.sh"

GATEWAY_PORT="${LOCAL_LLM_GATEWAY_PORT:-11500}"
for port in 8501 "$GATEWAY_PORT"; do
  if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "Port $port is already in use on this Mac (the local platform?). Stop it first:" >&2
    echo "  ./scripts/tmux_down.sh" >&2
    exit 1
  fi
done
ip=$(vm_ip)
echo "Tunnel to $GCP_INSTANCE ($ip): open http://127.0.0.1:8501  (Ctrl-C to close)"
# shellcheck disable=SC2046
exec ssh -N $(ssh_opts) -o ExitOnForwardFailure=yes \
  -L 8501:127.0.0.1:8501 -L "$GATEWAY_PORT:127.0.0.1:$GATEWAY_PORT" "$REMOTE_USER@$ip"
