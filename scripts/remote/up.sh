#!/bin/sh
# Start the GCP GPU machine (if it is stopped), copy this project to it, start the platform there
# and open an SSH tunnel so http://127.0.0.1:8501 on this Mac is the remote Local Agent Lab.
#   ./scripts/remote/up.sh            start + deploy + run + tunnel (Ctrl-C closes the tunnel only)
#   ./scripts/remote/up.sh --no-sync  skip copying the code
# The machine costs money while it runs: stop it with ./scripts/remote/down.sh when done.
set -eu
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$HERE/config.sh"

status=$(vm_status)
if [ "$status" != RUNNING ]; then
  echo "Starting $GCP_INSTANCE (was $status)..."
  gc start --async >/dev/null
  for _ in $(seq 1 60); do
    [ "$(vm_status)" = RUNNING ] && break
    sleep 5
  done
  [ "$(vm_status)" = RUNNING ] || { echo "Still not running; a STOCKOUT means try again later." >&2; exit 1; }
fi
ip=$(vm_ip)
echo "$GCP_INSTANCE is RUNNING at $ip"
for _ in $(seq 1 30); do  # sshd comes up a little after RUNNING
  remote true 2>/dev/null && break
  sleep 5
done

[ "${1:-}" = --no-sync ] || "$HERE/deploy.sh"
remote "cd ~/$REMOTE_DIR && ./scripts/tmux_up.sh"
exec "$HERE/tunnel.sh"
