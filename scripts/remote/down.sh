#!/bin/sh
# Stop the platform on the GPU machine and power the machine off; waits until GCP reports
# TERMINATED (the stop call itself may answer 403 for lack of zoneOperations.get, yet the stop
# goes through, so the status is what counts).
#   ./scripts/remote/down.sh           stop platform + machine
#   ./scripts/remote/down.sh --keep    stop the platform only, keep the machine running
set -eu
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$HERE/config.sh"

if [ "$(vm_status)" = RUNNING ]; then
  remote "cd ~/$REMOTE_DIR 2>/dev/null && ./scripts/tmux_down.sh && ./scripts/stop_db.sh" || true
fi
[ "${1:-}" = --keep ] && exit 0
gc stop --async >/dev/null 2>&1 || true
for _ in $(seq 1 60); do
  status=$(vm_status)
  [ "$status" = TERMINATED ] && { echo "$GCP_INSTANCE is TERMINATED."; exit 0; }
  sleep 5
done
echo "$GCP_INSTANCE is still $status; check it in the console." >&2
exit 1
