#!/bin/sh
# Copy the project's code and data (not models, virtualenv, results or secrets) to the GPU machine.
# Models are downloaded on the machine itself (Linux needs other builds than the Mac's MLX ones),
# storage/ keeps the remote's own state (database, sessions), and .env stays on this Mac: the
# remote .env is written by setup_linux.sh and holds no cloud keys unless you add them there.
set -eu
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_DIR=$(CDPATH= cd -- "$HERE/../.." && pwd)
. "$HERE/config.sh"

ip=$(vm_ip)
[ -n "$ip" ] || { echo "$GCP_INSTANCE has no external IP (is it running?)" >&2; exit 1; }
echo "Copying $PROJECT_DIR -> $REMOTE_USER@$ip:~/$REMOTE_DIR/"
rsync -az --delete \
  --exclude=.git/ --exclude=.venv/ --exclude=.runtime/ --exclude=__pycache__/ \
  --exclude=.pytest_cache/ --exclude=.ruff_cache/ \
  --exclude=/models/ --exclude=/storage/ --exclude=/runs/ \
  --exclude=.env --exclude='.env.*' --include=.env.example \
  --exclude=.DS_Store \
  -e "ssh $(ssh_opts | tr '\n' ' ')" \
  "$PROJECT_DIR/" "$REMOTE_USER@$ip:$REMOTE_DIR/"
