#!/bin/sh
set -eu
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
export PATH="/opt/homebrew/opt/postgresql@18/bin:$PATH"
# Homebrew (macOS) puts pg_ctl on PATH; Debian/Ubuntu keep it in /usr/lib/postgresql/<ver>/bin.
for d in /usr/lib/postgresql/*/bin; do [ -x "$d/pg_ctl" ] && PATH="$d:$PATH"; done
pg_ctl -D "$PROJECT_DIR/storage/pgdata" -m fast stop
