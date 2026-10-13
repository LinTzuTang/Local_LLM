#!/bin/sh
set -eu
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
export PATH="/opt/homebrew/opt/postgresql@18/bin:$PATH"
pg_ctl -D "$PROJECT_DIR/storage/pgdata" -m fast stop
