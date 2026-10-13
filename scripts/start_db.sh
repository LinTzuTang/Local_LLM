#!/bin/sh
# Start the simulated company PostgreSQL in the background (127.0.0.1:5433, data in storage/pgdata).
# First run: creates the cluster with a random superuser password saved to .env, then loads
# the fictional company data. Stop with ./scripts/stop_db.sh.
set -eu
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$PROJECT_DIR"
export PATH="/opt/homebrew/opt/postgresql@18/bin:$HOME/.local/bin:$PATH"
command -v pg_ctl >/dev/null || { echo "PostgreSQL not found: brew install postgresql@18" >&2; exit 1; }
DATA="$PROJECT_DIR/storage/pgdata"
PORT=5433
mkdir -p "$PROJECT_DIR/storage"
touch .env

if [ ! -f "$DATA/PG_VERSION" ]; then
  if ! grep -q '^LOCAL_LLM_PG_SUPERUSER_PASSWORD=' .env; then
    echo "LOCAL_LLM_PG_SUPERUSER_PASSWORD=$(openssl rand -hex 16)" >> .env
  fi
  PWFILE=$(mktemp)
  trap 'rm -f "$PWFILE"' EXIT
  grep '^LOCAL_LLM_PG_SUPERUSER_PASSWORD=' .env | cut -d= -f2 > "$PWFILE"
  initdb -D "$DATA" -U postgres -E UTF8 --no-locale \
    --auth-local=scram-sha-256 --auth-host=scram-sha-256 --pwfile="$PWFILE" >/dev/null
  echo "Created PostgreSQL cluster in storage/pgdata"
  FRESH=1
fi

if pg_ctl -D "$DATA" status >/dev/null 2>&1; then
  echo "PostgreSQL already running on 127.0.0.1:$PORT"
else
  # Listen on loopback only; the unix socket stays inside storage/.
  pg_ctl -D "$DATA" -l "$PROJECT_DIR/storage/postgres.log" -w \
    -o "-p $PORT -c listen_addresses=127.0.0.1 -k $DATA" start >/dev/null
  echo "PostgreSQL started on 127.0.0.1:$PORT (log: storage/postgres.log)"
fi

if [ "${FRESH:-0}" = 1 ]; then
  uv run python -m local_llm.company_db init
fi
