#!/bin/sh
# Build a portable zip without machine-specific files (venv, Ollama app, model weights, .env).
set -eu
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
NAME=$(basename "$PROJECT_DIR")
OUT="$PROJECT_DIR/dist/$NAME-$(date +%Y%m%d).zip"
mkdir -p "$PROJECT_DIR/dist"
rm -f "$OUT"
cd "$PROJECT_DIR/.."
zip -rq "$OUT" "$NAME" \
  -x "$NAME/.venv/*" "$NAME/.runtime/*" "$NAME/models/*" "$NAME/dist/*" \
     "$NAME/.env" "$NAME/.pytest_cache/*" "$NAME/.ruff_cache/*" \
     "*/__pycache__/*" "*.pyc" "*.DS_Store" "$NAME/.git/*"
echo "Created $OUT ($(du -h "$OUT" | cut -f1))"
