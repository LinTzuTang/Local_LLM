#!/bin/sh
# First-time setup on a new macOS or Linux machine after unzipping the project.
set -eu
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$PROJECT_DIR"
export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found; installing it to ~/.local/bin (official installer)."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# uv downloads Python 3.12 if needed and installs the exact locked dependencies.
uv sync --locked
uv run pytest -q
mkdir -p models/ollama runs

if command -v ollama >/dev/null 2>&1; then
  echo "Ollama found: $(command -v ollama)"
  echo "Next: ./scripts/start_ollama.sh, then in another terminal ./scripts/pull_model.sh"
else
  echo "Ollama is not installed. Install it, then run ./scripts/start_ollama.sh:"
  case "$(uname -s)" in
    Darwin) echo "  brew install ollama   (or download the app from https://ollama.com/download)" ;;
    Linux) echo "  curl -fsSL https://ollama.com/install.sh | sh" ;;
  esac
fi
