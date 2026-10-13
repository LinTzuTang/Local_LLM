#!/bin/sh
set -eu
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$PROJECT_DIR"
export PATH="$HOME/.local/bin:$PATH"
uv sync --locked
uv run local-agent doctor
uv run local-agent chat --prompt "Translate to Traditional Chinese: A local LLM runs on this computer without sending prompts to the cloud." --trace-file runs/chat.json
uv run local-agent agent --require-tool --trace-file runs/agent.json
