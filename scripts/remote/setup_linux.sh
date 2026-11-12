#!/bin/sh
# First-time setup on an Ubuntu/Debian machine with an NVIDIA GPU (run ON that machine, after
# ./scripts/remote/deploy.sh copied the project there):
#   ssh ... 'cd ~/Local_LLM && ./scripts/remote/setup_linux.sh'
# Installs ffmpeg, tmux, PostgreSQL and Ollama (pinned to the Mac's version), then the Python
# environment (./scripts/setup.sh), and writes a .env without any cloud keys.
set -eu
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$PROJECT_DIR"
OLLAMA_VERSION="${OLLAMA_VERSION:-0.40.13}"

[ "$(uname -s)" = Linux ] || { echo "Run this on the Linux GPU machine." >&2; exit 1; }
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader \
  || echo "提醒：找不到 nvidia-smi，語音模型會退回 CPU。" >&2

sudo apt-get update -q
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q ffmpeg tmux postgresql rsync curl \
  fonts-noto-cjk
# The platform runs its own cluster (127.0.0.1:5433, storage/pgdata); the distro's one on 5432
# is not used.
sudo systemctl disable --now postgresql >/dev/null 2>&1 || true

if ! ollama --version 2>/dev/null | grep -q "$OLLAMA_VERSION"; then
  curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION="$OLLAMA_VERSION" sh
fi
# The installer also starts a system service with its own model folder; the platform starts
# Ollama itself (scripts/start_ollama.sh) with the models in models/ollama.
sudo systemctl disable --now ollama >/dev/null 2>&1 || true

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Wrote .env from .env.example (no cloud keys: add them by hand if this machine may use them)."
fi
# 96 GB RTX PRO 6000: leave a few GB for the CUDA context and the gateway's speech models.
grep -q '^LOCAL_LLM_GPU_BUDGET_GB=' .env || echo "LOCAL_LLM_GPU_BUDGET_GB=90" >> .env

./scripts/setup.sh
echo
echo "Done. Start it with ./scripts/tmux_up.sh (or ./scripts/remote/up.sh from the Mac),"
echo "then pull models: ./scripts/pull_model.sh, and the gateway models from the Models page."
