#!/bin/sh
set -eu
[ "$(uname -s)" = Linux ] && [ "$(uname -m)" = aarch64 ] || { echo "Run on DGX Spark Arm64 Linux." >&2; exit 1; }
export PATH="$HOME/.local/bin:$PATH"
uvx --from huggingface-hub hf download nvidia/Qwen3.8-27B-NVFP4 --cache-dir "$HOME/.cache/huggingface/hub"
