#!/bin/sh
set -eu
[ "$(uname -s)" = Linux ] && [ "$(uname -m)" = aarch64 ] || { echo "Run on DGX Spark Arm64 Linux." >&2; exit 1; }
# Source: NVIDIA DGX Spark playbook, retrieved 2026-10-06; not tested on Spark yet.
# The loopback binding, context and sequence limits below are project choices.
exec docker run --rm --name local-llm-vllm --gpus all --ipc host \
  --ulimit memlock=-1 --ulimit stack=67108864 --entrypoint "" \
  -p 127.0.0.1:8000:8000 \
  -v "$HOME/.cache/huggingface/hub:/root/.cache/huggingface/hub:ro" \
  -e HF_HUB_OFFLINE=1 -e VLLM_USE_RUST_FRONTEND=1 \
  vllm/vllm-openai:qwen38 \
  vllm serve nvidia/Qwen3.8-27B-NVFP4 \
  --host 0.0.0.0 --port 8000 --tensor-parallel-size 1 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.7 \
  --max-model-len 8192 --max-num-seqs 2 --max-num-batched-tokens 4096 \
  --enable-chunked-prefill --async-scheduling --enable-prefix-caching \
  --load-format fastsafetensors --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml
