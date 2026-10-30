#!/bin/sh
# OCR models for the platform's "Compare OCR" tab (local_llm/ocr.py):
#   MiniCPM-OCR 1.0 into Ollama (Q8_0, vision projector, DSpark draft head; Ollama 0.40.12+),
#   MiniCPM-OCR Jev for llama-server (the gateway starts it on demand; brew install llama.cpp),
#   GLM-OCR and DeepSeek-OCR from the Ollama library.
#
#   ./scripts/setup_ocr_models.sh
set -eu
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$PROJECT_DIR"
HF=".venv/bin/hf"
OCR="models/ocr/MiniCPM-OCR-1.0-gguf"
JEV="models/ocr/MiniCPM-OCR-Jev-gguf"

"$HF" download openbmb/MiniCPM-OCR-1.0-gguf MiniCPM-OCR-1_0-Q8_0.gguf mmproj-model-f16-4x.gguf --local-dir "$OCR"
"$HF" download openbmb/MiniCPM-OCR-1.0-DSpark-gguf dspark-MiniCPM-OCR-1_0-Q8_0.gguf --local-dir "$OCR"
"$HF" download openbmb/MiniCPM-OCR-Jev-gguf MiniCPM-OCR-Jev-Q8_0.gguf mmproj-model-f16.gguf --local-dir "$JEV"

# The official Modelfile (OpenBMB/MiniCPM-OCR docs/ollama_deploy.md).
cat > "$OCR/Modelfile" <<'EOF'
FROM ./MiniCPM-OCR-1_0-Q8_0.gguf
FROM ./mmproj-model-f16-4x.gguf
DRAFT ./dspark-MiniCPM-OCR-1_0-Q8_0.gguf
REQUIRES 0.40.12
PARAMETER num_ctx 8192
PARAMETER temperature 0
PARAMETER draft_num_predict 15
EOF
(cd "$OCR" && ollama create minicpm-ocr -f Modelfile)
ollama pull glm-ocr
ollama pull deepseek-ocr:3b
command -v llama-server >/dev/null || echo "Jev needs llama-server: brew install llama.cpp"
