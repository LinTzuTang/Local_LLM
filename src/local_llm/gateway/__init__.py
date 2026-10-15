"""Model gateway: an Ollama-style local server for models Ollama cannot run.

Speech-to-text (mlx-whisper, mlx-audio, and Gemma 4 audio forwarded to the local Ollama),
document parsing (Docling) and a reranker (sentence-transformers) get what Ollama gives LLMs:
download with progress (/api/pull), lazy loading on first use, unloading after `keep_alive`,
/api/tags and /api/ps, plus OpenAI-style inference endpoints.

    uv run python -m local_llm.gateway      # http://127.0.0.1:11500, localhost only

Weights stay inside the project, next to models/ollama: Hugging Face downloads in models/hf
(HF_HOME), Docling artifacts in models/docling, locally converted MLX weights in models/mlx,
and one manifest per downloaded model in models/gateway.
"""

import os
from pathlib import Path

VERSION = "0.1.0"
PROJECT_DIR = Path(__file__).resolve().parents[3]
MODELS_DIR = PROJECT_DIR / "models"
HOST = "127.0.0.1"
PORT = int(os.getenv("LOCAL_LLM_GATEWAY_PORT", "11500"))
# Seconds a model stays in memory after its last request (Ollama's default is 5 minutes too).
DEFAULT_KEEP_ALIVE = float(os.getenv("LOCAL_LLM_GATEWAY_KEEP_ALIVE", "300"))
MAX_UPLOAD_BYTES = int(float(os.getenv("LOCAL_LLM_GATEWAY_MAX_UPLOAD_MB", "200")) * 1024**2)

# Hugging Face libraries read these when they are imported, so set them first.
os.environ["HF_HOME"] = str(MODELS_DIR / "hf")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
