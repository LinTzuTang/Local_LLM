import os
from dataclasses import dataclass
from urllib.parse import urlparse

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    provider: str
    model: str
    base_url: str
    max_tokens: int = 4096
    context: int = 16384
    timeout: float = 180
    api_key: str = "local"
    qwen_no_think: bool = True

    def __post_init__(self):
        if self.provider not in {"ollama", "vllm"}:
            raise ValueError("provider must be ollama or vllm")
        url = urlparse(self.base_url)
        if url.scheme not in {"http", "https"} or not url.hostname:
            raise ValueError("base_url must be an explicit HTTP(S) endpoint")
        if url.username or url.password:
            raise ValueError("Do not place credentials in endpoint URLs")
        if self.provider == "vllm" and not url.path.rstrip("/").endswith("/v1"):
            raise ValueError("vLLM base_url must end in /v1")
        if min(self.max_tokens, self.context, self.timeout) <= 0:
            raise ValueError("Token limits and timeout must be positive")


def settings(provider=None, model=None, base_url=None):
    load_dotenv()
    # No LangSmith account is required; tracing is opt-in through user configuration.
    os.environ.setdefault("LANGSMITH_TRACING", "false")
    selected = provider or os.getenv("LOCAL_LLM_PROVIDER", "ollama")
    fallback_model = "qwen3:30b-a3b" if selected == "ollama" else "Qwen/Qwen3-8B"
    fallback_url = "http://127.0.0.1:11434" if selected == "ollama" else "http://127.0.0.1:8000/v1"
    # Explicit provider override does not inherit a different provider's endpoint/model.
    same_provider = selected == os.getenv("LOCAL_LLM_PROVIDER", "ollama")
    return Settings(
        provider=selected,
        model=model or (os.getenv("LOCAL_LLM_MODEL") if same_provider else None) or fallback_model,
        base_url=(
            base_url or (os.getenv("LOCAL_LLM_BASE_URL") if same_provider else None) or fallback_url
        ).rstrip("/"),
        max_tokens=int(os.getenv("LOCAL_LLM_MAX_TOKENS", "4096")),
        context=int(os.getenv("LOCAL_LLM_CONTEXT", "16384")),
        timeout=float(os.getenv("LOCAL_LLM_TIMEOUT", "180")),
        api_key=os.getenv("LOCAL_LLM_API_KEY", "local"),
        qwen_no_think=os.getenv("LOCAL_LLM_QWEN_NO_THINK", "true").lower() == "true",
    )
