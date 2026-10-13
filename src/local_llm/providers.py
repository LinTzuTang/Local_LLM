import os

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from . import ollama_admin
from .config import Settings


def build_model(config: Settings, think: bool | None = None, temperature: float = 0):
    """`think=None` follows LOCAL_LLM_QWEN_NO_THINK; models that always think get think=True."""
    want_thinking = (not config.qwen_no_think) if think is None else think
    if config.provider == "ollama":
        return ChatOllama(
            model=config.model,
            base_url=config.base_url,
            temperature=temperature,
            num_ctx=config.context,
            num_gpu=int(os.environ["LOCAL_LLM_NUM_GPU"])
            if "LOCAL_LLM_NUM_GPU" in os.environ
            else None,
            num_predict=config.max_tokens,
            reasoning=ollama_admin.think_value(config.model, want_thinking, config.base_url),
            client_kwargs={"timeout": config.timeout},
        )
    extra = {}
    if "qwen3" in config.model.lower():
        extra["chat_template_kwargs"] = {"enable_thinking": want_thinking}
    return ChatOpenAI(
        model=config.model,
        base_url=config.base_url,
        api_key=config.api_key,
        temperature=temperature,
        max_tokens=config.max_tokens,
        timeout=config.timeout,
        max_retries=0,
        extra_body=extra,
    )
