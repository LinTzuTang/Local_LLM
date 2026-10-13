from langchain.agents import create_agent

from .config import Settings
from .providers import build_model
from .tools import calculator, search_models


def build_agent(config: Settings):
    return create_agent(
        model=build_model(config),
        tools=[calculator, search_models],
        system_prompt=(
            "You are a local assistant. Use calculator for every arithmetic question. "
            "Call a tool before answering arithmetic; never estimate. "
            "Use search_models for model-catalog questions. After receiving a tool result, "
            "answer briefly using that result. Do not repeatedly call the same tool. "
            "Follow the user's language."
        ),
    )
