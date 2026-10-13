import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from rich.console import Console

from .agent import build_agent
from .config import settings
from .providers import build_model

console = Console()


def main():
    if sys.argv[1:2] == ["code"]:  # the local coding agent has its own options
        from .coder.repl import main as code_main

        return code_main(sys.argv[2:])
    parser = argparse.ArgumentParser(
        description="LangChain local LLM demo",
        epilog="`local-agent code --help`: the fully local coding agent for this project.",
    )
    parser.add_argument(
        "mode", choices=["doctor", "chat", "agent", "code"], nargs="?", default="agent"
    )
    parser.add_argument("--prompt", default="Use the calculator to compute 37 * 19 + 8.")
    parser.add_argument("--provider", choices=["ollama", "vllm"])
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument(
        "--trace-file", help="Write local JSON trace (prompts and tool results included)"
    )
    parser.add_argument(
        "--require-tool", action="store_true", help="Fail if the agent skipped tools"
    )
    args = parser.parse_args()
    try:
        config = settings(args.provider, args.model, args.base_url)
        console.print(f"Backend: {config.provider} / {config.model} / {config.base_url}")
        if args.mode == "doctor":
            endpoint = "/api/tags" if config.provider == "ollama" else "/models"
            with httpx.Client(timeout=10, trust_env=False) as client:
                response = client.get(config.base_url + endpoint)
                response.raise_for_status()
                data = response.json()
            console.print_json(data=data)
            models = (
                [m["name"] for m in data.get("models", [])]
                if config.provider == "ollama"
                else [m["id"] for m in data.get("data", [])]
            )
            if config.model not in models:
                raise ValueError(
                    f"Configured model {config.model!r} not found; available: {models}"
                )
            return
        started = time.perf_counter()
        if args.mode == "chat":
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "You are a helpful local assistant. Keep your answer short."),
                    ("human", "{question}"),
                ]
            )
            response = (prompt | build_model(config)).invoke({"question": args.prompt})
            messages = [response]
        else:
            result = build_agent(config).invoke(
                {"messages": [{"role": "user", "content": args.prompt}]},
                config={"recursion_limit": 8},
            )
            messages = result["messages"]
        duration = time.perf_counter() - started
        tool_count = sum(isinstance(m, ToolMessage) for m in messages)
        for msg in messages:
            if isinstance(msg, ToolMessage):
                console.print(f"Tool {msg.name}: {msg.content}", markup=False)
        console.print(messages[-1].content, markup=False)
        console.print(f"Time: {duration:.2f}s / tool results: {tool_count}")
        if args.trace_file:
            path = Path(args.trace_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "recorded_at": datetime.now().astimezone().isoformat(),
                        "provider": config.provider,
                        "model": config.model,
                        "base_url": config.base_url,
                        "prompt": args.prompt,
                        "seconds": round(duration, 3),
                        "tool_results": tool_count,
                        "messages": [m.model_dump(mode="json") for m in messages],
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n"
            )
        if args.require_tool and tool_count == 0:
            raise ValueError(
                "The model answered without calling a tool; this is not a passing agent demo"
            )
    except Exception as exc:
        console.print(f"Error: {type(exc).__name__}: {exc}", style="red", markup=False)
        console.print("Start Ollama: ./scripts/start_ollama.sh ; download: ./scripts/pull_model.sh")
        sys.exit(1)


if __name__ == "__main__":
    main()
