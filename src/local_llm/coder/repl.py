"""Terminal front end of the local coding agent: `uv run local-agent code`.

Streams the model's thinking and answer, shows every tool call, renders each proposed write as
a unified diff and asks y / n / e (reject with a reason) / a (approve the rest of this turn).
Commands: /undo, /changes, /model, /mode, /think, /clear, /help, /exit.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from .. import ollama_admin
from .agent import (
    CODE_WRITE_TOOLS,
    DEFAULT_CONTEXT,
    MODES,
    RECOMMENDED_MODEL,
    CoderSession,
    best_coding_model,
    ordered_models,
    run_stats,
)
from .workspace import PROJECT_ROOT, EditError

console = Console(highlight=False)

HELP = """[bold]Commands[/]
  /undo [n]        undo the latest change of this session (or number n from /undo list)
  /undo list       list changes that can be undone
  /changes         change log (code backups and configuration versions)
  /model [name|n]  list models or switch (conversation is kept)
  /mode settings|code   settings only, or settings + code
  /think on|off|auto    model thinking
  /clear           start a new conversation
  /exit            quit
Approvals: y = approve, n = reject, e = reject with a reason for the agent, a = approve all
remaining changes of this turn."""


def _short(value, limit: int = 70) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    text = text.replace("\n", "⏎")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _call_text(call: dict) -> str:
    args = ", ".join(f"{k}={_short(v, 50)}" for k, v in (call.get("args") or {}).items())
    return f"{call['name']}({args})"


class Printer:
    """Turns session events into terminal output."""

    def __init__(self, show_thinking: bool = True):
        self.show_thinking = show_thinking
        self.state = None  # "reasoning" | "text" | None (what is being streamed)

    def _switch(self, state: str | None) -> None:
        if self.state != state and self.state is not None:
            console.print()
        if state == "reasoning" and self.state != "reasoning":
            console.print("💭 ", style="dim", end="")
        self.state = state

    def __call__(self, kind: str, data) -> None:
        if kind == "reasoning":
            if self.show_thinking:
                self._switch("reasoning")
                console.print(data, style="dim italic", end="", markup=False)
            return
        if kind == "text":
            self._switch("text")
            console.print(data, end="", markup=False)
            return
        self._switch(None)
        if isinstance(data, AIMessage):
            for call in data.tool_calls:
                style = "bold yellow" if call["name"] in WRITES else "cyan"
                console.print(f"→ {_call_text(call)}", style=style, markup=False)
        elif isinstance(data, ToolMessage):
            text = str(data.content)
            lines = text.splitlines() or [""]
            failed = data.status == "error" or text.startswith(("ERROR", "NOT SAVED", "User rej"))
            if data.name == "run_check" and len(lines) > 1:
                ok = lines[1].startswith("PASSED")
                console.print(f"  ← {lines[0]}: {lines[1]}", style="green" if ok else "red")
                if not ok:
                    console.print(_tail(lines[2:], 12), style="dim", markup=False)
                return
            style = "red" if failed else "dim"
            more = f"  (+{len(lines) - 3} lines)" if len(lines) > 3 else ""
            head = "\n    ".join(_short(x, 110) for x in lines[:3])
            console.print(f"  ← {data.name}: {head}{more}", style=style, markup=False)


WRITES = set(CODE_WRITE_TOOLS) | {
    "update_agent",
    "create_agent",
    "update_node",
    "add_node",
    "remove_node",
    "set_edges",
    "save_scenario",
}


def _tail(lines: list[str], n: int) -> str:
    return "\n".join("    " + x for x in lines[-n:])


def show_preview(request: dict, preview: dict) -> None:
    kind = preview.get("kind")
    target = preview.get("id") or "?"
    counts = ""
    if kind == "code":
        counts = f"  +{preview.get('added', 0)} −{preview.get('removed', 0)}"
        title = f"✏️  {request['name']} · {target}{counts}"
    else:
        title = f"⚙️  {request['name']} · {kind} {target}"
    body = Syntax(preview.get("diff") or "(no change)", "diff", theme="ansi_dark", word_wrap=True)
    console.print(Panel(body, title=title, title_align="left", border_style="yellow"))
    for e in preview.get("errors", []):
        console.print(f"  ✗ {e}", style="red", markup=False)
    for w in preview.get("warnings", []):
        console.print(f"  ! {w}", style="yellow", markup=False)


def ask_decisions(requests: list[dict], previews: list[dict], auto: dict) -> list[dict]:
    """y / n / e(reason) / a for each proposed write."""
    decisions = []
    for request, preview in zip(requests, previews, strict=True):
        show_preview(request, preview)
        if auto.get("all"):
            console.print("  ✓ approved (approve-all)", style="green")
            decisions.append({"type": "approve"})
            continue
        while True:
            try:
                answer = console.input("  Apply? [bold]y[/]es / [bold]n[/]o / [bold]e[/] reason / "
                                       "[bold]a[/]ll › ").strip().lower()  # fmt: skip
            except EOFError:
                answer = "n"
            if answer in ("y", "yes", ""):
                decisions.append({"type": "approve"})
            elif answer in ("a", "all"):
                auto["all"] = True
                decisions.append({"type": "approve"})
            elif answer in ("n", "no"):
                decisions.append({"type": "reject", "message": "The user rejected this change."})
            elif answer in ("e", "edit", "r", "reason"):
                reason = console.input("  Reason / what to do instead › ").strip()
                message = f"The user rejected this change: {reason or 'no reason given'}"
                decisions.append({"type": "reject", "message": message})
            else:
                continue
            break
    return decisions


def _stats_line(messages, seconds: float, approvals: int) -> str:
    s = run_stats(messages)
    speed = f" · {s['tokens_per_s']} tok/s" if s["tokens_per_s"] else ""
    return (
        f"⏱ {seconds:.1f}s · {s['llm_calls']} LLM calls · {s['tool_calls']} tool calls · "
        f"{approvals} approvals · {s['output_tokens']} tokens out · "
        f"max prompt {s['max_prompt_tokens']} tokens{speed}"
    )


def _installed(base_url: str) -> list[str]:
    try:
        return [m["name"] for m in ollama_admin.installed(base_url)]
    except Exception:
        return []


def _tool_models(base_url: str) -> list[str]:
    return [m for m in ordered_models(_installed(base_url)) if _supports_tools(m, base_url)]


def _supports_tools(model: str, base_url: str) -> bool:
    try:
        return ollama_admin.supports_tools(model, base_url)
    except Exception:
        return False


def pick_model(wanted: str | None, base_url: str) -> str:
    installed = _installed(base_url)
    if wanted:
        names = {n if ":" in n else f"{n}:latest" for n in installed}
        if (wanted if ":" in wanted else f"{wanted}:latest") not in names:
            raise SystemExit(
                f"Model {wanted!r} is not installed. Installed: {', '.join(sorted(installed))}\n"
                f"Download it with: ollama pull {wanted}"
            )
        return wanted
    model = best_coding_model(installed)
    if model:
        return model
    tools = _tool_models(base_url)
    if not tools:
        raise SystemExit(f"No tool-calling model installed. Try: ollama pull {RECOMMENDED_MODEL}")
    return tools[0]


def _think(value: str | None) -> bool | None:
    return {"on": True, "off": False}.get((value or "auto").lower())


def handle_command(line: str, session: CoderSession, printer: Printer) -> bool:
    """Run a /command; returns False to quit."""
    cmd, _, arg = line[1:].partition(" ")
    arg = arg.strip()
    if cmd in ("exit", "quit", "q"):
        return False
    if cmd in ("help", "h", "?"):
        console.print(HELP)
    elif cmd == "clear":
        session.history = []
        console.print("New conversation.", style="green")
    elif cmd == "changes":
        table = Table("#", "when", "type", "change", "state", box=None)
        for i, c in enumerate(session.changes(20), 1):
            state = "undone" if c.get("undone_at") or c["id"] in session.undone_config else ""
            table.add_row(str(i), c["at"][5:], c["type"], c["label"], state)
        console.print(table)
    elif cmd == "undo":
        options = session.undoable()
        if not options:
            console.print("Nothing to undo in this session.", style="yellow")
        elif arg == "list":
            for i, c in enumerate(options, 1):
                console.print(f"  {i}. {c['at'][11:]} {c['type']}: {c['label']}", markup=False)
        else:
            try:
                change = options[int(arg) - 1] if arg else options[0]
                console.print(f"Undo: {change['label']}?", markup=False)
                if change["type"] == "code":
                    diff = session.ws.entry_change(change["id"]).diff()
                    console.print(Syntax(diff, "diff", theme="ansi_dark"))
                if console.input("  Confirm undo [y/N] › ").strip().lower() in ("y", "yes"):
                    console.print("✓ " + session.undo(change), style="green", markup=False)
            except (ValueError, IndexError):
                console.print("Usage: /undo, /undo list, /undo <n>", style="yellow")
            except EditError as exc:
                console.print(f"✗ {exc}", style="red", markup=False)
    elif cmd == "model":
        models = _tool_models(session.base_url)
        if not arg:
            for i, m in enumerate(models, 1):
                mark = "●" if m == session.model else " "
                console.print(f" {mark} {i}. {m}", markup=False)
            console.print("Switch with /model <name or number>", style="dim")
        else:
            model = models[int(arg) - 1] if arg.isdigit() and 0 < int(arg) <= len(models) else arg
            if model not in models:
                console.print(f"{model} is not an installed tool-calling model.", style="red")
            else:
                session.rebuild(model=model)
                console.print(f"Model: {model} (conversation kept)", style="green")
    elif cmd == "mode":
        if arg not in MODES:
            console.print(f"Mode is {session.mode}; use /mode settings or /mode code")
        else:
            session.rebuild(mode=arg)
            console.print(f"Mode: {arg}", style="green")
    elif cmd == "think":
        session.think = _think(arg)
        session.rebuild()
        console.print(f"Thinking: {arg or 'auto'}", style="green")
    elif cmd == "thinking":
        printer.show_thinking = not printer.show_thinking
    else:
        console.print(f"Unknown command /{cmd}. /help lists them.", style="yellow")
    return True


def run_turn(session: CoderSession, prompt: str, printer: Printer, approve_all: bool) -> dict:
    """One user turn with live output; returns a record for logs and benchmarks."""
    auto = {"all": approve_all}
    approvals: list[list[dict]] = []

    def decide(requests, previews):
        decisions = ask_decisions(requests, previews, auto)
        approvals.append(decisions)
        return decisions

    started = time.perf_counter()
    try:
        added = session.send(prompt, decide, printer)
        interrupted = False
    except KeyboardInterrupt:
        added, interrupted = [], True
        console.print("\n⏹ interrupted", style="yellow")
    printer._switch(None)
    seconds = time.perf_counter() - started
    count = sum(len(d) for d in approvals)
    console.print(_stats_line(added, seconds, count), style="dim")
    return {
        "prompt": prompt,
        "seconds": round(seconds, 1),
        "interrupted": interrupted,
        "approvals": count,
        "decisions": approvals,
        "stats": run_stats(added),
        "messages": [json.loads(m.model_dump_json()) for m in added],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="local-agent code", description="Fully local coding agent for this project"
    )
    parser.add_argument("--model", help=f"Ollama model (default: best installed coding model, "
                        f"recommended {RECOMMENDED_MODEL})")  # fmt: skip
    parser.add_argument("--mode", choices=MODES, default="code",
                        help="settings: agents/scenarios only; code: settings + code")  # fmt: skip
    parser.add_argument("--root", default=str(PROJECT_ROOT),
                        help="project folder (e.g. a copy to experiment in)")  # fmt: skip
    parser.add_argument("--base-url", default=os.getenv("LOCAL_LLM_BASE_URL", "") or None)
    parser.add_argument("--ctx", type=int, default=DEFAULT_CONTEXT, help="context window")
    parser.add_argument("--think", choices=["on", "off", "auto"], default="auto")
    parser.add_argument("--temperature", type=float, default=None,
                        help="default: the model's own setting")  # fmt: skip
    parser.add_argument("--prompt", "-p", help="run this one instruction and exit")
    parser.add_argument("--approve-all", action="store_true",
                        help="approve every write without asking (use on a copy)")  # fmt: skip
    parser.add_argument("--log", help="write a JSON record of the run(s) to this file")
    parser.add_argument("--hide-thinking", action="store_true")
    args = parser.parse_args(argv)

    base_url = args.base_url if args.base_url and "11434" in args.base_url else None
    base_url = base_url or ollama_admin.DEFAULT_URL
    if ollama_admin.version(base_url) is None:
        raise SystemExit(f"Ollama is not reachable at {base_url}. Run ./scripts/start_ollama.sh")
    model = pick_model(args.model, base_url)
    root = Path(args.root).expanduser().resolve()
    if not (root / "src" / "local_llm").is_dir():
        raise SystemExit(f"{root} does not look like this project (no src/local_llm)")
    session = CoderSession(
        model,
        base_url,
        mode=args.mode,
        root=root,
        context=args.ctx,
        think=_think(args.think),
        temperature=args.temperature,
    )
    printer = Printer(show_thinking=not args.hide_thinking)
    console.print(
        Panel.fit(
            f"[bold]Local coding agent[/] · model [cyan]{model}[/] · mode [cyan]{args.mode}[/] · "
            f"context {args.ctx} · think {args.think}\nproject {root}\n"
            "Everything runs on this machine. Writes need your approval; /help for commands.",
            border_style="blue",
        )
    )
    if args.approve_all:
        console.print("⚠ --approve-all: every write is applied without asking.", style="yellow")
    record = {
        "started": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": model,
        "mode": args.mode,
        "root": str(root),
        "context": args.ctx,
        "think": args.think,
        "turns": [],
    }

    def save_log():
        if args.log:
            Path(args.log).parent.mkdir(parents=True, exist_ok=True)
            Path(args.log).write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n")

    if args.prompt:
        record["turns"].append(run_turn(session, args.prompt, printer, args.approve_all))
        save_log()
        return
    try:
        import readline  # noqa: F401  (line editing and history for input())
    except ImportError:
        pass
    while True:
        try:
            line = console.input("\n[bold cyan]you ›[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if not line:
            continue
        if line.startswith("/"):
            if not handle_command(line, session, printer):
                break
            continue
        record["turns"].append(run_turn(session, line, printer, args.approve_all))
        save_log()
    save_log()


if __name__ == "__main__":
    main(sys.argv[1:])
