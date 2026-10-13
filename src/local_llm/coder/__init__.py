"""A fully local coding agent for this project (no cloud model, no extra dependencies).

- `workspace`: project-scoped file tools (list, read, search, edit, write) with path safety,
  unified diffs, a backup of every changed file and undo, plus an allowlist of checks.
- `agent`: the LangChain agent that combines these code tools with the control center's
  configuration tools (agents.json / scenarios.json); every write waits for human approval.
- `repl`: the terminal front end, `uv run local-agent code`.
"""
