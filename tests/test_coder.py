"""The local coding agent's file tools: path safety, exact edits, backups, undo, checks.

No models are needed: the approval flow runs with a scripted fake chat model.
"""

import json

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from local_llm.coder import agent as coder_agent
from local_llm.coder.agent import (
    CoderSession,
    _answered,
    best_coding_model,
    code_tools,
    delete_session,
    list_sessions,
)
from local_llm.coder.workspace import (
    EditError,
    PathError,
    Workspace,
    denied_reason,
    glob_regex,
)
from local_llm.lab import control, store


@pytest.fixture
def ws(tmp_path) -> Workspace:
    root = tmp_path / "proj"
    (root / "src" / "pkg").mkdir(parents=True)
    (root / "src" / "pkg" / "mod.py").write_text(
        "def f():\n    return 1\n\n\ndef g():\n    return 1\n"
    )
    (root / "tests").mkdir()
    (root / "README.md").write_text("hello\n")
    (root / ".env").write_text("SECRET=hunter2\n")
    (root / ".env.example").write_text("SECRET=\n")
    (root / "models" / "blobs").mkdir(parents=True)
    (root / "models" / "blobs" / "weights.bin").write_text("SECRET weights")
    (root / "storage" / "pgdata").mkdir(parents=True)
    (root / ".venv" / "bin").mkdir(parents=True)
    (root / "data" / "lab").mkdir(parents=True)
    (root / "data" / "lab" / "agents.json").write_text('{"agents": []}\n')
    return Workspace(root)


# --- path safety ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "../outside.txt",
        "/etc/passwd",
        ".env",
        "src/.env",
        ".env.local",
        "models/blobs/weights.bin",
        "storage/pgdata/PG_VERSION",
        ".venv/bin/python",
        ".git/config",
        "server.key",
    ],
)
def test_paths_outside_or_secret_are_refused(ws, path):
    with pytest.raises(PathError):
        ws.resolve(path)


def test_symlink_out_of_the_project_is_refused(ws, tmp_path):
    (tmp_path / "secret.txt").write_text("outside")
    (ws.root / "link.txt").symlink_to(tmp_path / "secret.txt")
    with pytest.raises(PathError, match="outside the project"):
        ws.read_file("link.txt")


def test_write_only_restrictions(ws):
    assert ws.resolve("data/lab/agents.json")  # readable
    for path in ("data/lab/agents.json", "data/lab/scenarios.json", "storage/x.txt", "runs/a"):
        assert denied_reason(path, write=True)
        with pytest.raises(PathError):
            ws.resolve(path, write=True)
    assert "configuration tools" in denied_reason("data/lab/agents.json", write=True)
    assert ws.resolve(".env.example")  # templates are fine


def test_listing_and_search_hide_secrets_and_weights(ws):
    listing = ws.list_files("**/*")
    assert "src/pkg/mod.py" in listing and "README.md" in listing
    assert ".env\n" not in listing + "\n" and "models/" not in listing and ".venv" not in listing
    assert ws.list_files("*.py").splitlines()[1:] == ["src/pkg/mod.py"]
    assert "src/pkg/mod.py" in ws.list_files("src")
    found = ws.search("SECRET")
    assert ".env.example" in found
    assert "hunter2" not in found and "weights" not in found


def test_glob_regex():
    assert glob_regex("src/**/*.py").match("src/a/b/c.py")
    assert glob_regex("src/**/*.py").match("src/c.py")
    assert not glob_regex("src/*.py").match("src/a/c.py")
    assert glob_regex("*.{py,md}").match("x.md")


def test_read_file_numbers_lines_and_pages(ws):
    out = ws.read_file("src/pkg/mod.py", start=2, end=3)
    assert "lines 2-3 of 6" in out
    assert "    2\t    return 1" in out and "    1\t" not in out
    assert "more: read_file" in out
    with pytest.raises(PathError, match="no such file"):
        ws.read_file("src/pkg/nope.py")


# --- exact edits ---------------------------------------------------------------------------


def test_edit_must_match_exactly_once(ws):
    with pytest.raises(EditError, match="occurs 2 times"):
        ws.plan_edit("src/pkg/mod.py", "return 1", "return 2")
    with pytest.raises(EditError, match="not found"):
        ws.plan_edit("src/pkg/mod.py", "return 3", "return 2")
    with pytest.raises(EditError, match="line-number prefixes"):
        ws.plan_edit("src/pkg/mod.py", "    2\t    return 1", "x")
    with pytest.raises(EditError, match="identical"):
        ws.plan_edit("src/pkg/mod.py", "def f", "def f")
    change = ws.plan_edit("src/pkg/mod.py", "def g():\n    return 1", "def g():\n    return 2")
    assert change.action == "edit"
    assert "-    return 1" in change.diff() and "+    return 2" in change.diff()
    assert change.counts() == (1, 1)


def test_write_file_only_creates(ws):
    with pytest.raises(EditError, match="already exists"):
        ws.plan_write("README.md", "x")
    change = ws.plan_write("src/pkg/new.py", "X = 1")
    assert change.action == "create" and change.after == "X = 1\n"
    assert change.diff().startswith("--- /dev/null")


# --- backups and undo ----------------------------------------------------------------------


def test_backups_and_undo(ws):
    path = ws.root / "src" / "pkg" / "mod.py"
    original = path.read_text()
    edit = ws.apply(
        ws.plan_edit("src/pkg/mod.py", "def f():\n    return 1", "def f():\n    return 5")
    )
    new = ws.apply(ws.plan_write("src/pkg/new.py", "X = 1\n"))
    assert "return 5" in path.read_text() and (ws.root / "src/pkg/new.py").exists()
    backup = ws.backup_dir / edit["id"]
    assert (backup / "before").read_text() == original
    assert [c["id"] for c in ws.changes()] == [new["id"], edit["id"]]
    assert ws.code_changed_since(0)

    ws.undo()  # latest first: removes the created file
    assert not (ws.root / "src/pkg/new.py").exists()
    ws.undo(edit["id"])
    assert path.read_text() == original
    assert all(c.get("undone_at") for c in ws.changes())
    assert not ws.code_changed_since(0)
    with pytest.raises(EditError, match="nothing to undo"):
        ws.undo()


def test_undo_refuses_when_file_changed_later(ws):
    entry = ws.apply(ws.plan_edit("src/pkg/mod.py", "def f():", "def first():"))
    (ws.root / "src/pkg/mod.py").write_text("changed by hand\n")
    with pytest.raises(EditError, match="changed after"):
        ws.undo(entry["id"])


def test_apply_refuses_stale_plan(ws):
    change = ws.plan_edit("src/pkg/mod.py", "def f():", "def first():")
    (ws.root / "src/pkg/mod.py").write_text("something else\n")
    with pytest.raises(EditError, match="changed since"):
        ws.apply(change)


# --- checks --------------------------------------------------------------------------------


def test_only_allowlisted_checks_run(ws):
    with pytest.raises(ValueError, match="unknown check"):
        ws.run_check("rm -rf /")
    with pytest.raises(PathError):
        ws.run_check("pytest", "../")
    assert ws.run_check("compileall", "src").splitlines()[1].startswith("PASSED")
    (ws.root / "src" / "pkg" / "broken.py").write_text("def broken(:\n")
    assert "FAILED" in ws.run_check("compileall", "src/pkg/broken.py")


def test_pytest_check_reports_failures(ws):
    (ws.root / "tests" / "test_x.py").write_text("def test_x():\n    assert 1 == 2\n")
    out = ws.run_check("pytest", "tests/test_x.py")
    assert "FAILED" in out and "assert 1 == 2" in out


# --- tools and the approval flow -------------------------------------------------------------


def test_tools_return_errors_instead_of_raising(ws):
    read, write = code_tools(ws)
    tools = {t.name: t for t in read + write}
    assert set(tools) == {
        "list_files",
        "read_file",
        "search",
        "run_check",
        "edit_file",
        "write_file",
    }
    assert tools["read_file"].invoke({"path": ".env"}).startswith("ERROR")
    result = tools["edit_file"].invoke({"path": "src/pkg/mod.py", "old": "nope", "new": "x"})
    assert result.startswith("ERROR") and "not found" in result


def test_best_coding_model_prefers_coding_models():
    assert best_coding_model(["qwen3:1.7b", "qwen3:30b-a3b"]) == "qwen3:30b-a3b"
    assert (
        best_coding_model(["qwen3:30b-a3b", coder_agent.CODING_MODELS[0]])
        == (coder_agent.CODING_MODELS[0])
    )
    assert best_coding_model(["llama3:8b"]) is None


def test_unanswered_tool_calls_are_dropped_from_history():
    call = {"name": "edit_file", "args": {}, "id": "c1", "type": "tool_call"}
    done = {"name": "read_file", "args": {}, "id": "c2", "type": "tool_call"}
    messages = [
        AIMessage("", tool_calls=[done]),
        ToolMessage("ok", tool_call_id="c2"),
        AIMessage("", tool_calls=[call]),
    ]
    assert _answered(messages) == messages[:2]


class ScriptedModel(BaseChatModel):
    """Plays back AI messages (with tool calls) and accepts tool binding."""

    script: list

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self.script.pop(0))])

    def bind_tools(self, tools, **kwargs):
        return self


def _call(name: str, args: dict, i: int) -> AIMessage:
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": f"call{i}"}])


def test_session_pauses_for_approval_and_skips_invalid_edits(ws, monkeypatch):
    script = [
        _call("edit_file", {"path": "src/pkg/mod.py", "old": "missing", "new": "x"}, 1),
        _call("edit_file", {"path": "src/pkg/mod.py", "old": "def g():", "new": "def h():"}, 2),
        AIMessage("done"),
    ]
    monkeypatch.setattr(coder_agent, "build_model", lambda *a, **k: ScriptedModel(script=script))
    session = CoderSession("fake", root=ws.root)
    asked = []

    def decide(requests, previews):
        asked.append((requests, previews))
        return [{"type": "approve"}]

    added = session.send("rename g to h", decide)
    # The edit that cannot apply did not ask the human; its error went back to the model.
    assert len(asked) == 1
    (request,), (preview,) = asked[0]
    assert request["name"] == "edit_file" and preview["ok"]
    assert "+def h():" in preview["diff"]
    errors = [m for m in added if isinstance(m, ToolMessage) and "not found" in str(m.content)]
    assert errors
    assert "def h():" in (ws.root / "src/pkg/mod.py").read_text()
    assert added[-1].content == "done"
    # The change is in the session's change log and can be undone.
    (change,) = session.undoable()
    assert change["type"] == "code"
    session.undo(change)
    assert "def g():" in (ws.root / "src/pkg/mod.py").read_text()


def test_rejection_reason_reaches_the_model(ws, monkeypatch):
    script = [
        _call("write_file", {"path": "src/pkg/extra.py", "content": "X = 1\n"}, 1),
        AIMessage("ok, not creating it"),
    ]
    monkeypatch.setattr(coder_agent, "build_model", lambda *a, **k: ScriptedModel(script=script))
    session = CoderSession("fake", root=ws.root)
    added = session.send(
        "add a module", lambda r, p: [{"type": "reject", "message": "use src/pkg/mod.py"}]
    )
    assert not (ws.root / "src/pkg/extra.py").exists()
    rejected = next(m for m in added if isinstance(m, ToolMessage))
    assert "use src/pkg/mod.py" in str(rejected.content)


def test_sessions_are_saved_listed_and_resumed(ws, monkeypatch):
    scripts = iter(
        [
            [_call("list_agents", {}, 1), AIMessage("it defines g")],
            [AIMessage("still g")],
        ]
    )  # one script per model build: the first session, then the resumed one
    monkeypatch.setattr(
        coder_agent, "build_model", lambda *a, **k: ScriptedModel(script=next(scripts))
    )
    session = CoderSession("fake", root=ws.root, mode="settings")
    session.send("what is in mod.py?", lambda r, p: [])
    (row,) = list_sessions(ws.root)
    assert row["id"] == session.id and row["title"] == "what is in mod.py?"
    assert row["turns"] == 1 and row["mode"] == "settings" and "messages" not in row
    resumed = CoderSession.resume(session.id, root=ws.root)
    assert resumed.mode == "settings" and resumed.title == session.title
    assert [type(m) for m in resumed.history] == [type(m) for m in session.history]
    assert resumed.history[1].tool_calls[0]["name"] == "list_agents"
    added = resumed.send("and now?", lambda r, p: [])
    assert added[-1].content == "still g"
    assert list_sessions(ws.root)[0]["turns"] == 2
    delete_session(session.id, ws.root)
    assert list_sessions(ws.root) == []
    with pytest.raises(ValueError):
        delete_session("../x", ws.root)


# --- configuration tools for another copy of the project -----------------------------------


def test_config_preview_in_process(tmp_path, monkeypatch):
    for name in ("scenarios.json", "agents.json"):
        (tmp_path / name).write_text((store.LAB_DIR / name).read_text())
    monkeypatch.setattr(store, "SCENARIOS_FILE", tmp_path / "scenarios.json")
    monkeypatch.setattr(store, "AGENTS_FILE", tmp_path / "agents.json")
    monkeypatch.setattr(store, "HISTORY_DIR", tmp_path / "history")
    args = {"scenario_id": "hybrid_rag", "node_id": "answer", "changes": {"label": {"en": "A"}}}
    preview = control.preview("update_node", args)
    assert preview["ok"] and '"en": "A"' in preview["diff"]
    bad = control.preview("update_node", {**args, "node_id": "nope"})
    assert not bad["ok"] and "no node" in bad["errors"][0]


def test_config_tools_read_another_copy(tmp_path):
    lab = tmp_path / "data" / "lab"
    lab.mkdir(parents=True)
    agents = {"agents": [{"id": "x", "name": "X", "tools": [], "system_prompt": "s"}]}
    (lab / "agents.json").write_text(json.dumps(agents))
    (lab / "scenarios.json").write_text('{"scenarios": []}')
    read, write = control.config_tools(tmp_path)
    tools = {t.name: t for t in read + write}
    assert json.loads(tools["list_agents"].invoke({}))[0]["id"] == "x"
    assert tools["show_agent"].invoke({"agent_id": "nope"}).startswith("ERROR")
    assert control.history(tmp_path) == []
