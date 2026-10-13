"""The tools & skills catalogue: Agent Skills in data/lab/skills and the MCP server tools."""

import dataclasses
import json
import re
import subprocess

import pytest

from local_llm import mcp_tools, skills

NAME = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


# --- Skills -------------------------------------------------------------------------------


def test_every_skill_follows_the_format():
    installed = skills.list_skills()
    assert len(installed) >= 9
    for s in installed:
        assert NAME.match(s["name"]) and len(s["name"]) <= 64, s["name"]
        assert s["path"].endswith("/" + s["name"]), "name must match the folder"
        assert 20 < len(s["description"]) <= 1024, s["name"]
        assert s["description_zh"], f"{s['name']} needs description_zh"
        assert s["license"], f"{s['name']} needs a license"
        assert skills.get_skill(s["name"])["body"].strip()
        if s["source"].startswith("http"):  # imported: keep the licence text next to it
            assert s["license"] in {"Apache-2.0", "MIT"}, s["name"]
            assert "LICENSE.txt" in s["files"], s["name"]
        else:
            assert s["source"] == "local"


def test_company_skills_carry_the_house_rules():
    sql = skills.get_skill("sql-analyst")["body"]
    assert "quantity * oi.unit_price" in sql and "已取消" in sql
    assert "升級" in skills.get_skill("customer-reply")["body"]
    names = {s["name"] for s in skills.list_skills()}
    assert {"meeting-minutes", "management-memo", "contract-review", "data-privacy-check",
            "zh-tw-style", "quality-8d-report"} <= names  # fmt: skip


def test_load_skill_reads_instructions_and_extra_files():
    load = skills.skill_tool(["internal-comms", "sql-analyst"])
    body = load.invoke({"name": "internal-comms"})
    assert "examples/faq-answers.md" in body  # lists the files it can read next
    assert "##" in load.invoke({"name": "internal-comms", "file": "examples/faq-answers.md"})
    refused = load.invoke({"name": "internal-comms", "file": "../sql-analyst/SKILL.md"})
    assert "has no file" in refused
    assert "Unknown skill" in load.invoke({"name": "zh-tw-style"})  # not this agent's skill


def test_skill_prompt_lists_only_names_and_descriptions():
    prompt = skills.skill_prompt(["sql-analyst"])
    assert "sql-analyst:" in prompt and "load_skill" in prompt
    assert "SELECT" not in prompt  # the body stays out until load_skill
    assert skills.skill_prompt([]) == ""


def test_front_matter_falls_back_when_yaml_is_invalid():
    meta = skills._front_matter("name: demo\ndescription: uses: colons")
    assert meta == {"name": "demo", "description": "uses: colons"}
    folded = skills._front_matter("name: x\ndescription: >\n  one\n  two\n")
    assert folded["description"] == "one two"


# --- MCP tools ------------------------------------------------------------------------------


def test_mcp_tools_are_registered_with_tool_info():
    from local_llm.agents import TOOL_INFO, TOOLS, is_external

    for name in mcp_tools.TOOL_SPECS:
        assert name in TOOLS and name in TOOL_INFO
        runs, reads = TOOL_INFO[name].text("en")
        assert runs and reads
    assert is_external("mcp_fetch")
    assert not any(is_external(n) for n in mcp_tools.TOOL_SPECS if n != "mcp_fetch")
    # repo_path is fixed in code: the model never sees it
    assert "repo_path" not in json.dumps(mcp_tools.TOOL_SPECS["mcp_git_log"].schema)


def test_sandbox_paths_stay_inside_their_roots():
    assert mcp_tools.sandbox_path("workspace/a/b.md").is_relative_to(mcp_tools.WORKSPACE_DIR)
    for bad in ["/etc/passwd", "workspace/../../.env", "inbox/.hidden", "storage/x", ""]:
        with pytest.raises((ValueError, PermissionError)):
            mcp_tools.sandbox_path(bad)
    with pytest.raises(PermissionError):
        mcp_tools.sandbox_path("inbox/x.md", write=True)


@pytest.fixture
def hub(monkeypatch):
    pytest.importorskip("mcp_server_time")
    fresh = mcp_tools.McpHub()
    monkeypatch.setattr(mcp_tools, "HUB", fresh)
    yield fresh
    fresh.close()


def test_time_and_files_servers_answer_through_sync_tools(hub):
    now = json.loads(mcp_tools.call("mcp_time_get_current_time", {"timezone": "Asia/Taipei"}))
    assert now["timezone"] == "Asia/Taipei" and now["datetime"].endswith("+08:00")
    converted = mcp_tools.call(
        "mcp_time_convert_time",
        {"source_timezone": "Asia/Taipei", "time": "15:00", "target_timezone": "Asia/Tokyo"},
    )
    assert "16:00" in converted
    assert "Error" in mcp_tools.call("mcp_time_get_current_time", {"timezone": "Mars/Base"})

    name = "workspace/pytest-mcp-check.md"
    try:
        saved = mcp_tools.call("mcp_files_write_file", {"path": name, "content": "測試 ok"})
        assert "Saved" in saved
        assert mcp_tools.call("mcp_files_read_text_file", {"path": name}) == "測試 ok"
        assert name in mcp_tools.call("mcp_files_search_files", {"pattern": "pytest-mcp"})
    finally:
        (mcp_tools.WORKSPACE_DIR / "pytest-mcp-check.md").unlink(missing_ok=True)
    refused = mcp_tools.call("mcp_files_write_file", {"path": "inbox/x.md", "content": "x"})
    assert "read-only" in refused
    assert "not allowed" in mcp_tools.call(
        "mcp_files_read_text_file", {"path": "workspace/../../.env"}
    )
    assert hub.running("time") and hub.running("files")


def test_fetch_refuses_local_addresses_before_the_server(hub):
    assert "Blocked" in mcp_tools.call("mcp_fetch", {"url": "http://127.0.0.1:11434/api/tags"})
    assert not hub.running("fetch")  # refused in code: the server was never started


def test_declared_schemas_match_what_the_servers_offer(hub, monkeypatch, tmp_path):
    pytest.importorskip("mcp_server_git")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    git = mcp_tools.SERVERS["git"]
    monkeypatch.setitem(
        mcp_tools.SERVERS, "git", dataclasses.replace(git, args=("--repository", str(tmp_path)))
    )
    monkeypatch.setattr(mcp_tools, "GIT_REPO", tmp_path)
    for server in ("time", "git", "fetch", "files"):
        offered = {t["name"]: t for t in mcp_tools.discover(server)}
        for spec in (s for s in mcp_tools.TOOLS_SPEC if s.server == server):
            remote = offered[spec.remote]["schema"]
            assert set(spec.schema["properties"]) | set(spec.fixed) <= set(remote["properties"])
            assert set(remote.get("required", [])) <= set(spec.schema["required"]) | set(
                spec.fixed
            ), spec.name
            # only read-only server tools are given to agents, except the sandbox writer
            assert offered[spec.remote]["read_only"] or spec.name == "mcp_files_write_file"
    assert not offered.get("git_commit", {}).get("exposed", False)
