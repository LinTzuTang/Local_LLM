"""Scenario engine, retrieval helpers and the coding agent's change planning (no models needed)."""

import json

import pytest

from local_llm.lab import control, engine, rag, store
from local_llm.lab.textutil import render


def template_scenario(**extra) -> dict:
    """A runnable scenario made only of template nodes: fan-out, join and a router-free path."""
    return {
        "id": "t",
        "name": {"zh": "測試", "en": "test"},
        "inputs": [{"key": "question", "type": "text"}],
        "nodes": [
            {"id": "a", "type": "template", "params": {"template": "A:{question}", "output": "a"}},
            {"id": "b", "type": "template", "params": {"template": "B:{question}", "output": "b"}},
            {
                "id": "join",
                "type": "template",
                "join": "all",
                "params": {"template": "{a}+{b}", "output": "answer"},
            },
        ],
        "edges": [
            {"from": "START", "to": "a"},
            {"from": "START", "to": "b"},
            {"from": "a", "to": "join"},
            {"from": "b", "to": "join"},
            {"from": "join", "to": "END"},
        ],
        "output": "answer",
        **extra,
    }


def test_bundled_scenarios_are_valid():
    for s in store.scenarios():
        check = store.validate_scenario(s)
        assert check.ok, (s["id"], check.errors)


def test_parallel_branches_join_before_the_next_node():
    run = engine.Run(template_scenario(), {"question": "q"}, "http://127.0.0.1:9")
    run.execute()
    assert run.status == "done", run.error
    assert run.output() == "A:q+B:q"
    assert [e["node"] for e in run.trace].count("join") == 1
    assert {e["status"] for e in run.trace} == {"done"}


def test_validation_catches_broken_graphs():
    s = template_scenario()
    s["edges"] = [e for e in s["edges"] if e["to"] != "END"]
    assert any("no path to END" in e for e in store.validate_scenario(s).errors)
    s = template_scenario()
    s["nodes"][0]["params"]["bogus"] = 1
    assert any("unknown param" in e for e in store.validate_scenario(s).errors)
    s = template_scenario()
    s["edges"].append({"from": "a", "to": "b", "when": "x"})
    assert any("only router nodes" in e for e in store.validate_scenario(s).errors)
    s = template_scenario()
    s["nodes"][0]["params"]["template"] = "{missing}"
    assert any("{missing}" in w for w in store.validate_scenario(s).warnings)


def test_router_edges_must_match_options():
    s = template_scenario()
    s["nodes"].insert(
        0,
        {
            "id": "r",
            "type": "router",
            "params": {"instructions": "pick", "options": {"x": "X", "y": "Y"}},
        },
    )
    s["edges"] = [
        {"from": "START", "to": "r"},
        {"from": "r", "to": "a", "when": "x"},
        {"from": "r", "to": "b", "when": "z"},
        {"from": "a", "to": "join"},
        {"from": "b", "to": "join"},
        {"from": "join", "to": "END"},
    ]
    assert any("'z' is not an option" in e for e in store.validate_scenario(s).errors)


def test_render_formats_hits_and_keeps_other_braces():
    hits = [{"title": "Doc", "text": "hello"}]
    assert (
        render('{context} {"json": 1} {nope}', {"context": hits}) == '[1] (Doc) hello {"json": 1} '
    )


def test_rrf_rewards_agreement():
    a = [{"id": "x", "method": "bm25"}, {"id": "y", "method": "bm25"}]
    b = [{"id": "y", "method": "vector"}, {"id": "z", "method": "vector"}]
    fused = rag.rrf([a, b], k=60)
    assert fused[0]["id"] == "y"  # ranked by both
    assert fused[0]["ranks"] == {"bm25": 2, "vector": 1}


def test_cjk_bigrams_and_chunks():
    assert rag.tokenize("特休 SM-400") == ["sm-400", "特休"]
    chunks = rag.chunk_markdown("# T\n\n## A\n" + "字" * 1200 + "\n\n## B\nshort", size=500)
    assert all(len(c) <= 520 for c in chunks)
    assert chunks[-1].startswith("T > B")


@pytest.fixture
def lab_files(tmp_path, monkeypatch):
    for name in ("scenarios.json", "agents.json"):
        (tmp_path / name).write_text((store.LAB_DIR / name).read_text())
    monkeypatch.setattr(store, "SCENARIOS_FILE", tmp_path / "scenarios.json")
    monkeypatch.setattr(store, "AGENTS_FILE", tmp_path / "agents.json")
    monkeypatch.setattr(store, "HISTORY_DIR", tmp_path / "history")
    return tmp_path


def test_coding_agent_change_is_planned_saved_and_rolled_back(lab_files):
    args = {
        "scenario_id": "hybrid_rag",
        "node": {
            "id": "translate",
            "type": "llm",
            "label": {"zh": "翻譯", "en": "Translate"},
            "params": {"prompt": "Translate: {answer}", "output": "answer_en"},
        },
        "edges": [{"from": "answer", "to": "translate"}, {"from": "translate", "to": "END"}],
        "remove_edges": [{"from": "answer", "to": "END"}],
    }
    plan = control.plan_change("add_node", args)
    assert plan["check"].ok, plan["check"].errors
    assert '+      "id": "translate",' in store.diff(plan["before"], plan["after"])
    assert "Saved scenario" in control._apply("add_node", args)
    assert "translate" in [n["id"] for n in store.scenario("hybrid_rag")["nodes"]]
    entry = store.history(1)[0]
    assert store.rollback(entry["file"]).ok
    assert "translate" not in [n["id"] for n in store.scenario("hybrid_rag")["nodes"]]


def test_invalid_change_is_not_saved(lab_files):
    before = json.loads(store.SCENARIOS_FILE.read_text())
    result = control._apply(
        "update_node",
        {"scenario_id": "hybrid_rag", "node_id": "answer", "changes": {"type": "nope"}},
    )
    assert result.startswith("NOT SAVED")
    assert json.loads(store.SCENARIOS_FILE.read_text()) == before


def test_agent_update_validates_tools(lab_files):
    result = control._apply(
        "update_agent", {"agent_id": "translate", "changes": {"tools": ["rm_rf"]}}
    )
    assert result.startswith("NOT SAVED") and "unknown tools" in result
