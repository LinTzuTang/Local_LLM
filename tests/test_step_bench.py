import pytest

from local_llm import step_bench as sb


def test_model_key_and_result_parsing():
    assert sb.model_key("huihui_ai/Qwen3.8:27b") == "plat-huihui-ai-qwen3.8-27b"
    raw = "[[[{'pwd': '{}'}], 'done'], [[{'cd': '{\"folder\": \"a\"}'}, {'ls': '{}'}]]]"
    assert sb._steps(raw) == [
        [{"calls": [{"pwd": "{}"}]}, {"text": "done"}],
        [{"calls": [{"cd": '{"folder": "a"}'}, {"ls": "{}"}]}],
    ]
    assert sb._steps(None) == []


def test_estimate_counts_memory_prerequisites():
    assert sb.estimate_seconds("memory", 2, prereqs=5) == 5 * 120 + 2 * 40
    assert sb.estimate_seconds("multi_turn", 3) == 360


@pytest.mark.skipif(sb.missing("multi_turn") is not None, reason="BFCL files not on this machine")
def test_question_lists_load():
    mt = sb.multi_turn_questions()
    assert len(mt) == 100 and all(q["turns"] and q["category"].startswith("multi_turn") for q in mt)
    assert len(sb.search_questions()) == 20
    sets = sb.memory_sets("kv")
    assert set(sets) == {"finance", "healthcare", "notetaker"}
    assert all(s["prereqs"] and s["questions"] for s in sets.values())


@pytest.mark.skipif(sb.missing("company") is not None, reason="company test not on this machine")
def test_company_answers_load():
    d = sb.company_answers()
    assert len(d["questions"]) == len(sb.company_questions())
    act = [q for q in d["questions"] if q["expect"] == "act"]
    assert act and all(q["alternatives"] for q in act)
    assert all(q["alternatives"] == [] for q in d["questions"] if q["expect"] != "act")
    run = next(iter(d["runs"].values()))
    assert {"model", "cond", "main", "grades"} <= set(run)
