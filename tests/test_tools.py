import json

import pytest

from local_llm.tools import calculate, rank_models, search_models


def test_arithmetic():
    assert calculate("37 * 19 + 8") == 711
    assert calculate("-(3 + 4) / 2") == -3.5


@pytest.mark.parametrize(
    "expression",
    ["__import__('os').system('echo nope')", "a + 1", "2 ** 10000", "True", "[1, 2]", "1e99"],
)
def test_reject_non_arithmetic(expression):
    with pytest.raises(ValueError):
        calculate(expression)


ROWS = [
    {"id": "Qwen/Qwen3-1.7B", "name": "Qwen3-1.7B", "license": "apache-2.0", "notes": "小模型"},
    {
        "id": "Qwen/Qwen3-4B",
        "name": "Qwen3-4B",
        "license": "apache-2.0",
        "notes": "Qwen3-1.7B 下一步",
    },
    {"id": "microsoft/Phi-4-mini", "name": "Phi-4-mini", "license": "mit", "notes": "輕量"},
]


def test_search_ignores_json_keys_and_question_words():
    assert [r["id"] for r in rank_models("Qwen3-1.7B license", ROWS)] == [
        "Qwen/Qwen3-1.7B",
        "Qwen/Qwen3-4B",
    ]
    assert rank_models("license", ROWS) == []
    assert rank_models("report its license", ROWS) == []


def test_search_ranks_name_match_above_text_match_and_handles_chinese():
    assert rank_models("Qwen3-1.7B的授權", ROWS)[0]["id"] == "Qwen/Qwen3-1.7B"
    assert [r["id"] for r in rank_models("輕量", ROWS)] == ["microsoft/Phi-4-mini"]


def test_search_models_tool_on_real_catalog():
    result = json.loads(
        search_models.invoke("Search the catalog for Qwen3-1.7B and report its license.")
    )
    assert result[0]["id"] == "Qwen/Qwen3-1.7B"
    assert result[0]["license"] == "apache-2.0"
