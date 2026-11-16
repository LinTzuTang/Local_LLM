import pytest

from local_llm import tool_test
from local_llm.tool_test import check


def call(name, **args):
    return {"name": name, "arguments": args}


BFCL = {
    "kind": "calls",
    "style": "bfcl",
    "calls": [{"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}],
}


def test_bfcl_match_with_optional_and_number_types():
    assert check(BFCL, [call("calculate_triangle_area", base=10, height=5.0)])["ok"]
    assert check(BFCL, [call("calculate_triangle_area", base=10, height=5, unit="Units")])["ok"]


def test_bfcl_wrong_value_missing_and_extra_parameters():
    assert check(BFCL, [call("calculate_triangle_area", base=10, height=6)])["code"] == "mismatch"
    assert not check(BFCL, [call("calculate_triangle_area", base=10)])["ok"]
    assert not check(BFCL, [call("calculate_triangle_area", base=10, height=5, color="red")])["ok"]


def test_dotted_names_and_string_normalisation():
    expect = {"kind": "calls", "style": "bfcl", "calls": [{"math.gcd": {"city": ["New York, NY"]}}]}
    assert check(expect, [call("math_gcd", city="new york ny")])["ok"]


def test_parallel_calls_in_any_order_and_count():
    expect = {
        "kind": "calls",
        "style": "bfcl",
        "calls": [{"w": {"city": ["Paris"]}}, {"w": {"city": ["Rome"]}}],
    }
    assert check(expect, [call("w", city="Rome"), call("w", city="Paris")])["ok"]
    assert check(expect, [call("w", city="Rome")])["code"] == "count"


def test_nested_dict_answers():
    expect = {
        "kind": "calls",
        "style": "bfcl",
        "calls": [{"f": {"opts": [{"a": [1], "b": ["x", ""]}]}}],
    }
    assert check(expect, [call("f", opts={"a": 1})])["ok"]
    assert not check(expect, [call("f", opts={"a": 2})])["ok"]


def test_no_call_any_call_and_mention():
    assert check({"kind": "no_call"}, [], "I can't")["ok"]
    assert check({"kind": "no_call"}, [call("x")])["code"] == "called"
    assert check({"kind": "any_call"}, [call("x")])["ok"]
    mention = {"kind": "mention", "words": ["user_id", "lifestyle"]}
    assert check(mention, [], "Missing necessary parameters (user_id, lifestyle)")["ok"]
    assert check(mention, [], "Missing user_id")["code"] == "not_mentioned"


def test_acebench_numbered_names_alias_and_loose_strings():
    expect = {
        "kind": "calls",
        "style": "ace",
        "alias": {"fn_1": "Real.Api"},
        "calls": [{"Real.Api_1": {"tz": "ET"}}, {"Real.Api_2": {"tz": "PT"}}],
    }
    assert check(expect, [call("fn_1", tz="美国东部时间(ET)"), call("fn_1", tz="PT")])["ok"]


def test_samples_file_loads():
    samples = tool_test.load_samples()
    assert samples and {"bfcl_en", "bfcl_zhtw", "acebench"} <= {s["suite"] for s in samples}
    assert all(s["tools"] and s["messages"] and s["batch"] for s in samples)
    assert len({s["key"] for s in samples}) == len(samples)
    assert any(s["featured"] for s in samples)


def test_downloaded_set_uploads_back():
    raw = (tool_test.SETS / "bfcl_zhtw.json").read_bytes()
    got = tool_test.parse_set(raw)
    assert got["samples"][0]["suite"] == "bfcl_zhtw" and len(got["samples"]) > 100
    # an edited set: a bare list, one question, no expected answer
    one = tool_test.parse_set(b'[{"messages": [{"role": "user", "content": "hi"}]}]')
    assert (
        one["samples"][0]["expect"] == {"kind": "any_call"}
        and one["samples"][0]["suite"] == "uploaded"
    )
    for bad in (
        b"not json",
        b"[]",
        b'[{"messages": []}]',
        b'[{"messages": [{"role": "user"}], "expect": {"kind": "x"}}]',
    ):
        with pytest.raises(ValueError):
            tool_test.parse_set(bad)


def test_test_set_catalog_is_complete():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    cat = json.loads((root / "data" / "lab" / "datasets.json").read_text())
    for s in cat["sets"]:
        assert s["group"] in cat["groups"], s["id"]
        for field in ("name", "count", "what", "used"):
            assert s[field]["zh"] and s[field]["en"], (s["id"], field)
        assert s.get("use") or s.get("why_not"), s["id"]
        assert bool(s["files"]) != bool(s.get("why_not")), s["id"]  # a download or a reason
        for f in s["files"]:  # the stance questions are kept out of git on purpose
            assert (root / f).exists() or s["id"] == "taiwan_stance", f
