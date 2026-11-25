import json
from pathlib import Path

import pytest

from local_llm import aa_scores, code_bench

ROOT = Path(__file__).resolve().parents[1]


def check_table(columns: dict, rows: list[dict], name: str) -> None:
    width = len(columns["zh"])
    assert len(columns["en"]) == width and rows, name
    for r in rows:
        assert r["where"] in ("local", "cloud") and len(r["zh"]) == len(r["en"]) == width, name


def test_every_test_and_metric_states_its_method():
    cat = json.loads((ROOT / "data" / "lab" / "llm_benchmarks.json").read_text(encoding="utf-8"))
    assert cat["ours"]
    for t in cat["ours"]:
        status = t["status"]
        assert status in ("done", "running", "planned"), t["id"]
        assert t["date"] or status != "done", t["id"]  # a finished test says when it was run
        parts = ("name", "measures", "method") + (("why",) if "rows" not in t and "tables" not in t
                                                  else ("example", "limits"))  # fmt: skip
        for part in parts:
            assert t[part]["zh"].strip() and t[part]["en"].strip(), (t["id"], part)
        if status != "planned" and "rows" in t or "tables" in t:  # a conclusion comes first
            assert t["headline"]["zh"].strip() and t["headline"]["en"].strip(), t["id"]
        if "tables" in t:  # several tables under one test (capacity)
            for part in t["tables"]:
                assert part["title"]["zh"] and part["title"]["en"], t["id"]
                check_table(part["columns"], part["rows"], t["id"])
        elif "rows" in t:
            check_table(t["columns"], t["rows"], t["id"])
        else:  # no results yet: only a running or planned test
            assert status in ("running", "planned"), t["id"]
    uses = cat["uses"]
    assert len(uses["columns"]["zh"]) == len(uses["columns"]["en"]) == 4 and uses["rows"]
    for r in uses["rows"]:
        assert len(r["zh"]) == len(r["en"]) == 4 and all(c.strip() for c in r["zh"] + r["en"])
    aa = cat["aa"]
    assert aa["method"]["zh"] and aa["limits"]["zh"] and aa["url"].startswith("https://")
    for m in aa["metrics"]:
        assert m["field"] in aa_scores.FIELDS and m["unit"] in ("pct", "elo", "index")
        assert m["measures"]["zh"].strip() and m["measures"]["en"].strip(), m["field"]


def test_speech_and_ocr_tests_state_method_date_and_state():
    data = json.loads((ROOT / "data" / "lab" / "scores_methods.json").read_text(encoding="utf-8"))
    ids = {t["id"] for t in data["speech"]["tests"]}
    assert ids == {"asr", "diar", "live", "simul", "tts", "voice"}  # the views on the Scores page
    for t in data["speech"]["tests"] + data["ocr"]["tests"]:
        assert t["status"] in ("done", "running", "planned") and t["date"], t["id"]
        for part in ("name", "headline", "measures", "method", "limits"):
            assert t[part]["zh"].strip() and t[part]["en"].strip(), (t["id"], part)
    uses = data["ocr"]["uses"]
    assert uses["date"] and uses["rows"]
    for r in uses["rows"]:
        assert len(r["zh"]) == len(r["en"]) == len(uses["columns"]["zh"]) == 4


def test_aa_page_data_is_read_from_the_next_stream():
    models = [{"slug": "m-1", "intelligenceIndex": 20.1, "isOpenWeights": True,
               "headline": {"outputSpeed": 47.2}}]  # fmt: skip
    stream = 'x:{"initialModels":' + json.dumps(models) + ',"other":1}'
    half = len(stream) // 2
    html = "".join(
        f"<script>self.__next_f.push([1,{json.dumps(part)}])</script>"
        for part in (stream[:half], stream[half:])
    )
    got = aa_scores.variants(html)
    assert set(got) == {"m-1"}
    row = aa_scores.row(got["m-1"], "m")
    assert row["intelligenceIndex"] == 20.1 and row["outputSpeed"] == 47.2
    assert row["gdpval"] is None and row["url"].endswith("/models/m")
    assert aa_scores.is_local(row) and not aa_scores.is_local({"isOpenWeights": False})


def test_code_rows_explain_each_failure(tmp_path):
    model, think = "org/m:1b", True
    assert code_bench.key(model, think) == "org_m_1b"
    assert code_bench.key(model, False) == "org_m_1b-nothink"
    answers = tmp_path / "result" / "org_m_1b" / "lcb_result.json"
    answers.parent.mkdir(parents=True)
    rows = [
        {"question_id": "a", "content": "```python\nprint(1)\n```", "done_reason": "stop",
         "seconds": 3.0, "eval_count": 10},
        {"question_id": "b", "content": "", "done_reason": "length", "seconds": 9.0},
        {"question_id": "c", "content": "x", "done_reason": "stop", "seconds": 1.0},
    ]  # fmt: skip
    answers.write_text("\n".join(json.dumps(r) for r in rows))
    score = {"rows": [
        {"question_id": "a", "pass": True, "no_code": False, "code": "print(1)"},
        {"question_id": "b", "pass": False, "no_code": True, "code": ""},
        {"question_id": "c", "pass": False, "no_code": False, "code": "x"},
    ]}  # fmt: skip
    (tmp_path / "score.json").write_text(json.dumps(score))
    got = {r["id"]: r for r in code_bench.rows_from(tmp_path, model, think, ["a", "b", "c", "d"])}
    assert got["a"]["ok"] and got["a"]["code"] == "print(1)" and not got["a"]["why"]
    assert got["b"]["capped"] and "cap" in got["b"]["why"]
    assert got["c"]["why"] == "failed at least one test"
    assert not got["d"]["answered"] and got["d"]["why"].startswith("no answer")
    assert code_bench.estimate_minutes(["easy", "hard"]) == 7


@pytest.mark.skipif(code_bench.missing() is not None, reason="LiveCodeBench files not here")
def test_code_questions_are_the_batch_sample():
    qs = code_bench.questions()
    assert len(qs) == 60
    assert {d: sum(q["difficulty"] == d for q in qs) for d in ("easy", "medium", "hard")} == {
        "easy": 20, "medium": 20, "hard": 20}  # fmt: skip
    assert all(q["title"] and q["content"] for q in qs)
