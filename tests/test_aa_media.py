"""aa_media: reading AA's non-language-model leaderboards from synthetic pages (no network)."""

import json

import pytest

from local_llm import aa_media, aa_scores
from local_llm.aa_scores import page_data


def html(stream: str) -> str:
    """A page whose data stream is `stream`, split over two pushes like AA's pages."""
    half = len(stream) // 2
    return "".join(
        f"<script>self.__next_f.push([1,{json.dumps(part)}])</script>"
        for part in (stream[:half], stream[half:])
    )


def stt_stream() -> str:
    models = [
        {"id": "m1", "name": "Alpha, X", "openWeights": False, "creator": {"name": "X"},
         "representativeHostModel": {"id": "h1", "aaWerIndex": 0.05}},
        {"id": "m2", "name": "Beta", "openWeights": True, "creator": {"name": "Y"},
         "representativeHostModel": {"id": "h2", "aaWerIndex": 0.03}},
        {"id": "m3", "name": "Gamma", "openWeights": None, "creator": {"name": "Z"},
         "representativeHostModel": {"id": "h3", "aaWerIndex": None}},
    ]  # fmt: skip
    hosts = [{"id": "h1", "host": {"name": "X"}, "aaWerIndex": 0.05, "pricePer1kMinutes": 2.5,
              "performance": {"medianSpeedFactor": 40.0}}]  # fmt: skip
    # the second list only points at the first record, as AA's lists often do
    return (f'80:["$","$L85",null,{{"defaultSelectedModels":{json.dumps(models)}}}]\n'
            f'7c:{{"hostModels":{json.dumps(hosts)},'
            '"more":["$80:props:defaultSelectedModels:0"]}\n')  # fmt: skip


def test_stt_rows_join_the_representative_endpoint():
    rows = aa_media.stt_rows(page_data(html(stt_stream())))
    assert [r["model"] for r in rows] == [
        "Beta",
        "Alpha, X",
    ]  # lowest error first; no score left out
    assert rows[1] == {"model": "Alpha, X", "creator": "X", "open_weights": False, "wer": 0.05,
                       "speed": 40.0, "price": 2.5}  # fmt: skip
    assert rows[0]["speed"] is None and rows[0]["price"] is None  # endpoint not on the page


def test_streaming_rows_use_the_endpoint_and_its_provider():
    hosts = [
        {"id": "a", "name": "A, P", "displayName": "A Live", "host": {"name": "P"},
         "model": {"openWeights": True}, "aaWerStreamingIndex": 0.04,
         "timeToFinalTranscriptSeconds": 0.5, "pricePer1kMinutes": 6},
        {"id": "b", "name": "B", "displayName": None, "host": {"name": "Q"}, "model": {},
         "aaWerStreamingIndex": 0.02, "timeToFinalTranscriptSeconds": 1.2,
         "pricePer1kMinutes": None},
    ]  # fmt: skip
    rows = aa_media.stt_streaming_rows(page_data(html(f"5:{json.dumps(hosts)}\n")))
    assert [(r["model"], r["creator"], r["open_weights"]) for r in rows] == [
        ("B", "Q", None), ("A Live", "P", True)]  # fmt: skip
    assert rows[1]["final_s"] == 0.5 and rows[1]["wer"] == 0.04


def test_s2s_rows_take_the_overall_index_and_the_endpoint_latency():
    arena = [
        {"id": "r1", "name": "Voice A", "creatorName": "X", "sliceKey": "overall", "elo": 1050,
         "stsHostModelId": "h1", "stsQualityIndexV2Score": 70.5},
        {"id": "r2", "name": "Voice A", "creatorName": "X", "sliceKey": "agentic", "elo": 1100,
         "stsHostModelId": "h1", "stsQualityIndexV2Score": 99.0},
        {"id": "r3", "name": "Cascade", "creatorName": "Y", "sliceKey": "overall", "elo": 1111,
         "stsHostModelId": None, "stsQualityIndexV2Score": None},
        {"id": "r4", "name": "Voice B", "creatorName": "Z", "sliceKey": "overall", "elo": 990,
         "stsHostModelId": "h9", "stsQualityIndexV2Score": 80.0},
    ]  # fmt: skip
    hosts = [{"id": "h1", "model": {"openSource": False}, "host": {"name": "X"}, "bbaScore": 0.9,
              "timeToFirstAudioSeconds": 1.1, "costPerHourOfInputAudio": 4.2}]  # fmt: skip
    data = page_data(
        html(f'72:{{"arenaResults":{json.dumps(arena)},"hosts":{json.dumps(hosts)}}}\n')
    )
    rows = aa_media.s2s_rows(data)
    assert [(r["model"], r["index"]) for r in rows] == [("Voice B", 80.0), ("Voice A", 70.5)]
    assert rows[1]["ttfa"] == 1.1 and rows[1]["price"] == 4.2 and rows[1]["open_weights"] is False
    assert rows[0]["ttfa"] is None and rows[0]["open_weights"] is None  # endpoint not listed


def board_rows(*entries):
    return [{"formatted": {"rank": i + 1}, "values": v} for i, v in enumerate(entries)]


def test_elo_boards_take_the_overall_board_through_references():
    overall = board_rows(
        {"id": "a", "name": "A", "elo": 1100.5, "creator": {"name": "X"}, "openWeights": False,
         "pricePer1mCharacters": 40},
        {"id": "b", "name": "B", "elo": 1200.0, "creator": {"name": "Y"}, "openWeights": True,
         "pricePer1mCharacters": None},
    )  # fmt: skip
    accent = board_rows({"id": "a", "name": "A", "elo": 1300.0, "creator": {"name": "X"}})
    stream = (
        '47:["$","$L65",null,{"data":"$Q66","initialOpenWeightsOnly":false}]\n'
        '66:[[null,"$Q67"],["assistants","$Q68"]]\n'
        f"67:{json.dumps([[None, overall], ['en-us', accent]])}\n"
        f"68:{json.dumps([[None, accent]])}\n"
    )
    rows = aa_media.elo_rows("data", "pricePer1mCharacters")(page_data(html(stream)))
    assert rows == [
        {"model": "B", "creator": "Y", "open_weights": True, "elo": 1200.0, "price": None},
        {"model": "A", "creator": "X", "open_weights": False, "elo": 1100.5, "price": 40},
    ]


def test_elo_boards_inline_and_open_weights_from_the_link():
    with_audio = board_rows(
        {"id": "w", "name": "W", "elo": 1150.0, "creator": {"name": "X"}, "openWeightsUrl": None,
         "pricePerMinute": 12},
        {"id": "o", "name": "O", "elo": 1140.0, "creator": {"name": "Y"},
         "openWeightsUrl": "https://huggingface.co/o", "pricePerMinute": 4.8},
    )  # fmt: skip
    silent = board_rows({"id": "s", "name": "S", "elo": 1190.0, "creator": {"name": "Z"},
                         "openWeightsUrl": None})  # fmt: skip
    video = (f"70:{json.dumps([[None, with_audio]])}\n38a:{json.dumps([[None, silent]])}\n"
             '64:["$","$L6f",null,{"withAudioData":"$Q70",'
             '"withoutAudioData":"$Q38a"}]\n')  # fmt: skip
    rows = aa_media.elo_rows("withAudioData", "pricePerMinute")(page_data(html(video)))
    assert [(r["model"], r["open_weights"], r["price"]) for r in rows] == [
        ("W", False, 12), ("O", True, 4.8)]  # fmt: skip
    music = board_rows({"id": "m", "name": "M", "elo": 1160.5, "creator": {"name": "Q"}})
    stream = f'45:[["$","$L63",null,{{"title":"T","initialData":{json.dumps(music)}}}]]\n'
    rows = aa_media.elo_rows("initialData", None)(page_data(html(stream)))
    assert rows == [{"model": "M", "creator": "Q", "open_weights": None, "elo": 1160.5}]
    with pytest.raises(ValueError):
        aa_media.board(page_data(html('1:{"initialData":[{"x":1}]}\n')), "initialData")


def test_every_board_states_its_method_and_units():
    for cid, spec in aa_media.CATEGORIES.items():
        for part in ("name", "about"):
            assert spec[part]["zh"].strip() and spec[part]["en"].strip(), (cid, part)
        assert spec["metrics"], cid
        for m in spec["metrics"]:
            assert m["unit"] in ("pct", "elo", "index", "x", "usd", "s", "tps"), (cid, m["field"])
            assert isinstance(m["lower_better"], bool) and m["name"]["zh"] and m["name"]["en"]


def test_fetch_all_keeps_going_when_a_board_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(aa_media, "OUT", tmp_path / "aa_media.json")
    cache = tmp_path / "pages"
    cache.mkdir()
    for spec in aa_media.CATEGORIES.values():  # every page cached: no network
        (cache / f"{spec['page'].replace('/', '_')}.html").write_text("<html>changed</html>")
    (cache / "speech-to-text_non-streaming.html").write_text(html(stt_stream()))
    old = {"fetched": "2026-11-01", "categories": {"music": {"rows": [{"model": "Old"}]}}}
    aa_media.OUT.write_text(json.dumps(old))

    data = aa_media.fetch_all(cache)
    assert set(data["failed"]) == set(aa_media.CATEGORIES) - {"stt"}
    stt = data["categories"]["stt"]
    assert [r["model"] for r in stt["open_rows"]] == ["Beta"]
    assert (
        stt["total"] == 2
        and stt["rows"][0]["model"] == "Beta"
        and stt["url"].endswith("/non-streaming")
    )
    assert data["categories"]["music"] == old["categories"]["music"]  # the old copy stays
    assert "image" not in data["categories"]
    assert aa_media.load() == data

    (cache / "speech-to-text_non-streaming.html").write_text("<html>changed</html>")
    with pytest.raises(RuntimeError):  # nothing readable: the saved file is left alone
        aa_media.fetch_all(cache)
    assert aa_media.load() == data


def test_load_without_a_file(tmp_path, monkeypatch):
    monkeypatch.setattr(aa_media, "OUT", tmp_path / "missing.json")
    assert aa_media.load() is None
    aa_media.OUT.write_text("[1, 2]")
    assert aa_media.load() is None


def test_elo_boards_can_take_a_language_board():
    overall = board_rows({"id": "a", "name": "A", "elo": 1300.0, "creator": {"name": "X"}})
    zh = board_rows({"id": "b", "name": "B", "elo": 1100.0, "creator": {"name": "Y"},
                     "openWeights": True, "pricePer1mCharacters": 15})  # fmt: skip
    stream = (
        '47:["$","$L65",null,{"data":"$Q66"}]\n66:[[null,"$Q67"],["assistants","$Q68"]]\n'
        f"67:{json.dumps([[None, overall], ['en-us', overall], ['zh', zh]])}\n"
        f"68:{json.dumps([[None, overall]])}\n"
    )
    data = page_data(html(stream))
    assert [r["model"] for r in aa_media.elo_rows("data", None)(data)] == ["A"]
    rows = aa_media.elo_rows("data", "pricePer1mCharacters", key="zh")(data)
    assert rows == [
        {"model": "B", "creator": "Y", "open_weights": True, "elo": 1100.0, "price": 15}
    ]


def llm_model(mid, release, index, open_, size, **extra):
    return {"id": mid, "release": {"slug": release}, "name": f"{release} {mid}",
            "modelCreatorName": "C", "isOpenWeights": open_, "paramClass": size,
            "intelligenceIndex": index, "scores": {"gdpval-aa-normalized": 1400.5},
            "outputSpeed": 50.0, "price1mInputTokens": 0.5, "price1mOutputTokens": 3,
            **extra}  # fmt: skip


def test_llm_lists_keep_one_variant_and_split_open_and_small(monkeypatch):
    models = [
        llm_model("v1", "small-a", 15.0, True, "small"),
        llm_model("v2", "small-a", 20.0, True, "small"),  # the better variant of the same model
        llm_model("v3", "big-open", 40.0, True, "large"),
        llm_model("v4", "cloud", 60.0, False, None),
        llm_model("v5", "no-page", 12.0, True, "small"),  # its page fails: AA's group decides
        llm_model("v6", "mislabelled", 11.0, True, "small"),  # the count says 45B: not small
        llm_model("v7", "no-index", None, True, "small"),
    ]
    stream = f'9:{{"initialModels":{json.dumps(models)}}}\n'
    rows = aa_media.llm_rows(page_data(html(stream)))
    assert [r["model"] for r in rows] == [
        "cloud v4", "big-open v3", "small-a v2", "no-page v5", "mislabelled v6"]  # fmt: skip
    assert rows[2]["gdpval"] == 1400.5 and rows[2]["terminal_bench"] is None

    pages = {  # a model page lists its variants and comparable models, with parameters
        "small-a": [{"slug": "small-a", "id": "v2", "parameters": 27},
                    {"slug": "mislabelled", "id": "v6", "parameters": 45}],
        "big-open": [{"slug": "big-open", "id": "v3", "parameters": 700}],
    }  # fmt: skip

    def fake_fetch(page, cache=None):
        if page not in pages:
            raise OSError("not found")
        return html(f'1:{{"initialModels":{json.dumps(pages[page])}}}\n')

    monkeypatch.setattr(aa_scores, "fetch", fake_fetch)
    lists = aa_media.llm_lists(rows)
    assert [r["model"] for r in lists["rows"]] == [r["model"] for r in rows]
    assert [r["model"] for r in lists["open_rows"]] == [
        "big-open v3", "small-a v2", "no-page v5", "mislabelled v6"]  # fmt: skip
    assert [(r["model"], r["parameters"]) for r in lists["small_rows"]] == [
        ("small-a v2", 27), ("no-page v5", None)]  # fmt: skip
    assert all("_id" not in r for part in lists.values() for r in part)
