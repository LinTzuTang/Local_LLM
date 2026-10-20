"""Live voice assistant: OpenStreetMap / Open-Meteo lookups (HTTP mocked) and the engine's
turn-taking with a fake LLM, fake tools and fake speech synthesis."""

import json
import threading
import time

import httpx
import pytest

from local_llm import geo_tools, voice_agent
from local_llm.agents import TOOL_INFO, TOOLS, is_external

NEIHU = {"name": "內湖", "latitude": 25.0815, "longitude": 121.58809, "country_code": "TW",
         "population": 274538, "admin2": "臺北市", "admin3": "內湖區",
         "timezone": "Asia/Taipei"}  # fmt: skip


def _element(i, name, lat, lon, **tags):
    tags = {"name": name, "amenity": tags.pop("amenity", "restaurant"), **tags}
    return {"type": "node", "id": i, "lat": lat, "lon": lon, "tags": tags}


OVERPASS = [
    _element(1, "胖女孩早午餐", 25.0830, 121.5900),
    _element(2, "美而美", 25.0827, 121.5966, cuisine="breakfast"),
    _element(
        3,
        "營養早午餐",
        25.0820,
        121.5870,
        opening_hours="Mo-Su 06:30-14:00",
        **{"addr:street": "成功路三段187巷", "addr:housenumber": "2"},
    ),
    _element(4, "遠方的早午餐", 25.2000, 121.9000),  # far outside the radius
    _element(5, "路易莎咖啡", 25.0816, 121.5882, amenity="cafe"),
]
NOMINATIM = [
    {"osm_type": "node", "osm_id": 7, "lat": "25.0886", "lon": "121.5863",
     "name": "紅橘子精緻早午餐", "category": "amenity", "type": "restaurant",
     "extratags": {"opening_hours": "Mo-Su 06:00-14:00"},
     "address": {"road": "內湖路三段", "house_number": "60", "suburb": "內湖區"}},
    {"osm_type": "node", "osm_id": 1, "lat": "25.0830", "lon": "121.5900",
     "name": "胖女孩早午餐", "category": "amenity", "type": "restaurant",
     "address": {"road": "成功路三段", "suburb": "內湖區"}},
]  # fmt: skip


class Fake:
    """httpx.MockTransport handler recording requests; answers per host."""

    def __init__(self, overpass=None, nominatim=None, overpass_status=200, geocode=None):
        self.requests = []
        self.overpass = OVERPASS if overpass is None else overpass
        self.nominatim = NOMINATIM if nominatim is None else nominatim
        self.overpass_status, self.geocode = overpass_status, geocode

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        host = request.url.host
        if host == "geocoding-api.open-meteo.com":
            rows = self.geocode if self.geocode is not None else [NEIHU]
            return httpx.Response(200, json={"results": rows})
        if host == "api.open-meteo.com":
            return httpx.Response(200, json={
                "current": {"time": "2026-10-19T09:00", "temperature_2m": 24.5,
                            "apparent_temperature": 26.0, "relative_humidity_2m": 80,
                            "weather_code": 3, "wind_speed_10m": 8.0},
                "hourly": {"time": [f"2026-10-19T{h:02d}:00" for h in range(24)],
                           "precipitation_probability": [10] * 9 + [60] * 15},
                "daily": {"temperature_2m_min": [21.0], "temperature_2m_max": [27.0],
                          "precipitation_probability_max": [60], "weather_code": [80]},
            })  # fmt: skip
        if host == "overpass-api.de":
            if self.overpass_status != 200:
                return httpx.Response(self.overpass_status, text="busy")
            return httpx.Response(200, json={"elements": self.overpass})
        if host == "nominatim.openstreetmap.org":
            return httpx.Response(200, json=self.nominatim)
        return httpx.Response(404)


@pytest.fixture
def fake(monkeypatch):
    def install(**kw):
        handler = Fake(**kw)
        monkeypatch.setattr(geo_tools, "TRANSPORT", httpx.MockTransport(handler))
        monkeypatch.setattr(geo_tools, "NOMINATIM_INTERVAL_S", 0.0)
        geo_tools.clear_cache()
        return handler

    yield install
    geo_tools.clear_cache()


def test_weather_reports_now_today_and_rain(fake):
    handler = fake()
    result = geo_tools.weather("內湖")
    assert result["location"] == "內湖（臺北市內湖區）"
    assert result["temperature_c"] == 24.5 and result["condition"] == "陰"
    assert result["today"] == {"min_c": 21.0, "max_c": 27.0, "rain_chance_pct": 60,
                               "condition": "陣雨"}  # fmt: skip
    assert result["tomorrow"] is None  # the fixture has one day only
    assert result["rain_chance_next_6h_pct"] == 60
    assert "降雨機率 60%" in geo_tools.weather_summary(result)
    geo_tools.weather("內湖")  # cached: no second round of requests
    assert len(handler.requests) == 2


def test_geocode_strips_nearby_words_and_falls_back_to_short_name(fake):
    handler = fake()
    assert geo_tools.clean_place("內湖這附近") == "內湖"
    geo_tools.geocode("台北市內湖區")
    assert handler.requests[0].url.params["name"] == "台北市內湖區"
    fake(geocode=[], nominatim=[])
    with pytest.raises(geo_tools.GeoError):
        geo_tools.geocode("不存在的地方")


def test_geocode_asks_nominatim_when_geonames_has_no_major_place(fake):
    village = {**NEIHU, "name": "板橋", "population": None, "feature_code": "PPL"}
    fake(geocode=[village])  # only a village in GeoNames: OpenStreetMap is asked
    place = geo_tools.geocode("西門町")
    assert place["name"] == NOMINATIM[0]["name"] and place["country_code"] == "TW"
    district = {**NEIHU, "name": "板橋區", "feature_code": "PPLA2"}
    fake(geocode=[district])
    assert geo_tools.geocode("板橋區")["name"] == "板橋區"


def test_nearby_brunch_merges_sources_ranks_and_attributes(fake):
    handler = fake()
    result = geo_tools.nearby("內湖", "早午餐")
    names = [p["name"] for p in result["places"]]
    assert names[:3] == ["營養早午餐", "胖女孩早午餐", "紅橘子精緻早午餐"]  # brunch by distance
    assert "美而美" in names  # breakfast shop tops up the list
    assert "遠方的早午餐" not in names and "路易莎咖啡" not in names
    assert names.count("胖女孩早午餐") == 1  # same place from both sources kept once
    first = result["places"][0]
    assert first["opening_hours"] == "Mo-Su 06:30-14:00"
    assert first["address"] == "成功路三段187巷2號"
    assert 100 <= first["distance_m"] <= 300
    assert result["match"] == "brunch"
    assert result["attribution"] == "© OpenStreetMap contributors (ODbL)"
    assert result["sources"] == ["nominatim", "overpass"]
    nominatim = next(r for r in handler.requests if r.url.host == "nominatim.openstreetmap.org")
    assert set(nominatim.url.params) >= {"q", "viewbox", "bounded"}
    overpass = next(r for r in handler.requests if r.url.host == "overpass-api.de")
    assert "早午餐" in httpx.QueryParams(overpass.content.decode())["data"]


def test_nearby_says_so_when_osm_has_no_brunch(fake):
    cafe = {"osm_type": "node", "osm_id": 9, "lat": "25.0816", "lon": "121.5882",
            "name": "巷口咖啡", "category": "amenity", "type": "cafe"}  # fmt: skip
    fake(overpass=[], nominatim=[cafe])
    result = geo_tools.nearby("內湖", "早午餐")
    assert result["match"] == "nearby"
    assert "改列最近的咖啡店和餐廳" in result["note"]
    assert [p["name"] for p in result["places"]] == ["巷口咖啡"]
    assert result["places"][0]["match"] == "nearby"


def test_nearby_survives_overpass_errors(fake):
    fake(overpass_status=504)
    result = geo_tools.nearby("內湖", "早午餐")
    assert result["sources"] == ["nominatim"]
    assert any("Overpass" in e for e in result["errors"])
    assert result["places"]


def test_search_places_is_registered_as_a_network_tool(fake):
    fake()
    assert "search_places" in TOOLS and is_external("search_places")
    assert TOOL_INFO["search_places"].runs_en
    out = json.loads(TOOLS["search_places"].invoke({"location": "內湖", "query": "早午餐"}))
    assert out["places"][0]["name"] == "營養早午餐"


# -- engine ----------------------------------------------------------------------------------


def test_split_sentences_and_filler():
    pieces, rest = voice_agent.split_sentences("好的。內湖現在二十四度，有點悶熱，今天")
    assert pieces == ["好的。"] and rest == "內湖現在二十四度，有點悶熱，今天"
    pieces, rest = voice_agent.split_sentences(
        "內湖現在二十四度，有點悶熱，今天下午降雨機率六成，", soft=10
    )
    assert pieces == ["內湖現在二十四度，有點悶熱，", "今天下午降雨機率六成，"] and rest == ""
    assert voice_agent.filler("get_weather", {"location": "內湖"}, "我在內湖") == (
        "好，我幫你查一下內湖的天氣。"
    )
    assert voice_agent.filler("search_places", {"location": "內湖", "query": "早午餐"},
                              "謝謝，這附近有什麼好吃的早午餐店嗎") == (
        "不客氣！我幫你找一下內湖附近的早午餐店。"
    )  # fmt: skip
    assert voice_agent.filler("get_weather", {"location": "Neihu"}, "我在內湖") == (
        "好，我幫你查一下天氣。"
    )


def test_system_prompt_carries_location_and_network_rule():
    prompt = voice_agent.system_prompt({"location": "內湖"}, True)
    assert "使用者目前的位置：內湖" in prompt and "get_weather" in prompt
    assert "沒有開啟連網查詢" in voice_agent.system_prompt({}, False)


class ScriptedLLM:
    """Replies by the latest user message; records every call."""

    def __init__(self, delay=0.0):
        self.calls, self.delay = [], delay

    def __call__(self, base, model, messages, tools, think, max_tokens, cancel=None):
        self.calls.append({"messages": messages, "tools": tools})
        last = messages[-1]["content"].split("（使用者剛才說）")[-1]  # after the lookup results
        time.sleep(self.delay)
        if last.startswith("（系統：get_weather"):
            yield {"content": "內湖現在二十四度，陰天。"}
            yield {"content": "今天降雨機率六成，記得帶傘。"}
        elif last.startswith("（系統：search_places"):
            for piece in ["附近有營養早午餐，大約兩百公尺。", "還有胖女孩早午餐。",
                          "另外紅橘子精緻早午餐也不錯。"]:  # fmt: skip
                time.sleep(self.delay)
                yield {"content": piece}
        elif "天氣" in last:
            yield {"content": "你現在在哪裡？"}
        elif "內湖" in last:
            yield {"tool_calls": [{"function": {"name": "get_weather",
                                                "arguments": {"location": "內湖"}}}]}  # fmt: skip
        elif "早午餐" in last:
            location = "內湖" if "內湖" in messages[0]["content"] else "這附近"
            yield {"tool_calls": [{"function": {"name": "search_places", "arguments": {
                "location": location, "query": "早午餐"}}}]}  # fmt: skip
        else:
            yield {"content": "好的，祝你用餐愉快！"}
        yield {"done": True, "load_s": 0.0}


class Session:
    """A VoiceAgent with fakes, and a fake browser that plays each audio chunk at once."""

    def __init__(self, auto_play=True, **kw):
        self.events, self.auto_play = [], auto_play
        self.tool_args = []
        self.tool_gate = threading.Event()
        self.tool_gate.set()
        self.llm = kw.pop("chat", ScriptedLLM())

        def tools(name, args):
            self.tool_args.append((name, args))
            self.tool_gate.wait(5)
            if name == "get_weather":
                return {"location": "內湖", "temperature_c": 24}, "內湖 24°C"
            return {"places": [{"name": "營養早午餐"}]}, "營養早午餐"

        self.agent = voice_agent.VoiceAgent(
            self.emit, chat=self.llm, tools=tools, think=False, allow_network=True,
            speak=lambda text, voice, rate: (b"RIFF", 0.5), **kw,
        )  # fmt: skip
        self.agent.start(warm=False)

    def emit(self, event):
        self.events.append(event)
        if self.auto_play and event["type"] == "audio":
            for kind in ("playing", "played"):
                self.agent.on_client({"type": kind, "speech_id": event["speech_id"],
                                      "seq": event["seq"]})  # fmt: skip

    def say(self, text, start, end):
        self.agent.on_asr({"type": "speech_start", "t_audio": start}, self.agent.t0)
        self.agent.on_asr({"type": "speech_end", "t_audio": end}, self.agent.t0)
        self.agent.on_asr({"type": "final", "text": text}, self.agent.t0)

    def wait(self, predicate, timeout=5.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if predicate():
                return
            time.sleep(0.01)
        raise AssertionError(f"timed out; events: {[e['type'] for e in self.events]}")

    def idles(self):
        return sum(e["type"] == "idle" for e in self.events)

    def of(self, kind):
        return [e for e in self.events if e["type"] == kind]


def test_demo_conversation_asks_then_looks_up_in_background_and_remembers_place():
    s = Session()
    s.say("今天天氣如何", 0.0, 1.2)
    s.wait(lambda: s.idles() >= 1)
    assert s.of("assistant")[-1]["text"] == "你現在在哪裡？"
    assert s.agent.turns[0]["decision"] == "ask" and not s.tool_args

    s.say("我在內湖", 3.0, 3.8)
    s.wait(lambda: s.idles() >= 2)
    assert s.tool_args == [("get_weather", {"location": "內湖"})]
    texts = [e["text"] for e in s.of("audio")]
    assert texts[1:] == ["好，我幫你查一下內湖的天氣。", "內湖現在二十四度，陰天。",
                         "今天降雨機率六成，記得帶傘。"]  # fmt: skip
    kinds = [e["kind"] for e in s.of("audio")]
    assert kinds[1:] == ["filler", "answer", "answer"]
    start = s.of("tool_start")[0]
    assert start["name"] == "get_weather" and start["args"] == {"location": "內湖"}
    assert s.of("tool_end")[0]["ok"] is True
    assert s.agent.memory["location"] == "內湖"

    s.say("謝謝這附近有什麼好吃的早午餐店嗎", 8.0, 10.0)
    s.wait(lambda: s.idles() >= 3)
    assert s.tool_args[-1] == ("search_places", {"location": "內湖", "query": "早午餐"})
    # the weather result came along with the third turn's words, not in the system prompt
    third = [c for c in s.llm.calls if "早午餐店嗎" in c["messages"][-1]["content"]][0]["messages"]
    assert voice_agent.LOOKUP_RULE in third[-1]["content"] and "get_weather" in third[-1]["content"]
    assert voice_agent.LOOKUP_RULE not in third[0]["content"]
    assert any(e["text"] == "不客氣！我幫你找一下內湖附近的早午餐店。" for e in s.of("audio"))

    summary = s.agent.close()
    assert [t["decision"] for t in summary["turns"]] == ["ask", "tool", "tool"]
    turn = summary["turns"][1]
    assert turn["response_s"] is not None and turn["answer_s"] >= turn["response_s"]
    assert turn["tool_s"] is not None
    assert len(summary["tool_calls"]) == 2 and summary["memory"] == {"location": "內湖"}
    assert s.of("done")[0]["summary"]["llm"] == voice_agent.DEFAULT_LLM
    # the composing LLM got the tool result and no tools
    compose = [c for c in s.llm.calls if c["messages"][-1]["content"].startswith("（系統：")]
    assert compose and all(c["tools"] is None for c in compose)


def test_listening_continues_while_a_tool_runs():
    s = Session()
    s.tool_gate.clear()  # the lookup hangs until released
    s.say("我在內湖", 0.0, 0.8)
    s.wait(lambda: len(s.of("tool_start")) == 1 and len(s.of("audio")) == 1)
    s.say("你好", 2.0, 2.5)  # a new question while the weather lookup is running
    s.wait(lambda: any(e["text"] == "好的，祝你用餐愉快！" for e in s.of("audio")))
    assert not s.of("tool_end")
    s.tool_gate.set()
    s.wait(lambda: len(s.of("tool_end")) == 1 and s.idles() >= 1)
    assert s.of("audio")[-1]["kind"] == "answer"
    s.agent.close()


def test_barge_in_stops_playback_and_cuts_the_reply():
    s = Session(auto_play=False, chat=ScriptedLLM(delay=0.05))
    s.say("謝謝這附近有什麼好吃的早午餐店嗎", 0.0, 1.0)
    s.agent.memory["location"] = "內湖"
    s.wait(lambda: any(e["kind"] == "answer" for e in s.of("audio")))
    answer = next(e for e in s.of("audio") if e["kind"] == "answer")
    s.agent.on_client({"type": "playing", "speech_id": answer["speech_id"], "seq": 0})
    s.agent.on_asr({"type": "speech_start", "t_audio": 5.0}, s.agent.t0)  # the user talks
    s.wait(lambda: s.of("barge_in"))
    stopped = s.of("barge_in")[0]["speech_ids"]
    assert answer["speech_id"] in stopped
    s.wait(lambda: s.agent.turns[0].get("answer_interrupted") is not None)
    assert s.agent.turns[0]["answer_interrupted"] is True
    sent = [e for e in s.of("audio") if e["speech_id"] == answer["speech_id"]]
    assert len(sent) < 3  # the rest of the list was never synthesised
    assert s.agent.speeches[answer["speech_id"]]["interrupted"] is True
    assert "被使用者打斷" in s.agent.history[-1]["content"]
    s.agent.close()


def test_no_barge_in_when_switched_off_and_no_tools_without_network():
    s = Session(auto_play=False, barge_in=False)
    s.agent.allow_network = False
    s.say("我在內湖", 0.0, 0.8)
    s.wait(lambda: s.of("audio"))
    assert s.llm.calls[0]["tools"] is None
    s.agent.on_asr({"type": "speech_start", "t_audio": 3.0}, s.agent.t0)
    assert not s.of("barge_in")
    s.agent.close()


def test_tool_calls_are_capped_per_session():
    s = Session(call_limit=1)
    s.say("我在內湖", 0.0, 0.8)
    s.wait(lambda: s.idles() >= 1)
    s.say("我在內湖", 2.0, 2.8)
    s.wait(lambda: s.idles() >= 2)
    assert len(s.tool_args) == 1
    assert s.of("tool_end")[-1]["ok"] is False
    s.agent.close()


def test_append_only_asr_commits_after_speech_ends():
    s = Session(append_only=True)
    s.agent.on_asr({"type": "speech_start", "t_audio": 0.0}, s.agent.t0)
    for piece in ["今天", "天氣如何"]:
        s.agent.on_asr({"type": "final", "text": piece}, s.agent.t0)
    assert s.of("user_partial")[-1]["text"] == "今天天氣如何"
    s.agent.on_asr({"type": "speech_end", "t_audio": 1.0}, s.agent.t0)
    s.wait(lambda: s.of("user_final"))
    assert s.of("user_final")[0]["text"] == "今天天氣如何"
    s.agent.close()


def test_early_commit_at_speech_end_and_only_new_words_from_the_final():
    s = Session()
    a = s.agent
    a.on_asr({"type": "speech_start", "t_audio": 0.0}, a.t0)
    a.on_asr({"type": "partial", "text": "今天天氣如何"}, a.t0)
    a.on_asr({"type": "speech_end", "t_audio": 1.0}, a.t0)
    s.wait(lambda: s.of("user_final"))
    assert s.of("user_final")[0]["text"] == "今天天氣如何"
    assert a.turns[0]["early_commit"] is True
    # the recogniser's own endpoint arrives later with the same text: no second turn
    a.on_asr({"type": "partial", "text": "今天天氣如何"}, a.t0)
    a.on_asr({"type": "final", "text": "今天天氣如何"}, a.t0)
    s.wait(lambda: s.idles() >= 1)
    assert len(a.turns) == 1 and a.turns[0]["asr_text"] == "今天天氣如何"
    # the user went on before the endpoint: only the new words make the next turn
    a.on_asr({"type": "speech_start", "t_audio": 3.0}, a.t0)
    a.on_asr({"type": "partial", "text": "我在"}, a.t0)
    a.on_asr({"type": "speech_end", "t_audio": 3.5}, a.t0)
    a.on_asr({"type": "speech_start", "t_audio": 4.0}, a.t0)
    a.on_asr({"type": "partial", "text": "我在內湖"}, a.t0)
    assert s.of("user_partial")[-1]["text"] == "內湖"
    a.on_asr({"type": "final", "text": "我在內湖"}, a.t0)
    s.wait(lambda: len(s.of("user_final")) == 3)
    assert [e["text"] for e in s.of("user_final")][1:] == ["我在", "內湖"]
    assert voice_agent.after_prefix("謝謝，這附近", "謝謝這") == "附近"
    assert voice_agent.after_prefix("你好", "謝謝") is None
    a.close()


def test_english_turns_get_english_rules_and_spaced_pieces():
    assert voice_agent.ENGLISH_RULE not in voice_agent.system_prompt({}, True)
    assert voice_agent.ENGLISH_RULE in voice_agent.system_prompt({}, True, english=True)
    assert not voice_agent._CJK.search(voice_agent.compose_prompt(english=True))
    lookups = {"search_places": '{"places": [{"name": "營養早午餐"}]}'}
    said = voice_agent.with_lookups("第一家開到幾點？", lookups)
    assert said.startswith(voice_agent.LOOKUP_RULE) and "營養早午餐" in said
    assert said.endswith("第一家開到幾點？")
    assert voice_agent.with_lookups("hi", {}) == "hi"
    assert voice_agent.LOOKUP_RULE_EN in voice_agent.with_lookups("When?", lookups, english=True)
    assert voice_agent.join_pieces("zero,", "so you") == "zero, so you"
    assert voice_agent.join_pieces("好，", "我幫你查") == "好，我幫你查"


def test_geocode_queries_cover_geonames_gaps():
    assert geo_tools.geocode_queries("台北")[:2] == ["台北", "台北市"]
    assert "臺南市" in geo_tools.geocode_queries("台南")
    assert geo_tools.geocode_queries("Xinyi Taipei")[1:] == ["Taipei", "Xinyi"]


def test_promise_catches_lookups_announced_without_a_call():
    for said in (
        "我幫你搜尋一下內湖附近的早午餐店。",
        "我需要先幫你查詢一下板橋明天的天氣狀況。",
        "好，我幫你查一下天氣。",
        "Sure, let me search for coffee shops.",
    ):
        assert voice_agent.PROMISE.search(said), said
    assert not voice_agent.PROMISE.search("我可以幫你查天氣、找附近的餐廳。")


def test_lookup_text_keeps_what_follow_ups_need():
    weather = {"location": "內湖", "temperature_c": 24, "condition": "陰",
               "today": {"min_c": 21, "max_c": 27, "rain_chance_pct": 60, "condition": "陣雨"},
               "tomorrow": None}  # fmt: skip
    assert voice_agent.lookup_text("get_weather", weather) == (
        "內湖 現在 24°C 陰；今天 21–27°C 陣雨，降雨機率 60%"
    )
    places = {"location": "內湖", "query": "早午餐", "places": [
        {"name": "營養早午餐", "distance_m": 350, "opening_hours": "Mo-Su 06:30-14:00",
         "address": "成功路三段187巷2號", "kind": "餐廳"}]}  # fmt: skip
    text = voice_agent.lookup_text("search_places", places)
    assert (
        text == "內湖 附近的早午餐：1. 營養早午餐，350 公尺，Mo-Su 06:30-14:00，成功路三段187巷2號"
    )
