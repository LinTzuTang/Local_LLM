"""Cloud speech models as the live voice assistant (voice_cloud.py) against a fake WebSocket:
gpt-realtime's tool round trip, and gpt-live-1's replies cut at its silences."""

import asyncio
import base64
import json
import time

import numpy as np

from local_llm import voice_cloud

LOUD = base64.b64encode((np.ones(2400) * 8000).astype("<i2").tobytes()).decode()  # 0.1 s
QUIET = base64.b64encode(np.zeros(2400, dtype="<i2").tobytes()).decode()


class FakeSocket:
    """Plays scripted server events; `react` maps a client event type to events to send back."""

    def __init__(self, events, react=None):
        self.queue: asyncio.Queue = asyncio.Queue()
        for e in events:
            self.queue.put_nowait(e)
        self.react, self.sent = react or {}, []

    async def send(self, text):
        event = json.loads(text)
        self.sent.append(event)
        for reply in self.react.pop(event["type"], []):
            await self.queue.put(reply)

    async def recv(self):
        return json.dumps(await self.queue.get())

    def __aiter__(self):
        return self

    async def __anext__(self):
        event = await self.queue.get()
        if event is None:
            raise StopAsyncIteration
        if "sleep" in event:
            await asyncio.sleep(event["sleep"])
            return json.dumps({"type": "noop"})
        return json.dumps(event)

    async def close(self):
        await self.queue.put(None)


def session(cls, socket, **kw):
    events = []

    def emit(e):
        events.append(e)
        if e["type"] == "audio":
            for kind in ("playing", "played"):
                agent.on_client({"type": kind, "speech_id": e["speech_id"], "seq": e["seq"]})

    async def connect(path):
        return socket

    agent = cls(emit, connect=connect, **kw)
    agent.tools = lambda name, args: (
        {"location": args.get("location"), "temperature_c": 24},
        "24°C",
    )
    agent.start()
    return agent, events


def wait(predicate, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("timed out")


def text(rid, delta):
    return {"type": "response.output_audio_transcript.delta", "response_id": rid, "delta": delta}


def test_realtime_calls_the_tool_here_and_speaks_the_answer():
    call = {"type": "function_call", "name": "get_weather", "call_id": "x1",
            "arguments": json.dumps({"location": "內湖"})}  # fmt: skip
    heard = {"type": "conversation.item.input_audio_transcription.completed", "item_id": "i1",
             "transcript": "內湖天氣"}  # fmt: skip
    socket = FakeSocket(
        [{"sleep": 0.3}, heard, {"type": "response.created", "response": {"id": "r1"}},
         text("r1", "我查一下。"),
         {"type": "response.output_audio.delta", "response_id": "r1", "delta": LOUD},
         {"type": "response.done", "response": {"id": "r1", "output": [call]}}],
        react={"response.create": [
            {"type": "response.created", "response": {"id": "r2"}},
            text("r2", "內湖 24 度。"),
            {"type": "response.output_audio.delta", "response_id": "r2", "delta": LOUD},
            {"type": "response.done", "response": {"id": "r2", "output": []}}]},
    )  # fmt: skip
    agent, events = session(voice_cloud.RealtimeAgent, socket, allow_network=True)
    agent.on_asr({"type": "speech_start", "t_audio": 0.0}, agent.t0)  # the gateway's VAD
    agent.on_asr({"type": "speech_end", "t_audio": 0.2}, agent.t0)
    wait(
        lambda: any(
            e["type"] == "assistant" and e.get("final") and e["kind"] == "answer" for e in events
        )
    )
    update = next(e for e in socket.sent if e["type"] == "session.update")["session"]
    assert {t["name"] for t in update["tools"]} == {"get_weather", "search_places"}
    output = next(e for e in socket.sent if e["type"] == "conversation.item.create")["item"]
    assert output["call_id"] == "x1" and "24" in output["output"]
    kinds = [e["kind"] for e in events if e["type"] == "assistant" and e.get("final")]
    assert kinds == ["filler", "answer"]
    assert [e["type"] for e in events].count("tool_end") == 1
    finals = [e for e in events if e["type"] == "user_final"]
    assert [e["text"] for e in finals] == ["內湖天氣"]  # the caption is the service's transcript
    update_input = update["audio"]["input"]
    assert update_input["transcription"]["model"] == voice_cloud.TRANSCRIBE_DEPLOYMENT
    summary = agent.close()
    assert summary["engine"] == "gpt-realtime" and summary["turns"][0]["decision"] == "tool"
    assert (
        summary["turns"][0]["user"] == "內湖天氣" and summary["turns"][0]["speech_end"] is not None
    )


def test_realtime_without_network_gets_no_tools():
    socket = FakeSocket([])
    agent, _ = session(voice_cloud.RealtimeAgent, socket, allow_network=False)
    wait(lambda: socket.sent)
    update = socket.sent[0]["session"]
    assert "tools" not in update and "設定" in update["instructions"]
    agent.close()


def test_live_cuts_replies_at_its_silences_and_ignores_silent_audio():
    started = {"type": "session.started"}
    talk = [{"type": "session.output_audio.delta", "delta": LOUD}] * 3
    hush = [{"type": "session.output_audio.delta", "delta": QUIET}] * 5
    socket = FakeSocket(
        [started, *hush, *talk, {"sleep": 1.6}, *hush, *talk, {"sleep": 1.6}, *hush]
    )
    agent, events = session(voice_cloud.LiveAgent, socket)
    wait(lambda: len([e for e in events if e["type"] == "assistant" and e.get("final")]) == 2)
    finals = [e for e in events if e["type"] == "assistant" and e.get("final")]
    assert len({e["speech_id"] for e in finals}) == 2  # two replies, not one long one
    assert (
        socket.sent[0]["type"] == "session.start" and "delegation" not in socket.sent[0]["session"]
    )
    agent.close()


def test_live_starts_a_new_reply_once_the_user_has_spoken():
    started = {"type": "session.started"}
    talk = [{"type": "session.output_audio.delta", "delta": LOUD}] * 3
    socket = FakeSocket([started, *talk, {"sleep": 0.3}, *talk, {"sleep": 1.6}])
    agent, events = session(voice_cloud.LiveAgent, socket)
    wait(lambda: any(e["type"] == "audio" for e in events))
    agent.on_asr({"type": "speech_start", "t_audio": agent.now() + 0.05}, agent.t0)  # the user
    wait(lambda: len([e for e in events if e["type"] == "assistant" and e.get("final")]) == 2)
    agent.close()


def test_live_captions_are_its_own_transcript():
    started = {"type": "session.started"}
    heard = [
        {"type": "session.input_transcript.delta", "delta": d} for d in ("台中", "明天會下雨嗎")
    ]
    socket = FakeSocket([started, *heard, {"sleep": 1.5}])
    agent, events = session(voice_cloud.LiveAgent, socket)
    agent.on_asr({"type": "speech_start", "t_audio": 0.0}, agent.t0)
    agent.on_asr({"type": "speech_end", "t_audio": 0.1}, agent.t0)
    wait(lambda: any(e["type"] == "user_final" for e in events))
    final = next(e for e in events if e["type"] == "user_final")
    assert final["text"] == "台中明天會下雨嗎" and final["turn"] == 0
    assert any(e["type"] == "user_partial" for e in events)
    agent.close()


class GeminiSocket(FakeSocket):
    """Gemini Live's messages have no "type": `react` is keyed by their one top-level key."""

    async def send(self, text):
        event = json.loads(text)
        self.sent.append(event)
        for reply in self.react.pop(next(iter(event)), []):
            await self.queue.put(reply)


def gemini_turn(text="", audio=False, **flags):
    content = dict(flags)
    if audio:
        content["modelTurn"] = {"parts": [{"inlineData": {"data": LOUD}}]}
    if text:
        content["outputTranscription"] = {"text": text}
    return {"serverContent": content}


def test_gemini_live_calls_the_tool_here_and_speaks_the_answer():
    call = {"id": "f1", "name": "get_weather", "args": {"location": "台北"}}
    socket = GeminiSocket(
        [{"setupComplete": {}}, {"sleep": 0.3},
         {"serverContent": {"inputTranscription": {"text": "台 北 天 氣"}}},
         gemini_turn("我查一下。", audio=True), {"toolCall": {"functionCalls": [call]}}],
        react={"toolResponse": [gemini_turn("台北 24 度。", audio=True),
                                gemini_turn(turnComplete=True)]},
    )  # fmt: skip
    agent, events = session(voice_cloud.GeminiLiveAgent, socket, engine="gemini-3.8-live",
                            allow_network=True)  # fmt: skip
    agent.on_asr({"type": "speech_start", "t_audio": 0.0}, agent.t0)
    agent.on_asr({"type": "speech_end", "t_audio": 0.2}, agent.t0)
    wait(lambda: any(e["type"] == "assistant" and e.get("final") and e["kind"] == "answer"
                     for e in events))  # fmt: skip
    setup = socket.sent[0]["setup"]
    assert setup["model"].endswith("/gemini-3.8-live")
    declared = setup["tools"][0]["functionDeclarations"]
    assert {d["name"] for d in declared} == {"get_weather", "search_places"}
    reply = next(e for e in socket.sent if "toolResponse" in e)["toolResponse"]
    assert reply["functionResponses"][0]["id"] == "f1"
    assert "24" in reply["functionResponses"][0]["response"]["result"]
    kinds = [e["kind"] for e in events if e["type"] == "assistant" and e.get("final")]
    assert kinds == ["filler", "answer"]
    finals = [e["text"] for e in events if e["type"] == "user_final"]
    assert finals == ["台北天氣"]  # its transcript, spaces between characters dropped
    summary = agent.close()
    assert summary["kind"] == "turn" and summary["turns"][0]["decision"] == "tool"


def test_gemini_live_without_barge_in_asks_not_to_be_interrupted():
    socket = GeminiSocket([{"setupComplete": {}}])
    agent, _ = session(voice_cloud.GeminiLiveAgent, socket, engine="gemini-live-2.5",
                       barge_in=False)  # fmt: skip
    wait(lambda: socket.sent)
    setup = socket.sent[0]["setup"]
    assert setup["realtimeInputConfig"] == {"activityHandling": "NO_INTERRUPTION"}
    assert "tools" not in setup  # no network: nothing to look up
    agent.close()


class FakeNova:
    """A Bedrock bidirectional stream: input events recorded, output events scripted."""

    def __init__(self, events):
        self.out: asyncio.Queue = asyncio.Queue()
        for e in events:
            self.out.put_nowait(e)
        self.sent, stream = [], self

        class Input:
            async def send(self, chunk):
                stream.sent.append(json.loads(chunk.value.bytes_)["event"])

            async def close(self):
                await stream.out.put(None)

        self.input_stream = Input()

    async def await_output(self):
        stream = self

        class Output:
            async def receive(self):
                event = await stream.out.get()
                if event is None:
                    return None

                class Value:
                    bytes_ = json.dumps({"event": event}).encode()

                class Event:
                    value = Value()

                return Event()

        return None, Output()


def nova(kind, **body):
    return {kind: body}


def test_nova_joins_audio_blocks_of_one_reply_and_shows_the_speculative_text():
    stage = lambda s: json.dumps({"generationStage": s})  # noqa: E731
    stream = FakeNova([
        nova("contentStart", contentId="u", type="TEXT", role="USER"),
        nova("textOutput", contentId="u", role="USER", content="你好"),
        nova("contentEnd", contentId="u", stopReason="END_TURN"),
        nova("contentStart", contentId="t", type="TEXT", role="ASSISTANT",
             additionalModelFields=stage("SPECULATIVE")),
        nova("textOutput", contentId="t", role="ASSISTANT", content="你好！"),
        nova("contentEnd", contentId="t", stopReason="PARTIAL_TURN"),
        nova("contentStart", contentId="a1", type="AUDIO", role="ASSISTANT"),
        nova("audioOutput", contentId="a1", content=LOUD),
        nova("contentEnd", contentId="a1", stopReason="PARTIAL_TURN"),
        nova("contentStart", contentId="a2", type="AUDIO", role="ASSISTANT"),
        nova("audioOutput", contentId="a2", content=LOUD),
        nova("contentEnd", contentId="a2", stopReason="END_TURN"),
    ])  # fmt: skip

    async def connect(_):
        return stream

    events = []
    agent = voice_cloud.NovaSonicAgent(events.append, connect=connect)
    agent.start()
    wait(lambda: any(e["type"] == "assistant" and e.get("final") for e in events))
    finals = [e for e in events if e["type"] == "assistant" and e.get("final")]
    assert [(e["kind"], e["text"]) for e in finals] == [("reply", "你好！")]
    assert [e["text"] for e in events if e["type"] == "user_final"] == ["你好"]
    start = next(e for e in stream.sent if "promptStart" in e)["promptStart"]
    assert start["audioOutputConfiguration"]["voiceId"] == "tiffany"
    assert "toolConfiguration" not in start  # no network: no tools
    agent.close()
    assert {"sessionEnd": {}} in stream.sent
