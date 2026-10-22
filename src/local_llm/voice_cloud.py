"""Cloud speech models as the live voice assistant, for comparison: Azure OpenAI gpt-realtime
(② turn-based, calls the tools itself) and gpt-live-1 (③ full-duplex, delegates lookups to a
background text model). The microphone audio leaves this machine: the settings say so.

Both stream the microphone (16 kHz PCM16 from the browser, resampled to 24 kHz) to the
service and play its speech as it arrives, in pieces of about CHUNK_S seconds. No ASR runs
here: the user's captions are the service's own transcript (gpt-live-1 sends it anyway;
gpt-realtime transcribes with TRANSCRIBE_DEPLOYMENT), and a plain energy VAD in the gateway
marks when the user spoke, for the timeline and the latency figures. With the
network switch on they get the cascade's two lookups (voice_agent.run_tool: Open-Meteo,
OpenStreetMap), run here and shown on the page like the cascade's; with it off they are told
they cannot look anything up.

    gpt-realtime: WebSocket /openai/v1/realtime?model=<deployment>; the server's VAD (500 ms of
       silence) ends the user's turn and starts the reply, and cuts the reply off when the user
       speaks over it (the page stops playback at once). Function calls come in response.done;
       their results go back as function_call_output items, then response.create.
    gpt-live-1: WebSocket /openai/v1/live/sessions; one session.start with a Responses
       delegation (the background model and the tools). The model decides itself when to talk;
       a reply ends after GAP_S of silence from it. On this deployment long sessions fall
       behind real time and function tools can drop the connection (see the evaluation).

Credentials: AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY from the environment or the
project's .env (never sent to the page).
"""

import asyncio
import base64
import json
import os
import queue
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np

from . import voice_agent
from .voice_agent import join_pieces, speakable
from .voice_omni import OmniAgent, wav_bytes

PROJECT = Path(__file__).resolve().parents[2]
OUT_RATE = 24000
CHUNK_S = 0.5
GAP_S = 1.2
HEARD_S = 0.8  # gpt-live-1's transcript of the user is final this long after its last piece
TRANSCRIBE_DEPLOYMENT = os.getenv("LOCAL_LLM_REALTIME_TRANSCRIBE", "gpt-4o-mini-transcribe")
RESULT_WAIT_S = 5.0  # gpt-live-1 speaks a delegated result about 2 s after it is ready
VOICED_RMS = 0.01  # gpt-live-1's output audio below this is its silence
REALTIME_DEPLOYMENT = os.getenv("LOCAL_LLM_REALTIME_DEPLOYMENT", "gpt-realtime")
LIVE_BACKEND = os.getenv("LOCAL_LLM_LIVE_BACKEND", "gpt-5.5")
INSTRUCTIONS = (
    "你是即時語音助理。用台灣慣用的繁體中文口語回答，一到兩句、簡短自然；使用者說英文就用英文回答。"
    "使用者問天氣但沒說在哪裡時，先簡短反問地點；問題裡已經有地名就直接查。需要天氣或附近店家等即時資訊時，"
    "先用一句話告訴使用者你要查，再查，查到後根據結果回答；不要編造溫度、降雨機率、店名、營業時間或地址，"
    "查不到就說查不到。"
)
OFFLINE = (
    "你是即時語音助理。用台灣慣用的繁體中文口語回答，一到兩句、簡短自然；使用者說英文就用英文回答。"
    "這次沒有開放查詢：被問到即時天氣或附近店家時，老實說要先在設定打開連網查詢，不要編造。"
)
BACKEND_INSTRUCTIONS = (
    "Use the tools whenever current weather or nearby places are needed. Answer in the "
    "user's language (Taiwanese Mandarin in Traditional characters, or English), in one or "
    "two short spoken sentences, only from the tool results."
)
ENGINES = {
    "gpt-realtime": {
        "kind": "turn",
        "name": f"Azure {REALTIME_DEPLOYMENT}",
        "voice": "marin",
        "cloud": True,
    },  # fmt: skip
    "gpt-live-1": {
        "kind": "duplex",
        "name": "Azure gpt-live-1",
        "voice": "gpt-live-1",
        "cloud": True,
    },  # fmt: skip
}


def credentials() -> tuple[str, str] | None:
    env = dict(os.environ)
    dotenv = PROJECT / ".env"
    if dotenv.exists():
        for line in dotenv.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                env.setdefault(k.strip(), v.strip().strip("'\""))
    endpoint, key = env.get("AZURE_OPENAI_ENDPOINT"), env.get("AZURE_OPENAI_API_KEY")
    if not endpoint or not key:
        return None
    return endpoint.split("/openai")[0].rstrip("/").replace("https://", "wss://", 1), key


def available(engine: str) -> bool:
    return engine in ENGINES and credentials() is not None


def tool_specs() -> list[dict]:
    """The cascade's tools in the Realtime / Responses function form."""
    out = []
    for t in voice_agent.TOOL_SPECS:
        f = t["function"]
        out.append({"type": "function", "name": f["name"], "description": f["description"],
                    "parameters": f["parameters"]})  # fmt: skip
    return out


def to_24k(pcm16k: bytes) -> bytes:
    x = np.frombuffer(pcm16k, dtype="<i2").astype(np.float32)
    if not len(x):
        return b""
    n = int(len(x) * 1.5)
    y = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)
    return np.clip(y, -32768, 32767).astype("<i2").tobytes()


class CloudAgent(OmniAgent):
    """The shared part: a WebSocket client on its own event loop thread."""

    def __init__(self, emit: Callable[[dict], None], *, engine: str, allow_network: bool = False,
                 barge_in: bool = True, display=None, connect=None):  # fmt: skip
        super().__init__(emit, None, engine=engine, barge_in=barge_in, display=display)
        info = ENGINES[engine]
        self.llm, self.voice, self.allow_network = info["name"], info["voice"], allow_network
        self.mic: queue.Queue = queue.Queue()
        self.connect = connect  # for tests: an async factory returning a websocket-like object
        self.loop: asyncio.AbstractEventLoop | None = None
        self.ws = None
        self.pending: dict[str, bytearray] = {}  # speech id -> audio not yet sent
        self.caption: dict | None = None  # the user's turn the service is still transcribing
        self.last_heard = 0.0  # when its transcript of the user last grew
        self.net = threading.Thread(target=self._run, name=f"voice-{engine}", daemon=True)

    # -- plumbing -----------------------------------------------------------------------------

    def start(self, warm: bool = True) -> None:
        super().start(warm=False)
        self.net.start()

    def on_audio(self, data: bytes) -> None:
        self.mic.put(data)

    def _user(self, text: str, early: bool = False) -> None:
        """Captions only: the service takes the turns."""
        with self.lock:
            span = self.vad[-1] if self.vad else [None, None]
            turn = {"turn": len(self.turns), "user": text, "speech_start": span[0],
                    "speech_end": span[1] or self.now(), "asr_final": self.now(),
                    "speech_ids": [], "tool_calls": [], "assistant": "", "decision": None,
                    "early_commit": early}  # fmt: skip
            self.turns.append(turn)
        self.emit({"type": "user_final", "turn": turn["turn"], "text": text,
                   "speech_end": turn["speech_end"]})  # fmt: skip

    # -- the user's words: the service's own transcript ------------------------------------------

    def on_asr(self, event: dict, audio_t0: float | None = None) -> None:
        """Only the energy VAD comes here (no ASR): user speech on the timeline, and a turn
        opened as soon as the user starts, for the service's transcript to fill (the
        transcript can arrive after the model has begun to answer)."""
        super().on_asr(event, audio_t0)
        if event.get("type") in ("speech_start", "speech_end"):
            turn = self._caption_turn()
            if not turn["user"]:  # its transcript can come after its reply: hold the place
                self.emit({"type": "user_partial", "text": "…", "turn": turn["turn"]})

    def _caption_turn(self) -> dict:
        span = self.vad[-1] if self.vad else [self.now(), None]
        with self.lock:
            if self.caption is None:
                self.caption = {"turn": len(self.turns), "user": "", "speech_start": span[0],
                                "speech_end": span[1], "asr_final": None, "speech_ids": [],
                                "tool_calls": [], "assistant": "", "decision": None,
                                "early_commit": False}  # fmt: skip
                self.turns.append(self.caption)
            elif span[1] is not None:  # still the same words: the utterance went on
                self.caption["speech_end"] = span[1]
            return self.caption

    def _caption(self, text: str, final: bool = False) -> None:
        turn = self._caption_turn()
        turn["user"] = self.display(text).strip()
        self.last_heard = self.clock()
        if not final:
            self.emit({"type": "user_partial", "text": turn["user"], "turn": turn["turn"]})
            return
        with self.lock:
            self.caption = None
            turn["asr_final"] = self.now()
            if turn["speech_end"] is None:
                turn["speech_end"] = turn["asr_final"]
        self.emit({"type": "user_final", "turn": turn["turn"], "text": turn["user"],
                   "speech_end": turn["speech_end"]})  # fmt: skip

    def _run(self) -> None:
        self.loop = asyncio.new_event_loop()
        try:
            self.loop.run_until_complete(self._session())
        except Exception as exc:
            self.emit({"type": "error", "error": f"{self.llm}: {type(exc).__name__}: {exc}"})
        finally:
            self.ready.set()

    async def _open(self, path: str):
        if self.connect is not None:
            return await self.connect(path)
        import websockets

        creds = credentials()
        if creds is None:
            raise RuntimeError("AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY are not set")
        base, key = creds
        return await websockets.connect(base + path, additional_headers={"api-key": key},
                                        max_size=None)  # fmt: skip

    async def _send(self, event: dict) -> None:
        await self.ws.send(json.dumps(event))

    async def _microphone(self, kind: str) -> None:
        while not self.closed:
            chunks = []
            while not self.mic.empty():
                chunks.append(self.mic.get_nowait())
            if chunks:
                audio = base64.b64encode(to_24k(b"".join(chunks))).decode()
                await self._send({"type": kind, "audio": audio})
            await asyncio.sleep(0.05)

    def _turn(self) -> dict | None:
        return self.turns[-1] if self.turns else None

    def _speech_text(self, speech: dict, delta: str) -> None:
        speech["text"] += delta
        self.emit({"type": "assistant", "speech_id": speech["speech_id"], "turn": speech["turn"],
                   "kind": speech["kind"], "text": speakable(speech["text"]),
                   "final": False})  # fmt: skip

    def _speech_audio(self, speech: dict, pcm: bytes, flush: bool = False) -> None:
        buf = self.pending.setdefault(speech["speech_id"], bytearray())
        buf += pcm
        first = speech["seq"] == 0
        if buf and (flush or first or len(buf) >= CHUNK_S * OUT_RATE * 2):
            self.send_wav(speech, wav_bytes(bytes(buf), OUT_RATE))
            buf.clear()

    def _end_speech(self, speech: dict) -> None:
        if self.pending.get(speech["speech_id"]):
            self._speech_audio(speech, b"", flush=True)
        self.pending.pop(speech["speech_id"], None)
        turn = self.turns[speech["turn"]] if speech["turn"] is not None else None
        if turn is not None:
            said = speakable(speech["text"])
            key = "answer" if speech["kind"] == "answer" else "assistant"
            turn[key] = join_pieces(turn.get(key) or "", said) if turn.get(key) else said
            turn["llm_end"] = self.now()
        self._finish_speech(speech)
        if turn is not None:
            self._turn_metrics(turn)
        self._check_idle()

    def _barge(self, speeches: list[dict]) -> None:
        with self.lock:
            ids = {s["speech_id"] for s in speeches}
            stopped = sorted({sid for sid, _ in self.unplayed if sid in ids} | {
                s["speech_id"] for s in speeches if s.get("generating")})  # fmt: skip
            self.unplayed = {u for u in self.unplayed if u[0] not in ids}
            for s in speeches:
                s["interrupted"] = True
        if stopped:
            self.emit({"type": "barge_in", "speech_ids": stopped, "at": self.now()})

    async def _tool(self, name: str, args: dict, turn: dict | None) -> str:
        """Run one lookup here (with the page's tool events) and return its digest."""
        call = {"call_id": f"c{len(self.calls)}", "turn": turn["turn"] if turn else None,
                "name": name, "args": args, "start": self.now(), "end": None,
                "ok": None}  # fmt: skip
        self.calls.append(call)
        if turn is not None:
            turn["tool_calls"].append(call["call_id"])
            turn["decision"] = "tool"
        self.emit({"type": "tool_start", **{k: call[k] for k in ("call_id", "turn", "name", "args",
                                                                   "start")}})  # fmt: skip
        count = self.call_counts.get(name, 0)
        started = self.clock()
        with self.lock:
            self.running_tools += 1  # not idle while a lookup runs
        if count >= self.call_limit:
            result, summary = {"error": f"session limit of {self.call_limit} calls"}, "超過上限"
        else:
            self.call_counts[name] = count + 1
            try:
                result, summary = await asyncio.to_thread(self.tools, name, args)
            except Exception as exc:
                result, summary = {"error": str(exc)}, f"失敗：{exc}"
        with self.lock:
            self.running_tools -= 1
        call.update(end=self.now(), ok="error" not in result, summary=summary,
                    seconds=round(self.clock() - started, 3), result=result)  # fmt: skip
        self.emit({"type": "tool_end", "call_id": call["call_id"], "ok": call["ok"],
                   "summary": summary, "seconds": call["seconds"], "end": call["end"],
                   "result": voice_agent._short(name, result)})  # fmt: skip
        return voice_agent.tool_digest(name, result)

    def close(self) -> dict:
        self.closed = True  # the loops stop; the summary is built below
        if self.loop is not None and self.ws is not None:
            try:
                asyncio.run_coroutine_threadsafe(self.ws.close(), self.loop).result(5)
            except Exception:
                pass
        self.net.join(timeout=5)
        self.closed = False
        return super().close()


class RealtimeAgent(CloudAgent):
    """② gpt-realtime: the server's VAD takes the turns; the model calls the tools itself."""

    def __init__(self, emit, **kwargs):
        super().__init__(emit, engine="gpt-realtime", **kwargs)
        self.responses: dict[str, dict] = {}  # response id -> speech
        self.after_tool = False
        self.active = 0  # responses in progress (only one at a time may be created)
        self.heard: dict[str, str] = {}  # input item -> its transcript so far

    async def _session(self) -> None:
        started = self.clock()
        self.ws = await self._open(f"/openai/v1/realtime?model={REALTIME_DEPLOYMENT}")
        session = {
            "type": "realtime", "output_modalities": ["audio"],
            "instructions": INSTRUCTIONS if self.allow_network else OFFLINE,
            "audio": {"input": {"format": {"type": "audio/pcm", "rate": OUT_RATE},
                                "turn_detection": {"type": "server_vad", "silence_duration_ms": 500,
                                                   "create_response": True,
                                                   "interrupt_response": self.barge_in},
                                "transcription": {"model": TRANSCRIBE_DEPLOYMENT}},
                      "output": {"format": {"type": "audio/pcm", "rate": OUT_RATE},
                                 "voice": "marin"}},
        }  # fmt: skip
        if self.allow_network:
            session |= {"tools": tool_specs(), "tool_choice": "auto"}
        await self._send({"type": "session.update", "session": session})
        self.llm_load_s = round(self.clock() - started, 2)
        self.ready.set()
        self.emit({"type": "llm_ready", "llm": self.llm, "seconds": self.llm_load_s})
        mic = asyncio.create_task(self._microphone("input_audio_buffer.append"))
        try:
            async for message in self.ws:
                await self._event(json.loads(message))
                if self.closed:
                    break
        finally:
            mic.cancel()

    async def _event(self, e: dict) -> None:
        kind = e.get("type", "")
        if kind == "response.created":
            self.active += 1
            turn = self._turn()
            speech = self._new_speech(turn, "answer" if self.after_tool else "reply")
            if self.after_tool:
                with self.lock:
                    self.running_tools -= 1  # the answer to the lookups has started
            self.after_tool = False
            self.responses[e["response"]["id"]] = speech
            if turn is not None and turn.get("llm_start") is None:
                turn["llm_start"] = self.now()
        elif kind == "response.output_audio.delta":
            speech = self.responses.get(e.get("response_id"))
            if speech is not None and not speech["interrupted"]:
                self._speech_audio(speech, base64.b64decode(e["delta"]))
        elif kind == "response.output_audio_transcript.delta":
            speech = self.responses.get(e.get("response_id"))
            if speech is not None:
                self._speech_text(speech, e.get("delta", ""))
        elif kind == "conversation.item.input_audio_transcription.delta":
            item = e.get("item_id", "")
            self.heard[item] = self.heard.get(item, "") + e.get("delta", "")
            self._caption(self.heard[item])
        elif kind == "conversation.item.input_audio_transcription.completed":
            text = e.get("transcript") or self.heard.pop(e.get("item_id", ""), "")
            if text.strip():
                self._caption(text, final=True)
        elif kind == "input_audio_buffer.speech_started" and self.barge_in:
            live = [s for s in self.responses.values() if s.get("generating")
                    or any(sid == s["speech_id"] for sid, _ in self.unplayed)]  # fmt: skip
            self._barge(live)
        elif kind == "response.done":
            self.active = max(0, self.active - 1)
            resp = e.get("response") or {}
            speech = self.responses.pop(resp.get("id"), None)
            calls = [i for i in resp.get("output") or [] if i.get("type") == "function_call"]
            if calls:
                with self.lock:
                    self.running_tools += 1  # busy until the answer to the lookups is asked for
            if speech is not None:
                if calls:
                    speech["kind"] = "filler"
                turn = self.turns[speech["turn"]] if speech["turn"] is not None else None
                if turn is not None and not calls and turn.get("decision") is None:
                    text = speakable(speech["text"])
                    turn["decision"] = "ask" if text.endswith(("？", "?")) else "answer"
                self._end_speech(speech)
            if calls:
                asyncio.create_task(self._answer_calls(calls))
        elif kind == "error":
            self.emit(
                {"type": "error", "error": f"{self.llm}: {(e.get('error') or {}).get('message')}"}
            )

    async def _answer_calls(self, calls: list[dict]) -> None:
        """Busy from the function calls until the reply to their results starts (the next
        response.created), so the page does not take the gap for the end of the turn."""
        try:
            await self._answer_calls_now(calls)
        except Exception:
            with self.lock:
                self.running_tools -= 1
            self._check_idle()
            raise

    async def _answer_calls_now(self, calls: list[dict]) -> None:
        turn = self._turn()
        for item in calls:
            try:
                args = json.loads(item.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            digest = await self._tool(item.get("name"), args, turn)
            await self._send(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": item["call_id"],
                        "output": digest,
                    },
                }
            )
        for _ in range(300):  # another reply (to newer words) is still being spoken
            if not self.active:
                break
            await asyncio.sleep(0.1)
        self.after_tool = True
        await self._send({"type": "response.create"})


class LiveAgent(CloudAgent):
    """③ gpt-live-1: one model listens and talks all the time; lookups go to its delegate."""

    def __init__(self, emit, **kwargs):
        kwargs["barge_in"] = False  # the model decides
        super().__init__(emit, engine="gpt-live-1", **kwargs)
        self.speech: dict | None = None
        self.last_audio = 0.0
        self.step_called: set[str] = set()
        self.delegations: set[str] = set()  # background work in progress
        self.result_due: list[float] = []  # finished background work not spoken yet (deadlines)

    async def _session(self) -> None:
        started = self.clock()
        self.ws = await self._open("/openai/v1/live/sessions")
        session = {
            "model": "gpt-live-1",
            "instructions": INSTRUCTIONS if self.allow_network else OFFLINE,
        }
        if self.allow_network:
            session["delegation"] = {"type": "responses", "responses": {
                "model": LIVE_BACKEND, "instructions": BACKEND_INSTRUCTIONS, "tools": tool_specs(),
                "tool_choice": "auto", "parallel_tool_calls": False}}  # fmt: skip
        await self._send({"type": "session.start", "session": session})
        first = json.loads(await self.ws.recv())
        if first.get("type") != "session.started":
            raise RuntimeError(f"session.start: {json.dumps(first, ensure_ascii=False)[:300]}")
        self.llm_load_s = round(self.clock() - started, 2)
        self.ready.set()
        self.emit({"type": "llm_ready", "llm": self.llm, "seconds": self.llm_load_s})
        mic = asyncio.create_task(self._microphone("session.input_audio.append"))
        gaps = asyncio.create_task(self._gaps())
        try:
            async for message in self.ws:
                await self._event(json.loads(message))
                if self.closed:
                    break
        finally:
            mic.cancel()
            gaps.cancel()
            if self.speech is not None:
                self._end_speech(self.speech)
                self.speech = None

    def _current(self) -> dict:
        """The reply being spoken; a new one once the user has spoken since it began (in full
        duplex it may answer new words without a pause, which is a new reply all the same)."""
        if self.speech is not None and self.vad and self.speech.get("began") is not None:
            if self.vad[-1][0] is not None and self.vad[-1][0] > self.speech["began"]:
                self._close_speech()
        if self.speech is None:
            turn = self._turn()
            self.speech = self._new_speech(turn, "reply")
            self.speech["began"] = self.now()
            if turn is not None and turn.get("llm_start") is None:
                turn["llm_start"] = self.now()
        return self.speech

    def _close_speech(self) -> None:
        speech, self.speech = self.speech, None
        turn = self.turns[speech["turn"]] if speech["turn"] is not None else None
        if turn is not None and turn.get("decision") is None:
            text = speakable(speech["text"])
            turn["decision"] = "ask" if text.endswith(("？", "?")) else "answer"
        self._end_speech(speech)

    async def _gaps(self) -> None:
        """A reply ends after GAP_S without speech from the model."""
        while not self.closed:
            await asyncio.sleep(0.2)
            caption = self.caption  # no end event for its transcript: done once the user
            vad_closed = bool(self.vad) and self.vad[-1][1] is not None  # stopped and it went quiet
            if (
                caption
                and caption["user"]
                and vad_closed
                and self.clock() - self.last_heard > HEARD_S
            ):
                self._caption(caption["user"], final=True)
            if self.result_due and self.clock() > self.result_due[0]:  # never spoken
                self._speaks()
            if self.speech is not None and self.clock() - self.last_audio > GAP_S:
                self._close_speech()

    async def _event(self, e: dict) -> None:
        kind = e.get("type", "")
        if kind == "session.output_audio.delta":
            # it streams audio all the time, silence included: only speech starts or extends a reply
            pcm = base64.b64decode(e["delta"])
            x = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
            if len(x) and float(np.sqrt((x**2).mean())) > VOICED_RMS:
                self.last_audio = self.clock()
                self._speaks()
                self._speech_audio(self._current(), pcm)
            elif self.speech is not None:
                self._speech_audio(self.speech, pcm)  # a pause inside the reply
        elif kind == "session.input_transcript.delta":
            text = (self.caption or {}).get("user", "") + e.get("delta", "")
            self._caption(text)
        elif kind == "session.output_transcript.delta":
            self.last_audio = max(self.last_audio, self.clock() - GAP_S / 2)
            self._speech_text(self._current(), e.get("delta", ""))
            if (
                self.speech is not None
                and self.speech["text"].strip() == e.get("delta", "").strip()
            ):
                self.last_audio = self.clock()  # a new reply: give its audio time to arrive
        elif kind == "session.delegation.created":  # busy while the background model works
            did = (e.get("delegation") or {}).get("id")
            if did and did not in self.delegations:
                self.delegations.add(did)
                with self.lock:
                    self.running_tools += 1
        elif kind == "response.event":
            inner, did = e.get("event") or {}, e.get("delegation_id")
            item, step = inner.get("item") or {}, inner.get("type", "")
            if step == "response.output_item.done" and item.get("type") == "function_call":
                self.step_called.add(did)
                asyncio.create_task(self._answer_call(item))
            elif step in ("response.completed", "response.failed", "response.incomplete", "error"):
                # completed with a tool call: response.create continues it; otherwise it is done
                if did in self.step_called and step == "response.completed":
                    self.step_called.discard(did)
                elif did in self.delegations:
                    # the background is done, but the result is still to be spoken: busy
                    # until the model starts talking again (or RESULT_WAIT_S), see _speaks
                    self.delegations.discard(did)
                    self.result_due.append(self.clock() + RESULT_WAIT_S)
        elif kind == "session.closed":
            self.closed = True

    def _speaks(self) -> None:
        """It talks again (or gave up): the oldest finished background work is no longer due."""
        if self.result_due:
            self.result_due.pop(0)
            with self.lock:
                self.running_tools -= 1
            self._check_idle()

    async def _answer_call(self, item: dict) -> None:
        try:
            args = json.loads(item.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        digest = await self._tool(item.get("name"), args, self._turn())
        await self._send(
            {
                "type": "response.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": item.get("call_id"),
                    "output": digest,
                },
            }
        )
        await self._send({"type": "response.create"})


def make_agent(engine: str, emit: Callable[[dict], None], *, barge_in: bool = True,
               allow_network: bool = False, display=None) -> CloudAgent:  # fmt: skip
    if not available(engine):
        raise RuntimeError(f"{engine}: Azure OpenAI credentials are not set")
    cls = RealtimeAgent if engine == "gpt-realtime" else LiveAgent
    return cls(emit, barge_in=barge_in, allow_network=allow_network, display=display)


__all__ = ["ENGINES", "LiveAgent", "RealtimeAgent", "available", "make_agent"]
