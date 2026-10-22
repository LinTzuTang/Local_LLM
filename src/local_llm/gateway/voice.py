"""WebSocket /v1/voice/agent/ws: the live voice assistant (local_llm/voice_agent.py).

The browser streams 16 kHz mono PCM16 frames (~100 ms) as binary messages and sends JSON
text messages: {"type": "playing" | "played", "speech_id", "seq"[, "stopped"]} as it plays
the assistant's audio, and {"type": "stop"} to end. A worker thread runs the streaming ASR
model (gateway/live.py, VAD included) and hands every event to the VoiceAgent, which decides,
calls tools in the background, synthesises speech and emits JSON events back (the module
docstring of voice_agent lists them), ending with {"type": "done", "summary"}.

Query: asr (a streaming model), llm (an Ollama model), voice (a macOS `say` voice), rate
(words per minute, 0 = default), net (1 = allow the scenario's weather and place lookups),
barge (1 = let the user interrupt), s2tw (1 = show and use ASR text in Traditional Chinese),
engine ("cascade", the default; or a single speech model of local_llm/voice_omni.py:
"qwen3-omni" / "minicpm-o" turn-based, "minicpm-o-duplex" full-duplex; or a cloud one of
local_llm/voice_cloud.py, "gpt-realtime" / "gpt-live-1", which sends the audio to Azure; then
llm, voice and rate do not apply, net only for the cloud ones; the ASR draws the captions of
the local ones, while a cloud one gets no ASR at all: its own transcript is the caption).
The Ollama URL comes from LOCAL_LLM_BASE_URL (default 127.0.0.1:11434), never the client.
"""

import asyncio
import json
import os
import queue
import threading
import time

from fastapi import WebSocket

from .live import LiveSession
from .manager import GatewayError, Manager

STOP_GRACE_S = 0.4


def ollama_url() -> str:
    from local_llm import voice_agent

    url = os.getenv("LOCAL_LLM_BASE_URL", voice_agent.OLLAMA_URL)
    url = url.rstrip("/").removesuffix("/v1")
    return url if "11434" in url else voice_agent.OLLAMA_URL


async def voice_agent_ws(
    websocket: WebSocket,
    manager: Manager,
    asr: str,
    llm: str,
    voice: str,
    rate: int,
    net: bool,
    barge: bool,
    s2tw: bool,
    keep_alive: float | None = None,
    engine: str = "cascade",
    captions: bool = True,
) -> None:
    from local_llm import voice_agent, voice_cloud, voice_omni

    await websocket.accept()
    cloud = engine in voice_cloud.ENGINES  # hears and transcribes the audio itself
    # no ASR: a cloud model (its own transcript), or a local speech model without captions
    vad_only = cloud or (engine != "cascade" and not captions)
    try:
        if not vad_only:
            manager.require_installed(manager.spec(asr, "stream"))
    except GatewayError as exc:
        await websocket.send_json({"type": "error", "error": exc.message})
        await websocket.close()
        return
    loop = asyncio.get_running_loop()
    inbox: queue.Queue = queue.Queue()
    outbox: asyncio.Queue = asyncio.Queue()

    def emit(item) -> None:
        loop.call_soon_threadsafe(outbox.put_nowait, item)

    display = None
    if s2tw:
        import opencc

        display = opencc.OpenCC("s2tw").convert
    if engine != "cascade":
        try:
            if engine in voice_cloud.ENGINES:  # audio goes to Azure OpenAI (the page says so)
                agent = voice_cloud.make_agent(engine, emit, barge_in=barge, allow_network=net,
                                               display=display)  # fmt: skip
            else:
                agent = voice_omni.make_agent(engine, emit, barge_in=barge, display=display,
                                              captions=captions)  # fmt: skip
        except (RuntimeError, ValueError) as exc:
            await websocket.send_json({"type": "error", "error": str(exc)})
            await websocket.close()
            return
    else:
        # The constructor asks Ollama whether the model can switch thinking off: not on the loop.
        agent = await asyncio.to_thread(
            voice_agent.VoiceAgent, emit, llm=llm, voice=voice, rate=rate or None,
            allow_network=net, barge_in=barge, base_url=ollama_url(), display=display,
        )  # fmt: skip
    hears = getattr(agent, "on_audio", None)  # single speech models get the audio itself

    def listen() -> None:
        """No ASR here, only an energy VAD (the timeline, the latency figures, and for a local
        turn-based model the end of the user's turn): a cloud model captions with its own
        transcript (voice_cloud.py); a local speech model without captions shows none."""
        import numpy as np

        from local_llm import asr_streaming

        vad, t0 = asr_streaming.OnlineVad(asr_streaming.RATE), None
        try:
            emit({"type": "ready", "asr": None, "llm": agent.llm, "engine": engine,
                  "append_only": False, "load_seconds": 0.0})  # fmt: skip
            while (item := inbox.get()) is not None:
                data, arrived = item
                if t0 is None:  # the clock starts where the first frame's audio began
                    t0 = arrived - len(data) / 2 / asr_streaming.RATE
                agent.on_audio(data)
                x = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768
                for kind, seconds in vad.feed(x):
                    agent.on_asr({"type": f"speech_{kind}", "t_audio": seconds}, t0)
        except Exception as exc:
            emit({"type": "error", "error": f"{type(exc).__name__}: {exc}"})
        finally:
            time.sleep(STOP_GRACE_S)
            agent.close()
            emit(None)

    def work() -> None:
        # One thread for the ASR session: MLX arrays belong to the thread that made them.
        try:
            with manager.using(asr, "stream", keep_alive) as (backend, handle, load_seconds):
                live = LiveSession(backend.session(handle), display=display)
                agent.append_only = live.append_only
                emit({"type": "ready", "asr": asr, "llm": agent.llm, "engine": engine,
                      "append_only": live.append_only,
                      "load_seconds": round(load_seconds, 2)})  # fmt: skip
                while (item := inbox.get()) is not None:
                    if hears:
                        hears(item[0])
                    for event in live.feed(*item):
                        agent.on_asr(event, live.t0)
                for event in live.stop():
                    if event.get("type") != "done":
                        agent.on_asr(event, live.t0)
        except Exception as exc:
            emit({"type": "error", "error": exc.message if isinstance(exc, GatewayError)
                  else f"{type(exc).__name__}: {exc}"})  # fmt: skip
        finally:
            time.sleep(STOP_GRACE_S)
            agent.close()
            emit(None)

    agent.start()
    name = f"voice-{engine if vad_only else asr}"
    threading.Thread(target=listen if vad_only else work, name=name, daemon=True).start()

    async def receive() -> None:
        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("bytes"):
                    inbox.put((message["bytes"], time.perf_counter()))
                elif message.get("text"):
                    try:
                        data = json.loads(message["text"])
                    except json.JSONDecodeError:
                        continue
                    if data.get("type") == "stop":
                        break
                    agent.on_client(data)
        except Exception:  # the socket went away
            pass
        finally:
            inbox.put(None)

    receiver = asyncio.create_task(receive())
    open_socket = True
    while (event := await outbox.get()) is not None:
        if open_socket:
            try:
                await websocket.send_json(event)
            except Exception:
                open_socket = False  # keep draining so the worker can finish
    receiver.cancel()
    if open_socket:
        try:
            await websocket.close()
        except Exception:
            pass
