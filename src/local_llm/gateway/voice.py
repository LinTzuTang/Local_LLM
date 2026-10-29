"""WebSocket /v1/voice/agent/ws: the live voice assistant (local_llm/voice_agent.py).

The browser streams 16 kHz mono PCM16 frames (~100 ms) as binary messages and sends JSON
text messages: {"type": "playing" | "played", "speech_id", "seq"[, "stopped"]} as it plays
the assistant's audio, and {"type": "stop"} to end. A worker thread runs the streaming ASR
model (gateway/live.py, VAD included) and hands every event to the VoiceAgent, which decides,
calls tools in the background, synthesises speech and emits JSON events back (the module
docstring of voice_agent lists them), ending with {"type": "done", "summary"}.

Query: asr (a streaming model, or a whole-clip one: then an energy VAD cuts each utterance at
its pause and the model transcribes it whole once you stop, "pause" mode), llm (an Ollama
model), voice (a macOS `say` voice), rate (words per minute, 0 = default), net (1 = allow the
scenario's weather and place lookups), barge (1 = let the user interrupt), s2tw (1 = show
and use ASR text in Traditional Chinese), engine ("cascade", the default; or a single speech
model of local_llm/voice_omni.py: "qwen3-omni" / "minicpm-o" turn-based, "minicpm-o-duplex"
full-duplex; or a cloud one of local_llm/voice_cloud.py, "gpt-realtime" / "gpt-live-1",
which sends the audio to Azure; then llm, voice and rate do not apply, net only for the
cloud ones; the ASR draws the captions of the local ones, while a cloud one gets no ASR at
all: its own transcript is the caption). hear (1 = the cascade's front model hears the
user's audio instead of the ASR text, for the audio LLMs of voice_agent.AUDIO_LLMS; the ASR
then only draws the captions).
The Ollama URL comes from LOCAL_LLM_BASE_URL (default 127.0.0.1:11434), never the client.
"""

import asyncio
import json
import os
import queue
import threading
import time
from pathlib import Path

from fastapi import WebSocket

from .live import LiveSession
from .manager import GatewayError, Manager

STOP_GRACE_S = 0.4
PAUSE_PAD_S = 0.3  # audio kept before and after an utterance for a whole-clip model
PAUSE_GAP_S = 0.8  # quiet that ends an utterance for a whole-clip model


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
    hear: bool = False,
) -> None:
    from local_llm import voice_agent, voice_cloud, voice_omni

    await websocket.accept()
    cloud = engine in voice_cloud.ENGINES  # hears and transcribes the audio itself
    # no ASR: a cloud model (its own transcript), or a local speech model without captions
    vad_only = cloud or (engine != "cascade" and not captions)
    pause = False  # a whole-clip ASR model, transcribing each utterance after its pause
    try:
        if not vad_only:
            spec = manager.spec(asr)
            pause = spec.task == "asr"
            manager.require_installed(manager.spec(asr, "asr" if pause else "stream"))
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
            if engine in voice_cloud.ENGINES:  # audio goes to the cloud (the page says so)
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
            hears_audio=hear and llm in voice_agent.AUDIO_LLMS,
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

    def pause_asr() -> None:
        """A whole-clip model (not streaming): an energy VAD finds each utterance and, once it
        has ended (PAUSE_GAP_S of quiet), the model transcribes the whole of it, padded by
        PAUSE_PAD_S; the cascade then answers the final text. Nothing is shown while you
        talk; in the evaluation this was more accurate than true streaming (Qwen3-ASR-1.7B on
        the National Day speech: 2.9% vs Voxtral's 3.7%)."""
        import tempfile
        import wave

        import numpy as np

        from local_llm import asr_streaming

        rate = asr_streaming.RATE
        # 0.8 s of quiet ends an utterance, as the streaming models' endpoint does: at 0.6 s a
        # comma pause (「好，那就第一家吧，謝謝」 has 0.66 s) split one line into two turns.
        vad = asr_streaming.OnlineVad(rate, min_gap=PAUSE_GAP_S)
        t0, audio = None, bytearray()
        start, base = None, 0  # utterance start (s), seconds of audio dropped from `audio`
        jobs: queue.Queue = queue.Queue()

        def transcribe(pcm: bytes) -> str:
            with tempfile.NamedTemporaryFile(suffix=".wav") as f:
                with wave.open(f.name, "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(rate)
                    w.writeframes(pcm)
                out = manager.run(asr, "asr", keep_alive, audio=Path(f.name))
            return (out.get("text") or "").strip()

        def transcriber() -> None:  # one at a time, off the listening loop
            while (pcm := jobs.get()) is not None:
                try:
                    text = transcribe(pcm)
                except Exception as exc:
                    emit({"type": "error", "error": f"{asr}: {exc}"})
                    continue
                agent.on_asr({"type": "final", "text": text}, t0)

        worker = threading.Thread(target=transcriber, name=f"voice-{asr}-asr", daemon=True)
        try:
            started = time.perf_counter()
            transcribe(bytes(int(0.5 * rate) * 2))  # load and warm the model
            emit({"type": "ready", "asr": asr, "llm": agent.llm, "engine": engine,
                  "append_only": False, "pause": True,
                  "load_seconds": round(time.perf_counter() - started, 2)})  # fmt: skip
            worker.start()
            while (item := inbox.get()) is not None:
                data, arrived = item
                if t0 is None:
                    t0 = arrived - len(data) / 2 / rate
                audio += data
                if hears:
                    hears(data)
                x = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768
                for kind, seconds in vad.feed(x):
                    agent.on_asr({"type": f"speech_{kind}", "t_audio": seconds}, t0)
                    if kind == "start":
                        start = seconds
                    elif start is not None:
                        a = max(int((start - PAUSE_PAD_S - base) * rate), 0) * 2
                        b = int((seconds + PAUSE_PAD_S - base) * rate) * 2
                        jobs.put(bytes(audio[a:b]))
                        agent.on_asr({"type": "partial", "text": "…"}, t0)  # transcribing
                        start = None
                if start is None and len(audio) > 30 * rate * 2:  # keep the last 10 s
                    drop = len(audio) - 10 * rate * 2
                    del audio[:drop]
                    base += drop / 2 / rate
        except Exception as exc:
            emit({"type": "error", "error": exc.message if isinstance(exc, GatewayError)
                  else f"{type(exc).__name__}: {exc}"})  # fmt: skip
        finally:
            jobs.put(None)
            if worker.is_alive():
                worker.join(timeout=30)
            time.sleep(STOP_GRACE_S)
            agent.close()
            emit(None)

    def sim_next(data: dict) -> None:
        """The simulated user's next line (local_llm/user_sim.py), spoken by the chosen voice:
        {"type": "sim_line", "text", "done", "pcm" (base64 16 kHz PCM16), "seconds"}."""
        import base64
        import wave

        from local_llm import tts, user_sim

        try:
            history = []
            for turn in list(agent.turns):
                said = " ".join(x for x in (turn.get("assistant"), turn.get("answer")) if x)
                history.append((turn.get("user") or "", said.strip()))
            english = data.get("lang") == "en"
            line = user_sim.next_line(str(data.get("goal") or ""), history, english=english,
                                      base_url=ollama_url())  # fmt: skip
            pcm = ""
            if line["text"]:
                path = tts.synthesize(line["text"], str(data.get("voice") or "Tingting"), None,
                                      sample_rate=16000)  # fmt: skip
                with wave.open(str(path), "rb") as w:
                    pcm = base64.b64encode(w.readframes(w.getnframes())).decode()
            emit({"type": "sim_line", **line, "pcm": pcm})
        except Exception as exc:
            emit({"type": "sim_line", "text": "", "done": True, "pcm": "",
                  "error": f"{type(exc).__name__}: {exc}"})  # fmt: skip

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
    target = listen if vad_only else pause_asr if pause else work
    threading.Thread(target=target, name=name, daemon=True).start()

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
                    if data.get("type") == "sim_next":
                        threading.Thread(target=sim_next, args=(data,), name="voice-sim",
                                         daemon=True).start()  # fmt: skip
                        continue
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
