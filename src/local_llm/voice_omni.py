"""Single speech models as the live voice assistant: ② turn-based and ③ full-duplex.

The cascade (voice_agent.py) chains streaming ASR, a front LLM and macOS speech. Here one model
hears the user's audio and speaks back itself:

    ② turn-based (OmniAgent): the streaming ASR still runs, for the captions and its VAD; when
       the user has finished an utterance (the same early commit as the cascade), that stretch
       of microphone audio goes to the model, whose text and speech come back while it talks.
       A barge-in (the user speaking over it) stops the reply, as in the cascade.
       Engines: Qwen3-Omni 30B-A3B (qwen3_omni_server.py, its own environment) and MiniCPM-o
       4.5 (OpenBMB's llama-omni-server, simplex).
    ③ full-duplex (DuplexAgent): every second of microphone audio, speech or silence, goes to
       MiniCPM-o 4.5 in duplex mode, which decides each second whether to speak, back-channel
       or listen; the ASR only draws the captions. When it switches to listening while the
       user talks, the speech still queued in the browser is stopped.

Neither model has tools: their prompt says they are offline and must say so when asked for
live weather or shops. Both servers are started on demand (Sidecar) and stopped after
IDLE_S seconds unused, or when the other one is needed and nobody is using them (they need 20
and 8.5 GB; two conversations on different engines can run side by side).

The agents send the page the same events as the cascade (voice_agent.py's docstring), so the
browser component, the saved sessions and the History view work unchanged; "llm" is the
engine's name and "llm_ready" means the speech model has loaded.
"""

import atexit
import base64
import io
import json
import queue
import re
import shutil
import subprocess
import threading
import time
import wave
from collections.abc import Callable, Iterator
from pathlib import Path

import httpx

from .voice_agent import VoiceAgent, is_english, join_pieces, speakable

PROJECT = Path(__file__).resolve().parents[2]
RATE = 16000
WORK = PROJECT / "storage" / "voice_omni"
IDLE_S = 900
QWEN_DIR = PROJECT / "models" / "speech_extra" / "qwen3-omni-30b-4bit"
QWEN_PORT = 9070
QWEN_MLX_VLM = "mlx-vlm==0.7.9"
MINICPM_DIR = PROJECT / "models" / "minicpm-o"
MINICPM_SERVER = MINICPM_DIR / "src" / "build" / "bin" / "llama-omni-server"
MINICPM_GGUF = MINICPM_DIR / "MiniCPM-o-4_5-gguf" / "MiniCPM-o-4_5-Q4_K_M.gguf"
MINICPM_REF = (
    MINICPM_DIR
    / "src"
    / "tools"
    / "omni"
    / "assets"
    / "default_ref_audio"
    / "default_ref_audio.wav"
)
MINICPM_PORT = 9060
MINICPM_PROMPT = (
    "<|audio_end|>你的任務是用這種聲音當一個即時語音助理。請用台灣慣用的繁體中文口語回答，一到兩句、簡短自然；"
    "使用者說英文就用英文回答。你沒有連網，查不到即時天氣、預報或店家資訊；被問到時要老實說查不到，"
    "不要編造溫度、降雨機率、店名或營業時間。<|im_end|>\n<|im_start|>user\n"
)  # fmt: skip

AUDIO_ONLY = "（只有語音）"  # a duplex reply that came with no text
NO_CAPTION = "（語音，未顯示字幕）"  # a user turn when no ASR runs
# engine -> what the settings show; "kind" is the way of talking (② turn, ③ duplex)
ENGINES = {
    "qwen3-omni": {
        "kind": "turn",
        "name": "Qwen3-Omni 30B-A3B",
        "voice": "Chelsie",
        "size_gb": 20.0,
    },  # fmt: skip
    "minicpm-o": {"kind": "turn", "name": "MiniCPM-o 4.5", "voice": "MiniCPM-o", "size_gb": 8.6},
    "minicpm-o-duplex": {
        "kind": "duplex",
        "name": "MiniCPM-o 4.5",
        "voice": "MiniCPM-o",
        "size_gb": 8.6,
    },  # fmt: skip
}


def available(engine: str) -> bool:
    """Whether the engine's weights (and runtime) are on this machine."""
    if engine == "qwen3-omni":
        return (QWEN_DIR / "config.json").exists() and shutil.which("uv") is not None
    if engine in ("minicpm-o", "minicpm-o-duplex"):
        return MINICPM_SERVER.exists() and MINICPM_GGUF.exists() and MINICPM_REF.exists()
    return False


def wav_bytes(pcm: bytes, rate: int) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return out.getvalue()


def wav_duration(data: bytes) -> float:
    with wave.open(io.BytesIO(data), "rb") as w:
        return w.getnframes() / w.getframerate()


# -- the model servers ---------------------------------------------------------------------------


class Sidecar:
    """A model server process on 127.0.0.1:`port`, started on first use and stopped after
    IDLE_S seconds without use (or when another sidecar starts)."""

    running: dict[str, "Sidecar"] = {}
    lock = threading.Lock()

    def __init__(self, name: str, port: int, cmd: list[str], cwd: Path, ready_s: float = 900):
        self.name, self.port, self.cmd, self.cwd, self.ready_s = name, port, cmd, cwd, ready_s
        self.url = f"http://127.0.0.1:{port}"
        self.proc: subprocess.Popen | None = None
        self.last_used = time.monotonic()
        self.users = 0  # conversations using it now: never stopped while above 0

    def healthy(self) -> bool:
        try:
            return httpx.get(f"{self.url}/health", timeout=2).status_code == 200
        except httpx.HTTPError:
            return False

    def ensure(self) -> None:
        with Sidecar.lock:
            self.last_used = time.monotonic()
            for other in list(Sidecar.running.values()):
                if other is not self and other.users <= 0:  # only idle ones make room
                    other.stop()
            if self.proc is not None and self.proc.poll() is None and self.healthy():
                return
            if self.healthy():  # left running by an earlier gateway
                Sidecar.running[self.name] = self
                return
            WORK.mkdir(parents=True, exist_ok=True)
            log = (WORK / f"{self.name}.log").open("w")
            self.proc = subprocess.Popen(
                self.cmd, cwd=self.cwd, stdout=log, stderr=subprocess.STDOUT
            )
            Sidecar.running[self.name] = self
        deadline = time.monotonic() + self.ready_s
        while time.monotonic() < deadline:
            if self.healthy():
                return
            if self.proc.poll() is not None:
                raise RuntimeError(f"{self.name} exited; see {WORK / (self.name + '.log')}")
            time.sleep(1)
        raise RuntimeError(f"{self.name} did not start within {self.ready_s:.0f} s")

    def touch(self) -> None:
        self.last_used = time.monotonic()

    def acquire(self) -> None:
        with Sidecar.lock:
            self.users += 1
            self.last_used = time.monotonic()

    def release(self) -> None:
        with Sidecar.lock:
            self.users = max(0, self.users - 1)
            self.last_used = time.monotonic()

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None
        Sidecar.running.pop(self.name, None)


def _reaper() -> None:
    while True:
        time.sleep(60)
        with Sidecar.lock:
            for car in list(Sidecar.running.values()):
                if car.users <= 0 and time.monotonic() - car.last_used > IDLE_S:
                    car.stop()


threading.Thread(target=_reaper, name="voice-omni-reaper", daemon=True).start()
atexit.register(lambda: [car.stop() for car in list(Sidecar.running.values())])


def qwen_sidecar() -> Sidecar:
    car = Sidecar.running.get("qwen3-omni")
    if car is None:
        script = Path(__file__).with_name("qwen3_omni_server.py")
        cmd = [shutil.which("uv") or "uv", "run", "--no-project", "--with", QWEN_MLX_VLM, "python",
               str(script), "--model", str(QWEN_DIR), "--port", str(QWEN_PORT)]  # fmt: skip
        car = Sidecar("qwen3-omni", QWEN_PORT, cmd, PROJECT)
    return car


def minicpm_sidecar() -> Sidecar:
    car = Sidecar.running.get("minicpm-o")
    if car is None:
        cmd = [str(MINICPM_SERVER), "--host", "127.0.0.1", "--port", str(MINICPM_PORT),
               "--model", str(MINICPM_GGUF), "-ngl", "99", "--ctx-size", "8192",
               "--repeat-penalty", "1.05", "--temp", "0.7"]  # fmt: skip
        car = Sidecar("minicpm-o", MINICPM_PORT, cmd, MINICPM_SERVER.parents[2], ready_s=300)
    return car


# -- the models ----------------------------------------------------------------------------------


class QwenOmni:
    """Turn-based Qwen3-Omni through qwen3_omni_server.py: ("text", str) then ("wav", bytes)."""

    def __init__(self, sidecar: Callable[[], Sidecar] = qwen_sidecar):
        self.sidecar = sidecar()
        self.lang = "zh"
        self.holding = False

    def start(self, lang: str = "zh") -> None:
        if not self.holding:  # this conversation keeps the server running until close()
            self.sidecar.acquire()
            self.holding = True
        self.sidecar.ensure()
        self.lang = lang
        httpx.post(f"{self.sidecar.url}/reset", json={"lang": lang}, timeout=30).raise_for_status()

    def reply(self, pcm: bytes, lang: str) -> Iterator[tuple[str, object]]:
        self.sidecar.touch()
        body = {"audio": base64.b64encode(pcm).decode(), "lang": lang}
        with httpx.stream("POST", f"{self.sidecar.url}/reply", json=body, timeout=180) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                event = json.loads(line) if line.strip() else {}
                if event.get("type") == "text":
                    yield "text", event["text"]
                elif event.get("type") == "audio":
                    yield "wav", wav_bytes(base64.b64decode(event["pcm"]), 24000)
                elif event.get("type") == "done":
                    return

    def close(self) -> None:
        if self.holding:
            self.sidecar.release()
            self.holding = False


class MiniCPM:
    """MiniCPM-o 4.5 through llama-omni-server (simplex for turns, duplex for ticks).

    This build only applies `assistant_prompt` with media_type 2 and does not prefill the
    reference voice itself: prefill cnt 0 with it builds the system prompt. Audio goes in as
    1-second 16 kHz WAV files; the speech comes back as WAV files the server writes under
    out_dir (simplex: round_000/tts_wav, every turn, since the server captures round_idx by
    reference; duplex: tts_wav, names reused)."""

    def __init__(self, duplex: bool = False, sidecar: Callable[[], Sidecar] = minicpm_sidecar):
        self.duplex, self.sidecar = duplex, sidecar()
        self.cnt, self.dir = 1, WORK / "minicpm-o" / str(int(time.time() * 1000))
        self.seen: dict[str, tuple[int, int]] = {}

    @property
    def url(self) -> str:
        return self.sidecar.url

    def start(self, lang: str = "zh") -> None:
        self.sidecar.acquire()  # held until close()
        self.sidecar.ensure()
        shutil.rmtree(self.dir, ignore_errors=True)
        (self.dir / "in").mkdir(parents=True)
        (self.dir / "out").mkdir()
        body = {"media_type": 2, "use_tts": True, "duplex_mode": self.duplex, "tts_gpu_layers": 99,
                "token2wav_device": "gpu:0", "output_dir": str(self.dir / "out"),
                "assistant_prompt": MINICPM_PROMPT}  # fmt: skip
        httpx.post(f"{self.url}/v1/stream/omni_init", json=body, timeout=600).raise_for_status()
        self._prefill(MINICPM_REF, 0)

    def _prefill(self, path: Path, cnt: int) -> None:
        r = httpx.post(
            f"{self.url}/v1/stream/prefill",
            timeout=120,
            json={"audio_path_prefix": str(path), "img_path_prefix": "", "cnt": cnt},
        )
        r.raise_for_status()

    def _chunk(self, pcm: bytes) -> Path:
        path = self.dir / "in" / f"chunk_{self.cnt:05d}.wav"
        path.write_bytes(wav_bytes(pcm.ljust(RATE * 2, b"\0")[: RATE * 2], RATE))
        return path

    def _decode(self, out: queue.Queue | None = None) -> tuple[str, bool]:
        text, listen = "", False
        body = {"debug_dir": str(self.dir / "out"), "stream": True, "round_idx": self.cnt}
        with httpx.stream("POST", f"{self.url}/v1/stream/decode", json=body, timeout=120) as r:
            for line in r.iter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                piece = event.get("content") or ""
                text += piece
                listen = listen or bool(event.get("is_listen"))
                if piece and out is not None:
                    out.put(("text", piece))
                if event.get("stop"):
                    break
        return text, listen and not text.strip()

    def _new_wavs(self, folder: Path, settle: bool) -> list[bytes]:
        """WAV files written (or rewritten) since last time, in order; a file counts once its
        size has not changed for 50 ms (or right away when `settle` is off)."""
        out = []
        files = sorted(folder.glob("wav_*.wav"), key=lambda p: int(re.sub(r"\D", "", p.stem) or 0))
        for p in files:
            try:
                st = p.stat()
            except FileNotFoundError:
                continue
            key = (st.st_mtime_ns, st.st_size)
            if self.seen.get(str(p)) == key or st.st_size <= 44:
                continue
            if settle:
                time.sleep(0.05)
                if p.stat().st_size != st.st_size:
                    continue
            self.seen[str(p)] = key
            out.append(p.read_bytes())
        return out

    def reply(self, pcm: bytes, lang: str) -> Iterator[tuple[str, object]]:
        """Simplex: the utterance as 1-second chunks, then decode; text and WAVs as they come."""
        self.sidecar.touch()
        for pos in range(0, len(pcm), RATE * 2):
            self._prefill(self._chunk(pcm[pos : pos + RATE * 2]), self.cnt)
            self.cnt += 1
        tts = self.dir / "out" / "round_000" / "tts_wav"
        shutil.rmtree(tts, ignore_errors=True)
        self.seen = {}
        events: queue.Queue = queue.Queue()
        done = threading.Event()

        def decode() -> None:
            try:
                self._decode(events)
            finally:
                done.set()

        threading.Thread(target=decode, name="minicpm-decode", daemon=True).start()
        deadline = None
        while True:
            while not events.empty():
                yield events.get()
            for data in self._new_wavs(tts, settle=True):
                yield "wav", data
            finished = done.is_set() and (tts / "generation_done.flag").exists()
            if done.is_set() and deadline is None:
                deadline = time.monotonic() + 60
            if finished or (deadline and time.monotonic() > deadline):
                for data in self._new_wavs(tts, settle=False):
                    yield "wav", data
                while not events.empty():
                    yield events.get()
                return
            time.sleep(0.05)

    def tick(self, pcm: bytes) -> dict:
        """Duplex: one second of audio in; what it said in that second, whether it listened,
        and the speech written since the last tick."""
        self.sidecar.touch()
        self._prefill(self._chunk(pcm), self.cnt)
        text, listen = self._decode()
        self.cnt += 1
        return {
            "text": text,
            "listen": listen,
            "wavs": self._new_wavs(self.dir / "out" / "tts_wav", settle=True),
        }

    def close(self) -> None:
        self.sidecar.release()
        shutil.rmtree(self.dir, ignore_errors=True)


def make_model(engine: str):
    if engine == "qwen3-omni":
        return QwenOmni()
    if engine == "minicpm-o":
        return MiniCPM(duplex=False)
    if engine == "minicpm-o-duplex":
        return MiniCPM(duplex=True)
    raise ValueError(f"unknown voice engine {engine!r}")


# -- the agents ----------------------------------------------------------------------------------


class OmniAgent(VoiceAgent):
    """② turn-based: the cascade's turn-taking, with one speech model instead of LLM + TTS."""

    MARGIN_S = 0.3  # audio kept before the VAD's speech start and after its end

    def __init__(
        self,
        emit: Callable[[dict], None],
        model,
        *,
        engine: str,
        barge_in: bool = True,
        display: Callable[[str], str] | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ):
        info = ENGINES.get(engine, {})
        super().__init__(emit, llm=info.get("name", engine), voice=info.get("voice", engine),
                         allow_network=False, barge_in=barge_in, think=None, display=display,
                         clock=clock)  # fmt: skip
        self.engine, self.model = engine, model
        self.pcm = bytearray()  # everything the microphone sent, 16 kHz PCM16
        self.audio_offset = 0.0  # session clock - audio time
        self.ready = threading.Event()
        self.lang = "zh"
        self.cursor = 0  # bytes of microphone audio given to the duplex model
        # Without captions no ASR runs (these models do not say what they heard): a turn
        # starts at the VAD's end of speech, with no text.
        self.captions = True

    def on_audio(self, data: bytes) -> None:
        with self.lock:
            self.pcm += data

    def on_asr(self, event: dict, audio_t0: float | None = None) -> None:
        if audio_t0 is not None:
            self.audio_offset = audio_t0 - self.t0
        super().on_asr(event, audio_t0)
        if not self.captions and event.get("type") == "speech_end":
            self._user(NO_CAPTION)

    def _warm(self) -> None:
        started = self.clock()
        try:
            self.model.start("zh")
            self.llm_load_s = round(self.clock() - started, 2)
            self.emit({"type": "llm_ready", "llm": self.llm, "seconds": self.llm_load_s})
        except Exception as exc:
            self.emit({"type": "error", "error": f"{self.llm}: {exc}"})
        finally:
            with self.lock:  # what was said while the model loaded is not replayed (duplex)
                self.cursor = len(self.pcm) - len(self.pcm) % 2
            self.ready.set()

    def utterance(self, turn: dict) -> bytes:
        """The turn's stretch of microphone audio (VAD speech start/end, with a margin)."""
        start = (turn.get("speech_start") or 0.0) - self.audio_offset - self.MARGIN_S
        end = (turn.get("speech_end") or self.now()) - self.audio_offset + self.MARGIN_S
        with self.lock:
            a = max(0, int(start * RATE)) * 2
            b = min(len(self.pcm) // 2, int(end * RATE)) * 2
            return bytes(self.pcm[a:b]) if b > a else bytes(self.pcm[-RATE * 2 :])

    def send_wav(self, speech: dict, data: bytes, text: str = "") -> None:
        duration = wav_duration(data)
        with self.lock:
            seq = speech["seq"]
            speech["seq"] += 1
            self.unplayed.add((speech["speech_id"], seq))
            turn = self.turns[speech["turn"]] if speech["turn"] is not None else None
            if turn is not None and turn.get("first_audio_sent") is None:
                turn["first_audio_sent"] = self.now()
        self.emit(
            {
                "type": "audio",
                "speech_id": speech["speech_id"],
                "seq": seq,
                "turn": speech["turn"],
                "kind": speech["kind"],
                "text": text,
                "duration": round(duration, 3),
                "wav": base64.b64encode(data).decode(),
            }
        )

    def _respond(self, turn: dict) -> None:
        self.ready.wait(900)
        epoch = self.epoch
        lang = "en" if is_english(turn["user"]) else "zh"
        if turn["turn"] == 0 and lang != self.lang and isinstance(self.model, QwenOmni):
            self.model.start(lang)  # Qwen3-Omni's system prompt follows the first turn's language
        self.lang = lang
        audio = self.utterance(turn)
        started = self.clock()
        turn["llm_start"] = self.now()
        self.emit({"type": "llm", "turn": turn["turn"], "phase": "start", "what": "reply"})
        speech = self._new_speech(turn, "reply")
        replies = self.model.reply(audio, lang)
        try:
            for kind, value in replies:
                if epoch != self.epoch:
                    break
                if turn.get("llm_first_s") is None:
                    turn["llm_first_s"] = round(self.clock() - started, 3)
                if kind == "text":
                    value = self.display(value)  # MiniCPM-o writes Simplified: shown in Traditional
                    speech["text"] = join_pieces(speech["text"], value) if speech["text"] else value
                    self.emit(
                        {
                            "type": "assistant",
                            "speech_id": speech["speech_id"],
                            "turn": turn["turn"],
                            "kind": "reply",
                            "text": speakable(speech["text"]),
                            "final": False,
                        }
                    )
                elif kind == "wav":
                    self.send_wav(speech, value)
        finally:
            replies.close()  # a barge-in closes the stream, which stops the speech
        interrupted = epoch != self.epoch
        turn["llm_s"] = round(self.clock() - started, 3)
        turn["llm_end"] = self.now()
        self.emit({"type": "llm", "turn": turn["turn"], "phase": "end", "what": "reply"})
        text = speakable(speech["text"])
        turn["decision"] = (
            "interrupted" if interrupted else ("ask" if text.endswith(("？", "?")) else "answer")
        )
        turn["assistant"] = text
        self._finish_speech(speech)
        self._turn_metrics(turn)

    def close(self) -> dict:
        summary = super().close()
        try:
            self.model.close()
        except Exception:
            pass
        return summary

    def summary(self) -> dict:
        return {
            **super().summary(),
            "engine": self.engine,
            "kind": ENGINES.get(self.engine, {}).get("kind") or getattr(self, "engine_kind", None),
        }


def join_wavs(parts: list[bytes]) -> bytes:
    """Speech files of one reply as one WAV (they share the model's sample rate)."""
    pcm, rate = bytearray(), RATE
    for data in parts:
        with wave.open(io.BytesIO(data)) as w:
            rate = w.getframerate()
            pcm += w.readframes(w.getnframes())
    return wav_bytes(bytes(pcm), rate)


class DuplexAgent(OmniAgent):
    """③ full-duplex: the model hears every second and decides itself when to talk.

    The server's text of a reply often lacks its beginning (its duplex decode returns only the
    last piece it wrote each second, while the speech is complete: "即時天氣。" for a reply
    that says "抱歉，我查不到即時天氣。"). With `transcribe` (a whole-clip ASR model, given by
    the gateway) the shown text of each finished reply is what its speech says."""

    TAIL_TICKS = 2

    def __init__(self, emit: Callable[[dict], None], model, *,
                 transcribe: Callable[[bytes], str] | None = None, **kwargs):  # fmt: skip
        kwargs["barge_in"] = False  # the model decides; see _tick
        super().__init__(emit, model, **kwargs)
        self.transcribe = transcribe
        self.quiet = 0  # listening seconds since its last speech
        self.dropping = 0  # seconds of a cut-off reply's late speech still to drop
        self.speech: dict | None = None
        self.ticker = threading.Thread(target=self._ticks, name="voice-duplex", daemon=True)

    def start(self, warm: bool = True) -> None:
        super().start(warm)
        self.ticker.start()

    def _user(self, text: str, early: bool = False) -> None:
        """Captions only: the turn is recorded for the page, the model already heard it."""
        with self.lock:
            span = self.vad[-1] if self.vad else [None, None]
            turn = {
                "turn": len(self.turns),
                "user": text,
                "speech_start": span[0],
                "speech_end": span[1] or self.now(),
                "asr_final": self.now(),
                "speech_ids": [],
                "tool_calls": [],
                "assistant": "",
                "decision": "answer",
                "early_commit": early,
            }
            self.turns.append(turn)
            speech = self.speech  # it began answering before the caption came: this turn's
            if speech is not None and speech["turn"] is None:
                speech["turn"] = turn["turn"]
                turn["speech_ids"].append(speech["speech_id"])
                turn["llm_start"] = self.now()
        self.emit(
            {
                "type": "user_final",
                "turn": turn["turn"],
                "text": text,
                "speech_end": turn["speech_end"],
            }
        )

    def _ticks(self) -> None:
        self.ready.wait(900)
        step = RATE * 2
        while not self.closed:
            with self.lock:
                have = len(self.pcm) - self.cursor
            if have < step:
                time.sleep(0.05)
                continue
            with self.lock:
                chunk = bytes(self.pcm[self.cursor : self.cursor + step])
                self.cursor += step
            try:
                started = time.perf_counter()
                out = self.model.tick(chunk)
                took = time.perf_counter() - started
                self._log(f"{self.now():8.2f} tick {took:.2f}s listen={out['listen']} "
                          f"wavs={len(out['wavs'])} {out['text']!r}")  # fmt: skip
                self._tick(out)
            except Exception as exc:
                self.emit({"type": "error", "error": f"{self.llm}: {type(exc).__name__}: {exc}"})
                time.sleep(1)

    def _log(self, line: str) -> None:
        """One line per second of the duplex loop, in storage/voice_omni/minicpm-o-duplex.log."""
        try:
            with (WORK / "minicpm-o-duplex.log").open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass

    def _tick(self, out: dict) -> None:
        """One second's output. Its speech files lag its text by up to two seconds, so a reply
        ends after TAIL_TICKS listening seconds without new speech, or at once when it gives
        way to the user (then the rest of that reply is dropped)."""
        turn = self.turns[-1] if self.turns else None
        text, wavs = out["text"].strip(), out["wavs"]
        if self.dropping:
            self.dropping = 0 if text else self.dropping - 1
            wavs = [] if not text else wavs
        if text or wavs:
            self.quiet = 0
            if self.speech is None:
                self.speech = self._new_speech(turn, "reply")
                if turn is not None and turn.get("llm_start") is None:
                    turn["llm_start"] = self.now()
            speech = self.speech
            if text:
                piece = self.display(text)
                speech["text"] = join_pieces(speech["text"], piece) if speech["text"] else piece
                self.emit({"type": "assistant", "speech_id": speech["speech_id"],
                           "turn": speech["turn"], "kind": "reply",
                           "text": speakable(speech["text"]), "final": False})  # fmt: skip
            for data in wavs:
                self.send_wav(speech, data)
                speech.setdefault("wavs", []).append(data)
        if self.speech is None or not out["listen"] or text:
            return
        speech = self.speech
        if bool(self.vad) and self.vad[-1][1] is None:  # it gave way to the user: stop it
            with self.lock:
                stopped = sorted({sid for sid, _ in self.unplayed if sid == speech["speech_id"]})
                self.unplayed = {u for u in self.unplayed if u[0] != speech["speech_id"]}
                speech["interrupted"] = bool(stopped)
            if stopped:
                self.emit({"type": "barge_in", "speech_ids": stopped, "at": self.now()})
            self.dropping = self.TAIL_TICKS
        else:
            self.quiet += 0 if wavs else 1
            if self.quiet < self.TAIL_TICKS:
                return
        self.speech, self.quiet = None, 0
        if self.transcribe is not None and speech.get("wavs"):
            try:  # the caption from what it said, not from its (partial) text
                heard = self.transcribe(join_wavs(speech["wavs"])).strip()
            except Exception:
                heard = ""
            if heard:
                speech["model_text"], speech["text"] = speech["text"], self.display(heard)
        if not speech["text"].strip() and speech["seq"]:  # it spoke without writing anything
            speech["text"] = AUDIO_ONLY
        if speech["turn"] is not None:
            t = self.turns[speech["turn"]]
            t["assistant"] = speakable(join_pieces(t.get("assistant") or "", speech["text"]))
            t["llm_end"] = self.now()
        self._finish_speech(speech)
        if speech["turn"] is not None:
            self._turn_metrics(self.turns[speech["turn"]])
        self._check_idle()

    def close(self) -> dict:
        if self.speech is not None:
            self._finish_speech(self.speech)
            self.speech = None
        return super().close()


def make_agent(engine: str, emit: Callable[[dict], None], *, barge_in: bool = True,
               display: Callable[[str], str] | None = None, captions: bool = True,
               transcribe: Callable[[bytes], str] | None = None) -> OmniAgent:  # fmt: skip
    """`transcribe` (WAV bytes -> text): the duplex agent captions its replies with it."""
    if not available(engine):
        raise RuntimeError(f"{engine} is not installed on this machine")
    model = make_model(engine)
    if ENGINES[engine]["kind"] == "duplex":
        agent = DuplexAgent(emit, model, engine=engine, barge_in=barge_in, display=display,
                            transcribe=transcribe)  # fmt: skip
    else:
        agent = OmniAgent(emit, model, engine=engine, barge_in=barge_in, display=display)
    agent.captions = captions
    return agent


__all__ = ["ENGINES", "DuplexAgent", "OmniAgent", "available", "make_agent"]
