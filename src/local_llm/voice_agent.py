"""A live voice assistant in the style of OpenAI's GPT-Live, as a local cascade.

GPT-Live keeps a fast voice layer talking while slower work (tool calls, deeper reasoning)
runs in the background, speaks the result when it is ready, and lets the user interrupt.
Here the same turn-taking runs on this machine with separate parts:

    microphone PCM -> streaming ASR (gateway/live.py) -> final utterance
      -> front LLM (Ollama, thinking off, tool definitions) decides:
           (a) answer directly, (b) ask a clarifying question, or (c) call a tool
      -> (c): speak a short filler at once ("好，我幫你查一下內湖的天氣"), run the tool in a
         background thread while listening on, then let the LLM turn the result into a
         short spoken answer
      -> macOS `say` per sentence -> WAV to the browser, which plays it
    user starts speaking while the assistant talks (VAD speech_start) -> "barge_in": the
    browser stops playback at once and the reply being generated is cut off.

`VoiceAgent` is the engine, independent of the transport: the gateway's WebSocket endpoint
(gateway/voice.py) feeds it ASR events (`on_asr`) and the browser's playback reports
(`on_client`), and sends out whatever it `emit`s. The LLM, the tools and the speech
synthesiser are injectable, so tests run it without Ollama, the network or `say`.

Events sent to the page (all times are seconds on the session clock):
    user_partial / user_final {turn, text}, vad {state}, assistant {turn, speech_id, kind,
    text, final}, audio {speech_id, seq, wav (base64), duration, text}, barge_in
    {speech_ids}, tool_start {call_id, turn, name, args}, tool_end {call_id, ok, summary,
    seconds, result}, turn {turn metrics}, idle, llm_ready, error, done {summary}.
Messages from the page: {"type": "playing" | "played", "speech_id", "seq"[, "stopped"]}.
"""

import base64
import json
import queue
import re
import threading
import time
from collections.abc import Callable, Iterator
from datetime import datetime

import httpx

from . import geo_tools
from .speech_live import speakable

OLLAMA_URL = "http://127.0.0.1:11434"
# Chosen by measuring the three demo turns (16 runs per model, thinking off): Gemma 4 E4B,
# Qwen3.6 35B-A3B, Gemma 4 12B and Qwen3.8 27B made every tool decision right; E4B decided
# fastest (median 0.10 s, warm) and needs the least memory. qwen3:1.7b missed 5 of 18.
DEFAULT_LLM = "gemma4:e4b"
CANDIDATE_LLMS = ("gemma4:e4b", "qwen3.6:35b-a3b-coding", "gemma4:12b", "qwen3.8:27b",
                  "qwen3:1.7b")  # fmt: skip
DEFAULT_VOICE = "Meijia"
ALWAYS_THINKS = {"qwen3:30b-a3b"}
CALL_LIMIT = 6  # lookups per tool per session when the network switch is on
HISTORY_MESSAGES = 10
MAX_TOKENS = 220
KEEP_ALIVE = "30m"
COMMIT_DELAY_S = 0.35  # append-only ASR (Voxtral): wait this long after speech ends
WEEKDAYS = "一二三四五六日"

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "查詢某個地點現在的天氣與今天的降雨機率（Open-Meteo）。只有在知道使用者在哪裡時"
                "才呼叫；不知道地點就先問使用者。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "地名，用使用者說的中文名稱，例如「內湖」「台北市信義區」",
                    }
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_places",
            "description": (
                "用 OpenStreetMap 找某個地點附近約 1.5 公里內的店家（早午餐、咖啡、餐廳等），"
                "回傳店名、距離、營業時間與地址。"
                "使用者說「這附近」時，location 用使用者目前的位置。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "地名，例如「內湖」"},
                    "query": {
                        "type": "string",
                        "description": "要找的店家類型，例如「早午餐」「咖啡」「拉麵」",
                    },
                },
                "required": ["location", "query"],
            },
        },
    },
]
TOOL_NAMES = tuple(t["function"]["name"] for t in TOOL_SPECS)


def system_prompt(memory: dict, allow_network: bool, now: datetime | None = None) -> str:
    """The front agent's instructions: spoken Taiwanese Mandarin, when to ask and when to
    call a tool. `memory` holds what the conversation established (the user's location)."""
    now = now or datetime.now()
    lines = [
        "你是在使用者電腦上執行的即時語音助理，你說的每句話都會被語音合成念出來。",
        "規則：",
        "1. 用台灣慣用的繁體中文口語回答，一到兩句、不超過 40 個字；使用者說英文就用英文回答。",
        "2. 不要用 Markdown、條列、表情符號或網址。",
    ]
    if allow_network:
        lines += [
            "3. 使用者問天氣但你不知道他在哪裡時，不要猜地點，只要簡短反問：「你現在在哪裡？」",
            "4. 知道地點後要查天氣就呼叫 get_weather；問附近的店家、早午餐、咖啡或餐廳就呼叫"
            " search_places。「這附近」「這裡」指使用者目前的位置。",
            "5. 需要查詢時直接呼叫工具，不要只回答要去查。"
            "沒有查詢結果就不要說出任何天氣數字或店名。",
            "6. 其他問題直接簡短回答。",
        ]
    else:
        lines += [
            "3. 這個場景目前沒有開啟連網查詢，你查不到即時天氣或店家；被問到時，簡短說明需要"
            "在設定打開「允許連網查詢」，不要編造。",
            "4. 其他問題直接簡短回答。",
        ]
    lines.append(f"現在時間：{now:%Y-%m-%d} 星期{WEEKDAYS[now.weekday()]} {now:%H:%M}。")
    if memory.get("location"):
        lines.append(f"使用者目前的位置：{memory['location']}（使用者先前說的）。")
    return "\n".join(lines)


def compose_prompt() -> str:
    """Instructions for turning a tool result into the spoken answer."""
    return "\n".join([
        "你是即時語音助理，你的回答會被念出來。根據系統提供的查詢結果回答使用者剛才的問題。",
        "用台灣慣用的繁體中文口語，兩到三句、不超過 80 個字，不用 Markdown、條列、表情符號或網址。",
        "天氣：說現在的溫度和天氣、今天的降雨機率，再給一句簡短建議（例如要不要帶傘）。",
        "店家：挑兩到三家說店名和大約距離；只能說結果裡有的店，不要編造；如果結果說明找到的不是"
        "早午餐店而是附近其他店，要老實說。距離用「大約幾百公尺」的說法。",
        "查詢失敗時，簡短道歉並說明查不到。",
    ])  # fmt: skip


# -- the parts it talks to (replaceable in tests) ---------------------------------------------


def ollama_chat(base_url: str, model: str, messages: list[dict], tools: list | None = None,
                think: bool | None = False, max_tokens: int = MAX_TOKENS,
                cancel: threading.Event | None = None) -> Iterator[dict]:  # fmt: skip
    """Stream an Ollama /api/chat call: {"content": text} pieces, {"tool_calls": [...]},
    then {"done": True, "load_s", "prompt_tokens", "output_tokens"}. Closing the iterator
    (or setting `cancel`) closes the connection, which stops the generation."""
    body = {
        "model": model,
        "messages": messages,
        "stream": True,
        "keep_alive": KEEP_ALIVE,
        "options": {"temperature": 0.2, "num_predict": max_tokens, "num_ctx": 4096},
    }
    if tools:
        body["tools"] = tools
    if think is not None:
        body["think"] = think
    with httpx.Client(timeout=120, trust_env=False) as client:
        with client.stream("POST", base_url.rstrip("/") + "/api/chat", json=body) as response:
            if response.is_error:
                response.read()
                raise RuntimeError(f"Ollama HTTP {response.status_code}: {response.text[:300]}")
            for line in response.iter_lines():
                if cancel is not None and cancel.is_set():
                    return
                if not line:
                    continue
                chunk = json.loads(line)
                if "error" in chunk:
                    raise RuntimeError(chunk["error"])
                message = chunk.get("message") or {}
                if message.get("content"):
                    yield {"content": message["content"]}
                if message.get("tool_calls"):
                    yield {"tool_calls": message["tool_calls"]}
                if chunk.get("done"):
                    yield {"done": True, "load_s": (chunk.get("load_duration") or 0) / 1e9,
                           "prompt_tokens": chunk.get("prompt_eval_count"),
                           "output_tokens": chunk.get("eval_count")}  # fmt: skip


def think_value(model: str, base_url: str = OLLAMA_URL) -> bool | None:
    """The `think` value for a spoken reply: off wherever the model allows it.

    Newer Ollama models (Qwen3.6, Qwen3.8, Gemma 4) switch thinking off with think=false
    even though their templates do not show it (ollama_admin.thinking_mode reports them as
    always thinking); only the Thinking-2507 build behind qwen3:30b-a3b really cannot, so it
    gets True and its reasoning is kept out of the speech. None = no thinking support."""
    from . import ollama_admin

    try:
        capabilities = ollama_admin.show(model, base_url).get("capabilities", [])
    except httpx.HTTPError:
        return None
    if "thinking" not in capabilities:
        return None
    return model in ALWAYS_THINKS


def run_tool(name: str, args: dict) -> tuple[dict, str]:
    """(result, one-line summary) of a lookup; failures come back as {"error": ...}."""
    try:
        if name == "get_weather":
            result = geo_tools.weather(str(args.get("location") or ""))
            return result, geo_tools.weather_summary(result)
        if name == "search_places":
            query = str(args.get("query") or "早午餐")
            result = geo_tools.nearby(str(args.get("location") or ""), query)
            return result, geo_tools.nearby_summary(result)
        return {"error": f"unknown tool {name}"}, f"unknown tool {name}"
    except (geo_tools.GeoError, httpx.HTTPError, KeyError, ValueError) as exc:
        return {"error": str(exc)}, f"失敗：{exc}"


def synthesize(text: str, voice: str, rate: int | None = None) -> tuple[bytes, float]:
    """(WAV bytes, seconds of audio): macOS speech in-process (about 0.05 s a sentence),
    `say` as the fallback (tts.speak_wav)."""
    from . import tts

    return tts.speak_wav(text, voice, rate)


# -- text helpers -----------------------------------------------------------------------------

_SENTENCE_END = "。！？!?；;…\n"
REPAIR = "（系統：請直接呼叫工具查詢，不要回覆文字。）"
PROMISE = re.compile(r"(幫你|我來|我先|讓我)?(查|找)(一下|看看|查|找)|let me (check|look)", re.I)
_CJK = re.compile(r"[一-鿿]")


def split_sentences(buffer: str, final: bool = False, soft: int = 18) -> tuple[list[str], str]:
    """Pieces of streamed text ready to be spoken, and the rest. A piece ends at a sentence
    mark, or at a comma once it is `soft` characters long (so speech starts sooner)."""
    out, start = [], 0
    for i, ch in enumerate(buffer):
        if ch in _SENTENCE_END or (ch in "，,、：:" and i + 1 - start >= soft):
            piece = buffer[start : i + 1].strip()
            if piece.strip(_SENTENCE_END + "，,、：: "):
                out.append(piece)
            start = i + 1
    rest = buffer[start:]
    if final and rest.strip():
        out.append(rest.strip())
        rest = ""
    return out, rest


_PUNCT = " \t\n，,。.！!？?、；;：:…「」\"'"


def after_prefix(text: str, prefix: str) -> str | None:
    """What `text` adds after `prefix`, comparing without spaces and punctuation; None when
    `text` does not start with `prefix`."""
    i = 0
    for ch in prefix:
        if ch in _PUNCT:
            continue
        while i < len(text) and text[i] in _PUNCT:
            i += 1
        if i >= len(text) or text[i] != ch:
            return None
        i += 1
    return text[i:].lstrip(_PUNCT)


def is_english(text: str) -> bool:
    latin = sum(ch.isascii() and ch.isalpha() for ch in text or "")
    return latin >= 6 and not _CJK.search(text or "")


def filler(name: str, args: dict, user_text: str) -> str:
    """What the assistant says the moment it decides to look something up."""
    place = str(args.get("location") or "").strip()
    place = place if _CJK.search(place) else ""
    if is_english(user_text):
        what = "the weather" if name == "get_weather" else "places nearby"
        return f"Sure, let me check {what}."
    thanks = "不客氣！" if re.search(r"謝謝|谢谢|感謝|感谢", user_text or "") else "好，"
    if name == "get_weather":
        return f"{thanks}我幫你查一下{place}的天氣。" if place else f"{thanks}我幫你查一下天氣。"
    query = str(args.get("query") or "店家").strip() or "店家"
    if not query.endswith(("店", "廳", "館")) and len(query) <= 4 and _CJK.search(query):
        query += "店"
    where = f"{place}附近" if place else "附近"
    return f"{thanks}我幫你找一下{where}的{query}。"


def tool_digest(name: str, result: dict) -> str:
    """The tool result as compact JSON for the composing LLM (no URLs to read out)."""
    if "error" in result:
        return json.dumps({"error": result["error"]}, ensure_ascii=False)
    if name == "search_places":
        keep = {k: result.get(k) for k in ("location", "query", "match", "note")}
        keep["places"] = [
            {k: p.get(k) for k in ("name", "kind", "distance_m", "opening_hours", "address")}
            for p in result.get("places", [])
        ]
        return json.dumps(keep, ensure_ascii=False)
    return json.dumps({k: v for k, v in result.items() if k != "source"}, ensure_ascii=False)


def _arguments(call: dict) -> dict:
    args = (call.get("function") or {}).get("arguments") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    return args if isinstance(args, dict) else {}


# -- the engine -------------------------------------------------------------------------------


class VoiceAgent:
    def __init__(
        self,
        emit: Callable[[dict], None],
        *,
        llm: str = DEFAULT_LLM,
        voice: str = DEFAULT_VOICE,
        rate: int | None = None,
        allow_network: bool = False,
        barge_in: bool = True,
        base_url: str = OLLAMA_URL,
        chat: Callable[..., Iterator[dict]] = ollama_chat,
        tools: Callable[[str, dict], tuple[dict, str]] = run_tool,
        speak: Callable[[str, str, int | None], tuple[bytes, float]] = synthesize,
        think: bool | None | str = "auto",
        display: Callable[[str], str] | None = None,
        append_only: bool = False,
        early_commit: bool = True,
        call_limit: int = CALL_LIMIT,
        clock: Callable[[], float] = time.perf_counter,
    ):
        self.emit_raw, self.llm, self.voice, self.rate = emit, llm, voice, rate
        self.allow_network, self.barge_in, self.base_url = allow_network, barge_in, base_url
        self.chat, self.tools, self.speak = chat, tools, speak
        self.think = think_value(llm, base_url) if think == "auto" else think
        self.display = display or (lambda text: text)
        self.append_only, self.call_limit, self.clock = append_only, call_limit, clock
        self.t0 = clock()
        self.lock = threading.RLock()
        self.inbox: queue.Queue = queue.Queue()
        self.history: list[dict] = []  # plain {"role", "content"} turns for the LLM
        self.memory: dict = {}
        self.turns: list[dict] = []
        self.calls: list[dict] = []
        self.speeches: dict[str, dict] = {}
        self.unplayed: set[tuple[str, int]] = set()  # audio chunks sent, not yet played out
        self.vad: list[list] = []  # [start, end] of user speech, session clock
        self.call_counts: dict[str, int] = {}
        self.epoch = 0  # bumped by a barge-in: replies of older epochs stop
        self.busy = False
        self.running_tools = 0
        self.closed = False
        self.was_idle = True
        self.partial = ""  # text of the utterance so far (append-only ASR: its pieces)
        self.early_commit = early_commit
        self.awaiting: dict | None = None  # committed at speech end, final text pending
        self.commit_timer: threading.Timer | None = None
        self.llm_load_s: float | None = None
        self.brain = threading.Thread(target=self._loop, name="voice-agent", daemon=True)

    # -- plumbing -----------------------------------------------------------------------------

    def now(self) -> float:
        return round(self.clock() - self.t0, 3)

    def emit(self, event: dict) -> None:
        if not self.closed or event.get("type") == "done":
            self.emit_raw({**event, "t": self.now()})

    def start(self, warm: bool = True) -> None:
        self.brain.start()
        if warm:
            threading.Thread(target=self._warm, name="voice-agent-warm", daemon=True).start()

    def _warm(self) -> None:
        """Load the LLM and fill its prompt cache with the system prompt and tools, and
        render the commonest filler once, so the first turn does not pay for loading."""
        started = self.clock()
        try:
            messages = [{"role": "system", "content": system_prompt({}, self.allow_network)},
                        {"role": "user", "content": "你好"}]  # fmt: skip
            tools = TOOL_SPECS if self.allow_network else None
            for item in self.chat(self.base_url, self.llm, messages, tools, self.think, 1):
                if item.get("done"):
                    break
            self.llm_load_s = round(self.clock() - started, 2)
            self.speak("好，我幫你查一下。", self.voice, self.rate)
            self.emit({"type": "llm_ready", "llm": self.llm, "seconds": self.llm_load_s})
        except Exception as exc:  # the page shows it; turns will report their own errors
            self.emit({"type": "error", "error": f"{self.llm}: {exc}"})

    # -- input: speech recognition --------------------------------------------------------------

    def on_asr(self, event: dict, audio_t0: float | None = None) -> None:
        """One event of gateway/live.py. `audio_t0` is the LiveSession's clock origin, so
        audio times (t_audio, VAD) can be put on the session clock.

        A turn starts when the utterance is over: for append-only models (Voxtral) shortly
        after the VAD's speech end; for the others at the VAD's speech end too when
        `early_commit` is on (the partial text is final by then, and the recogniser's own
        endpoint needs another ~0.5 s of silence), otherwise at the recogniser's final text.
        The final text of an early-committed utterance only adds what came after it."""
        kind = event.get("type")
        offset = (audio_t0 - self.t0) if audio_t0 is not None else 0.0
        if kind == "speech_start":
            t = round(event["t_audio"] + offset, 3)
            with self.lock:
                self.vad.append([t, None])
                if self.commit_timer:  # still the same utterance (append-only ASR)
                    self.commit_timer.cancel()
                    self.commit_timer = None
                if self.awaiting:
                    self.awaiting["resumed"] = True
            self.emit({"type": "vad", "state": "start", "at": t})
            self._maybe_barge_in(t)
        elif kind == "speech_end":
            t = round(event["t_audio"] + offset, 3)
            with self.lock:
                if self.vad and self.vad[-1][1] is None:
                    self.vad[-1][1] = t
            self.emit({"type": "vad", "state": "end", "at": t})
            if self.append_only and self.partial.strip():
                with self.lock:
                    self.commit_timer = threading.Timer(COMMIT_DELAY_S, self._commit)
                    self.commit_timer.start()
            elif not self.append_only and self.early_commit and self.partial.strip():
                text = self.partial.strip()
                with self.lock:
                    self.partial = ""
                    self.awaiting = {"text": text, "resumed": False}
                self._user(text, early=True)
        elif kind in ("partial", "final"):
            text = event.get("display") or self.display(event.get("text") or "")
            if self.append_only:
                if not self.partial and not text.strip(_PUNCT):
                    return  # punctuation that belongs to the utterance already committed
                self.partial += text
                self.emit({"type": "user_partial", "text": self.partial})
                return
            if self.awaiting:  # text of an utterance already committed at its speech end
                rest = after_prefix(text, self.awaiting["text"])
                if rest is None:  # the recogniser revised it
                    rest = text if self.awaiting["resumed"] else ""
                if kind == "final":
                    with self.lock:
                        committed, self.awaiting = self.awaiting["text"], None
                    for turn in reversed(self.turns):
                        if turn["user"] == committed:
                            turn["asr_text"] = text
                            break
                text = rest
            if kind == "partial":
                self.partial = text
                if text.strip():
                    self.emit({"type": "user_partial", "text": text})
            else:
                self.partial = ""
                if text.strip(_PUNCT):
                    self._user(text.strip())

    def _commit(self) -> None:
        with self.lock:
            text, self.partial, self.commit_timer = self.partial.strip(), "", None
        if text.strip(_PUNCT):  # a late full stop alone is not a turn
            self._user(text)

    def _user(self, text: str, early: bool = False) -> None:
        with self.lock:
            span = self.vad[-1] if self.vad else [None, None]
            turn = {"turn": len(self.turns), "user": text, "speech_start": span[0],
                    "speech_end": span[1], "asr_final": self.now(), "speech_ids": [],
                    "tool_calls": [], "assistant": "", "decision": None,
                    "early_commit": early}  # fmt: skip
            if turn["speech_end"] is None:  # final text before the VAD saw the end
                turn["speech_end"] = turn["asr_final"]
            self.turns.append(turn)
            self.was_idle = False
        self.emit({"type": "user_final", "turn": turn["turn"], "text": text,
                   "speech_end": turn["speech_end"]})  # fmt: skip
        self.inbox.put(("user", turn))

    def _maybe_barge_in(self, t: float) -> None:
        with self.lock:
            talking = bool(self.unplayed) or (self.busy and self._current_speech())
            if not (self.barge_in and talking):
                return
            self.epoch += 1
            stopped = sorted({sid for sid, _ in self.unplayed} | set(self._current_speech()))
            self.unplayed.clear()
            for sid in stopped:
                speech = self.speeches.get(sid)
                if speech and speech.get("end") is None:
                    speech["interrupted"] = True
                    speech["end"] = t
        self.emit({"type": "barge_in", "speech_ids": stopped, "at": t})
        self._check_idle()

    def _current_speech(self) -> list[str]:
        return [sid for sid, s in self.speeches.items() if s.get("generating")]

    # -- input: playback reports from the page --------------------------------------------------

    def on_client(self, message: dict) -> None:
        sid, seq = message.get("speech_id"), message.get("seq", 0)
        speech = self.speeches.get(sid)
        if not speech:
            return
        now = self.now()
        turn = self.turns[speech["turn"]] if speech["turn"] is not None else None
        if message.get("type") == "playing":
            with self.lock:
                first = speech["start"] is None
                if first:
                    speech["start"] = now
                if turn is not None and turn.get("first_audio") is None:
                    turn["first_audio"] = now
                if turn is not None and speech["kind"] == "answer":
                    turn.setdefault("answer_audio", now)
            self.emit({"type": "speaking", "speech_id": sid, "seq": seq, "on": True})
            if first and turn is not None:
                self._turn_metrics(turn)
        elif message.get("type") == "played":
            with self.lock:
                self.unplayed.discard((sid, seq))
                if message.get("stopped"):
                    speech["interrupted"] = True
                left = any(s == sid for s, _ in self.unplayed)
                finished = not left and not speech["generating"]
                if finished and speech["end"] is None:
                    speech["end"] = now
            if finished:
                self.emit({"type": "speaking", "speech_id": sid, "on": False})
                if turn is not None:
                    self._turn_metrics(turn)
            self._check_idle()

    # -- the brain: one thread, one item at a time ---------------------------------------------

    def _loop(self) -> None:
        while (item := self.inbox.get()) is not None:
            kind, payload = item
            with self.lock:
                self.busy = True
            try:
                if kind == "user":
                    self._respond(payload)
                elif kind == "tool":
                    self._answer_from_tool(payload)
            except Exception as exc:
                self.emit({"type": "error", "error": f"{type(exc).__name__}: {exc}"})
            finally:
                with self.lock:
                    self.busy = False
                self._check_idle()

    def _messages(self, system: str, extra: list[dict]) -> list[dict]:
        return [{"role": "system", "content": system}, *self.history[-HISTORY_MESSAGES:], *extra]

    def _remember(self, role: str, text: str) -> None:
        if not text:
            return
        if self.history and self.history[-1]["role"] == role == "assistant":
            self.history[-1]["content"] += text
        else:
            self.history.append({"role": role, "content": text})

    def _respond(self, turn: dict) -> None:
        """Front agent: answer, ask, or start tool calls and say a filler at once."""
        epoch = self.epoch
        tools = TOOL_SPECS if self.allow_network else None
        messages = self._messages(
            system_prompt(self.memory, self.allow_network),
            [{"role": "user", "content": turn["user"]}],
        )
        started = self.clock()
        turn["llm_start"] = self.now()
        self.emit({"type": "llm", "turn": turn["turn"], "phase": "start", "what": "decide"})
        speech = self._new_speech(turn, "reply")
        buffer, calls = "", []
        for item in self.chat(self.base_url, self.llm, messages, tools, self.think, MAX_TOKENS):
            if (item.get("content") or item.get("tool_calls")) and turn.get("llm_first_s") is None:
                turn["llm_first_s"] = round(self.clock() - started, 3)
            if item.get("content"):
                buffer += item["content"]
                pieces, buffer = split_sentences(buffer)
                if not self._say(speech, pieces, epoch):
                    break
            if item.get("tool_calls"):
                calls += item["tool_calls"]
            if item.get("done"):
                turn["llm_load_s"] = round(item.get("load_s") or 0, 2)
        interrupted = epoch != self.epoch
        if not interrupted:
            pieces, _ = split_sentences(buffer, final=True)
            self._say(speech, pieces, epoch)
        self._remember("user", turn["user"])
        calls = [c for c in calls if (c.get("function") or {}).get("name") in TOOL_NAMES]
        if not calls and not interrupted and self.allow_network and PROMISE.search(speech["text"]):
            # It said it would look something up but called nothing: ask once more for the
            # call; what it said already serves as the filler.
            turn["repaired"] = True
            nudge = [{"role": "assistant", "content": speech["text"]},
                     {"role": "user", "content": REPAIR}]  # fmt: skip
            for item in self.chat(self.base_url, self.llm, messages + nudge, tools, self.think,
                                  MAX_TOKENS):  # fmt: skip
                calls += item.get("tool_calls") or []
            calls = [c for c in calls if (c.get("function") or {}).get("name") in TOOL_NAMES]
        turn["llm_s"] = round(self.clock() - started, 3)
        turn["llm_end"] = self.now()
        self.emit({"type": "llm", "turn": turn["turn"], "phase": "end", "what": "decide"})
        if interrupted:  # the user went on talking: their next words get the answer
            turn["decision"] = "interrupted"
        elif calls:
            turn["decision"] = "tool"
            first = None
            for call in calls[:2]:
                name, args = call["function"]["name"], _arguments(call)
                first = first or (name, args)
                self._start_tool(turn, name, args)  # runs while the filler is spoken
            speech["kind"] = "filler"  # what it said, or the instant line, holds the floor
            if not speakable(speech["text"]):  # the model said nothing itself
                self._say(speech, [filler(first[0], first[1], turn["user"])], epoch)
        else:
            text = speakable(speech["text"])
            turn["decision"] = "ask" if text.endswith(("？", "?")) else "answer"
        turn["assistant"] = speakable(speech["text"])
        mark = "……（被使用者打斷）" if interrupted and turn["assistant"] else ""
        self._remember("assistant", turn["assistant"] + mark)
        self._finish_speech(speech)
        self._turn_metrics(turn)

    def _start_tool(self, turn: dict, name: str, args: dict) -> None:
        place = str(args.get("location") or "").strip()
        if place and _CJK.search(place):
            self.memory["location"] = geo_tools.clean_place(place)
        count = self.call_counts.get(name, 0)
        call = {"call_id": f"c{len(self.calls)}", "turn": turn["turn"], "name": name,
                "args": args, "start": self.now(), "end": None, "ok": None}  # fmt: skip
        self.calls.append(call)
        turn["tool_calls"].append(call["call_id"])
        self.emit({"type": "tool_start", **{k: call[k] for k in ("call_id", "turn", "name",
                                                                   "args", "start")}})  # fmt: skip
        if count >= self.call_limit:
            limit = f"超過這次對話的上限（{self.call_limit} 次）"
            error = {"error": f"session limit of {self.call_limit} calls"}
            self._tool_done(call, error, limit, 0.0, counted=False)
            return
        self.call_counts[name] = count + 1
        with self.lock:
            self.running_tools += 1

        def work() -> None:
            started = self.clock()
            try:
                result, summary = self.tools(name, args)
            except Exception as exc:
                result, summary = {"error": str(exc)}, f"失敗：{exc}"
            self._tool_done(call, result, summary, self.clock() - started)

        threading.Thread(target=work, name=f"tool-{name}", daemon=True).start()

    def _tool_done(self, call: dict, result: dict, summary: str, seconds: float,
                   counted: bool = True) -> None:  # fmt: skip
        call.update(end=self.now(), ok="error" not in result, summary=summary,
                    seconds=round(seconds, 3), result=result)  # fmt: skip
        self.emit({"type": "tool_end", "call_id": call["call_id"], "ok": call["ok"],
                   "summary": summary, "seconds": call["seconds"], "end": call["end"],
                   "result": _short(call["name"], result)})  # fmt: skip
        self.inbox.put(("tool", call))
        if counted:
            with self.lock:
                self.running_tools -= 1

    def _answer_from_tool(self, call: dict) -> None:
        turn = self.turns[call["turn"]]
        epoch = self.epoch
        note = (
            f"（系統：{call['name']} 的查詢結果如下，請據此回答使用者剛才的問題"
            f"「{turn['user']}」。）\n{tool_digest(call['name'], call['result'])}"
        )
        messages = self._messages(compose_prompt(), [{"role": "user", "content": note}])
        started = self.clock()
        call["compose_start"] = self.now()
        self.emit({"type": "llm", "turn": turn["turn"], "phase": "start", "what": "compose"})
        speech = self._new_speech(turn, "answer", call_id=call["call_id"])
        buffer = ""
        for item in self.chat(self.base_url, self.llm, messages, None, self.think, MAX_TOKENS):
            if item.get("content"):
                if call.get("compose_first_s") is None:
                    call["compose_first_s"] = round(self.clock() - started, 3)
                buffer += item["content"]
                pieces, buffer = split_sentences(buffer)
                if not self._say(speech, pieces, epoch):
                    break
        if epoch == self.epoch:
            pieces, buffer = split_sentences(buffer, final=True)
            self._say(speech, pieces, epoch)
        call["compose_s"] = round(self.clock() - started, 3)
        self.emit({"type": "llm", "turn": turn["turn"], "phase": "end", "what": "compose"})
        self._finish_speech(speech)
        spoken = speakable(speech["text"])
        turn["answer"] = spoken
        turn["answer_interrupted"] = epoch != self.epoch
        suffix = "（被使用者打斷）" if epoch != self.epoch else ""
        self._remember("assistant", spoken + suffix)
        self._turn_metrics(turn)

    # -- speech output -----------------------------------------------------------------------

    def _new_speech(self, turn: dict | None, kind: str, call_id: str | None = None) -> dict:
        sid = f"s{len(self.speeches)}"
        speech = {"speech_id": sid, "turn": turn["turn"] if turn else None, "kind": kind,
                  "call_id": call_id, "text": "", "seq": 0, "generating": True,
                  "start": None, "end": None, "interrupted": False, "tts_s": 0.0}  # fmt: skip
        with self.lock:
            self.speeches[sid] = speech
        if turn is not None:
            turn["speech_ids"].append(sid)
        return speech

    def _finish_speech(self, speech: dict) -> None:
        with self.lock:
            speech["generating"] = False
            if not speech["seq"]:  # nothing was said
                self.speeches.pop(speech["speech_id"], None)
                return
            left = any(s == speech["speech_id"] for s, _ in self.unplayed)
            if not left and speech["end"] is None:
                speech["end"] = self.now()
        self.emit({"type": "assistant", "speech_id": speech["speech_id"], "turn": speech["turn"],
                   "kind": speech["kind"], "text": speakable(speech["text"]),
                   "final": True})  # fmt: skip

    def _say(self, speech: dict, pieces: list[str], epoch: int) -> bool:
        """Synthesize and send pieces of the reply; False once a barge-in cut it off."""
        for piece in pieces:
            if epoch != self.epoch:
                return False
            text = speakable(piece)
            if not text:
                continue
            started = self.clock()
            try:
                wav, duration = self.speak(text, self.voice, self.rate)
            except Exception as exc:
                self.emit({"type": "error", "error": f"speech synthesis failed: {exc}"})
                continue
            speech["tts_s"] = round(speech["tts_s"] + self.clock() - started, 3)
            if epoch != self.epoch:
                return False
            with self.lock:
                seq = speech["seq"]
                speech["seq"] += 1
                speech["text"] += text
                self.unplayed.add((speech["speech_id"], seq))
                turn = self.turns[speech["turn"]] if speech["turn"] is not None else None
                if turn is not None and turn.get("first_audio_sent") is None:
                    turn["first_audio_sent"] = self.now()
            self.emit({"type": "assistant", "speech_id": speech["speech_id"],
                       "turn": speech["turn"], "kind": speech["kind"],
                       "text": speech["text"], "final": False})  # fmt: skip
            self.emit({"type": "audio", "speech_id": speech["speech_id"], "seq": seq,
                       "turn": speech["turn"], "kind": speech["kind"], "text": text,
                       "duration": round(duration, 3),
                       "wav": base64.b64encode(wav).decode()})  # fmt: skip
        return epoch == self.epoch

    # -- bookkeeping ----------------------------------------------------------------------------

    def _turn_metrics(self, turn: dict) -> None:
        self.emit({"type": "turn", **metrics(turn, self.calls)})

    def _check_idle(self) -> None:
        with self.lock:
            idle = (not self.busy and self.inbox.empty() and not self.running_tools
                    and not self.unplayed and not self._current_speech())  # fmt: skip
            changed = idle and not self.was_idle
            self.was_idle = idle
        if changed:
            self.emit({"type": "idle"})

    def close(self) -> dict:
        """Stop the engine and return the session summary (also emitted as "done")."""
        with self.lock:
            if self.commit_timer:
                self.commit_timer.cancel()
        if self.append_only and self.partial.strip():
            self._commit()
        self.inbox.put(None)
        if self.brain.is_alive():
            self.brain.join(timeout=5)
        summary = self.summary()
        self.closed = True
        self.emit({"type": "done", "summary": summary})
        return summary

    def summary(self) -> dict:
        speeches = [
            {
                k: s.get(k)
                for k in (
                    "speech_id",
                    "turn",
                    "kind",
                    "call_id",
                    "text",
                    "start",
                    "end",
                    "interrupted",
                    "tts_s",
                )
            }  # fmt: skip
            for s in self.speeches.values()
        ]
        calls = [{k: v for k, v in c.items() if k != "result"} | {
            "result": _short(c["name"], c.get("result") or {})} for c in self.calls]  # fmt: skip
        return {
            "llm": self.llm,
            "voice": self.voice,
            "allow_network": self.allow_network,
            "barge_in": self.barge_in,
            "llm_load_s": self.llm_load_s,
            "turns": [metrics(t, self.calls) for t in self.turns],
            "tool_calls": calls,
            "speeches": speeches,
            "user_speech": [list(v) for v in self.vad],
            "memory": dict(self.memory),
        }


def _short(name: str, result: dict) -> dict:
    """What the page shows of a tool result (and what the run file keeps)."""
    if "error" in result:
        return {"error": result["error"]}
    if name == "search_places":
        return {
            "location": result.get("location"),
            "match": result.get("match"),
            "note": result.get("note"),
            "sources": result.get("sources"),
            "places": [
                {k: p.get(k) for k in ("name", "distance_m", "opening_hours", "address", "osm")}
                for p in result.get("places", [])
            ],
            "attribution": result.get("attribution"),
        }
    if name == "get_weather":
        return {k: result.get(k) for k in ("location", "temperature_c", "condition", "today",
                                           "rain_chance_next_6h_pct", "source")}  # fmt: skip
    return result


def metrics(turn: dict, calls: list[dict]) -> dict:
    """A turn's timings (session clock, seconds) and the latencies derived from them:
    response_s = user stopped speaking -> the assistant's first audio started playing;
    answer_s = user stopped speaking -> the spoken tool answer started playing."""
    out = {k: turn.get(k) for k in (
        "turn", "user", "assistant", "answer", "decision", "speech_start", "speech_end",
        "asr_final", "llm_start", "llm_end", "llm_first_s", "llm_s", "llm_load_s", "repaired",
        "first_audio_sent", "first_audio",
        "answer_audio", "answer_interrupted", "tool_calls", "speech_ids")}  # fmt: skip
    end = turn.get("speech_end")

    def since(t):
        return round(t - end, 3) if t is not None and end is not None else None

    out["asr_s"] = since(turn.get("asr_final"))
    out["response_s"] = since(turn.get("first_audio"))
    out["answer_s"] = since(turn.get("answer_audio"))
    mine = [c for c in calls if c["call_id"] in (turn.get("tool_calls") or [])]
    out["tool_s"] = round(sum(c.get("seconds") or 0 for c in mine), 3) if mine else None
    out["compose_first_s"] = next((c.get("compose_first_s") for c in mine), None)
    return out
