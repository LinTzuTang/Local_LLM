"""Single speech models as the live voice assistant (voice_omni.py) with fake models: a
turn-based reply from the utterance's own audio, a barge-in that closes the reply, and the
full-duplex loop that gives way to the user."""

import threading
import time

import pytest

from local_llm import voice_omni


@pytest.fixture(autouse=True)
def _work(tmp_path, monkeypatch):
    monkeypatch.setattr(voice_omni, "WORK", tmp_path)


RATE = voice_omni.RATE
WAV = voice_omni.wav_bytes(b"\0\0" * 2400, 24000)  # 0.1 s


class TurnModel:
    def __init__(self, delay=0.0, pieces=3):
        self.delay, self.pieces = delay, pieces
        self.heard: list[bytes] = []
        self.closed = threading.Event()
        self.started = []

    def start(self, lang="zh"):
        self.started.append(lang)

    def reply(self, pcm, lang):
        self.heard.append(pcm)
        try:
            yield "text", "你好，" if lang == "zh" else "Hello, "
            for _ in range(self.pieces):
                time.sleep(self.delay)
                yield "wav", WAV
            yield "text", "有什麼需要嗎？" if lang == "zh" else "how can I help?"
        finally:
            self.closed.set()

    def close(self):
        pass


class Session:
    def __init__(self, agent_cls=voice_omni.OmniAgent, model=None, auto_play=True, **kw):
        self.events, self.auto_play = [], auto_play
        self.model = model or TurnModel()
        self.agent = agent_cls(self.emit, self.model, engine=kw.pop("engine", "qwen3-omni"), **kw)
        self.agent.start(warm=False)
        self.agent.ready.set()

    def emit(self, event):
        self.events.append(event)
        if self.auto_play and event["type"] == "audio":
            for kind in ("playing", "played"):
                self.agent.on_client(
                    {"type": kind, "speech_id": event["speech_id"], "seq": event["seq"]}
                )

    def audio(self, seconds, value=1):
        self.agent.on_audio(int(value).to_bytes(2, "little", signed=True) * int(seconds * RATE))

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

    def of(self, kind):
        return [e for e in self.events if e["type"] == kind]


def test_turn_model_hears_the_utterance_and_speaks_its_own_audio():
    s = Session()
    s.audio(1.0, 0)  # silence
    s.audio(1.5, 7)  # the user, 1.0-2.5 s
    s.audio(1.0, 0)
    s.say("你好", 1.0, 2.5)
    s.wait(lambda: s.agent.turns and s.agent.turns[0].get("decision"))
    heard = s.model.heard[0]
    margin = voice_omni.OmniAgent.MARGIN_S
    assert len(heard) == round((1.5 + 2 * margin) * RATE) * 2  # speech plus a margin each side
    assert s.agent.turns[0]["decision"] == "ask"  # it ended with a question
    assert s.agent.turns[0]["assistant"] == "你好，有什麼需要嗎？"
    assert [e["kind"] for e in s.of("audio")] == ["reply"] * 3
    assert (
        s.agent.turns[0]["first_audio"] is not None and s.agent.turns[0]["llm_first_s"] is not None
    )
    summary = s.agent.close()
    assert summary["engine"] == "qwen3-omni" and summary["kind"] == "turn"
    assert summary["llm"] == "Qwen3-Omni 30B-A3B"


def test_barge_in_closes_the_reply():
    s = Session(model=TurnModel(delay=0.1, pieces=20), auto_play=False)
    s.audio(3.0, 5)
    s.say("講個故事", 0.0, 1.0)
    s.wait(lambda: s.of("audio"))
    first = s.of("audio")[0]
    s.agent.on_client({"type": "playing", "speech_id": first["speech_id"], "seq": 0})
    s.agent.on_asr({"type": "speech_start", "t_audio": 2.0}, s.agent.t0)  # the user cuts in
    s.wait(lambda: s.of("barge_in"))
    assert s.model.closed.wait(2)
    s.wait(lambda: s.agent.turns[0].get("decision") == "interrupted")
    assert len(s.of("audio")) < 20
    s.agent.close()


def test_english_first_turn_restarts_qwen_in_english():
    model = TurnModel()
    model.__class__ = type("FakeQwen", (TurnModel, voice_omni.QwenOmni), {})
    s = Session(model=model)
    s.audio(2.0, 3)
    s.say("Hello there", 0.0, 1.5)
    s.wait(lambda: s.agent.turns and s.agent.turns[0].get("decision"))
    assert model.started == ["en"]
    assert s.agent.turns[0]["assistant"].startswith("Hello")
    s.agent.close()


class DuplexModel:
    """Talks in ticks 2-3, then listens."""

    def __init__(self):
        self.ticks = 0

    def start(self, lang="zh"):
        pass

    def tick(self, pcm):
        self.ticks += 1
        if self.ticks in (2, 3):
            return {"text": "嗯，" if self.ticks == 2 else "好的。", "listen": False, "wavs": [WAV]}
        return {"text": "", "listen": True, "wavs": []}

    def close(self):
        pass


def test_duplex_talks_on_its_own_and_gives_way_to_the_user():
    model = DuplexModel()
    s = Session(agent_cls=voice_omni.DuplexAgent, model=model, auto_play=False,
                engine="minicpm-o-duplex")  # fmt: skip
    s.agent.on_asr({"type": "speech_start", "t_audio": 0.0}, s.agent.t0)
    s.audio(1.0, 4)
    s.agent.on_asr({"type": "speech_end", "t_audio": 1.0}, s.agent.t0)
    s.agent.on_asr({"type": "final", "text": "你好"}, s.agent.t0)
    s.audio(2.0, 0)  # ticks 2 and 3: it talks
    s.wait(lambda: len(s.of("audio")) == 2)
    assert not s.agent.inbox.qsize()  # captions only: no cascade turn was queued
    s.agent.on_asr({"type": "speech_start", "t_audio": 3.0}, s.agent.t0)  # the user talks again
    s.audio(1.0, 6)  # tick 4: it listens while the user speaks
    s.wait(lambda: s.of("barge_in"))
    assert s.of("barge_in")[0]["speech_ids"] == [s.of("audio")[0]["speech_id"]]
    assert s.agent.turns[0]["assistant"] == "嗯，好的。"
    assert not s.agent.barge_in  # the model decides, not the VAD
    summary = s.agent.close()
    assert summary["kind"] == "duplex"


def test_engines_report_whether_installed(tmp_path, monkeypatch):
    monkeypatch.setattr(voice_omni, "QWEN_DIR", tmp_path / "missing")
    monkeypatch.setattr(voice_omni, "MINICPM_SERVER", tmp_path / "missing" / "server")
    assert not voice_omni.available("qwen3-omni")
    assert not voice_omni.available("minicpm-o-duplex")
    assert not voice_omni.available("cascade")
    assert set(voice_omni.ENGINES) == {"qwen3-omni", "minicpm-o", "minicpm-o-duplex"}


class LateSpeechModel(DuplexModel):
    """Writes the text first and its speech one and two seconds later (as MiniCPM-o does)."""

    def tick(self, pcm):
        self.ticks += 1
        if self.ticks == 2:
            return {"text": "你好。", "listen": False, "wavs": []}
        if self.ticks in (3, 4):
            return {"text": "", "listen": True, "wavs": [WAV]}
        return {"text": "", "listen": True, "wavs": []}


def test_duplex_reply_waits_for_its_late_speech():
    s = Session(agent_cls=voice_omni.DuplexAgent, model=LateSpeechModel(),
                engine="minicpm-o-duplex")  # fmt: skip
    s.audio(7.0, 0)
    s.wait(lambda: any(e["type"] == "assistant" and e["final"] for e in s.events))
    finals = [e for e in s.of("assistant") if e["final"]]
    assert [e["text"] for e in finals] == ["你好。"]  # one reply, not a second empty one
    assert {e["speech_id"] for e in s.of("audio")} == {finals[0]["speech_id"]}
    assert len(s.of("audio")) == 2
    s.agent.close()


def test_without_captions_the_turn_starts_at_the_end_of_speech():
    s = Session()
    s.agent.captions = False  # no ASR runs: only the VAD's events come
    s.audio(2.0, 5)
    s.agent.on_asr({"type": "speech_start", "t_audio": 0.2}, s.agent.t0)
    s.agent.on_asr({"type": "speech_end", "t_audio": 1.5}, s.agent.t0)
    s.wait(lambda: s.agent.turns and s.agent.turns[0].get("decision"))
    assert s.agent.turns[0]["user"] == voice_omni.NO_CAPTION
    assert len(s.model.heard) == 1 and s.model.heard[0]  # it got the audio all the same
    s.agent.close()


def test_a_server_in_use_is_not_stopped_for_another(monkeypatch):
    stopped = []
    monkeypatch.setattr(voice_omni.Sidecar, "healthy", lambda self: True)
    monkeypatch.setattr(voice_omni.Sidecar, "stop", lambda self: stopped.append(self.name))
    monkeypatch.setattr(voice_omni.Sidecar, "running", {})
    busy = voice_omni.Sidecar("busy", 1, ["true"], voice_omni.PROJECT)
    idle = voice_omni.Sidecar("idle", 2, ["true"], voice_omni.PROJECT)
    busy.acquire()
    voice_omni.Sidecar.running.update(busy=busy, idle=idle)
    voice_omni.Sidecar("new", 3, ["true"], voice_omni.PROJECT).ensure()
    assert stopped == ["idle"]  # the one a conversation holds keeps running
    busy.release()
    assert busy.users == 0
