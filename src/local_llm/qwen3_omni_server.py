"""Qwen3-Omni-30B-A3B as a small local HTTP server, for the live voice assistant's turn-based mode.

    uv run --no-project --with mlx-vlm==0.7.9 python src/local_llm/qwen3_omni_server.py \
        --model models/speech_extra/qwen3-omni-30b-4bit --port 9070

mlx-vlm (which runs Qwen3-Omni's thinker, talker and code2wav on MLX) pins a newer mlx and
many packages the platform does not use, so the model runs in its own throw-away environment
and the gateway talks to it over HTTP (local_llm/voice_omni.py starts it on demand). This file
imports nothing from local_llm. It serves one conversation at a time:

    GET  /health                -> {"ok": true}
    POST /reset {"lang"}        -> a new conversation (system prompt in zh or en)
    POST /reply {"audio", "lang"}
        audio: base64 16 kHz mono PCM16, the user's whole utterance. The reply streams as
        newline-delimited JSON: {"type": "text", "text"} once the thinker has written it, then
        {"type": "audio", "pcm"} (base64 24 kHz mono PCM16: the first piece at once, then
        about 0.6 s each) as the talker speaks, then {"type": "done"}. Closing the
        connection stops the speech.

The earlier turns stay in the prompt (the user's audio and the model's text, last HISTORY
turns). It exits by itself after IDLE_S seconds without a request.
"""

import argparse
import base64
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

RATE = 16000
OUT_RATE = 24000
CHUNK_S = 0.6  # speech sent to the gateway in pieces of about this length
HISTORY = 6
IDLE_S = 900
AUDIO = "<|audio_start|><|audio_pad|><|audio_end|>"
SYSTEM = {
    "zh": "你是即時語音助理。請用台灣慣用的繁體中文口語回答，一到兩句、簡短自然；"
    "使用者說英文就用英文回答。你沒有連網，查不到即時天氣、預報或店家資訊；"
    "被問到時要老實說查不到，不要編造溫度、降雨機率、店名或營業時間。",
    "en": "You are a real-time voice assistant. Reply in natural spoken English, one or two short "
    "sentences. You have no internet access and cannot look up live weather, forecasts or nearby "
    "shops; when asked, say honestly that you cannot look it up, and never make up temperatures, "
    "rain chances, shop names or opening hours.",
}  # fmt: skip


class Omni:
    def __init__(self, path: str, speaker: str):
        import mlx.core as mx
        from mlx_vlm import load

        self.mx, self.speaker = mx, speaker
        self.model, self.processor = load(path)
        self.eos = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.lock = threading.Lock()
        self.lang, self.turns = "zh", []  # [(audio, reply)]
        for _ in self.reply(np.zeros(RATE, dtype=np.float32), "zh", keep=False):  # compile kernels
            pass

    def reset(self, lang: str) -> None:
        self.lang, self.turns = (lang if lang in SYSTEM else "zh"), []

    def reply(self, audio: np.ndarray, lang: str, keep: bool = True):
        mx = self.mx
        turns = self.turns[-HISTORY:]
        prompt = f"<|im_start|>system\n{SYSTEM[self.lang if keep else lang]}<|im_end|>\n"
        for _, said in turns:
            prompt += (
                f"<|im_start|>user\n{AUDIO}<|im_end|>\n<|im_start|>assistant\n{said}<|im_end|>\n"
            )
        prompt += f"<|im_start|>user\n{AUDIO}<|im_end|>\n<|im_start|>assistant\n"
        audios = [a for a, _ in turns] + [audio]
        inputs = self.processor(text=prompt, audio=audios, sampling_rate=RATE, return_tensors="np")
        text, pending, sent = "", [], 0
        try:
            for kind, value in self.model.generate_stream(
                mx.array(inputs["input_ids"]), speaker=self.speaker, thinker_eos_token_id=self.eos,
                thinker_max_new_tokens=256, chunk_size=12, left_context_size=25,
                input_features=mx.array(inputs["input_features"]),
                feature_attention_mask=mx.array(inputs["feature_attention_mask"]),
            ):  # fmt: skip
                if kind == "text":
                    ids = [x for x in value if x != self.eos]
                    text = self.processor.tokenizer.decode(ids, skip_special_tokens=True).strip()
                    if keep:
                        self.turns.append((audio, text))
                    yield {"type": "text", "text": text}
                else:  # the first piece at once, then about CHUNK_S at a time
                    pending.append(np.array(value).reshape(-1))
                    if sent == 0 or sum(len(x) for x in pending) >= CHUNK_S * OUT_RATE:
                        yield self._audio(pending)
                        pending, sent = [], sent + 1
            if pending:
                yield self._audio(pending)
        finally:
            mx.clear_cache()

    @staticmethod
    def _audio(pieces: list) -> dict:
        pcm = (np.clip(np.concatenate(pieces), -1, 1) * 32767).astype("<i2")
        return {"type": "audio", "pcm": base64.b64encode(pcm.tobytes()).decode()}


def serve(omni: Omni, port: int) -> None:
    last = [time.monotonic()]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet
            pass

        def _json(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            last[0] = time.monotonic()
            self._json(200, {"ok": True}) if self.path == "/health" else self._json(404, {})

        def do_POST(self):
            last[0] = time.monotonic()
            body = json.loads(
                self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}"
            )
            if self.path == "/reset":
                with omni.lock:
                    omni.reset(body.get("lang", "zh"))
                return self._json(200, {"ok": True})
            if self.path != "/reply":
                return self._json(404, {})
            pcm = (
                np.frombuffer(base64.b64decode(body["audio"]), dtype="<i2").astype(np.float32)
                / 32768
            )
            with omni.lock:
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.end_headers()
                try:
                    for event in omni.reply(pcm, body.get("lang", "zh")):
                        self.wfile.write((json.dumps(event) + "\n").encode())
                        self.wfile.flush()
                        last[0] = time.monotonic()
                    self.wfile.write(b'{"type": "done"}\n')
                except (BrokenPipeError, ConnectionResetError):  # the user cut in: stop speaking
                    pass

    def watchdog():
        while time.sleep(30) is None:
            if time.monotonic() - last[0] > IDLE_S:
                os._exit(0)

    threading.Thread(target=watchdog, daemon=True).start()
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=9070)
    parser.add_argument("--speaker", default="Chelsie")
    args = parser.parse_args()
    serve(Omni(args.model, args.speaker), args.port)


if __name__ == "__main__":
    main()
