"""OCR engines on this Mac, and MiniCPM-OCR Jev, which scores an OCR result against the image.

    from local_llm import ocr
    text = ocr.read("minicpm-ocr", Path("page.png"))                 # Ollama
    text = ocr.read("ocrmac", Path("page.png"), gateway=GATEWAY)     # through the model gateway
    score = gateway_client.ocr_score(Path("block.png"), text)        # Jev, 0-1

Every engine sits behind one of the platform's two front doors (docs: HackMD 〈地端 AI 模型推論
架構〉): the OCR models are in Ollama, the rest in the model gateway.
- minicpm-ocr: MiniCPM-OCR 1.0 (OpenBMB, 1.3B) in Ollama, Q8_0 GGUF with its vision projector and
  the DSpark draft head for speculative decoding (models/ocr/MiniCPM-OCR-1.0-gguf/Modelfile,
  `ollama create minicpm-ocr`, Ollama 0.40.12 or later); prompt "Text Recognition:".
- glm-ocr, deepseek-ocr: GLM-OCR and DeepSeek-OCR 3B in Ollama.
- ocrmac: macOS Vision (ocrmac, accurate, zh-Hant / zh-Hans / en-US), gateway /v1/ocr.
- docling: the gateway's document parser (layout, TableFormer, macOS Vision OCR).
read() is the client side (Ollama or the gateway); read_local() runs the gateway's engines in
this process and is what /v1/ocr calls.

Jev (MiniCPM-OCR Jev, 2B) is a decision model: llama.cpp's llama-server serves it at
/v1/systemone and answers with the probability that the text matches the image; it is given the
image, the text and the text drawn as an image (render: Pillow and a system CJK font here, where
the official service renders with Chromium on Linux). It judges one region (a paragraph, a
title), not a whole page. jev_score() starts that llama-server on first use as a gateway sidecar
(voice_omni.Sidecar: stopped after 15 idle minutes); the platform reaches it through
/v1/ocr/score.

plain() and distance() are the scoring of scripts/ocr_eval: markup, case, NFKC and whitespace
ignored, normalised edit distance.
"""

from __future__ import annotations

import base64
import io
import re
import shutil
import unicodedata
from pathlib import Path

import httpx

PROJECT = Path(__file__).resolve().parents[2]
MODELS = PROJECT / "models" / "ocr"
OCR_GGUF = MODELS / "MiniCPM-OCR-1.0-gguf"
JEV_GGUF = MODELS / "MiniCPM-OCR-Jev-gguf"
JEV_MODEL = JEV_GGUF / "MiniCPM-OCR-Jev-Q8_0.gguf"
JEV_PORT = 18498
OLLAMA = "http://127.0.0.1:11434"
FONT = "/System/Library/Fonts/STHeiti Medium.ttc"

# id -> facts shown on the platform; size in GB on disk.
ENGINES = {
    "minicpm-ocr": {
        "label": "MiniCPM-OCR 1.0",
        "runtime": "Ollama",
        "size_gb": 2.0,
        "licence": "Apache-2.0",
        "ollama": "minicpm-ocr",
        "setup": "scripts/setup_ocr_models.sh",
    },
    "glm-ocr": {
        "label": "GLM-OCR",
        "runtime": "Ollama",
        "size_gb": 2.7,
        "licence": "MIT",
        "ollama": "glm-ocr",
    },
    "deepseek-ocr": {
        "label": "DeepSeek-OCR 3B",
        "runtime": "Ollama",
        "size_gb": 6.7,
        "licence": "MIT",
        "ollama": "deepseek-ocr:3b",
    },
    "ocrmac": {"label": "macOS Vision", "runtime": "gateway", "size_gb": None, "licence": "macOS"},
    "docling": {
        "label": "Docling + macOS Vision",
        "runtime": "gateway",
        "size_gb": 0.4,
        "licence": "MIT",
    },
}
PROMPTS = {
    "minicpm-ocr": "Text Recognition:",
    "glm-ocr": "Text Recognition:",
    "deepseek-ocr": "<|grounding|>Convert the document to markdown.",
}
GATEWAY_ENGINES = {"ocrmac"}  # read_local() runs these; /v1/ocr serves them


def available(engine: str) -> tuple[bool, str]:
    """(usable on this machine, reason when not), for the engines outside Ollama."""
    if engine == "ocrmac":
        try:
            import ocrmac  # noqa: F401
        except ImportError:
            return False, "uv add ocrmac"
    return True, ""


def jev_available() -> tuple[bool, str]:
    if not shutil.which("llama-server"):
        return False, "llama-server not found: brew install llama.cpp"
    if not JEV_MODEL.exists():
        return False, "Jev not downloaded: scripts/setup_ocr_models.sh"
    return True, ""


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _bytes(image: Path | bytes) -> bytes:
    return image if isinstance(image, bytes) else Path(image).read_bytes()


def read(engine: str, image: Path | bytes, gateway: str | None = None, ollama: str = OLLAMA) -> str:
    """The text (or Markdown) of the image, read by one engine: Ollama or the gateway."""
    data = _bytes(image)
    if engine in PROMPTS:
        body = {
            "model": ENGINES[engine]["ollama"],
            "stream": False,
            "think": False,
            "keep_alive": "10m",
            "messages": [{"role": "user", "content": PROMPTS[engine], "images": [b64(data)]}],
            "options": {"temperature": 0, "num_ctx": 16384, "num_predict": 8192},
        }
        r = httpx.post(f"{ollama}/api/chat", json=body, timeout=900)
        if r.status_code == 400 and "think" in r.text:  # models without a thinking switch
            body.pop("think")
            r = httpx.post(f"{ollama}/api/chat", json=body, timeout=900)
        r.raise_for_status()
        text = r.json()["message"]["content"]  # DeepSeek's boxes dropped
        return re.sub(r"<\|det\|>.*?<\|/det\|>|<\|/?ref\|>", "", text, flags=re.S)
    from . import gateway_client

    kwargs = {"base_url": gateway} if gateway else {}
    if engine in GATEWAY_ENGINES:
        return gateway_client.ocr(data, engine, **kwargs)
    if engine == "docling":
        return gateway_client.parse_document(data, filename="page.png", **kwargs)["markdown"]
    raise ValueError(f"unknown OCR engine {engine}")


def read_local(engine: str, image: Path | bytes) -> str:
    """Run a gateway engine in this process (the gateway's /v1/ocr)."""
    if engine != "ocrmac":
        raise ValueError(f"not a local OCR engine: {engine}")
    from ocrmac import ocrmac
    from PIL import Image

    with Image.open(io.BytesIO(_bytes(image))) as img:
        lines = ocrmac.OCR(
            img.convert("RGB"),
            recognition_level="accurate",
            language_preference=["zh-Hant", "zh-Hans", "en-US"],
        ).recognize()
    return "\n".join(text for text, _, _ in lines)


def render(text: str, width: int, size: int = 28) -> bytes:
    """The text drawn black on white, wrapped to `width` pixels (Jev's candidate image)."""
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype(FONT, size)
    width, step = max(width, 200), int(size * 1.35)
    lines = []
    for para in text.split("\n"):
        line = ""
        for ch in para:
            if line and font.getlength(line + ch) > width - 16:
                lines.append(line)
                line = ""
            line += ch
        lines.append(line)
    img = Image.new("RGB", (width, 16 + step * len(lines)), "white")
    draw = ImageDraw.Draw(img)
    for k, line in enumerate(lines):
        draw.text((8, 8 + step * k), line, fill="black", font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def data_url(data: bytes) -> str:
    """PNG data URL; images longer than 200:1 are padded with white, as the official SDK does."""
    from PIL import Image

    with Image.open(io.BytesIO(data)) as img:
        w, h = img.size
        if max(w, h) / max(1, min(w, h)) > 200:
            size = (w, w // 200 + 1) if w >= h else (h // 200 + 1, h)
            canvas = Image.new("RGB", size, "white")
            canvas.paste(img.convert("RGB"), (0, 0))
            buf = io.BytesIO()
            canvas.save(buf, format="PNG")
            data = buf.getvalue()
    return "data:image/png;base64," + b64(data)


def jev_body(image: bytes, text: str) -> dict:
    """The /v1/systemone request of the SDK's llama.cpp backend: original first, render second."""
    from PIL import Image

    with Image.open(io.BytesIO(image)) as img:
        width = img.size[0]
    return {
        "state": text,
        "files": [data_url(image), data_url(render(text, width))],
        "questions": {"consistency": {"type": "choice", "criteria": {"0": None, "1": None}}},
    }


def jev_sidecar():
    """Jev's llama-server as a sidecar of this process (started on first use, stopped after
    15 idle minutes or when another sidecar needs the room)."""
    from .voice_omni import Sidecar

    car = Sidecar.running.get("minicpm-ocr-jev")
    if car is None:
        cmd = [shutil.which("llama-server") or "llama-server", "-m", str(JEV_MODEL),
               "--mmproj", str(JEV_GGUF / "mmproj-model-f16.gguf"), "-ngl", "99", "-c", "32768",
               "--host", "127.0.0.1", "--port", str(JEV_PORT)]  # fmt: skip
        car = Sidecar("minicpm-ocr-jev", JEV_PORT, cmd, PROJECT, ready_s=180)
    return car


def jev_score(image: Path | bytes, text: str) -> float:
    """Probability (0-1) that `text` is exactly what the image shows (runs Jev here)."""
    car = jev_sidecar()
    car.acquire()
    try:
        car.ensure()
        r = httpx.post(f"{car.url}/v1/systemone", json=jev_body(_bytes(image), text), timeout=120)
        r.raise_for_status()
        return float(r.json()["answers"]["consistency"]["probabilities"]["1"])
    finally:
        car.release()


# OmniDocBench categories that count as page content (headers, footers, page numbers, figures
# and masked regions do not).
KEEP = {
    "title",
    "text_block",
    "figure_caption",
    "table_caption",
    "table_footnote",
    "figure_footnote",
    "reference",
    "equation_caption",
    "code_txt",
    "equation_isolated",
    "table",
}


def omnidoc_truth(page: dict) -> str:
    """An OmniDocBench page's annotated content in reading order: text, tables as HTML,
    display formulas as LaTeX."""
    blocks = sorted(
        (b for b in page["layout_dets"] if b["category_type"] in KEEP and not b.get("ignore")),
        key=lambda b: b.get("order") or 0,
    )
    return "\n".join(b.get("html") or b.get("latex") or b.get("text") or "" for b in blocks)


def sample_image() -> bytes:
    """A small Traditional Chinese page made on the spot (no file needed), with its text."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (980, 420), "white")
    draw = ImageDraw.Draw(img)
    title, body = ImageFont.truetype(FONT, 34), ImageFont.truetype(FONT, 26)
    draw.text((40, 30), SAMPLE_TEXT[0], fill="black", font=title)
    for k, line in enumerate(SAMPLE_TEXT[1:]):
        draw.text((40, 100 + 52 * k), line, fill="black", font=body)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


SAMPLE_TEXT = [
    "第 42 週產銷會議紀錄",
    "一、CNC 產線稼動率 87%，較上週提升 3 個百分點。",
    "二、客訴案件 2 件：包裝破損 1 件、規格不符 1 件，已開立 8D 報告。",
    "三、原物料：鋁錠報價每噸 2,350 美元，採購部建議分兩批下單。",
    "四、下週重點：新北廠導入 AOI 檢測，預計 11 月 15 日完成驗收。",
    "五、出席：生產部、品保部、採購部、業務部，共 12 人。",
]


def plain(text: str) -> str:
    """Text for scoring: HTML tags, LaTeX commands, Markdown marks, case and whitespace dropped."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"[\s#*|`$\\{}^_~>=\-:]+", "", text)


def distance(hyp: str, ref: str) -> float:
    """Normalised edit distance of plain(hyp) to plain(ref): 0 identical, 1 nothing in common."""
    from rapidfuzz.distance import Levenshtein

    return Levenshtein.normalized_distance(plain(hyp), plain(ref))
