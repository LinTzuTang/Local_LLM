"""OCR engines on this Mac, and MiniCPM-OCR Jev, which scores an OCR result against the image.

    from local_llm import ocr
    text = ocr.read("minicpm-ocr", Path("page.png"))
    score = ocr.jev_score(Path("block.png"), text)  # 0-1, higher = more likely correct

Engines (ENGINES), each reading a whole image in one request:
- minicpm-ocr: MiniCPM-OCR 1.0 (OpenBMB, 1.3B) Q8_0 GGUF in llama-server (llama.cpp, Metal),
  prompt "Text Recognition:", temperature 0. llama-server is started on first use and kept
  running (LlamaServer); one already serving the same GGUF on the port is reused.
- glm-ocr, deepseek-ocr: GLM-OCR and DeepSeek-OCR 3B in Ollama.
- ocrmac: macOS Vision through ocrmac (accurate, zh-Hant / zh-Hans / en-US).
- docling: the gateway's document parser (layout, TableFormer, macOS Vision OCR).

Jev (MiniCPM-OCR Jev, 2B) is a decision model: llama.cpp serves it at /v1/systemone and
answers with the probability that the text matches the image; it is given the image, the text
and the text drawn as an image (render: Pillow and a system CJK font here, where the official
service renders with Chromium on Linux). It judges one region (a paragraph, a title), not a
whole page.

plain() and distance() are the scoring of scripts/ocr_eval: markup, case, NFKC and whitespace
ignored, normalised edit distance.
"""

from __future__ import annotations

import base64
import io
import re
import shutil
import subprocess
import time
import unicodedata
from pathlib import Path

import httpx

PROJECT = Path(__file__).resolve().parents[2]
MODELS = PROJECT / "models" / "ocr"
OCR_GGUF = MODELS / "MiniCPM-OCR-1.0-gguf"
JEV_GGUF = MODELS / "MiniCPM-OCR-Jev-gguf"
OLLAMA = "http://127.0.0.1:11434"
FONT = "/System/Library/Fonts/STHeiti Medium.ttc"

# id -> facts shown on the platform; size in GB on disk.
ENGINES = {
    "minicpm-ocr": {
        "label": "MiniCPM-OCR 1.0",
        "runtime": "llama.cpp",
        "size_gb": 1.7,
        "licence": "Apache-2.0",
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
    "ocrmac": {"label": "macOS Vision", "runtime": "macOS", "size_gb": None, "licence": "macOS"},
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


class LlamaServer:
    """One llama-server process for one GGUF on one port, started on first use."""

    def __init__(self, model: Path, mmproj: Path, port: int, ctx: int):
        self.model, self.mmproj, self.port, self.ctx = model, mmproj, port, ctx
        self.url = f"http://127.0.0.1:{port}"
        self.process: subprocess.Popen | None = None

    def serving(self) -> bool:
        try:
            r = httpx.get(f"{self.url}/v1/models", timeout=2)
            return r.status_code == 200 and any(
                self.model.name in m.get("id", "") for m in r.json().get("data", [])
            )
        except (httpx.HTTPError, ValueError):
            return False

    def ensure(self, timeout: float = 120) -> str:
        if self.serving():
            return self.url
        if not shutil.which("llama-server"):
            raise RuntimeError("llama-server not found: brew install llama.cpp")
        if not self.model.exists():
            raise RuntimeError(f"model not downloaded: {self.model}")
        self.process = subprocess.Popen(
            [
                "llama-server",
                "-m",
                str(self.model),
                "--mmproj",
                str(self.mmproj),
                "-ngl",
                "99",
                "-c",
                str(self.ctx),
                "--port",
                str(self.port),
                "--jinja",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited ({self.process.returncode}); is port {self.port} in use?"
                )
            if self.serving():
                return self.url
            time.sleep(0.5)
        raise RuntimeError("llama-server did not start in time")

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
        self.process = None


OCR_SERVER = LlamaServer(
    OCR_GGUF / "MiniCPM-OCR-1_0-Q8_0.gguf", OCR_GGUF / "mmproj-model-f16-4x.gguf", 18499, 16384
)
JEV_SERVER = LlamaServer(
    JEV_GGUF / "MiniCPM-OCR-Jev-Q8_0.gguf", JEV_GGUF / "mmproj-model-f16.gguf", 18498, 32768
)


def available(engine: str) -> tuple[bool, str]:
    """(usable, reason when not)."""
    if engine == "minicpm-ocr":
        if not shutil.which("llama-server"):
            return False, "brew install llama.cpp"
        return OCR_SERVER.model.exists(), "hf download openbmb/MiniCPM-OCR-1.0-gguf"
    if engine == "ocrmac":
        try:
            import ocrmac  # noqa: F401
        except ImportError:
            return False, "uv add ocrmac"
    return True, ""


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _bytes(image: Path | bytes) -> bytes:
    return image if isinstance(image, bytes) else Path(image).read_bytes()


def read(engine: str, image: Path | bytes, gateway: str | None = None, ollama: str = OLLAMA) -> str:
    """The text (or Markdown) of the image, read by one engine."""
    data = _bytes(image)
    if engine == "minicpm-ocr":
        url = OCR_SERVER.ensure()
        body = {
            "temperature": 0,
            "max_tokens": 8192,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64(data)}"},
                        },
                        {"type": "text", "text": PROMPTS[engine]},
                    ],
                }
            ],
        }
        r = httpx.post(f"{url}/v1/chat/completions", json=body, timeout=900)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    if engine in ("glm-ocr", "deepseek-ocr"):
        body = {
            "model": ENGINES[engine]["ollama"],
            "stream": False,
            "keep_alive": "10m",
            "messages": [{"role": "user", "content": PROMPTS[engine], "images": [b64(data)]}],
            "options": {"temperature": 0, "num_ctx": 16384, "num_predict": 8192},
        }
        r = httpx.post(f"{ollama}/api/chat", json=body, timeout=900)
        r.raise_for_status()
        text = r.json()["message"]["content"]
        return re.sub(
            r"<\|det\|>.*?<\|/det\|>|<\|/?ref\|>", "", text, flags=re.S
        )  # DeepSeek's boxes
    if engine == "ocrmac":
        from ocrmac import ocrmac
        from PIL import Image

        with Image.open(io.BytesIO(data)) as img:
            lines = ocrmac.OCR(
                img.convert("RGB"),
                recognition_level="accurate",
                language_preference=["zh-Hant", "zh-Hans", "en-US"],
            ).recognize()
        return "\n".join(text for text, _, _ in lines)
    if engine == "docling":
        from . import gateway_client

        kwargs = {"base_url": gateway} if gateway else {}
        return gateway_client.parse_document(data, filename="page.png", **kwargs)["markdown"]
    raise ValueError(f"unknown OCR engine {engine}")


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


def jev_score(image: Path | bytes, text: str) -> float:
    """Probability (0-1) that `text` is exactly what the image shows."""
    url = JEV_SERVER.ensure()
    r = httpx.post(f"{url}/v1/systemone", json=jev_body(_bytes(image), text), timeout=120)
    r.raise_for_status()
    return float(r.json()["answers"]["consistency"]["probabilities"]["1"])


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
