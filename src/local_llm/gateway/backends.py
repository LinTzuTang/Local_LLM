"""How each kind of model is loaded, run and released.

Heavy libraries are imported inside `load`/`run`, so the gateway starts fast and only pays for
what is actually used. `load` returns (handle, device); `run` gets the handle plus the request
inputs and returns plain JSON-able data. Bad input raises ValueError (HTTP 400).
"""

import gc
import re
import sys
import threading
from pathlib import Path

from .registry import ModelSpec


class Backend:
    task = ""

    def load(self, spec: ModelSpec, path: Path) -> tuple[object, str]:
        raise NotImplementedError

    def run(self, handle, **inputs) -> dict:
        raise NotImplementedError

    def release(self, handle) -> None:
        """Drop references the library keeps on its own (memory is freed by free_memory)."""


class WhisperBackend(Backend):
    """Speech-to-text with mlx-whisper on the Apple GPU."""

    task = "asr"
    # mlx-whisper keeps "the" model in a module-level holder, so transcriptions take turns.
    _lock = threading.Lock()

    def load(self, spec, path):
        import mlx.core as mx
        from mlx_whisper.load_models import load_model

        return {"model": load_model(str(path), dtype=mx.float16), "path": str(path)}, "mlx"

    def run(self, handle, audio: Path, language: str | None = None, prompt: str | None = None):
        import mlx_whisper
        from mlx_whisper.audio import SAMPLE_RATE, load_audio
        from mlx_whisper.transcribe import ModelHolder

        try:
            samples = load_audio(str(audio))  # ffmpeg: any audio/video format, 16 kHz mono
        except RuntimeError as exc:
            raise ValueError(f"could not decode the audio file: {exc}") from exc
        with self._lock:
            ModelHolder.model, ModelHolder.model_path = handle["model"], handle["path"]
            result = mlx_whisper.transcribe(
                samples,
                path_or_hf_repo=handle["path"],
                language=language,
                initial_prompt=prompt,
                verbose=None,
            )
        return {
            "text": result["text"].strip(),
            "segments": [
                {
                    "start": round(seg["start"], 2),
                    "end": round(seg["end"], 2),
                    "text": seg["text"].strip(),
                }
                for seg in result["segments"]
            ],  # fmt: skip
            "language": result.get("language"),
            "duration": round(len(samples) / SAMPLE_RATE, 2),
        }

    def release(self, handle):
        from mlx_whisper.transcribe import ModelHolder

        with self._lock:
            if ModelHolder.model is handle["model"]:
                ModelHolder.model = ModelHolder.model_path = None


class DoclingBackend(Backend):
    """PDF/Office/HTML -> Markdown with Docling's standard pipeline (layout + TableFormer)."""

    task = "doc"

    def load(self, spec, path):
        from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
        from docling.document_converter import (
            DocumentConverter,
            ImageFormatOption,
            PdfFormatOption,
        )

        options = PdfPipelineOptions(
            artifacts_path=path,  # models/docling; Docling never downloads when this is set
            do_table_structure=True,
            accelerator_options=AcceleratorOptions(device=AcceleratorDevice.MPS),
        )
        options.table_structure_options.mode = TableFormerMode.ACCURATE
        options.do_ocr, options.ocr_options, ocr = ocr_options()
        converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF, InputFormat.IMAGE, InputFormat.DOCX,
                             InputFormat.PPTX, InputFormat.HTML, InputFormat.MD],
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=options),
                InputFormat.IMAGE: ImageFormatOption(pipeline_options=options),
            },
        )  # fmt: skip
        converter.initialize_pipeline(InputFormat.PDF)  # load the models now, not on first use
        # Docling runs layout on MPS but keeps TableFormer on the CPU (faster there).
        return {"converter": converter, "ocr": ocr}, "mps+cpu"

    def run(self, handle, document: Path):
        from docling.exceptions import ConversionError

        try:
            result = handle["converter"].convert(document)
        except ConversionError as exc:
            raise ValueError(f"could not parse the document: {exc}") from exc
        doc = result.document
        return {
            "markdown": join_cjk_lines(doc.export_to_markdown()),
            "pages": len(result.pages) or doc.num_pages(),
            "tables": len(doc.tables),
            "ocr": handle["ocr"],
        }


# A PDF line break inside Chinese text comes out as a space ("表示 不續約者"), which breaks
# keyword search; Chinese does not put spaces between characters, so drop exactly those in
# paragraphs (headings such as "第一條 合約標的" and table rows keep their spacing).
_CJK = r"[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef]"
_CJK_GAP = re.compile(rf"(?<={_CJK}) (?={_CJK})")


def join_cjk_lines(markdown: str) -> str:
    return "\n".join(
        line if line.lstrip().startswith(("#", "|")) else _CJK_GAP.sub("", line)
        for line in markdown.split("\n")
    )


def ocr_options():
    """(do_ocr, options, engine name): macOS Vision via ocrmac (reads Traditional Chinese)."""
    from docling.datamodel.pipeline_options import OcrMacOptions

    try:
        import ocrmac  # noqa: F401
    except ImportError:
        return False, OcrMacOptions(), "off"
    return True, OcrMacOptions(lang=["zh-Hant", "en-US"]), "ocrmac"


class CrossEncoderBackend(Backend):
    """Reranker: scores (query, passage) pairs with a sentence-transformers CrossEncoder."""

    task = "rerank"

    def load(self, spec, path):
        import torch
        from sentence_transformers import CrossEncoder

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model = CrossEncoder(
            str(path),
            device=device,
            max_length=spec.options.get("max_length", 1024),
            local_files_only=True,
            # Half precision halves memory on the GPU with no visible change in ranking.
            model_kwargs={"dtype": torch.float16} if device == "mps" else None,
        )
        return model, device

    def run(self, handle, query: str, documents: list[str]):
        scores = handle.predict(
            [(query, d) for d in documents], batch_size=16, show_progress_bar=False
        )
        return {"scores": [float(s) for s in scores]}


BACKENDS = {
    "mlx-whisper": WhisperBackend(),
    "docling": DoclingBackend(),
    "cross-encoder": CrossEncoderBackend(),
}


def free_memory() -> None:
    """Return freed model memory to macOS (after the last reference was dropped)."""
    gc.collect()
    if "torch" in sys.modules:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    if "mlx.core" in sys.modules:
        import mlx.core as mx

        mx.clear_cache()
