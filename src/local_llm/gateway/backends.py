"""How each kind of model is loaded, run and released.

Heavy libraries are imported inside `load`/`run`, so the gateway starts fast and only pays for
what is actually used. `load` returns (handle, device); `run` gets the handle plus the request
inputs and returns plain JSON-able data. Bad input raises ValueError (HTTP 400).
"""

import gc
import re
import sys
import threading
import time
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


# MLX runs one GPU graph at a time per process; two speech models transcribing at once (a
# scenario with parallel branches) take turns instead of risking concurrent Metal work.
MLX_LOCK = threading.Lock()
SAMPLE_RATE = 16000


class mlx_turn:
    """`with mlx_turn() as turn:` holds MLX_LOCK; turn.waited is the time spent queuing,
    reported as queue_seconds so a parallel run does not look like a slow model."""

    def __enter__(self):
        start = time.perf_counter()
        MLX_LOCK.acquire()
        self.waited = round(time.perf_counter() - start, 2)
        return self

    def __exit__(self, *exc):
        MLX_LOCK.release()


def decode_audio(path: Path):
    """Any audio/video file -> 16 kHz mono float32 samples (ffmpeg, through mlx-whisper)."""
    from mlx_whisper.audio import load_audio

    try:
        return load_audio(str(path))
    except RuntimeError as exc:
        raise ValueError(f"could not decode the audio file: {exc}") from exc


def _segments(raw, text: str, duration: float) -> list[dict]:
    """Timed segments when the model gives them, else one segment for the whole file."""
    rows = [s for s in raw or [] if isinstance(s, dict) and "start" in s and "end" in s]
    if not rows:
        return [{"start": 0.0, "end": round(duration, 2), "text": text}] if text else []
    return [
        {"start": round(float(s["start"]), 2), "end": round(float(s["end"]), 2),
         "text": str(s.get("text", "")).strip()}
        for s in rows
    ]  # fmt: skip


class WhisperBackend(Backend):
    """Speech-to-text with mlx-whisper on the Apple GPU.

    The catalog's `options.decode` is passed to mlx_whisper.transcribe, e.g.
    `{"condition_on_previous_text": false}`, which stops Whisper large-v3(-turbo) from
    looping on long real recordings. task="translate" is Whisper's built-in speech -> English
    translation (mode="translate"); models that lost it in fine-tuning set `"translate": false`
    (large-v3-turbo; Breeze-ASR-25 and Taiwan-Tongues still write Chinese when asked)."""

    task = "asr"

    def load(self, spec, path):
        import mlx.core as mx
        from mlx_whisper.load_models import load_model

        model = load_model(str(path), dtype=mx.float16)
        handle = {"model": model, "path": str(path), "decode": spec.options.get("decode", {}),
                  "translate": spec.options.get("translate")}  # fmt: skip
        return handle, "mlx"

    def run(
        self,
        handle,
        audio: Path,
        language: str | None = None,
        prompt: str | None = None,
        mode: str = "transcribe",
    ):
        import mlx_whisper
        from mlx_whisper.transcribe import ModelHolder

        if mode == "translate" and handle.get("translate") is False:
            raise ValueError(
                "this model does not translate (it writes the original language); use "
                "whisper-large-v3, an audio LLM, or transcribe and translate with an LLM"
            )
        samples = decode_audio(audio)  # ffmpeg: any audio/video format, 16 kHz mono
        # mlx-whisper keeps "the" model in a module-level holder, so transcriptions take turns.
        with mlx_turn() as turn:
            ModelHolder.model, ModelHolder.model_path = handle["model"], handle["path"]
            result = mlx_whisper.transcribe(
                samples,
                path_or_hf_repo=handle["path"],
                language=language,
                initial_prompt=prompt,
                verbose=None,
                task=mode,  # "transcribe" or "translate"
                **handle.get("decode", {}),
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
            "queue_seconds": turn.waited,
        }

    def release(self, handle):
        from mlx_whisper.transcribe import ModelHolder

        with MLX_LOCK:
            if ModelHolder.model is handle["model"]:
                ModelHolder.model = ModelHolder.model_path = None


# Qwen3-ASR names its languages in English ("Chinese"); the API takes ISO codes like Whisper.
LANGUAGE_NAMES = {
    "zh": "Chinese", "en": "English", "yue": "Cantonese", "ja": "Japanese", "ko": "Korean",
    "de": "German", "fr": "French", "es": "Spanish", "it": "Italian", "pt": "Portuguese",
    "ru": "Russian", "ar": "Arabic", "vi": "Vietnamese", "th": "Thai", "id": "Indonesian",
    "ms": "Malay", "hi": "Hindi", "tr": "Turkish", "nl": "Dutch", "sv": "Swedish",
    "da": "Danish", "fi": "Finnish", "pl": "Polish", "cs": "Czech", "el": "Greek",
    "hu": "Hungarian", "ro": "Romanian", "fa": "Persian", "fil": "Filipino",
}  # fmt: skip


def mlx_audio_kwargs(options: dict, language: str | None, prompt: str | None) -> dict:
    """generate() keyword arguments for an mlx-audio STT model, from its catalog options:
    `generate` (fixed kwargs), `language_names` (pass "Chinese" instead of "zh") and
    `prompt_arg` (the model's context / hotword field, if it has one)."""
    kwargs = dict(options.get("generate", {}))
    if language:
        names = options.get("language_names")
        kwargs["language"] = LANGUAGE_NAMES.get(language, language) if names else language
    if prompt and options.get("prompt_arg"):
        kwargs[options["prompt_arg"]] = prompt
    return kwargs


class MlxAudioBackend(Backend):
    """Speech-to-text with mlx-audio (Qwen3-ASR, SenseVoice, Fun-ASR-Nano, ...) on the GPU."""

    task = "asr"

    def load(self, spec, path):
        from mlx_audio.stt.utils import load_model

        # A Path (not a str) is never resolved against the Hugging Face Hub.
        return {"model": load_model(Path(path)), "options": spec.options}, "mlx"

    def run(
        self,
        handle,
        audio: Path,
        language: str | None = None,
        prompt: str | None = None,
        mode: str = "transcribe",
    ):
        import mlx.core as mx

        if mode == "translate":
            raise ValueError("this model only transcribes; translate its transcript with an LLM")
        samples = decode_audio(audio)
        duration = len(samples) / SAMPLE_RATE
        kwargs = mlx_audio_kwargs(handle["options"], language, prompt)
        with mlx_turn() as turn:
            result = handle["model"].generate(mx.array(samples), **kwargs)
        text = (result.text or "").strip()
        return {
            "text": text,
            "segments": _segments(result.segments, text, duration),
            "language": result.language if isinstance(result.language, str) else language,
            "duration": round(duration, 2),
            "queue_seconds": turn.waited,
        }


# Gemma 4 and other audio LLMs follow an instruction instead of a decoding API.
INSTRUCTION = (
    "Transcribe this audio verbatim. Output only the transcript: no title, no timestamps, "
    "no speaker labels, no commentary and no translation."
)
LANGUAGE_HINTS = {
    "zh": "The speech is Mandarin Chinese from Taiwan; write it in Traditional Chinese "
    "characters with full-width punctuation.",
    "en": "The speech is English.",
}


TRANSLATE_INSTRUCTION = (
    "Translate this speech into English. Output only the English translation: no title, no "
    "timestamps, no speaker labels, no commentary and no transcript of the original language."
)


def audio_instruction(
    options: dict, language: str | None, prompt: str | None, mode: str = "transcribe"
) -> str:
    if mode == "translate":
        parts = [options.get("translate_instruction", TRANSLATE_INSTRUCTION)]
        if language and language != "en":
            parts.append(f"The speech language code is '{language}'.")
        if prompt:
            parts.append(f"Context that may help with names and terms: {prompt}")
        return " ".join(parts)
    parts = [options.get("instruction", INSTRUCTION)]
    if language:
        parts.append(LANGUAGE_HINTS.get(language, f"The speech language code is '{language}'."))
    if prompt:
        parts.append(f"Context that may help with names and terms: {prompt}")
    return " ".join(parts)


def split_points(samples, max_seconds: float, rate: int = SAMPLE_RATE) -> list[int]:
    """Chunk boundaries (sample indices, ending with len) no longer than max_seconds, each cut
    placed at the quietest 50 ms frame of the chunk's last quarter, so words are not split."""
    import numpy as np

    total, size = len(samples), int(max_seconds * rate)
    cuts, start, frame = [], 0, rate // 20
    while total - start > size:
        lo, hi = start + size * 3 // 4, start + size
        window = np.asarray(samples[lo:hi], dtype=np.float32)
        frames = len(window) // frame
        if frames:
            energy = (window[: frames * frame].reshape(frames, frame) ** 2).mean(axis=1)
            cut = lo + int(energy.argmin()) * frame + frame // 2
        else:
            cut = hi
        cuts.append(cut)
        start = cut
    return [*cuts, total]


class OllamaAudioBackend(Backend):
    """An audio-capable LLM in the local Ollama (Gemma 4) used as a speech recogniser.

    Ollama loads and unloads the weights; the gateway only keeps the tag, so "loaded" here
    means "used recently" and unloading asks Ollama to free the model."""

    task = "asr"

    def load(self, spec, path):
        from . import ollama

        try:
            caps = ollama.capabilities(spec.ollama)
        except Exception as exc:
            raise RuntimeError(f"Ollama has no model {spec.ollama}: {exc}") from exc
        if "audio" not in caps:
            raise RuntimeError(f"Ollama does not accept audio input for {spec.ollama}")
        return {"tag": spec.ollama, "options": spec.options}, "ollama"

    def run(
        self,
        handle,
        audio: Path,
        language: str | None = None,
        prompt: str | None = None,
        mode: str = "transcribe",
    ):
        from . import ollama

        samples = decode_audio(audio)
        options = handle["options"]
        instruction = audio_instruction(options, language, prompt, mode)
        bounds = split_points(samples, options.get("chunk_seconds", 240))
        # About 5 characters per second of speech in Chinese; allow plenty of headroom.
        per_second = options.get("max_tokens_per_second", 12)
        segments, ollama_load, start = [], 0.0, 0
        for end in bounds:
            max_tokens = int(64 + per_second * (end - start) / SAMPLE_RATE)
            reply = ollama.chat_audio(handle["tag"], instruction,
                                      ollama.wav_bytes(samples[start:end]), max_tokens)  # fmt: skip
            text = (reply.get("message", {}).get("content") or "").strip()
            ollama_load += (reply.get("load_duration") or 0) / 1e9
            segment = {"start": round(start / SAMPLE_RATE, 2), "end": round(end / SAMPLE_RATE, 2),
                       "text": text}  # fmt: skip
            if reply.get("done_reason") == "length":
                segment["truncated"] = True  # hit max_tokens: usually a repetition loop
            segments.append(segment)
            start = end
        joiner = "" if language in ("zh", "ja", "yue") and mode != "translate" else " "
        return {
            "text": joiner.join(s["text"] for s in segments if s["text"]),
            "segments": segments,
            "language": language,
            "duration": round(len(samples) / SAMPLE_RATE, 2),
            "ollama_load_seconds": round(ollama_load, 2),
            "truncated": any(s.get("truncated") for s in segments),
        }

    def release(self, handle):
        from . import ollama

        try:
            ollama.unload(handle["tag"])
        except Exception:  # Ollama stopped: nothing left to free
            pass


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


class SherpaDiarizationBackend(Backend):
    """Who spoke when: sherpa-onnx speaker diarization on the CPU (onnxruntime).

    pyannote segmentation-3.0 (ONNX) finds speech and speaker turns in 10 s windows, a speaker
    embedding model (3D-Speaker CAM++) describes each turn, and the turns are clustered into
    speakers: into `num_speakers` clusters when the caller knows the count, else by a cosine
    distance `threshold` (catalog option; lower = more speakers)."""

    task = "diarize"

    def load(self, spec, path):
        import os

        import sherpa_onnx

        options = spec.options
        threads = max(1, min(8, (os.cpu_count() or 4) // 2))
        segmentation = sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(Path(path) / options["segmentation"])
            ),
            num_threads=threads,
        )
        embedding = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(Path(path) / options["embedding"]), num_threads=threads
        )

        def config(num_speakers: int | None, threshold: float | None):
            clustering = sherpa_onnx.FastClusteringConfig(
                num_clusters=num_speakers or -1,
                threshold=threshold or options.get("threshold", 0.5),
            )
            return sherpa_onnx.OfflineSpeakerDiarizationConfig(
                segmentation=segmentation,
                embedding=embedding,
                clustering=clustering,
                min_duration_on=options.get("min_duration_on", 0.3),
                min_duration_off=options.get("min_duration_off", 0.5),
            )

        first = config(None, None)
        if not first.validate():
            raise RuntimeError("diarization model files are missing or invalid")
        return {"model": sherpa_onnx.OfflineSpeakerDiarization(first), "config": config}, "cpu"

    def run(
        self,
        handle,
        audio: Path,
        num_speakers: int | None = None,
        threshold: float | None = None,
    ):
        import numpy as np

        if num_speakers is not None and not 1 <= num_speakers <= 20:
            raise ValueError("num_speakers must be between 1 and 20")
        samples = np.ascontiguousarray(decode_audio(audio), dtype=np.float32)
        model = handle["model"]
        model.set_config(handle["config"](num_speakers, threshold))
        result = model.process(samples).sort_by_start_time()
        order: dict[int, int] = {}  # cluster id -> 0, 1, ... in order of first appearance
        segments = [
            {"start": round(s.start, 2), "end": round(s.end, 2),
             "speaker": order.setdefault(int(s.speaker), len(order))}
            for s in result
        ]  # fmt: skip
        return {
            "segments": segments,
            "speakers": len({s["speaker"] for s in segments}),
            "duration": round(len(samples) / SAMPLE_RATE, 2),
        }


BACKENDS = {
    "mlx-whisper": WhisperBackend(),
    "mlx-audio": MlxAudioBackend(),
    "ollama-audio": OllamaAudioBackend(),
    "docling": DoclingBackend(),
    "cross-encoder": CrossEncoderBackend(),
    "sherpa-diarization": SherpaDiarizationBackend(),
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
