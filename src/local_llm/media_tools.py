"""Agent tools for the inbox (storage/inbox/): list files, transcribe audio, parse documents.

Users drop meeting recordings and documents into the inbox (the "語音與文件" page has an upload
box); agents only see file names and call these tools. Transcription and parsing run in the
local model gateway (127.0.0.1:11500), so the files never leave this Mac.

Inbox names are plain file names: no directories, no hidden files, nothing that resolves
outside the inbox (checked again on every read).
"""

import re
import shutil
import unicodedata
from datetime import datetime
from pathlib import Path

import httpx
from langchain.tools import tool

from . import gateway_client

PROJECT_DIR = Path(__file__).resolve().parents[2]
INBOX_DIR = PROJECT_DIR / "storage" / "inbox"
SAMPLES_DIR = PROJECT_DIR / "data" / "lab" / "samples"
SAMPLE_FILES = ("meeting_zh.wav", "supplier_contract.pdf")
AUDIO_TYPES = {".wav", ".mp3", ".m4a", ".mp4", ".aac", ".flac", ".ogg", ".opus", ".webm", ".mov"}
DOCUMENT_TYPES = {".pdf", ".docx", ".pptx", ".html", ".htm", ".md", ".png", ".jpg", ".jpeg"}
ASR_MODEL = "breeze-asr-25"  # Taiwanese Mandarin, Traditional Chinese output
MAX_TOOL_CHARS = 12_000  # about 8k tokens of Chinese: room for the answer in a 16k context
_cache: dict[tuple, str] = {}  # (tool, name, size, mtime) -> result; files rarely change


def safe_name(name: str) -> str:
    """A plain, harmless file name for the inbox (raises ValueError if nothing is left)."""
    name = unicodedata.normalize("NFC", name.replace("\\", "/").split("/")[-1]).strip()
    name = re.sub(r"[^\w.\-()（）\[\] ]+", "_", name).lstrip(". ")[:120]
    if not name or not Path(name).stem:
        raise ValueError("invalid file name")
    return name


def inbox_path(name: str) -> Path:
    """Path of an existing inbox file; refuses directories, hidden files and anything outside."""
    if not name or name != Path(name).name or name.startswith(".") or "\\" in name:
        raise ValueError(f"'{name}' is not a plain inbox file name; use list_inbox_files")
    path = (INBOX_DIR / name).resolve()
    if path.parent != INBOX_DIR.resolve() or not path.is_file():
        raise FileNotFoundError(f"'{name}' is not in the inbox; use list_inbox_files")
    return path


def kind(name: str) -> str:
    suffix = Path(name).suffix.lower()
    return "audio" if suffix in AUDIO_TYPES else "document" if suffix in DOCUMENT_TYPES else "other"


def add_samples() -> list[str]:
    """Copy the demo meeting recording and supplier contract into the inbox."""
    INBOX_DIR.mkdir(parents=True, exist_ok=True)
    for name in SAMPLE_FILES:
        shutil.copy2(SAMPLES_DIR / name, INBOX_DIR / name)
    return list(SAMPLE_FILES)


def list_files() -> list[dict]:
    if not INBOX_DIR.exists():
        add_samples()  # first use: start with the demo files so the example prompts work
    rows = []
    for p in sorted(INBOX_DIR.iterdir()):
        if p.is_file() and not p.is_symlink() and not p.name.startswith("."):
            stat = p.stat()
            modified = datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="minutes")
            size_kb = round(stat.st_size / 1024)
            rows.append({"name": p.name, "kind": kind(p.name), "size_kb": size_kb,
                         "modified": modified})  # fmt: skip
    return rows


def save(name: str, data: bytes) -> str:
    """Store an upload in the inbox under a safe name (overwrites a file of the same name)."""
    name = safe_name(name)
    INBOX_DIR.mkdir(parents=True, exist_ok=True)
    (INBOX_DIR / name).write_bytes(data)
    return name


def remove(name: str) -> None:
    inbox_path(name).unlink()


def _truncate(text: str, hint: str) -> str:
    if len(text) <= MAX_TOOL_CHARS:
        return text
    return (
        text[:MAX_TOOL_CHARS]
        + f"\n\n[Truncated: showing the first {MAX_TOOL_CHARS:,} of {len(text):,} characters. "
        + hint
        + "]"
    )


def _cached(tool_name: str, path: Path, compute) -> str:
    stat = path.stat()
    key = (tool_name, path.name, stat.st_size, stat.st_mtime_ns)
    if key not in _cache:
        _cache[key] = compute()
    return _cache[key]


def _gateway_problem(exc: Exception) -> str:
    if isinstance(exc, httpx.ConnectError):
        return ("ERROR: the local model gateway is not running. Ask the user to start it with "
                "./scripts/start_gateway.sh (or ./scripts/tmux_up.sh).")  # fmt: skip
    return f"ERROR: {exc}"


def _clock(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"


@tool
def list_inbox_files() -> str:
    """List the files users put in the inbox (meeting recordings, contracts, reports...) with
    their type and size. Call this first to get exact file names."""
    rows = list_files()
    if not rows:
        return "The inbox is empty. Ask the user to upload files on the Speech & documents page."
    return "\n".join(
        f"- {r['name']} ({r['kind']}, {r['size_kb']} KB, {r['modified']})" for r in rows
    )


@tool
def transcribe_audio(filename: str) -> str:
    """Transcribe an audio file from the inbox (e.g. a meeting recording) into text with
    timestamps, using a local speech-to-text model. `filename` must be an exact name from
    list_inbox_files."""
    try:
        path = inbox_path(filename)
    except (ValueError, FileNotFoundError) as exc:
        return f"ERROR: {exc}"
    if kind(path.name) != "audio":
        return f"ERROR: {filename} is not an audio file; use parse_document for documents."

    def compute() -> str:
        result = gateway_client.transcribe(path, model=ASR_MODEL)
        lines = [
            f"[{_clock(s['start'])}-{_clock(s['end'])}] {s['text']}" for s in result["segments"]
        ]
        header = (f"Transcript of {path.name} ({result['duration']:.0f} s of audio, "
                  f"model {result['model']}, language {result['language']}):")  # fmt: skip
        return header + "\n" + "\n".join(lines)

    try:
        text = _cached("transcribe", path, compute)
    except (httpx.HTTPError, gateway_client.GatewayError) as exc:
        return _gateway_problem(exc)
    return _truncate(text, "The rest of the recording was cut; summarise what is shown.")


@tool
def parse_document(filename: str) -> str:
    """Read a document from the inbox (PDF, Word, PowerPoint, HTML, Markdown or an image) and
    return its content as Markdown, tables included. `filename` must be an exact name from
    list_inbox_files."""
    try:
        path = inbox_path(filename)
    except (ValueError, FileNotFoundError) as exc:
        return f"ERROR: {exc}"
    if kind(path.name) != "document":
        return f"ERROR: {filename} is not a supported document; use transcribe_audio for audio."

    def compute() -> str:
        result = gateway_client.parse_document(path)
        return (f"Content of {path.name} ({result['pages']} pages, {result['tables']} tables):\n\n"
                + result["markdown"])  # fmt: skip

    try:
        text = _cached("parse", path, compute)
    except (httpx.HTTPError, gateway_client.GatewayError) as exc:
        return _gateway_problem(exc)
    return _truncate(text, "Tell the user only the first part of the document was read.")


TOOLS = [list_inbox_files, transcribe_audio, parse_document]
# What each tool really does, for agents.ToolInfo (kept here so this module does not import
# agents.py, which imports this one).
TOOL_TEXT = {
    "list_inbox_files": {
        "runs": "本機 Python 函式：列出收件匣裡的檔案",
        "reads": "storage/inbox/（唯讀）",
        "runs_en": "Local Python: lists the files in the inbox",
        "reads_en": "storage/inbox/ (read-only)",
    },
    "transcribe_audio": {
        "runs": "本機模型閘道：POST 127.0.0.1:11500/v1/audio/transcriptions（Breeze-ASR-25，MLX）",
        "reads": "storage/inbox/ 的音訊檔（不離開這台電腦）",
        "runs_en": "Local model gateway: POST 127.0.0.1:11500/v1/audio/transcriptions "
        "(Breeze-ASR-25, MLX)",
        "reads_en": "Audio files in storage/inbox/ (never leave this Mac)",
    },
    "parse_document": {
        "runs": "本機模型閘道：POST 127.0.0.1:11500/v1/documents/parse（Docling）",
        "reads": "storage/inbox/ 的文件（不離開這台電腦）",
        "runs_en": "Local model gateway: POST 127.0.0.1:11500/v1/documents/parse (Docling)",
        "reads_en": "Documents in storage/inbox/ (never leave this Mac)",
    },
}
