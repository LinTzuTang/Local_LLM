"""Rebuild the demo files in data/lab/samples/ from their source text (macOS only).

    uv run python scripts/make_samples.py

meeting_zh.txt -> meeting_zh.wav: each "[voice] line" is spoken with macOS `say`, then ffmpeg
joins the lines with short pauses into 16 kHz mono WAV (what speech-to-text models expect).
supplier_contract.html -> supplier_contract.pdf: rendered by Chrome through Playwright.
"""

import re
import subprocess
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

SAMPLES = Path(__file__).resolve().parents[1] / "data" / "lab" / "samples"
PAUSE_S = 0.6
RATE = 205  # words per minute; macOS default is about 175


def make_meeting_audio() -> Path:
    lines = [
        re.match(r"\[(.+?)\]\s*(.+)", line).groups()
        for line in (SAMPLES / "meeting_zh.txt").read_text().splitlines()
        if line.startswith("[")
    ]
    out = SAMPLES / "meeting_zh.wav"
    with tempfile.TemporaryDirectory() as tmp:
        parts = []
        for i, (voice, text) in enumerate(lines):
            part = Path(tmp) / f"{i:02d}.aiff"
            subprocess.run(["say", "-v", voice, "-r", str(RATE), "-o", str(part), text], check=True)
            parts.append(part)
        inputs = [arg for p in parts for arg in ("-i", str(p))]
        # Resample each line, add a pause after it, then concatenate them in order.
        chains = [
            f"[{i}:a]aresample=16000,aformat=channel_layouts=mono,apad=pad_dur={PAUSE_S}[a{i}]"
            for i in range(len(parts))
        ]
        joined = "".join(f"[a{i}]" for i in range(len(parts)))
        graph = ";".join(chains) + f";{joined}concat=n={len(parts)}:v=0:a=1[out]"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", graph,
             "-map", "[out]", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(out)],
            check=True,
        )  # fmt: skip
    return out


def make_contract_pdf() -> Path:
    out = SAMPLES / "supplier_contract.pdf"
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page()
        page.goto((SAMPLES / "supplier_contract.html").as_uri())
        page.pdf(path=str(out), format="A4", prefer_css_page_size=True, print_background=True)
        browser.close()
    return out


if __name__ == "__main__":
    for path in (make_meeting_audio(), make_contract_pdf()):
        print(f"wrote {path} ({path.stat().st_size / 1e3:.0f} kB)")
