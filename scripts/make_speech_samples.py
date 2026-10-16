"""Join the evaluation's test sentences into two longer sample recordings for the speech page.

    PYTHONPATH=src .venv/bin/python scripts/make_speech_samples.py [librispeech] [hokkien]

The speech evaluation (docs/SPEECH_EVALUATION.md) scored models on 20 short clips per
dataset. This script joins those 20 clips, in the evaluation's order, with 0.8 s of silence
between them, into one 16 kHz mono FLAC, and writes the reference transcript next to it (one
line per clip, `#` header lines with the source and licence):

- LibriSpeech test-clean (CC BY 4.0): data/lab/samples/librispeech_20_en.flac. The licence
  allows redistribution with attribution, so the file is in the repo.
- MOE 臺灣台語常用詞辭典 example sentences (CC BY-ND 3.0 TW): storage/speech_samples/
  hokkien_moe_20_nan.flac. NoDerivatives forbids sharing a joined (derived) file, so it is
  only built locally, in a gitignored folder; the reference is the dictionary's 漢字.

Inputs are the evaluation's manifests (storage/speech_eval/data/<dataset>.json), its result
files (storage/speech_eval/results/asr_<dataset>.json, for which clips were used) and the
downloaded audio under storage/speech_eval/raw/; `scripts/speech_eval/prepare_data.py
librispeech hokkien_moe` downloads them first.
"""

import json
import sys
from pathlib import Path

from local_llm import asr_metrics, audio_aug

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "storage" / "speech_eval"
GAP_S = 0.8

DATASETS = {
    "librispeech": {
        "dataset": "en_librispeech",
        "out": asr_metrics.SAMPLES_DIR / "librispeech_20_en.flac",
        "header": [
            "LibriSpeech test-clean: 20 utterances joined with 0.8 s of silence, 16 kHz mono.",
            "Source: LibriSpeech ASR corpus (Panayotov, Chen, Povey, Khudanpur, ICASSP 2015),",
            "https://www.openslr.org/12/ (read LibriVox audiobooks).",
            "Licence: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/); joined and",
            "resampled only, the words are unchanged. One line per utterance, in order:",
        ],
    },
    "hokkien": {
        "dataset": "hokkien_moe",
        "out": asr_metrics.LOCAL_SAMPLES_DIR / "hokkien_moe_20_nan.flac",
        "header": [
            "教育部《臺灣台語常用詞辭典》例句 20 句（台語），每句間隔 0.8 秒，16 kHz 單聲道。",
            "Source: Ministry of Education, Dictionary of Frequently-Used Taiwan Taigi,",
            "https://sutian.moe.edu.tw/ (example sentences and their recordings).",
            "Licence: CC BY-ND 3.0 TW. NoDerivatives: this joined file is built locally only",
            "(storage/, not in the repo) and must not be redistributed. Reference = 漢字.",
        ],
    },
}


def used_items(dataset: str) -> list[str]:
    """The clip ids the evaluation scored, in the order they first appear in its runs."""
    runs = json.loads((EVAL / "results" / f"asr_{dataset}.json").read_text())["runs"]
    out: list[str] = []
    for run in runs:
        if run["item"] not in out:
            out.append(run["item"])
    return out


def build(name: str) -> Path:
    spec = DATASETS[name]
    manifest = json.loads((EVAL / "data" / f"{spec['dataset']}.json").read_text())
    by_id = {m["id"]: m for m in manifest}
    items = [by_id[i] for i in used_items(spec["dataset"])]
    clips = [audio_aug.load(ROOT / m["audio"]) for m in items]
    lines, offset = [f"# {h}" for h in spec["header"]], 0.0
    for m, clip in zip(items, clips, strict=True):
        seconds = len(clip) / audio_aug.SR
        lines.append(f"# {offset:7.2f}-{offset + seconds:7.2f}s  {m['id']}  {m['source']}")
        offset += seconds + GAP_S
    lines += [m["ref"].strip() for m in items]
    out = audio_aug.save_flac(spec["out"], audio_aug.join(clips, GAP_S))
    out.with_suffix(".txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{out.relative_to(ROOT)}: {len(items)} clips, {offset - GAP_S:.1f} s")
    return out


if __name__ == "__main__":
    for name in sys.argv[1:] or list(DATASETS):
        build(name)
