"""Scoring speech-recognition output: character error rate (CER) for Chinese.

Chinese has no spaces between words, so ASR is scored per character: CER = (substitutions +
deletions + insertions) / reference length, from a Levenshtein alignment. Before aligning,
both texts are normalised so formatting does not count as an error: punctuation and spaces
are removed, full-width letters and digits become half-width, Latin letters are lowercased,
Simplified characters are converted to Traditional (OpenCC `s2tw`), because Whisper,
Qwen3-ASR and SenseVoice write Simplified while Breeze-ASR-25 and the reference are in
Traditional, and numbers written with digits are spelled out in Chinese ("87%" ->
"百分之八十七", "250" -> "二百五十"; 兩 counts as 二), because models with inverse text
normalisation write digits where the speaker said words.
"""

import re
import unicodedata
from functools import cache
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
SAMPLES_DIR = PROJECT_DIR / "data" / "lab" / "samples"


@cache
def _to_traditional():
    import opencc

    return opencc.OpenCC("s2tw")


DIGITS = "零一二三四五六七八九"
_NUMBER = re.compile(r"(\d+)(?:\.(\d+))?(%?)")


def spell_number(n: int) -> str:
    """Chinese reading of a non-negative integer below 10^8 (15 -> 十五, 105 -> 一百零五)."""
    if n == 0:
        return DIGITS[0]
    if n >= 10**4:
        high, low = divmod(n, 10**4)
        rest = spell_number(low) if low else ""
        return spell_number(high) + "萬" + (DIGITS[0] if 0 < low < 1000 else "") + rest
    out, zero = "", False
    for value, unit in ((1000, "千"), (100, "百"), (10, "十"), (1, "")):
        d = n // value % 10
        if d:
            out += (DIGITS[0] if zero and out else "") + DIGITS[d] + unit
            zero = False
        else:
            zero = bool(out)
    return out[1:] if out.startswith("一十") else out


def spell_numbers(text: str) -> str:
    """Write the digit numbers in a text as Chinese words; long digit strings (years, phone
    numbers) are read digit by digit."""

    def word(match: re.Match) -> str:
        whole, frac, percent = match.groups()
        if len(whole) > 8 or (len(whole) > 1 and whole.startswith("0")):
            spoken = "".join(DIGITS[int(d)] for d in whole)
        elif len(whole) == 4 and text[match.end() : match.end() + 1] == "年":
            spoken = "".join(DIGITS[int(d)] for d in whole)
        else:
            spoken = spell_number(int(whole))
        if frac:
            spoken += "點" + "".join(DIGITS[int(d)] for d in frac)
        return ("百分之" + spoken) if percent else spoken

    return _NUMBER.sub(word, text)


def normalize(text: str, canonical: bool = True) -> str:
    """Letters, digits and CJK characters only; `canonical` also unifies the script and
    spells out numbers (see the module docstring)."""
    text = unicodedata.normalize("NFKC", text or "").lower()
    if canonical:
        text = _to_traditional().convert(spell_numbers(text)).replace("兩", "二")
    return "".join(ch for ch in text if unicodedata.category(ch)[0] in "LN")


def edit_ops(ref: str, hyp: str) -> dict:
    """Levenshtein counts {"sub", "del", "ins"} turning ref into hyp."""
    # One row of (cost, sub, del, ins) per hyp prefix; O(len(ref) * len(hyp)) time.
    prev = [(j, 0, 0, j) for j in range(len(hyp) + 1)]
    for i, r in enumerate(ref, 1):
        row = [(i, 0, i, 0)]
        for j, h in enumerate(hyp, 1):
            diag = prev[j - 1]
            if r == h:
                best = diag
            else:
                best = min(
                    (diag[0] + 1, diag[1] + 1, diag[2], diag[3]),
                    (prev[j][0] + 1, prev[j][1], prev[j][2] + 1, prev[j][3]),
                    (row[j - 1][0] + 1, row[j - 1][1], row[j - 1][2], row[j - 1][3] + 1),
                )
            row.append(best)
        prev = row
    _, sub, dele, ins = prev[-1]
    return {"sub": sub, "del": dele, "ins": ins}


def cer(reference: str, hypothesis: str, canonical: bool = True) -> dict | None:
    """{"cer", "errors", "ref_chars", "sub", "del", "ins"}, or None without a reference."""
    ref, hyp = normalize(reference, canonical), normalize(hypothesis, canonical)
    if not ref:
        return None
    ops = edit_ops(ref, hyp)
    errors = ops["sub"] + ops["del"] + ops["ins"]
    return {"cer": round(errors / len(ref), 4), "errors": errors, "ref_chars": len(ref), **ops}


def script(text: str) -> str | None:
    """The script of Chinese text, "traditional" or "simplified" (by how much OpenCC s2tw
    changes it); None when there are hardly any Chinese characters."""
    cjk = [ch for ch in text or "" if "\u4e00" <= ch <= "\u9fff"]
    if len(cjk) < 10:
        return None
    converted = _to_traditional().convert("".join(cjk))
    changed = sum(a != b for a, b in zip(cjk, converted, strict=False))
    return "simplified" if changed / len(cjk) > 0.03 else "traditional"


def alignment(reference: str, hypothesis: str) -> list[tuple[str, str, str]]:
    """(op, reference part, hypothesis part) over the normalised texts, op being "equal",
    "replace", "delete" (missed by the model) or "insert" (added by the model); for showing
    where a transcript differs from the reference."""
    import difflib

    ref, hyp = normalize(reference), normalize(hypothesis)
    matcher = difflib.SequenceMatcher(None, ref, hyp, autojunk=False)
    return [(op, ref[a:b], hyp[c:d]) for op, a, b, c, d in matcher.get_opcodes()]


_SPEAKER = re.compile(r"^\s*\[[^\]]*\]\s*")


def reference_text(source: str) -> str:
    """The spoken text of a sample script: drops `#` comment lines and `[voice]` prefixes
    (the format of data/lab/samples/meeting_zh.txt, which scripts/make_samples.py reads)."""
    lines = [_SPEAKER.sub("", line).strip() for line in source.splitlines()]
    return "\n".join(line for line in lines if line and not line.startswith("#"))


def sample_reference(audio: Path) -> str | None:
    """Reference transcript for a bundled sample (meeting_zh.wav -> meeting_zh.txt)."""
    script = Path(audio).with_suffix(".txt")
    if script.parent.resolve() != SAMPLES_DIR.resolve() or not script.exists():
        return None
    return reference_text(script.read_text(encoding="utf-8"))
