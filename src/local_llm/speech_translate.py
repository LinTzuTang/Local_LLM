"""Speech translation helpers: LLM prompts, scoring and the live-input simulation.

Three ways to turn Chinese speech into English text are compared on the speech page:
cascade (a speech model's transcript translated by an LLM, `translation_prompt`), direct (an
audio LLM told to translate) and Whisper's own task="translate". `score` gives chrF and BLEU
against a reference translation (sacrebleu), and `chunk_bounds` / `live_schedule` simulate a
live speaker: the audio arrives in N-second chunks and each chunk can only be processed once
it has fully arrived and the previous one is done.
"""

import re

SAMPLE_RATE = 16000
# ISO code -> English name, for prompts.
LANGUAGES = {"zh": "Chinese", "en": "English", "ja": "Japanese", "ko": "Korean"}


def is_translategemma(model: str) -> bool:
    return model.split("/")[-1].startswith("translategemma")


def translation_prompt(
    text: str, model: str, source: str = "zh", target: str = "en", script: str | None = None
) -> str:
    """The user message that asks `model` to translate `text`.

    TranslateGemma expects its fixed template (ollama.com/library/translategemma): one user
    message, two blank lines before the text; Chinese is coded zh-TW (Traditional) or zh-Hans
    (Simplified) by the script the text is written in. Other chat models get a plain
    instruction that asks for the translation only."""
    src, tgt = LANGUAGES.get(source, source), LANGUAGES.get(target, target)
    if is_translategemma(model):
        src_code = source
        if source == "zh":
            src_code = "zh-Hans" if script == "simplified" else "zh-TW"
        return (
            f"You are a professional {src} ({src_code}) to {tgt} ({target}) translator. Your "
            f"goal is to accurately convey the meaning and nuances of the original {src} text "
            f"while adhering to {tgt} grammar, vocabulary, and cultural sensitivities.\n"
            f"Produce only the {tgt} translation, without any additional explanations or "
            f"commentary. Please translate the following {src} text into {tgt}:\n\n\n{text}"
        )
    return (
        f"Translate the following {src} text into {tgt}. It is a speech-recognition "
        f"transcript, so it may lack punctuation or contain recognition errors; translate the "
        f"intended meaning. Output only the {tgt} translation, with no notes, quotes or "
        f"explanations.\n\n{text}"
    )


_PREFIX = re.compile(r"^\s*(?:english\s+)?translation\s*:\s*", re.I)


def clean_output(text: str) -> str:
    """Strip a leading "Translation:" label and wrapping quotes some models add."""
    text = _PREFIX.sub("", text.strip())
    if len(text) > 1 and text[0] == text[-1] and text[0] in "\"'“”":
        text = text[1:-1].strip()
    return text


def score(hypothesis: str, reference: str) -> dict | None:
    """chrF (character n-gram F-score, robust for short texts) and BLEU, both 0-100, of one
    translation against one reference; None without a reference."""
    if not reference.strip() or not hypothesis.strip():
        return None
    import sacrebleu

    return {
        "chrf": round(sacrebleu.sentence_chrf(hypothesis, [reference]).score, 1),
        "bleu": round(sacrebleu.sentence_bleu(hypothesis, [reference]).score, 1),
    }


def chunk_bounds(
    n_samples: int, chunk_s: float, rate: int = SAMPLE_RATE, min_s: float = 1.0
) -> list[tuple[int, int]]:
    """(start, end) sample ranges of consecutive `chunk_s`-second chunks covering the audio,
    as a live speaker's audio would arrive. A last piece shorter than `min_s` is added to the
    chunk before it (too short to recognise on its own)."""
    if chunk_s <= 0:
        raise ValueError("chunk length must be positive")
    size = max(int(chunk_s * rate), 1)
    bounds = [(a, min(a + size, n_samples)) for a in range(0, n_samples, size)]
    if len(bounds) > 1 and bounds[-1][1] - bounds[-1][0] < min_s * rate:
        last = bounds.pop()
        bounds[-1] = (bounds[-1][0], last[1])
    return bounds


def live_schedule(chunk_ends: list[float], compute: list[float]) -> list[dict]:
    """When each chunk's text would appear for a live listener.

    Chunk i has fully arrived at `chunk_ends[i]` seconds after the speaker started; it is
    processed once it has arrived and the previous chunk is finished, taking `compute[i]`
    seconds. Returns per chunk {"arrival", "start", "finish", "lag"}, where lag = finish -
    arrival: how far the text trails the speaker. A lag that grows chunk after chunk means the
    model cannot keep up with real time."""
    rows, busy_until = [], 0.0
    for arrival, seconds in zip(chunk_ends, compute, strict=True):
        start = max(arrival, busy_until)
        finish = start + seconds
        busy_until = finish
        rows.append(
            {
                "arrival": round(arrival, 2),
                "start": round(start, 2),
                "finish": round(finish, 2),
                "lag": round(finish - arrival, 2),
            }
        )
    return rows
