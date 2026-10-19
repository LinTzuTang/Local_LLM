"""Pure helpers for live captions and the voice-chat assistant on the speech page.

Live captions are simulated from a recording: the audio is cut into short chunks
(speech_translate.chunk_bounds), each chunk is transcribed on its own, and
speech_translate.live_schedule turns the measured processing times into when each chunk's text
would appear for a live listener. `latency_stats` summarises that, and `caption_at` gives the
caption a viewer would see at a given moment of the talk (for replaying the run).

The voice chat is a cascade: speech recognition -> a chat LLM -> text to speech. `messages`
builds the LLM conversation from the earlier turns, and `speakable` strips what should not be
read aloud (Markdown, emoji, thinking blocks).
"""

import re
import statistics

from local_llm.diarization import join_text


def join_texts(texts: list[str]) -> str:
    """Chunk transcripts joined into one caption: no space between Chinese characters, one
    between words."""
    out = ""
    for t in texts:
        out = join_text(out, (t or "").strip())
    return out


def latency_stats(chunks: list[dict]) -> dict:
    """Summary of a live run's chunks ({"end", "compute", "finish", "lag"} each): when the
    first text appeared, the mean / median / worst lag behind the speaker, the lag after the
    last chunk, and whether the model kept up (processing faster than the audio arrives)."""
    if not chunks:
        return {}
    lags = [c["lag"] for c in chunks]
    audio = chunks[-1]["end"] - chunks[0].get("start", 0.0)
    compute = sum(c["compute"] for c in chunks)
    return {
        "first_s": round(chunks[0]["finish"], 2),
        "mean_lag": round(statistics.fmean(lags), 2),
        "median_lag": round(statistics.median(lags), 2),
        "max_lag": round(max(lags), 2),
        "lag_end": round(lags[-1], 2),
        "compute_s": round(compute, 2),
        "rtf": round(compute / audio, 4) if audio > 0 else None,
        "keeps_up": compute <= audio,
    }


def caption_at(chunks: list[dict], t: float) -> str:
    """The caption shown `t` seconds after the speaker started: every chunk whose text had
    come back (finish <= t), in order."""
    return join_texts([c.get("text", "") for c in chunks if c.get("finish", 0) <= t])


# -- voice chat -------------------------------------------------------------------------------


def system_prompt(language: str | None = "zh") -> str:
    """Instructions for a spoken assistant: short, plain sentences that sound right read
    aloud, in the language of the question (Traditional Chinese for Chinese)."""
    return (
        "You are a helpful voice assistant running fully on this computer. Your reply will be "
        "read aloud by a speech synthesiser, so answer in one to three short sentences of "
        "plain spoken text: no Markdown, lists, headings, tables, code, URLs or emoji. Reply "
        "in the language the user speaks; for Chinese, use Traditional Chinese as used in "
        "Taiwan. If the question was misheard and makes no sense, ask the user to repeat it."
        + {
            "zh": " The user is speaking Chinese: reply in Traditional Chinese.",
            "en": " The user is speaking English: reply in English, even if earlier turns "
            "were in Chinese.",
        }.get(language or "", "")
    )


def messages(history: list[dict], question: str, system: str, max_turns: int = 4) -> list[dict]:
    """The chat for the LLM: system prompt, the last `max_turns` turns of `history`
    ({"question", "answer"} each, oldest first) and the new question."""
    out = [{"role": "system", "content": system}]
    for turn in history[-max_turns:] if max_turns else []:
        if turn.get("question") and turn.get("answer"):
            out.append({"role": "user", "content": turn["question"]})
            out.append({"role": "assistant", "content": turn["answer"]})
    out.append({"role": "user", "content": question})
    return out


_THINK = re.compile(r"<think>.*?</think>", re.S)
_EMOJI = re.compile("[\U0001f000-\U0001faff☀-➿️‍]")


def speakable(text: str) -> str:
    """Text to hand to the speech synthesiser: thinking blocks, Markdown markup, links and
    emoji removed, whitespace collapsed."""
    text = _THINK.sub("", text or "")
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)  # [label](url) -> label
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"^\s{0,3}(#{1,6}|[-*+]|\d+[.)])\s+", "", text, flags=re.M)
    text = re.sub(r"[*_`~|>#]", "", text)
    text = _EMOJI.sub("", text)
    return re.sub(r"\s+", " ", text).strip()
