"""A simulated user for the live voice assistant: an LLM plays the person talking to it.

    from local_llm import user_sim
    line = user_sim.next_line("你在內湖，想知道今天天氣……", [("今天天氣如何？", "你在哪裡？")])
    # {"text": "我在內湖。", "done": False, "seconds": 0.6}

A fixed test script says its next line whatever the assistant answered, so a conversation can
drift apart (the assistant asks something the script does not answer). Here the next line is
written from the conversation so far and a goal (data/lab/voice_scripts.json `goal`), one short
spoken sentence at a time, until the goal is met ("done") or the turns run out. The gateway
speaks it with the chosen voice into the assistant's microphone (gateway/voice.py, sim_next).
"""

import json
import re
import time

import httpx

OLLAMA_URL = "http://127.0.0.1:11434"
MODEL = "gemma4:12b"
SYSTEM = (
    "你在測試一個語音助理，扮演打電話給它的使用者。依照「你的目標」和到目前為止的對話，"
    "寫出你下一句要說的話：一句口語、20 字以內，像真人講話，不要旁白、不要括號、不要表情符號。"
    "回應助理剛才說的內容：它問你問題就回答；它說要查就等它，不要重複問；它給了答案就照目標往下走。"
    "目標都完成（通常是道謝道別）時，在最後一句設 done=true。"
    '只輸出 JSON：{"say": "下一句", "done": false}'
)
SYSTEM_EN = (
    "You are testing a voice assistant by playing the user who calls it. From 'your goal' and "
    "the conversation so far, write the next thing you say: one short spoken sentence (under "
    "15 words), natural, no narration, brackets or emoji. Respond to what the assistant just "
    "said: answer its questions; if it is looking something up, do not ask again; when it has "
    "answered, move on with the goal. When the goal is done (usually saying thanks and "
    'goodbye), set done=true on that last line. Output JSON only: {"say": "...", "done": false}'
)


def prompt(goal: str, history: list[tuple[str, str]], english: bool) -> str:
    lines = []
    for user, assistant in history:
        if user:
            lines.append(("You: " if english else "你：") + user)
        if assistant:
            lines.append(("Assistant: " if english else "助理：") + assistant)
    talk = "\n".join(lines) or (
        "(nothing yet: say the first line)" if english else "（還沒開始，說第一句）"
    )
    if english:
        return f"Your goal: {goal}\n\nConversation so far:\n{talk}\n\nYour next line:"
    return f"你的目標：{goal}\n\n到目前為止的對話：\n{talk}\n\n你的下一句："


def parse(text: str) -> dict:
    """{"say", "done"} from the model's reply (a JSON object, or failing that plain text).
    Ollama has no structured output for Gemma 4 (format=json answers 501), so the JSON is
    asked for in the prompt and read out of the text here."""
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            data = json.loads(match.group(0))
            say = str(data.get("say") or "").strip()
            return {"text": say, "done": bool(data.get("done")) or not say}
        except (json.JSONDecodeError, AttributeError):
            pass
    say = text.strip().strip('"')
    return {"text": say, "done": not say}


def next_line(goal: str, history: list[tuple[str, str]], *, english: bool = False,
              base_url: str = OLLAMA_URL, model: str = MODEL) -> dict:  # fmt: skip
    """The simulated user's next line: {"text", "done", "seconds"}."""
    started = time.perf_counter()
    body = {
        "model": model,
        "stream": False,
        "think": False,
        "keep_alive": "30m",
        "options": {"temperature": 0.4, "num_predict": 120},
        "messages": [
            {"role": "system", "content": SYSTEM_EN if english else SYSTEM},
            {"role": "user", "content": prompt(goal, history, english)},
        ],
    }
    with httpx.Client(timeout=120, trust_env=False) as client:
        r = client.post(base_url.rstrip("/") + "/api/chat", json=body)
        r.raise_for_status()
        reply = (r.json().get("message") or {}).get("content") or ""
    return parse(reply) | {"seconds": round(time.perf_counter() - started, 2)}
