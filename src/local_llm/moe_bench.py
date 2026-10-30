"""MoE vs dense language models in Ollama: memory, generation speed, prompt speed, TMMLU+ accuracy.

    from local_llm import moe_bench
    row = moe_bench.speed("qwen3:30b-a3b")
    row = moe_bench.quality("qwen3:30b-a3b", moe_bench.questions(50))

Pairs (PAIRS), as close as the installed models allow (same family and quantisation, MoE or
not); C is a reference only (different families):
  A  qwen3:30b-a3b (MoE, 128 experts, 8 used)      vs qwen3:32b (dense)             Q4_K_M, GGUF
  B  qwen3.6:35b-a3b-coding (MoE, about 3B active) vs qwen3.8:27b (dense)           nvfp4, MLX
  C  gpt-oss:20b (MoE, 32 experts, 4 used)         vs mistral-small3.2:24b (dense)  reference only
  D  gemma4:26b (MoE, 128 experts, 8 used)         vs gemma4:31b (dense)            nvfp4, MLX
  E  qwen3.6:35b-a3b-nvfp4 (MoE, about 3B active)  vs qwen3.6:27b-nvfp4 (dense)     nvfp4, MLX
A, D and E are the fair pairs (same family, generation and quantisation); B mixes a coding MoE
with a general dense model a generation apart and is kept for the first results; E replaces it.

speed(): the model alone in Ollama (the others unloaded first), temperature 0, thinking off
where the model allows it, num_ctx 20480: memory as Ollama reports it for the loaded model
(/api/ps, weights plus the KV cache for that context); generation tokens/s over a short Chinese
question (256 tokens, median of 3 after a warm-up); prompt (prefill) tokens/s and the time to
the first token on a long Chinese prompt of about 4K and 16K tokens (32 tokens out).

quality(): TMMLU+ (ikala/tmmluplus, Taiwan exam questions, all 67 subjects' test split) drawn
with seed 1, the same questions for every model, temperature 0, one letter as the answer (the
first A-D standing alone). Ollama's think=false does not stop qwen3:30b-a3b from reasoning in
its reply, so the Qwen models get the prompt raw in their chat format with an empty thinking
block, as their template does with thinking off; gpt-oss cannot turn reasoning off and reasons
at "low".

Expectation: memory follows total parameters, generation speed follows active parameters
(decoding reads every active weight once per token). scripts/llm_eval/run_moe_dense.py runs
the full test (200 questions) and exports data/lab/moe_benchmark.json for the platform.
"""

from __future__ import annotations

import csv
import random
import re
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import httpx

PROJECT = Path(__file__).resolve().parents[2]
OLLAMA = "http://127.0.0.1:11434"
PAIRS = {
    "A": (("qwen3:30b-a3b", "MoE"), ("qwen3:32b", "dense")),
    "B": (("qwen3.6:35b-a3b-coding", "MoE"), ("qwen3.8:27b", "dense")),
    "C": (("gpt-oss:20b", "MoE"), ("mistral-small3.2:24b", "dense")),
    "D": (("gemma4:26b", "MoE"), ("gemma4:31b", "dense")),
    "E": (("qwen3.6:35b-a3b-nvfp4", "MoE"), ("qwen3.6:27b-nvfp4", "dense")),
}
MODELS = {model: (pair, kind) for pair, members in PAIRS.items() for model, kind in members}
SHORT = "請用三段話介紹台灣的半導體產業：發展歷史、目前的地位、未來的挑戰。"
NUM_CTX = 20480
CHATML = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
ASK = (
    "以下是一題選擇題，請只回答正確選項的字母（A、B、C 或 D），不要解釋。\n\n"
    "{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n\n答案："
)
CORPUS = PROJECT / "data" / "lab" / "corpus" / "company_kb"


def long_prompt(chars: int, source: Path | None = None) -> str:
    """About `chars` characters of Chinese text (repeated as needed) and a one-line task."""
    if source and source.exists():
        text = source.read_text(encoding="utf-8")
    else:
        text = "\n\n".join(p.read_text(encoding="utf-8") for p in sorted(CORPUS.glob("*.md")))
    while len(text) < chars:
        text += "\n" + text
    return text[:chars] + "\n\n請用一句話總結上面這份文件的主題。"


def generate(
    model: str,
    prompt: str,
    num_predict: int,
    num_ctx: int = NUM_CTX,
    raw: bool = False,
    base_url: str = OLLAMA,
) -> dict:
    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "raw": raw,
        "think": "low"
        if model.startswith("gpt-oss")
        else False,  # gpt-oss cannot turn reasoning off
        "keep_alive": "10m",
        "options": {"temperature": 0, "seed": 1, "num_predict": num_predict, "num_ctx": num_ctx},
    }
    with httpx.Client(timeout=1800) as c:
        r = c.post(f"{base_url}/api/generate", json=body)
        if r.status_code == 400 and "think" in r.text:  # models without a thinking switch
            body.pop("think")
            r = c.post(f"{base_url}/api/generate", json=body)
    r.raise_for_status()
    d = r.json()
    return {
        "response": d.get("response", ""),
        "thought": bool(d.get("thinking")) or "<think>" in d.get("response", ""),
        "prompt_tokens": d.get("prompt_eval_count"),
        "prompt_s": d.get("prompt_eval_duration", 0) / 1e9,
        "eval_tokens": d.get("eval_count"),
        "eval_s": d.get("eval_duration", 0) / 1e9,
        "load_s": d.get("load_duration", 0) / 1e9,
    }


def unload_all(base_url: str = OLLAMA) -> None:
    with httpx.Client(timeout=120) as c:
        for m in c.get(f"{base_url}/api/ps").json().get("models", []):
            c.post(f"{base_url}/api/generate", json={"model": m["name"], "keep_alive": 0})
    time.sleep(3)


def loaded_size(model: str, base_url: str = OLLAMA) -> dict:
    with httpx.Client(timeout=60) as c:
        for m in c.get(f"{base_url}/api/ps").json().get("models", []):
            if m["name"] == model:
                return {
                    "size_gb": round(m["size"] / 1e9, 1),
                    "vram_gb": round(m.get("size_vram", 0) / 1e9, 1),
                }
    return {}


def info(model: str, base_url: str = OLLAMA) -> dict:
    with httpx.Client(timeout=60) as c:
        d = c.post(f"{base_url}/api/show", json={"model": model}).json()
    mi, det = d.get("model_info", {}), d.get("details", {})
    experts = {k.split(".")[-1]: v for k, v in mi.items() if "expert" in k}
    return {
        "family": det.get("family"),
        "params": det.get("parameter_size"),
        "quant": det.get("quantization_level"),
        **experts,
    }


def speed(
    model: str,
    base_url: str = OLLAMA,
    source: Path | None = None,
    progress: Callable[[str], None] = lambda _: None,
) -> dict:
    pair, kind = MODELS.get(model, ("-", "?"))
    progress("unload")
    unload_all(base_url)
    started = time.perf_counter()
    progress("load")
    warm = generate(model, "你好", 8, base_url=base_url)
    row = {
        "model": model,
        "pair": pair,
        "kind": kind,
        **info(model, base_url),
        "load_s": round(time.perf_counter() - started, 1),
        **loaded_size(model, base_url),
    }
    runs = []
    for i in range(3):
        progress(f"generate {i + 1}/3")
        runs.append(generate(model, SHORT, 256, base_url=base_url))
    row["gen_tok_s"] = round(
        statistics.median(r["eval_tokens"] / r["eval_s"] for r in runs if r["eval_s"]), 1
    )
    row["short_first_s"] = round(statistics.median(r["prompt_s"] for r in runs), 3)
    for label, chars in (("4k", 5200), ("16k", 21000)):
        progress(f"prefill {label}")
        r = generate(model, long_prompt(chars, source), 32, base_url=base_url)
        row[f"prefill_{label}_tokens"] = r["prompt_tokens"]
        row[f"prefill_{label}_tok_s"] = (
            round(r["prompt_tokens"] / r["prompt_s"], 1) if r["prompt_s"] else None
        )
        row[f"first_{label}_s"] = round(r["prompt_s"], 2)
        row[f"gen_after_{label}_tok_s"] = (
            round(r["eval_tokens"] / r["eval_s"], 1) if r["eval_s"] else None
        )
    row["warm_load_s"] = round(warm["load_s"], 1)
    row["at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    return row


def questions(n: int, seed: int = 1) -> list[dict]:
    """`n` TMMLU+ test questions drawn from all subjects (downloaded once into the HF cache)."""
    from huggingface_hub import hf_hub_download, list_repo_files

    files = sorted(
        f
        for f in list_repo_files("ikala/tmmluplus", repo_type="dataset")
        if f.endswith("_test.csv")
    )
    pool = []
    for f in files:
        path = hf_hub_download("ikala/tmmluplus", f, repo_type="dataset")
        with open(path, encoding="utf-8") as fh:
            subject = f.split("/")[-1].removesuffix("_test.csv")
            pool += [{"subject": subject, **row} for row in csv.DictReader(fh)]
    return random.Random(seed).sample(pool, n)


def answer_letter(reply: str) -> str | None:
    reply = re.sub(r"<think>.*?(</think>|$)", "", reply, flags=re.S).strip()
    return next(iter(re.findall(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])", reply)), None)


def quality(
    model: str,
    items: list[dict],
    base_url: str = OLLAMA,
    progress: Callable[[int, int], None] = lambda i, n: None,
    cancelled: Callable[[], bool] = lambda: False,
) -> dict:
    unload_all(base_url)
    answers = []
    for i, q in enumerate(items):
        if cancelled():
            break
        progress(i, len(items))
        if model.startswith("qwen"):
            r = generate(
                model, CHATML.format(ASK.format(**q)), 16, num_ctx=4096, raw=True, base_url=base_url
            )
        else:
            r = generate(model, ASK.format(**q), 2048, num_ctx=4096, base_url=base_url)
        letter = answer_letter(r["response"])
        answers.append(
            {
                "i": i,
                "answer": letter,
                "correct": letter == q["answer"].strip(),
                "thought": r["thought"],
                "long": len(
                    re.sub(r"<think>.*?(</think>|$)", "", r["response"], flags=re.S).strip()
                )
                > 20,
            }
        )
    pair, kind = MODELS.get(model, ("-", "?"))
    n = len(answers)
    return {
        "model": model,
        "pair": pair,
        "kind": kind,
        "questions": n,
        "correct": sum(a["correct"] for a in answers),
        "no_letter": sum(a["answer"] is None for a in answers),
        "thought": sum(a["thought"] for a in answers),
        "long": sum(a["long"] for a in answers),
        "accuracy": round(sum(a["correct"] for a in answers) / n, 3) if n else None,
        "answers": answers,
        "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
