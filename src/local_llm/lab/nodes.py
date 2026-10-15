"""Node types for scenario graphs.

Each node reads values from the shared scenario state (`data`), calls at most one model or
algorithm, and writes its result back under the key named by its `output` parameter. The
parameter specs below are also what the control-center coding agent reads to edit scenarios,
so keep their help text precise.
"""

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage

from .. import catalog
from ..streaming import Backend, stream_chat
from . import decide, rag
from .textutil import lookup, preview, render

DEFAULT_LLM = catalog.default_model()
DEFAULT_ROUTER = "nimble:9b"


@dataclass(frozen=True)
class Param:
    help: str
    default: object = None
    required: bool = False


@dataclass(frozen=True)
class NodeType:
    key: str
    label: dict  # {"zh": ..., "en": ...}
    icon: str
    color: str  # fill colour in the flow diagram
    summary: str  # one line for the coding agent
    params: dict[str, Param]
    run: Callable
    models: Callable[[dict], list[str]] = field(default=lambda p: [])


def _input(p: dict) -> str:
    return p.get("input", "question")


# --- implementations: run(params, data, step) -> dict of state updates -----------------------


def run_llm(p: dict, data: dict, step) -> dict:
    prompt = render(p["prompt"], data)
    system = render(p.get("system", ""), data)
    step.inputs["prompt"] = preview(prompt, 1500)
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    thinking, answer, stats = "", "", None
    for event in stream_chat(
        Backend("Ollama", "ollama", step.ollama_url),
        p.get("model", DEFAULT_LLM),
        messages,
        think=p.get("think", False),
        temperature=p.get("temperature", 0.3),
        max_tokens=p.get("max_tokens", 4096),
        num_ctx=p.get("num_ctx", 16384),
    ):
        if step.cancelled():
            break
        if event.kind == "thinking":
            thinking += event.text
            step.stream = f"💭 … {len(thinking)}"
        elif event.kind == "content":
            answer += event.text
            step.stream = answer
        else:
            stats = event.stats
    if stats:
        step.detail.update(
            output_tokens=stats.output_tokens,
            prompt_tokens=stats.prompt_tokens,
            tokens_per_s=stats.decode_tps and round(stats.decode_tps, 1),
            first_token_s=stats.ttft_s and round(stats.ttft_s, 2),
        )
    if thinking:
        step.detail["thinking"] = thinking
    return {p["output"]: answer.strip()}


def run_agent(p: dict, data: dict, step) -> dict:
    from ..agents import AGENTS, build
    from ..config import Settings

    spec = AGENTS[p["agent"]]
    model = p.get("model") or spec.model or DEFAULT_LLM
    agent = build(spec, Settings("ollama", model, step.ollama_url, max_tokens=4096, context=16384))
    question = lookup(_input(p), data)
    step.inputs["question"] = preview(question)
    trail, answer = [], ""
    for update in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        config={
            "recursion_limit": p.get("max_steps", 14),
            "configurable": {"thread_id": str(uuid.uuid4())},
        },
        stream_mode="updates",
    ):
        for value in update.values():
            for m in (value or {}).get("messages", []):
                if isinstance(m, AIMessage):
                    for call in m.tool_calls:
                        trail.append({"call": call["name"], "args": call["args"]})
                    if m.content and not m.tool_calls:
                        answer = m.content
                elif isinstance(m, ToolMessage):
                    trail.append({"result": m.name, "content": preview(m.content, 600)})
                step.stream = " → ".join(t.get("call") or "✓" for t in trail) or "…"
        if step.cancelled():
            break
    step.detail["tool_trail"] = trail
    step.detail["tool_calls"] = sum("call" in t for t in trail)
    return {p.get("output", "answer"): answer}


def run_router(p: dict, data: dict, step) -> dict:
    text = lookup(_input(p), data)
    step.inputs["text"] = preview(text)
    options = p["options"]
    if p.get("method", "nimble") == "nimble":
        result = decide.system_one(
            {"text": text},
            {"route": {"type": "choice", "instructions": p["instructions"], "criteria": options}},
            p.get("model", DEFAULT_ROUTER),
            step.ollama_url,
        )
        answer = decide.summarize(result["answers"]["route"])
        step.detail["usage"] = result.get("usage")
    else:
        answer = decide.choose_with_llm(
            text, p["instructions"], options, p.get("model", DEFAULT_LLM), step.ollama_url
        )
    step.detail["probabilities"] = answer["probabilities"]
    step.detail["confidence"] = answer["confidence"]
    step.stream = f"→ {answer['value']}"
    return {p.get("output", "route"): answer["value"]}


def run_classify(p: dict, data: dict, step) -> dict:
    text = lookup(_input(p), data)
    step.inputs["text"] = preview(text)
    result = decide.system_one(
        {"text": text}, p["questions"], p.get("model", DEFAULT_ROUTER), step.ollama_url
    )
    labels = {}
    for name, raw in result["answers"].items():
        answer = decide.summarize(raw)
        labels[name] = answer["value"]
        step.detail[name] = answer
    step.detail["usage"] = result.get("usage")
    return {**labels, p.get("output", "labels"): labels}


def _hit_rows(hits: list[dict]) -> list[dict]:
    return [
        {"title": h["title"], "score": h["score"], "text": h["text"][:160], **h.get("ranks", {})}
        for h in hits
    ]


def run_bm25(p: dict, data: dict, step) -> dict:
    query = lookup(_input(p), data)
    role = render(p.get("role") or "", data) or None
    hits = rag.search_bm25(p.get("collection", "company_kb"), query, p.get("top_k", 8), role)
    step.inputs.update(query=preview(query), role=role or "—")
    step.detail["hits"] = _hit_rows(hits)
    return {p["output"]: hits}


def run_vector(p: dict, data: dict, step) -> dict:
    query = lookup(_input(p), data)
    role = render(p.get("role") or "", data) or None
    hits = rag.search_vector(
        p.get("collection", "company_kb"), query, p.get("top_k", 8), role, step.ollama_url
    )
    step.inputs.update(query=preview(query), role=role or "—")
    step.detail["hits"] = _hit_rows(hits)
    return {p["output"]: hits}


def run_rrf(p: dict, data: dict, step) -> dict:
    rankings = [data.get(key) or [] for key in p["inputs"]]
    hits = rag.rrf(rankings, p.get("k", 60), p.get("top_k", 8))
    step.detail["hits"] = _hit_rows(hits)
    return {p["output"]: hits}


def run_rerank(p: dict, data: dict, step) -> dict:
    from .. import gateway_client

    query = lookup(p.get("query", "question"), data)
    hits = data.get(p["input"]) or []
    if not hits:
        return {p["output"]: []}
    results = gateway_client.rerank(
        query, [h["text"] for h in hits], p.get("model", "bge-reranker-v2-m3"), p.get("top_n", 4)
    )
    reranked = []
    for r in results[: p.get("top_n", 4)]:
        hit = hits[r["index"]]
        ranks = {**hit.get("ranks", {}), "before": r["index"] + 1}
        reranked.append({**hit, "score": round(r["relevance_score"], 4), "ranks": ranks})
    step.detail["hits"] = _hit_rows(reranked)
    return {p["output"]: reranked}


def run_transcribe(p: dict, data: dict, step) -> dict:
    from .. import asr_metrics, diarization, gateway_client

    path = Path(lookup(p.get("input", "audio_path"), data))
    model = p.get("model", "breeze-asr-25")
    prompt = render(p["prompt"], data) if p.get("prompt") else None
    translate = p.get("task", "transcribe") == "translate"
    call = gateway_client.translate if translate else gateway_client.transcribe
    result = call(path, model, p.get("language"), prompt=prompt)
    step.inputs["audio"] = path.name
    # Speech models on the Apple GPU take turns; time spent waiting is not the model's.
    seconds = max((result.get("seconds") or 0) - (result.get("queue_seconds") or 0), 0)
    duration = result.get("duration")
    step.detail.update(
        audio_seconds=duration,
        segments=len(result.get("segments") or []),
        model_seconds=round(seconds, 2),
        queue_seconds=result.get("queue_seconds"),
        load_seconds=result.get("load_seconds"),
        # Real-time factor: processing time / audio length (lower is faster).
        rtf=round(seconds / duration, 3) if seconds and duration else None,
    )
    if result.get("truncated"):
        step.detail["truncated"] = True  # an audio LLM hit its output cap (repetition loop)
    text, tag = result["text"], model
    reference = lookup(p["reference"], data) if p.get("reference") else ""
    if len(asr_metrics.normalize(reference)) >= 5:  # "none" / "無" means no reference
        score = asr_metrics.cer(reference, text)
        step.detail["cer"] = score["cer"]
        tag += f" · CER {score['cer'] * 100:.1f}%"
    turns = data.get(p["speakers"]) if p.get("speakers") else None
    if turns:
        label = p.get("speaker_label") or "講者 {n}"
        labelled = diarization.label_segments(turns, result.get("segments") or [])
        sep = ": " if label.isascii() else "："
        text = diarization.transcript(labelled, label, sep)
        step.detail["speakers"] = len({t["speaker"] for t in labelled})
    if p.get("tag_model"):
        text = f"[{tag}] {text}"
    return {p.get("output", "transcript"): text}


def run_diarize(p: dict, data: dict, step) -> dict:
    from .. import diarization, gateway_client

    path = Path(lookup(p.get("input", "audio_path"), data))
    count = p.get("num_speakers")
    result = gateway_client.diarize(
        path, int(count) if count else None, model=p.get("model", "speaker-diarization")
    )
    turns = result["segments"]
    step.inputs["audio"] = path.name
    names = diarization.speaker_names((t["speaker"] for t in turns), "S{n}")
    step.detail.update(
        speakers=result["speakers"],
        turns=len(turns),
        audio_seconds=result.get("duration"),
        model_seconds=result.get("seconds"),
        load_seconds=result.get("load_seconds"),
    )
    step.stream = " ".join(
        f"{names[t['speaker']]} {t['start']:.0f}–{t['end']:.0f}s"
        for t in diarization.merge_turns(turns)
    )
    return {p.get("output", "speaker_turns"): turns}


def run_parse(p: dict, data: dict, step) -> dict:
    from .. import gateway_client

    path = Path(lookup(p.get("input", "document_path"), data))
    result = gateway_client.parse_document(path, model=p.get("model", "docling"))
    step.inputs["document"] = path.name
    step.detail.update(
        pages=result.get("pages"), tables=result.get("tables"), model_seconds=result.get("seconds")
    )
    return {p.get("output", "document"): result["markdown"]}


def run_template(p: dict, data: dict, step) -> dict:
    return {p["output"]: render(p["template"], data)}


# --- registry ------------------------------------------------------------------------------------

OUTPUT = Param("State key to write the result to.", required=True)
INPUT = Param("State key to read (e.g. `question`), or a template like `{a}\\n{b}`.", "question")
COLLECTION = Param("Knowledge-base collection (folder under data/lab/corpus).", "company_kb")
ROLE = Param(
    "Department filter: only chunks tagged `all` or this role (sales/hr/support). "
    "May be a template such as `{route}`. Empty = no filter.",
    "",
)

NODE_TYPES = {
    t.key: t
    for t in [
        NodeType(
            "llm",
            {"zh": "LLM", "en": "LLM"},
            "🧠",
            "#EEF2FF",
            "One chat-model call with a prompt template.",
            {
                "model": Param("Ollama chat model.", DEFAULT_LLM),
                "prompt": Param("User prompt; `{key}` inserts a state value.", required=True),
                "system": Param("System prompt (template).", ""),
                "output": OUTPUT,
                "temperature": Param("Sampling temperature.", 0.3),
                "max_tokens": Param("Output limit (thinking models need more).", 4096),
                "num_ctx": Param("Context window.", 16384),
                "think": Param("Enable thinking where the model allows switching it.", False),
            },
            run_llm,
            lambda p: [p.get("model", DEFAULT_LLM)],
        ),
        NodeType(
            "agent",
            {"zh": "Agent", "en": "Agent"},
            "🤖",
            "#E0F2FE",
            "A tool-using agent from data/lab/agents.json (ReAct loop until it answers).",
            {
                "agent": Param("Agent id in data/lab/agents.json.", required=True),
                "model": Param("Ollama model with tool support (empty = agent default).", None),
                "input": INPUT,
                "output": Param("State key for the final answer.", "answer"),
                "max_steps": Param("Recursion limit for the agent loop.", 14),
            },
            run_agent,
            lambda p: [p.get("model") or DEFAULT_LLM],
        ),
        NodeType(
            "router",
            {"zh": "路由", "en": "Router"},
            "🔀",
            "#FEF3C7",
            "Picks one option; outgoing edges with `when: <option>` decide where to go next.",
            {
                "method": Param("`nimble` (decision model) or `llm` (chat model, JSON).", "nimble"),
                "model": Param("Decision model (nimble) or chat model (llm).", DEFAULT_ROUTER),
                "input": INPUT,
                "instructions": Param("What to decide.", required=True),
                "options": Param("Object {option_key: description}.", required=True),
                "output": Param("State key for the chosen option.", "route"),
            },
            run_router,
            lambda p: [p.get("model", DEFAULT_ROUTER)],
        ),
        NodeType(
            "classify",
            {"zh": "分類", "en": "Classify"},
            "🏷️",
            "#FEF3C7",
            "Several typed questions at once with a decision model (choice / score / noul).",
            {
                "model": Param("Decision model.", DEFAULT_ROUTER),
                "input": INPUT,
                "questions": Param(
                    "Object {name: {type: choice|score|noul, instructions, criteria}}; "
                    "choice criteria = {key: description}, score criteria = [levels].",
                    required=True,
                ),
                "output": Param("State key for {name: value} of all answers.", "labels"),
            },
            run_classify,
            lambda p: [p.get("model", DEFAULT_ROUTER)],
        ),
        NodeType(
            "retrieve_bm25",
            {"zh": "關鍵字檢索 BM25", "en": "Keyword search (BM25)"},
            "🔤",
            "#ECFDF5",
            "Lexical search (BM25, CJK bigrams) over a knowledge-base collection.",
            {
                "collection": COLLECTION,
                "input": INPUT,
                "top_k": Param("Number of passages.", 8),
                "role": ROLE,
                "output": OUTPUT,
            },
            run_bm25,
        ),
        NodeType(
            "retrieve_vector",
            {"zh": "向量檢索", "en": "Vector search"},
            "🧭",
            "#ECFDF5",
            "Dense search with the collection's embedding model (Ollama /api/embed).",
            {
                "collection": COLLECTION,
                "input": INPUT,
                "top_k": Param("Number of passages.", 8),
                "role": ROLE,
                "output": OUTPUT,
            },
            run_vector,
            lambda p: [
                (rag.manifest(p.get("collection", "company_kb")) or {}).get(
                    "embed_model", rag.DEFAULT_EMBED
                )
            ],
        ),
        NodeType(
            "rrf",
            {"zh": "RRF 融合", "en": "RRF fusion"},
            "🔗",
            "#ECFDF5",
            "Reciprocal Rank Fusion of several result lists (score = Σ 1/(k+rank)).",
            {
                "inputs": Param("List of state keys holding result lists.", required=True),
                "k": Param("RRF constant.", 60),
                "top_k": Param("Number of fused passages.", 8),
                "output": OUTPUT,
            },
            run_rrf,
        ),
        NodeType(
            "rerank",
            {"zh": "重新排序", "en": "Rerank"},
            "🎯",
            "#ECFDF5",
            "Cross-encoder reranking through the model gateway.",
            {
                "model": Param("Gateway reranker id.", "bge-reranker-v2-m3"),
                "query": Param("State key or template for the query.", "question"),
                "input": Param("State key holding the passages to rerank.", required=True),
                "top_n": Param("Passages to keep.", 4),
                "output": OUTPUT,
            },
            run_rerank,
            lambda p: [p.get("model", "bge-reranker-v2-m3")],
        ),
        NodeType(
            "transcribe",
            {"zh": "語音轉文字", "en": "Speech to text"},
            "🎙️",
            "#FCE7F3",
            "Speech recognition through the model gateway.",
            {
                "model": Param(
                    "Gateway ASR id: breeze-asr-25, whisper-large-v3(-turbo), qwen3-asr-1.7b, "
                    "qwen3-asr-0.6b, sensevoice-small, fun-asr-nano, or an audio LLM served "
                    "through Ollama (gemma4:e2b, gemma4:e4b).",
                    "breeze-asr-25",
                ),
                "input": Param("State key with the audio file path.", "audio_path"),
                "language": Param("Language hint, e.g. zh or en (empty = auto).", None),
                "prompt": Param(
                    "Optional context (names, terms) to steer recognition; `{key}` allowed. "
                    "Ignored by models without a context input (SenseVoice).",
                    None,
                ),
                "reference": Param(
                    "Optional state key with a reference transcript: the character error rate "
                    "(CER, script/number/punctuation-normalised) is recorded; 'none' skips it.",
                    None,
                ),
                "tag_model": Param(
                    "Prefix the transcript with the model id and CER, e.g. `[breeze-asr-25 · "
                    "CER 6.2%] …`, so a later prompt knows which model wrote it.",
                    False,
                ),
                "task": Param(
                    "'transcribe' (default) or 'translate': speech -> English text in one step "
                    "(whisper-large-v3 or an audio LLM such as gemma4:e4b; Breeze, "
                    "Taiwan-Tongues, large-v3-turbo, Qwen3-ASR and SenseVoice cannot).",
                    "transcribe",
                ),
                "speakers": Param(
                    "Optional state key with speaker turns from a diarize node: the output "
                    "becomes a speaker-labelled transcript, one '講者 1：…' line per turn. "
                    "Needs a model with sentence timestamps (Whisper family, Breeze-ASR-25).",
                    None,
                ),
                "speaker_label": Param(
                    "Speaker name pattern for `speakers`, with {n} = 1, 2, … in order of first "
                    "appearance, e.g. 'Speaker {n}'.",
                    "講者 {n}",
                ),
                "output": Param("State key for the transcript.", "transcript"),
            },
            run_transcribe,
            lambda p: [p.get("model", "breeze-asr-25")],
        ),
        NodeType(
            "diarize",
            {"zh": "說話者分段", "en": "Speaker diarization"},
            "🗣️",
            "#FCE7F3",
            "Who spoke when, through the model gateway (pyannote segmentation + CAM++).",
            {
                "model": Param("Gateway diarization id.", "speaker-diarization"),
                "input": Param("State key with the audio file path.", "audio_path"),
                "num_speakers": Param(
                    "Number of speakers if known (empty = estimate from the audio).", None
                ),
                "output": Param(
                    "State key for the speaker turns [{start, end, speaker}] (seconds; "
                    "speakers numbered 0, 1, … in order of first appearance); pass it to a "
                    "transcribe node's `speakers`.",
                    "speaker_turns",
                ),
            },
            run_diarize,
            lambda p: [p.get("model", "speaker-diarization")],
        ),
        NodeType(
            "parse_document",
            {"zh": "文件解析", "en": "Parse document"},
            "📄",
            "#FCE7F3",
            "PDF/DOCX → Markdown through the model gateway (Docling).",
            {
                "model": Param("Gateway document model id.", "docling"),
                "input": Param("State key with the document path.", "document_path"),
                "output": Param("State key for the Markdown.", "document"),
            },
            run_parse,
            lambda p: [p.get("model", "docling")],
        ),
        NodeType(
            "template",
            {"zh": "組合", "en": "Template"},
            "🧩",
            "#F3F4F6",
            "Combines state values into text without calling a model.",
            {"template": Param("Text with `{key}` placeholders.", required=True), "output": OUTPUT},
            run_template,
        ),
    ]
}


def describe_node_types() -> list[dict]:
    """Documentation for the coding agent (and the scenario editor)."""
    return [
        {
            "type": t.key,
            "summary": t.summary,
            "params": {
                name: {"help": p.help, "default": p.default, "required": p.required}
                for name, p in t.params.items()
            },
        }
        for t in NODE_TYPES.values()
    ]
