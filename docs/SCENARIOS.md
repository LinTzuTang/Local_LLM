# Multi-agent scenarios

A scenario is a small JSON graph in `data/lab/scenarios.json`. The engine
(`src/local_llm/lab/engine.py`) compiles it into a LangGraph `StateGraph` and records a trace
of every node execution. The Scenarios page uses that trace to draw the live flow diagram and
the timeline. The control-center coding agent edits the same JSON.

## Format

```jsonc
{
  "id": "hybrid_rag",                       // snake_case, unique
  "category": "rag",                        // routing | rag | speech | documents | multi_model | pipeline
  "name": {"zh": "混合檢索 RAG", "en": "Hybrid RAG"},
  "description": {"zh": "…", "en": "…"},
  "inputs": [{"key": "question", "type": "text", "label": {"zh": "問題", "en": "Question"}}],
  "examples": {"zh": ["…"], "en": ["…"]},
  "nodes": [
    {"id": "bm25", "type": "retrieve_bm25", "label": {"zh": "BM25", "en": "BM25"},
     "params": {"input": "question", "top_k": 8, "output": "bm25_hits"}},
    {"id": "fuse", "type": "rrf", "join": "all",
     "params": {"inputs": ["bm25_hits", "vector_hits"], "output": "fused"}}
  ],
  "edges": [{"from": "START", "to": "bm25"}, {"from": "bm25", "to": "fuse"}],
  "output": "answer"                        // state key shown as the result
}
```

- **State**: every node reads from one shared dict and writes its result under
  `params.output`. Parallel branches add their own keys; a reducer merges them.
- **Templates**: in `prompt`, `system`, `template`, `input`, `query` and `role`,
  `{key}` inserts a state value. Retrieval hits are formatted as numbered passages
  (`[1] (title) text`), which makes citations easy.
- **Edges**: an edge goes `START → node → … → END`. Several edges leaving one node run in
  parallel. A node with `"join": "all"` waits for every incoming branch.
- **Routing**: a `router` node writes the chosen option, and its outgoing edges carry
  `"when": "<option>"`. Use `"when": "*"` for a fallback.
- **Inputs**: inputs can be `text`, `choice` (with `options`), `audio` or `document`. Audio and
  document inputs are file paths; the UI offers a bundled sample or an upload.

## Node types

| type | uses | does |
|---|---|---|
| `llm` | Ollama chat model | One call with a prompt template; streams; records tokens, tokens/s, thinking |
| `agent` | Ollama model + tools | Runs an agent from `data/lab/agents.json` (ReAct loop); records every tool call |
| `router` | nimble (`/v1/systemone`) or an LLM | Picks one option and records the probabilities; conditional edges follow it |
| `classify` | nimble | Several typed questions at once (`choice` / `score` / `noul`), e.g. team + urgency + refund |
| `retrieve_bm25` | — | BM25 over a collection (CJK bigrams), with an optional department filter |
| `retrieve_vector` | Ollama embedding (bge-m3) | Dense search, with the same department filter |
| `rrf` | — | Reciprocal Rank Fusion: score = Σ 1/(k + rank) |
| `rerank` | gateway cross-encoder | Reranks passages (bge-reranker-v2-m3) |
| `transcribe` | gateway ASR | Speech to text (Breeze-ASR-25, Whisper) |
| `parse_document` | gateway Docling | PDF/DOCX → Markdown, including tables |
| `template` | — | Combines state values into text |

The parameter list for each type, with defaults, is in `src/local_llm/lab/nodes.py`. The
Scenarios page lists them under "Node type reference", and the coding agent can call
`list_node_types`.

## Validation and history

`store.validate_scenario` runs before every save. It rejects:

- unknown node types or params, or missing required params
- edges to unknown nodes
- conditional edges that don't leave a router, or `when` values that aren't router options
- nodes that can't be reached from START, or have no path to END

It warns, without blocking, when a `{key}` isn't produced by any node or input.

Every save keeps the previous version in `storage/lab/history/`, and the control center can
restore it.

## Adding a scenario

1. Ask the control center, for example: "Create a scenario that translates a question to
   English, answers it with hybrid RAG and translates the answer back".
2. Or open the scenario's **Definition** tab, edit the JSON (validated live, with a diff) and
   save.
3. Or edit `data/lab/scenarios.json` directly. The page picks up changes on the next rerun.

New node types go in `nodes.py`. Write a `run(params, data, step)` function that returns the
state updates, and register a `NodeType` with its params so validation, the UI and the coding
agent all know about it.
