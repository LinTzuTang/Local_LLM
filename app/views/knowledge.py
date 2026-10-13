import time

import jobs
import pandas as pd
import streamlit as st
import ui
from common import installed_names, kept, ollama_url
from i18n import tr

from local_llm import catalog
from local_llm.lab import rag

ui.page_header(
    "📚",
    tr("知識庫 RAG", "Knowledge base (RAG)"),
    tr(
        "RAG 場景用的文件庫。流程：解析（Markdown 直接讀，PDF/DOCX 交給模型服務的 Docling）→ "
        "依標題切段 → BM25 關鍵字索引＋向量索引（Ollama embedding）。每份文件標記哪些部門可讀，"
        "檢索時先過濾權限，機密段落不會進到 prompt。",
        "Documents for the RAG scenarios. Pipeline: parse (Markdown as is, PDF/DOCX via the "
        "gateway's Docling) → split by headings → BM25 keyword index + vector index (Ollama "
        "embeddings). Each document is tagged with the departments allowed to read it; retrieval "
        "filters by permission first, so restricted passages never reach a prompt.",
    ),
)


def pipeline_dot() -> str:
    """How a question becomes a cited answer (the hybrid RAG scenario)."""

    def n(name: str, title: str, sub: str, fill: str) -> str:
        return (
            f'{name} [label=<<b>{title}</b><br/><font point-size="9" color="#475569">{sub}'
            f'</font>>, fillcolor="{fill}"]'
        )

    return "\n".join(
        [
            "digraph {",
            'graph [rankdir=LR, bgcolor="transparent", nodesep=0.25, ranksep=0.35]',
            'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, '
            'color="#CBD5E1", margin="0.12,0.06"]',
            'edge [color="#94A3B8", arrowsize=0.7, fontname="Helvetica", fontsize=9, '
            'fontcolor="#64748B"]',
            n("docs", tr("📄 文件", "📄 Documents"), "md · pdf · docx", "#F8FAFC"),
            n("parse", tr("解析＋切段", "Parse + chunk"), "Docling · headings", "#FCE7F3"),
            n("idx", tr("兩種索引", "Two indexes"), "BM25 · bge-m3", "#ECFDF5"),
            n(
                "q",
                tr("❓ 問題", "❓ Question"),
                tr("＋提問者部門", "+ asker's department"),
                "#EEF2FF",
            ),
            n(
                "acl",
                tr("🔒 權限過濾", "🔒 Permission filter"),
                tr("只留可讀文件", "readable docs only"),
                "#FEF3C7",
            ),
            n("bm25", "🔤 BM25", tr("精確詞、代號", "exact terms, codes"), "#ECFDF5"),
            n(
                "vec",
                tr("🧭 向量", "🧭 Vector"),
                tr("語意、跨語言", "meaning, cross-language"),
                "#ECFDF5",
            ),
            n("rrf", "🔗 RRF", "Σ 1/(60+rank)", "#ECFDF5"),
            n("rr", "🎯 Rerank", "bge-reranker-v2-m3", "#CCFBF1"),
            n("llm", tr("🧠 LLM 回答", "🧠 LLM answer"), tr("引用 [n]", "cites [n]"), "#EEF2FF"),
            "docs -> parse -> idx",
            "q -> acl; idx -> acl [style=dotted]",
            "acl -> bm25; acl -> vec; bm25 -> rrf; vec -> rrf",
            'rrf -> rr [label=" top 8"]; rr -> llm [label=" top 4"]',
            "}",
        ]
    )


st.graphviz_chart(pipeline_dot(), use_container_width=True)

ROLES = {
    "all": tr("全公司", "Everyone"),
    "sales": tr("業務部", "Sales"),
    "hr": tr("人資部", "HR"),
    "support": tr("客服部", "Support"),
}
collections = rag.collections()
if not collections:
    st.info(tr("還沒有任何文件庫。", "No collections yet."))
    st.stop()
collection = st.selectbox(tr("文件庫", "Collection"), collections, **kept("kb_collection"))
docs = rag.documents(collection)
info = rag.manifest(collection)

c1, c2, c3, c4 = st.columns(4)
c1.metric(tr("文件", "Documents"), len(docs))
c2.metric(tr("段落", "Chunks"), info["chunks"] if info else "—")
c3.metric(tr("Embedding 模型", "Embedding model"), info["embed_model"] if info else "—")
c4.metric(tr("建立時間", "Built"), info["built_at"][5:16] if info else tr("尚未建立", "not yet"))


# --- index build (background job) ---------------------------------------------------------------
def build_work(job: jobs.Job, collection: str, model: str, url: str) -> None:
    """Worker thread: no Streamlit calls (and no tr()), so progress text is language-neutral."""
    for event in rag.build_index(collection, model, url):
        if event["stage"] == "parse":
            job.progress = event["done"] / max(event["total"], 1) * 0.8
            job.detail = f"📄 {event['doc']}"
        elif event["stage"] == "embed":
            job.progress, job.detail = 0.85, f"🧭 {model} × {event['total']}"
        else:
            job.state["manifest"] = event


embedders = [
    n for n in installed_names() if "embedding" in ((catalog.find(n) or {}).get("caps") or [])
] or [rag.DEFAULT_EMBED]
job = jobs.find("kb_build", collection=collection)
building = bool(job and job.running)
with st.container(border=True):
    a, b, c = st.columns([2, 1, 3])
    model = a.selectbox(
        tr("Embedding 模型", "Embedding model"),
        embedders,
        index=next((i for i, m in enumerate(embedders) if m.startswith(rag.DEFAULT_EMBED)), 0),
    )
    if b.button(
        tr("重建索引", "Rebuild index"), type="primary", disabled=building, use_container_width=True
    ):
        url = ollama_url()
        jobs.start(
            jobs.Job(
                "kb_build",
                tr(f"建立索引 {collection}", f"Index {collection}"),
                "views/knowledge.py",
                state={"collection": collection},
            ),
            lambda j: build_work(j, collection, model, url),
        )
        st.rerun()
    if building:

        @st.fragment(run_every=1)
        def progress(j: jobs.Job) -> None:
            if not j.running:
                st.rerun()
            st.progress(j.progress or 0.0, text=j.detail or "…")

        with c:
            progress(job)
    elif job and job.status == "error":
        c.error(job.error)
    elif job and job.status == "done":
        m = job.state.get("manifest", {})
        c.success(
            tr(
                f"完成：{m.get('documents')} 份文件、{m.get('chunks')} 段，{m.get('seconds')} 秒",
                f"Done: {m.get('documents')} documents, {m.get('chunks')} chunks "
                f"in {m.get('seconds')} s",
            )
        )

tab_search, tab_docs, tab_upload = st.tabs(
    [
        tr("🔎 檢索比較", "🔎 Compare retrieval"),
        tr("📄 文件與權限", "📄 Documents & access"),
        tr("⬆️ 新增文件", "⬆️ Add documents"),
    ]
)

# --- search playground ----------------------------------------------------------------------------
with tab_search:
    st.markdown(
        tr(
            "同一個問題，並排比較 **BM25（關鍵字）**、**向量（語意）**、**RRF 融合** 與 **Rerank** 的排名。"
            "試試英文問中文文件、或含產品代號的問題，看兩種檢索各自的強項。",
            "One query, four rankings side by side: **BM25 (keywords)**, **vector (meaning)**, "
            "**RRF fusion** and **rerank**. Try an English question over Chinese documents, or "
            "one with a product code, to see where each method shines.",
        )
    )
    RERANK_DEMOS = [
        "How fast must we answer when a customer's line stops?",
        "What discount can a junior sales rep give?",
        "員工可以把客戶合約貼到 ChatGPT 嗎？",
    ]
    st.markdown(
        tr(
            "**需要 Rerank 的問題**：這幾題融合後的第 1 名是錯的段落（例如「調薪」），"
            "cross-encoder 逐段細讀後才把正確段落排到第 1：",
            "**Questions that need reranking**: after fusion, the top passage is wrong (e.g. "
            "the pay-raise rule); the cross-encoder reads each passage with the question and "
            "moves the right one to #1:",
        )
    )
    st.session_state.setdefault(
        "kb_query", tr("SM-400 交期多久？", "How long is the SM-400 lead time?")
    )
    for i, demo in enumerate(RERANK_DEMOS):
        st.button(
            f"🎯 {demo}",
            key=f"kb_demo_{i}",
            on_click=st.session_state.__setitem__,
            args=("kb_query", demo),
        )
    a, b, c = st.columns([4, 1.4, 1])
    query = a.text_input(tr("問題", "Query"), key="kb_query")
    role = b.selectbox(
        tr("以哪個部門身分", "As department"),
        ["", "sales", "hr", "support"],
        format_func=lambda r: tr("不過濾", "No filter") if not r else ROLES[r],
    )
    top_k = c.number_input("top k", 2, 10, 5)
    if not info:
        st.warning(tr("請先建立索引。", "Build the index first."))
    elif query:
        started = time.perf_counter()
        try:
            bm25 = rag.search_bm25(collection, query, top_k * 2, role or None)
            vector = rag.search_vector(collection, query, top_k * 2, role or None, ollama_url())
        except Exception as exc:
            st.error(f"{type(exc).__name__}: {exc}")
            st.stop()
        fused = rag.rrf([bm25, vector], top_k=top_k * 2)
        retrieval_s = time.perf_counter() - started
        reranked, rerank_note = None, None
        try:
            from local_llm import gateway_client

            started = time.perf_counter()
            results = gateway_client.rerank(query, [h["text"] for h in fused], top_n=top_k)
            reranked = [
                {
                    **fused[r["index"]],
                    "score": round(r["relevance_score"], 4),
                    "before": r["index"] + 1,
                }
                for r in results
            ]
            rerank_note = f"{time.perf_counter() - started:.2f}s"
            if reranked and reranked[0]["before"] != 1:
                st.success(
                    tr(
                        f"🎯 Rerank 把融合後第 {reranked[0]['before']} 名的段落拉到第 1 名："
                        f"「{reranked[0]['text'].split(chr(10), 1)[-1][:60]}…」",
                        f"🎯 Rerank moved fusion's #{reranked[0]['before']} to #1: "
                        f"“{reranked[0]['text'].split(chr(10), 1)[-1][:60]}…”",
                    )
                )
        except Exception as exc:
            rerank_note = tr(f"無法使用：{exc}", f"unavailable: {exc}")

        def table(hits: list[dict], extras: tuple = ()) -> None:
            rows = [
                {
                    "#": i,
                    tr("文件", "Doc"): h["title"],
                    "score": h["score"],
                    **{f"{tr('原排名', 'rank')} {x}": h.get(x) for x in extras},
                    tr("段落", "Passage"): h["text"].split("\n", 1)[-1][:80],
                }
                for i, h in enumerate(hits[:top_k], start=1)
            ]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        st.caption(tr(f"檢索 {retrieval_s:.2f}s", f"retrieval {retrieval_s:.2f}s"))
        cols = st.columns(2)
        with cols[0]:
            st.markdown("**🔤 BM25**")
            table(bm25)
            st.markdown("**🔗 RRF** — score = Σ 1/(60 + rank)")
            for h in fused:
                h["bm25"], h["vector"] = h["ranks"].get("bm25"), h["ranks"].get("vector")
            table(fused, ("bm25", "vector"))
        with cols[1]:
            st.markdown(f"**🧭 {tr('向量', 'Vector')}** · `{info['embed_model']}`")
            table(vector)
            st.markdown(f"**🎯 Rerank** · `bge-reranker-v2-m3` · {rerank_note}")
            if reranked:
                table(reranked, ("before",))
            else:
                st.info(
                    tr(
                        "Rerank 需要模型服務（gateway）與 bge-reranker-v2-m3。",
                        "Rerank needs the model gateway and bge-reranker-v2-m3.",
                    )
                )

# --- documents & access -----------------------------------------------------------------------------
with tab_docs:
    st.dataframe(
        pd.DataFrame(
            [
                {
                    tr("標題", "Title"): d["title"],
                    tr("檔案", "File"): d["file"],
                    tr("可讀部門", "Readable by"): "、".join(ROLES.get(r, r) for r in d["access"]),
                    "KB": round(d["bytes"] / 1024, 1),
                }
                for d in docs
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    doc = st.selectbox(
        tr("檢視／修改權限", "View / change access"),
        docs,
        format_func=lambda d: f"{d['title']} ({d['file']})",
    )
    if doc:
        access = st.multiselect(
            tr("可讀部門", "Readable by"),
            list(ROLES),
            default=doc["access"],
            format_func=ROLES.get,
            key=f"access_{doc['file']}",
        )
        if access != doc["access"] and st.button(tr("更新權限", "Update access")):
            rag.set_document_meta(collection, doc["file"], doc["title"], access)
            st.toast(tr("已更新，重建索引後生效。", "Updated; rebuild the index to apply."))
            st.rerun()
        path = rag.CORPUS_DIR / collection / doc["file"]
        if path.suffix in rag.TEXT_SUFFIXES:
            with st.expander(tr("內容", "Content")):
                st.markdown(path.read_text())

# --- upload -------------------------------------------------------------------------------------------
with tab_upload:
    st.markdown(
        tr(
            "Markdown/TXT 直接加入；PDF、DOCX、PPTX、HTML 會在建立索引時交給模型服務（Docling）解析。",
            "Markdown/TXT are added as is; PDF, DOCX, PPTX and HTML are parsed by the gateway "
            "(Docling) when the index is built.",
        )
    )
    file = st.file_uploader(
        tr("選擇檔案", "Choose a file"), type=["md", "txt", "pdf", "docx", "pptx", "html"]
    )
    if file:
        name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in file.name)
        title = st.text_input(tr("標題", "Title"), file.name.rsplit(".", 1)[0])
        access = st.multiselect(
            tr("可讀部門", "Readable by"), list(ROLES), default=["all"], format_func=ROLES.get
        )
        exists = (rag.CORPUS_DIR / collection / name).exists()
        if exists:
            st.warning(tr("同名檔案會被覆蓋。", "A file with this name will be replaced."))
        if st.button(tr("加入文件庫", "Add to collection"), type="primary", disabled=not access):
            (rag.CORPUS_DIR / collection / name).write_bytes(file.getvalue())
            rag.set_document_meta(collection, name, title, access)
            st.toast(tr("已加入，請重建索引。", "Added; rebuild the index."))
            st.rerun()
