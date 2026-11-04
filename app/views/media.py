"""Other models: compare OCR models (ocr_compare.py) and try the model gateway's document parsing
and reranker. Language models are on views/llm.py, speech models on views/speech.py.

Every result shows the HTTP call that produced it (the platform's theme: make API calls
visible). Long calls run as background jobs, so switching pages does not stop them.
"""

import json

import jobs
import ocr_compare
import pandas as pd
import streamlit as st
import ui
from common import download_control, gateway_url
from i18n import tr

from local_llm import gateway_client, media_tools

URL = gateway_url()
SAMPLE_AUDIO = media_tools.SAMPLES_DIR / "meeting_zh.wav"
SAMPLE_DOC = media_tools.SAMPLES_DIR / "supplier_contract.pdf"
AUDIO_TYPES = sorted(t.lstrip(".") for t in media_tools.AUDIO_TYPES)
DOC_TYPES = sorted(t.lstrip(".") for t in media_tools.DOCUMENT_TYPES)
PAGE_TASKS = {"doc", "rerank"}  # gateway models this page uses; speech ones are on views/speech.py
TASK_LABELS = {
    "asr": tr("語音轉文字", "Speech to text"),
    "doc": tr("文件解析", "Document parsing"),
    "rerank": "Rerank",
    "diarize": tr("說話者分段", "Diarization"),
}

ui.page_header(
    "document_scanner",
    tr("其他模型比較", "Other models"),
    tr(
        "語言與語音以外的模型：圖片與掃描文件轉文字（OCR 比較）、PDF／Word 解析成 Markdown、檢索結果重新排序（Rerank）。文件解析與 reranker 由本機的**模型閘道**提供，OCR 模型在 Ollama；資料不會離開這台電腦，每個結果下方都列出實際呼叫的 API。語言模型請到「語言模型比較」，語音模型請到「語音模型比較」。",
        "Models other than language and speech ones: images and scans to text (Compare OCR), PDF/Word to Markdown, reranking retrieval results. Document parsing and the reranker come from the local **model gateway**, the OCR models run in Ollama; nothing leaves this Mac and every result shows the API call behind it. Language models are on Language models, speech models on Compare speech models.",
    ),
)

version = gateway_client.version(URL)
if version is None:
    st.error(tr(f"連不到模型閘道（{URL}）。", f"Cannot reach the model gateway ({URL})."))
    st.markdown(tr("在 terminal 執行其中一個：", "Start it in a terminal with one of:"))
    st.code(
        "./scripts/tmux_up.sh        # "
        + tr("在 tmux 背景執行（建議）", "runs in tmux, in the background (recommended)")
        + "\n./scripts/start_gateway.sh  # "
        + tr("在目前的 terminal 執行", "runs in this terminal")
        + "\nuv run python -m local_llm.gateway",
        "sh",
    )
    st.caption(tr("說明見 docs/GATEWAY.md。", "See docs/GATEWAY.md."))
    st.stop()

models = {m["id"]: m for m in gateway_client.models(URL)}
loaded = {m["id"]: m for m in gateway_client.loaded(URL)}


# -- helpers --------------------------------------------------------------------------------


def status_panel() -> None:
    title = tr("模型閘道狀態", "Gateway status") + f" · v{version} · `{URL}`"
    with st.expander(title, expanded=False):
        st.caption(tr("資料來自 `GET /api/tags` 與 `GET /api/ps`。",
                      "From `GET /api/tags` and `GET /api/ps`."))  # fmt: skip
        rows = []
        for m in models.values():
            if m["task"] not in PAGE_TASKS:
                continue
            ps = loaded.get(m["id"])
            rows.append(
                {
                    tr("模型", "Model"): m["id"],
                    tr("任務", "Task"): TASK_LABELS.get(m["task"], m["task"]),
                    # table cells cannot show icons: plain-text marks
                    tr("已下載", "Downloaded"): "✓" if m["installed"] else "—",
                    tr("大小", "Size"): f"{m['size_gb']} GB",
                    tr("記憶體中", "In memory"): f"● {ps['device']}" if ps else "—",
                    tr("卸載時間", "Unloads at"): ((ps or {}).get("expires_at") or "")[11:19],
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        for model_id in loaded:
            if st.button(tr(f"卸載 {model_id}", f"Unload {model_id}"), key=f"unload_{model_id}"):
                gateway_client.unload(model_id, URL)
                st.rerun()


def model_select(task: str, key: str) -> str | None:
    """Gateway model for a task; offers the download when it is missing."""
    options = [m for m in models if models[m]["task"] == task]

    def label(model_id: str) -> str:
        m = models[model_id]
        mark = "" if m["installed"] else "↓ "  # select box options cannot show icons
        return f"{mark}{m['name']} ({m['size_gb']} GB)"

    choice = st.selectbox(tr("模型", "Model"), options, format_func=label, key=key)
    if choice and not models[choice]["installed"]:
        st.warning(tr(f"{choice} 尚未下載。", f"{choice} is not downloaded yet."))
        download_control(choice, key=f"{key}_pull", source="gateway", page="views/media.py",
                         label=tr(f"下載 {choice}", f"Download {choice}"))  # fmt: skip
        return None
    return choice


def pick_file(kind: str, key: str) -> tuple[str, bytes] | None:
    """(file name, bytes) from an upload, the bundled sample or the inbox."""
    sources = {
        "sample": tr("範例檔", "Sample file"),
        "upload": tr("上傳", "Upload"),
        "inbox": tr("收件匣", "Inbox"),
    }
    source = st.segmented_control(tr("來源", "Source"), list(sources), format_func=sources.get,
                                  default="sample", required=True, key=f"{key}_source")  # fmt: skip
    if source == "upload":
        types = AUDIO_TYPES if kind == "audio" else DOC_TYPES
        upload = st.file_uploader(tr("選擇檔案", "Choose a file"), type=types, key=f"{key}_up")
        return (upload.name, upload.getvalue()) if upload else None
    if source == "inbox":
        names = [r["name"] for r in media_tools.list_files() if r["kind"] == kind]
        if not names:
            st.info(tr("收件匣裡沒有這類檔案。", "No files of this kind in the inbox."))
            return None
        name = st.selectbox(tr("檔案", "File"), names, key=f"{key}_inbox")
        return name, media_tools.inbox_path(name).read_bytes()
    sample = SAMPLE_AUDIO if kind == "audio" else SAMPLE_DOC
    st.caption(tr(f"範例：`{sample.name}`（虛構的公司內容，原稿在 data/lab/samples/）",
                  f"Sample: `{sample.name}` (fictional company content; source in data/lab/samples/)"))  # fmt: skip
    return sample.name, sample.read_bytes()


def start(key: str, title: str, work) -> None:
    job = jobs.start(jobs.Job("media", title, "views/media.py"), work)
    st.session_state[key] = job.id
    st.rerun()


@st.fragment(run_every=1)
def wait_for(job: jobs.Job) -> None:
    if not job.running:
        st.rerun()
    st.info(
        f"{job.title} · {job.elapsed():.0f}s · "
        + tr(
            "第一次使用要先載入模型；可以切換頁面，結果會保留。",
            "The first call loads the model; you can switch pages, the result is kept.",
        ),  # fmt: skip
        icon=":material/progress_activity:",
    )


def current_job(key: str) -> jobs.Job | None:
    """The page's last job: shows progress while running, returns it when finished."""
    job = jobs.get(st.session_state.get(key))
    if job is None:
        return None
    if job.running:
        wait_for(job)
        return None
    if job.status == "error":
        st.error(job.error)
        return None
    return job


def api_call(method: str, path: str, request: dict, curl: str, response: dict) -> None:
    with st.expander(tr(f"API 呼叫：`{method} {path}`", f"API call: `{method} {path}`")):
        st.markdown(tr("**請求**（Python：`local_llm.gateway_client`）",
                       "**Request** (Python: `local_llm.gateway_client`)"))  # fmt: skip
        st.json(request)
        st.markdown(tr("**等效的 curl**", "**Equivalent curl**"))
        st.code(curl, "sh")
        st.markdown(tr("**回應**", "**Response**"))
        st.json(response, expanded=1)


# -- page -----------------------------------------------------------------------------------

status_panel()
tab_ocr, tab_doc, tab_rerank = st.tabs(
    [
        ":material/document_scanner: " + tr("OCR 比較", "Compare OCR"),
        ":material/description: " + tr("文件解析", "Document parsing"),
        ":material/sort: " + tr("重新排序（Rerank）", "Rerank"),
    ]
)

with tab_doc:
    left, right = st.columns([1, 1])
    with left:
        picked_doc = pick_file("document", "doc")
    with right:
        doc_model = model_select("doc", "doc_model")
        st.caption(
            tr(
                "Docling 標準流程：版面分析（Heron，Apple GPU）＋表格結構（TableFormer）；掃描頁面用 macOS Vision OCR（支援繁中）。支援 PDF、DOCX、PPTX、HTML、Markdown 與圖片。",
                "Docling standard pipeline: layout analysis (Heron, Apple GPU) + table structure (TableFormer); scanned pages use macOS Vision OCR (reads Traditional Chinese). Accepts PDF, DOCX, PPTX, HTML, Markdown and images.",
            )
        )
        go_doc = st.button(tr("解析", "Parse"), type="primary",
                           disabled=not (picked_doc and doc_model))  # fmt: skip
    if go_doc and picked_doc and doc_model:
        name, data = picked_doc

        def doc_work(job, name=name, data=data, model=doc_model):
            job.state["request"] = {"file": f"{name} ({len(data) / 1024:.0f} KB)", "model": model}
            job.state["result"] = gateway_client.parse_document(
                data, filename=name, model=model, base_url=URL
            )

        start("media_doc_job", tr(f"解析：{name}", f"Parse {name}"), doc_work)
    if job := current_job("media_doc_job"):
        res, req = job.state["result"], job.state["request"]
        st.divider()
        a, b, c, d = st.columns(4)
        a.metric(tr("頁數", "Pages"), res["pages"])
        b.metric(tr("表格", "Tables"), res["tables"])
        c.metric(tr("處理時間", "Processing time"), f"{res['seconds']:.1f} s")
        d.metric(tr("其中載入模型", "Of which model load"), f"{res.get('load_seconds', 0):.1f} s")
        rendered, raw = st.tabs([tr("排版後", "Rendered"), "Markdown"])
        with rendered:
            with st.container(height=520, border=True):
                st.markdown(res["markdown"])
        with raw:
            st.code(res["markdown"], "markdown", height=520)
        st.download_button(tr("下載 Markdown", "Download Markdown"), res["markdown"],
                           file_name=req["file"].split(" (")[0].rsplit(".", 1)[0] + ".md")  # fmt: skip
        md = res["markdown"]
        shown = {**res, "markdown": md[:400] + (f"… ({len(md):,} chars)" if len(md) > 400 else "")}
        api_call("POST", "/v1/documents/parse", req,
                 f"curl {URL}/v1/documents/parse -F file=@{req['file'].split(' (')[0]} -F model={req['model']}",
                 shown)  # fmt: skip

DEFAULT_PASSAGES = """乙方未依確認之交期交貨者，依逾期天數按逾期部分之貨款計算違約金：1 至 7 日每日 0.2%。
違約金總額以該批訂單貨款之 10% 為上限。
東昇機電通知，伺服馬達要延後兩週，預計十月二十四日才會到廠。
本合約以中華民國法律為準據法，以臺灣臺中地方法院為第一審管轄法院。
品保課已經查出原因，是三號車床的刀具磨耗。
員工餐廳本週三中午供應牛肉麵。
Late deliveries are charged 0.3% of the late amount per day from day 8 to day 14."""

with tab_ocr:
    ocr_compare.view()

with tab_rerank:
    st.caption(
        tr(
            "Reranker 會把「問題」和每一段文字一起讀，給出 0–1 的相關分數。RAG 先用向量搜尋找出幾十段候選，再用 reranker 排出最相關的幾段交給 LLM。",
            "A reranker reads the query together with each passage and scores relevance from 0 to 1. RAG first retrieves dozens of candidates with vector search, then a reranker picks the best few for the LLM.",
        )
    )
    left, right = st.columns([2, 1])
    with left:
        query = st.text_input(tr("問題", "Query"), "供應商延遲交貨要罰多少？", key="rr_query")
        passages_text = st.text_area(tr("候選段落（一行一段）", "Passages (one per line)"),
                                     DEFAULT_PASSAGES, height=200, key="rr_passages")  # fmt: skip
    with right:
        rr_model = model_select("rerank", "rr_model")
        passages = [p.strip() for p in passages_text.splitlines() if p.strip()]
        top_n = st.number_input("top_n", 1, max(len(passages), 1), max(len(passages), 1))
        go_rr = st.button(tr("排序", "Rerank"), type="primary",
                          disabled=not (rr_model and query.strip() and passages))  # fmt: skip
    if go_rr and rr_model and passages:

        def rr_work(job, query=query, passages=passages, model=rr_model, top_n=int(top_n)):
            job.state["request"] = {"model": model, "query": query, "documents": passages,
                                    "top_n": top_n}  # fmt: skip
            job.state["result"] = gateway_client.rerank(query, passages, model=model, top_n=top_n,
                                                        base_url=URL)  # fmt: skip

        start("media_rr_job", tr("Rerank", "Rerank"), rr_work)
    if job := current_job("media_rr_job"):
        results, req = job.state["result"], job.state["request"]
        st.divider()
        chart = pd.DataFrame(
            [{"passage": f"#{r['index']} {r['document'][:28]}", "score": r["relevance_score"]}
             for r in results]
        )  # fmt: skip
        st.bar_chart(chart, x="passage", y="score", horizontal=True, sort="-score",
                     x_label="", y_label=tr("相關分數", "Relevance score"))  # fmt: skip
        st.dataframe(
            pd.DataFrame([{tr("排名", "Rank"): i + 1, tr("原順序", "Index"): r["index"],
                           tr("分數", "Score"): round(r["relevance_score"], 4),
                           tr("段落", "Passage"): r["document"]} for i, r in enumerate(results)]),
            hide_index=True, width="stretch",
        )  # fmt: skip
        body = json.dumps({**req, "documents": [d[:20] + "…" for d in req["documents"]]},
                          ensure_ascii=False)  # fmt: skip
        api_call("POST", "/v1/rerank", req,
                 f"curl {URL}/v1/rerank -H 'content-type: application/json' -d '{body}'",
                 {"model": req["model"], "results": results})  # fmt: skip
