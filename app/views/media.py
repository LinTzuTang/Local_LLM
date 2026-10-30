"""Speech & documents: try the model gateway's speech-to-text, document parsing and reranker, and
compare OCR models (ocr_compare.py).

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
AUDIO_MIME = {"mp3": "audio/mpeg", "m4a": "audio/mp4", "mp4": "audio/mp4", "mov": "video/quicktime"}
TASK_LABELS = {
    "asr": tr("語音轉文字", "Speech to text"),
    "doc": tr("文件解析", "Document parsing"),
    "rerank": "Rerank",
    "diarize": tr("說話者分段", "Diarization"),
}

ui.page_header(
    "mic",
    tr("語音與文件", "Speech & documents"),
    tr(
        "Ollama 只跑 LLM 與 embedding。語音轉文字、文件解析、reranker 由本機的**模型閘道**提供，用法和 Ollama 一樣：一鍵下載、第一次使用時載入、閒置後自動卸載。資料不會離開這台電腦；每個結果下方都列出實際呼叫的 API。",
        "Ollama only runs LLMs and embeddings. Speech-to-text, document parsing and reranking come from the local **model gateway**, which behaves like Ollama: one-click download, loaded on first use, unloaded when idle. Nothing leaves this Mac; every result shows the API call behind it.",
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


def clock(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{seconds % 60:05.2f}"


# -- page -----------------------------------------------------------------------------------

status_panel()
tab_asr, tab_doc, tab_ocr, tab_rerank, tab_inbox = st.tabs(
    [
        ":material/mic: " + tr("語音轉文字", "Speech to text"),
        ":material/description: " + tr("文件解析", "Document parsing"),
        ":material/document_scanner: " + tr("OCR 比較", "Compare OCR"),
        ":material/sort: " + tr("重新排序（Rerank）", "Rerank"),
        ":material/inbox: " + tr("收件匣（給 agent）", "Inbox (for agents)"),
    ]
)

with tab_asr:
    st.page_link(
        "views/speech.py",
        label=tr(
            "想並排比較多個語音模型、測噪音與電話音質、翻成英文或分辨會議講者？到「語音模型比較」",
            "Compare several speech models side by side, test noise and telephone audio, "
            "translate to English or tell meeting speakers apart: Compare speech models",
        ),
        icon=":material/graphic_eq:",
    )
    left, right = st.columns([1, 1])
    with left:
        picked = pick_file("audio", "asr")
        if picked:
            ext = picked[0].rsplit(".", 1)[-1].lower()
            st.audio(picked[1], format=AUDIO_MIME.get(ext, f"audio/{ext}"))
    with right:
        asr_model = model_select("asr", "asr_model")
        languages = {
            "zh": tr("中文", "Chinese"),
            "en": tr("英文", "English"),
            "auto": tr("自動偵測", "Detect"),
        }
        language = st.segmented_control(tr("語言", "Language"), list(languages), required=True,
                                        format_func=languages.get, default="zh", key="asr_lang")  # fmt: skip
        prompt = st.text_input(
            tr("提示（initial prompt，選填）", "Prompt (initial prompt, optional)"),
            "以下是繁體中文的會議錄音。" if language == "zh" else "",
            key=f"asr_prompt_{language}",
            help=tr(
                "Whisper 會延續提示的寫法：寫繁體中文可讓 Whisper 輸出繁體；也可以放人名、產品名等專有名詞。Breeze-ASR-25 本身就輸出繁體。",
                "Whisper continues the prompt's style: a Traditional-Chinese prompt steers Whisper to Traditional characters; names and product terms help too. Breeze-ASR-25 outputs Traditional Chinese by itself.",
            ),
        )
        go = st.button(tr("轉成文字", "Transcribe"), type="primary",
                       disabled=not (picked and asr_model))  # fmt: skip
    if go and picked and asr_model:
        name, data = picked
        lang = None if language == "auto" else language

        def asr_work(job, name=name, data=data, model=asr_model, lang=lang, prompt=prompt):
            job.state["request"] = {"file": f"{name} ({len(data) / 1024:.0f} KB)", "model": model,
                                    "language": lang, "prompt": prompt or None}  # fmt: skip
            job.state["result"] = gateway_client.transcribe(
                data, model=model, language=lang, filename=name, prompt=prompt or None,
                base_url=URL,
            )  # fmt: skip

        start("media_asr_job", tr(f"轉文字：{name}", f"Transcribe {name}"), asr_work)
    if job := current_job("media_asr_job"):
        res, req = job.state["result"], job.state["request"]
        st.divider()
        a, b, c, d = st.columns(4)
        a.metric(tr("音訊長度", "Audio length"), f"{res['duration']:.1f} s")
        b.metric(tr("處理時間", "Processing time"), f"{res['seconds']:.1f} s")
        c.metric(tr("速度", "Speed"), f"{res['duration'] / max(res['seconds'], 0.01):.1f}×",
                 help=tr("音訊長度 ÷ 處理時間", "audio length ÷ processing time"))  # fmt: skip
        d.metric(tr("其中載入模型", "Of which model load"), f"{res.get('load_seconds', 0):.1f} s")
        st.markdown(f"**{res['model']}** · {tr('語言', 'language')} `{res['language']}`")
        st.text_area(tr("逐字稿", "Transcript"), res["text"], height=150)
        segments = pd.DataFrame(
            [{tr("開始", "Start"): clock(s["start"]), tr("結束", "End"): clock(s["end"]),
              tr("內容", "Text"): s["text"]} for s in res["segments"]]
        )  # fmt: skip
        st.dataframe(segments, hide_index=True, width="stretch", height=280)
        lines = "\n".join(
            f"[{clock(s['start'])}-{clock(s['end'])}] {s['text']}" for s in res["segments"]
        )
        st.download_button(tr("下載逐字稿（含時間軸）", "Download transcript (with timestamps)"),
                           lines, file_name="transcript.txt")  # fmt: skip
        form = " ".join(f"-F {k}={json.dumps(v, ensure_ascii=False)}" for k, v in req.items()
                        if v and k != "file")  # fmt: skip
        shown = {
            **res,
            "segments": res["segments"][:3]
            + ([f"… {len(res['segments']) - 3} more"] if len(res["segments"]) > 3 else []),
        }
        api_call("POST", "/v1/audio/transcriptions", req,
                 f"curl {URL}/v1/audio/transcriptions -F file=@{req['file'].split(' (')[0]} {form}",
                 shown)  # fmt: skip

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

with tab_inbox:
    st.markdown(
        tr(
            "Agent 看不到你的硬碟，只能透過工具讀 **收件匣**（`storage/inbox/`）裡的檔案：`list_inbox_files` 列出檔案、`transcribe_audio` 轉逐字稿、`parse_document` 解析文件；後兩個呼叫上面同一個模型閘道。在 Agent 頁選「會議與文件助理」即可使用。",
            "Agents cannot see your disk; they read files in the **inbox** (`storage/inbox/`) through tools: `list_inbox_files`, `transcribe_audio` and `parse_document` (the last two call the same model gateway as above). Pick the 'Meetings & documents assistant' on the Agent page.",
        )
    )
    uploads = st.file_uploader(tr("加入檔案", "Add files"), type=AUDIO_TYPES + DOC_TYPES,
                               accept_multiple_files=True, key="inbox_upload")  # fmt: skip
    a, b, c = st.columns([1, 1, 2])
    if a.button(tr("放進收件匣", "Add to inbox"), type="primary", disabled=not uploads):
        saved = [media_tools.save(u.name, u.getvalue()) for u in uploads]
        st.success(tr("已加入：", "Added: ") + "、".join(saved))
    if b.button(tr("放入範例檔", "Add the samples")):
        st.success(tr("已加入：", "Added: ") + "、".join(media_tools.add_samples()))
    if c.button(
        tr("用「會議與文件助理」分析", "Analyse with the meetings & documents assistant"),
        icon=":material/smart_toy:",
    ):
        st.session_state.agent_type = "meeting_docs"  # preselects it on the Agent page
        st.switch_page("views/agent.py")
    files = media_tools.list_files()
    if files:
        kinds = {
            "audio": tr("音訊", "audio"),
            "document": tr("文件", "document"),
            "other": tr("其他", "other"),
        }
        st.dataframe(
            pd.DataFrame([{tr("檔名", "Name"): f["name"], tr("類型", "Type"): kinds[f["kind"]],
                           "KB": f["size_kb"], tr("修改時間", "Modified"): f["modified"]}
                          for f in files]),
            hide_index=True, width="stretch",
        )  # fmt: skip
        d1, d2 = st.columns([3, 1])
        victim = d1.selectbox(tr("刪除檔案", "Delete a file"), [f["name"] for f in files],
                              label_visibility="collapsed")  # fmt: skip
        if d2.button(tr("刪除", "Delete"), key="inbox_delete"):
            media_tools.remove(victim)
            st.rerun()
    else:
        st.info(tr("收件匣是空的。", "The inbox is empty."))
