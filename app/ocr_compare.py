"""OCR tab of the Speech & documents page: read one image with several local OCR engines
(local_llm/ocr.py), side by side, timed, optionally scored by MiniCPM-OCR Jev; and the benchmark
on OmniDocBench pages (data/lab/ocr_benchmark.json, exported by scripts/ocr_eval/run_ocr.py)."""

import json

import jobs
import pandas as pd
import streamlit as st
from common import PROJECT_DIR, gateway_url, installed_names, kept, ollama_url
from i18n import tr

from local_llm import ocr

DATA = PROJECT_DIR / "data" / "lab" / "ocr_benchmark.json"
TEST_PAGES = PROJECT_DIR / "storage" / "ocr_eval" / "pages.json"
TEST_TRUTH = PROJECT_DIR / "storage" / "ocr_eval" / "gt.json"
LANGS = {
    "traditional_chinese": ("繁中", "Traditional Chinese"),
    "simplified_chinese": ("簡中", "Simplified Chinese"),
    "english": ("英文", "English"),
    "en_ch_mixed": ("中英混合", "Mixed"),
}

def engine_label(engine: str) -> str:
    e = ocr.ENGINES[engine]
    size = f"，{e['size_gb']} GB" if e["size_gb"] else ""
    return f"{e['label']}（{e['runtime']}{size}）"


# -- benchmark ---------------------------------------------------------------------------------


def bench_view() -> None:
    try:
        data = json.loads(DATA.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        st.info(tr("找不到評測結果（data/lab/ocr_benchmark.json）。",
                   "No benchmark results found (data/lab/ocr_benchmark.json)."))  # fmt: skip
        return
    st.markdown(
        tr(
            f"**OmniDocBench {data['pages']} 頁（繁中 {data['by_language']['traditional_chinese']}、簡中 "
            f"{data['by_language']['simplified_chinese']}、英文 {data['by_language']['english']}、中英混合 "
            f"{data['by_language']['en_ch_mixed']}）與其中 {data['real5_pages']} 頁的實拍版**（{data['date']}，"
            f"{data['machine']}）。數字是和人工標註的差距（正規化編輯距離，0 = 完全一樣，越低越好）。",
            f"**{data['pages']} OmniDocBench pages ({data['by_language']['traditional_chinese']} Traditional "
            f"Chinese, {data['by_language']['simplified_chinese']} Simplified, {data['by_language']['english']} "
            f"English, {data['by_language']['en_ch_mixed']} mixed) and {data['real5_pages']} of them "
            f"photographed** ({data['date']}, {data['machine']}). Numbers are the distance to the "
            "human annotation (normalised edit distance, 0 = identical, lower is better).",
        )
    )
    rows = []
    for r in data["rows"]:
        rows.append({
            tr("模型", "Model"): r["label"], tr("執行", "Runtime"): r["runtime"],
            tr("大小（GB）", "Size (GB)"): r["size_gb"],
            tr("全部（乾淨）", "All (clean)"): r["clean"],
            **{tr(*LANGS[k]): r["by_language"].get(k) for k in LANGS},
            tr("實拍", "Photographed"): r.get("real5"),
            tr("同頁乾淨", "Same pages clean"): r.get("real5_clean"),
            tr("秒／頁", "s / page"): r["s_page"],
        })  # fmt: skip
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.3f") for c in df.columns
                                if c not in (tr("模型", "Model"), tr("執行", "Runtime"),
                                             tr("大小（GB）", "Size (GB)"), tr("秒／頁", "s / page"))})  # fmt: skip
    chart = df.melt(id_vars=[tr("模型", "Model")], value_vars=[tr("全部（乾淨）", "All (clean)"), tr("實拍", "Photographed")],
                    var_name=tr("頁面", "Pages"), value_name=tr("差距", "Distance"))  # fmt: skip
    st.bar_chart(chart, x=tr("模型", "Model"), y=tr("差距", "Distance"), color=tr("頁面", "Pages"),
                 stack=False, horizontal=True)  # fmt: skip
    if data.get("notes"):
        st.markdown(tr(*data["notes"]))
    st.caption(
        tr(
            "每個模型都是整頁一次送進去（不先切版面）；官方 MiniCPM-OCR、GLM-OCR 流程會先用版面模型切塊再逐塊辨識，分數會更好。"
            "計分是自製的簡化版（去掉 Markdown／HTML／LaTeX 標記、忽略空白大小寫），不能和官方公布的 OmniDocBench 分數直接比。"
            "實拍：Real5-OmniDocBench 的光照、掃描、螢幕翻拍、歪斜、彎曲，輪流各一種；繁中頁面沒有實拍版。"
            "腳本：scripts/ocr_eval/run_ocr.py。",
            "Every model reads the whole page in one request (no layout model first); the official "
            "MiniCPM-OCR and GLM-OCR pipelines cut the page into regions first and score higher. "
            "The scoring is our own simplification (Markdown / HTML / LaTeX removed, whitespace and "
            "case ignored), not comparable with published OmniDocBench scores. Photographed: "
            "Real5-OmniDocBench illumination, scanning, screen photo, skew and warping in turn; the "
            "Traditional Chinese pages have no photographed version. Script: scripts/ocr_eval/run_ocr.py.",
        )
    )
    jev = data.get("jev")
    if jev:
        st.subheader(
            tr("MiniCPM-OCR Jev：自動判斷 OCR 對不對", "MiniCPM-OCR Jev: is the OCR right?")
        )
        st.markdown(
            tr(
                f"從上面的乾淨頁面切出 {jev['regions']} 個文字區塊，各由 MiniCPM-OCR 與 macOS Vision 辨識一次"
                f"（共 {jev['candidates']} 個結果，{jev['correct']} 個與標註完全相同）。Jev 的分數分辨對錯的能力 "
                f"**AUROC {jev['auroc']}**（0.5 = 亂猜，1 = 完全分得開）；每個結果打分數中位數 {jev['jev_s']} 秒。",
                f"{jev['regions']} text regions cut from the clean pages above, each read by MiniCPM-OCR "
                f"and macOS Vision ({jev['candidates']} results, {jev['correct']} identical to the "
                f"annotation). How well Jev's score separates right from wrong: **AUROC {jev['auroc']}** "
                f"(0.5 = guessing, 1 = perfect); median {jev['jev_s']} s per result.",
            )
        )
        pick = pd.DataFrame([
            {tr("做法", "Method"): "MiniCPM-OCR", tr("完全正確", "Exact"): jev["minicpm-ocr"]["exact"],
             tr("平均差距", "Mean distance"): jev["minicpm-ocr"]["ned"]},
            {tr("做法", "Method"): "macOS Vision", tr("完全正確", "Exact"): jev["ocrmac"]["exact"],
             tr("平均差距", "Mean distance"): jev["ocrmac"]["ned"]},
            {tr("做法", "Method"): tr("兩個挑 Jev 分數高的", "Higher Jev score of the two"),
             tr("完全正確", "Exact"): jev["jev_pick"]["exact"], tr("平均差距", "Mean distance"): jev["jev_pick"]["ned"]},
            {tr("做法", "Method"): tr("上限：兩個有一個對", "Ceiling: either is right"),
             tr("完全正確", "Exact"): jev["oracle"]["exact"], tr("平均差距", "Mean distance"): None},
        ])  # fmt: skip
        st.dataframe(pick, hide_index=True, width="stretch")
        accept = pd.DataFrame([
            {tr("自動通過門檻", "Auto-accept above"): t, tr("通過比例", "Passed"): jev[f"accept_{t}"]["pass_rate"],
             tr("通過的有多少真的對", "Of those, right"): jev[f"accept_{t}"]["precision"]}
            for t in (0.5, 0.8, 0.9)
        ])  # fmt: skip
        st.dataframe(accept, hide_index=True, width="stretch")
        st.caption(
            tr(
                "Jev 要一張「把候選文字畫出來的圖」；官方用 Linux 上的 Chromium 渲染，這裡用 Pillow 與系統中文字型，分數可能因此偏移。"
                "只測文字區塊（不含公式、表格）。腳本：scripts/ocr_eval/run_jev.py。",
                "Jev also takes the candidate text drawn as an image; the official service renders "
                "with Chromium on Linux, here Pillow and a system CJK font, which may move the "
                "scores. Text regions only (no formulas or tables). Script: scripts/ocr_eval/run_jev.py.",
            )
        )


# -- try it -------------------------------------------------------------------------------------


def ocr_work(job: jobs.Job, image: bytes, engines: list[str], score: bool, truth: str | None,
             gateway: str, ollama: str) -> None:  # fmt: skip
    import time

    for engine, row in zip(engines, job.state["results"], strict=True):
        if job.cancel.is_set():
            break
        row["status"] = "running"
        job.detail = ocr.ENGINES[engine]["label"]
        start = time.perf_counter()
        try:
            row["text"] = ocr.read(engine, image, gateway=gateway, ollama=ollama)
            row["seconds"] = round(time.perf_counter() - start, 2)
            if truth is not None:
                row["distance"] = round(ocr.distance(row["text"], truth), 3)
            row["status"] = "done"
        except Exception as exc:  # show it and go on with the next engine
            row["status"], row["error"] = "error", str(exc)[:300]
        job.progress = sum(r["status"] in ("done", "error") for r in job.state["results"]) / len(
            engines
        )
    if score and not job.cancel.is_set():
        job.detail = "Jev"
        try:
            for row in job.state["results"]:
                if row.get("text"):
                    row["jev"] = round(ocr.jev_score(image, row["text"]), 3)
        except Exception as exc:
            job.state["jev_error"] = str(exc)[:300]
        finally:
            ocr.JEV_SERVER.stop()  # 2.7 GB; started again when needed
    job.status = "cancelled" if job.cancel.is_set() else "done"


def render(job: jobs.Job) -> None:
    results = job.state["results"]
    if job.running:
        st.progress(job.progress or 0.0, text=job.detail)
    cols = st.columns(min(len(results), 3))
    for k, row in enumerate(results):
        with cols[k % len(cols)]:
            with st.container(border=True):
                st.markdown(f"**{ocr.ENGINES[row['engine']]['label']}**")
                facts = []
                if row.get("seconds") is not None:
                    facts.append(tr(f"{row['seconds']} 秒", f"{row['seconds']} s"))
                if row.get("distance") is not None:
                    facts.append(tr(f"與標註差距 {row['distance']}", f"distance {row['distance']}"))
                if row.get("jev") is not None:
                    facts.append(f"Jev {row['jev']}")
                if facts:
                    st.caption(" · ".join(facts))
                if row["status"] == "error":
                    st.error(row["error"])
                elif row.get("text") is not None:
                    st.code(row["text"] or "（空白）", language=None, wrap_lines=True, height=360)
                else:
                    st.caption(tr("等待中…" if row["status"] == "waiting" else "辨識中…",
                                  "Waiting…" if row["status"] == "waiting" else "Reading…"))  # fmt: skip
    if job.state.get("jev_error"):
        st.warning("Jev: " + job.state["jev_error"])
    if job.state.get("score") and not job.running:
        st.caption(
            tr(
                "Jev 是為單一區塊（一段文字、一個標題）設計的；整頁圖片的分數僅供參考。",
                "Jev is made for one region (a paragraph, a title); on a whole page its score is only indicative.",
            )
        )


@st.fragment(run_every=1)
def live(job: jobs.Job) -> None:
    if not job.running:
        st.rerun()
    render(job)


def try_view() -> None:
    sources = {"sample": tr("內建範例（繁中會議紀錄）", "Built-in sample (Chinese minutes)"),
               "upload": tr("上傳圖片", "Upload an image")}  # fmt: skip
    pages = json.loads(TEST_PAGES.read_text()) if TEST_PAGES.exists() else []
    if pages:
        sources["test"] = tr("評測用的 OmniDocBench 頁面", "An OmniDocBench test page")
    source = st.radio(
        tr("圖片", "Image"),
        list(sources),
        format_func=sources.get,
        horizontal=True,
        **kept("ocr_source"),
    )
    image, truth = None, None
    if source == "sample":
        image, truth = ocr.sample_image(), "\n".join(ocr.SAMPLE_TEXT)
    elif source == "upload":
        up = st.file_uploader(
            tr("圖片（PNG／JPG）", "Image (PNG / JPG)"), type=["png", "jpg", "jpeg"]
        )
        image = up.getvalue() if up else None
    else:

        def page_label(i: int) -> str:
            p = pages[i]
            kind = f" · {p['real5_kind']}" if p.get("real5_kind") else ""
            return f"{i + 1}. {tr(*LANGS[p['language']])} · {p['source']}{kind}"

        i = st.selectbox(
            tr("頁面", "Page"), range(len(pages)), format_func=page_label, **kept("ocr_page")
        )
        photographed = bool(pages[i].get("real5")) and st.toggle(tr("用實拍版", "Photographed version"),
                                                                 **kept("ocr_real5"))  # fmt: skip
        image = open(pages[i]["real5" if photographed else "clean"], "rb").read()
        truth_pages = {p["page_info"]["image_path"]: p for p in json.loads(TEST_TRUTH.read_text())}
        truth = ocr.omnidoc_truth(truth_pages[pages[i]["page"]])
    if image:
        st.image(image, width=480 if source != "sample" else "content")

    installed = set(installed_names())
    usable, why = {}, {}
    for engine, e in ocr.ENGINES.items():
        ok, reason = ocr.available(engine)
        if (
            e.get("ollama")
            and e["ollama"] not in installed
            and f"{e['ollama']}:latest" not in installed
        ):
            ok, reason = False, f"ollama pull {e['ollama']}"
        usable[engine] = ok
        why[engine] = reason
    default = [e for e in ("minicpm-ocr", "glm-ocr", "ocrmac") if usable[e]]
    engines = st.multiselect(tr("OCR 模型", "OCR models"), [e for e in ocr.ENGINES if usable[e]],
                             default=default, format_func=engine_label, **kept("ocr_engines"))  # fmt: skip
    missing = [f"{ocr.ENGINES[e]['label']}：`{why[e]}`" for e in ocr.ENGINES if not usable[e]]
    if missing:
        st.caption(tr("還不能用：", "Not available yet: ") + "；".join(missing))
    score = st.toggle(tr("用 MiniCPM-OCR Jev 打正確性分數", "Score with MiniCPM-OCR Jev"),
                      disabled=not ocr.JEV_SERVER.model.exists(), **kept("ocr_jev"))  # fmt: skip

    job = jobs.get(st.session_state.get("ocr_job"))
    busy = bool(job and job.running)
    if st.button(tr("開始辨識", "Read"), type="primary", disabled=busy or not (image and engines)):
        job = jobs.Job("ocr", tr(f"OCR 比較 {len(engines)} 個模型", f"Compare {len(engines)} OCR models"),
                       "views/ocr.py",
                       state={"score": score,
                              "results": [{"engine": e, "status": "waiting", "text": None} for e in engines]})  # fmt: skip
        gw, ol = gateway_url(), ollama_url()
        jobs.start(job, lambda j: ocr_work(j, image, engines, score, truth, gw, ol))
        st.session_state.ocr_job = job.id
        st.rerun()
    if job and job.running:
        live(job)
    elif job:
        render(job)


def view() -> None:
    """The tab: a mode switch, then the chosen view."""
    st.caption(
        tr(
            "同一張圖交給多個 OCR 模型（一次一個，時間才公平），並排比較讀出的文字與速度；"
            "也能請 MiniCPM-OCR Jev 替每個結果打 0～1 的正確性分數。",
            "Give the same image to several OCR models in turn (one at a time, so timings are "
            "fair) and compare the text and speed; MiniCPM-OCR Jev can score each result from 0 to 1.",
        )
    )
    MODES = {
        "try": (tr("試跑", "Try it"), "play_arrow"),
        "bench": (tr("評測總表", "Benchmark"), "leaderboard"),
    }
    st.session_state.setdefault("ocr_mode", "try")
    mode = (
        st.segmented_control(
            tr("模式", "Mode"),
            list(MODES),
            format_func=lambda m: f":material/{MODES[m][1]}: {MODES[m][0]}",
            label_visibility="collapsed",
            **kept("ocr_mode"),
        )
        or "try"
    )


    if mode == "bench":
        bench_view()
    else:
        try_view()
