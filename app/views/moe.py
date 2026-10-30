"""MoE vs dense: what a mixture-of-experts model buys on this Mac (memory, generation speed,
prompt speed, TMMLU+ accuracy), from the measured pairs (data/lab/moe_benchmark.json, exported
by scripts/llm_eval/run_moe_dense.py), and a run of the same test (local_llm/moe_bench.py) on
the models chosen here, in the background."""

import json

import altair as alt
import jobs
import pandas as pd
import streamlit as st
import ui
from common import PROJECT_DIR, installed_names, kept, ollama_url, require_ollama, save_run
from i18n import tr

from local_llm import moe_bench

DATA = PROJECT_DIR / "data" / "lab" / "moe_benchmark.json"
KIND_COLORS = alt.Scale(domain=["MoE", "dense"], range=["#2563EB", "#94A3B8"])

ui.page_header(
    "account_tree",
    tr("MoE vs Dense", "MoE vs dense"),
    tr(
        "MoE（混合專家）模型每個字只動用一小部分參數：記憶體照總參數算，速度卻接近小模型。"
        "這裡比較同家族、參數量相近的 MoE 與一般（dense）模型，在這台 Mac 上的記憶體、速度與答題正確率。",
        "A mixture-of-experts model uses only a small part of its parameters for each token: "
        "memory follows the total, speed is close to a small model. Same-family MoE and dense "
        "models of similar size, compared on this Mac for memory, speed and accuracy.",
    ),
)
require_ollama()

MODES = {
    "results": (tr("實測結果", "Results"), "leaderboard"),
    "run": (tr("自己跑一次", "Run it"), "play_arrow"),
}
st.session_state.setdefault("moe_mode", "results")
mode = (
    st.segmented_control(
        tr("模式", "Mode"),
        list(MODES),
        format_func=lambda m: f":material/{MODES[m][1]}: {MODES[m][0]}",
        label_visibility="collapsed",
        **kept("moe_mode"),
    )
    or "results"
)


def principle() -> None:
    with st.expander(tr("MoE 的原理（30 秒版）", "How MoE works (in 30 seconds)")):
        st.markdown(
            tr(
                "- 一般模型（dense）每產生一個字，**全部**參數都要算一遍。\n"
                "- MoE 把中間的前饋層拆成很多「專家」（例如 qwen3:30b-a3b 有 128 個），每個字由路由器挑幾個（8 個）來算，"
                "所以 30B 的模型每個字只動用約 3B 參數（名稱裡的 **A3B** = active 3B）。\n"
                "- **記憶體**：所有專家都要放在記憶體裡，所以照總參數算（30B 級）。\n"
                "- **生成速度**：產生每個字要把用到的權重讀一遍，讀得少就快，所以接近 3B 模型。\n"
                "- **讀長文（prefill）**：計算量也只算用到的專家，在 Mac 上一樣快很多。\n"
                "- 適合記憶體夠大、但算力或頻寬有限的機器——正好是 Mac 的統一記憶體。",
                "- A dense model computes **all** its parameters for every token.\n"
                "- MoE splits the feed-forward layers into many experts (128 in qwen3:30b-a3b); a "
                "router picks a few (8) per token, so a 30B model uses about 3B per token (the "
                "**A3B** in the name = active 3B).\n"
                "- **Memory**: every expert must be loaded, so it follows the total (30B class).\n"
                "- **Generation speed**: each token reads the weights it uses, so it is close to a 3B model.\n"
                "- **Long prompts (prefill)**: the compute also counts only the experts used, so it is much faster on a Mac too.\n"
                "- Good for machines with plenty of memory but limited compute or bandwidth, such as a Mac's unified memory.",
            )
        )


def results_view() -> None:
    principle()
    try:
        data = json.loads(DATA.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        st.info(
            tr(
                "找不到實測結果（data/lab/moe_benchmark.json）。",
                "No results found (data/lab/moe_benchmark.json).",
            )
        )
        return
    st.markdown(
        tr(
            f"**{data['machine']}，Ollama，每個模型單獨載入**（{data['date']}）。A、B 組是同家族、同量化的 MoE 與 dense；"
            f"C 組不同家族，只當參考。正確率是 TMMLU+ {data['questions']} 題（台灣各類考試的選擇題）。",
            f"**{data['machine']}, Ollama, each model loaded alone** ({data['date']}). Pairs A and B "
            "are same-family, same-quantisation MoE and dense models; C mixes families and is for "
            f"reference only. Accuracy: {data['questions']} TMMLU+ questions (Taiwan exam multiple choice).",
        )
    )
    df = pd.DataFrame(data["rows"])
    df["label"] = df["pair"] + " · " + df["model"]
    df["active"] = df.apply(
        lambda r: (
            f"{r['expert_used_count']}/{r['expert_count']}"
            if pd.notna(r.get("expert_count"))
            else "—"
        ),
        axis=1,
    )

    def bars(column: str, title: str) -> alt.Chart:
        return alt.Chart(df).mark_bar().encode(
            y=alt.Y("label:N", title=None, sort=None), x=alt.X(f"{column}:Q", title=title),
            color=alt.Color("kind:N", title=None, scale=KIND_COLORS),
            tooltip=["model", "kind", column],
        ).properties(height=220)  # fmt: skip

    left, right = st.columns(2)
    left.altair_chart(
        bars(
            "gen_tok_s",
            tr("生成速度（tokens/秒，越長越快）", "Generation (tokens/s, longer = faster)"),
        ),
        width="stretch",
    )
    right.altair_chart(
        bars(
            "first_16k_s",
            tr(
                "讀 16K 長文到第一個字（秒，越短越好）",
                "16K prompt to first token (s, shorter = better)",
            ),
        ),
        width="stretch",
    )
    left.altair_chart(bars("size_gb", tr("記憶體（GB）", "Memory (GB)")), width="stretch")
    right.altair_chart(
        bars("tmmlu_accuracy", tr("TMMLU+ 正確率", "TMMLU+ accuracy")), width="stretch"
    )
    shown = df[["pair", "kind", "model", "params", "quant", "active", "size_gb", "gen_tok_s", "prefill_16k_tok_s",
                "first_16k_s", "tmmlu_accuracy"]].rename(columns={
        "pair": tr("組", "Pair"), "kind": tr("類型", "Type"), "model": tr("模型", "Model"),
        "params": tr("總參數", "Total"), "quant": tr("量化", "Quant"), "active": tr("專家（用／總）", "Experts (used/all)"),
        "size_gb": tr("記憶體 GB", "Memory GB"), "gen_tok_s": tr("生成 tok/s", "Gen tok/s"),
        "prefill_16k_tok_s": tr("讀長文 tok/s", "Prefill tok/s"), "first_16k_s": tr("16K 首字（秒）", "16K first (s)"),
        "tmmlu_accuracy": "TMMLU+",
    })  # fmt: skip
    st.dataframe(shown, hide_index=True, width="stretch",
                 column_config={"TMMLU+": st.column_config.NumberColumn(format="percent")})  # fmt: skip
    st.markdown(
        tr(
            "**結論**：參數量相近時，MoE 的答題正確率與 dense 差不到 3 個百分點（200 題的誤差範圍內），"
            "但生成快 2.7～7 倍、讀長文快 4～6 倍；記憶體則照總參數算，不會比較省。",
            "**Takeaway**: at similar size, MoE answers within 3 points of dense (inside the "
            "error margin of 200 questions), but generates 2.7-7x and reads long prompts 4-6x "
            "faster; memory follows the total parameters, so it is not smaller.",
        )
    )
    st.caption(
        tr(
            "記憶體是 Ollama 回報的載入量，含 20K context 的 KV 快取。生成：中文短問題、256 個字、3 次取中位數；"
            "讀長文：約 4K 與 16K token 的中文文件。思考一律關閉（gpt-oss 關不掉，用最低強度）。"
            "腳本：scripts/llm_eval/run_moe_dense.py；方法：src/local_llm/moe_bench.py。",
            "Memory is what Ollama reports when loaded, including the KV cache for a 20K context. "
            "Generation: a short Chinese question, 256 tokens, median of 3; prefill: Chinese "
            "documents of about 4K and 16K tokens. Thinking off (gpt-oss cannot, so it reasons "
            "at low). Script: scripts/llm_eval/run_moe_dense.py; method: src/local_llm/moe_bench.py.",
        )
    )


def moe_work(job: jobs.Job, models: list[str], n_questions: int, base_url: str) -> None:
    items = moe_bench.questions(n_questions) if n_questions else []
    for row in job.state["results"]:
        if job.cancel.is_set():
            break
        model = row["model"]
        row["status"] = "running"
        try:
            row["speed"] = moe_bench.speed(model, base_url=base_url,
                                           progress=lambda step, m=model: setattr(job, "detail", f"{m}：{step}"))  # fmt: skip
            if items:
                q = moe_bench.quality(model, items, base_url=base_url,
                                      progress=lambda i, n, m=model: setattr(job, "detail", f"{m}：TMMLU+ {i + 1}/{n}"),
                                      cancelled=job.cancel.is_set)  # fmt: skip
                row["quality"] = {k: v for k, v in q.items() if k != "answers"}
            row["status"] = "done"
        except Exception as exc:
            row["status"], row["error"] = "error", str(exc)[:300]
        job.progress = sum(r["status"] in ("done", "error") for r in job.state["results"]) / len(
            models
        )
    moe_bench.unload_all(base_url)
    job.state["path"] = str(
        save_run("moe", {"questions": n_questions, "results": job.state["results"]})
    )
    job.status = "cancelled" if job.cancel.is_set() else "done"


def render(job: jobs.Job) -> None:
    if job.running:
        st.progress(job.progress or 0.0, text=job.detail)
    rows = []
    for r in job.state["results"]:
        s, q = r.get("speed") or {}, r.get("quality") or {}
        rows.append({tr("模型", "Model"): r["model"], tr("類型", "Type"): s.get("kind", moe_bench.MODELS.get(r["model"], ("", "?"))[1]),
                     tr("狀態", "Status"): r["status"], tr("記憶體 GB", "Memory GB"): s.get("size_gb"),
                     tr("生成 tok/s", "Gen tok/s"): s.get("gen_tok_s"), tr("16K 首字（秒）", "16K first (s)"): s.get("first_16k_s"),
                     "TMMLU+": q.get("accuracy"), tr("錯誤", "Error"): r.get("error")})  # fmt: skip
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={"TMMLU+": st.column_config.NumberColumn(format="percent")})  # fmt: skip
    if job.state.get("path"):
        st.caption(tr(f"已儲存：`{job.state['path']}`", f"Saved: `{job.state['path']}`"))


@st.fragment(run_every=1)
def live(job: jobs.Job) -> None:
    if not job.running:
        st.rerun()
    render(job)


def run_view() -> None:
    installed = set(installed_names())
    known = [m for m in moe_bench.MODELS if m in installed]
    others = sorted(m for m in installed if m not in moe_bench.MODELS)

    def label(model: str) -> str:
        if model in moe_bench.MODELS:
            pair, kind = moe_bench.MODELS[model]
            return f"{model}（{pair} 組 {kind}）"
        return model

    models = st.multiselect(tr("模型（建議成對選）", "Models (best chosen in pairs)"), known + others,
                            default=[m for m in ("qwen3:30b-a3b", "qwen3:32b") if m in installed],
                            format_func=label, **kept("moe_models"))  # fmt: skip
    n = st.select_slider(tr("TMMLU+ 題數（0 = 只測速度）", "TMMLU+ questions (0 = speed only)"),
                         options=[0, 20, 50, 100, 200], value=20, **kept("moe_questions"))  # fmt: skip
    st.caption(
        tr(
            "每個模型會單獨載入（其他模型先卸載），速度測試一個約 1～2 分鐘；dense 32B 讀 16K 長文要一分多鐘。"
            "題目第一次會從 Hugging Face 下載 TMMLU+（約 10 MB）。",
            "Each model is loaded alone (others unloaded first); the speed test takes about 1-2 "
            "minutes per model, a dense 32B needs over a minute for the 16K prompt alone. The "
            "first run downloads TMMLU+ from Hugging Face (about 10 MB).",
        )
    )
    job = jobs.get(st.session_state.get("moe_job"))
    busy = bool(job and job.running)
    if st.button(tr("開始測試", "Run"), type="primary", disabled=busy or not models):
        job = jobs.Job("moe", tr(f"MoE 測試 {len(models)} 個模型", f"MoE test, {len(models)} models"), "views/moe.py",
                       state={"results": [{"model": m, "status": "waiting"} for m in models]})  # fmt: skip
        url = ollama_url()
        jobs.start(job, lambda j: moe_work(j, models, n, url))
        st.session_state.moe_job = job.id
        st.rerun()
    if job and job.running:
        live(job)
    elif job:
        render(job)


if mode == "run":
    run_view()
else:
    results_view()
