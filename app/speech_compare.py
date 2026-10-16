"""Transcribe mode of the speech-model page (views/speech.py), and saved speech runs.

The same recording goes to several ASR models in turn (one at a time, so the timings are
fair): the gateway's Whisper-family, Qwen3-ASR, SenseVoice and Fun-ASR models, and Gemma 4
through Ollama. Results sit side by side with processing time, real-time factor and, when a
reference transcript is given, the character error rate (asr_metrics.cer).

Conditions: the same audio can also be run with background noise (white, pink, babble or
office, at a chosen SNR) and as telephone audio (local_llm.audio_aug); the augmented WAVs are
cached under storage/speech_cache, and the results become a model x condition CER table.

Every speech run is saved as `runs/asr_compare-*.json` with a `mode` field ("transcribe",
"translate", "meeting", "live", "tts", "voice_chat"; old runs have none and are
transcriptions); `history_view` and `history_summary` draw them on the History page.
"""

import html
from pathlib import Path

import altair as alt
import jobs
import pandas as pd
import speech_common as sc
import speech_live
import speech_meeting
import speech_translate
import speech_tts
import speech_voice
import streamlit as st
from common import kept, save_run
from i18n import tr

from local_llm import asr_metrics, audio_aug, gateway_client

MAX_MODELS = 10
SAVED = ("model", "condition", "text", "seconds", "load_s", "compute_s", "duration", "rtf",
         "segments", "truncated", "script", "score", "drift", "error")  # fmt: skip
SCRIPTS = {"traditional": ("繁體", "Traditional"), "simplified": ("簡體", "Simplified")}
SNRS = [0, 5, 10, 15, 20, 30]
CONDITIONS = ("clean", "noise", "phone", "noise_phone")


# -- conditions (pure) ------------------------------------------------------------------------


def condition_key(c: dict) -> str:
    if c["id"] == "clean":
        return "clean"
    if c["id"] == "phone":
        return "phone"
    noise = f"{c['kind']}{c['snr']}db"
    return noise if c["id"] == "noise" else f"phone_{noise}"


def make_conditions(chosen: list[str], kind: str, snr: int) -> list[dict]:
    """Condition dicts in a fixed order (clean first, so drift can be measured against it)."""
    out = []
    for cid in CONDITIONS:
        if cid in chosen:
            c = {"id": cid} | ({"kind": kind, "snr": snr} if "noise" in cid else {})
            out.append(c | {"key": condition_key(c)})
    return out


def noise_name(kind: str) -> str:
    return {
        "white": tr("白噪音", "white noise"),
        "pink": tr("粉紅噪音", "pink noise"),
        "babble": tr("人聲嘈雜", "babble"),
        "office": tr("辦公室", "office"),
    }.get(kind, kind)


def condition_label(c: dict | None) -> str:
    if not c or c.get("id", "clean") == "clean":
        return tr("乾淨", "Clean")
    if c["id"] == "phone":
        return tr("電話音質", "Telephone")
    noise = f"{noise_name(c['kind'])} {c['snr']} dB"
    if c["id"] == "noise":
        return noise[:1].upper() + noise[1:]
    return tr(f"電話＋{noise}", f"Telephone + {noise}")


def prepare_conditions(
    source: Path, conditions: list[dict], babble: list[Path], cache_dir: Path
) -> None:
    """Write each condition's WAV (cached by content hash and condition); sets c["file"]."""
    samples, folder = None, cache_dir / sc.cache_key(source)
    for c in conditions:
        if c["id"] == "clean":
            c["file"] = str(source)
            continue
        out = folder / f"{c['key']}.wav"
        if not out.exists():
            if samples is None:
                samples = audio_aug.load(source)
            x = samples
            if "noise" in c["id"]:
                x = audio_aug.add_noise(x, c["snr"], c["kind"], seed=0, babble_sources=babble)
            if "phone" in c["id"]:
                x = audio_aug.telephone(x)
            audio_aug.save(out, x)
        c["file"] = str(out)


# -- worker (no Streamlit calls) ------------------------------------------------------------


def asr_work(job: jobs.Job, url: str) -> None:
    """Transcribe with each model (under each condition), score, save the run."""
    state = job.state
    results, reference, conditions = state["results"], state["reference"], state["conditions"]
    if len(conditions) > 1:
        job.detail = "noise / telephone"
        source = Path(state["audio"]) if isinstance(state["audio"], str) else state["audio"]
        prepare_conditions(source, conditions, state["babble"], sc.CACHE_DIR)
    files = {c["key"]: c.get("file") for c in conditions}
    for i, r in enumerate(results, start=1):
        if job.cancel.is_set():
            break
        r["status"] = "running"
        job.detail = f"{i}/{len(results)} {r['model']}"
        if len(conditions) > 1:
            job.detail += f" · {r['condition']}"
        job.progress = (i - 1) / len(results)
        audio = Path(files[r["condition"]]) if len(conditions) > 1 else state["audio"]
        if isinstance(audio, str):
            audio = Path(audio)
        try:
            out = gateway_client.transcribe(
                audio, r["model"], state["language"] or None,
                filename=state["filename"], prompt=state["prompt"] or None,
                keep_alive=0 if state["unload"] and r is _last_of(results, r) else None,
                base_url=url,
            )  # fmt: skip
        except Exception as exc:  # shown in the model's column
            r.update(status="done", error=str(exc))
            continue
        # Time the model spent on this audio: without loading it and without waiting for
        # another speech model to free the GPU.
        load = (out.get("load_seconds") or 0) + (out.get("ollama_load_seconds") or 0)
        compute = max(out["seconds"] - load - (out.get("queue_seconds") or 0), 0.01)
        duration = out.get("duration") or 0
        r.update(
            text=out["text"],
            seconds=out["seconds"],
            load_s=round(load, 2),
            compute_s=round(compute, 2),
            duration=duration,
            rtf=round(compute / duration, 4) if duration else None,
            segments=len(out.get("segments") or []),
            truncated=bool(out.get("truncated")),
            script=asr_metrics.script(out["text"]),
            score=asr_metrics.score(reference, out["text"], state["language"])
            if reference
            else None,
            status="done",
        )
        clean = next(
            (x for x in results if x["model"] == r["model"] and x["condition"] == "clean"), None
        )
        if not reference and clean is not r and clean and clean.get("text"):
            r["drift"] = asr_metrics.score(clean["text"], out["text"], state["language"])
    for r in results:
        if r["status"] != "done":  # stopped before this model's turn
            r["status"] = "done"
    job.progress = 1.0
    path = save_run(
        "asr_compare",
        {
            "mode": "transcribe",
            "audio": state["audio_name"],
            "source": state["source"],
            "language": state["language"] or None,
            "prompt": state["prompt"] or None,
            "reference": reference or None,
            "conditions": [
                {k: v for k, v in c.items() if k != "file"} for c in conditions
            ] if len(conditions) > 1 else None,
            "stopped": job.cancel.is_set(),
            "results": [{k: r.get(k) for k in SAVED} for r in results],
        },
    )  # fmt: skip
    state["path"] = path.relative_to(path.parents[1])


def _last_of(results: list[dict], r: dict) -> dict:
    """The model's last run in the list (unload only after its last condition)."""
    return [x for x in results if x["model"] == r["model"]][-1]


# -- drawing --------------------------------------------------------------------------------


def diff_html(reference: str, text: str) -> str:
    """The transcript against the reference: missed text struck out, added text underlined."""
    out = []
    for op, ref, hyp in asr_metrics.alignment(reference, text):
        ref, hyp = html.escape(ref), html.escape(hyp)
        if op == "equal":
            out.append(hyp)
        if op in ("replace", "delete") and ref:
            out.append(f'<del style="color:#DC2626;background:#DC262614">{ref}</del>')
        if op in ("replace", "insert") and hyp:
            out.append(f'<ins style="color:#15803D;background:#15803D18">{hyp}</ins>')
    return f'<div style="line-height:1.9;font-size:.95rem">{"".join(out)}</div>'


def facts(r: dict) -> str:
    parts = [tr("處理", "processing") + f" {r['compute_s']:.2f}s"]
    if r.get("rtf"):
        parts.append(f"RTF {r['rtf']:.3f} · {1 / r['rtf']:.0f}×")
    if (r.get("load_s") or 0) >= 0.5:
        parts.append(tr("另載入", "plus load") + f" {r['load_s']:.1f}s")
    if r.get("score"):
        parts.append(f"{r['score'].get('metric', 'CER')} {sc.pct(r['score']['cer'])}")
    elif r.get("drift"):
        parts.append(tr("與乾淨版差異", "vs clean") + f" {sc.pct(r['drift']['cer'])}")
    return " · ".join(parts)


def model_card(r: dict, names: dict, reference: str) -> None:
    with st.container(border=True):
        st.markdown(f"**{names.get(r['model'], r['model'])}**")
        st.caption(f"`{r['model']}`")
        if r["status"] == "waiting":
            st.info(tr("等待中…", "Waiting…"))
            return
        if r["status"] == "running":
            st.info(tr("辨識中…", "Transcribing…"), icon=":material/progress_activity:")
            return
        if r.get("error"):
            st.error(r["error"])
            return
        if r.get("compute_s") is None:
            st.caption(tr("沒有執行（比較中途停止）", "Not run (the comparison was stopped)"))
            return
        st.caption(facts(r))
        if r.get("truncated"):
            st.warning(
                tr(
                    "輸出達到上限（音訊 LLM 開始重複），逐字稿被截斷。",
                    "The output hit its limit (the audio LLM started repeating); the transcript "
                    "is cut off.",
                )
            )
        with st.container(height=230, border=False):
            st.write(sc.show(r["text"]) or tr("（沒有辨識出文字）", "(no text recognised)"))
        if reference and r.get("text"):
            with st.expander(tr("對照原稿", "Against the reference")):
                st.caption(
                    tr(
                        "已統一繁簡、數字寫法並去掉標點；紅色刪除線＝漏掉或聽錯的原文，"
                        "綠色底線＝模型多出或寫錯的字。",
                        "Script, number format and punctuation are unified; red strike-through "
                        "= reference text missed or misheard, green underline = what the model "
                        "wrote instead or added.",
                    )
                )
                st.markdown(diff_html(reference, r["text"]), unsafe_allow_html=True)


def metric_of(results: list[dict]) -> str:
    """ "WER" when the runs were scored by words (English), else "CER"."""
    scores = [r.get("score") or r.get("drift") or {} for r in results]
    return "WER" if any(x.get("metric") == "WER" for x in scores) else "CER"


def metric_help(metric: str) -> str:
    if metric == "WER":
        return tr("詞錯誤率 WER（以英文單字計算，越低越好）", "Word error rate (lower is better)")
    return tr("字錯率 CER（越低越好）", "Character error rate (lower is better)")


def summary_table(results: list[dict], names: dict) -> pd.DataFrame:
    rows = []
    for r in results:
        if r.get("error") or r.get("compute_s") is None:
            continue
        score = r.get("score") or {}
        script = SCRIPTS.get(r.get("script"))
        rows.append(
            {
                tr("模型", "Model"): names.get(r["model"], r["model"]),
                "CER": score.get("cer"),
                tr("錯字（替換／漏／多）", "Errors (sub / del / ins)"): (
                    f"{score['sub']} / {score['del']} / {score['ins']}" if score else None
                ),
                tr("處理（秒）", "Processing (s)"): r["compute_s"],
                "RTF": r.get("rtf"),
                tr("載入（秒）", "Load (s)"): r.get("load_s"),
                tr("文字", "Script"): tr(*script) if script else "—",
                tr("字數", "Characters"): (
                    len(asr_metrics.words_en(r["text"]))
                    if (r.get("score") or {}).get("metric") == "WER"
                    else len(asr_metrics.normalize(r["text"], canonical=False))
                ),
            }
        )
    return pd.DataFrame(rows)


def table_config(table: pd.DataFrame, reference: str, metric: str = "CER") -> dict:
    config = {
        "RTF": st.column_config.NumberColumn(
            help=tr(
                "即時倍率＝處理時間÷錄音長度，越低越快（0.05 = 比即時快 20 倍）",
                "Real-time factor = processing time ÷ audio length; lower is faster "
                "(0.05 = 20× faster than real time)",
            ),
            format="%.3f",
        ),
        tr("處理（秒）", "Processing (s)"): st.column_config.NumberColumn(format="%.2f"),
        tr("載入（秒）", "Load (s)"): st.column_config.NumberColumn(format="%.1f"),
    }
    if reference:
        config["CER"] = st.column_config.ProgressColumn(
            label=metric,
            help=metric_help(metric),
            min_value=0.0,
            max_value=max(0.3, float(table["CER"].max() or 0)),
            format="percent",
        )
    return config


def single_summary(results: list[dict], names: dict, reference: str) -> None:
    table = summary_table(results, names)
    if table.empty:
        return
    st.subheader(tr("比較結果", "Results"))
    if reference:
        table = table.sort_values("CER")
    else:
        table = table.drop(columns=["CER", tr("錯字（替換／漏／多）", "Errors (sub / del / ins)")])
    metric = metric_of(results)
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config=table_config(table, reference, metric),
    )
    model_col = tr("模型", "Model")
    charts = st.columns(2)
    if reference:
        charts[0].caption(metric_help(metric))
        charts[0].bar_chart(
            table.rename(columns={"CER": metric}),
            x=model_col,
            y=metric,
            horizontal=True,
            sort=metric,
        )
    charts[1].caption(tr("即時倍率 RTF（越低越快）", "Real-time factor (lower is faster)"))
    charts[1].bar_chart(table, x=model_col, y="RTF", horizontal=True, sort="RTF")


def condition_matrix(
    results: list[dict], conditions: list[dict], names: dict, reference: str
) -> pd.DataFrame:
    """Long table: one row per model x condition with its error rate (CER against the
    reference, or the change from the model's own clean transcript without one)."""
    labels = {c["key"]: condition_label(c) for c in conditions}
    rows = []
    for r in results:
        if r.get("error") or r.get("compute_s") is None:
            continue
        value = (
            (r.get("score") or {}).get("cer")
            if reference
            else (r.get("drift") or {}).get("cer", 0.0 if r.get("condition") == "clean" else None)
        )
        rows.append(
            {
                "model": names.get(r["model"], r["model"]),
                "condition": labels.get(r.get("condition", "clean"), r.get("condition")),
                "value": value,
                "rtf": r.get("rtf"),
            }
        )
    return pd.DataFrame(rows)


def conditions_summary(
    results: list[dict], conditions: list[dict], names: dict, reference: str
) -> None:
    long = condition_matrix(results, conditions, names, reference)
    if long.empty:
        return
    st.subheader(tr("不同條件下的表現", "Across conditions"))
    order = [condition_label(c) for c in conditions]
    wide = long.pivot_table(index="model", columns="condition", values="value", aggfunc="first")
    wide = wide.reindex(columns=[o for o in order if o in wide.columns]).reset_index()
    wide = wide.rename(columns={"model": tr("模型", "Model")})
    fmt = st.column_config.NumberColumn(format="percent")
    st.dataframe(
        wide,
        hide_index=True,
        width="stretch",
        column_config={o: fmt for o in order},
    )
    measure = (
        metric_help(metric_of(results)).split("（")[0].split(" (")[0]
        if reference
        else tr("與乾淨版逐字稿的差異", "Change from the clean transcript")
    )
    st.caption(
        measure
        + tr("（越低越好）。", " (lower is better). ")
        + (
            ""
            if reference
            else tr(
                "沒有參考原稿，所以拿每個模型自己在乾淨音訊上的逐字稿當基準。",
                "Without a reference, each model's own transcript of the clean audio is the "
                "baseline.",
            )
        )
    )
    chart = (
        alt.Chart(long.dropna(subset=["value"]))
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("condition:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("model:N"),
            y=alt.Y("value:Q", title=measure, axis=alt.Axis(format="%")),
            color=alt.Color("model:N", title=tr("模型", "Model")),
            tooltip=[
                alt.Tooltip("model:N", title=tr("模型", "Model")),
                alt.Tooltip("condition:N", title=tr("條件", "Condition")),
                alt.Tooltip("value:Q", title=measure, format=".1%"),
                alt.Tooltip("rtf:Q", title="RTF", format=".3f"),
            ],
        )
        .properties(height=280)
    )
    st.altair_chart(chart, width="stretch")


def cards(results: list[dict], names: dict, reference: str) -> None:
    per_row = 3 if len(results) > 2 else max(len(results), 1)
    for start in range(0, len(results), per_row):
        cols = st.columns(per_row, gap="small")
        for col, r in zip(cols, results[start : start + per_row], strict=False):
            with col:
                model_card(r, names, reference)


def condition_picker(conditions: list[dict], key: str) -> dict:
    labels = {c["key"]: condition_label(c) for c in conditions}
    chosen = st.segmented_control(
        tr("顯示哪個條件的逐字稿", "Transcripts under"),
        list(labels),
        format_func=labels.get,
        default=conditions[0]["key"],
        key=key,
    )
    return next((c for c in conditions if c["key"] == chosen), conditions[0])


def render(job: jobs.Job) -> None:
    state, names = job.state, sc.names()
    results, reference, conditions = state["results"], state["reference"], state["conditions"]
    st.markdown(f"#### {state['audio_name']}")
    if len(conditions) > 1:
        c = condition_picker(conditions, key=f"cond_view_{job.id}")
        if c.get("file") and Path(c["file"]).exists():
            st.audio(c["file"])
        cards([r for r in results if r["condition"] == c["key"]], names, reference)
    else:
        cards(results, names, reference)
    if job.running:
        sc.running_bar(job)
        return
    if job.error:
        st.error(job.error)
    if len(conditions) > 1:
        conditions_summary(results, conditions, names, reference)
    else:
        single_summary(results, names, reference)
    notes = [
        tr(
            "處理時間不含載入模型與等待 GPU；同一段錄音依序交給每個模型。",
            "Processing time excludes loading the model and waiting for the GPU; the models "
            "run one after another on the same recording.",
        )
    ]
    if reference and metric_of(results) == "WER":
        notes.append(
            tr(
                "WER 以英文單字計算：先轉小寫、把數字讀成英文、去掉標點再逐字比對。",
                "WER is computed per word after lower-casing, spelling out numbers and removing "
                "punctuation.",
            )
        )
    elif reference:
        notes.append(
            tr(
                "CER 先統一繁簡（OpenCC）、把阿拉伯數字讀成中文、去掉標點與空白再逐字比對。",
                "CER is computed per character after unifying the script (OpenCC), spelling "
                "out digits in Chinese and removing punctuation and spaces.",
            )
        )
    st.caption(" ".join(notes))
    sc.saved_caption(state)


@st.fragment(run_every=1)
def live(job: jobs.Job) -> None:
    if not job.running:
        st.rerun()
    render(job)


# -- page -----------------------------------------------------------------------------------


def model_select(status: dict, labels: dict, key: str, label: str, max_n: int) -> list[str]:
    options = [m for m in labels if m in status]
    st.session_state.setdefault(key, [m for m in options if status[m]["installed"]][:max_n])
    st.session_state[key] = [m for m in st.session_state[key] if m in options]
    return st.multiselect(
        label,
        options,
        format_func=lambda m: sc.model_label(m, status, labels),
        max_selections=max_n,
        **kept(key),
    )


def conditions_form() -> tuple[list[str], str, int]:
    """The robustness options: which conditions, noise kind and SNR."""
    st.session_state.setdefault("asr_conditions", ["clean", "noise", "phone"])
    labels = {
        "clean": tr("乾淨（原始）", "Clean (original)"),
        "noise": tr("加噪音", "With noise"),
        "phone": tr("電話音質", "Telephone"),
        "noise_phone": tr("電話＋噪音", "Telephone + noise"),
    }
    chosen = st.pills(
        tr("條件", "Conditions"),
        list(labels),
        format_func=labels.get,
        selection_mode="multi",
        **kept("asr_conditions"),
    )
    a, b = st.columns(2)
    st.session_state.setdefault("asr_noise_kind", "babble")
    kind = a.selectbox(
        tr("噪音種類", "Noise"),
        audio_aug.NOISE_KINDS,
        format_func=lambda k: {
            "white": tr("白噪音（風扇、嘶聲）", "White (hiss, fans)"),
            "pink": tr("粉紅噪音（空調、車流）", "Pink (air conditioning, traffic)"),
            "babble": tr("人聲嘈雜（其他人在說話）", "Babble (other people talking)"),
            "office": tr("辦公室（空調＋遠處交談＋鍵盤）", "Office (hum + distant talk + keys)"),
        }[k],
        **kept("asr_noise_kind"),
    )
    st.session_state.setdefault("asr_snr", 10)
    snr = b.select_slider(
        tr("訊噪比 SNR（dB，越低越吵）", "Signal-to-noise ratio (dB; lower = noisier)"),
        SNRS,
        **kept("asr_snr"),
    )
    st.caption(
        tr(
            "噪音與電話音質在本機用 numpy 產生（`local_llm.audio_aug`，固定亂數種子）；人聲嘈雜"
            "混入其他範例錄音。電話＝300–3400 Hz 帶通、降到 8 kHz、μ-law 編碼。",
            "Noise and telephone audio are generated locally with numpy (`local_llm.audio_aug`, "
            "fixed seed); babble mixes in the other sample recordings. Telephone = 300–3400 Hz "
            "band-pass, 8 kHz sampling and μ-law coding.",
        )
    )
    return list(chosen or []), kind, snr


def transcribe_page(url: str, status: dict) -> None:
    labels = sc.names()
    selected = model_select(
        status, labels, "asr_models",
        tr(f"要比較的語音模型（最多 {MAX_MODELS} 個；↓ 表示尚未下載）",
           f"Speech models to compare (up to {MAX_MODELS}; ↓ = not downloaded yet)"),
        MAX_MODELS,
    )  # fmt: skip
    sc.download_missing(selected, status, "asr")
    sc.licence_warning(selected, labels)

    chosen = sc.audio_choice("asr")
    sc.follow("asr", "reference", chosen, sc.sample_reference)
    c1, c2 = st.columns([1, 2])
    languages = {
        "zh": tr("中文", "Chinese"),
        "en": tr("英文", "English"),
        "": tr("自動偵測", "Detect"),
    }
    picked = (chosen[2], chosen[3]) if chosen else None
    if chosen and chosen[2] == "sample" and st.session_state.get("asr_language_for") != picked:
        st.session_state.asr_language = sc.sample_language(chosen[3])
    st.session_state.asr_language_for = picked
    st.session_state.setdefault("asr_language", "zh")
    language = c1.selectbox(
        tr("語言", "Language"), list(languages), format_func=languages.get, **kept("asr_language")
    )
    if chosen and chosen[2] == "sample" and Path(chosen[3]).stem.endswith("_nan"):
        c1.caption(
            tr(
                "台語錄音：模型沒有台語選項，用最接近的「中文」提示。",
                "Taiwanese Hokkien: the models have no Hokkien option, so the closest hint, "
                "Chinese, is used.",
            )
        )
    prompt = c2.text_input(
        tr("上下文提示（選填）", "Context prompt (optional)"),
        placeholder=tr("例：人名、產品名、專有名詞", "e.g. names, products, terms"),
        help=tr(
            "Whisper 當作開頭提示、Qwen3-ASR 當作系統提示、Gemma 當作補充說明；SenseVoice 不使用。"
            "公平比較時建議留空。",
            "Whisper uses it as its initial prompt, Qwen3-ASR as a system prompt and Gemma as "
            "extra context; SenseVoice ignores it. Leave it empty for a fair comparison.",
        ),
        **kept("asr_prompt"),
    )
    reference = st.text_area(
        tr(
            "參考原稿（選填，用來計算錯誤率）",
            "Reference transcript (optional, for the error rate)",
        ),
        height=110,
        **kept("asr_reference"),
    )
    robust = st.toggle(
        tr("同時測試噪音與電話音質", "Also test with noise and telephone audio"),
        help=tr(
            "同一段錄音再做出加噪音、電話音質的版本，每個模型都跑一遍，比較字錯率怎麼變。",
            "Makes noisy and telephone versions of the same recording, runs every model on "
            "each, and shows how the error rate changes.",
        ),
        **kept("asr_robust"),
    )
    conds, kind, snr = (["clean"], "babble", 10)
    if robust:
        with st.container(border=True):
            conds, kind, snr = conditions_form()
    unload = st.toggle(
        tr(
            "每個模型用完就卸載（省記憶體；下次會重新載入）",
            "Unload each model after use (saves memory; it loads again next time)",
        ),
        **kept("asr_unload"),
    )

    job = jobs.get(st.session_state.get("asr_job"))
    busy = bool(job and job.running)
    ready = [m for m in selected if status[m]["installed"]]
    conditions = make_conditions(conds or ["clean"], kind, snr)
    runs = len(ready) * len(conditions)
    plural = "" if len(ready) == 1 else "s"
    label = (
        tr(f"開始比較（{len(ready)} 個模型）", f"Compare ({len(ready)} model{plural})")
        if len(conditions) == 1
        else tr(
            f"開始比較（{len(ready)} 個模型 × {len(conditions)} 種條件）",
            f"Compare ({len(ready)} model{plural} × {len(conditions)} conditions)",
        )
    )
    if st.button(
        label, type="primary", icon=":material/play_arrow:",
        disabled=busy or not ready or chosen is None or not conditions,
    ):  # fmt: skip
        audio, filename, source, audio_name = chosen
        if len(conditions) > 1:
            audio = sc.cached_audio(audio, filename)
        babble = [sc.SAMPLES_DIR / s for s in sc.BABBLE_SAMPLES if s != audio_name]
        job = jobs.Job(
            "asr_compare",
            tr(f"比較 {len(ready)} 個語音模型", f"Compare {len(ready)} speech models")
            + (tr(f"（{runs} 次）", f" ({runs} runs)") if len(conditions) > 1 else ""),
            sc.PAGE,
            state={
                "audio": str(audio) if isinstance(audio, Path) else audio,
                "filename": filename,
                "audio_name": audio_name,
                "source": source,
                "language": language,
                "prompt": prompt.strip(),
                "reference": reference.strip(),
                "unload": unload,
                "conditions": conditions,
                "babble": babble,
                "results": [
                    {"model": m, "condition": c["key"], "status": "waiting"}
                    for m in ready
                    for c in conditions
                ],
            },
        )
        jobs.start(job, lambda j: asr_work(j, url))
        st.session_state.asr_job = job.id
        st.rerun()
    if chosen is None:
        st.caption(tr("先選擇或錄一段音訊。", "Choose or record some audio first."))
    if job and job.running:
        live(job)
    elif job:
        render(job)


# -- history --------------------------------------------------------------------------------


def open_page_link() -> None:
    st.page_link(
        sc.PAGE,
        label=tr("開啟語音模型比較頁", "Open Compare speech models"),
        icon=":material/graphic_eq:",
    )


def history_view(data: dict) -> None:
    """A saved speech run (runs/asr_compare-*.json) on the History page."""
    views = {
        "translate": speech_translate.history_view,
        "meeting": speech_meeting.history_view,
        "live": speech_live.history_view,
        "tts": speech_tts.history_view,
        "voice_chat": speech_voice.history_view,
    }
    views.get(data.get("mode") or "transcribe", transcribe_history)(data)
    open_page_link()


def transcribe_history(data: dict) -> None:
    names = sc.names()
    results = [r for r in data.get("results") or [] if isinstance(r, dict)]
    reference = data.get("reference") or ""
    conditions = [c for c in data.get("conditions") or [] if isinstance(c, dict)]
    for r in results:
        r.setdefault("status", "done")
        r.setdefault("condition", "clean")
    duration = next((r.get("duration") for r in results if r.get("duration")), None)
    facts_line = [tr("錄音：", "Audio: ") + f"`{data.get('audio', '?')}`"]
    if duration:
        facts_line.append(tr(f"{duration:.0f} 秒", f"{duration:.0f} s"))
    if data.get("language"):
        facts_line.append(tr("語言：", "Language: ") + data["language"])
    if data.get("prompt"):
        facts_line.append(tr("上下文提示：", "Context prompt: ") + data["prompt"])
    if conditions:
        facts_line.append(
            tr("條件：", "Conditions: ") + "、".join(condition_label(c) for c in conditions)
        )
    if data.get("stopped"):
        facts_line.append(tr("中途停止", "stopped early"))
    st.caption(" · ".join(facts_line))
    if len(conditions) > 1:
        conditions_summary(results, conditions, names, reference)
        c = condition_picker(conditions, key="hist_cond_view")
        cards([r for r in results if r["condition"] == c["key"]], names, reference)
    else:
        table = summary_table(results, names)
        if not table.empty:
            if not reference:
                table = table.drop(
                    columns=["CER", tr("錯字（替換／漏／多）", "Errors (sub / del / ins)")]
                )
            else:
                table = table.sort_values("CER")
            st.dataframe(
                table,
                hide_index=True,
                width="stretch",
                column_config=table_config(table, reference, metric_of(results)),
            )
        cards(results, names, reference)
    if reference:
        with st.expander(tr("參考原稿", "Reference transcript")):
            st.write(reference)


def history_summary(data: dict) -> dict:
    """Card fields for the History list: models, best score, total processing time."""
    summaries = {
        "translate": speech_translate.history_summary,
        "meeting": speech_meeting.history_summary,
        "live": speech_live.history_summary,
        "tts": speech_tts.history_summary,
        "voice_chat": speech_voice.history_summary,
    }
    if (mode := data.get("mode") or "transcribe") in summaries:
        return summaries[mode](data)
    results = [r for r in data.get("results") or [] if isinstance(r, dict)]
    clean = [r for r in results if r.get("condition", "clean") == "clean"] or results
    scored = [(r["model"], r["score"]["cer"]) for r in clean if r.get("score")]
    metric = next((r["score"].get("metric") for r in clean if r.get("score")), None) or "CER"
    seconds = sum(r.get("compute_s") or 0 for r in results)
    models = []
    for r in results:
        if r.get("model") and r["model"] not in models:
            models.append(r["model"])
    # Cached across languages (history.summarize): no tr() here.
    title = str(data.get("audio") or "?")
    if data.get("conditions"):
        title += " · " + " / ".join(c.get("key", "?") for c in data["conditions"])
    return {
        "title": title,
        "models": models,
        "seconds": round(seconds, 2) if seconds else None,
        "snippet": next((r.get("text", "")[:220] for r in results if r.get("text")), ""),
        "best": (min(scored, key=lambda x: x[1]), metric) if scored else None,
        "error": bool(results) and all(r.get("error") for r in results),
    }


def best_label(best) -> str:
    """The best score for the History card: "CER 3.2% · model" (or chrF / DER)."""
    if not best:
        return ""
    (model, value), metric = best
    shown = f"{value:.1f}" if metric == "chrF" else sc.pct(value)
    return f"{metric} {shown} · {model}"
