"""Meeting mode, "compare models": diarization models chosen here, run on the test meetings
that come with reference speaker turns, scored by DER in the background.

Test meetings: the bundled samples with an RTTM in data/lab/speech_refs (TESTSET). Models: the
gateway's diarization models (catalog task "diarize"), plus Azure AI Speech when cloud models
are switched on (the audio is uploaded). Models that take a speaker count (sherpa-onnx, Azure)
can run with the true count, with none, or both; the end-to-end ones (Nemotron 3, Sortformer)
always estimate it. DER as on the analysis page: local_llm.diarization.der, 0.25 s collar.
"""

import jobs
import pandas as pd
import speech_common as sc
import streamlit as st
from common import cloud_enabled, kept, save_run
from i18n import tr

from local_llm import catalog, elevenlabs, gateway_client
from local_llm import diarization as dz

TESTSET = ["meeting_zh.wav", "alimeeting_r8001_zh.flac", "alimeeting_r8009_zh.flac"]
AZURE = "azure-diarization"
ELEVEN = "elevenlabs-diarization"
CLOUD = (AZURE, ELEVEN)
COLLAR_S = 0.25


def takes_count(model: str) -> bool:
    """Whether the model can be told how many speakers there are."""
    if model in CLOUD:
        return True
    spec = next((m for m in catalog.gateway_models("diarize") if m["id"] == model), {})
    return spec.get("backend") == "sherpa-diarization"


def model_names() -> dict[str, str]:
    names = sc.names("diarize")
    if cloud_enabled():
        names[AZURE] = tr("Azure AI Speech 講者分離（雲端）", "Azure AI Speech diarization (cloud)")
        if elevenlabs.available():
            names[ELEVEN] = tr("ElevenLabs Scribe v2 講者分離（雲端）",
                               "ElevenLabs Scribe v2 diarization (cloud)")  # fmt: skip
    return names


def speakers_in(reference: list[dict]) -> int:
    return len({t["speaker"] for t in reference})


def compare_work(job: jobs.Job, url: str) -> None:
    state = job.state
    plan = [(m, mode, s) for m in state["models"] for s in state["samples"]
            for mode in state["modes"] if mode == "estimated" or takes_count(m)]  # fmt: skip
    rows = state.setdefault("rows", [])
    for k, (model, mode, sample) in enumerate(plan):
        if job.cancel.is_set():
            break
        job.detail, job.progress = f"{model} · {sc.sample_label(sample)}", k / max(len(plan), 1)
        reference = dz.parse_reference(sc.ref_file(sample, ".rttm"))
        count = speakers_in(reference) if mode == "known" else None
        path = sc.sample_path(sample)
        try:
            if model == AZURE:
                out = dz.azure_diarize(path, count or 10)
            elif model == ELEVEN:
                out = elevenlabs.diarize(path, count)
            else:
                out = gateway_client.diarize(path, count, model=model, keep_alive=120, base_url=url)
        except Exception as exc:  # one failure should not stop the rest
            rows.append({"model": model, "mode": mode, "sample": sample, "error": str(exc)})
            continue
        score = dz.der(reference, out["segments"], collar=COLLAR_S)
        rows.append({"model": model, "mode": mode, "sample": sample, "der": score["der"],
                     "miss": score["miss"], "false_alarm": score["false_alarm"],
                     "confusion": score["confusion"], "speakers": out.get("speakers"),
                     "ref_speakers": speakers_in(reference), "seconds": out.get("seconds")})  # fmt: skip
    job.progress = 1.0
    path = save_run("asr_compare", {"mode": "meeting_compare", "models": state["models"],
                                    "samples": state["samples"], "modes": state["modes"],
                                    "rows": rows, "stopped": job.cancel.is_set()})  # fmt: skip
    state["path"] = path.relative_to(path.parents[1])


def results_table(rows: list[dict], names: dict) -> None:
    done = [r for r in rows if "der" in r]
    for r in rows:
        if r.get("error"):
            st.error(f"{names.get(r['model'], r['model'])} · {sc.sample_label(r['sample'])}：{r['error']}")
    if not done:
        return
    modes = {"known": tr("告訴人數", "count given"), "estimated": tr("自己判斷", "estimated")}
    df = pd.DataFrame(done)
    df["model"] = df["model"].map(lambda m: names.get(m, m))
    df["mode"] = df["mode"].map(modes)
    df["sample"] = df["sample"].map(sc.sample_label)
    df["cell"] = df.apply(lambda r: f"{r['der']:.1%}（{r['speakers']}/{r['ref_speakers']} 人）", axis=1)
    table = df.pivot_table(index=["model", "mode"], columns="sample", values="cell",
                           aggfunc="first", sort=False)  # fmt: skip
    table.index.names = [tr("模型", "Model"), tr("人數", "Count")]
    st.dataframe(table, width="stretch")
    st.caption(tr(
        "DER（講者錯誤率，越低越好）＝漏掉＋多出＋認錯人的時間 ÷ 參考講者說話時間，邊界前後 0.25 秒不計；"
        "括號是模型分出的人數／實際人數。",
        "DER (lower = better) = missed + false alarm + wrong speaker time / reference speech, "
        "0.25 s around boundaries not scored; brackets: speakers found / actual.",
    ))  # fmt: skip
    with st.expander(tr("明細（漏掉、多出、認錯人、處理秒數）", "Details (miss, false alarm, confusion, seconds)")):
        st.dataframe(df[["model", "mode", "sample", "der", "miss", "false_alarm", "confusion", "seconds"]],
                     hide_index=True, width="stretch")  # fmt: skip


def compare_page(url: str, status: dict) -> None:
    names = model_names()
    st.caption(tr(
        "選幾個講者辨識模型，在附參考講者時間的測試會議上跑，算 DER 比較（在背景執行）。"
        "Nemotron 3、Sortformer 這類端到端模型自己判斷人數，不接受指定人數。",
        "Pick diarization models and run them on the test meetings that come with reference "
        "speaker turns, scored by DER (runs in the background). End-to-end models such as "
        "Nemotron 3 and Sortformer estimate the speaker count and take none.",
    ))  # fmt: skip
    st.session_state.setdefault("mc_models", [m for m in names if (status.get(m) or {}).get("installed")][:3])
    models = st.multiselect(tr("模型", "Models"), list(names), format_func=lambda m: names[m],
                            **kept("mc_models"))  # fmt: skip
    st.session_state.setdefault("mc_samples", TESTSET)
    samples = st.multiselect(tr("測試會議", "Test meetings"), TESTSET, format_func=sc.sample_label,
                             **kept("mc_samples"))  # fmt: skip
    mode_names = {"estimated": tr("自己判斷人數", "Estimate the count"),
                  "known": tr("告訴實際人數（模型支援時）", "Give the true count (where supported)")}  # fmt: skip
    st.session_state.setdefault("mc_modes", ["estimated", "known"])
    modes = st.pills(tr("人數", "Speaker count"), list(mode_names), format_func=mode_names.get,
                     selection_mode="multi", **kept("mc_modes"))  # fmt: skip
    gateway = [m for m in models if m not in CLOUD]
    sc.download_missing(gateway, status, "mc")
    ready = [m for m in models if m in CLOUD or (status.get(m) or {}).get("installed")]
    if any(m in CLOUD for m in models):
        st.caption(tr("雲端模型（Azure、ElevenLabs）會把錄音上傳到雲端。",
                      "Cloud models (Azure, ElevenLabs) upload the recordings."))  # fmt: skip
    job = jobs.get(st.session_state.get("mc_job"))
    busy = bool(job and job.running)
    if st.button(tr("開始比較", "Compare"), type="primary", icon=":material/play_arrow:",
                 disabled=busy or not ready or not samples or not modes):  # fmt: skip
        job = jobs.Job("speech_meeting_compare", tr("講者辨識模型比較", "Diarization models compared"),
                       sc.PAGE, state={"models": ready, "samples": samples, "modes": modes})  # fmt: skip
        jobs.start(job, lambda j: compare_work(j, url))
        st.session_state.mc_job = job.id
        st.rerun()
    if job and job.running:
        live_view(job, names)
    elif job:
        render(job, names)


def render(job: jobs.Job, names: dict) -> None:
    if job.running:
        sc.running_bar(job)
    elif job.error:
        st.error(job.error)
    results_table(job.state.get("rows") or [], names)
    if not job.running:
        sc.saved_caption(job.state)


@st.fragment(run_every=1)
def live_view(job: jobs.Job, names: dict) -> None:
    if not job.running:
        st.rerun()
    render(job, names)


def history_view(data: dict) -> None:
    names = model_names() | sc.names("diarize")
    bits = [tr("測試會議：", "Test meetings: ") + "、".join(sc.sample_label(s) for s in data.get("samples") or [])]
    if data.get("stopped"):
        bits.append(tr("中途停止", "stopped early"))
    st.caption(" · ".join(bits))
    results_table([r for r in data.get("rows") or [] if isinstance(r, dict)], names)


def history_summary(data: dict) -> dict:
    rows = [r for r in data.get("rows") or [] if isinstance(r, dict) and "der" in r]
    best = min(rows, key=lambda r: r["der"], default=None)
    # Cached across languages (history.summarize): no tr() here.
    return {
        "title": f"diarization × {len(data.get('samples') or [])}",
        "models": list(data.get("models") or []),
        "seconds": round(sum(r.get("seconds") or 0 for r in rows), 2) or None,
        "snippet": "",
        "best": ((best["model"], best["der"]), "DER") if best else None,
        "error": not rows,
    }
