"""The live voice assistant's evaluation tables, drawn under its page (voice_panel.py).

- Ways of talking: the measured results of every configuration tried, local and cloud
  (data/lab/voice_benchmark.json, from the speech evaluation: 7 test scripts, 23 lines).
- Front LLMs: the cascade's front model compared on the same scripts.
- Your runs: the sessions saved on this machine (runs/voice_agent-*.json) grouped by their
  settings, with median latencies per turn, so streaming and whole-clip recognition, models
  and voices can be compared from what was actually run here.
"""

import json
import statistics
from pathlib import Path

import altair as alt
import pandas as pd
import speech_common as sc
import streamlit as st
import voice_assistant as va
from common import PROJECT_DIR
from i18n import tr

DATA = PROJECT_DIR / "data" / "lab" / "voice_benchmark.json"
RUNS = PROJECT_DIR / "runs"
KINDS = {"cascade": ("① 串接式（Cascade）", "① Cascade"),
         "turn": ("② 回合式（Turn-based + Barge-in）", "② Turn-based + Barge-in"),
         "duplex": ("③ 全雙工（Full-Duplex）", "③ Full-Duplex")}  # fmt: skip
KIND_COLOURS = alt.Scale(domain=[tr(*v) for v in KINDS.values()],
                         range=["#4F46E5", "#0EA5E9", "#F97316"])  # fmt: skip


def load() -> dict | None:
    try:
        return json.loads(DATA.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:.0f}%"


def _sec(x: float | None) -> str:
    return "—" if x is None else f"{x:.2f} 秒" if x < 2 else f"{x:.1f} 秒"


def systems_frame(data: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                tr("系統", "System"): tr(s["name"], s["name_en"]),
                tr("對話方式", "Way"): tr(*KINDS[s["kind"]]),
                tr("跑在哪", "Where"): tr("本機", "local") if s["where"] == "local"
                else tr("雲端", "cloud"),
                tr("能查資料", "Lookups"): "✅" if s["tools"] else "—",
                tr("說完到開口", "Starts talking"): _sec(s["start_s"]),
                tr("說出查到的答案", "Says the answer"): _sec(s["answer_s"]),
                tr("判斷正確", "Decisions right"): s["decisions"] or "—",
                tr("繁體", "Traditional"): _pct(s["traditional"]),
                tr("不亂編", "Grounded"): _pct(s["grounded"]),
                tr("自然度", "Natural"): "—" if s["natural"] is None else f"{s['natural']:.2f}",
                tr("一句話", "In short"): tr(s["note"], s["note_en"]),
            }
            for s in data["systems"]
        ]
    )  # fmt: skip


def systems_chart(data: dict) -> alt.Chart:
    rows = [{"system": tr(s["name"], s["name_en"]), "kind": tr(*KINDS[s["kind"]]),
             "start": s["start_s"]} for s in data["systems"] if s["start_s"] is not None]  # fmt: skip
    df = pd.DataFrame(rows)
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            y=alt.Y("system:N", sort="x", title=None, axis=alt.Axis(labelLimit=320)),
            x=alt.X(
                "start:Q",
                title=tr("說完到開口（秒，越短越好）", "Starts talking (s, lower is better)"),
            ),
            color=alt.Color(
                "kind:N", scale=KIND_COLOURS, title=None, legend=alt.Legend(orient="bottom")
            ),  # fmt: skip
            tooltip=["system", "kind", "start"],
        )
    )


def front_frame(data: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                tr("前台 LLM", "Front LLM"): r["model"],
                tr("判斷正確", "Decisions right"): r["decisions"],
                tr("開始回應", "First token"): _sec(r["first_s"]),
                tr("不亂編", "Grounded"): _pct(r["grounded"]),
                tr("自然度", "Natural"): "—" if r["natural"] is None else f"{r['natural']:.2f}",
                tr("記憶體", "Memory"): "—" if r["memory_gb"] is None else f"{r['memory_gb']} GB",
            }
            for r in data["front_llms"]
        ]
    )


# -- your runs ----------------------------------------------------------------------------------


def _median(values: list) -> str:
    """Median seconds as text, "—" when no turn has the figure (e.g. no lookup answered)."""
    values = [v for v in values if isinstance(v, int | float)]
    return f"{statistics.median(values):.2f}" if values else "—"


def asr_kind(asr: str | None) -> str:
    if not asr:
        return tr("模型自己聽", "the model itself")
    if asr in va.PAUSE_MODELS:
        return tr("非串流", "whole-clip")
    return tr("串流", "streaming")


def config_label(settings: dict, names: dict) -> str:
    engine = settings.get("engine") or "cascade"
    asr = settings.get("asr") or ""
    if engine == "cascade":
        hear = settings.get("hear")
        heard = tr("只做字幕", "captions only") if hear else asr_kind(asr)
        llm = (settings.get("llm") or "?") + (
            tr("（直接聽語音）", " (hears the audio)") if hear else ""
        )
        bits = [f"{names.get(asr, asr)}（{heard}）", llm, settings.get("voice") or "?"]
        return tr("① 串接：", "① Cascade: ") + " → ".join(bits)
    kind = tr(*KINDS.get(va_kind(engine), KINDS["turn"]))
    return f"{kind}：{engine}"


def va_kind(engine: str) -> str:
    from local_llm import voice_cloud, voice_omni

    info = voice_omni.ENGINES.get(engine) or voice_cloud.ENGINES.get(engine) or {}
    return info.get("kind", "cascade" if engine == "cascade" else "turn")


@st.cache_data(ttl=60, show_spinner=False)
def runs_frame(lang: str) -> pd.DataFrame:
    """Saved sessions grouped by their settings: count, turns and median latencies."""
    del lang  # part of the cache key: the labels are translated
    names = sc.names("stream") | sc.names("asr")
    groups: dict[str, dict] = {}
    for path in sorted(RUNS.glob("voice_agent-*.json")):
        try:
            run = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        turns = run.get("turns") or []
        if not turns:
            continue
        settings = run.get("settings") or {}
        label = config_label(settings, names)
        g = groups.setdefault(label, {"runs": 0, "turns": 0, "asr": [], "start": [],
                                      "answer": [], "last": ""})  # fmt: skip
        g["runs"] += 1
        g["turns"] += len(turns)
        g["asr"] += [t.get("asr_s") for t in turns]
        g["start"] += [t.get("response_s") for t in turns]
        g["answer"] += [t.get("answer_s") for t in turns]
        g["last"] = max(g["last"], str(run.get("recorded_at") or path.stem[12:]))
    rows = [
        {
            tr("配置", "Configuration"): label,
            tr("次數", "Sessions"): g["runs"],
            tr("輪數", "Turns"): g["turns"],
            tr("辨識（秒）", "ASR (s)"): _median(g["asr"]),
            tr("說完到開口（秒）", "Starts talking (s)"): _median(g["start"]),
            tr("說出答案（秒）", "Says the answer (s)"): _median(g["answer"]),
            tr("最近一次", "Latest"): g["last"][:16].replace("T", " "),
        }
        for label, g in groups.items()
    ]
    df = pd.DataFrame(rows)
    return df.sort_values(df.columns[-1], ascending=False) if len(df) else df


# -- section ------------------------------------------------------------------------------------


def section() -> None:
    st.subheader(tr("評測總表", "Evaluation"))
    data = load()
    a, b, c = st.tabs([tr("各種做法", "Ways of talking"), tr("前台 LLM", "Front LLMs"),
                       tr("你在平台上跑過的配置", "Your runs")])  # fmt: skip
    with a:
        if not data:
            st.info(tr("找不到評測結果（data/lab/voice_benchmark.json）。",
                       "No results found (data/lab/voice_benchmark.json)."))  # fmt: skip
        else:
            st.caption(
                tr(
                    "7 段測試對話、23 句台詞（就是上面的測試腳本），以說話速度送入；說完到開口、說出答案取中位數。"
                    "判斷正確只算能查資料的系統；不亂編、自然度由 AI 評審（Azure gpt-5.4）逐句評。"
                    f"（{data.get('date', '')}，這台 Mac；雲端各跑 1 次）",
                    "7 test conversations, 23 lines (the test scripts above), fed at speaking pace; "
                    "medians of the start and answer times. Decisions count for systems with "
                    "lookups only; grounded and natural come from an AI judge (Azure gpt-5.4). "
                    f"({data.get('date', '')}, this Mac; each cloud system run once)",
                )
            )
            st.altair_chart(systems_chart(data), width="stretch")
            st.dataframe(systems_frame(data), hide_index=True, width="stretch")
            st.caption(
                tr(
                    "② 回合式（Turn-based + Barge-in）與 ③ 全雙工（Full-Duplex）的分法："
                    "Gemini Live、Nova Sonic、gpt-realtime 的連線是雙向同步串流（Bidirectional "
                    "Streaming，可插話），但仍等你講完才回，所以算回合式；全雙工是模型每一刻自己決定要不要說話，"
                    "能邊聽邊附和（gpt-live-1、MiniCPM-o 全雙工）。",
                    "Turn-based + barge-in vs full-duplex: Gemini Live, Nova Sonic and gpt-realtime "
                    "use bidirectional streaming (you can cut in) but still wait until you finish, so they count as "
                    "turn-based; a full-duplex model decides every moment whether to speak and "
                    "can back-channel while listening (gpt-live-1, MiniCPM-o full-duplex).",
                )
            )
            if data.get("not_run"):
                with st.expander(
                    tr("還沒測、不能測的即時語音模型", "Real-time voice models not run")
                ):
                    st.dataframe(
                        pd.DataFrame([
                            {tr("模型", "Model"): tr(zh, en), tr("類型", "Kind"): tr(kz, ke),
                             tr("原因", "Why"): tr(wz, we)}
                            for zh, en, kz, ke, wz, we in data["not_run"]
                        ]),
                        hide_index=True,
                        width="stretch",
                    )  # fmt: skip
    with b:
        if data:
            st.caption(
                tr(
                    "串接式的前台 LLM：同樣 7 段對話、10 個模型（關閉思考）。判斷正確＝該反問、該查、該直接回答有沒有做對"
                    "（46 個判斷）；開始回應＝使用者說完到 LLM 吐出第一個字。",
                    "The cascade's front LLM: the same 7 conversations, 10 models (thinking off). "
                    "Decisions = asked back, looked up or answered when it should (46 in all); "
                    "first token = end of the user's speech to the LLM's first token.",
                )
            )
            st.dataframe(front_frame(data), hide_index=True, width="stretch")
    with c:
        df = runs_frame(st.session_state.get("lang", "zh"))
        if df.empty:
            st.info(tr("還沒有儲存的對話。", "No saved sessions yet."))
        else:
            st.caption(
                tr(
                    "這台電腦上存下來的對話（執行紀錄），依設定分組，每輪的延遲取中位數：辨識＝說完到定稿文字；"
                    "說完到開口＝說完到助理第一個聲音；說出答案＝說完到念出查詢結果。串流與非串流辨識、"
                    "不同 LLM 和聲音可以直接對照。測試腳本與麥克風的對話都算在內。",
                    "Sessions saved on this machine (run history), grouped by settings, medians "
                    "per turn: ASR = end of speech to the final text; starts talking = to the "
                    "assistant's first sound; says the answer = to the spoken lookup result. "
                    "Compare streaming and whole-clip recognition, LLMs and voices directly. "
                    "Test-script and microphone sessions both count.",
                )
            )
            st.dataframe(df, hide_index=True, width="stretch")
