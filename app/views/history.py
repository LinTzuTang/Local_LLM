"""Run history: every recorded run in runs/*.json as a readable activity feed.

Each run is a card (what was asked, when, how long, which models, how it ended); opening one
shows a view tailored to its kind: a scenario re-draws its flow diagram from the recorded
trace, a model comparison puts the answers side by side, an agent run becomes a conversation
with its LLM and tool calls. Raw JSON is still one click away for audits.
"""

import html
import json
import re
import time
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path

import flow_ui
import pandas as pd
import streamlit as st
import ui
from common import RUNS_DIR, kept
from i18n import pick, tr

from local_llm.lab import store
from local_llm.lab.nodes import NODE_TYPES

# family -> (icon, zh, en, colour)
KINDS = {
    "scenario": ("🕸️", "多 Agent 場景", "Scenario", "#7C3AED"),
    "compare": ("⚖️", "模型比較", "Model comparison", "#4F46E5"),
    "agent": ("🤖", "Agent", "Agent", "#0284C7"),
    "agent_compare": ("🆚", "多 Agent 比較", "Agent comparison", "#0F766E"),
    "departments": ("🏢", "部門 Agent", "Department agents", "#0F766E"),
    "control": ("🧑‍💻", "總控 Coding Agent", "Coding agent", "#334155"),
    "backend": ("🏎️", "後端效能", "Backend benchmark", "#D97706"),
    "other": ("📄", "其他", "Other", "#64748B"),
}
# status -> (icon, zh, en, colour)
STATUSES = {
    "done": ("✓", "完成", "Done", "#15803D"),
    "error": ("✗", "失敗", "Failed", "#DC2626"),
    "stopped": ("⏹", "已停止", "Stopped", "#D97706"),
}
VERDICTS = {  # department / agent-comparison verdicts
    "answered": ("✅", "有回答", "Answered", "#15803D"),
    "denied": ("🔒", "權限不足", "Permission denied", "#B45309"),
    "error": ("⚠️", "執行失敗", "Failed", "#DC2626"),
    "stopped": ("⏹️", "已停止", "Stopped", "#64748B"),
}
STAMP = re.compile(r"-\d{8}-\d{6}$")
READ_ONLY = ("list_", "show_", "get_", "read_", "search_", "describe_", "find_", "view_", "grep")
PAGE_SIZE = 12

CSS = """
<style>
.hx-strip { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px;
  margin: .2rem 0 1rem 0; }
.hx-stat { border: 1px solid var(--lab-border); border-radius: 14px; padding: 10px 16px;
  background: #FFFFFF; }
.hx-stat .v { font-size: 1.45rem; font-weight: 750; color: #0F172A; line-height: 1.25; }
.hx-stat .k { font-size: .78rem; color: var(--lab-muted); }
.hx-day { font-size: .76rem; font-weight: 700; color: #64748B; letter-spacing: .06em;
  text-transform: uppercase; margin: 1.1rem 0 .1rem 2px; }
.hx-row { display: flex; gap: 14px; align-items: flex-start; min-width: 0; }
.hx-icon { flex: 0 0 42px; height: 42px; border-radius: 12px; display: flex;
  align-items: center; justify-content: center; font-size: 1.3rem; }
.hx-main { flex: 1 1 auto; min-width: 0; }
.hx-title { font-weight: 650; font-size: 1rem; color: #0F172A; line-height: 1.35;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.hx-snip { color: #64748B; font-size: .86rem; line-height: 1.4; margin: 1px 0 6px 0;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.hx-meta { flex: 0 0 auto; text-align: right; font-size: .8rem; color: #64748B;
  line-height: 1.7; white-space: nowrap; }
.hx-chips { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
.hx-chip { display: inline-flex; align-items: center; gap: 4px; padding: 1px 9px;
  border-radius: 999px; font-size: .75rem; font-weight: 650; line-height: 1.55;
  border: 1px solid; white-space: nowrap; }
.hx-chips .lab-pill { font-size: .74rem; }
.hx-head .eyebrow { font-size: .78rem; font-weight: 700; letter-spacing: .05em;
  text-transform: uppercase; }
.hx-head h2 { font-size: 1.45rem; font-weight: 750; margin: .15rem 0 .5rem 0; padding: 0;
  line-height: 1.3; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical;
  overflow: hidden; }
.hx-head .meta { display: flex; flex-wrap: wrap; gap: 8px 14px; align-items: center;
  color: #64748B; font-size: .85rem; }
.hx-sec { font-size: .8rem; font-weight: 700; color: #475569; letter-spacing: .05em;
  text-transform: uppercase; margin: .2rem 0 .4rem 0; }
.hx-quote { border-left: 3px solid #C7D2FE; background: #F8FAFC; border-radius: 0 10px 10px 0;
  padding: 8px 14px; color: #1E293B; white-space: pre-wrap; font-size: .95rem; }
.hx-colhead { display: flex; justify-content: space-between; align-items: center; gap: 8px; }
.hx-colhead b { font-size: 1rem; }
</style>
"""


# --- parsing (cached by file mtime, language-neutral) ------------------------------------------


def text_of(content) -> str:
    """LangChain message content: a string or a list of content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(c.get("text", "")) if isinstance(c, dict) else str(c) for c in content
        ).strip()
    return "" if content is None else str(content)


def final_answer(messages: list) -> str:
    for m in reversed(messages or []):
        if isinstance(m, dict) and m.get("type") == "ai" and not m.get("tool_calls"):
            if text := text_of(m.get("content")).strip():
                return text
    return ""


def snippet(text: str, limit: int = 220) -> str:
    text = re.sub(r"```.*?```", " ", text or "", flags=re.S)
    text = re.sub(r"[#*_`>|]+|-{3,}", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def short_input(value) -> str:
    """Scenario inputs: file paths (audio, documents) show as their file name."""
    if isinstance(value, str) and "\n" not in value and len(value) < 400 and "/" in value:
        path = Path(value)
        if path.suffix and " " not in path.name:
            return path.name
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def family(kind: str, data: dict) -> str:
    head = kind.split("-", 1)[0]
    if head in KINDS:
        return head
    if head in ("coder", "coding", "code"):
        return "control"  # the control-center coding agent, under any of its names
    return "agent" if isinstance(data.get("messages"), list) else "other"


def entries(data: dict) -> list[tuple[str, dict]]:
    """Per-agent results of a departments / agent-comparison run, as (key, result)."""
    res = data.get("results") or data.get("agents") or data.get("runs") or {}
    if isinstance(res, dict):
        return [(str(k), v) for k, v in res.items() if isinstance(v, dict)]
    if isinstance(res, list):
        return [
            (str(r.get("agent") or r.get("agent_id") or r.get("id") or f"#{i + 1}"), r)
            for i, r in enumerate(res)
            if isinstance(r, dict)
        ]
    return []


def entry_answer(r: dict) -> str:
    return text_of(r.get("answer") or r.get("output") or "") or final_answer(r.get("messages"))


def ai_seconds(messages: list) -> float | None:
    total = sum(
        (m.get("response_metadata") or {}).get("total_duration") or 0
        for m in messages or []
        if isinstance(m, dict) and m.get("type") == "ai"
    )
    return round(total / 1e9, 2) if total else None


def tool_calls_of(messages: list) -> list[dict]:
    return [
        c
        for m in messages or []
        if isinstance(m, dict) and m.get("type") == "ai"
        for c in m.get("tool_calls") or []
    ]


def is_change(name: str) -> bool:
    return not name.startswith(READ_ONLY)


def verdict_id(value) -> str | None:
    """Older records stored the translated label (e.g. "🔒 權限不足") instead of the id."""
    if not value:
        return None
    text = str(value)
    if text in VERDICTS:
        return text
    for key, (_, zh, en, _) in VERDICTS.items():
        if zh in text or en.lower() in text.lower():
            return key
    return text


def normal_status(value) -> str | None:
    if value in ("done", "ok", "success", "finished", "completed"):
        return "done"
    if value in ("error", "failed", "failure"):
        return "error"
    if value in ("cancelled", "canceled", "stopped"):
        return "stopped"
    return None


@st.cache_data(show_spinner=False, max_entries=64)
def load(path: str, mtime: float) -> dict:
    data = json.loads(Path(path).read_text())
    return data if isinstance(data, dict) else {"value": data}


@st.cache_data(show_spinner=False, max_entries=5000)
def summarize(path: str, mtime: float) -> dict:
    """What a card needs; strings stay language-neutral (translated when drawn)."""
    p = Path(path)
    summary = {
        "name": p.name,
        "kind": STAMP.sub("", p.stem),
        "family": "other",
        "ts": mtime,
        "title": p.stem,
        "snippet": "",
        "models": [],
        "status": "done",
        "seconds": None,
        "ref": None,
        "facts": {},
    }
    try:
        data = load(path, mtime)
    except (OSError, ValueError) as exc:
        summary.update(status="error", snippet=f"{type(exc).__name__}: {exc}")
        return summary
    kind = str(data.get("kind") or summary["kind"])
    fam = family(kind, data)
    summary.update(kind=kind, family=fam)
    if isinstance(data.get("recorded_at"), str):
        try:
            summary["ts"] = datetime.fromisoformat(data["recorded_at"]).timestamp()
        except ValueError:
            pass
    prompt = text_of(
        data.get("prompt") or data.get("question") or data.get("instruction") or data.get("task")
    )
    status = normal_status(data.get("status"))
    if not status:
        status = "error" if data.get("error") else "stopped" if data.get("stopped") else "done"
    seconds = data.get("seconds") or data.get("wall_seconds")
    models = [data["model"]] if isinstance(data.get("model"), str) else []
    facts: dict = {}
    snip = ""

    if fam == "scenario":
        ref = data.get("scenario") or kind.split("-", 1)[-1]
        summary["ref"] = ref
        inputs = data.get("inputs") or {}
        first = next((short_input(v) for v in inputs.values() if v), "")
        prompt = first
        models = list(data.get("models") or [])
        snip = snippet(text_of(data.get("output")))
        facts["steps"] = len(data.get("steps") or [])
    elif fam == "compare":
        results = [r for r in data.get("results") or [] if isinstance(r, dict)]
        models = [r.get("model") for r in results if r.get("model")]
        totals = [((r.get("stats") or {}).get("total_s") or 0) for r in results]
        seconds = round(sum(totals), 2) if any(totals) else None
        scored = [
            (r.get("model"), (r.get("judge") or {}).get("score"))
            for r in results
            if isinstance(r.get("judge"), dict)
            and isinstance((r.get("judge") or {}).get("score"), int | float)
        ]
        if scored:
            facts["top"] = max(scored, key=lambda x: x[1])
        if data.get("judge"):
            facts["judge"] = data["judge"]
        facts["n"] = len(results)
        snip = snippet(text_of(results[0].get("answer"))) if results else ""
        if results and all(not r.get("answer") and not r.get("stats") for r in results):
            status = "error"
    elif fam in ("agent", "control"):
        messages = data.get("messages") or []
        summary["ref"] = data.get("agent") or (kind.split("-", 1)[1] if "-" in kind else None)
        snip = snippet(final_answer(messages))
        calls = tool_calls_of(messages)
        facts["tools"] = (
            data["tool_calls"] if isinstance(data.get("tool_calls"), int) else len(calls)
        )
        if fam != "agent":
            facts["changes"] = sum(is_change(c.get("name", "")) for c in calls)
            seconds = seconds or ai_seconds(messages)
        if not prompt:
            human = next((m for m in messages if m.get("type") == "human"), None)
            prompt = text_of(human.get("content")) if human else ""
    elif fam in ("departments", "agent_compare"):
        items = entries(data)
        verdicts = Counter(verdict_id(r.get("verdict")) for _, r in items if r.get("verdict"))
        facts["verdicts"] = dict(verdicts)
        facts["n"] = len(items)
        for _, r in items:
            if isinstance(r.get("model"), str) and r["model"] not in models:
                models.append(r["model"])
        if not seconds:
            secs = [r.get("seconds") for _, r in items if isinstance(r.get("seconds"), int | float)]
            seconds = max(secs) if secs else None
        stopped = any(r.get("stopped") for _, r in items)
        failed = [r for _, r in items if r.get("error")]
        status = (
            "stopped" if stopped else "error" if items and len(failed) == len(items) else status
        )
        snip = next((snippet(entry_answer(r)) for _, r in items if entry_answer(r)), "")
    elif fam == "backend":
        rows = [r for r in data.get("results") or [] if isinstance(r, dict)]
        for r in rows:
            m = r.get("模型") or r.get("model")
            if m and m not in models:
                models.append(m)
        facts["n"] = len(rows)
    else:
        snip = snippet(text_of(data.get("output") or data.get("answer") or ""))
        if not snip and isinstance(data.get("messages"), list):
            snip = snippet(final_answer(data["messages"]))

    summary.update(
        title=prompt or summary["title"],
        snippet=snip,
        models=[m for m in models if isinstance(m, str)],
        status=status,
        seconds=seconds if isinstance(seconds, int | float) else None,
        facts=facts,
    )
    summary["search"] = " ".join(
        [kind, p.name, summary["title"], snip, " ".join(summary["models"])]
    ).lower()
    return summary


def all_runs() -> list[dict]:
    files = []
    for f in RUNS_DIR.glob("*.json"):
        try:
            files.append((f, f.stat().st_mtime))
        except OSError:
            continue
    runs = [summarize(str(f), mtime) for f, mtime in files]
    return sorted(runs, key=lambda r: r["ts"], reverse=True)


# --- small presentation helpers (page side, translated) ----------------------------------------


def esc(text) -> str:
    return html.escape(str(text))


def kind_meta(fam: str, kind: str = "") -> tuple[str, str, str]:
    """Icon, label and colour; runs of an unknown kind are labelled with their own kind."""
    icon, zh, en, color = KINDS.get(fam, KINDS["other"])
    return icon, kind if fam == "other" and kind else tr(zh, en), color


def chip(text: str, color: str) -> str:
    return (
        f'<span class="hx-chip" style="color:{color};border-color:{color}33;'
        f'background:{color}0F">{esc(text)}</span>'
    )


def status_chip(status: str) -> str:
    icon, zh, en, color = STATUSES.get(status, STATUSES["done"])
    return chip(f"{icon} {tr(zh, en)}", color)


def verdict_chip(verdict: str, count: int | None = None) -> str:
    icon, zh, en, color = VERDICTS.get(verdict, ("•", verdict, verdict, "#64748B"))
    label = f"{icon} {tr(zh, en)}" + (f" × {count}" if count and count > 1 else "")
    return chip(label, color)


def model_pills(models: list[str], limit: int = 4) -> str:
    shown = [flow_ui.short_model(m) for m in models[:limit]]
    more = [f"+{len(models) - limit}"] if len(models) > limit else []
    return "".join(f'<span class="lab-pill">{esc(m)}</span>' for m in shown + more)


def ago(ts: float) -> str:
    secs = max(0.0, time.time() - ts)
    if secs < 60:
        return tr("剛剛", "just now")
    if secs < 3600:
        n = int(secs // 60)
        return tr(f"{n} 分鐘前", f"{n} min ago")
    if secs < 86400:
        n = int(secs // 3600)
        return tr(f"{n} 小時前", f"{n} h ago")
    n = int(secs // 86400)
    if n < 7:
        return tr(f"{n} 天前", f"{n} day{'s' if n > 1 else ''} ago")
    return day_label(datetime.fromtimestamp(ts).date())


def day_label(day: date) -> str:
    today = date.today()
    weekday_zh = "一二三四五六日"[day.weekday()]
    full = tr(f"{day.month} 月 {day.day} 日（週{weekday_zh}）", day.strftime("%a, %b %-d"))
    if day == today:
        return tr("今天", "Today") + " · " + full
    if day == today - timedelta(days=1):
        return tr("昨天", "Yesterday") + " · " + full
    return full if day.year == today.year else f"{day.year} · {full}"


def duration(secs: float | None) -> str:
    if secs is None:
        return "—"
    if secs < 60:
        return f"{secs:.1f}s"
    return f"{int(secs // 60)}m {int(secs % 60):02d}s"


def names() -> tuple[dict, dict]:
    """Scenario and agent definitions by id (for human names; missing files are fine)."""
    try:
        scenarios = {s["id"]: s for s in store.scenarios()}
    except Exception:
        scenarios = {}
    try:
        agents = {a["id"]: a for a in store.agents()}
    except Exception:
        agents = {}
    return scenarios, agents


def ref_name(run: dict, scenarios: dict, agents: dict) -> str:
    ref = run.get("ref")
    if run["family"] == "scenario":
        return pick(scenarios[ref]["name"]) if ref in scenarios else str(ref or "")
    if run["family"] == "agent" and ref:
        return pick(agents[ref]["name"]) if ref in agents else str(ref)
    return ""


def agent_label(key: str, r: dict, agents: dict) -> str:
    agent_id = r.get("agent") or r.get("agent_id") or key
    label = r.get("label") or r.get("name")
    if label:
        return pick(label)
    return pick(agents[agent_id]["name"]) if agent_id in agents else str(agent_id)


def section(title: str) -> None:
    st.markdown(f'<div class="hx-sec">{esc(title)}</div>', unsafe_allow_html=True)


def quote(text: str) -> None:
    st.markdown(f'<div class="hx-quote">{esc(text)}</div>', unsafe_allow_html=True)


def stat_strip(items: list[tuple[str, str]]) -> None:
    cells = "".join(
        f'<div class="hx-stat"><div class="v">{esc(v)}</div><div class="k">{esc(k)}</div></div>'
        for k, v in items
    )
    cols = len(items)
    st.markdown(
        f'<div class="hx-strip" style="grid-template-columns:repeat({cols},minmax(0,1fr))">'
        f"{cells}</div>",
        unsafe_allow_html=True,
    )


def open_run(name: str) -> None:
    st.query_params["run"] = name


def close_run() -> None:
    if "run" in st.query_params:
        del st.query_params["run"]


# --- list view --------------------------------------------------------------------------------


def card_facts(run: dict, scenarios: dict, agents: dict) -> str:
    """Chips after the kind: agent / scenario name, verdicts, judge winner, models."""
    parts = []
    if name := ref_name(run, scenarios, agents):
        parts.append(chip(name, "#475569"))
    facts = run["facts"]
    if top := facts.get("top"):
        parts.append(chip(f"🏆 {flow_ui.short_model(str(top[0]))} {top[1]}/10", "#B45309"))
    for verdict, count in (facts.get("verdicts") or {}).items():
        parts.append(verdict_chip(verdict, count))
    if facts.get("changes"):
        n = facts["changes"]
        parts.append(chip(tr(f"✏️ {n} 項變更", f"✏️ {n} change{'s' if n > 1 else ''}"), "#334155"))
    return "".join(parts) + model_pills(run["models"])


def run_card(run: dict, scenarios: dict, agents: dict) -> None:
    icon, label, color = kind_meta(run["family"], run["kind"])
    title = run["title"]
    if run["family"] == "scenario":
        title = ref_name(run, scenarios, agents) or title
        snip = run["title"] if run["title"] != run["name"].removesuffix(".json") else ""
        snip = snip or run["snippet"]
    else:
        snip = run["snippet"]
    meta = f"{status_chip(run['status'])}<br>{esc(ago(run['ts']))}"
    if run["seconds"] is not None:
        meta += f" · ⏱ {duration(run['seconds'])}"
    facts = card_facts(run, scenarios, agents) if run["family"] != "scenario" else ""
    if run["family"] == "scenario":
        facts = model_pills(run["models"])
    with st.container(border=True):
        left, right = st.columns([10, 1.7], vertical_alignment="center")
        left.markdown(
            f'<div class="hx-row"><div class="hx-icon" style="background:{color}14">{icon}</div>'
            f'<div class="hx-main"><div class="hx-title" title="{esc(title)}">{esc(title)}</div>'
            f'<div class="hx-snip">{esc(snip) or "&nbsp;"}</div>'
            f'<div class="hx-chips">{chip(label, color)}{facts}</div></div>'
            f'<div class="hx-meta">{meta}</div></div>',
            unsafe_allow_html=True,
        )
        right.button(
            tr("查看", "Open"),
            key=f"open_{run['name']}",
            on_click=open_run,
            args=(run["name"],),
            icon=":material/arrow_forward:",
            icon_position="right",
            width="stretch",
        )


def list_view(runs: list[dict]) -> None:
    scenarios, agents = names()
    counts = Counter(r["family"] for r in runs)
    order = [f for f in KINDS if counts.get(f)]
    models = sorted({m for r in runs for m in r["models"]}, key=str.lower)

    top = st.columns([2.3, 1.9, 3.0, 2.8], vertical_alignment="bottom")
    query = top[0].text_input(
        tr("搜尋", "Search"),
        placeholder=tr("搜尋問題、答案、模型、場景…", "Search prompts, answers, models…"),
        **kept("hist_query"),
    )
    chosen_models = top[1].multiselect(
        tr("模型", "Model"),
        models,
        format_func=flow_ui.short_model,
        placeholder=tr("所有模型", "All models"),
        **kept("hist_models"),
    )
    periods = {
        "today": tr("今天", "Today"),
        "7d": tr("7 天", "7 days"),
        "30d": tr("30 天", "30 days"),
        "custom": tr("自訂", "Custom"),
    }
    with top[3]:
        period = st.segmented_control(
            tr("時間", "When"),
            list(periods),
            format_func=periods.get,
            wrap=True,
            **kept("hist_period"),
        )
    lo, hi = None, None
    today = date.today()
    if period == "today":
        lo = today
    elif period == "7d":
        lo = today - timedelta(days=6)
    elif period == "30d":
        lo = today - timedelta(days=29)
    elif period == "custom":
        days = [datetime.fromtimestamp(r["ts"]).date() for r in runs]
        first = min(days) if days else today
        picked = top[3].date_input(
            tr("日期範圍", "Date range"),
            value=(first, today),
            min_value=first,
            max_value=max([today, *days]),
            key="hist_range",
        )
        if isinstance(picked, tuple | list) and picked:
            lo, hi = picked[0], picked[-1]

    with top[2]:
        statuses = st.pills(
            tr("狀態", "Status"),
            list(STATUSES),
            selection_mode="multi",
            format_func=lambda s: f"{STATUSES[s][0]} {tr(STATUSES[s][1], STATUSES[s][2])}",
            wrap=True,
            **kept("hist_status"),
        )
    kinds = st.pills(
        tr("類型", "Type"),
        order,
        selection_mode="multi",
        format_func=lambda f: f"{KINDS[f][0]} {kind_meta(f)[1]} · {counts[f]}",
        wrap=True,
        **kept("hist_kinds"),
    )

    words = [w for w in (query or "").lower().split() if w]

    def keep(r: dict) -> bool:
        if kinds and r["family"] not in kinds:
            return False
        if statuses and r["status"] not in statuses:
            return False
        if chosen_models and not set(chosen_models) & set(r["models"]):
            return False
        day = datetime.fromtimestamp(r["ts"]).date()
        if (lo and day < lo) or (hi and day > hi):
            return False
        if words:
            hay = r.get("search", "") + " " + ref_name(r, scenarios, agents).lower()
            return all(w in hay for w in words)
        return True

    shown = [r for r in runs if keep(r)]
    done = sum(r["status"] == "done" for r in shown)
    stat_strip(
        [
            (
                tr("筆紀錄", "Runs"),
                f"{len(shown)}" + (f" / {len(runs)}" if len(shown) != len(runs) else ""),
            ),
            (tr("成功率", "Succeeded"), f"{done / len(shown):.0%}" if shown else "—"),
            (tr("用到的模型", "Models used"), str(len({m for r in shown for m in r["models"]}))),
            (tr("累計執行時間", "Total run time"), duration(sum(r["seconds"] or 0 for r in shown))),
        ]
    )
    if not shown:
        st.info(tr("沒有符合條件的紀錄。", "No runs match these filters."))
        return

    limit = st.session_state.get("hist_limit", PAGE_SIZE)
    current_day = None
    for r in shown[:limit]:
        day = datetime.fromtimestamp(r["ts"]).date()
        if day != current_day:
            current_day = day
            st.markdown(f'<div class="hx-day">{esc(day_label(day))}</div>', unsafe_allow_html=True)
        run_card(r, scenarios, agents)
    if len(shown) > limit:
        left = len(shown) - limit
        st.button(
            tr(f"顯示更多（還有 {left} 筆）", f"Show more ({left} more)"),
            on_click=st.session_state.__setitem__,
            args=("hist_limit", limit + PAGE_SIZE),
            width="stretch",
        )


# --- detail views -------------------------------------------------------------------------------


def detail_header(run: dict, data: dict, scenarios: dict, agents: dict) -> None:
    icon, label, color = kind_meta(run["family"], run["kind"])
    title = run["title"]
    eyebrow = label
    if name := ref_name(run, scenarios, agents):
        if run["family"] == "scenario":
            title = name
        else:
            eyebrow += f" · {name}"
    when = datetime.fromtimestamp(run["ts"])
    meta = [
        status_chip(run["status"]),
        f"<span>🕒 {esc(when.strftime('%Y-%m-%d %H:%M'))} · {esc(ago(run['ts']))}</span>",
    ]
    if run["seconds"] is not None:
        meta.append(f"<span>⏱ {duration(run['seconds'])}</span>")
    meta.append(f'<span class="hx-chips">{model_pills(run["models"], limit=8)}</span>')
    meta.append(f'<span class="lab-pill">{esc(run["name"])}</span>')
    with st.container(border=True):
        st.markdown(
            f'<div class="hx-head hx-row"><div class="hx-icon" style="background:{color}14">'
            f'{icon}</div><div class="hx-main"><div class="eyebrow" style="color:{color}">'
            f"{esc(eyebrow)}</div><h2>{esc(title)}</h2>"
            f'<div class="meta">{"".join(meta)}</div></div></div>',
            unsafe_allow_html=True,
        )


def metrics(items: list[tuple[str, object]]) -> None:
    stat_strip([(k, str(v)) for k, v in items])


def safely(draw, what: str) -> bool:
    """Old records may not match today's renderers; fall back instead of failing the page."""
    try:
        draw()
        return True
    except Exception as exc:  # noqa: BLE001 - a history viewer must survive odd records
        st.caption(tr(f"無法顯示{what}：", f"Could not draw the {what}: ") + f"{exc}")
        return False


def scenario_view(data: dict, run: dict, scenarios: dict) -> None:
    steps = [s for s in data.get("steps") or [] if isinstance(s, dict)]
    sc = scenarios.get(run["ref"])
    metrics(
        [
            (tr("執行時間", "Duration"), duration(run["seconds"])),
            (tr("執行的步驟", "Steps run"), len(steps)),
            (tr("用到的模型", "Models"), len(run["models"])),
            (
                tr("平行步驟最多", "Max parallel steps"),
                max_parallel(steps) if steps else "—",
            ),
        ]
    )
    inputs = data.get("inputs") or {}
    if inputs:
        labels = {i["key"]: pick(i.get("label")) for i in (sc or {}).get("inputs", [])}
        with st.container(border=True):
            section(tr("輸入", "Input"))
            for key, value in inputs.items():
                if len(inputs) > 1 or labels.get(key):
                    st.caption(labels.get(key) or key)
                quote(short_input(value))
    if data.get("error"):
        st.error(data["error"])
    with st.container(border=True):
        section(tr("流程（依這次的紀錄重畫）", "Flow (re-drawn from this run's trace)"))
        if sc:
            safely(
                lambda: st.graphviz_chart(
                    flow_ui.flow_dot(sc, steps, finished=True), width="content"
                ),
                tr("流程圖", "flow diagram"),
            )
            st.caption(
                tr(
                    "粗線是這次實際走過的路徑；虛線框是沒有執行的節點。",
                    "Bold edges are the path this run took; dashed boxes did not run.",
                )
            )
        else:
            st.caption(
                tr(
                    f"場景「{run['ref']}」已不存在，無法重畫流程圖。",
                    f"Scenario '{run['ref']}' no longer exists, so the diagram cannot be drawn.",
                )
            )
    known = steps and all(s.get("type") in NODE_TYPES and "start" in s for s in steps)
    tab_result, tab_time, tab_steps = st.tabs(
        [
            tr("📝 結果", "📝 Result"),
            tr("⏱️ 時間軸", "⏱️ Timeline"),
            tr(f"🧩 各步驟（{len(steps)}）", f"🧩 Steps ({len(steps)})"),
        ]
    )
    with tab_result:
        output = data.get("output")
        if isinstance(output, str) and output.strip():
            with st.container(border=True):
                st.markdown(output)
        elif output:
            st.json(output)
        else:
            st.caption(tr("這次沒有產生結果。", "This run produced no result."))
    with tab_time:
        if known:
            safely(
                lambda: flow_ui.timeline(steps, running=False, elapsed=run["seconds"] or 0),
                tr("時間軸", "timeline"),
            )
            st.caption(
                tr(
                    "每一列是一個節點；同時開始的列代表平行執行，最長的那條就是瓶頸。",
                    "One row per node; rows that start together ran in parallel, and the "
                    "longest bar is the bottleneck.",
                )
            )
        else:
            st.caption(tr("沒有可用的時間資料。", "No timing data recorded."))
    with tab_steps:
        if not known or not safely(
            lambda: flow_ui.step_details(steps, elapsed=run["seconds"] or 0),
            tr("步驟", "steps"),
        ):
            for s in steps:
                with st.expander(
                    f"{s.get('status', '')} · {pick(s.get('label')) or s.get('node')}"
                ):
                    st.json(s, expanded=1)


def max_parallel(steps: list[dict]) -> int:
    best = 0
    for s in steps:
        start = s.get("start")
        if start is None:
            continue
        best = max(
            best,
            sum(
                1
                for t in steps
                if t.get("start") is not None
                and t["start"] <= start + 1e-6
                and (t.get("end") or float("inf")) > start
            ),
        )
    return best


def compare_view(data: dict, run: dict) -> None:
    results = [r for r in data.get("results") or [] if isinstance(r, dict)]
    stats = [r.get("stats") or {} for r in results]
    fastest = min((s["first_answer_s"] for s in stats if s.get("first_answer_s")), default=None)
    best_tps = max((s["decode_tps"] for s in stats if s.get("decode_tps")), default=None)
    top = run["facts"].get("top")
    metrics(
        [
            (tr("比較的模型", "Models compared"), len(results)),
            (tr("最快首字", "Fastest first token"), f"{fastest:.2f}s" if fastest else "—"),
            (tr("最高速度", "Best speed"), f"{best_tps:.0f} tok/s" if best_tps else "—"),
            (
                tr("裁判最高分", "Top judge score"),
                f"{top[1]}/10 · {flow_ui.short_model(str(top[0]))}" if top else "—",
            ),
        ]
    )
    long_prompt(run, text_of(data.get("prompt")))
    notes = []
    if data.get("system"):
        notes.append(tr("系統提示：", "System prompt: ") + str(data["system"]))
    notes.append(
        tr("思考模式：開", "Thinking: on")
        if data.get("think")
        else tr("思考模式：關", "Thinking: off")
    )
    if data.get("judge"):
        notes.append(tr("裁判模型：", "Judge: ") + str(data["judge"]))
    if data.get("stopped"):
        notes.append(tr("⏹ 中途停止", "⏹ stopped early"))
    st.caption(" · ".join(notes))
    if not results:
        return

    rows = []
    for r, s in zip(results, stats, strict=True):
        judge = r.get("judge") if isinstance(r.get("judge"), dict) else {}
        rows.append(
            {
                tr("模型", "Model"): r.get("model"),
                tr("裁判分數", "Judge score"): judge.get("score"),
                tr("首字（秒）", "First token (s)"): s.get("first_answer_s"),
                "tokens/s": s.get("decode_tps"),
                tr("輸出 tokens", "Output tokens"): s.get("output_tokens"),
                tr("總時間（秒）", "Total (s)"): s.get("total_s"),
                tr("載入（秒）", "Load (s)"): s.get("load_s"),
            }
        )
    df = pd.DataFrame(rows)
    max_tps = float(df["tokens/s"].max() or 0) if "tokens/s" in df else 0
    with st.container(border=True):
        section(tr("指標比較", "Metrics"))
        st.dataframe(
            df,
            hide_index=True,
            width="stretch",
            column_config={
                tr("裁判分數", "Judge score"): st.column_config.ProgressColumn(
                    min_value=0, max_value=10, format="%d / 10"
                ),
                "tokens/s": st.column_config.ProgressColumn(
                    min_value=0, max_value=max(max_tps, 1.0), format="%.0f"
                ),
                tr("首字（秒）", "First token (s)"): st.column_config.NumberColumn(format="%.2f"),
                tr("總時間（秒）", "Total (s)"): st.column_config.NumberColumn(format="%.1f"),
                tr("載入（秒）", "Load (s)"): st.column_config.NumberColumn(format="%.1f"),
            },
        )

    section(tr("答案並排", "Answers side by side"))
    per_row = 3 if len(results) > 4 or len(results) == 3 else 2 if len(results) > 1 else 1
    for start in range(0, len(results), per_row):
        cols = st.columns(per_row, gap="medium")
        for col, r, s in zip(
            cols, results[start : start + per_row], stats[start : start + per_row], strict=False
        ):
            judge = r.get("judge") if isinstance(r.get("judge"), dict) else {}
            score = judge.get("score")
            with col.container(border=True):
                badge = chip(f"🏆 {score}/10", "#B45309") if score is not None else ""
                st.markdown(
                    f'<div class="hx-colhead"><b>{esc(r.get("model", "?"))}</b>{badge}</div>',
                    unsafe_allow_html=True,
                )
                facts = []
                if s.get("first_answer_s") is not None:
                    facts.append(tr("首字", "first token") + f" {s['first_answer_s']:.2f}s")
                if s.get("decode_tps"):
                    facts.append(f"{s['decode_tps']:.0f} tok/s")
                if s.get("total_s") is not None:
                    facts.append(tr("總計", "total") + f" {s['total_s']:.1f}s")
                if facts:
                    st.caption(" · ".join(facts))
                answer_box(text_of(r.get("answer")) or tr("（沒有回答）", "(no answer)"))
                if judge.get("reason"):
                    st.caption(tr("裁判評語：", "Judge: ") + str(judge["reason"]))
                if r.get("thinking"):
                    with st.expander(tr("💭 思考過程", "💭 Thinking")):
                        st.code(r["thinking"][-6000:], language=None, wrap_lines=True)


def tool_network(name: str) -> bool:
    try:
        from local_llm.agents import TOOL_INFO

        info = TOOL_INFO.get(name)
        return bool(info and info.network)
    except Exception:
        return False


def tool_entry(name: str, args, message: dict | None) -> dict:
    content = text_of(message.get("content")) if message else ""
    return {
        "name": name or "tool",
        "args": args,
        "content": content,
        "ran": message is not None,
        "rejected": bool(message)
        and (message.get("status") == "error" or content.startswith("User rejected")),
        "network": tool_network(name or ""),
    }


def turns(messages: list) -> tuple[list[dict], dict | None]:
    """Each LLM call paired with the tool runs it asked for, plus the final answer call."""
    results = {m.get("tool_call_id"): m for m in messages if m.get("type") == "tool"}
    last = max(
        (i for i, m in enumerate(messages) if m.get("type") == "ai" and not m.get("tool_calls")),
        default=None,
    )
    used, steps, final = set(), [], None
    for i, m in enumerate(messages):
        if m.get("type") == "ai":
            meta, usage = m.get("response_metadata") or {}, m.get("usage_metadata") or {}
            step = {
                "calls": [],
                "seconds": (meta.get("total_duration") or 0) / 1e9 or None,
                "tokens": (usage.get("input_tokens"), usage.get("output_tokens")),
                "thinking": (m.get("additional_kwargs") or {}).get("reasoning_content"),
                "text": text_of(m.get("content")),
            }
            for c in m.get("tool_calls") or []:
                res = results.get(c.get("id"))
                if res is not None:
                    used.add(id(res))
                step["calls"].append(tool_entry(c.get("name"), c.get("args"), res))
            if i == last:
                final = step
            elif step["calls"] or step["text"]:
                steps.append(step)
        elif m.get("type") == "tool" and id(m) not in used:  # a result without its call
            steps.append(
                {
                    "calls": [tool_entry(m.get("name"), None, m)],
                    "seconds": None,
                    "tokens": (None, None),
                    "thinking": None,
                    "text": "",
                }
            )
    return steps, final


def step_title(i: int, step: dict) -> str:
    calls = step["calls"]
    if not calls:
        return f"{i}. 💬 " + tr("中間回覆", "Interim reply")
    names_ = ", ".join(f"`{c['name']}`" for c in calls)
    if any(c["rejected"] for c in calls):
        icon, state = "⛔", tr("人工拒絕，未執行", "rejected by a human, not run")
    elif not all(c["ran"] for c in calls):
        icon, state = "⏸️", tr("未執行", "not run")
    elif any(c["network"] for c in calls):
        icon, state = "🌐", tr("外部呼叫", "external call")
    else:
        icon, state = "🔧", tr("本機工具", "local tool")
    parts = [f"{i}. {icon} {names_}", state]
    if step["seconds"]:
        parts.append(f"🧠 {step['seconds']:.1f}s")
    tin, tout = step["tokens"]
    if tin or tout:
        parts.append(f"{tin or '?'} → {tout or '?'} tokens")
    return " · ".join(parts)


def as_code(value, limit: int) -> tuple[str, str | None]:
    if isinstance(value, str):
        text = value.strip()
        try:
            text = json.dumps(json.loads(text), ensure_ascii=False, indent=1)
            return text[:limit], "json"
        except ValueError:
            return text[:limit], None
    return json.dumps(value, ensure_ascii=False, indent=1)[:limit], "json"


def step_body(step: dict, limit: int = 4000) -> None:
    if step["thinking"]:
        st.caption(tr("💭 模型的思考", "💭 Model's reasoning"))
        st.code(step["thinking"][-3000:], language=None, wrap_lines=True, height=140)
    for c in step["calls"]:
        if c["args"] is not None:
            st.caption(tr(f"呼叫 `{c['name']}` 的參數", f"Arguments for `{c['name']}`"))
            st.code(*as_code(c["args"] or {}, limit), wrap_lines=True)
        if c["ran"]:
            st.caption(tr(f"`{c['name']}` 的結果", f"Result of `{c['name']}`"))
            st.code(*as_code(c["content"], limit), wrap_lines=True)
    if step["text"]:
        st.markdown(step["text"])


def step_list(messages: list, compact: bool = False) -> None:
    """LLM decisions with the tool runs they triggered, numbered; details open on click."""
    steps, _ = turns(messages)
    if not steps:
        st.caption(tr("沒有呼叫工具，直接回答。", "Answered directly, no tool calls."))
        return
    for i, step in enumerate(steps, start=1):
        if compact:  # inside a column or an expander, where expanders cannot nest
            st.markdown(f"**{step_title(i, step)}**")
            step_body(step, limit=1200)
        else:
            with st.expander(step_title(i, step)):
                step_body(step)


def answer_box(text: str, height: int = 380) -> None:
    """Long answers scroll inside a fixed box so side-by-side cards stay comparable."""
    long = len(text) > 600 or text.count("\n") > 14
    with st.container(height=height if long else "content", border=False):
        st.markdown(text)


def conversation(data: dict, prompt: str, avatar: str = "🤖") -> None:
    messages = [m for m in data.get("messages") or [] if isinstance(m, dict)]
    steps, final = turns(messages)
    with st.chat_message("user"):
        st.markdown(prompt or "—")
    with st.chat_message("assistant", avatar=avatar):
        if steps:
            section(tr(f"過程（{len(steps)} 步）", f"Steps ({len(steps)})"))
            step_list(messages)
            section(tr("回答", "Answer"))
        if final and final["thinking"]:
            with st.expander(tr("💭 思考過程", "💭 Thinking")):
                st.code(final["thinking"][-6000:], language=None, wrap_lines=True)
        if final and final["text"]:
            st.markdown(final["text"])
        elif data.get("stopped"):
            st.caption(tr("⏹ 在回答前被停止。", "⏹ Stopped before answering."))
        else:
            st.caption(tr("沒有最後回答。", "No final answer."))


def decisions_summary(decisions) -> Counter:
    flat = []
    for d in decisions or []:
        flat.extend(d if isinstance(d, list) else [d])
    return Counter(str(x.get("type")) if isinstance(x, dict) else str(x) for x in flat)


def decision_chips(decisions) -> str:
    labels = {
        "approve": ("✅", "核准", "approved", "#15803D"),
        "reject": ("❌", "拒絕", "rejected", "#DC2626"),
        "edit": ("✏️", "修改後核准", "edited", "#D97706"),
    }
    out = []
    for kind, n in decisions_summary(decisions).items():
        icon, zh, en, color = labels.get(kind, ("•", kind, kind, "#64748B"))
        out.append(chip(f"{icon} {tr(zh, en)} × {n}", color))
    return "".join(out)


def agent_view(data: dict, run: dict, agents: dict) -> None:
    messages = [m for m in data.get("messages") or [] if isinstance(m, dict)]
    llm_calls = data.get("llm_calls") or sum(m.get("type") == "ai" for m in messages)
    metrics(
        [
            (tr("執行時間", "Duration"), duration(run["seconds"])),
            (tr("LLM 呼叫", "LLM calls"), llm_calls),
            (tr("工具呼叫", "Tool calls"), run["facts"].get("tools", 0)),
            (
                tr("外部呼叫／被拒絕", "External / rejected"),
                f"{data.get('external_calls', 0)} / {data.get('rejected_calls', 0)}",
            ),
        ]
    )
    agent = agents.get(run.get("ref"))
    policy = data.get("policy") or {}
    chips = []
    if policy:
        chips.append(
            chip(tr("🌐 允許連網", "🌐 Network allowed"), "#0284C7")
            if policy.get("allow_network")
            else chip(tr("🔒 不連網", "🔒 Offline"), "#475569")
        )
        if policy.get("approve_external"):
            chips.append(
                chip(tr("✋ 外部呼叫需核准", "✋ External calls need approval"), "#B45309")
            )
    if data.get("think"):
        chips.append(ui.badge("thinking"))
    if dec := decision_chips(data.get("decisions")):
        chips.append(dec)
    if agent or chips:
        with st.container(border=True):
            if agent:
                st.markdown(f"**{esc(pick(agent['name']))}**")
                st.caption(pick(agent.get("description")))
            if chips:
                st.markdown(f'<div class="hx-chips">{"".join(chips)}</div>', unsafe_allow_html=True)
    conversation(data, text_of(data.get("prompt")) or run["title"])


def long_prompt(run: dict, text: str) -> None:
    """The header clamps the title to three lines; show long prompts in full."""
    if len(text) > 140 or text.count("\n") > 2:
        with st.container(border=True):
            section(tr("完整問題", "Full prompt"))
            quote(text)


def multi_view(data: dict, run: dict, agents: dict) -> None:
    items = entries(data)
    verdicts = run["facts"].get("verdicts") or {}
    metrics(
        [
            (tr("同時執行的 agent", "Agents run"), len(items)),
            (tr("總耗時", "Wall time"), duration(run["seconds"])),
            *[
                (
                    f"{VERDICTS[v][0]} {tr(VERDICTS[v][1], VERDICTS[v][2])}"
                    if v in VERDICTS
                    else v,
                    n,
                )
                for v, n in verdicts.items()
            ][:2],
        ]
    )
    long_prompt(run, run["title"])
    if data.get("model"):
        st.caption(
            tr("所有 agent 都用同一個模型：", "Every agent used the same model: ")
            + str(data["model"])
        )
    if not items:
        st.caption(tr("沒有各 agent 的結果。", "No per-agent results recorded."))
        return
    per_row = min(3, len(items))
    for start in range(0, len(items), per_row):
        cols = st.columns(per_row, gap="medium")
        for col, (key, r) in zip(cols, items[start : start + per_row], strict=False):
            with col.container(border=True):
                verdict = verdict_id(r.get("verdict"))
                head = verdict_chip(verdict) if verdict else ""
                st.markdown(
                    f'<div class="hx-colhead"><b>{esc(agent_label(key, r, agents))}</b>{head}</div>',
                    unsafe_allow_html=True,
                )
                facts = []
                if r.get("model") and r.get("model") != data.get("model"):
                    facts.append(str(r["model"]))
                if isinstance(r.get("seconds"), int | float):
                    facts.append(f"⏱ {duration(r['seconds'])}")
                messages = [m for m in r.get("messages") or [] if isinstance(m, dict)]
                steps, _ = turns(messages)
                tools = sum(len(s["calls"]) for s in steps)
                if messages:
                    facts.append(
                        tr(f"{tools} 次工具呼叫", f"{tools} tool call{'s' if tools != 1 else ''}")
                    )
                if facts:
                    st.caption(" · ".join(facts))
                if r.get("error"):
                    st.error(r["error"])
                answer_box(entry_answer(r) or tr("（沒有回答）", "(no answer)"), height=340)
                if steps:
                    with st.expander(tr(f"過程（{len(steps)} 步）", f"Steps ({len(steps)})")):
                        step_list(messages, compact=True)


def flatten(value, prefix: str = "") -> list[tuple[str, object]]:
    if isinstance(value, dict) and value:
        out = []
        for k, v in value.items():
            out += flatten(v, f"{prefix}.{k}" if prefix else str(k))
        return out
    return [(prefix, value)]


TARGET_KEYS = ("scenario_id", "agent_id", "node_id", "path", "file")


def change_card(call: dict, result: str) -> None:
    args = call.get("args") or {}
    target = " → ".join(str(args[k]) for k in TARGET_KEYS if args.get(k))
    rest = [
        (key.removeprefix("changes."), value)
        for key, value in flatten({k: v for k, v in args.items() if k not in TARGET_KEYS})
    ]
    st.markdown(f"✏️ **`{call.get('name')}`**" + (f" · {esc(target)}" if target else ""))
    if rest and len(rest) <= 12:
        lines = [f"| {tr('欄位', 'Field')} | {tr('新值', 'New value')} |", "| --- | --- |"]
        for key, value in rest:
            shown = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            shown = str(shown).replace("|", "\\|").replace("\n", " ")[:200]
            lines.append(f"| `{key or '—'}` | `{shown}` |")
        st.markdown("\n".join(lines))
    elif rest:
        st.code(json.dumps(args, ensure_ascii=False, indent=2)[:6000], language="json")
    if result:
        st.caption(tr("結果：", "Result: ") + result[:500])


def control_view(data: dict, run: dict) -> None:
    messages = [m for m in data.get("messages") or [] if isinstance(m, dict)]
    calls = tool_calls_of(messages)
    changes = [c for c in calls if is_change(c.get("name", ""))]
    results = {
        m.get("tool_call_id"): text_of(m.get("content"))
        for m in messages
        if m.get("type") == "tool"
    }
    decided = decisions_summary(data.get("decisions"))
    metrics(
        [
            (tr("LLM 呼叫", "LLM calls"), sum(m.get("type") == "ai" for m in messages)),
            (tr("查看設定", "Lookups"), len(calls) - len(changes)),
            (tr("變更", "Changes"), len(changes)),
            (
                tr("人工核准", "Approved by a human"),
                decided.get("approve", 0) + decided.get("edit", 0),
            ),
        ]
    )
    long_prompt(run, run["title"])
    with st.container(border=True):
        section(tr("變更內容", "Changes"))
        if dec := decision_chips(data.get("decisions")):
            st.markdown(f'<div class="hx-chips">{dec}</div>', unsafe_allow_html=True)
        if not changes and not any(data.get(k) for k in ("diff", "diffs", "patch", "changes")):
            st.caption(tr("這次沒有修改任何設定。", "Nothing was changed in this run."))
        for c in changes:
            change_card(c, results.get(c.get("id"), ""))
        for key in ("diff", "diffs", "patch"):
            value = data.get(key)
            for d in value if isinstance(value, list) else [value] if value else []:
                st.code(
                    d if isinstance(d, str) else json.dumps(d, ensure_ascii=False, indent=1),
                    language="diff",
                )
        if isinstance(data.get("changes"), list | dict) and data["changes"]:
            st.json(data["changes"], expanded=1)
    with st.chat_message("assistant", avatar="🧑‍💻"):
        steps, final = turns(messages)
        if steps:
            section(tr(f"過程（{len(steps)} 步）", f"Steps ({len(steps)})"))
            step_list(messages)
            section(tr("回覆", "Reply"))
        st.markdown((final or {}).get("text") or tr("（沒有最後回覆）", "(no final reply)"))


def table_view(data: dict, run: dict) -> None:
    if data.get("prompt"):
        with st.container(border=True):
            section(tr("題目", "Prompt"))
            quote(text_of(data["prompt"]))
    rows = [r for r in data.get("results") or [] if isinstance(r, dict)]
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def generic_view(data: dict, run: dict) -> None:
    """Unknown kinds: show the readable parts; everything else is in the raw JSON."""
    scalars = {
        k: v
        for k, v in data.items()
        if isinstance(v, str | int | float | bool)
        and k not in ("kind", "recorded_at")
        and not (isinstance(v, str) and len(v) > 160)
    }
    texts = {k: v for k, v in data.items() if isinstance(v, str) and len(v) > 160}
    if scalars:
        with st.container(border=True):
            section(tr("欄位", "Fields"))
            st.dataframe(
                pd.DataFrame(
                    {
                        tr("欄位", "Field"): list(scalars),
                        tr("值", "Value"): [str(v) for v in scalars.values()],
                    }
                ),
                hide_index=True,
                width="stretch",
                height=35 * (len(scalars) + 1) + 3,
            )
    for key, value in texts.items():
        with st.container(border=True):
            section(key)
            st.markdown(value)
    for key, value in data.items():
        if key == "messages" and isinstance(value, list):
            conversation(data, run["title"])
        elif isinstance(value, list) and value and all(isinstance(x, dict) for x in value):
            flat = [x for x in value if all(not isinstance(v, dict | list) for v in x.values())]
            if len(flat) == len(value):
                section(key)
                st.dataframe(pd.DataFrame(value), hide_index=True, width="stretch")


def detail_view(run: dict, runs: list[dict]) -> None:
    path = RUNS_DIR / run["name"]
    idx = next(i for i, r in enumerate(runs) if r["name"] == run["name"])
    bar = st.columns([2.2, 5, 1.6, 1.6, 1.6], vertical_alignment="center")
    bar[0].button(
        tr("所有紀錄", "All runs"),
        icon=":material/arrow_back:",
        on_click=close_run,
        type="tertiary",
    )
    bar[2].download_button(
        "JSON",
        path.read_bytes() if path.exists() else b"",
        file_name=run["name"],
        mime="application/json",
        icon=":material/download:",
        on_click="ignore",
        width="stretch",
    )
    newer = runs[idx - 1]["name"] if idx > 0 else None
    older = runs[idx + 1]["name"] if idx + 1 < len(runs) else None
    bar[3].button(
        tr("較新", "Newer"),
        icon=":material/chevron_left:",
        disabled=not newer,
        on_click=open_run,
        args=(newer,),
        width="stretch",
    )
    bar[4].button(
        tr("較舊", "Older"),
        icon=":material/chevron_right:",
        icon_position="right",
        disabled=not older,
        on_click=open_run,
        args=(older,),
        width="stretch",
    )
    try:
        data = load(str(path), path.stat().st_mtime)
    except (OSError, ValueError) as exc:
        st.error(tr("無法讀取這筆紀錄：", "Could not read this run: ") + str(exc))
        return
    scenarios, agents = names()
    detail_header(run, data, scenarios, agents)
    fam = run["family"]
    views = {
        "scenario": lambda: scenario_view(data, run, scenarios),
        "compare": lambda: compare_view(data, run),
        "agent": lambda: agent_view(data, run, agents),
        "departments": lambda: multi_view(data, run, agents),
        "agent_compare": lambda: multi_view(data, run, agents),
        "control": lambda: control_view(data, run),
        "backend": lambda: table_view(data, run),
    }
    if not safely(views.get(fam, lambda: generic_view(data, run)), tr("這筆紀錄", "this run")):
        generic_view(data, run)
    with st.expander(tr("原始 JSON", "Raw JSON"), icon=":material/data_object:"):
        st.caption(f"`{path.relative_to(RUNS_DIR.parent)}`")
        st.json(data, expanded=1)


# --- page -----------------------------------------------------------------------------------------

st.markdown(CSS, unsafe_allow_html=True)
ui.page_header(
    "📜",
    tr("執行紀錄", "Run history"),
    tr(
        "每次模型比較、agent、場景與總控的執行都會自動存檔。點開任何一筆就能重看當時的問題、"
        "流程、每個步驟與結果，也能下載 JSON 帶到別台機器對照。",
        "Every model comparison, agent, scenario and control-center run is saved "
        "automatically. Open any of them to replay the question, the flow, every step and the "
        "result, or download the JSON to compare on another machine.",
    ),
)

runs = all_runs()
if not runs:
    st.info(
        tr(
            "還沒有紀錄。到「模型比較」、「Agent 工作室」或「場景展示」執行一次就會出現在這裡。",
            "No runs yet. Run something on Compare models, Agent studio or the scenario gallery "
            "and it shows up here.",
        )
    )
    st.stop()

wanted = st.query_params.get("run")
selected = next((r for r in runs if r["name"] == wanted), None) if wanted else None
if wanted and not selected:
    st.warning(tr(f"找不到紀錄 `{wanted}`。", f"Run `{wanted}` was not found."))
if selected:
    # Opening a run from far down the list should start at the top of the page.
    st.html(
        "<script>for (const s of ['[data-testid=\"stMain\"]', 'section.main', "
        "'[data-testid=\"stAppViewContainer\"]']) { const el = window.parent.document"
        ".querySelector(s) || document.querySelector(s); if (el) el.scrollTo(0, 0); }"
        "window.scrollTo(0, 0);</script>",
        unsafe_allow_javascript=True,
    )
    detail_view(selected, runs)
else:
    list_view(runs)
