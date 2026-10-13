"""Model library: every local model at a glance, with capability filters and one-click downloads.

The Ollama catalogue comes from data/catalog.json: `caps` are Ollama's capability badges and
`tags` our derived labels (code, long context, MoE, licence ...); both use the keys of ui.CAPS,
so a capability looks the same here as on every other page. Downloads run as background jobs.
The "non-Ollama" view lists the speech, document and reranker models served by the gateway.
"""

import html

import jobs
import pandas as pd
import streamlit as st
import ui
from common import download_control, gateway_url, ollama_url, pull_progress, require_ollama
from i18n import current, tr

from local_llm import catalog, gateway_client, ollama_admin

lang = current()
data = catalog.load_catalog()
sep = ", " if lang == "en" else "、"


def loc(obj: dict, field: str):
    return catalog.localized(obj, field, lang)


def esc(text) -> str:
    return html.escape(str(text or ""))


tasks: dict[str, str] = loc(data, "tasks")
fit_labels = catalog.FIT_LABELS_EN if lang == "en" else catalog.FIT_LABELS
tiers = catalog.tiers()
all_models = catalog.models()

st.markdown(
    """
<style>
.lab-stats {display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px;
  margin: .2rem 0 .6rem 0;}
.lab-stat {border: 1px solid var(--lab-border); border-radius: 14px; padding: 10px 16px;
  background: linear-gradient(180deg, #FFFFFF, #F8FAFC);}
.lab-stat .v {font-size: 1.55rem; font-weight: 750; line-height: 1.2; color: #0F172A;}
.lab-stat .k {font-size: .8rem; color: var(--lab-muted);}
.tier-strip {display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 10px;}
.tier-tile {border: 1px solid var(--lab-border); border-radius: 12px; padding: 10px 12px;
  background: #FFFFFF;}
.tier-tile .t {font-weight: 700;} .tier-tile .r {color: var(--lab-primary); font-size: .78rem;
  font-weight: 650;} .tier-tile .s {font-size: .8rem; color: #475569; margin: 4px 0;}
.tier-tile .n {font-size: .76rem; color: var(--lab-muted);}
.mc-section {display: flex; align-items: baseline; gap: 10px; margin: 1.1rem 0 .1rem 0;}
.mc-section h3 {margin: 0; padding: 0; font-size: 1.25rem; font-weight: 750;}
.mc-section .d {color: var(--lab-muted); font-size: .88rem;}
.mc-section .c {margin-left: auto; font-size: .78rem; font-weight: 650; color: #475569;
  background: #F1F5F9; border-radius: 999px; padding: 2px 10px; white-space: nowrap;}
.mc-top {display: flex; justify-content: space-between; align-items: flex-start; gap: 8px;}
.mc-id {min-width: 0;}
.mc-name {font-weight: 750; font-size: 1.04rem; line-height: 1.3; overflow: hidden;
  text-overflow: ellipsis; white-space: nowrap; color: #0F172A;}
.mc-sub {color: var(--lab-muted); font-size: .8rem; margin-top: 1px;}
.mc-flags {display: flex; gap: 4px; flex-shrink: 0;}
.mc-flag {font-size: .72rem; font-weight: 700; padding: 2px 8px; border-radius: 999px;
  background: #FEF3C7; color: #92400E; white-space: nowrap;}
.mc-flag.default {background: #EEF2FF; color: #4338CA;}
.mc-specs {display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0 6px 0;}
.mc-spec {font-size: .76rem; background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px;
  padding: 1px 8px; color: #334155; white-space: nowrap;}
.mc-note {font-size: .87rem; color: #1F2937; line-height: 1.5; margin: 6px 0 8px 0;
  display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden;}
.mc-foot {display: flex; gap: 10px; align-items: center; font-size: .78rem;
  color: var(--lab-muted); border-top: 1px dashed #E5E7EB; padding-top: 7px; margin-bottom: .9rem;}
.mc-foot .lic {margin-left: auto; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  max-width: 55%;}
.fit-ok {color: #15803D; font-weight: 650;} .fit-tight {color: #B45309; font-weight: 650;}
.fit-no {color: #B91C1C; font-weight: 650;}
@media (max-width: 900px) {.lab-stats {grid-template-columns: repeat(2, minmax(0, 1fr));}
  .tier-strip {grid-template-columns: repeat(2, minmax(0, 1fr));}}
</style>
""",
    unsafe_allow_html=True,
)

ui.page_header(
    "📚",
    tr("模型庫", "Model library"),
    tr(
        "這台 Mac 跑得動的地端模型一覽：用能力、任務、大小與授權篩選，一鍵在背景下載。徽章同時列出 Ollama 標示的能力（工具、思考、看圖、音訊、向量、決策）與平台整理的標籤（程式、長上下文、MoE、可否商用）。",
        "Every local model this Mac can run, at a glance: filter by capability, task, size and licence, and download in the background with one click. Badges show Ollama's capability labels (tools, thinking, vision, audio, embedding, decision) plus our own (code, long context, MoE, commercial use).",
    ),
    eyebrow=tr("模型", "Models"),
)
require_ollama()
installed = {catalog.normalize(m["name"]) for m in ollama_admin.installed(ollama_url())}


def is_installed(m: dict) -> bool:
    return catalog.normalize(m["tag"]) in installed


def stat(value, label: str) -> str:
    return f'<div class="lab-stat"><div class="v">{esc(value)}</div><div class="k">{esc(label)}</div></div>'


st.markdown(
    '<div class="lab-stats">'
    + stat(len(all_models), tr("個 Ollama／GGUF 模型", "Ollama / GGUF models"))
    + stat(sum(map(is_installed, all_models)), tr("已下載到這台 Mac", "downloaded to this Mac"))
    + stat(sum(bool(m.get("recommended")) for m in all_models), tr("⭐ 平台推薦", "⭐ recommended"))
    + stat(
        len(catalog.other_models()),
        tr("非 Ollama 模型（語音、文件、Reranker）", "non-Ollama models (speech, docs, rerank)"),
    )
    + "</div>",
    unsafe_allow_html=True,
)
st.caption(
    f"{loc(data, 'memory_note')} "
    + tr(
        f"資料來源：[Ollama 本機模型](https://ollama.com/search?c=local)，查核日期 {data['checked_date']}。",
        f"Source: [Ollama local models](https://ollama.com/search?c=local), checked on {data['checked_date']}.",
    )
)

if pulls := [j for j in jobs.active() if j.kind == "pull"]:
    with st.container(border=True):
        st.markdown(
            tr(
                "**下載中**（在背景進行，切換頁面不會中斷）",
                "**Downloading** (in the background; switching pages won't stop it)",
            )
        )
        for job in pulls:
            pull_progress(job, key=f"top_{job.id}")


# --- non-Ollama models (model gateway) --------------------------------------------------------
def licence_keys(text: str | None) -> list[str]:
    low = (text or "").lower()
    if "nc" in low.replace("-", " ").split() or "非商用" in low or "non-commercial" in low:
        return ["non_commercial"]
    return ["commercial_ok"] if low.startswith(("apache", "mit")) else []


def gateway_card(m: dict, status: dict | None, loaded: dict | None, up: bool) -> None:
    """Download / delete / loaded state of one model served by the model gateway."""
    have = bool(status and status["installed"])
    entry = next(
        (o for o in catalog.other_models() if o.get("gateway", {}).get("id") == m["id"]), {}
    )
    keys = [
        m["task"],
        *(["taiwan"] if entry.get("taiwan") else []),
        *licence_keys(m.get("license")),
    ]
    keys += ["installed"] if have else []
    size = (status or {}).get("size_bytes")
    specs = [f"{m['size_gb']} GB"]
    if size:
        specs.append(tr(f"磁碟上 {size / 1e9:.2f} GB", f"{size / 1e9:.2f} GB on disk"))
    # The gateway may serve its own pipeline for an entry (Docling: Heron + TableFormer instead of
    # granite-docling); then the entry's size, runtime and note do not describe what is served.
    own = m.get("name") == entry.get("name")
    if own and entry.get("params"):
        specs.insert(0, loc(entry, "params"))
    if own and entry.get("mac"):
        specs.append(loc(data, "mac_labels")[entry["mac"]])
    flags = f'<span class="mc-flag">⭐ {tr("推薦", "Pick")}</span>' if m.get("recommended") else ""
    note = " ".join(x for x in (loc(entry, "use"), loc(entry, "note")) if x)
    note = note if own and note else loc(m, "note")
    state = tr("⚪ 未載入", "⚪ Not loaded")
    if loaded:
        until = (loaded["expires_at"] or "")[11:16] or tr("不限", "never")
        state = tr(f"🟢 已載入（{loaded['device']}，{until} 卸載）",
                   f"🟢 Loaded ({loaded['device']}, unloads at {until})")  # fmt: skip
    with st.container(border=True, height="stretch", gap="xsmall"):
        st.markdown(
            f'<div class="mc-top"><div class="mc-id"><div class="mc-name" title="{esc(loc(m, "name"))}">{esc(loc(m, "name"))}</div>'
            f'<div class="mc-sub">{tr("模型閘道", "Model gateway")} · <code>{esc(m["id"])}</code></div></div>'
            f'<div class="mc-flags">{flags}</div></div>'
            f'<div class="mc-specs">{"".join(f"<span class=mc-spec>{esc(s)}</span>" for s in specs)}</div>'
            + ui.badges(keys)
            + f'<div class="mc-note" title="{esc(note)}">{esc(note)}</div>',
            unsafe_allow_html=True,
        )
        st.space("stretch")
        st.markdown(
            f'<div class="mc-foot"><span>{esc(state if have else "")}</span>'
            f'<span class="lic" title="{esc(loc(m, "license"))}">⚖️ {esc(loc(m, "license"))}</span></div>',
            unsafe_allow_html=True,
        )
        job = jobs.find("pull", model=m["id"])
        if up and not have and job and job.running:
            download_control(m["id"], key=f"gw_pull_{m['id']}", source="gateway")
            return
        row = st.container(horizontal=True, gap="small")
        if not up:
            row.button(tr("⬇ 下載", "⬇ Download"), key=f"gw_pull_{m['id']}", disabled=True,
                       help=tr("模型閘道沒有在執行", "The model gateway is not running"))  # fmt: skip
        elif have:
            if loaded and row.button(tr("卸載", "Unload"), key=f"gw_unload_{m['id']}"):
                gateway_client.unload(m["id"], gateway_url())
                st.rerun()
            with row.popover(tr("🗑 刪除", "🗑 Delete")):
                st.caption(tr(f"刪除 {m['id']} 的權重？之後可以再下載。",
                              f"Delete the weights of {m['id']}? You can download them again later."))  # fmt: skip
                if st.button(tr("確定刪除", "Delete"), key=f"gw_del_{m['id']}", type="primary"):
                    gateway_client.delete(m["id"], gateway_url())
                    st.rerun()
        else:
            download_control(m["id"], key=f"gw_pull_{m['id']}", container=row,
                             label=tr("⬇ 下載", "⬇ Download"), source="gateway")  # fmt: skip
        with row.popover(tr("詳細", "Details"), icon=":material/info:"):
            st.markdown(f"**{loc(m, 'name')}** · `{m['id']}`")
            st.caption(tr("模型閘道怎麼準備這個模型", "How the gateway prepares this model"))
            st.write(loc(m, "note"))
            for label, field in (
                (tr("執行方式", "Runtime"), "runtime"),
                (tr("安裝", "Install"), "install"),
            ):
                if value := loc(entry, field):
                    st.markdown(
                        f"**{label}**：{value}" if lang == "zh" else f"**{label}**: {value}"
                    )
            st.code(
                f'curl -X POST {gateway_url()}/api/pull -d \'{{"model": "{m["id"]}"}}\'',
                language="bash",
            )
            if repo := m.get("repo") or entry.get("repo"):
                st.markdown(
                    f"[{tr('開啟 Hugging Face 模型頁', 'Open on Hugging Face')} ↗](https://huggingface.co/{repo})"
                )


def other_models_view() -> None:
    """Speech, document parsing and reranker models that Ollama cannot run."""
    st.info(loc(data, "other_runtimes_note"))
    url = gateway_url()
    version = gateway_client.version(url)
    if version:
        status = {m["id"]: m for m in gateway_client.models(url)}
        loaded = {m["id"]: m for m in gateway_client.loaded(url)}
        st.caption(
            tr(f"模型閘道 v{version} 執行中：`{url}`（`GET /api/tags`、`GET /api/ps`；下載是 `POST /api/pull`）。",
               f"Model gateway v{version} running at `{url}` (`GET /api/tags`, `GET /api/ps`; downloads use `POST /api/pull`).")
        )  # fmt: skip
    else:
        status, loaded = {}, {}
        st.warning(
            tr(f"模型閘道沒有在執行（{url}），無法下載下列模型。請執行 `./scripts/tmux_up.sh`（在 tmux 背景執行）或 `./scripts/start_gateway.sh`。",
               f"The model gateway is not running ({url}), so these models cannot be downloaded. Run `./scripts/tmux_up.sh` (in tmux) or `./scripts/start_gateway.sh`.")
        )  # fmt: skip
    with st.expander(
        tr(
            "Ollama、模型閘道與其他執行方式的差別",
            "Ollama vs. the model gateway vs. other runtimes",
        )
    ):
        st.markdown(
            tr(
                "| | Ollama 模型 | 模型閘道（本機 11500 port） | 其他非 Ollama 模型 |\n|---|---|---|---|\n"
                "| 安裝 | 平台按「下載」 | 平台按「下載」 | 自己裝 Python 套件與權重 |\n"
                "| 呼叫方式 | `/api/chat` | `/v1/audio/transcriptions`、`/v1/documents/parse`、`/v1/rerank` | 各自的 Python API |\n"
                "| 記憶體 | 自動載入／卸載 | 第一次使用時載入，閒置 5 分鐘自動卸載 | 自己管理，與 Ollama 共用 64GB |\n"
                "| Mac GPU | Metal／MLX | MLX（語音）、MPS（版面分析、reranker） | 看框架：MLX ✅、PyTorch 🟡、PaddlePaddle 只有 CPU |\n"
                "| 平台頁面 | 對話、Agent、比較 | 「語音與文件」頁、agent 工具（會議與文件助理） | 要自己包成工具才用得到 |",
                "| | Ollama models | Model gateway (local port 11500) | Other non-Ollama models |\n|---|---|---|---|\n"
                "| Install | Download button | Download button | Install Python packages and weights yourself |\n"
                "| API | `/api/chat` | `/v1/audio/transcriptions`, `/v1/documents/parse`, `/v1/rerank` | Each library's own Python API |\n"
                "| Memory | Loaded / unloaded automatically | Loaded on first use, unloaded after 5 idle minutes | Your job; shares the 64GB with Ollama |\n"
                "| Mac GPU | Metal / MLX | MLX (speech), MPS (layout, reranker) | Depends: MLX ✅, PyTorch 🟡, PaddlePaddle CPU only |\n"
                "| Pages | Chat, Agent, Compare | Speech & documents page, agent tools (meetings & documents assistant) | Must be wrapped as tools first |",
            )
        )
    mac = loc(data, "mac_labels")
    for key, label in loc(data, "other_categories").items():
        members = catalog.other_models(key)
        section(f"{ui.CAPS[key][0]} {label}", "", len(members))
        served = [m for m in catalog.gateway_models()
                  if any(o.get("gateway", {}).get("id") == m["id"] for o in members)]  # fmt: skip
        for start in range(0, len(served), 2):
            cols = st.columns(2)
            for col, m in zip(cols, served[start : start + 2], strict=False):
                with col:
                    gateway_card(m, status.get(m["id"]), loaded.get(m["id"]), bool(version))
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        tr("模型", "Model"): ("⭐ " if m.get("recommended") else "")
                        + m["name"]
                        + (" 🇹🇼" if m.get("taiwan") else ""),
                        tr("模型閘道", "Gateway"): f"✅ {m['gateway']['id']}"
                        if m.get("gateway")
                        else "",
                        tr("用途", "Use"): loc(m, "use"),
                        tr("參數", "Params"): loc(m, "params"),
                        tr("在這台 Mac", "On this Mac"): mac[m["mac"]],
                        tr("執行方式", "Runtime"): loc(m, "runtime"),
                        tr("授權", "License"): loc(m, "license"),
                        "Hugging Face": f"https://huggingface.co/{m['repo']}"
                        if "/" in m["repo"] and " " not in m["repo"]
                        else None,
                        tr("安裝", "Install"): loc(m, "install"),
                        tr("備註", "Note"): loc(m, "note"),
                    }
                    for m in members
                ]
            ),
            hide_index=True,
            width="stretch",
            column_config={
                "Hugging Face": st.column_config.LinkColumn(display_text=tr("開啟", "Open"))
            },
        )
    st.caption(
        tr(
            "Ollama 上也有 OCR 與文件模型：在「Ollama 模型」勾選「📄 OCR／文件」能力，可找到 glm-ocr、deepseek-ocr、granite3.2-vision、nemotron3。決策模型（nimble、clef、tev1…）勾選「🎯 決策分類」。",
            "Ollama also has OCR and document models: in 'Ollama models', pick the '📄 OCR / docs' capability to find glm-ocr, deepseek-ocr, granite3.2-vision and nemotron3. Decision models (nimble, clef, tev1 …) are under '🎯 Decision'.",
        )
    )


# --- Ollama catalogue -------------------------------------------------------------------------
def section(title: str, description: str, count: int) -> None:
    st.markdown(
        f'<div class="mc-section"><h3>{esc(title)}</h3><span class="d">{esc(description)}</span>'
        f'<span class="c">{tr(f"{count} 個模型", f"{count} models")}</span></div>',
        unsafe_allow_html=True,
    )


def tier_range(t: dict) -> str:
    index = tiers.index(t)
    low = tiers[index - 1]["max_gb"] if index else None
    if t["max_gb"] is None:
        return f">{low}GB"
    return f"≤{t['max_gb']}GB" if low is None else f"{low}–{t['max_gb']}GB"


def search_text(m: dict) -> str:
    parts = [m["tag"], m["family"], m.get("params", ""), m["description"], m.get("note", ""),
             m.get("note_en", ""), m.get("license", ""), m.get("license_en", "")]  # fmt: skip
    for t in m["tasks"]:
        parts += [data["tasks"].get(t, ""), data["tasks_en"].get(t, "")]
    for k in catalog.all_capabilities(m):
        parts += [k, *ui.CAPS.get(k, ("", "", ""))[1:3]]
    return " ".join(parts).casefold()


def model_card(m: dict, group: str) -> None:
    """One product card; `group` keeps widget keys unique when a model is listed twice."""
    tag = m["tag"]
    have = is_installed(m)
    hf = catalog.is_huggingface(tag)
    name, origin = tag, "Ollama"
    if hf:  # hf.co/<user>/<repo>-GGUF:<quant> -> "<repo>" and "Hugging Face GGUF · <quant>"
        repo, _, quant = tag.split("/")[-1].partition(":")
        name, origin = repo.removesuffix("-GGUF"), f"Hugging Face GGUF · {quant}"
    tier = catalog.tier_of(m)
    flags = ""
    if m.get("default"):
        flags += f'<span class="mc-flag default">{tr("平台預設", "Default")}</span>'
    if m.get("recommended"):
        flags += f'<span class="mc-flag">⭐ {tr("推薦", "Pick")}</span>'
    specs = [
        (f"<b>{esc(m.get('params'))}</b>", tr("參數量", "Parameters")),
        (f"{m['size_gb']} GB", tr("這台 Mac 的下載大小", "Download size on this Mac")),
        (
            f"{catalog.context_label(m)} ctx",
            tr("Context 長度（tokens）", "Context window (tokens)"),
        ),
        (
            f"↻ {esc(m.get('updated'))}",
            tr("上游最後更新（年-月）", "Last updated upstream (year-month)"),
        ),
    ]
    spec_html = "".join(
        f'<span class="mc-spec" title="{esc(tip)}">{value}</span>' for value, tip in specs
    )
    keys = catalog.all_capabilities(m) + (["installed"] if have else [])
    licence = loc(m, "license")
    with st.container(border=True, height="stretch", gap="xsmall"):
        st.markdown(
            f'<div class="mc-top"><div class="mc-id"><div class="mc-name" title="{esc(tag)}">{esc(name)}</div>'
            f'<div class="mc-sub">{esc(m["family"])} · {esc(origin)}</div></div>'
            f'<div class="mc-flags">{flags}</div></div>'
            f'<div class="mc-specs">{spec_html}</div>'
            + ui.badges(keys)
            + f'<div class="mc-note" title="{esc(loc(m, "note"))}">{esc(loc(m, "note"))}</div>',
            unsafe_allow_html=True,
        )
        st.space("stretch")  # keeps the footer and buttons aligned across a row of cards
        st.markdown(
            f'<div class="mc-foot"><span class="fit-{m["fit"]}">{esc(fit_labels[m["fit"]])}</span>'
            f"<span>{esc(loc(tier, 'name'))}</span>"
            f'<span class="lic" title="{esc(licence)}">⚖️ {esc(licence)}</span></div>',
            unsafe_allow_html=True,
        )
        job = jobs.find("pull", model=tag)
        if not have and job and job.running:  # full-width live progress instead of the buttons
            download_control(tag, key=f"pull_{group}_{tag}")
            return
        row = st.container(horizontal=True, gap="small")
        if have:
            with row.popover(tr("🗑 刪除", "🗑 Delete"), disabled=bool(m.get("default")),
                             help=tr("平台預設模型不能刪除", "The platform default cannot be deleted")
                             if m.get("default") else None):  # fmt: skip
                st.caption(tr(f"從 Ollama 刪除 {tag}？之後可以再下載。",
                              f"Delete {tag} from Ollama? You can download it again later."))  # fmt: skip
                if st.button(tr("確定刪除", "Delete"), key=f"del_{group}_{tag}", type="primary"):
                    ollama_admin.delete(tag, ollama_url())
                    st.rerun()
        elif m["fit"] == "no":
            row.button(tr("⬇ 下載", "⬇ Download"), key=f"pull_{group}_{tag}", disabled=True,
                       help=tr("超過這台 Mac 的記憶體，建議放在 DGX Spark。",
                               "Too big for this Mac's memory; use a DGX Spark."))  # fmt: skip
        else:
            download_control(tag, key=f"pull_{group}_{tag}", container=row,
                             label=tr("⬇ 下載", "⬇ Download"))  # fmt: skip
        with row.popover(tr("詳細", "Details"), icon=":material/info:"):
            details(m, tier)


def details(m: dict, tier: dict) -> None:
    st.markdown(f"**`{m['tag']}`**")
    st.caption(tr("上游介紹（原文）", "Upstream description"))
    st.markdown(f"> {loc(m, 'description')}")
    st.caption(tr("平台建議", "Our take"))
    st.write(loc(m, "note"))
    context = f"{catalog.context_label(m)} ({m.get('context', 0):,} tokens)"
    ollama_caps = sep.join(ui.cap_label(c) for c in m.get("caps") or [])
    our_tags = sep.join(ui.cap_label(c) for c in m.get("tags") or [])
    rows = {
        tr("系列", "Family"): m["family"],
        tr("參數量", "Parameters"): m.get("params", ""),
        tr("Context 長度", "Context window"): context,
        tr("下載大小", "Download size"): f"{m['size_gb']} GB · {loc(tier, 'name')}",
        tr("在這台 Mac", "On this Mac"): fit_labels[m["fit"]],
        tr("適合任務", "Good for"): sep.join(tasks[t] for t in m["tasks"]),
        tr("Ollama 能力", "Ollama capabilities"): ollama_caps or tr("純文字", "text only"),
        tr("平台標籤", "Our labels"): our_tags or "—",
        tr("授權", "Licence"): loc(m, "license"),
        tr("上游更新", "Updated upstream"): m.get("updated", ""),
    }
    st.markdown(
        "\n".join(f"- **{k}**：{v}" if lang == "zh" else f"- **{k}**: {v}" for k, v in rows.items())
    )
    st.code(catalog.pull_command(m), language="bash")
    st.markdown(f"[{tr('開啟模型頁', 'Open the model page')} ↗]({catalog.page_url(m)})")


def grid(members: list[dict], group: str, per_row: int = 3) -> None:
    for start in range(0, len(members), per_row):
        cols = st.columns(per_row)
        for col, m in zip(cols, members[start : start + per_row], strict=False):
            with col:
                model_card(m, group)


SOURCES = {
    "ollama": tr("🦙 Ollama 模型", "🦙 Ollama models"),
    "other": tr(
        "🎙️ 非 Ollama（語音、文件、Reranker）", "🎙️ Non-Ollama (speech, documents, rerankers)"
    ),
}
source = st.segmented_control(
    tr("模型來源", "Model source"),
    list(SOURCES),
    format_func=SOURCES.get,
    default="ollama",
    required=True,
    key="models_source",
    persist_state="session",
    label_visibility="collapsed",
)
if source == "other":
    other_models_view()
    st.stop()

with st.expander(tr("級距總覽：各級距適合的任務", "Size tiers and the tasks they suit")):
    tiles = []
    for t in tiers:
        members = [m for m in all_models if catalog.tier_of(m)["key"] == t["key"]]
        tiles.append(
            f'<div class="tier-tile"><div class="t">{esc(loc(t, "name"))} '
            f'<span class="r">{esc(tier_range(t))}</span></div>'
            f'<div class="s">{esc(loc(t, "summary"))}</div>'
            f'<div class="n">{esc(sep.join(loc(t, "suggested")))}</div>'
            f'<div class="n"><b>{tr(f"{len(members)} 個模型", f"{len(members)} models")}</b></div></div>'
        )
    st.markdown(f'<div class="tier-strip">{"".join(tiles)}</div>', unsafe_allow_html=True)

GROUPS = {
    "tier": tr("依級距", "By size"),
    "task": tr("依任務", "By task"),
    "none": tr("不分組", "No grouping"),
}
SORTS = {
    "featured": tr("推薦優先", "Recommended first"),
    "small": tr("由小到大", "Smallest first"),
    "large": tr("由大到小", "Largest first"),
    "newest": tr("最近更新", "Recently updated"),
    "name": tr("名稱", "Name"),
}
cap_options = [
    k
    for k in ui.CAPS
    if k != "installed" and any(k in catalog.all_capabilities(m) for m in all_models)
]
tier_keys = [t["key"] for t in tiers]
tier_by_key = {t["key"]: t for t in tiers}

with st.container(border=True):
    c1, c2, c3 = st.columns([3.2, 2, 1.4], vertical_alignment="bottom")
    query = c1.text_input(
        tr("搜尋", "Search"),
        placeholder=tr(
            "搜尋名稱、系列或用途，例如 qwen、看圖、coding、台灣",
            "Search name, family or use, e.g. qwen, vision, coding, Taiwan",
        ),
        type="search",
        live="400ms",
        key="mf_query",
        persist_state="session",
        label_visibility="collapsed",
    )
    group = c2.segmented_control(
        tr("分組", "Group by"), list(GROUPS), format_func=GROUPS.get, default="tier",
        required=True, key="mf_group", persist_state="session", label_visibility="collapsed",
    )  # fmt: skip
    order = c3.selectbox(
        tr("排序", "Sort"), list(SORTS), format_func=SORTS.get, key="mf_sort",
        persist_state="session", label_visibility="collapsed",
    )  # fmt: skip
    caps_sel = st.pills(
        tr("能力（全部符合）", "Capabilities (must have all)"), cap_options, selection_mode="multi",
        format_func=ui.cap_label, key="mf_caps", persist_state="session",
    )  # fmt: skip
    c1, c2 = st.columns([1.35, 1])
    task_sel = c1.pills(
        tr("任務", "Task"), list(tasks), selection_mode="multi", format_func=tasks.get,
        key="mf_tasks", persist_state="session", wrap=True,
    )  # fmt: skip
    tier_sel = c2.pills(
        tr("大小級距", "Size tier"), tier_keys, selection_mode="multi",
        format_func=lambda k: f"{loc(tier_by_key[k], 'name')} {tier_range(tier_by_key[k])}",
        key="mf_tiers", persist_state="session", wrap=True,
    )  # fmt: skip
    c1, c2, c3 = st.columns([2.6, 1, 1], vertical_alignment="bottom")
    fam_sel = c1.multiselect(
        tr("系列", "Family"), catalog.families(), placeholder=tr("全部系列", "All families"),
        key="mf_family", persist_state="session",
    )  # fmt: skip
    only_installed = c2.toggle(
        tr("只看已下載", "Installed only"), key="mf_installed", persist_state="session"
    )
    only_rec = c3.toggle(
        tr("只看 ⭐ 推薦", "⭐ Recommended only"), key="mf_rec", persist_state="session"
    )

terms = (query or "").casefold().split()
rows = []
for m in all_models:
    caps = set(catalog.all_capabilities(m))
    if caps_sel and not set(caps_sel) <= caps:
        continue
    if task_sel and not set(task_sel) & set(m["tasks"]):
        continue
    if tier_sel and catalog.tier_of(m)["key"] not in tier_sel:
        continue
    if fam_sel and m["family"] not in fam_sel:
        continue
    if (only_installed and not is_installed(m)) or (only_rec and not m.get("recommended")):
        continue
    if terms and not all(t in search_text(m) for t in terms):
        continue
    rows.append(m)

SORT_KEYS = {
    "featured": lambda m: (
        not m.get("default"),
        not m.get("recommended"),
        not is_installed(m),
        m["size_gb"],
    ),
    "small": lambda m: m["size_gb"],
    "large": lambda m: -m["size_gb"],
    "newest": lambda m: tuple(-int(x) for x in m.get("updated", "0-0").split("-")),
    "name": lambda m: m["tag"].casefold(),
}
rows.sort(key=SORT_KEYS[order or "featured"])

FILTER_DEFAULTS = {"mf_query": "", "mf_caps": [], "mf_tasks": [], "mf_tiers": [], "mf_family": [],
                   "mf_installed": False, "mf_rec": False}  # fmt: skip


def clear_filters() -> None:
    st.session_state.update(FILTER_DEFAULTS)


c1, c2 = st.columns([4, 1], vertical_alignment="center")
c1.caption(
    tr(
        f"顯示 {len(rows)} / {len(all_models)} 個模型",
        f"Showing {len(rows)} of {len(all_models)} models",
    )
)
c2.button(tr("清除篩選", "Clear filters"), type="tertiary", icon=":material/filter_alt_off:",
          on_click=clear_filters, width="stretch")  # fmt: skip

if not rows:
    st.info(
        tr(
            "沒有符合條件的模型，試著移除一些篩選。",
            "No model matches these filters; try removing some.",
        )
    )
elif group == "task":
    for key, label in tasks.items():
        if members := [m for m in rows if key in m["tasks"]]:
            section(label, "", len(members))
            grid(members, key)
elif group == "none":
    grid(rows, "all")
else:
    for t in tiers:
        if members := [m for m in rows if catalog.tier_of(m)["key"] == t["key"]]:
            section(f"{loc(t, 'name')} · {tier_range(t)}", loc(t, "summary"), len(members))
            grid(members, t["key"])

with st.expander(tr("下載其他 Ollama 模型", "Download another Ollama model")):
    tag = st.text_input(
        tr("模型名稱", "Model name"),
        placeholder=tr(
            "例如 qwen3:14b，見 https://ollama.com/search?c=local",
            "e.g. qwen3:14b — see https://ollama.com/search?c=local",
        ),
    )
    if tag.strip():
        download_control(tag.strip(), key="pull_custom", label=tr("下載", "Download"))
