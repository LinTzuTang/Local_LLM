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


def plain(label: str) -> str:
    """Catalogue labels start with an emoji mark ("✅ fits"); the page draws its own icon."""
    head, _, rest = label.partition(" ")
    return rest if rest and not any(ch.isalnum() for ch in head) else label


fit_labels = {k: plain(v) for k, v in (catalog.FIT_LABELS_EN if lang == "en"
                                       else catalog.FIT_LABELS).items()}  # fmt: skip
mac_labels = {k: plain(v) for k, v in loc(data, "mac_labels").items()}
# status -> (icon, colour) for "fits on this Mac" and "runs on" marks
FIT_ICONS = {
    "ok": ("check", "#15803D"),
    "tight": ("warning", "#B45309"),
    "no": ("block", "#B91C1C"),
}
MAC_ICONS = {"mlx": ("bolt", "#15803D"), "mps": ("speed", "#B45309"), "cpu": ("warning", "#B91C1C"),
             "onnx": ("memory", "#15803D"), "cuda": ("block", "#B91C1C"),
             "native": ("laptop_mac", "#15803D")}  # fmt: skip
lang_labels: dict[str, str] = loc(data, "lang_labels")


def mark(label: str, glyph: tuple[str, str]) -> str:
    return f'<span class="ml-state" style="color:{glyph[1]}">{ui.icon(glyph[0])}{esc(label)}</span>'


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
/* list layout: one compact row per model */
.ml-grid {display: grid; gap: 12px; align-items: center;
  grid-template-columns: minmax(0, 2.3fr) minmax(0, .7fr) minmax(0, .6fr) minmax(0, .55fr)
    minmax(0, 3fr) minmax(0, .9fr);}
.ml-grid.cloud {grid-template-columns: minmax(0, 2.1fr) minmax(0, 1.15fr) minmax(0, .6fr)
    minmax(0, 3.2fr) minmax(0, .9fr);}
.ml-head {font-size: .7rem; font-weight: 650; text-transform: uppercase; letter-spacing: .05em;
  color: var(--lab-muted); padding: 2px 0 6px 0;}
.ml-name {display: flex; align-items: center; gap: 6px; min-width: 0;}
.ml-name b {font-weight: 700; font-size: .95rem; color: #0F172A; overflow: hidden;
  text-overflow: ellipsis; white-space: nowrap;}
.ml-sub {color: var(--lab-muted); font-size: .76rem; overflow: hidden; text-overflow: ellipsis;
  white-space: nowrap;}
.ml-num {font-size: .86rem; color: #1F2937; font-variant-numeric: tabular-nums;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
.ml-num.dim {color: #475569;}
.ml-state {font-size: .8rem; white-space: nowrap; display: inline-flex; align-items: center;
  gap: 4px;}
.ml-state .ms {font-size: 16px; vertical-align: 0;}
.ml-state.have {color: #15803D; font-weight: 650;}
.mc-flag {font-size: .68rem; font-weight: 700; padding: 1px 7px; border-radius: 999px;
  background: #FEF3C7; color: #92400E; white-space: nowrap; flex-shrink: 0;
  display: inline-flex; align-items: center; gap: 2px;}
.mc-flag .ms {font-size: 12px; vertical-align: 0;}
.mc-flag.default {background: #EEF2FF; color: #4338CA;}
.mc-chip {font-size: .74rem; font-weight: 650; padding: 1px 8px; border-radius: 999px;
  white-space: nowrap; display: inline-flex; align-items: center; gap: 3px; border: 1px solid;}
.mc-chip .ms {font-size: 13px; vertical-align: 0;}
.mc-chip.catalog {color: #475569; background: #F8FAFC; border-color: #CBD5E1;
  border-style: dashed;}
.mc-chip.builtin, .mc-chip.deployed {color: #15803D; background: #F0FDF4; border-color: #BBF7D0;}
.mc-chip.idle {color: #64748B; background: #F8FAFC; border-color: #E2E8F0;}
.mc-chip.cloud {color: #C2410C; background: #FFF7ED; border-color: #FDBA74;}
[class*="st-key-mrow_cat_"] .ml-name b {color: #334155; font-weight: 650;}
[class*="st-key-mrow_"] {border-top: 1px solid #EEF2F7; padding-top: 6px;}
[class*="st-key-mrow_"] button {min-height: 2rem; padding-top: 2px; padding-bottom: 2px;}
.mc-section .ms {color: var(--lab-primary); font-size: 1.2rem; vertical-align: -0.15em;}
.fit-ok {color: #15803D; font-weight: 650;} .fit-tight {color: #B45309; font-weight: 650;}
.fit-no {color: #B91C1C; font-weight: 650;}
@media (max-width: 900px) {.lab-stats {grid-template-columns: repeat(2, minmax(0, 1fr));}
  .tier-strip {grid-template-columns: repeat(2, minmax(0, 1fr));}}
</style>
""",
    unsafe_allow_html=True,
)

ui.page_header(
    "apps",
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


def stat(value, label: str, glyph: str = "") -> str:
    mark = ui.icon(glyph, color="var(--lab-primary)") + " " if glyph else ""
    return (
        f'<div class="lab-stat"><div class="v">{esc(value)}</div>'
        f'<div class="k">{mark}{esc(label)}</div></div>'
    )


st.markdown(
    '<div class="lab-stats">'
    + stat(len(all_models), tr("個 Ollama／GGUF 模型", "Ollama / GGUF models"), "deployed_code")
    + stat(
        sum(map(is_installed, all_models)),
        tr("已下載到這台 Mac", "downloaded to this Mac"),
        "download_done",
    )
    + stat(
        sum(bool(m.get("recommended")) for m in all_models),
        tr("平台推薦", "recommended"),
        "star",
    )
    + stat(
        len(catalog.other_models()),
        tr("非 Ollama 模型（語音、文件、Reranker）", "non-Ollama models (speech, docs, rerank)"),
        "graphic_eq",
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
def chip(text: str, cls: str, glyph: str = "", tip: str = "") -> str:
    """Small rounded status chip (catalog-only, cloud, deployed ...)."""
    title = f' title="{esc(tip)}"' if tip else ""
    return f'<span class="mc-chip {cls}"{title}>{ui.icon(glyph) if glyph else ""}{esc(text)}</span>'


def runs_on(key: str | None) -> str:
    """Short 'runs on' mark for a list row, with the full label as tooltip."""
    if not key:
        return "—"
    short = {"native": tr("內建", "BUILT-IN")}.get(key, key.upper())
    return f'<span title="{esc(mac_labels[key])}">{mark(short, MAC_ICONS[key])}</span>'


def lang_text(m: dict) -> str:
    return sep.join(lang_labels.get(k, k) for k in m.get("langs") or [])


def gateway_row(m: dict, status: dict | None, loaded: dict | None, up: bool) -> None:
    """One list row: download / delete / loaded state of a model served by the model gateway."""
    have = bool(status and status["installed"])
    entry = next(
        (o for o in catalog.other_models() if o.get("gateway", {}).get("id") == m["id"]), {}
    )
    keys = [
        m["task"],
        *(["taiwan"] if entry.get("taiwan") else []),
        *catalog.licence_keys(entry or m),
    ]
    size = (status or {}).get("size_bytes")
    # The gateway may serve its own pipeline for an entry (Docling: Heron + TableFormer instead of
    # granite-docling); then the entry's size, runtime and note do not describe what is served.
    own = m.get("name") == entry.get("name")
    params = loc(entry, "params") if own and entry.get("params") else "—"
    runs = runs_on(entry.get("mac")) if own else "—"
    note = " ".join(x for x in (loc(entry, "use"), loc(entry, "note")) if x)
    note = note if own and note else loc(m, "note")
    flags = (
        f'<span class="mc-flag">{ui.icon("star")}{tr("推薦", "Pick")}</span>'
        if m.get("recommended")
        else ""
    )
    if loaded:
        until = (loaded["expires_at"] or "")[11:16] or tr("不限", "never")
        state = (f'<span class="ml-state have">{ui.icon("memory")}'
                 + tr(f"已載入（{loaded['device']}，{until} 卸載）",
                      f"Loaded ({loaded['device']}, until {until})") + "</span>")  # fmt: skip
    elif have:
        state = f'<span class="ml-state have">{ui.icon("check_circle")}{tr("已下載", "Installed")}</span>'
    else:
        state = f'<span class="ml-state">{esc(tr("未下載", "Not downloaded"))}</span>'
    disk = tr(f"磁碟 {size / 1e9:.2f} GB", f"{size / 1e9:.2f} GB on disk") if size else ""
    facts, actions = st.container(key=f"mrow_gw_{m['id']}").columns(
        ROW_COLS, gap="small", vertical_alignment="center"
    )
    facts.markdown(
        f'<div class="ml-grid"><div title="{esc(note)}">'
        f'<div class="ml-name"><b>{esc(loc(m, "name"))}</b>{flags}</div>'
        f'<div class="ml-sub"><code>{esc(m["id"])}</code> · {esc(loc(m, "license"))}</div></div>'
        f'<span class="ml-num"><b>{esc(params)}</b></span>'
        f'<span class="ml-num" title="{esc(disk)}">{m["size_gb"]} GB</span>'
        f'<span class="ml-num dim">{runs}</span>'
        f"<div>{ui.badges(keys, small=True, limit=5)}</div>"
        f"<div>{state}</div></div>",
        unsafe_allow_html=True,
    )
    with actions:
        job = jobs.find("pull", model=m["id"])
        if up and not have and job and job.running:
            download_control(m["id"], key=f"gw_pull_{m['id']}", source="gateway")
            return
        row = st.container(horizontal=True, gap="small", horizontal_alignment="right")
        if not up:
            row.button(DOWNLOAD, key=f"gw_pull_{m['id']}", disabled=True,
                       help=tr("模型閘道沒有在執行", "The model gateway is not running"))  # fmt: skip
        elif have:
            if loaded and row.button("", key=f"gw_unload_{m['id']}", icon=":material/eject:",
                                     help=tr("卸載", "Unload")):  # fmt: skip
                gateway_client.unload(m["id"], gateway_url())
                st.rerun()
            with row.popover("", icon=":material/delete:", help=tr("刪除", "Delete")):
                st.caption(tr(f"刪除 {m['id']} 的權重？之後可以再下載。",
                              f"Delete the weights of {m['id']}? You can download them again later."))  # fmt: skip
                if st.button(tr("確定刪除", "Delete"), key=f"gw_del_{m['id']}", type="primary"):
                    gateway_client.delete(m["id"], gateway_url())
                    st.rerun()
        else:
            download_control(m["id"], key=f"gw_pull_{m['id']}", container=row,
                             label=DOWNLOAD, source="gateway")  # fmt: skip
        with row.popover("", icon=":material/info:", help=tr("詳細", "Details")):
            st.markdown(f"**{loc(m, 'name')}** · `{m['id']}`")
            if note:
                st.write(note)
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


CATALOG_TIP = tr(
    "可在模型庫查看；尚未接上平台，還不能一鍵下載，需自行安裝",
    "Listed in the library for reference; not wired into the platform yet, so there is no "
    "one-click download (install it yourself)",
)
CLOUD_TIP = tr(
    "雲端付費服務：錄音或文字會上傳到服務商，不在這台 Mac 執行",
    "Paid cloud service: the audio or text is uploaded; it does not run on this Mac",
)


def served_tip(via: str) -> str:
    return tr(f"平台已可用：經模型閘道的 {via}", f"Usable now through the gateway's {via}")


def catalog_row(m: dict, index: int) -> None:
    """One list row of a catalog-only entry: facts, a repo link and details, no download."""
    note = " ".join(x for x in (loc(m, "use"), loc(m, "note")) if x)
    keys = [m["category"], *(["taiwan"] if m.get("taiwan") else []), *catalog.licence_keys(m),
            *(["narration"] if m.get("narration") else [])]  # fmt: skip
    sub = sep.join(x for x in (lang_text(m), loc(m, "license")) if x)
    flags = (
        f'<span class="mc-flag">{ui.icon("star")}{tr("推薦", "Pick")}</span>'
        if m.get("recommended")
        else ""
    )
    via = m.get("served_by")
    if m.get("mac") == "native":
        state = chip(tr("內建，免下載", "Built in"), "builtin", "laptop_mac")
    elif via:  # a mode of a model the platform already serves
        state = chip(tr(f"經 {via} 可用", f"Via {via}"), "builtin", "link", served_tip(via))
    else:
        state = chip(tr("尚未接上平台", "Not in platform yet"), "catalog", "bookmark", CATALOG_TIP)
    params = loc(m, "params")
    size = m.get("size_gb")
    size_text = "—" if size is None else (tr("免下載", "none") if size == 0 else f"{size} GB")
    facts, actions = st.container(key=f"mrow_cat_{m['category']}_{index}").columns(
        ROW_COLS, gap="small", vertical_alignment="center"
    )
    facts.markdown(
        f'<div class="ml-grid"><div title="{esc(note)}">'
        f'<div class="ml-name"><b>{esc(loc(m, "name"))}</b>{flags}</div>'
        f'<div class="ml-sub">{esc(sub)}</div></div>'
        f'<span class="ml-num" title="{esc(params)}"><b>{esc(params)}</b></span>'
        f'<span class="ml-num">{esc(size_text)}</span>'
        f'<span class="ml-num dim">{runs_on(m.get("mac"))}</span>'
        f"<div>{ui.badges(keys, small=True, limit=4)}</div>"
        f"<div>{state}</div></div>",
        unsafe_allow_html=True,
    )
    with actions:
        row = st.container(horizontal=True, gap="small", horizontal_alignment="right")
        if url := catalog.other_url(m):
            row.link_button("", url, icon=":material/open_in_new:",
                            help=tr("開啟模型頁", "Open the model page"))  # fmt: skip
        with row.popover("", icon=":material/info:", help=tr("詳細", "Details")):
            st.markdown(f"**{loc(m, 'name')}**")
            if m.get("mac") == "native":
                st.caption(
                    tr("macOS 內建，在這台 Mac 直接可用", "Built into macOS; ready on this Mac")
                )
            else:
                st.caption(served_tip(via) if via else CATALOG_TIP)
            if note:
                st.write(note)
            facts_md = {
                tr("Repo", "Repo"): f"`{loc(m, 'repo')}`",
                tr("參數量", "Parameters"): loc(m, "params"),
                tr("下載大小", "Download size"): size_text,
                tr("語言", "Languages"): lang_text(m) or "—",
                tr("在這台 Mac", "On this Mac"): mac_labels.get(m.get("mac"), "—"),
                tr("執行方式", "Runtime"): loc(m, "runtime"),
                tr("安裝", "Install"): loc(m, "install"),
                tr("授權", "Licence"): loc(m, "license"),
                tr("釋出", "Released"): m.get("released"),
            }
            st.markdown("\n".join(
                f"- **{k}**：{v}" if lang == "zh" else f"- **{k}**: {v}"
                for k, v in facts_md.items() if v
            ))  # fmt: skip
            if url:
                st.markdown(f"[{tr('開啟模型頁', 'Open the model page')} ↗]({url})")


def cloud_row(m: dict) -> None:
    """One list row of a paid cloud speech model (Azure OpenAI, Google, AWS, ElevenLabs …):
    no download, data leaves the Mac."""
    task = ui.cap_text(m["category"])
    state = (
        chip(tr("已開通", "Available"), "deployed", "check_circle")
        if m.get("deployed")
        else chip(tr("未開通", "Not set up"), "idle", "radio_button_unchecked")
    )
    note = " ".join(x for x in (loc(m, "use"), loc(m, "note")) if x)
    facts, actions = st.container(key=f"mrow_cloud_{m['name']}").columns(
        ROW_COLS, gap="small", vertical_alignment="center"
    )
    facts.markdown(
        f'<div class="ml-grid cloud"><div title="{esc(note)}">'
        f'<div class="ml-name"><b>{esc(m["name"])}</b></div>'
        f'<div class="ml-sub" title="{esc(loc(m, "api"))}">'
        f"{esc(m.get('provider', 'Azure OpenAI'))} · {esc(loc(m, 'api'))}</div></div>"
        f"<div>{ui.badges(cloud_keys(m), small=True, limit=3)}</div>"
        f"<span>{chip(tr('雲端', 'Cloud'), 'cloud', 'cloud_upload', CLOUD_TIP)}</span>"
        f'<div class="ml-sub" style="white-space:normal">{esc(loc(m, "use"))}</div>'
        f"<div>{state}</div></div>",
        unsafe_allow_html=True,
    )
    with actions:
        row = st.container(horizontal=True, gap="small", horizontal_alignment="right")
        if m.get("url"):
            row.link_button("", m["url"], icon=":material/open_in_new:",
                            help=tr("開啟官方文件", "Open the official docs"))  # fmt: skip
        with row.popover("", icon=":material/info:", help=tr("詳細", "Details")):
            st.markdown(f"**{m['name']}** · {task}")
            st.caption(CLOUD_TIP)
            if note:
                st.write(note)


def cloud_keys(m: dict) -> list[str]:
    """Badges of a cloud row: its task, paid, and whether online videos often use it."""
    return [m["category"], *(["paid"] if m.get("paid", True) else []),
            *(["narration"] if m.get("narration") else [])]  # fmt: skip


def category_label(key: str) -> str:
    if key == "cloud":
        return f"{ui.cap_icon('cloud')} " + tr("雲端（付費）", "Cloud (paid)")
    return f"{ui.cap_icon(key)} {ui.cap_text(key)}"


def candidates_table(members: list[dict]) -> None:
    """The category as one table, for comparing runtimes and install steps side by side."""
    st.dataframe(
        pd.DataFrame(
            [
                {
                    tr("模型", "Model"): ("★ " if m.get("recommended") else "")
                    + loc(m, "name")
                    + (tr("（台灣）", " (Taiwan)") if m.get("taiwan") else ""),
                    tr("狀態", "Status"): f"✓ {m['gateway']['id']}"
                    if m.get("gateway")
                    else tr("尚未接上平台", "Not in platform yet"),
                    tr("用途", "Use"): loc(m, "use"),
                    tr("語言", "Languages"): lang_text(m),
                    tr("參數", "Params"): loc(m, "params"),
                    tr("在這台 Mac", "On this Mac"): mac_labels[m["mac"]],
                    tr("執行方式", "Runtime"): loc(m, "runtime"),
                    tr("授權", "License"): loc(m, "license"),
                    tr("模型頁", "Model page"): catalog.other_url(m),
                    tr("安裝", "Install"): loc(m, "install"),
                    tr("備註", "Note"): loc(m, "note"),
                }
                for m in members
            ]
        ),
        hide_index=True,
        width="stretch",
        column_config={
            tr("模型頁", "Model page"): st.column_config.LinkColumn(display_text=tr("開啟", "Open"))
        },
    )


# Ollama / gateway / catalog-only / cloud, one row per aspect: (zh cells, en cells).
RUNTIME_ROWS = [
    (["", "Ollama 模型", "模型閘道（本機 11500 port）", "尚未接上平台", "雲端（Azure OpenAI）"],
     ["", "Ollama models", "Model gateway (local port 11500)", "Not in platform yet",
      "Cloud (Azure OpenAI)"]),
    (["安裝", "平台按「下載」", "平台按「下載」", "自己裝 Python 套件與權重",
      "不用下載，要有 Azure 部署"],
     ["Install", "Download button", "Download button",
      "Install Python packages and weights yourself", "Nothing to download; needs a deployment"]),
    (["呼叫方式", "`/api/chat`",
      "`/v1/audio/transcriptions`、`/v1/documents/parse`、`/v1/rerank`", "各自的 Python API",
      "Azure OpenAI API"],
     ["API", "`/api/chat`", "`/v1/audio/transcriptions`, `/v1/documents/parse`, `/v1/rerank`",
      "Each library's own Python API", "Azure OpenAI API"]),
    (["記憶體", "自動載入／卸載", "第一次使用時載入，閒置 5 分鐘自動卸載",
      "自己管理，與 Ollama 共用 64GB", "不佔本機"],
     ["Memory", "Loaded / unloaded automatically",
      "Loaded on first use, unloaded after 5 idle minutes", "Your job; shares the 64GB with Ollama",
      "None on this Mac"]),
    (["Mac GPU", "Metal／MLX", "MLX（語音）、MPS（版面分析、reranker）",
      "看框架：MLX 可用、PyTorch 部分可用、CUDA 專用的不能跑", "—"],
     ["Mac GPU", "Metal / MLX", "MLX (speech), MPS (layout, reranker)",
      "Depends: MLX yes, PyTorch partly, CUDA-only no", "—"]),
    (["資料", "留在本機", "留在本機", "留在本機", "**上傳到雲端**"],
     ["Data", "Stays on this Mac", "Stays on this Mac", "Stays on this Mac",
      "**Uploaded to the cloud**"]),
    (["平台頁面", "對話、Agent、比較", "「語音與文件」頁、agent 工具（會議與文件助理）",
      "只在模型庫列出", "只在模型庫列出"],
     ["Pages", "Chat, Agent, Compare", "Speech & documents page, agent tools",
      "Listed in the library only", "Listed in the library only"]),
]  # fmt: skip


def runtime_table() -> str:
    rows = ["| " + " | ".join(tr(*pair) for pair in zip(zh, en, strict=True)) + " |"
            for zh, en in RUNTIME_ROWS]  # fmt: skip
    return "\n".join([rows[0], "|---" * len(RUNTIME_ROWS[0][0]) + "|", *rows[1:]])


def other_models_view() -> None:
    """Speech, document parsing and reranker models that Ollama cannot run, plus cloud ones."""
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
        st.markdown(runtime_table())
    categories: dict[str, str] = loc(data, "other_categories")
    notes: dict[str, str] = loc(data, "other_category_notes")
    options = [*categories, *(["cloud"] if catalog.cloud_models() else [])]
    with st.container(border=True):
        c1, c2 = st.columns([3, 1.3], vertical_alignment="center")
        query = c1.text_input(
            tr("搜尋", "Search"), type="search", live="400ms", key="mo_query",
            placeholder=tr("搜尋名稱、用途或授權，例如 台語、streaming、MLX、apache",
                           "Search name, use or licence, e.g. Hokkien, streaming, MLX, apache"),
            persist_state="session", label_visibility="collapsed",
        )  # fmt: skip
        only_platform = c2.toggle(
            tr("只看已接上平台", "In platform only"), key="mo_platform", persist_state="session",
            help=tr("隱藏只列在模型庫、還不能一鍵下載的模型", "Hide catalog-only entries"),
        )  # fmt: skip
        picked = st.pills(
            tr("類別（不選＝全部）", "Category (none = all)"), options, format_func=category_label,
            key="mo_category", persist_state="session", wrap=True,
        )  # fmt: skip
    terms = (query or "").casefold().split()

    def hit(m: dict) -> bool:
        text = " ".join(str(v) for v in m.values() if isinstance(v, str | list)).casefold()
        text += " " + lang_text(m).casefold() + " " + mac_labels.get(m.get("mac"), "").casefold()
        return all(t in text for t in terms)

    shown = 0
    for key, label in categories.items():
        if picked and picked != key:
            continue
        members = [m for m in catalog.other_models(key) if hit(m)]
        served_ids = {o["gateway"]["id"] for o in members if o.get("gateway")}
        served = [m for m in catalog.gateway_models() if m["id"] in served_ids]
        listed = [] if only_platform else [m for m in members if catalog.is_catalog_only(m)]
        listed.sort(key=lambda m: not m.get("recommended"))
        if not served and not listed:
            continue
        shown += 1
        section(label, "", len(served) + len(listed), ui.CAPS.get(key, ("",))[0])
        if notes.get(key):
            st.caption(notes[key])
        with st.container(border=True, gap="xsmall"):
            list_head(GW_HEAD)
            for m in served:
                gateway_row(m, status.get(m["id"]), loaded.get(m["id"]), bool(version))
            for i, m in enumerate(listed):
                catalog_row(m, i)
        with st.expander(tr("以表格比較（含安裝方式）", "Compare as a table (with install steps)")):
            candidates_table([m for m in members if m.get("gateway") or m in listed])
    clouds = [m for m in catalog.cloud_models() if hit(m)]
    if clouds and not only_platform and picked in (None, "cloud"):
        shown += 1
        section(tr("雲端語音模型（付費）", "Cloud speech models (paid)"), "",
                len(clouds), "cloud")  # fmt: skip
        st.caption(loc(data, "cloud_note"))
        with st.container(border=True, gap="xsmall"):
            list_head([tr("服務商與 API", "Provider & API"), tr("任務", "Task"),
                       tr("位置", "Where"), tr("用途", "Use"), tr("狀態", "Status")],
                      "cloud")  # fmt: skip
            for m in sorted(clouds, key=lambda m: not m.get("deployed")):
                cloud_row(m)
    if not shown:
        st.info(tr("沒有符合條件的模型，試著移除一些篩選。",
                   "No model matches these filters; try removing some."),
                icon=":material/search_off:")  # fmt: skip
    st.caption(
        tr(
            f"Ollama 上也有 OCR 與文件模型：在「Ollama 模型」勾選「{ui.cap_label('ocr')}」能力，可找到 glm-ocr、deepseek-ocr、granite3.2-vision、nemotron3。決策模型（nimble、clef、tev1…）勾選「{ui.cap_label('decision')}」。",
            f"Ollama also has OCR and document models: in 'Ollama models', pick the '{ui.cap_label('ocr')}' capability to find glm-ocr, deepseek-ocr, granite3.2-vision and nemotron3. Decision models (nimble, clef, tev1 …) are under '{ui.cap_label('decision')}'.",
        )
    )


# --- Ollama catalogue -------------------------------------------------------------------------
def section(title: str, description: str, count: int, glyph: str = "") -> None:
    mark = ui.icon(glyph) + " " if glyph else ""
    st.markdown(
        f'<div class="mc-section"><h3>{mark}{esc(title)}</h3><span class="d">{esc(description)}</span>'
        f'<span class="c">{tr(f"{count} 個模型", f"{count} models")}</span></div>',
        unsafe_allow_html=True,
    )


ROW_COLS = [6.2, 1.55]  # model facts | actions
DOWNLOAD = ":material/download: " + tr("下載", "Download")
GW_HEAD = [tr("模型", "Model"), tr("參數", "Params"), tr("大小", "Size"), tr("執行", "Runs on"),
           tr("用途與授權", "Use & licence"), tr("狀態", "Status")]  # fmt: skip


def flag_html(m: dict) -> str:
    out = ""
    if m.get("default"):
        out += f'<span class="mc-flag default">{tr("平台預設", "Default")}</span>'
    if m.get("recommended"):
        out += f'<span class="mc-flag">{ui.icon("star")}{tr("推薦", "Pick")}</span>'
    return out


def list_head(labels: list[str], grid: str = "") -> None:
    """Column titles above a list of model rows (same grid as the rows)."""
    left, _ = st.columns(ROW_COLS, gap="small")
    left.markdown(
        f'<div class="ml-grid {grid} ml-head">' + "".join(f"<span>{esc(x)}</span>" for x in labels)
        + "</div>",
        unsafe_allow_html=True,
    )  # fmt: skip


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


def model_row(m: dict, group: str) -> None:
    """One compact list row; `group` keeps widget keys unique when a model is listed twice."""
    tag = m["tag"]
    have = is_installed(m)
    name, origin = tag, "Ollama"
    if catalog.is_huggingface(tag):  # hf.co/<user>/<repo>-GGUF:<quant> -> "<repo>" + quant
        repo, _, quant = tag.split("/")[-1].partition(":")
        name, origin = repo.removesuffix("-GGUF"), f"Hugging Face GGUF · {quant}"
    sub = sep.join(x for x in (m["family"], origin) if x)
    if m.get("updated"):
        sub += tr(f" · 更新 {m['updated']}", f" · updated {m['updated']}")
    if have:
        state = f'<span class="ml-state have">{ui.icon("check_circle")}{tr("已下載", "Installed")}</span>'
    else:
        state = mark(fit_labels[m["fit"]], FIT_ICONS[m["fit"]])
    caps = catalog.all_capabilities(m)
    facts, actions = st.container(key=f"mrow_{group}_{tag}").columns(
        ROW_COLS, gap="small", vertical_alignment="center"
    )
    facts.markdown(
        f'<div class="ml-grid"><div title="{esc(loc(m, "note"))}">'
        f'<div class="ml-name"><b>{esc(name)}</b>{flag_html(m)}</div>'
        f'<div class="ml-sub">{esc(sub)}</div></div>'
        f'<span class="ml-num" title="{esc(m.get("params"))}"><b>{esc(m.get("params"))}</b></span>'
        f'<span class="ml-num">{m["size_gb"]} GB</span>'
        f'<span class="ml-num dim">{esc(catalog.context_label(m))}</span>'
        f"<div>{ui.badges(caps, small=True, limit=3)}</div>"
        f"<div>{state}</div></div>",
        unsafe_allow_html=True,
    )
    with actions:
        job = jobs.find("pull", model=tag)
        if not have and job and job.running:  # live progress instead of the buttons
            download_control(tag, key=f"pull_{group}_{tag}")
            return
        row = st.container(horizontal=True, gap="small", horizontal_alignment="right")
        if have:
            with row.popover(
                "", icon=":material/delete:", disabled=bool(m.get("default")),
                help=tr("平台預設模型不能刪除", "The platform default cannot be deleted")
                if m.get("default") else tr("刪除", "Delete"),
            ):  # fmt: skip
                st.caption(tr(f"從 Ollama 刪除 {tag}？之後可以再下載。",
                              f"Delete {tag} from Ollama? You can download it again later."))  # fmt: skip
                if st.button(tr("確定刪除", "Delete"), key=f"del_{group}_{tag}", type="primary"):
                    ollama_admin.delete(tag, ollama_url())
                    st.rerun()
        elif m["fit"] == "no":
            row.button(DOWNLOAD, key=f"pull_{group}_{tag}", disabled=True,
                       help=tr("超過這台 Mac 的記憶體，建議放在 DGX Spark。",
                               "Too big for this Mac's memory; use a DGX Spark."))  # fmt: skip
        else:
            download_control(tag, key=f"pull_{group}_{tag}", container=row, label=DOWNLOAD)
        with row.popover("", icon=":material/info:", help=tr("詳細", "Details")):
            details(m, catalog.tier_of(m))


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
        tr(
            "在這台 Mac", "On this Mac"
        ): f":material/{FIT_ICONS[m['fit']][0]}: {fit_labels[m['fit']]}",
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


PAGE_SIZE = 25


SOURCES = {
    "ollama": ":material/deployed_code: " + tr("Ollama 模型", "Ollama models"),
    "other": ":material/graphic_eq: "
    + tr("非 Ollama（語音、文件、Reranker）", "Non-Ollama (speech, documents, rerankers)"),
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
    # Task, size tier and family live in a popover so the list starts higher on the page.
    more = sum(len(st.session_state.get(k) or []) for k in ("mf_tasks", "mf_tiers", "mf_family"))
    c1, c2, c3, _ = st.columns([1.3, 1, 1.2, 2.6], vertical_alignment="center")
    with c1.popover(
        tr("更多篩選", "More filters") + (f" · {more}" if more else ""),
        icon=":material/filter_list:",
        width="stretch",
    ):
        task_sel = st.pills(
            tr("任務", "Task"), list(tasks), selection_mode="multi", format_func=tasks.get,
            key="mf_tasks", persist_state="session", wrap=True,
        )  # fmt: skip
        tier_sel = st.pills(
            tr("大小級距", "Size tier"), tier_keys, selection_mode="multi",
            format_func=lambda k: f"{loc(tier_by_key[k], 'name')} {tier_range(tier_by_key[k])}",
            key="mf_tiers", persist_state="session", wrap=True,
        )  # fmt: skip
        fam_sel = st.multiselect(
            tr("系列", "Family"), catalog.families(), placeholder=tr("全部系列", "All families"),
            key="mf_family", persist_state="session",
        )  # fmt: skip
    only_installed = c2.toggle(
        tr("只看已下載", "Installed only"), key="mf_installed", persist_state="session"
    )
    only_rec = c3.toggle(tr("只看推薦", "Recommended only"), key="mf_rec", persist_state="session")

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


# Rows in display order, with their group: (group key, title, description, icon, model).
if group == "task":
    entries = [(k, label, "", "task_alt", m) for k, label in tasks.items()
               for m in rows if k in m["tasks"]]  # fmt: skip
elif group == "none":
    entries = [("all", "", "", "", m) for m in rows]
else:
    entries = [(t["key"], f"{loc(t, 'name')} · {tier_range(t)}", loc(t, "summary"), "memory", m)
               for t in tiers for m in rows if catalog.tier_of(m)["key"] == t["key"]]  # fmt: skip
group_sizes: dict[str, int] = {}
for key, *_ in entries:
    group_sizes[key] = group_sizes.get(key, 0) + 1

# Paging: a new search or filter starts again at page 1.
pages = max(1, -(-len(entries) // PAGE_SIZE))
signature = repr([query, caps_sel, task_sel, tier_sel, fam_sel, only_installed, only_rec,
                  group, order])  # fmt: skip
if st.session_state.get("mf_signature") != signature:
    st.session_state.mf_signature = signature
    st.session_state.mf_page = 1
if st.session_state.get("mf_page") not in range(1, pages + 1):
    st.session_state.mf_page = 1

c1, c2, c3 = st.columns([2.2, 2.6, 1], vertical_alignment="center")
page = st.session_state.mf_page
first, last = (page - 1) * PAGE_SIZE, min(page * PAGE_SIZE, len(entries))
c1.caption(
    tr(
        f"符合 {len(rows)} / {len(all_models)} 個模型"
        + (f" · 第 {first + 1}–{last} 列" if pages > 1 else ""),
        f"{len(rows)} of {len(all_models)} models match"
        + (f" · rows {first + 1}–{last}" if pages > 1 else ""),
    )
)
if pages > 1:
    with c2:
        page = st.segmented_control(
            tr("頁數", "Page"), list(range(1, pages + 1)), key="mf_page", required=True,
            format_func=lambda n: tr(f"第 {n} 頁", f"Page {n}") if n == 1 else str(n),
            label_visibility="collapsed",
        )  # fmt: skip
        first, last = (page - 1) * PAGE_SIZE, min(page * PAGE_SIZE, len(entries))
c3.button(tr("清除篩選", "Clear filters"), type="tertiary", icon=":material/filter_alt_off:",
          on_click=clear_filters, width="stretch")  # fmt: skip

HEAD = [tr("模型", "Model"), tr("參數", "Params"), tr("大小", "Size"), "Context",
        tr("能力", "Capabilities"), tr("這台 Mac", "This Mac")]  # fmt: skip
if not rows:
    st.info(
        tr(
            "沒有符合條件的模型，試著移除一些篩選。",
            "No model matches these filters; try removing some.",
        ),
        icon=":material/search_off:",
    )
shown_group = None
box = None
for key, title, description, glyph, m in entries[first:last]:
    if key != shown_group:
        shown_group = key
        if title:
            section(title, description, group_sizes[key], glyph)
        box = st.container(border=True, gap="xsmall")
        with box:
            list_head(HEAD)
    with box:
        model_row(m, key)

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
