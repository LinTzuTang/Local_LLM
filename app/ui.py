"""Shared look and feel: global CSS, page headers, capability badges and small HTML pieces.

Pages call `page_header(...)` instead of `st.title` and `badges([...])` wherever models,
agents or nodes are described, so the same capability always looks the same everywhere.
No web fonts are loaded (the app must work offline): the system font stack on macOS gives
SF Pro + PingFang TC.
"""

import html

import streamlit as st
from i18n import tr

# capability key -> (icon, zh, en, colour)
CAPS = {
    "chat": ("💬", "對話", "Chat", "#4F46E5"),
    "tools": ("🛠️", "工具呼叫", "Tools", "#0284C7"),
    "thinking": ("🧠", "推理思考", "Thinking", "#7C3AED"),
    "vision": ("👁️", "看圖", "Vision", "#D97706"),
    "code": ("💻", "程式", "Code", "#334155"),
    "embedding": ("🧭", "向量嵌入", "Embedding", "#059669"),
    "decision": ("🎯", "決策分類", "Decision", "#DC2626"),
    "audio": ("🎧", "聽音訊", "Audio in", "#DB2777"),
    "asr": ("🎙️", "語音轉文字", "Speech-to-text", "#DB2777"),
    "tts": ("🔊", "語音合成", "Text-to-speech", "#BE185D"),
    "ocr": ("📄", "OCR／文件", "OCR / docs", "#475569"),
    "doc": ("📑", "文件解析", "Doc parsing", "#475569"),
    "rerank": ("📶", "重新排序", "Rerank", "#0F766E"),
    "translate": ("🌐", "翻譯", "Translation", "#2563EB"),
    "taiwan": ("🇹🇼", "台灣在地", "Taiwan", "#B91C1C"),
    "multilingual": ("🗺️", "多語言", "Multilingual", "#1D4ED8"),
    "long_context": ("📚", "長上下文", "Long context", "#6D28D9"),
    "moe": ("⚡", "MoE 省算力", "MoE", "#CA8A04"),
    "commercial_ok": ("✅", "可商用", "Commercial OK", "#15803D"),
    "non_commercial": ("⛔", "非商用", "Non-commercial", "#B45309"),
    "installed": ("✔", "已下載", "Installed", "#15803D"),
}

CSS = """
<style>
:root { --lab-primary: #4F46E5; --lab-muted: #64748B; --lab-border: #E5E7EB; }
html, body, [class*="css"] {
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "PingFang TC",
               "Noto Sans TC", "Helvetica Neue", Arial, sans-serif;
}
[data-testid="stAppDeployButton"], [data-testid="stMainMenu"] { display: none; }
.block-container { padding-top: 3.2rem; max-width: 1280px; }
h1, h2, h3, h4 { letter-spacing: -0.01em; }
/* bordered containers become cards */
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]) {
  border-radius: 14px; border-color: var(--lab-border);
  box-shadow: 0 1px 2px rgba(15, 23, 42, .04), 0 2px 8px rgba(15, 23, 42, .04);
  background: #FFFFFF;
}
div[data-testid="stExpander"] details { border-radius: 12px; }
button[kind="primary"] { border-radius: 10px; }
button[kind="secondary"] { border-radius: 10px; }
/* page header */
.lab-header { display: flex; align-items: center; gap: 14px; margin: 0 0 .25rem 0; }
.lab-header .icon { font-size: 1.9rem; width: 52px; height: 52px; border-radius: 14px;
  display: flex; align-items: center; justify-content: center;
  background: linear-gradient(135deg, #EEF2FF, #E0F2FE); }
.lab-header h1 { font-size: 1.85rem; font-weight: 750; margin: 0; padding: 0; line-height: 1.2; }
.lab-header .eyebrow { font-size: .78rem; font-weight: 650; text-transform: uppercase;
  letter-spacing: .06em; color: var(--lab-primary); }
.lab-sub { color: var(--lab-muted); font-size: .98rem; margin: .1rem 0 1.1rem 0; max-width: 980px; }
/* badges */
.lab-badges { display: flex; flex-wrap: wrap; gap: 6px; margin: 2px 0 4px 0; }
.lab-badge { display: inline-flex; align-items: center; gap: 4px; padding: 2px 9px;
  border-radius: 999px; font-size: .78rem; font-weight: 600; line-height: 1.5;
  border: 1px solid; white-space: nowrap; }
.lab-pill { display: inline-block; padding: 1px 8px; border-radius: 6px; font-size: .78rem;
  background: #F1F5F9; color: #334155;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.lab-dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 6px; }
.lab-kv { color: var(--lab-muted); font-size: .85rem; }
.lab-big { font-size: 1.6rem; font-weight: 750; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(icon: str, title: str, subtitle: str = "", eyebrow: str = "") -> None:
    """Consistent page title block (use instead of st.title)."""
    eyebrow_html = f'<div class="eyebrow">{html.escape(eyebrow)}</div>' if eyebrow else ""
    st.markdown(
        f'<div class="lab-header"><div class="icon">{icon}</div><div>{eyebrow_html}'
        f"<h1>{html.escape(title)}</h1></div></div>"
        + (f'<div class="lab-sub">{subtitle}</div>' if subtitle else ""),
        unsafe_allow_html=True,
    )


def badge(key: str, text: str | None = None) -> str:
    icon, zh, en, color = CAPS.get(key, ("•", key, key, "#64748B"))
    label = html.escape(text or tr(zh, en))
    return (
        f'<span class="lab-badge" style="color:{color};border-color:{color}33;'
        f'background:{color}0F">{icon} {label}</span>'
    )


def badges(keys, extra: list[str] | None = None) -> str:
    """HTML row of capability badges (render with st.markdown(..., unsafe_allow_html=True))."""
    parts = [badge(k) for k in dict.fromkeys(keys) if k]
    parts += [f'<span class="lab-pill">{html.escape(x)}</span>' for x in extra or []]
    return f'<div class="lab-badges">{"".join(parts)}</div>'


def show_badges(keys, extra: list[str] | None = None, container=st) -> None:
    container.markdown(badges(keys, extra), unsafe_allow_html=True)


def cap_label(key: str) -> str:
    icon, zh, en, _ = CAPS.get(key, ("•", key, key, ""))
    return f"{icon} {tr(zh, en)}"
