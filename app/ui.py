"""Shared look and feel: global CSS, page headers, capability badges and small HTML pieces.

Pages call `page_header(...)` instead of `st.title` and `badges([...])` wherever models,
agents or nodes are described, so the same capability always looks the same everywhere.
No web fonts are loaded (the app must work offline): the system font stack on macOS gives
SF Pro + PingFang TC.

Icons: one set everywhere, Google's Material Symbols (Rounded). Native widgets take
`":material/name:"` (`mi(name)` builds it); custom HTML uses `icon(name)`, which renders the
same font Streamlit bundles. Functions that take an icon accept "name", ":material/name:" or,
for backward compatibility, an emoji.
"""

import html
import re

import streamlit as st
from i18n import tr

# capability key -> (Material Symbols icon name, zh, en, colour)
CAPS = {
    "chat": ("chat_bubble", "對話", "Chat", "#4F46E5"),
    "tools": ("build", "工具呼叫", "Tools", "#0284C7"),
    "thinking": ("psychology", "推理思考", "Thinking", "#7C3AED"),
    "vision": ("visibility", "看圖", "Vision", "#D97706"),
    "code": ("code", "程式", "Code", "#334155"),
    "embedding": ("scatter_plot", "向量嵌入", "Embedding", "#059669"),
    "decision": ("alt_route", "決策分類", "Decision", "#DC2626"),
    "audio": ("headphones", "聽音訊", "Audio in", "#DB2777"),
    "asr": ("mic", "語音轉文字", "Speech-to-text", "#DB2777"),
    "tts": ("volume_up", "語音合成", "Text-to-speech", "#BE185D"),
    "ocr": ("document_scanner", "OCR／文件", "OCR / docs", "#475569"),
    "doc": ("description", "文件解析", "Doc parsing", "#475569"),
    "rerank": ("sort", "重新排序", "Rerank", "#0F766E"),
    "diarize": ("record_voice_over", "說話者分段", "Diarization", "#9333EA"),
    "stream": ("subtitles", "串流辨識", "Streaming ASR", "#C026D3"),
    "voice_chat": ("voice_chat", "語音對話", "Voice chat", "#7C3AED"),
    "speech_translate": ("interpreter_mode", "語音翻譯", "Speech translation", "#2563EB"),
    "vad": ("graphic_eq", "人聲偵測", "Voice activity", "#0891B2"),
    "translate": ("translate", "翻譯", "Translation", "#2563EB"),
    "taiwan": ("location_on", "台灣在地", "Taiwan", "#B91C1C"),
    "multilingual": ("language", "多語言", "Multilingual", "#1D4ED8"),
    "long_context": ("article", "長上下文", "Long context", "#6D28D9"),
    "moe": ("bolt", "MoE 省算力", "MoE", "#CA8A04"),
    "commercial_ok": ("verified", "可商用", "Commercial OK", "#15803D"),
    "non_commercial": ("block", "非商用", "Non-commercial", "#B45309"),
    "custom_license": ("gavel", "自訂授權", "Custom licence", "#92400E"),
    "gated": ("lock", "需同意條款", "Gated", "#475569"),
    "paid": ("payments", "付費", "Paid", "#B91C1C"),
    "narration": ("record_voice_over", "影片旁白常用", "Common for narration", "#7C3AED"),
    "cloud": ("cloud", "雲端", "Cloud", "#EA580C"),
    "installed": ("download_done", "已下載", "Installed", "#15803D"),
}
_NAME = re.compile(r"[a-z0-9_]+")
_SHORTCODE = re.compile(r":material/([a-z0-9_]+):")


def _name(icon: str) -> str | None:
    """Material Symbols name of ':material/x:' or a bare 'x'; None for emoji or other text."""
    icon = (icon or "").strip()
    if m := _SHORTCODE.fullmatch(icon):
        return m.group(1)
    return icon if _NAME.fullmatch(icon) else None


def mi(icon: str) -> str:
    """':material/x:' for native widgets (st.button icon=, labels, tabs, st.markdown)."""
    name = _name(icon)
    return f":material/{name}:" if name else icon


def icon(name: str, size: str | None = None, color: str | None = None, cls: str = "") -> str:
    """Material Symbols icon as HTML, for st.markdown(..., unsafe_allow_html=True).

    Uses the Material Symbols Rounded font that Streamlit ships (offline, same glyphs as
    `:material/x:` in native widgets). Emoji or other text is returned escaped, unchanged.
    """
    ms = _name(name)
    if not ms:
        return html.escape(name or "")
    style = "".join([f"font-size:{size};" if size else "", f"color:{color};" if color else ""])
    style_attr = f' style="{style}"' if style else ""
    return f'<span class="ms {cls}"{style_attr} translate="no" aria-hidden="true">{ms}</span>'


_icon = icon  # page_header's `icon` argument shadows the function


def html_icons(text: str) -> str:
    """Turn ':material/x:' shortcodes inside an HTML string into icon spans."""
    return _SHORTCODE.sub(lambda m: icon(m.group(1)), text)


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
[data-testid="stPopoverBody"] { min-width: 360px; }
button[kind="primary"] { border-radius: 10px; }
button[kind="secondary"] { border-radius: 10px; }
/* page header */
.lab-header { display: flex; align-items: center; gap: 14px; margin: 0 0 .25rem 0; }
.ms { font-family: "Material Symbols Rounded"; font-weight: normal; font-style: normal;
  font-size: 1.15em; line-height: 1; letter-spacing: normal; text-transform: none;
  display: inline-block; white-space: nowrap; word-wrap: normal; direction: ltr;
  font-feature-settings: "liga"; -webkit-font-smoothing: antialiased; vertical-align: -0.18em;
  user-select: none; }
.lab-header .icon { font-size: 1.9rem; width: 48px; height: 48px; border-radius: 14px;
  display: flex; align-items: center; justify-content: center; flex-shrink: 0;
  color: #4F46E5; background: linear-gradient(135deg, #EEF2FF, #E0F2FE);
  box-shadow: inset 0 0 0 1px #E0E7FF; }
.lab-header .icon .ms { font-size: 28px; vertical-align: 0; }
.lab-header h1 { font-size: 1.85rem; font-weight: 750; margin: 0; padding: 0; line-height: 1.2; }
.lab-header .eyebrow { font-size: .78rem; font-weight: 650; text-transform: uppercase;
  letter-spacing: .06em; color: var(--lab-primary); }
.lab-sub { color: var(--lab-muted); font-size: .98rem; margin: .1rem 0 1.1rem 0; max-width: 980px; }
/* badges */
.lab-badges { display: flex; flex-wrap: wrap; gap: 6px; margin: 2px 0 4px 0; }
.lab-badge { display: inline-flex; align-items: center; gap: 4px; padding: 2px 9px;
  border-radius: 999px; font-size: .78rem; font-weight: 600; line-height: 1.5;
  border: 1px solid; white-space: nowrap; }
.lab-badge .ms, .lab-pill .ms { font-size: 15px; vertical-align: 0; }
.lab-badges.sm { gap: 4px; margin: 0; }
.lab-badges.sm .lab-badge { padding: 0 7px; font-size: .72rem; }
.lab-badges.sm .lab-badge .ms { font-size: 13px; }
.lab-pill { display: inline-flex; align-items: center; gap: 4px; padding: 1px 8px;
  border-radius: 6px; font-size: .78rem;
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
    """Consistent page title block (use instead of st.title).

    icon: a Material Symbols name ("hub" or ":material/hub:"); an emoji still works.
    """
    eyebrow_html = f'<div class="eyebrow">{html.escape(eyebrow)}</div>' if eyebrow else ""
    st.markdown(
        f'<div class="lab-header"><div class="icon">{_icon(icon)}</div>'
        f"<div>{eyebrow_html}<h1>{html.escape(title)}</h1></div></div>"
        + (f'<div class="lab-sub">{html_icons(subtitle)}</div>' if subtitle else ""),
        unsafe_allow_html=True,
    )


def badge(key: str, text: str | None = None) -> str:
    glyph, zh, en, color = CAPS.get(key, ("", key, key, "#64748B"))
    label = html.escape(text or tr(zh, en))
    mark = icon(glyph) if glyph else ""
    return (
        f'<span class="lab-badge" style="color:{color};border-color:{color}33;'
        f'background:{color}0F">{mark} {label}</span>'
    )


def pill(text: str) -> str:
    """Grey monospace chip; a leading ':material/x:' becomes an icon."""
    return f'<span class="lab-pill">{html_icons(html.escape(text))}</span>'


def badges(keys, extra: list[str] | None = None, small: bool = False, limit: int = 0) -> str:
    """HTML row of capability badges (render with st.markdown(..., unsafe_allow_html=True)).

    extra: grey chips after the badges (":material/x: text" shows an icon). small: compact
    size for list rows. limit: show at most this many badges and a "+n" chip for the rest.
    """
    keys = [k for k in dict.fromkeys(keys) if k]
    rest = keys[limit:] if limit else []
    parts = [badge(k) for k in (keys[:limit] if limit else keys)]
    if rest:
        names = ", ".join(tr(*CAPS.get(k, ("", k, k))[1:3]) for k in rest)
        parts.append(f'<span class="lab-pill" title="{html.escape(names)}">+{len(rest)}</span>')
    parts += [pill(x) for x in extra or []]
    return f'<div class="lab-badges{" sm" if small else ""}">{"".join(parts)}</div>'


def show_badges(keys, extra: list[str] | None = None, container=st) -> None:
    container.markdown(badges(keys, extra), unsafe_allow_html=True)


def cap_icon(key: str) -> str:
    """':material/x:' of a capability, for native widgets."""
    return mi(CAPS.get(key, ("circle",))[0])


def cap_text(key: str) -> str:
    """Plain capability name (select box options and table cells cannot show icons)."""
    _, zh, en, _ = CAPS.get(key, ("", key, key, ""))
    return tr(zh, en)


def cap_label(key: str) -> str:
    """Icon + name for places that render Markdown (pills, st.markdown, help, tabs)."""
    return f"{cap_icon(key)} {cap_text(key)}"
