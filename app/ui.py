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
# Local vs cloud, the same on every chart and table of the speech pages: blue runs on this
# machine, orange in the cloud (audio or text leaves it).
LOCAL_COLOR, CLOUD_COLOR = "#2563EB", "#EA580C"
LOCAL_TINT, CLOUD_TINT = "rgba(37, 99, 235, .16)", "rgba(234, 88, 12, .16)"


def where_names() -> tuple[str, str]:
    from i18n import tr

    return tr("本機", "Local"), tr("雲端", "Cloud")


def where_scale():
    """Altair colour scale for a "where" field holding where_names()."""
    import altair as alt

    local, cloud = where_names()
    return alt.Scale(domain=[local, cloud], range=[LOCAL_COLOR, CLOUD_COLOR])


def is_cloud(where) -> bool:
    return str(where).strip().lower().startswith(("雲端", "cloud"))


def where_rows(df, column: str):
    """A dataframe Styler tinting each row by its `column` (local blue, cloud orange; values
    such as 「雲端（Azure）」 count as cloud)."""

    def tint(row):
        colour = CLOUD_TINT if is_cloud(row[column]) else LOCAL_TINT
        return [f"background-color: {colour}"] * len(row)

    return df.style.apply(tint, axis=1)


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


# Two looks, picked with the day/night switch (theme_switch.py) or the system setting: "lab bench"
# (warm paper, ink) by day, "lab at night" (soft charcoal) by night. One muted teal throughout,
# monospace for labels and numbers; colours are kept low in saturation so long sessions are easy
# on the eyes. Streamlit's own widgets use the same values from .streamlit/config.toml.
PRIMARY = "#4E9C93"  # bars and diagram accents: reads on both backgrounds
PRIMARY_TINT = "rgba(78, 156, 147, .14)"

PALETTES = {
    "light": {
        "primary": "#2F7F76", "primary-strong": "#24655E", "tint": "#E4EFEC",
        "signal": "#4FA39A", "paper": "#F5F4F0", "surface": "#FFFFFF", "surface-2": "#F1EFE9",
        "text": "#22282B", "heading": "#15191B", "text-2": "#3E474A", "text-3": "#545D60",
        "muted": "#6B7174", "faint": "#8A9194", "border": "#DDD9D0", "border-strong": "#CBC6BB",
        "track": "#E9E6DF", "other": "#CFCAC0", "ok": "#2F855A", "warn": "#A16207",
        "warn-2": "#C05621", "bad": "#B83232", "header": "rgba(245, 244, 240, .88)",
        "glow-a": "rgba(78, 156, 147, .06)", "glow-b": "rgba(122, 160, 200, .06)",
        "card-sheen": "rgba(255, 255, 255, 0)", "selection": "#CFE5E0",
    },
    "dark": {
        "primary": "#6BBAAF", "primary-strong": "#9ED3CA", "tint": "rgba(107, 186, 175, .12)",
        "signal": "#7CC4B9", "paper": "#141B1D", "surface": "#182123", "surface-2": "#1D2829",
        "text": "#D8E1DF", "heading": "#EEF3F2", "text-2": "#B8C6C4", "text-3": "#A2B2B0",
        "muted": "#8A9997", "faint": "#7D8C8A", "border": "#2B3A3C", "border-strong": "#3A4C4E",
        "track": "#243235", "other": "#4A5F62", "ok": "#7FC79B", "warn": "#E0B866",
        "warn-2": "#E2A27A", "bad": "#E59A9A", "header": "rgba(20, 27, 29, .82)",
        "glow-a": "rgba(107, 186, 175, .05)", "glow-b": "rgba(130, 165, 205, .05)",
        "card-sheen": "rgba(255, 255, 255, .02)", "selection": "rgba(107, 186, 175, .35)",
    },
}  # fmt: skip


def theme() -> str:
    """'light' or 'dark': what the browser shows now (the switch or the system setting)."""
    try:
        kind = st.context.theme.type
    except Exception:  # outside a browser session (tests, scripts)
        kind = None
    return kind if kind in PALETTES else "dark"


def is_dark() -> bool:
    return theme() == "dark"


def on_dark(color: str) -> str:
    """A colour readable as text on the dark background: dark ones are mixed with white."""
    try:
        r, g, b = (int(color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return color
    lum = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255
    if lum >= 0.45:
        return color
    t = min(0.6, 0.45 - lum + 0.25)  # the darker, the more white
    r, g, b = (round(c + (255 - c) * t) for c in (r, g, b))
    return f"#{r:02X}{g:02X}{b:02X}"


_DOT_FILL = re.compile(r'(fillcolor\s*=\s*")([^"]+)(")')
_DOT_LINE = re.compile(r'((?<!fill)color\s*=\s*")(#[0-9A-Fa-f]{6})(")')
_DOT_TEXT = re.compile(r'((?:fontcolor\s*=\s*|<font[^>]*?color=)")(#[0-9A-Fa-f]{6}|white|black)(")')
_NAMED = {"white": "#FFFFFF", "black": "#000000"}


def _hex_lum(color: str) -> float | None:
    color = _NAMED.get(color.lower(), color)
    try:
        r, g, b = (int(color.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4))
    except ValueError:
        return None
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _hls_hex(color: str, lightness: float, max_sat: float) -> str:
    import colorsys

    color = _NAMED.get(color.lower(), color)
    r, g, b = (int(color.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4))
    h, _, s = colorsys.rgb_to_hls(r, g, b)
    r, g, b = colorsys.hls_to_rgb(h, lightness, min(s, max_sat))
    return f"#{round(r * 255):02X}{round(g * 255):02X}{round(b * 255):02X}"


def themed_dot(dot: str) -> str:
    """A Graphviz diagram drawn for the current theme. Diagrams are written with light pastel
    nodes and dark text; at night the pastel fills become deep tints of the same hue, light
    borders darken, dark text turns light (and unset text, black by default, too)."""
    if not is_dark():
        return dot

    def fill(m: re.Match) -> str:
        lum = _hex_lum(m.group(2))
        if lum is None or lum < 0.6:  # already a strong colour (START, status): keep
            return m.group(0)
        return m.group(1) + _hls_hex(m.group(2), 0.19, 0.32) + m.group(3)

    def line(m: re.Match) -> str:
        lum = _hex_lum(m.group(2))
        if lum is None or lum < 0.7:
            return m.group(0)
        return m.group(1) + _hls_hex(m.group(2), 0.40, 0.35) + m.group(3)

    def text(m: re.Match) -> str:
        colour = _NAMED.get(m.group(2).lower(), m.group(2))
        lum = _hex_lum(colour)
        if lum is None or lum >= 0.45:
            return m.group(0)
        return m.group(1) + on_dark(colour) + m.group(3)

    dot = _DOT_FILL.sub(fill, dot)
    dot = _DOT_LINE.sub(line, dot)
    dot = _DOT_TEXT.sub(text, dot)
    # Text with no colour set is black: make the default light (later attributes still win).
    dot = re.sub(r"\bnode\s*\[", 'node [fontcolor="#D8E1DF", ', dot)
    return dot


def chip_style(color: str) -> str:
    """Inline style of a coloured chip (text, faint border and fill) for the current theme."""
    text = on_dark(color) if is_dark() else color
    return f"color:{text};border-color:{text}40;background:{text}12"


CSS = """
<style>
:root {
  --lab-ink: #142326; --lab-ink-2: #1A2D30;
  --lab-mono: "JetBrains Mono", ui-monospace, "SF Mono", Menlo, "DejaVu Sans Mono", monospace;
  --lab-sans: Inter, -apple-system, BlinkMacSystemFont, "PingFang TC", "Noto Sans TC",
    "Noto Sans CJK TC", "Helvetica Neue", Arial, sans-serif;
}
html, body, [class*="css"] { font-family: var(--lab-sans); font-feature-settings: "cv11", "ss03"; }
[data-testid="stAppDeployButton"], [data-testid="stMainMenu"] { display: none; }
.block-container { padding-top: 3.2rem; max-width: 1280px; }
h1, h2, h3, h4 { letter-spacing: -0.02em; }
/* card title: a small tinted icon tile before the text (ui.card_title) */
.lab-card-t { display: flex; align-items: center; gap: 9px; font-weight: 700; font-size: 1rem;
  color: var(--lab-heading); margin: 0 0 .35rem 0; }
.lab-card-t .ms { width: 26px; height: 26px; border-radius: 7px; font-size: 17px;
  display: inline-flex; align-items: center; justify-content: center; vertical-align: 0;
  color: var(--lab-primary); background: var(--lab-tint); }
::selection { background: var(--lab-selection); }
.stApp { background:
  radial-gradient(900px 500px at 85% -10%, var(--lab-glow-b), transparent 60%),
  radial-gradient(800px 500px at -10% 10%, var(--lab-glow-a), transparent 60%),
  var(--lab-paper); }

/* top bar: a quiet underline under the active page */
[data-testid="stHeader"] { background: var(--lab-header); backdrop-filter: blur(10px);
  border-bottom: 1px solid var(--lab-border); }
[data-testid="stTopNavLink"], [data-testid="stTopNavSection"] { border-radius: 8px; }
[data-testid="stTopNavLink"][aria-current="page"] { background: transparent; border-radius: 0;
  box-shadow: inset 0 -2px 0 var(--lab-primary); }
[data-testid="stTopNavLink"][aria-current="page"] span { color: var(--lab-heading);
  font-weight: 700; }

/* bordered containers become flat cards */
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]) {
  border-radius: 12px; border-color: var(--lab-border);
  background: linear-gradient(180deg, var(--lab-card-sheen), transparent), var(--lab-surface);
  transition: border-color .25s;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]):hover {
  border-color: var(--lab-border-strong); }
div[data-testid="stExpander"] details { border-radius: 12px; border-color: var(--lab-border);
  background: var(--lab-surface); }
div[data-testid="stExpander"] summary:hover { color: var(--lab-primary); }
[data-testid="stPopoverBody"] { min-width: 360px; }
button[kind="primary"], button[kind="secondary"] { border-radius: 9px; font-weight: 600; }
button[kind="secondary"] { background: var(--lab-surface); }
button[kind="secondary"]:hover { border-color: var(--lab-primary); color: var(--lab-primary); }
button[kind="primary"] { transition: transform .15s, filter .15s; }
button[kind="primary"]:hover { transform: translateY(-1px); filter: brightness(1.06); }
[data-baseweb="tab-list"] { gap: 4px; }
[data-baseweb="tab"] { font-weight: 600; }

/* page header: ink tile with the page icon */
.lab-header { display: flex; align-items: center; gap: 16px; margin: 0 0 .3rem 0; }
.ms { font-family: "Material Symbols Rounded"; font-weight: normal; font-style: normal;
  font-size: 1.15em; line-height: 1; letter-spacing: normal; text-transform: none;
  display: inline-block; white-space: nowrap; word-wrap: normal; direction: ltr;
  font-feature-settings: "liga"; -webkit-font-smoothing: antialiased; vertical-align: -0.18em;
  user-select: none; }
.lab-header .icon { width: 50px; height: 50px; border-radius: 13px; flex-shrink: 0;
  display: flex; align-items: center; justify-content: center; color: #8FCFC5;
  background: radial-gradient(circle at 30% 20%, #24403F, var(--lab-ink) 75%);
  box-shadow: inset 0 0 0 1px rgba(255,255,255,.07), 0 6px 16px -8px rgba(10, 20, 22, .45); }
.lab-header .icon .ms { font-size: 27px; vertical-align: 0; }
.lab-header h1 { font-size: 1.9rem; font-weight: 760; margin: 0; padding: 0; line-height: 1.15;
  color: var(--lab-heading); }
.lab-header .eyebrow { font-size: .76rem; font-weight: 700; letter-spacing: .06em;
  text-transform: uppercase; color: var(--lab-primary); }
.lab-eyebrow { font-family: var(--lab-mono); font-size: .72rem; font-weight: 600;
  text-transform: uppercase; letter-spacing: .12em; color: var(--lab-primary); }
.lab-sub { color: var(--lab-muted); font-size: .98rem; margin: .15rem 0 1.2rem 0;
  max-width: 980px; }

/* section title: numbered monospace label and a hairline */
.lab-section { display: flex; align-items: baseline; gap: 12px; margin: 1.6rem 0 .7rem 0;
  padding-bottom: .45rem; border-bottom: 1px solid var(--lab-border); }
.lab-section .n { font-family: var(--lab-mono); font-size: .72rem; font-weight: 700;
  letter-spacing: .08em; color: var(--lab-primary); padding: 1px 8px; border-radius: 6px;
  background: var(--lab-tint); border: 1px solid var(--lab-border); align-self: center; }
.lab-section h3 { margin: 0; padding: 0; font-size: 1.22rem; font-weight: 740;
  color: var(--lab-heading); }
.lab-section .d { margin-left: auto; color: var(--lab-muted); font-size: .84rem; }

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
  border-radius: 6px; font-size: .78rem; background: var(--lab-surface-2);
  color: var(--lab-text-2); border: 1px solid var(--lab-border); font-family: var(--lab-mono); }
.lab-dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 6px; }
.lab-kv { color: var(--lab-muted); font-size: .85rem; }
.lab-big { font-size: 1.6rem; font-weight: 750; font-variant-numeric: tabular-nums; }
/* Run history is a plain link (main.py), which Streamlit always puts before the section menus;
   move it to the end of the top bar. The menu sets each item's order inline. */
.rc-overflow-item:has(a[data-testid="stTopNavLink"][href$="/history"]) { order: 90 !important; }
</style>
"""


def _palette_css(name: str) -> str:
    rules = "; ".join(f"--lab-{k}: {v}" for k, v in PALETTES[name].items())
    return f"<style>:root {{ {rules}; }}</style>"


def card_title(icon_name: str, title: str) -> None:
    """A card's title with a small tinted icon tile, the same on every card."""
    st.markdown(
        f'<div class="lab-card-t">{_icon(icon_name)}<span>{html.escape(title)}</span></div>',
        unsafe_allow_html=True,
    )


def section(number: str, title: str, note: str = "") -> None:
    """A numbered section title ("01", "系統現況"), the same on every page that uses one."""
    note_html = f'<span class="d">{html.escape(note)}</span>' if note else ""
    st.markdown(
        f'<div class="lab-section"><span class="n">{html.escape(number)}</span>'
        f"<h3>{html.escape(title)}</h3>{note_html}</div>",
        unsafe_allow_html=True,
    )


def inject_css() -> None:
    st.markdown(_palette_css(theme()) + CSS, unsafe_allow_html=True)


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
    return f'<span class="lab-badge" style="{chip_style(color)}">{mark} {label}</span>'


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
