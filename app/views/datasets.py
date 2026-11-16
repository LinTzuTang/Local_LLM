"""Test sets: every test set we used (data/lab/datasets.json), what it tests, its source and
licence, how to use it on the platform, and a download. Language-model tool-calling sets download
in the format the Tool calling mode of views/llm.py reads back (Upload a set); speech samples
download as their audio with the reference text, which the speech pages take as an upload.
Sets the platform cannot run (multi-turn, search + memory) or must not offer (company-internal tools)
are listed with the reason and no download.
"""

import io
import json
import zipfile
from pathlib import Path

import streamlit as st
import ui
from i18n import pick, tr

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "data" / "lab" / "datasets.json"

ui.page_header(
    "dataset",
    tr("測試資料集", "Test sets"),
    tr(
        "我們評測用過的測試資料：測什麼、從哪來、授權、在平台上怎麼用。可以下載的資料集，"
        "下載後就能在平台上使用：語言模型的工具呼叫題組可在「語言模型比較 → 工具呼叫測試」上傳，"
        "語音範例可在語音頁面上傳並貼上參考逐字稿。",
        "The test data behind our evaluations: what each set tests, where it comes from, its "
        "licence and how to use it here. Downloadable sets work on the platform as they are: "
        "tool-calling sets upload into Language models → Tool calling, speech samples upload on "
        "the speech pages with their reference text.",
    ),
    eyebrow=tr("模型", "Models"),
)


@st.cache_data
def catalog(mtime: float) -> dict:
    return json.loads(CATALOG.read_text())


@st.cache_data(max_entries=8)
def payload(paths: tuple[str, ...], mtimes: tuple[float, ...]) -> bytes:
    """One file as it is, several as a zip (flat: the names are unique)."""
    if len(paths) == 1:
        return (ROOT / paths[0]).read_bytes()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            z.write(ROOT / p, Path(p).name)
    return buf.getvalue()


def size_text(n: int) -> str:
    return f"{n / 1e6:.1f} MB" if n >= 1e6 else f"{max(n / 1e3, 1):.0f} KB"


if not CATALOG.exists():
    st.warning(tr("找不到 data/lab/datasets.json。", "data/lab/datasets.json is missing."))
    st.stop()
cat = catalog(CATALOG.stat().st_mtime)
groups = cat["groups"]
st.session_state.setdefault("ds_group", "llm")
group = (
    st.segmented_control(
        tr("類別", "Group"),
        list(groups),
        format_func=lambda g: f"{pick(groups[g])}（{sum(s['group'] == g for s in cat['sets'])}）",
        key="ds_group",
        label_visibility="collapsed",
    )
    or "llm"
)

if group == "llm":
    with st.expander(
        tr("題目全部放上平台會不會太多？", "Is the whole question bank too much for the platform?"),
        icon=":material/help:",
    ):
        st.markdown(
            tr(
                "不會。工具呼叫的三組題目共 1,198 題、約 4 MB，伺服器啟動後只讀一次並快取；"
                "頁面上先選題庫、再選類型，一個清單最多 80 題，而且可以打字搜尋，"
                "所以不會拖慢平台。多輪、搜尋＋記憶要模擬環境或外部服務，平台上無法執行，只列說明。",
                "No. The three tool-calling sets are 1,198 questions, about 4 MB, read once per "
                "server process and cached. The page narrows them by suite and category first "
                "(at most 80 in a list, searchable), so they do not slow the platform. Multi-turn "
                "and search + memory need a simulated environment or outside services, so they "
                "are only described.",
            )
        )

for s in [x for x in cat["sets"] if x["group"] == group]:
    files = [f for f in s["files"] if (ROOT / f).exists()]
    with st.container(border=True):
        head, side = st.columns([5, 2], vertical_alignment="top")
        head.markdown(f"#### {pick(s['name'])}")
        lic = pick(s["license"]) if isinstance(s["license"], dict) else s["license"]
        side.markdown(
            f":blue-badge[{pick(s['count'])}]  :gray-badge[{lic}]",
        )
        st.markdown(pick(s["what"]))
        st.markdown(f"**{tr('用在', 'Used for')}**：{pick(s['used'])}")
        if s.get("use"):
            st.markdown(f"**{tr('在平台上怎麼用', 'On the platform')}**：{pick(s['use'])}")
        if s.get("why_not"):
            st.caption(":material/block: " + pick(s["why_not"]))
        foot = st.columns([2, 3], vertical_alignment="center")
        if s["files"] and files:
            paths = tuple(files)
            data = payload(paths, tuple((ROOT / f).stat().st_mtime for f in paths))
            name = Path(paths[0]).name if len(paths) == 1 else f"{s['id']}.zip"
            foot[0].download_button(
                tr(f"下載（{size_text(len(data))}）", f"Download ({size_text(len(data))})"),
                data,
                file_name=name,
                mime="application/json" if name.endswith(".json") else None,
                icon=":material/download:",
                key=f"ds_dl_{s['id']}",
                on_click="ignore",
            )
            foot[1].caption(
                ("、".join(Path(f).name for f in paths))
                if len(paths) > 1
                else tr("JSON 檔，可直接上傳回平台", "JSON, uploads back as it is")
                if name.endswith(".json")
                else ""
            )
        elif s["files"]:
            foot[0].caption(tr("這台伺服器上沒有這個檔案。", "This server does not have the file."))
        if s.get("source"):
            foot[1].markdown(f"[{tr('來源', 'Source')}]({s['source']})")
