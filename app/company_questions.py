"""Question browser of the company-tools test (Multi-step mode, llm_steps_page.py): every question
with what counts as correct, and how each model did on it in the batch runs (with what a model that
failed actually called). The answer keys in words and the grades come from the test's own script
(step_bench.company_answers).
"""

import streamlit as st
from common import kept
from i18n import tr

from local_llm import step_bench

PAGE_SIZE = 15
GROUP_LABEL = {
    "D": tr("查詢", "Look-ups"),
    "S": tr("單一動作", "One action"),
    "L": tr("多步驟（要先查再做）", "Several steps"),
    "P": tr("日期、時間、數字", "Dates, times, numbers"),
    "N": tr("不需要工具", "No tool needed"),
    "A": tr("資訊不足要反問", "Missing information: ask"),
}
EXPECT = {
    "act": tr(
        "動手做：下面任一種做法都算對；不能多做其他修改，也不能在其他系統做修改。",
        "Do it: any one of the ways below passes; no other changes, none in another system.",
    ),
    "ask": tr(
        "資訊不足：不能做任何修改，回覆要反問（先查詢再說明也可以）。",
        "Missing information: no change at all, and the reply asks back (looking things up "
        "first is fine).",
    ),
    "answer": tr(
        "不需要工具：直接回答，不能做任何修改（查詢可以）。",
        "No tool needed: answer directly, no change at all (look-ups are fine).",
    ),
}
WHY = {
    "wrong args": tr("工具對、參數錯", "Right tool, wrong arguments"),
    "wrong / missing tool": tr("沒呼叫到該用的工具", "Did not call the tool needed"),
    "other system": tr("在其他系統做了修改", "Changed something in another system"),
    "check failed": tr("時段或會議室不符合條件", "Time slot or room does not fit"),
    "called a tool": tr("不需要工具卻做了修改", "Made a change when none was needed"),
    "acted": tr("該反問卻直接動手", "Acted instead of asking"),
    "did not ask": tr("沒有反問", "Did not ask back"),
    "request error": tr("請求失敗", "Request failed"),
}


@st.cache_data(ttl=600, show_spinner=False)
def data() -> dict:
    return step_bench.company_answers()


def model_name(m: str) -> str:
    return step_bench.CLOUD[m][0] if m in step_bench.CLOUD else m


def why_text(why: str) -> str:
    if why.startswith("extra writes:"):
        return tr("多做了修改：", "Extra changes: ") + why.split(":", 1)[1].strip()
    return WHY.get(why, why)


def render() -> None:
    try:
        with st.spinner(tr("讀取題目與批次結果…", "Loading questions and batch results…")):
            d = data()
    except Exception as e:  # noqa: BLE001 - the script's own message is the useful part
        st.error(tr(f"讀不到題目：{e}", f"Could not load the questions: {e}"))
        return
    questions = d["questions"]
    st.caption(
        tr(
            f"全部 {len(questions)} 題：每題的要求、算對的做法，以及批次測試裡各模型有沒有答對。",
            f"All {len(questions)} questions: the request, what counts as correct, and which "
            "models passed it in the batch runs.",
        )
    )

    c1, c2, c3 = st.columns([2, 3, 1])
    groups = ["*"] + list(dict.fromkeys(q["group"] for q in questions))
    group = c1.selectbox(
        tr("類型", "Category"),
        groups,
        format_func=lambda g: (
            tr("全部類型", "All categories")
            if g == "*"
            else f"{GROUP_LABEL.get(g, g)}（{sum(q['group'] == g for q in questions)}）"
        ),
        **kept("company_q_group"),
    )
    find = c2.text_input(
        tr("搜尋題目文字或編號", "Search the question text or id"), **kept("company_q_find")
    ).strip()
    cond = c3.selectbox(
        tr("批次結果的條件", "Batch condition"), ["A", "B", "C"], **kept("company_q_cond")
    )
    runs = [r for r in d["runs"].values() if r["main"] and r["cond"] == cond]

    pool = [
        q
        for q in questions
        if (group == "*" or q["group"] == group)
        and (not find or find.lower() in (q["id"] + q["text"]).lower())
    ]
    if not pool:
        st.info(tr("沒有符合的題目。", "No question matches."))
        return
    pages = (len(pool) - 1) // PAGE_SIZE + 1
    page = 1
    if pages > 1:
        page = (
            st.segmented_control(
                tr(
                    f"共 {len(pool)} 題，每頁 {PAGE_SIZE} 題",
                    f"{len(pool)} questions, {PAGE_SIZE} a page",
                ),
                list(range(1, pages + 1)),
                default=1,
                key=f"company_q_page_{group}_{find}",
            )
            or 1
        )

    for q in pool[(page - 1) * PAGE_SIZE : page * PAGE_SIZE]:
        graded = [(r["model"], r["grades"][q["id"]]) for r in runs if q["id"] in r["grades"]]
        ok = sorted(m for m, g in graded if g["ok"])
        bad = sorted((m, g) for m, g in graded if not g["ok"])
        score = tr(
            f"{cond} {len(ok)}/{len(graded)} 個模型答對", f"{cond} {len(ok)}/{len(graded)} passed"
        )
        with st.expander(f"{q['id']}　{q['text']}　·　{score}"):
            question(q, ok, bad)


def question(q: dict, ok: list, bad: list) -> None:
    left, right = st.columns([3, 2])
    with left:
        st.markdown(f"**{tr('應有的行為', 'Expected')}**：{EXPECT[q['expect']]}")
        for n, alt in enumerate(q["alternatives"], 1):
            if len(q["alternatives"]) > 1:
                st.caption(tr(f"做法 {n}", f"Way {n}"))
            st.code("\n".join(alt), language=None, wrap_lines=True)
        if q["ask_ok"]:
            st.info(
                tr(
                    "這題題意有歧義，照做或先反問都算對。",
                    "Ambiguous: doing it or asking first both pass.",
                ),
                icon=":material/help:",
            )
        if q["check"]:
            st.caption(q["check"])
    with right:
        st.markdown(f"**{tr('批次結果', 'Batch results')}**")
        st.caption(
            tr("答對：", "Passed: ") + ("、".join(model_name(m) for m in ok) or tr("沒有", "none"))
        )
        if not bad:
            return
        st.dataframe(
            [{tr("沒答對", "Failed"): model_name(m), tr("原因", "Why"): why_text(g["why"])}
             for m, g in bad],
            hide_index=True,
            use_container_width=True,
        )  # fmt: skip
        pick = st.selectbox(
            tr("看這個模型實際做了什麼", "See what this model did"),
            [m for m, _ in bad],
            format_func=model_name,
            key=f"company_q_see_{q['id']}",
        )
        g = dict(bad)[pick]
        st.code(
            "\n".join(g.get("calls") or []) or tr("（沒有呼叫工具）", "(no call)"),
            wrap_lines=True,
        )
        if g.get("final"):
            st.caption(g["final"])
