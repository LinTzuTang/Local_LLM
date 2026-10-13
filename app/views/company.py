import html

import pandas as pd
import streamlit as st
import ui
from i18n import current, pick, tr

from local_llm import company_db

lang = current()
ui.page_header(
    "🗄️",
    tr("公司資料庫", "Company database"),
    tr(
        f"{company_db.COMPANY}的模擬 PostgreSQL（人名與公司皆為虛構）。用任何<b>資料庫身分</b>預覽資料、"
        "管理身分與權限、上傳自己的 CSV／Excel。權限由 PostgreSQL 本身強制，不是靠 prompt。",
        f"Simulated PostgreSQL of {company_db.COMPANY} (all people and companies are fictional). "
        "Preview data as any <b>database identity</b>, manage identities and their permissions, "
        "and upload your own CSV / Excel. PostgreSQL itself enforces the permissions, not the "
        "prompt.",
    ),
    eyebrow="PostgreSQL · democo",
)

if error := company_db.status():
    st.error(tr(f"連不到資料庫：{error}", f"Cannot reach the database: {error}"))
    st.markdown(
        tr(
            "在 terminal 執行 `./scripts/start_db.sh`（第一次會自動建立並載入模擬資料）。",
            "Run `./scripts/start_db.sh` in a terminal (the first run creates the database and "
            "loads the sample data).",
        )
    )
    st.stop()

try:  # first visit: create the example identities (finance, management, intern)
    if created := company_db.ensure_seed_identities():
        st.toast(tr("已建立範例身分：", "Created example identities: ") + ", ".join(created))
except Exception as exc:  # noqa: BLE001 - the page still works with the built-in identities
    st.warning(tr(f"無法建立範例身分：{exc}", f"Could not create the example identities: {exc}"))

idents = company_db.identities()
ROLES = [i["role"] for i in idents]
IDENT = {i["role"]: i for i in idents}
LABELS = {r: pick(IDENT[r]["label"]) for r in ROLES}
tables = company_db.admin_tables()
names = [t["table"] for t in tables]
info = {t["table"]: t for t in tables}
access = company_db.access_matrix(ROLES)

# Access modes in the grant editor (stable ids; labels follow the language).
MODES = {
    "none": tr("⛔ 不可讀", "⛔ No access"),
    "all": tr("✅ 全部欄位", "✅ All columns"),
    "some": tr("🟡 部分欄位", "🟡 Some columns"),
}


def readable(role: str, table: str) -> list[str]:
    return access[role].get(table, ([], 0))[0]


def mode_of(role: str, table: str) -> str:
    columns, total = access[role].get(table, ([], len(info[table]["columns"])))
    return "none" if not columns else "all" if len(columns) == total else "some"


def cell(role: str, table: str) -> str:
    columns, total = access[role].get(table, ([], len(info[table]["columns"])))
    mode = mode_of(role, table)
    if mode == "none":
        return "⛔"
    return "✅ " + tr("全部", "all") if mode == "all" else f"🟡 {len(columns)}/{total}"


def ident_label(role: str) -> str:
    return f"{LABELS[role]} · {role}"


def column_chips(visible: list[str], columns: list[str]) -> str:
    """Every column of a table as a pill: green when visible, struck-through when hidden."""
    parts = []
    for col in columns:
        name = html.escape(col)
        if col in visible:
            parts.append(
                f'<span class="lab-pill" style="background:#ECFDF5;color:#047857">✓ {name}</span>'
            )
        else:
            parts.append(
                '<span class="lab-pill" style="background:#F8FAFC;color:#94A3B8;'
                f'text-decoration:line-through">{name}</span>'
            )
    return f'<div class="lab-badges">{"".join(parts)}</div>'


def matrix_style(value: str) -> str:
    if value.startswith("✅"):
        return "background-color: #ECFDF5; color: #047857"
    if value.startswith("🟡"):
        return "background-color: #FFFBEB; color: #B45309"
    if value.startswith("⛔"):
        return "background-color: #F8FAFC; color: #94A3B8"
    return ""


preview_tab, ident_tab, upload_tab = st.tabs(
    [
        tr("👀 以身分預覽", "👀 Preview as identity"),
        tr("🪪 身分與權限", "🪪 Identities & permissions"),
        tr("⬆️ 上傳資料", "⬆️ Upload data"),
    ]
)

with preview_tab:
    c1, c2 = st.columns(2)
    table = c1.selectbox(
        tr("資料表", "Table"),
        names,
        key="preview_table",
        format_func=lambda t: f"{t}　{info[t]['description'] or ''}",
    )
    viewer = c2.selectbox(
        tr("以誰的身分看", "View as"),
        ["admin", *ROLES],
        format_func=lambda r: (
            tr("管理員（superuser，全部欄位）", "Admin (superuser, all columns)")
            if r == "admin"
            else ident_label(r)
        ),
        key="preview_viewer",
    )
    columns = info[table]["columns"]
    visible = columns if viewer == "admin" else readable(viewer, table)
    hidden = [c for c in columns if c not in visible]
    who = tr("管理員", "Admin") if viewer == "admin" else LABELS[viewer]

    with st.container(border=True):
        top_left, top_right = st.columns([3, 2], vertical_alignment="center")
        top_left.markdown(
            f"**{table}**　{info[table]['description'] or ''}  \n"
            + tr(
                f"{info[table]['rows']:,} 筆 · {len(columns)} 個欄位",
                f"{info[table]['rows']:,} rows · {len(columns)} columns",
            )
        )
        if not visible:
            top_right.markdown(f"### ⛔ {tr('不可讀', 'No access')}")
        else:
            top_right.markdown(
                f"### 👁️ {len(visible)} / {len(columns)} " + tr("欄看得到", "columns visible")
            )
        st.markdown(column_chips(visible, columns), unsafe_allow_html=True)
        st.caption(
            tr(
                "綠色＝這個身分看得到的欄位；灰色刪除線＝隱藏欄位（查詢會被資料庫拒絕）。",
                "Green = columns this identity can read; grey struck-through = hidden columns "
                "(the database refuses queries on them).",
            )
        )

    if not visible:
        # No st.stop() inside a tab: it would also stop the other tabs from rendering.
        st.warning(
            tr(
                f"{who} 不能讀這張表；agent 查詢時會收到 permission denied。",
                f"{who} cannot read this table; an agent querying it gets “permission denied”.",
            )
        )
    else:
        try:
            if viewer == "admin":
                cols, rows = company_db.admin_preview(table)
            else:
                # Connects as that PostgreSQL role: exactly what its agents would see.
                cols, rows = company_db.preview(viewer, table, visible, limit=50)
        except Exception as exc:  # noqa: BLE001
            st.error(f"{type(exc).__name__}: {exc}")
        else:
            st.dataframe(pd.DataFrame(rows, columns=cols), hide_index=True, width="stretch")
            st.caption(
                tr(
                    "只顯示前 50 筆。非管理員的預覽是用該身分自己的資料庫帳號查詢，畫面就是它的 agent 實際看得到的資料。",
                    "First 50 rows only. A non-admin preview logs in with that identity's own "
                    "database account, so this is exactly what its agents can see.",
                )
            )

    st.markdown(f"##### {tr('這張表各身分的權限', 'Who can read this table')}")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    tr("身分", "Identity"): ident_label(r),
                    tr("權限", "Access"): MODES[mode_of(r, table)],
                    tr("看得到", "Visible"): ", ".join(readable(r, table)) or "—",
                    tr("隱藏", "Hidden"): ", ".join(
                        c for c in columns if c not in readable(r, table)
                    )
                    or "—",
                }
                for r in ROLES
            ]
        ).style.map(matrix_style, subset=[tr("權限", "Access")]),
        hide_index=True,
        width="stretch",
    )

with ident_tab:
    st.markdown(
        tr(
            "每個**資料庫身分**都是一個 PostgreSQL 登入角色：只能 `SELECT`、預設唯讀交易、單一查詢 5 秒逾時、"
            "最多 10 條連線。Agent 定義裡的 `db_role` 決定它用哪個身分連線。三個部門身分是內建的；"
            "其他身分可以在這裡新增、調整權限或刪除（存在 `storage/lab/identities.json`，密碼存在 `.env`）。",
            "Each **database identity** is a PostgreSQL login role: `SELECT` only, read-only "
            "transactions by default, a 5 s statement timeout and at most 10 connections. An "
            "agent's `db_role` picks the identity it connects with. The three department "
            "identities are built in; create, re-grant or delete others here (kept in "
            "`storage/lab/identities.json`, passwords in `.env`).",
        )
    )

    # Identity cards
    for row in range(0, len(ROLES), 3):
        for col, role in zip(st.columns(3, gap="small"), ROLES[row : row + 3], strict=False):
            ident = IDENT[role]
            n_tables = sum(1 for t in names if readable(role, t))
            users = company_db.agents_using(role)
            with col.container(border=True):
                st.markdown(f"**{LABELS[role]}**")
                st.markdown(
                    ui.badges(
                        [],
                        [
                            role,
                            tr("內建", "built-in") if ident["builtin"] else tr("自訂", "custom"),
                        ],
                    ),
                    unsafe_allow_html=True,
                )
                st.caption(pick(ident["description"]) or "—")
                st.markdown(
                    tr(
                        f"可讀 **{n_tables}** / {len(names)} 張表",
                        f"Reads **{n_tables}** / {len(names)} tables",
                    )
                    + "  \n"
                    + tr("使用的 agent：", "Used by agents: ")
                    + (", ".join(f"`{u}`" for u in users) or "—")
                )

    st.markdown(f"#### {tr('權限矩陣', 'Permission matrix')}")
    st.caption(
        tr(
            "列＝資料表，欄＝身分。✅ 全部欄位　🟡 部分欄位（可讀/全部）　⛔ 不可讀。資料直接取自 PostgreSQL 的權限檢查。",
            "Rows = tables, columns = identities. ✅ all columns · 🟡 some columns "
            "(readable/total) · ⛔ no access. Read straight from PostgreSQL's privilege checks.",
        )
    )
    matrix = pd.DataFrame(
        [
            {
                tr("資料表", "Table"): t,
                tr("說明", "Description"): info[t]["description"] or "",
                **{LABELS[r]: cell(r, t) for r in ROLES},
            }
            for t in names
        ]
    )
    st.dataframe(
        matrix.style.map(matrix_style, subset=[LABELS[r] for r in ROLES]),
        hide_index=True,
        width="stretch",
        height=(len(names) + 1) * 35 + 3,
    )

    st.markdown(f"#### {tr('調整權限', 'Edit permissions')}")
    with st.container(border=True):
        e1, e2 = st.columns(2)
        if (new_role := st.session_state.pop("_next_grant_role", None)) in ROLES:
            st.session_state.grant_role = new_role  # just created: grant it access next
        role = e1.selectbox(
            tr("身分", "Identity"), ROLES, format_func=ident_label, key="grant_role"
        )
        # After an upload, open the new table here (set before the widget exists).
        if (new_table := st.session_state.pop("_next_perm_table", None)) in names:
            st.session_state.perm_table = new_table
        if st.session_state.get("perm_table") not in names:
            st.session_state.pop("perm_table", None)
        table = e2.selectbox(
            tr("資料表", "Table"),
            names,
            key="perm_table",
            format_func=lambda t: f"{t}　{info[t]['description'] or ''}",
        )
        columns = info[table]["columns"]
        current_cols = readable(role, table)
        mode = st.radio(
            tr("權限", "Access"),
            list(MODES),
            index=list(MODES).index(mode_of(role, table)),
            format_func=MODES.get,
            key=f"mode_{role}_{table}",
            horizontal=True,
        )
        picked = (
            st.multiselect(
                tr("可讀欄位", "Readable columns"),
                columns,
                default=current_cols or columns,
                key=f"cols_{role}_{table}",
            )
            if mode == "some"
            else None
        )
        if IDENT[role]["builtin"]:
            st.caption(
                tr(
                    "內建部門身分的調整會立即生效；重新執行 `company_db init` 會回到預設權限。",
                    "Changes to a built-in identity apply immediately; re-running "
                    "`company_db init` restores its defaults.",
                )
            )
        empty = mode == "some" and not picked
        if empty:
            st.warning(
                tr(
                    "選了「部分欄位」但沒有勾任何欄位；請至少選一欄，或改成「不可讀」。",
                    "“Some columns” is selected but no column is ticked; pick at least one, or "
                    "switch to “No access”.",
                )
            )
        if st.button(
            tr("套用權限", "Apply permissions"), type="primary", disabled=empty, key="apply_grant"
        ):
            # set_access: None = no access, [] = whole table, list = only those columns
            cols = None if mode == "none" else [] if mode == "all" else picked
            if cols and set(cols) == set(columns):
                cols = []  # every column picked = whole table
            try:
                company_db.set_access(table, role, cols)
            except Exception as exc:  # noqa: BLE001
                st.error(f"{type(exc).__name__}: {exc}")
            else:
                st.toast(
                    tr(
                        f"{LABELS[role]} 對 {table} 的權限已更新。",
                        f"Updated {LABELS[role]}'s access to {table}.",
                    )
                )
                st.rerun()
        if table.startswith("uploads.") and st.button(
            tr("刪除這張上傳的資料表", "Delete this uploaded table"), key="drop_upload"
        ):
            company_db.drop_table(table)
            st.rerun()

    new_col, del_col = st.columns(2, gap="medium")
    with new_col.container(border=True):
        st.markdown(f"#### ➕ {tr('新增身分', 'New identity')}")
        with st.form("new_identity", clear_on_submit=False, border=False):
            suffix = st.text_input(
                tr("身分代號（agent_ 後面的部分）", "Identity id (after agent_)"),
                placeholder="qa",
                key="new_suffix",
                help=tr(
                    "小寫英文、數字、底線；會建立 PostgreSQL 角色 agent_〈代號〉",
                    "Lowercase letters, digits, _; creates the PostgreSQL role agent_<id>",
                ),
            )
            n1, n2 = st.columns(2)
            label_zh = n1.text_input(
                tr("名稱（中文）", "Label (Chinese)"), placeholder="品保部", key="new_label_zh"
            )
            label_en = n2.text_input(
                tr("名稱（英文）", "Label (English)"), placeholder="Quality", key="new_label_en"
            )
            desc_zh = st.text_input(
                tr("說明（中文）", "Description (Chinese)"),
                placeholder="產品、工單與採購，不含人資",
                key="new_desc_zh",
            )
            desc_en = st.text_input(
                tr("說明（英文）", "Description (English)"),
                placeholder="Products, tickets and purchasing; no HR",
                key="new_desc_en",
            )
            template = st.selectbox(
                tr("起始權限", "Start with"),
                ["", *ROLES],
                format_func=lambda r: (
                    tr("沒有任何權限（之後再開放）", "No access (grant later)")
                    if not r
                    else tr(f"複製「{LABELS[r]}」的權限", f"Copy {LABELS[r]}'s access")
                ),
            )
            submitted = st.form_submit_button(tr("建立身分", "Create identity"), type="primary")
        if submitted:
            grants = {}
            if template:
                for t in names:
                    cols, total = access[template].get(t, ([], 0))
                    if cols:
                        grants[t] = [] if len(cols) == total else cols
            try:
                record = company_db.create_identity(
                    "agent_" + suffix.strip().lower().removeprefix("agent_"),
                    {"zh": label_zh.strip(), "en": label_en.strip()},
                    {"zh": desc_zh.strip(), "en": desc_en.strip()},
                    grants,
                )
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))
            else:
                st.toast(tr(f"已建立身分 {record['role']}", f"Created identity {record['role']}"))
                for k in (
                    "new_suffix",
                    "new_label_zh",
                    "new_label_en",
                    "new_desc_zh",
                    "new_desc_en",
                ):
                    st.session_state.pop(k, None)  # start the next identity from an empty form
                st.session_state["_next_grant_role"] = record["role"]
                st.rerun()

    with del_col.container(border=True):
        st.markdown(f"#### 🗑️ {tr('刪除身分', 'Delete identity')}")
        custom = [r for r in ROLES if not IDENT[r]["builtin"]]
        if not custom:
            st.caption(tr("目前沒有自訂身分。", "There are no custom identities."))
        else:
            target = st.selectbox(
                tr("自訂身分", "Custom identity"), custom, format_func=ident_label, key="del_role"
            )
            users = company_db.agents_using(target)
            if users:
                st.warning(
                    tr(
                        f"還有 agent 使用這個身分（{', '.join(users)}），請先在 Agent 工作室改掉它們的資料庫身分。",
                        f"Agents still use this identity ({', '.join(users)}); change their "
                        "database identity in the Agent studio first.",
                    )
                )
            st.caption(
                tr(
                    "刪除會移除 PostgreSQL 角色、它的所有權限與 .env 裡的密碼。內建部門身分不能刪除。",
                    "Deleting removes the PostgreSQL role, all its privileges and its password "
                    "in .env. Built-in department identities cannot be deleted.",
                )
            )
            sure = st.checkbox(
                tr(f"確定刪除 {target}", f"Yes, delete {target}"), key=f"sure_{target}"
            )
            if st.button(
                tr("刪除", "Delete"), disabled=not sure or bool(users), key="delete_identity"
            ):
                try:
                    company_db.delete_identity(target)
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))
                else:
                    st.toast(tr(f"已刪除 {target}", f"Deleted {target}"))
                    st.rerun()

with upload_tab:
    st.markdown(
        tr(
            "上傳 CSV 或 Excel，會建立成資料庫裡的新資料表。**上傳後預設沒有任何身分能讀**，"
            "要到「🪪 身分與權限」開放。",
            "Upload a CSV or Excel file to create a new table in the database. **By default no "
            "identity can read an uploaded table**; grant access under “🪪 Identities & "
            "permissions”.",
        )
    )
    file = st.file_uploader(tr("選擇檔案", "Choose a file"), type=["csv", "xlsx"])
    df = None
    if file:
        try:
            df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
        except Exception as exc:
            st.error(tr(f"讀不了這個檔案：{exc}", f"Cannot read this file: {exc}"))
    if file and df is not None:
        st.markdown(
            tr(
                f"**預覽**：{len(df)} 筆、{len(df.columns)} 個欄位",
                f"**Preview**: {len(df)} rows, {len(df.columns)} columns",
            )
        )
        st.dataframe(df.head(10), hide_index=True, width="stretch")
        default_name = "".join(
            ch if ch.isascii() and (ch.isalnum() or ch == "_") else "_"
            for ch in file.name.rsplit(".", 1)[0].lower()
        ).strip("_")
        c1, c2 = st.columns([1, 2])
        schema = c1.selectbox(
            tr("放在哪個 schema", "Schema"), ["uploads", "hr", "sales", "inventory", "support"]
        )
        table_name = c2.text_input(
            tr("資料表名稱（小寫英文、數字、底線）", "Table name (lowercase letters, digits, _)"),
            default_name or "my_table",
        )
        description = st.text_input(
            tr(
                "資料表說明（agent 會看到，寫清楚有助於它查詢）",
                "Table description (the agent sees it; a clear one helps it query)",
            ),
            placeholder=tr(
                "例如：2026 年各專案預算與負責人", "e.g. 2026 budget and owner of each project"
            ),
        )
        exists = f"{schema}.{table_name}" in info
        replace = (
            st.checkbox(tr("已存在則覆蓋", "Replace if it exists"), value=False)
            if exists
            else False
        )
        if exists and not replace:
            st.warning(
                tr(
                    f"`{schema}.{table_name}` 已存在；勾選「覆蓋」或換個名稱。",
                    f"`{schema}.{table_name}` already exists; tick “Replace” or pick another name.",
                )
            )
        if st.button(
            tr("匯入資料庫", "Import into the database"),
            type="primary",
            disabled=exists and not replace,
        ):
            try:
                n = company_db.import_dataframe(df, schema, table_name, description, replace)
            except Exception as exc:
                st.error(tr(f"匯入失敗：{exc}", f"Import failed: {exc}"))
            else:
                st.session_state["_next_perm_table"] = f"{schema}.{table_name}"  # grant next
                # A toast survives the rerun that refreshes the table list.
                st.toast(
                    tr(
                        f"已匯入 {n} 筆到 {schema}.{table_name}；到「🪪 身分與權限」開放給需要的身分。",
                        f"Imported {n} rows into {schema}.{table_name}; grant access to the "
                        "identities that need it under “🪪 Identities & permissions”.",
                    )
                )
                st.rerun()
