"""Simulated company database: a fictional Taiwanese precision-parts maker on PostgreSQL.

`python -m local_llm.company_db init` creates the schema, the access roles and synthetic
data. Agents connect as a per-department role, so PostgreSQL itself enforces what each
agent may read (e.g. the sales agent cannot see salaries) and that nothing is written.
"""

import json
import os
import random
import re
import secrets
import sys
import threading
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import psycopg
from dotenv import load_dotenv, set_key, unset_key
from faker import Faker
from langchain.tools import tool
from psycopg import sql

PROJECT_DIR = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_DIR / ".env"
DB_NAME = "democo"
COMPANY = "示範精密工業股份有限公司（虛構）"
MAX_ROWS = 100
# Schemas holding company data; 'uploads' receives tables imported from the platform.
SCHEMAS = ("hr", "sales", "inventory", "support", "uploads")

# Department roles: what each agent identity may read. Column lists limit sensitive tables.
ROLES = {
    "agent_sales": {
        "label": "業務部",
        "schemas": ["sales", "inventory", "support"],
        "columns": {
            "hr.departments": ["id", "name"],
            "hr.employees": ["id", "emp_no", "name", "dept_id", "title", "email", "status"],
        },
    },
    "agent_hr": {
        "label": "人資部",
        "schemas": ["hr"],
        "columns": {},
    },
    "agent_support": {
        "label": "客服部",
        "schemas": ["support"],
        "columns": {
            "sales.customers": ["id", "name", "industry", "city", "tier"],
            "sales.products": ["id", "sku", "name", "category", "unit_price"],
            "inventory.warehouses": ["id", "name", "city"],
            "inventory.stock": ["product_id", "warehouse_id", "qty_on_hand"],
            "hr.employees": ["id", "name", "title", "email"],
        },
    },
}

ROLE_TEXT = {  # English labels and both descriptions for the UI; Chinese labels live in ROLES
    "agent_sales": (
        "Sales",
        "業務、庫存與客服資料；員工只看基本資料，看不到薪資與電話",
        "Sales, inventory and support data; basic staff info, no salaries or phone numbers",
    ),
    "agent_hr": (
        "HR",
        "全部人資資料（含薪資）；看不到業務、庫存與客服",
        "All HR data including salaries; no sales, inventory or support",
    ),
    "agent_support": (
        "Support",
        "客服工單；客戶、產品、庫存與員工只看部分欄位",
        "Support tickets; limited customer, product, stock and staff columns",
    ),
}

# Extra identities created on first use (stored in storage/lab/identities.json, editable on
# the Company database page). grants: {table: None = no access, [] = all columns, [cols]}.
SEED_IDENTITIES = [
    {
        "role": "agent_finance",
        "label": {"zh": "財務部", "en": "Finance"},
        "description": {
            "zh": "訂單、金額、產品成本、採購與庫存；客戶含信用額度；看不到任何人資資料",
            "en": "Orders, amounts, product cost, purchasing and stock; customers incl. credit "
            "limits; no HR data",
        },
        "grants": {
            "sales.orders": [],
            "sales.order_items": [],
            "sales.products": [],
            "sales.customers": ["id", "name", "industry", "city", "tier", "credit_limit"],
            "inventory.purchase_orders": [],
            "inventory.stock": [],
            "inventory.warehouses": [],
        },
    },
    {
        "role": "agent_management",
        "label": {"zh": "管理層", "en": "Management"},
        "description": {
            "zh": "全公司資料，但看不到員工薪資與電話",
            "en": "All company data except staff salaries and phone numbers",
        },
        "grants": {
            "hr.departments": [],
            "hr.employees": [
                "id",
                "emp_no",
                "name",
                "dept_id",
                "title",
                "hire_date",
                "email",
                "status",
            ],  # fmt: skip
            "hr.leave_requests": [],
            "sales.products": [],
            "sales.customers": [],
            "sales.orders": [],
            "sales.order_items": [],
            "inventory.warehouses": [],
            "inventory.stock": [],
            "inventory.purchase_orders": [],
            "support.tickets": [],
        },
    },
    {
        "role": "agent_intern",
        "label": {"zh": "實習生", "en": "Intern"},
        "description": {
            "zh": "只看產品型錄（不含成本）與倉庫",
            "en": "Product catalogue (without cost) and warehouses only",
        },
        "grants": {
            "sales.products": ["id", "sku", "name", "category", "unit_price"],
            "inventory.warehouses": [],
        },
    },
]


def identities() -> list[dict]:
    """Database identities an agent (or a preview) can use: one PostgreSQL role each.

    The three department roles are built in; more are created on the Company database page
    and kept in storage/lab/identities.json (their passwords are in .env like the others).
    """
    builtin = [
        {
            "role": role,
            "label": {"zh": rules["label"], "en": ROLE_TEXT.get(role, (role, "", ""))[0]},
            "description": {
                "zh": ROLE_TEXT.get(role, ("", "", ""))[1],
                "en": ROLE_TEXT.get(role, ("", "", ""))[2],
            },
            "builtin": True,
        }
        for role, rules in ROLES.items()
    ]
    custom = [
        {
            "role": i["role"],
            "label": i["label"],
            "description": i.get("description") or {"zh": "", "en": ""},
            "builtin": False,
        }
        for i in _custom_identities()
    ]
    return builtin + custom


def identity_roles() -> list[str]:
    return [i["role"] for i in identities()]


SCHEMA_SQL = """
CREATE SCHEMA hr; CREATE SCHEMA sales; CREATE SCHEMA inventory; CREATE SCHEMA support;

CREATE TABLE hr.departments (id int PRIMARY KEY, name text NOT NULL, manager_id int);
COMMENT ON TABLE hr.departments IS '部門';
COMMENT ON COLUMN hr.departments.manager_id IS '部門主管，對應 hr.employees.id';

CREATE TABLE hr.employees (
  id int PRIMARY KEY, emp_no text UNIQUE NOT NULL, name text NOT NULL,
  dept_id int REFERENCES hr.departments(id), title text, hire_date date,
  email text, phone text, salary int, status text);
COMMENT ON TABLE hr.employees IS '員工（含離職）';
COMMENT ON COLUMN hr.employees.emp_no IS '員工編號';
COMMENT ON COLUMN hr.employees.title IS '職稱';
COMMENT ON COLUMN hr.employees.hire_date IS '到職日';
COMMENT ON COLUMN hr.employees.salary IS '月薪（新台幣，機密）';
COMMENT ON COLUMN hr.employees.status IS '在職 或 離職';

CREATE TABLE hr.leave_requests (
  id int PRIMARY KEY, employee_id int REFERENCES hr.employees(id), leave_type text,
  start_date date, end_date date, days numeric(4,1), status text);
COMMENT ON TABLE hr.leave_requests IS '請假紀錄';
COMMENT ON COLUMN hr.leave_requests.leave_type IS '特休、病假、事假、婚假、喪假、公假';
COMMENT ON COLUMN hr.leave_requests.status IS '核准、待審、駁回';

CREATE TABLE sales.products (
  id int PRIMARY KEY, sku text UNIQUE, name text, category text,
  unit_price int, unit_cost int);
COMMENT ON TABLE sales.products IS '產品';
COMMENT ON COLUMN sales.products.unit_price IS '牌價（新台幣）';
COMMENT ON COLUMN sales.products.unit_cost IS '成本（新台幣）';

CREATE TABLE sales.customers (
  id int PRIMARY KEY, name text, industry text, city text, tier text,
  credit_limit int, owner_id int REFERENCES hr.employees(id));
COMMENT ON TABLE sales.customers IS '客戶（虛構公司）';
COMMENT ON COLUMN sales.customers.tier IS '客戶等級 A/B/C';
COMMENT ON COLUMN sales.customers.credit_limit IS '信用額度（新台幣）';
COMMENT ON COLUMN sales.customers.owner_id IS '負責業務，對應 hr.employees.id';

CREATE TABLE sales.orders (
  id int PRIMARY KEY, order_no text UNIQUE, customer_id int REFERENCES sales.customers(id),
  order_date date, status text, sales_rep_id int REFERENCES hr.employees(id));
COMMENT ON TABLE sales.orders IS '訂單';
COMMENT ON COLUMN sales.orders.status IS '處理中、已出貨、已完成、已取消';

CREATE TABLE sales.order_items (
  order_id int REFERENCES sales.orders(id), product_id int REFERENCES sales.products(id),
  quantity int, unit_price int, PRIMARY KEY (order_id, product_id));
COMMENT ON TABLE sales.order_items IS '訂單明細；金額 = quantity * unit_price（成交價）';

CREATE TABLE inventory.warehouses (id int PRIMARY KEY, name text, city text);
COMMENT ON TABLE inventory.warehouses IS '倉庫';

CREATE TABLE inventory.stock (
  product_id int REFERENCES sales.products(id), warehouse_id int REFERENCES inventory.warehouses(id),
  qty_on_hand int, reorder_point int, PRIMARY KEY (product_id, warehouse_id));
COMMENT ON TABLE inventory.stock IS '庫存；qty_on_hand 低於 reorder_point 需補貨';

CREATE TABLE inventory.purchase_orders (
  id int PRIMARY KEY, po_no text UNIQUE, supplier text, product_id int REFERENCES sales.products(id),
  quantity int, order_date date, expected_date date, status text);
COMMENT ON TABLE inventory.purchase_orders IS '採購單';
COMMENT ON COLUMN inventory.purchase_orders.status IS '已下單、運送中、已入庫';

CREATE TABLE support.tickets (
  id int PRIMARY KEY, ticket_no text UNIQUE, customer_id int REFERENCES sales.customers(id),
  product_id int REFERENCES sales.products(id), created_at timestamp, channel text,
  category text, priority text, status text, subject text, description text,
  assignee_id int REFERENCES hr.employees(id), resolved_at timestamp);
COMMENT ON TABLE support.tickets IS '客服工單';
COMMENT ON COLUMN support.tickets.channel IS '電話、Email、LINE、業務轉介';
COMMENT ON COLUMN support.tickets.category IS '品質異常、交期查詢、安裝問題、退換貨、技術諮詢、報價';
COMMENT ON COLUMN support.tickets.priority IS '低、中、高、緊急';
COMMENT ON COLUMN support.tickets.status IS '待處理、處理中、已解決、已結案';
"""

DEPARTMENTS = [
    ("總經理室", 4, ["總經理", "特助", "秘書"]),
    ("業務部", 40, ["業務經理", "資深業務", "業務專員", "業務助理"]),
    ("研發部", 45, ["研發經理", "資深工程師", "機構工程師", "韌體工程師"]),
    ("生產部", 90, ["廠長", "生產組長", "技術員", "作業員"]),
    ("品保部", 20, ["品保經理", "品保工程師", "檢驗員"]),
    ("採購部", 12, ["採購經理", "採購專員"]),
    ("倉儲物流部", 25, ["倉儲主管", "倉管員", "物流專員"]),
    ("客服部", 18, ["客服主管", "技術客服", "客服專員"]),
    ("人資部", 8, ["人資經理", "人資專員"]),
    ("財務部", 10, ["財務經理", "會計", "出納"]),
    ("資訊部", 12, ["資訊經理", "系統工程師", "網管"]),
]
SALARY = {
    "總經理": 280000, "經理": 120000, "廠長": 130000, "主管": 85000, "組長": 62000,
    "資深": 75000, "工程師": 62000, "特助": 70000, "專員": 45000, "業務": 48000,
    "技術員": 40000, "作業員": 33000, "檢驗員": 36000, "助理": 36000, "會計": 46000,
    "出納": 40000, "倉管員": 36000, "秘書": 42000, "網管": 52000,
}  # fmt: skip
PRODUCT_LINES = [
    ("線性滑軌", "LG", [12, 15, 20, 25, 30], 2800),
    ("滾珠螺桿", "BS", [16, 20, 25, 32], 4200),
    ("精密軸承", "BR", [6000, 6202, 7205, 7308], 650),
    ("伺服馬達", "SM", [200, 400, 750, 1000, 2000], 18500),
    ("減速機", "RG", [60, 90, 120], 12000),
    ("光電感測器", "PS", [10, 30, 50], 1500),
    ("運動控制器", "MC", [4, 8, 16], 26000),
]
CITIES = ["台北市", "新北市", "桃園市", "新竹市", "新竹縣", "台中市", "彰化縣", "台南市", "高雄市"]
INDUSTRIES = [
    "工具機",
    "半導體設備",
    "自動化設備",
    "汽車零組件",
    "醫療器材",
    "電子組裝",
    "包裝機械",
]
NAME_HEADS = list("宏昇鼎聖華永泰勝冠瑞達鴻信益全佳凱群晟璟")
NAME_TAILS = ["精機", "科技", "機械", "工業", "自動化", "光電", "實業", "精密"]
SUPPLIERS = ["德芮金屬（虛構）", "穎昌鑄造（虛構）", "日鋒電機（虛構）", "力群電子（虛構）"]
TICKET_TEMPLATES = {
    "品質異常": ("{p} 出貨後異音", "客戶反映 {p} 安裝運轉約兩週後出現異音，請協助判斷是否為品質問題。"),
    "交期查詢": ("{o} 交期確認", "客戶詢問訂單 {o} 的 {p} 預計出貨日，產線排程需要確認。"),
    "安裝問題": ("{p} 安裝對位困難", "客戶依手冊安裝 {p} 時對位困難，詢問是否有安裝治具或影片。"),
    "退換貨": ("{p} 規格下錯申請換貨", "客戶下單時規格選錯，希望將 {p} 換成其他尺寸。"),
    "技術諮詢": ("{p} 負載與壽命計算", "客戶想確認 {p} 在其設備負載條件下的預估壽命。"),
    "報價": ("{p} 大量採購報價", "客戶明年度預計大量採購 {p}，希望取得階梯報價。"),
}  # fmt: skip


def conninfo(user: str, password: str, dbname: str = DB_NAME) -> str:
    host = os.getenv("LOCAL_LLM_PG_HOST", "127.0.0.1")
    port = os.getenv("LOCAL_LLM_PG_PORT", "5433")
    return psycopg.conninfo.make_conninfo(
        host=host, port=port, dbname=dbname, user=user, password=password, connect_timeout=5
    )


def _password(name: str) -> str:
    load_dotenv(ENV_FILE)
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} missing in .env; run ./scripts/start_db.sh first")
    return value


def init(rng_seed: int = 7) -> None:
    load_dotenv(ENV_FILE)
    superpass = _password("LOCAL_LLM_PG_SUPERUSER_PASSWORD")
    with psycopg.connect(conninfo("postgres", superpass, "postgres"), autocommit=True) as c:
        c.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(DB_NAME)))
        c.execute(sql.SQL("CREATE DATABASE {} ENCODING 'UTF8'").format(sql.Identifier(DB_NAME)))
        for role in ROLES:
            env = f"LOCAL_LLM_PG_{role.upper()}_PASSWORD"
            password = os.getenv(env) or secrets.token_hex(16)
            set_key(str(ENV_FILE), env, password)
            exists = c.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", [role]).fetchone()
            verb = "ALTER" if exists else "CREATE"
            c.execute(
                sql.SQL("{} ROLE {} LOGIN PASSWORD {} CONNECTION LIMIT 10").format(
                    sql.SQL(verb), sql.Identifier(role), sql.Literal(password)
                )
            )
            # Read-only and time-limited regardless of the SQL an agent sends.
            c.execute(
                sql.SQL("ALTER ROLE {} SET default_transaction_read_only = on").format(
                    sql.Identifier(role)
                )
            )
            c.execute(
                sql.SQL("ALTER ROLE {} SET statement_timeout = '5s'").format(sql.Identifier(role))
            )
    with psycopg.connect(conninfo("postgres", superpass)) as db:
        db.execute(SCHEMA_SQL)
        _load_data(db, random.Random(rng_seed))
        db.execute(
            sql.SQL("COMMENT ON DATABASE {} IS {}").format(
                sql.Identifier(DB_NAME), sql.Literal(f"{COMPANY} 模擬資料，所有人名與公司皆為虛構")
            )
        )
        db.execute("REVOKE ALL ON SCHEMA public FROM PUBLIC")
        for role, rules in ROLES.items():
            ident = sql.Identifier(role)
            db.execute(
                sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(sql.Identifier(DB_NAME), ident)
            )
            for schema in rules["schemas"]:
                s = sql.Identifier(schema)
                db.execute(sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(s, ident))
                db.execute(
                    sql.SQL("GRANT SELECT ON ALL TABLES IN SCHEMA {} TO {}").format(s, ident)
                )
            for table, columns in rules["columns"].items():
                schema, name = table.split(".")
                db.execute(
                    sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(sql.Identifier(schema), ident)
                )
                db.execute(
                    sql.SQL("GRANT SELECT ({}) ON {} TO {}").format(
                        sql.SQL(", ").join(map(sql.Identifier, columns)),
                        sql.Identifier(schema, name),
                        ident,
                    )
                )
    # Extra identities: their roles survive in the cluster, their grants in identities.json.
    for ident in _custom_identities():
        _restore_identity(ident)
    ensure_seed_identities()
    print(f"Loaded {COMPANY} into database {DB_NAME!r}; identities: {', '.join(identity_roles())}")


def _load_data(db, rng: random.Random) -> None:
    fake = Faker("zh_TW")
    fake.seed_instance(rng.randrange(10**6))
    today = date(2026, 10, 12)

    employees, departments, by_dept = [], [], {}
    for dept_id, (dept, size, titles) in enumerate(DEPARTMENTS, start=1):
        for i in range(size):
            title = titles[0] if i == 0 else rng.choice(titles[1:] or titles)
            emp_id = len(employees) + 1
            base = next((v for k, v in SALARY.items() if k in title), 45000)
            employees.append(
                (
                    emp_id, f"E{emp_id:04d}", fake.name(), dept_id, title,
                    today - timedelta(days=rng.randrange(60, 365 * 18)),
                    f"e{emp_id:04d}@democo.example", f"09{rng.randrange(10**8):08d}",
                    int(base * rng.uniform(0.9, 1.25)) // 100 * 100,
                    "離職" if rng.random() < 0.05 else "在職",
                )
            )  # fmt: skip
            by_dept.setdefault(dept, []).append(emp_id)
        departments.append((dept_id, dept, by_dept[dept][0]))
    db.cursor().executemany("INSERT INTO hr.departments VALUES (%s, %s, %s)", departments)
    db.cursor().executemany(
        "INSERT INTO hr.employees VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", employees
    )

    leave_types = ["特休"] * 6 + ["病假"] * 3 + ["事假"] * 2 + ["婚假", "喪假", "公假"]
    leaves = []
    for i in range(1, 1601):
        start = date(2026, 1, 2) + timedelta(days=rng.randrange(270))
        days = rng.choice([0.5, 1, 1, 1, 2, 3, 5])
        leaves.append(
            (i, rng.randrange(1, len(employees) + 1), rng.choice(leave_types), start,
             start + timedelta(days=max(int(days) - 1, 0)), days,
             rng.choices(["核准", "待審", "駁回"], [85, 10, 5])[0])
        )  # fmt: skip
    db.cursor().executemany("INSERT INTO hr.leave_requests VALUES (%s,%s,%s,%s,%s,%s,%s)", leaves)

    products = []
    for category, prefix, sizes, base in PRODUCT_LINES:
        for size in sizes:
            for grade in ("標準級", "精密級"):
                price = int(
                    base * (1 + sizes.index(size) * 0.35) * (1.4 if grade == "精密級" else 1)
                )
                pid = len(products) + 1
                sku = f"{prefix}-{size}{'P' if grade == '精密級' else 'N'}"
                products.append(
                    (pid, sku, f"{category} {prefix}{size}（{grade}）", category,
                     price // 10 * 10, int(price * rng.uniform(0.55, 0.7)))
                )  # fmt: skip
    db.cursor().executemany("INSERT INTO sales.products VALUES (%s,%s,%s,%s,%s,%s)", products)

    sales_reps = [e for e in by_dept["業務部"] if employees[e - 1][9] == "在職"]
    customers, names = [], set()
    while len(customers) < 400:
        name = (
            f"{rng.choice(NAME_HEADS)}{rng.choice(NAME_HEADS)}{rng.choice(NAME_TAILS)}股份有限公司"
        )
        if name in names:
            continue
        names.add(name)
        tier = rng.choices("ABC", [15, 35, 50])[0]
        customers.append(
            (len(customers) + 1, name, rng.choice(INDUSTRIES), rng.choice(CITIES), tier,
             {"A": 5_000_000, "B": 2_000_000, "C": 600_000}[tier], rng.choice(sales_reps))
        )  # fmt: skip
    db.cursor().executemany("INSERT INTO sales.customers VALUES (%s,%s,%s,%s,%s,%s,%s)", customers)

    orders, items = [], []
    weights = [{"A": 6, "B": 3, "C": 1}[c[4]] for c in customers]
    for oid in range(1, 4201):
        day = date(2026, 1, 2) + timedelta(days=rng.randrange(283))
        customer = rng.choices(customers, weights)[0]
        age = (today - day).days
        status = (
            "已取消" if rng.random() < 0.04
            else "處理中" if age < 10 else "已出貨" if age < 25 else "已完成"
        )  # fmt: skip
        orders.append((oid, f"SO{day:%y%m}{oid:05d}", customer[0], day, status, customer[6]))
        for product in rng.sample(products, rng.randint(1, 4)):
            discount = {"A": 0.88, "B": 0.94, "C": 1.0}[customer[4]]
            items.append((oid, product[0], rng.choice([2, 4, 5, 10, 20, 50]),
                          int(product[4] * discount) // 10 * 10))  # fmt: skip
    db.cursor().executemany("INSERT INTO sales.orders VALUES (%s,%s,%s,%s,%s,%s)", orders)
    db.cursor().executemany("INSERT INTO sales.order_items VALUES (%s,%s,%s,%s)", items)

    warehouses = [(1, "桃園總倉", "桃園市"), (2, "台中倉", "台中市"), (3, "高雄倉", "高雄市")]
    db.cursor().executemany("INSERT INTO inventory.warehouses VALUES (%s,%s,%s)", warehouses)
    stock = [
        (p[0], w[0], rng.randrange(0, 400), rng.choice([30, 50, 80]))
        for p in products
        for w in warehouses
    ]
    db.cursor().executemany("INSERT INTO inventory.stock VALUES (%s,%s,%s,%s)", stock)
    pos = []
    for i in range(1, 321):
        day = date(2026, 1, 5) + timedelta(days=rng.randrange(280))
        expected = day + timedelta(days=rng.randrange(14, 60))
        status = "已入庫" if expected < today else rng.choice(["已下單", "運送中"])
        pos.append((i, f"PO{day:%y%m}{i:04d}", rng.choice(SUPPLIERS), rng.choice(products)[0],
                    rng.choice([50, 100, 200, 500]), day, expected, status))  # fmt: skip
    db.cursor().executemany(
        "INSERT INTO inventory.purchase_orders VALUES (%s,%s,%s,%s,%s,%s,%s,%s)", pos
    )

    support_staff = by_dept["客服部"]
    tickets = []
    for i in range(1, 951):
        order = rng.choice(orders)
        customer_id, order_no = order[2], order[1]
        product = products[rng.choice([it for it in items if it[0] == order[0]])[1] - 1]
        category = rng.choices(list(TICKET_TEMPLATES), [22, 25, 15, 10, 18, 10])[0]
        subject, body = TICKET_TEMPLATES[category]
        created = datetime.combine(order[3], datetime.min.time()) + timedelta(
            days=rng.randrange(3, 40), hours=rng.randrange(8, 18)
        )
        if created.date() > today:
            created = datetime.combine(today, datetime.min.time()) - timedelta(
                hours=rng.randrange(1, 200)
            )
        age = (datetime.combine(today, datetime.min.time()) - created).days
        status = (
            rng.choice(["待處理", "處理中"])
            if age < 5
            else rng.choices(["處理中", "已解決", "已結案"], [10, 30, 60])[0]
        )
        resolved = (
            created + timedelta(hours=rng.randrange(2, 120))
            if status in ("已解決", "已結案")
            else None
        )
        tickets.append(
            (i, f"CS{created:%y%m}{i:04d}", customer_id, product[0], created,
             rng.choice(["電話", "Email", "LINE", "業務轉介"]), category,
             rng.choices(["低", "中", "高", "緊急"], [20, 45, 27, 8])[0], status,
             subject.format(p=product[2], o=order_no), body.format(p=product[2], o=order_no),
             rng.choice(support_staff), resolved)
        )  # fmt: skip
    db.cursor().executemany(
        "INSERT INTO support.tickets VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", tickets
    )


# --- Agent tools ----------------------------------------------------------------------


def _query(role: str, query: str, params=None, limit: int = MAX_ROWS):
    password = _password(f"LOCAL_LLM_PG_{role.upper()}_PASSWORD")
    with psycopg.connect(conninfo(role, password)) as db:
        db.read_only = True  # second guard on top of the role default
        cursor = db.execute(query, params)
        columns = [c.name for c in cursor.description or []]
        rows = cursor.fetchmany(limit + 1)
    return columns, rows


def _json_value(value):
    """SUM/AVG return Decimal: keep them numbers (not strings) so the model reads amounts."""
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    return str(value)


def _json(columns, rows, limit=MAX_ROWS) -> str:
    data = [dict(zip(columns, row, strict=True)) for row in rows[:limit]]
    note = f"\n(only the first {limit} rows are shown)" if len(rows) > limit else ""
    return json.dumps(data, ensure_ascii=False, default=_json_value) + note


def make_tools(role: str) -> list:
    """Database tools bound to one PostgreSQL role (the agent's department identity)."""

    @tool
    def db_list_tables() -> str:
        """List the company database tables you are allowed to read, with descriptions."""
        cols, rows = _query(
            role,
            """SELECT t.table_schema || '.' || t.table_name AS table,
                      obj_description((quote_ident(t.table_schema) || '.' ||
                                       quote_ident(t.table_name))::regclass) AS description
               FROM information_schema.tables t
               WHERE t.table_schema = ANY(%s)
                 AND EXISTS (SELECT 1 FROM information_schema.column_privileges p
                             WHERE p.table_schema = t.table_schema
                               AND p.table_name = t.table_name
                               AND p.privilege_type = 'SELECT')
               ORDER BY 1""",
            [list(SCHEMAS)],
        )
        # pg_class lists every table to any role; report how many are off-limits (count only)
        # so the model says "no permission" instead of "that table does not exist".
        _, total = _query(
            role,
            """SELECT count(*) FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
               WHERE n.nspname = ANY(%s) AND c.relkind = 'r'""",
            [list(SCHEMAS)],
        )
        hidden = total[0][0] - len(rows)
        if not hidden:
            return _json(cols, rows)
        return (
            f"PERMISSION NOTICE: {hidden} more company table(s) exist that your role is not "
            "allowed to read. If the user asks about data not in the tables below, answer that "
            "your role has no permission to access it.\n" + _json(cols, rows)
        )

    @tool
    def db_describe_table(table: str) -> str:
        """Show the columns (name, type, description) of one table, e.g. 'sales.orders'.
        Only columns you are allowed to read are listed."""
        schema, _, name = table.partition(".")
        cols, rows = _query(
            role,
            """SELECT c.column_name, c.data_type,
                      col_description((quote_ident(c.table_schema) || '.' ||
                                       quote_ident(c.table_name))::regclass,
                                      c.ordinal_position) AS description
               FROM information_schema.columns c
               WHERE c.table_schema = %s AND c.table_name = %s
                 AND EXISTS (SELECT 1 FROM information_schema.column_privileges p
                             WHERE p.table_schema = c.table_schema
                               AND p.table_name = c.table_name
                               AND p.column_name = c.column_name
                               AND p.privilege_type = 'SELECT')
               ORDER BY c.ordinal_position""",
            [schema, name],
        )
        if not rows:
            return f"No readable table named {table!r}."
        # pg_attribute lists every column to any user; the difference is what this role
        # cannot read. Report only the count so the model says "no permission", not
        # "the data does not exist", without revealing the hidden column names.
        _, total = _query(
            role,
            """SELECT count(*) FROM pg_attribute
               WHERE attrelid = (quote_ident(%s) || '.' || quote_ident(%s))::regclass
                 AND attnum > 0 AND NOT attisdropped""",
            [schema, name],
        )
        hidden = total[0][0] - len(rows)
        if not hidden:
            return _json(cols, rows)
        return (
            f"PERMISSION NOTICE: this table has {hidden} more column(s) that your role is not "
            "allowed to read. If the user asks for data not in the columns below (for example "
            "salary or cost), answer that your role has no permission to access it.\n"
            + _json(cols, rows)
        )

    @tool
    def db_query(query: str) -> str:
        """Run one read-only PostgreSQL SELECT query on the company database and return rows
        as JSON (max 100 rows). Use schema-qualified table names such as sales.orders."""
        if not query.strip().lower().startswith(("select", "with")):
            return "Only SELECT queries are allowed."
        try:
            cols, rows = _query(role, query)
        except psycopg.errors.InsufficientPrivilege as exc:
            return f"Permission denied by the database for role {role}: {exc}".strip()
        except psycopg.Error as exc:
            return f"SQL error: {exc}".strip()
        return _json(cols, rows)

    return [db_list_tables, db_describe_table, db_query]


def readable_columns(role: str) -> dict[str, tuple[list[str], int]]:
    """{'schema.table': (columns this role may read, total columns)} for company tables."""
    _, rows = _query(
        role,
        """SELECT a.attrelid::regclass::text, a.attname,
                  has_column_privilege(a.attrelid, a.attnum, 'SELECT')
           FROM pg_attribute a JOIN pg_class c ON c.oid = a.attrelid
           JOIN pg_namespace n ON n.oid = c.relnamespace
           WHERE n.nspname = ANY(%s) AND c.relkind = 'r'
             AND a.attnum > 0 AND NOT a.attisdropped
           ORDER BY 1, a.attnum""",
        [list(SCHEMAS)],
        limit=10_000,
    )
    result: dict[str, tuple[list[str], int]] = {}
    for table, column, allowed in rows:
        columns, total = result.get(table, ([], 0))
        result[table] = (columns + [column] if allowed else columns, total + 1)
    return result


def preview(role: str, table: str, columns: list[str] | None = None, limit: int = 20):
    """First rows of a table as `role` sees them (only permitted columns are selected).

    `columns` defaults to every column the role may read; PermissionError if there are none.
    """
    if columns is None:
        columns = readable_columns(role).get(table, ([], 0))[0]
        if not columns:
            raise PermissionError(f"{role} cannot read {table}")
    schema, name = table.split(".")
    query = sql.SQL("SELECT {} FROM {} LIMIT {}").format(
        sql.SQL(", ").join(map(sql.Identifier, columns)),
        sql.Identifier(schema, name),
        sql.Literal(limit),
    )
    password = _password(f"LOCAL_LLM_PG_{role.upper()}_PASSWORD")
    with psycopg.connect(conninfo(role, password)) as db:
        db.read_only = True
        cursor = db.execute(query)
        return [c.name for c in cursor.description], cursor.fetchall()


# --- Admin (platform page): uses the superuser; agents never get these --------------

IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")
PG_TYPES = {"i": "bigint", "u": "bigint", "f": "double precision", "b": "boolean", "M": "timestamp"}


def _admin():
    return psycopg.connect(conninfo("postgres", _password("LOCAL_LLM_PG_SUPERUSER_PASSWORD")))


def admin_tables() -> list[dict]:
    """Every company table with its row count, columns and description."""
    with _admin() as db:
        rows = db.execute(
            """SELECT n.nspname || '.' || c.relname, obj_description(c.oid), c.oid
               FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
               WHERE n.nspname = ANY(%s) AND c.relkind = 'r' ORDER BY 1""",
            [list(SCHEMAS)],
        ).fetchall()
        result = []
        for name, description, oid in rows:
            columns = [
                r[0]
                for r in db.execute(
                    "SELECT attname FROM pg_attribute WHERE attrelid = %s AND attnum > 0 "
                    "AND NOT attisdropped ORDER BY attnum",
                    [oid],
                )
            ]
            count = db.execute(
                sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(*name.split(".")))
            ).fetchone()[0]
            result.append(
                {"table": name, "description": description, "columns": columns, "rows": count}
            )
    return result


def admin_preview(table: str, limit: int = 50):
    with _admin() as db:
        cursor = db.execute(
            sql.SQL("SELECT * FROM {} LIMIT {}").format(
                sql.Identifier(*table.split(".")), sql.Literal(limit)
            )
        )
        return [c.name for c in cursor.description], cursor.fetchall()


def import_dataframe(df, schema: str, table: str, description: str = "", replace=False) -> int:
    """Create `schema.table` from a pandas DataFrame and load its rows.

    New tables are readable by no department role until permissions are granted.
    """
    if schema not in SCHEMAS:
        raise ValueError(f"schema must be one of {SCHEMAS}")
    if not IDENTIFIER.match(table):
        raise ValueError("表名只能用小寫英文、數字、底線，且不能以數字開頭")
    df = df.copy()
    df.columns = [str(c).strip() or f"col_{i}" for i, c in enumerate(df.columns, start=1)]
    if len(set(df.columns)) != len(df.columns):
        raise ValueError("欄位名稱重複")
    target = sql.Identifier(schema, table)
    columns = [
        sql.SQL("{} {}").format(sql.Identifier(c), sql.SQL(PG_TYPES.get(df[c].dtype.kind, "text")))
        for c in df.columns
    ]
    with _admin() as db:
        db.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema)))
        if replace:
            db.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(target))
        db.execute(sql.SQL("CREATE TABLE {} ({})").format(target, sql.SQL(", ").join(columns)))
        if description:
            db.execute(
                sql.SQL("COMMENT ON TABLE {} IS {}").format(target, sql.Literal(description))
            )
        records = df.astype(object).where(df.notna(), None).itertuples(index=False, name=None)
        with db.cursor().copy(
            sql.SQL("COPY {} ({}) FROM STDIN").format(
                target, sql.SQL(", ").join(map(sql.Identifier, df.columns))
            )
        ) as copy:
            for record in records:
                copy.write_row(record)
    return len(df)


def set_access(table: str, role: str, columns: list[str] | None) -> None:
    """Replace `role`'s read access to `table`: None = no access, [] = whole table,
    otherwise only the listed columns. Applied with GRANT/REVOKE inside PostgreSQL.

    Works for every identity; for the extra identities the rule is also saved in
    storage/lab/identities.json so that `init` (which recreates the database) restores it.
    """
    if role not in identity_roles():
        raise ValueError(f"unknown identity {role}")
    with _admin() as db:
        _set_access(db, table, role, columns)
    if role not in ROLES:
        with _STORE_LOCK:
            store = _read_store()
            for ident in store["identities"]:
                if ident["role"] == role:
                    grants = ident.setdefault("grants", {})
                    if columns is None:
                        grants.pop(table, None)
                    else:
                        grants[table] = list(columns)
            _write_store(store)


def _set_access(db, table: str, role: str, columns: list[str] | None) -> None:
    schema, name = table.split(".")
    target, ident = sql.Identifier(schema, name), sql.Identifier(role)
    all_columns = [
        r[0]
        for r in db.execute(
            "SELECT attname FROM pg_attribute WHERE attrelid = %s::regclass AND attnum > 0 "
            "AND NOT attisdropped",
            [f"{schema}.{name}" if IDENTIFIER.match(name) else f'{schema}."{name}"'],
        )
    ]
    db.execute(sql.SQL("REVOKE ALL ON {} FROM {}").format(target, ident))
    db.execute(
        sql.SQL("REVOKE SELECT ({}) ON {} FROM {}").format(
            sql.SQL(", ").join(map(sql.Identifier, all_columns)), target, ident
        )
    )
    if columns is None:
        return
    unknown = sorted(set(columns) - set(all_columns))
    if unknown:
        raise ValueError(f"{table} has no column(s) {unknown}")
    db.execute(sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(sql.Identifier(schema), ident))
    if columns:
        db.execute(
            sql.SQL("GRANT SELECT ({}) ON {} TO {}").format(
                sql.SQL(", ").join(map(sql.Identifier, columns)), target, ident
            )
        )
    else:
        db.execute(sql.SQL("GRANT SELECT ON {} TO {}").format(target, ident))


def access_matrix(roles: list[str] | None = None) -> dict[str, dict[str, tuple[list[str], int]]]:
    """{role: {'schema.table': (readable columns, total columns)}} for every identity, read
    from PostgreSQL's own privilege checks (one admin query instead of one login per role)."""
    roles = roles or identity_roles()
    with _admin() as db:
        rows = db.execute(
            """SELECT r.rolname, n.nspname || '.' || c.relname, a.attname,
                      has_column_privilege(r.oid, a.attrelid, a.attnum, 'SELECT')
               FROM pg_roles r
               CROSS JOIN pg_attribute a
               JOIN pg_class c ON c.oid = a.attrelid
               JOIN pg_namespace n ON n.oid = c.relnamespace
               WHERE r.rolname = ANY(%s) AND n.nspname = ANY(%s) AND c.relkind = 'r'
                 AND a.attnum > 0 AND NOT a.attisdropped
               ORDER BY 1, 2, a.attnum""",
            [roles, list(SCHEMAS)],
        ).fetchall()
    result: dict[str, dict[str, tuple[list[str], int]]] = {r: {} for r in roles}
    for role, table, column, allowed in rows:
        columns, total = result[role].get(table, ([], 0))
        result[role][table] = (columns + [column] if allowed else columns, total + 1)
    return result


# --- Extra identities (beyond the three built-in departments) ---------------------------

LAB_DIR = PROJECT_DIR / "storage" / "lab"
IDENTITIES_FILE = LAB_DIR / "identities.json"
AGENTS_FILE = PROJECT_DIR / "data" / "lab" / "agents.json"
ROLE_NAME = re.compile(r"^agent_[a-z][a-z0-9_]{1,30}$")
_STORE_LOCK = threading.RLock()


def _env_key(role: str) -> str:
    return f"LOCAL_LLM_PG_{role.upper()}_PASSWORD"


def _read_store() -> dict:
    try:
        data = json.loads(IDENTITIES_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        data = {}
    data.setdefault("seeded", False)
    data.setdefault("identities", [])
    return data


def _write_store(data: dict) -> None:
    IDENTITIES_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = IDENTITIES_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(IDENTITIES_FILE)


def _custom_identities() -> list[dict]:
    return _read_store()["identities"]


def _superuser():
    return psycopg.connect(
        conninfo("postgres", _password("LOCAL_LLM_PG_SUPERUSER_PASSWORD"), "postgres"),
        autocommit=True,
    )


def _login_role(c, role: str, password: str) -> None:
    """Create or update a LOGIN role that can only read, with a statement time limit."""
    ident = sql.Identifier(role)
    exists = c.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", [role]).fetchone()
    c.execute(
        sql.SQL(
            "{} ROLE {} LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE PASSWORD {} CONNECTION LIMIT 10"
        ).format(  # fmt: skip
            sql.SQL("ALTER" if exists else "CREATE"), ident, sql.Literal(password)
        )
    )
    c.execute(sql.SQL("ALTER ROLE {} SET default_transaction_read_only = on").format(ident))
    c.execute(sql.SQL("ALTER ROLE {} SET statement_timeout = '5s'").format(ident))


def _restore_identity(ident: dict) -> None:
    """Recreate an extra identity's role (if needed) and its grants, e.g. after `init`."""
    role, env = ident["role"], _env_key(ident["role"])
    load_dotenv(ENV_FILE)
    password = os.getenv(env) or secrets.token_hex(16)
    with _superuser() as c:
        _login_role(c, role, password)
    set_key(str(ENV_FILE), env, password)
    os.environ[env] = password
    with _admin() as db:
        db.execute(
            sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(
                sql.Identifier(DB_NAME), sql.Identifier(role)
            )
        )
        tables = {
            r[0]
            for r in db.execute(
                "SELECT n.nspname || '.' || c.relname FROM pg_class c "
                "JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE n.nspname = ANY(%s) AND c.relkind = 'r'",
                [list(SCHEMAS)],
            )
        }
        for table, columns in (ident.get("grants") or {}).items():
            if table in tables:  # skip uploaded tables that were deleted since
                _set_access(db, table, role, columns)


def create_identity(
    role: str, label: dict, description: dict | None = None, grants: dict | None = None
) -> dict:
    """Create a read-only database identity (PostgreSQL LOGIN role) for agents.

    label / description: {"zh": ..., "en": ...}. grants: {table: None | [] | [columns]}.
    Like the built-in roles: SELECT only, default_transaction_read_only, 5 s statement
    timeout, at most 10 connections; the generated password is stored in .env.
    """
    role = role.strip().lower()
    if not ROLE_NAME.match(role):
        raise ValueError(
            "身分代號需為 agent_ 開頭，接小寫英文、數字或底線（例如 agent_finance） / "
            "The identity id must be agent_ followed by lowercase letters, digits or _"
        )
    if not (label.get("zh") or label.get("en")):
        raise ValueError("請填寫身分名稱 / A label is required")
    with _STORE_LOCK:
        if role in identity_roles():
            raise ValueError(f"{role} 已存在 / {role} already exists")
        record = {
            "role": role,
            "label": {
                "zh": label.get("zh") or label.get("en"),
                "en": label.get("en") or label["zh"],
            },
            "description": {
                "zh": (description or {}).get("zh", ""),
                "en": (description or {}).get("en", ""),
            },
            "grants": {t: c for t, c in (grants or {}).items() if c is not None},
            "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        _restore_identity(record)  # PostgreSQL first: nothing is listed if this fails
        store = _read_store()
        store["identities"].append(record)
        _write_store(store)
    return record


def agents_using(role: str) -> list[str]:
    """Ids of agent definitions (data/lab/agents.json) that connect as `role`."""
    try:
        agents = json.loads(AGENTS_FILE.read_text(encoding="utf-8"))["agents"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return []
    return [a["id"] for a in agents if a.get("db_role") == role]


def delete_identity(role: str) -> None:
    """Drop an extra identity: its privileges, its PostgreSQL role and its password."""
    if role in ROLES:
        raise ValueError("內建部門身分不能刪除 / Built-in department identities cannot be deleted")
    if role not in {i["role"] for i in _custom_identities()}:
        raise ValueError(f"unknown identity {role}")
    if users := agents_using(role):
        raise ValueError(
            f"仍有 agent 使用這個身分：{', '.join(users)} / Still used by agents: {', '.join(users)}"
        )
    ident = sql.Identifier(role)
    with _superuser() as c:
        exists = c.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", [role]).fetchone()
    if exists:
        with _admin() as db:  # privileges in democo, including CONNECT on the database
            db.execute(sql.SQL("DROP OWNED BY {}").format(ident))
        with _superuser() as c:
            c.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE usename = %s", [role]
            )
            c.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(ident))
    with _STORE_LOCK:
        store = _read_store()
        store["identities"] = [i for i in store["identities"] if i["role"] != role]
        _write_store(store)
    if ENV_FILE.exists():
        unset_key(str(ENV_FILE), _env_key(role))
    os.environ.pop(_env_key(role), None)


def update_identity(role: str, label: dict | None = None, description: dict | None = None) -> None:
    """Rename or re-describe an extra identity (its grants are edited with set_access)."""
    with _STORE_LOCK:
        store = _read_store()
        for ident in store["identities"]:
            if ident["role"] == role:
                if label:
                    ident["label"] = {**ident["label"], **{k: v for k, v in label.items() if v}}
                if description is not None:
                    ident["description"] = {**ident.get("description", {}), **description}
                _write_store(store)
                return
    raise ValueError(f"unknown or built-in identity {role}")


def ensure_seed_identities() -> list[str]:
    """Create the example identities (finance, management, intern) once; returns new roles.

    Runs on the first visit of the Company database page and after `init`. Deleting a seed
    identity later is respected (the store remembers that seeding happened).
    """
    with _STORE_LOCK:
        store = _read_store()
        if store["seeded"]:
            return []
        existing = {i["role"] for i in store["identities"]}
        created = []
        for seed in SEED_IDENTITIES:
            if seed["role"] not in existing:
                create_identity(**seed)
                created.append(seed["role"])
        store = _read_store()
        store["seeded"] = True
        _write_store(store)
        return created


def drop_table(table: str) -> None:
    schema, name = table.split(".")
    if schema != "uploads":
        raise ValueError("只能刪除自己上傳（uploads）的資料表")
    with _admin() as db:
        db.execute(sql.SQL("DROP TABLE {}").format(sql.Identifier(schema, name)))


def status() -> str | None:
    """None when the database is reachable with the sales role, else an error message."""
    try:
        _query("agent_sales", "SELECT 1")
        return None
    except (psycopg.Error, RuntimeError) as exc:
        return str(exc).strip()


if __name__ == "__main__":
    if sys.argv[1:] == ["init"]:
        init()
    else:
        print("usage: python -m local_llm.company_db init")
