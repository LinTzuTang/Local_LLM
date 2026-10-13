---
name: sql-analyst
description: Explore and query the company PostgreSQL database (democo) safely and with the company's official metric definitions. Use before answering any question that needs numbers from company data - revenue, orders, customers, products, stock, purchase orders, support tickets, headcount - with the db_list_tables, db_describe_table and db_query tools.
description_zh: 安全、正確地查詢公司資料庫（democo），並使用公司統一的指標定義。回答任何需要公司數字的問題（營收、訂單、客戶、產品、庫存、採購、客服工單、人數）前先載入，搭配 db_list_tables／db_describe_table／db_query 使用。
source: local
license: internal
---

# SQL 分析師：公司資料庫查詢規範

示範精密工業股份有限公司（虛構）的資料在 PostgreSQL `democo`。你用的資料庫身分決定你看得到哪些表與欄位，這是由資料庫本身強制的。

## 工作流程

1. 不確定有哪些表 → 先 `db_list_tables`。要用的表 → 先 `db_describe_table('schema.table')` 看欄位（只會列出你這個身分可讀的欄位）。
2. 一次只寫**一個** `SELECT`（可以用 `WITH`）。表名一律加 schema，例如 `sales.orders`。
3. **在資料庫裡彙總**：用 `GROUP BY`、`SUM`、`COUNT`、`ORDER BY … LIMIT 20`。不要撈明細回來自己加總（工具最多只回 100 列，會漏算）。
4. 結果不合理（例如 0、負數、比上月多 10 倍）→ 先檢查 JOIN 與篩選條件再回答。
5. 回答要附：數字、期間、篩選條件、單位（新台幣）。使用者要求或數字重要時，附上你用的 SQL。

## 公司統一的指標定義

| 指標 | 定義 |
|---|---|
| **營收** | `SUM(oi.quantity * oi.unit_price)`，來自 `sales.order_items`（`unit_price` 是成交價），**排除 `sales.orders.status = '已取消'` 的訂單** |
| 訂單數 | `COUNT(DISTINCT o.id)`，同樣排除「已取消」 |
| 期間 | 依 `sales.orders.order_date`；「本月」「今年」用 `current_date` 計算 |
| 訂單狀態 | 處理中、已出貨、已完成、已取消；「有效訂單」= 非已取消 |
| 牌價 vs 成交價 | 牌價 `sales.products.unit_price`；成交價 `sales.order_items.unit_price`（A 級客戶約 88 折、B 級約 94 折） |
| **毛利** | `SUM(oi.quantity * (oi.unit_price - p.unit_cost))`；`unit_cost`（成本）是機密，客服、實習生等身分讀不到 |
| 客戶等級 | `sales.customers.tier`：A／B／C |
| 需補貨 | `inventory.stock.qty_on_hand < reorder_point` |
| 未結工單 | `support.tickets.status IN ('待處理','處理中')` |
| 處理時間 | `resolved_at - created_at`（只算已解決／已結案） |
| 在職人數 | `hr.employees.status = '在職'` |

## 範本

```sql
-- 各月營收（排除已取消）
SELECT date_trunc('month', o.order_date)::date AS month,
       SUM(oi.quantity * oi.unit_price) AS revenue
FROM sales.orders o
JOIN sales.order_items oi ON oi.order_id = o.id
WHERE o.status <> '已取消'
GROUP BY 1 ORDER BY 1;

-- 今年前 10 大客戶
SELECT c.name, c.tier, SUM(oi.quantity * oi.unit_price) AS revenue
FROM sales.orders o
JOIN sales.order_items oi ON oi.order_id = o.id
JOIN sales.customers c ON c.id = o.customer_id
WHERE o.status <> '已取消'
  AND o.order_date >= date_trunc('year', current_date)
GROUP BY c.name, c.tier
ORDER BY revenue DESC LIMIT 10;

-- 各產品類別營收占比
SELECT p.category, SUM(oi.quantity * oi.unit_price) AS revenue,
       round(100.0 * SUM(oi.quantity * oi.unit_price)
             / SUM(SUM(oi.quantity * oi.unit_price)) OVER (), 1) AS pct
FROM sales.orders o
JOIN sales.order_items oi ON oi.order_id = o.id
JOIN sales.products p ON p.id = oi.product_id
WHERE o.status <> '已取消'
GROUP BY p.category ORDER BY revenue DESC;
```

## 安全與權限

- 只讀：只能下 `SELECT`。不要嘗試 INSERT／UPDATE／DELETE／DDL／GRANT。
- 遇到 `permission denied` 或 `PERMISSION NOTICE`：直接告訴使用者「我的資料庫身分沒有權限讀取這些資料」，**不要換表或換欄位繞過**，也不要猜數字。
- 不要在回答列出個資明細（電話、薪資、個人資料）；需要時只給彙總或人數。
- 查不到就說查不到，並說明用了什麼條件；不要編造。

## 數字格式

- 金額：NT$1,234,567；大數字可加「約 123.5 萬元」。
- 百分比保留一位小數；成長率寫清楚比較基準（例如「較 2026 年 8 月成長 12.3%」）。
