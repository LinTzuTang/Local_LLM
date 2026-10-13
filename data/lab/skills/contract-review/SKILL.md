---
name: contract-review
description: Review a supplier or customer contract (for example one parsed with parse_document) against the company checklist - summarise key terms, flag risky or missing clauses with a risk level and suggested wording, and list questions for legal. Use when asked to review, summarise or compare contract terms. Gives a first-pass review, not legal advice.
description_zh: 依公司檢核表審閱供應商或客戶合約（例如用 parse_document 解析的檔案）：整理關鍵條款、標出有風險或缺漏的條款（含風險等級與修改建議），並列出要問法務的問題。審閱、摘要或比較合約條款時使用。屬初步檢視，不是法律意見。
source: local
license: internal
---

# 合約審閱

## 步驟

1. 讀完整份合約（檔案在 inbox 就先 `parse_document`）。
2. 做「合約摘要表」。
3. 逐項對照下方檢核表，標出風險與缺漏。
4. 引用時寫**條號＋原文短句**（例：第 8.2 條「乙方不負任何間接損害賠償責任」）；找不到條號就寫段落位置。
5. 不確定的解讀寫「需法務確認」，不要自己下法律結論。

## 公司標準（虛構示範公司的內部基準）

| 項目 | 公司可接受 | 高風險訊號 |
|---|---|---|
| 付款條件 | 採購：月結 60 天；銷售：月結 90 天內 | 預付超過 30%、收款超過 120 天 |
| 交期遲延罰則 | 每日 0.1%，上限合約金額 10% | 無上限、按日 0.5% 以上 |
| 保固 | 12 個月或依產品規格 | 超過 24 個月、或「終身」 |
| 責任上限 | 以合約金額或 12 個月付款總額為上限 | 無上限、包含間接損失與營業損失 |
| 保密 | 期間至少 3 年，雙方對等 | 單方義務、無期限的競業限制 |
| 智慧財產 | 圖面、模具、規格歸屬寫清楚 | 我方圖面或模具歸對方 |
| 準據法／管轄 | 中華民國法律；臺灣新竹或臺北地方法院 | 外國法、外國仲裁 |
| 價格調整 | 有原物料或匯率調整機制 | 長約固定價且無調整條款 |
| 續約／終止 | 雙方可書面通知終止 | 自動續約且難以終止、最低採購量 |

另外檢查：規格與驗收標準（圖面版本、公差、材質、檢驗方式）、不良品處理與召回、不可抗力、個資與出口管制、轉包限制、簽約權限與用印。

## 輸出格式

```markdown
## 一、合約摘要
| 項目 | 內容 |
|---|---|
| 當事人 | |
| 標的 | |
| 期間 | |
| 金額／價格 | |
| 付款條件 | |
| 交期 | |

## 二、風險條款
| 條款 | 風險 | 說明 | 建議修改 |
|---|---|---|---|
| 第 x 條「…」 | 🔴 高／🟡 中／🟢 低 | | |

## 三、缺漏條款
- 〈檢核表有、合約沒寫的項目〉

## 四、請法務確認
1. 〈問題〉
```

最後一行固定加註：「以上為 AI 初步檢視，非法律意見；簽署前請法務確認。」
