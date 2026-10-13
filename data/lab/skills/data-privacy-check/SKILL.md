---
name: data-privacy-check
description: Classify information before it leaves the company or reaches a wider audience - external or network tools, web searches, email, reports for customers or suppliers. Finds personal data (PII), secrets and confidential business data, then masks, aggregates or blocks it and gives a send / fix / do-not-send verdict. Use before every network tool call and before sharing results outside the team.
description_zh: 資料送出公司或擴大分享前的分類檢查（外部／連網工具、網路搜尋、Email、給客戶或供應商的報告）：找出個資、密碼金鑰與機密營業資料，遮罩、彙總或擋下，並給出「可送出／修改後送出／不可送出」的結論。每次呼叫連網工具前、把結果分享到團隊外之前使用。
source: local
license: internal
---

# 送出前資料分類檢查

台灣《個人資料保護法》與公司保密規範都適用。連網工具（`get_weather`、`wikipedia_search`、`fetch_webpage`、`mcp_fetch`）的參數會離開這台電腦；人工核准前，用這份檢查說明你要送出什麼。

## 分類

| 等級 | 例子（本公司） | 可以送出嗎 |
|---|---|---|
| 🔴 秘密／憑證 | 密碼、API key、token、`.env` 內容、資料庫連線字串、私鑰 | **絕不送出**，也不要寫進回答 |
| 🔴 個資 | 員工姓名＋電話／email／員工編號、身分證字號、地址、生日、**薪資**、請假與病假紀錄（健康相關屬特種個資） | 不送出；必要時去識別化 |
| 🟠 機密 | 產品成本 `unit_cost`、毛利、成交價與折扣、客戶信用額度、合約條款、供應商報價、未公開圖面與規格 | 不送出公司外；內部依職務分享 |
| 🟡 內部 | 組織架構、流程、彙總後的訂單統計、工單數量 | 去識別、彙總後經核准可送 |
| 🟢 公開 | 公司官網資訊、產品型錄與牌價、公開法規、天氣與一般知識 | 可以送出 |

客戶公司名稱屬「內部」：搜尋或外部工具的關鍵字**不要帶客戶名、人名、訂單號**。

## 偵測線索

- 身分證字號：英文字母＋1 或 2＋8 碼數字（例：A123456789）
- 手機：09 開頭 10 碼（0912-345-678、0912345678）
- Email、地址（路／街＋號）、生日日期
- 金額欄位旁出現「成本」「薪」「底價」「折扣」「信用額度」
- 長串英數（32 碼以上）、`sk-`、`password=`、`postgres://`

## 處置方式

- 遮罩：王小明 → 王○明；0912-345-678 → 0912-xxx-678；A123456789 → A12xxxxx89
- 彙總：「3 位員工請病假共 7 天」取代逐筆名單；金額改區間或占比
- 移除：秘密／憑證一律刪掉
- 替代：用產品類別、地區、客戶等級取代具名資料

## 輸出格式

```markdown
### 送出前檢查
| 內容 | 等級 | 處置 |
|---|---|---|
| 〈項目〉 | 🔴/🟠/🟡/🟢 | 保留／遮罩／彙總／移除 |

**結論**：✅ 可以送出／⚠️ 修改後送出（附修改後版本）／⛔ 不可送出（原因）
```

不確定就歸到較嚴格的等級。核准仍由人決定；這份檢查是讓核准的人看清楚要送出什麼。
