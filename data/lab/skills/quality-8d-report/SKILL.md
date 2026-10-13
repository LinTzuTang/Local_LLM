---
name: quality-8d-report
description: Turn a customer quality complaint (a 品質異常 support ticket - noise, wear, failure, wrong dimension, batch defect) into an 8D problem-solving report (D0-D8) with 5W2H problem statement, containment within 24 hours, root cause (5 Why, fishbone) and corrective and preventive actions. Use when a ticket or the user reports a product defect or asks for an 8D or corrective-action report.
description_zh: 把客戶品質異常（客服工單「品質異常」：異音、磨耗、故障、尺寸不符、批量不良）整理成 8D 報告（D0–D8）：5W2H 問題描述、24 小時內圍堵、根本原因（5 Why、魚骨圖）、矯正與預防措施。工單或使用者回報產品不良、或要求 8D／矯正措施報告時使用。
source: local
license: internal
---

# 品質異常 8D 報告

## 公司時限（虛構示範公司）

D3 圍堵：24 小時內｜D4–D5 原因與對策：7 天內｜D8 結案：30 天內。A 級客戶或安全相關問題，D3 當天回報品保經理。

## 先蒐集事實（有資料庫工具時）

```sql
-- 同一產品近 90 天的品質異常工單（看是否為批量問題）
SELECT t.ticket_no, t.created_at::date, t.priority, t.status, t.subject
FROM support.tickets t
WHERE t.product_id = 〈產品 id〉 AND t.category = '品質異常'
  AND t.created_at >= current_date - 90
ORDER BY t.created_at DESC;
```

沒有的資料（批號、不良數、使用條件）列進「待蒐集資料」，不要自己補。

## 報告格式

```markdown
# 8D 報告：〈產品型號〉〈問題一句話〉（〈工單號〉）

| D | 項目 | 內容 |
|---|---|---|
| D0 | 緊急應變 | 是否有安全疑慮？是否需立即通知客戶停用？ |
| D1 | 小組 | 品保（召集）、生產、研發、客服、業務 |
| D2 | 問題描述（5W2H） | What 現象／Where 發生位置與客戶產線／When 首次發生與出貨日／Who 客戶與發現者／Why 已知影響／How 怎麼發現／How many 數量與不良率 |
| D3 | 暫時對策（圍堵） | 廠內庫存、在途品、客戶端庫存的篩選與隔離；替代品安排 |
| D4 | 根本原因 | 發生原因與流出原因分開寫；5 Why；魚骨圖（人、機、料、法、環、測） |
| D5 | 永久對策 | 對應每個根本原因的對策，含負責人與日期 |
| D6 | 執行與驗證 | 用數據證明有效（例：改善後 3 批抽檢 0 不良） |
| D7 | 預防再發 | 更新 SOP、FMEA、管制計畫；水平展開到同系列產品與其他產線 |
| D8 | 結案 | 客戶確認、團隊表揚、經驗分享 |

## 待蒐集資料
- 〈清單〉
```

## 原則

- **事實與推測分開**：推測一律標「（假設，待驗證）」。D4 沒有證據前不能寫成定論。
- 對事不對人：原因寫流程或條件，不寫個人疏失。
- 要給客戶的版本：拿掉內部成本、人名、供應商名稱，只留現象、圍堵、對策與時程（語氣依 customer-reply）。
