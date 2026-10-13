---
name: meeting-minutes
description: Turn a meeting transcript or rough notes into structured meeting minutes in Taiwanese business format - summary, decisions, action items with owner and due date, open questions. Use after transcribe_audio, or when the user pastes meeting notes or asks for 會議記錄.
description_zh: 把會議逐字稿或筆記整理成台灣企業常用格式的會議記錄：摘要、決議、待辦（負責人與期限）、待確認事項。在 transcribe_audio 之後，或使用者貼上會議筆記、要求整理會議記錄時使用。
source: local
license: internal
---

# 會議記錄

## 原則

- **只寫逐字稿／筆記裡有的內容。** 沒提到的出席者、日期、負責人不要補；不確定就寫「（待確認）」。
- 語音轉文字可能聽錯產品型號、人名、數字：與上下文不符時保留原文並加註「（疑似辨識錯誤）」。
- 相對日期（「下週三」「月底前」）換算成實際日期；不知道今天日期就先呼叫時間工具（`current_time` 或 `mcp_time_get_current_time`）。
- 數字、單位、型號照原文，不要四捨五入。
- 去掉口語贅字（嗯、那個、就是說），但不改變意思。
- 人事、薪資、客訴賠償等敏感內容：照記，但在最後加「⚠️ 含敏感資訊，請限縮發送範圍」。

## 格式

```markdown
# 會議記錄：〈主題〉

- 日期／時間：2026 年 10 月 13 日（二）14:00–15:00
- 地點／形式：〈會議室或線上〉
- 主持人：〈 〉　記錄：AI 草稿（待人工確認）
- 出席：〈依逐字稿出現的人名／職稱〉

## 一、會議摘要
1. 〈3–5 點，每點一句話，先講結論〉

## 二、決議事項
| # | 決議 | 備註 |
|---|---|---|
| 1 | 〈已拍板的事〉 | |

## 三、待辦事項
| # | 事項 | 負責人 | 期限 | 狀態 |
|---|---|---|---|---|
| 1 | 〈動詞開頭，可驗收〉 | 〈人名或單位〉 | 2026/10/20 | 未開始 |

## 四、待確認／未決事項
- 〈討論了但沒有結論的事，寫明要誰確認〉

## 五、下次會議
- 〈時間與議題；沒有提到就寫「未訂」〉
```

## 決議 vs 待辦 vs 討論

- **決議**：會中明確同意的事（「就這樣決定」「OK 照這個做」）。
- **待辦**：有人要去做的具體動作；每一筆都要有負責人與期限，沒講就寫「（待指派）」「（待訂）」。
- 只是討論、意見分歧 → 放「待確認」，不要寫成決議。

## 存檔（有檔案工具時）

若可以使用 `mcp_files_write_file`，把定稿存成 `workspace/YYYY-MM-DD-會議記錄-〈主題〉.md`，並在回答中告訴使用者檔名。
