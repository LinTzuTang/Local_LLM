# 範例檔 / Samples

| 檔案 | 內容 | 來源與授權 |
| --- | --- | --- |
| `meeting_zh.wav` / `meeting_zh.txt` | 88 秒虛構產銷會議，macOS 合成語音（兩位講者） | 本專案自製：`uv run python scripts/make_samples.py` 由 `meeting_zh.txt` 重建 |
| `supplier_contract.pdf` / `.html` | 虛構供應合約 | 本專案自製（同上） |
| `national_day_2026_zh.wav` / `national_day_2026_zh.txt` | 真人錄音：賴清德總統 115 年國慶演說節錄，4 分 29 秒，照稿宣讀 | 見下 |
| `national_day_2026_en.txt` | 上一列同一段落的**官方英譯**（翻譯評測的參考譯文） | 見下 |

## national_day_2026_zh：總統府國慶演說節錄

- **音訊**：總統府 YouTube 官方頻道 `@presidentialoffice`，〈20261010 總統出席「中華民國中樞暨各界慶祝115年國慶大會」〉
  <https://www.youtube.com/watch?v=ozVNA0bl5uU>（2026-10-10 上傳，25:00）。
  取 07:13.8–11:42.7（「各位國人同胞，山巔之城……」到「……也願意留下來。」），轉成 16 kHz 單聲道 WAV。
  下載：`yt-dlp -f bestaudio -x --audio-format wav`（2026-10-15）；剪輯：
  `ffmpeg -ss 433.8 -to 702.7 -i <下載檔> -ac 1 -ar 16000 -c:a pcm_s16le national_day_2026_zh.wav`。
- **文字**：總統府新聞〈總統發表國慶演說「臺灣 山巔之城，地上的鹽，世上的光」〉演說全文的同一段落
  <https://www.president.gov.tw/NEWS/40414>（115 年 10 月 10 日）。
- **授權**：
  - 影片說明寫明「◎版權屬總統府所有，使用請加註〔總統府提供〕」，也就是註明出處即可使用，不是「須經同意方可使用」的特別聲明。
  - 總統府網站〈政府網站資料開放宣告〉（<https://www.president.gov.tw/Page/18>）宣告網站資料採
    [政府資料開放授權條款-第1版](https://data.gov.tw/license)：可重製、改作、公開傳輸，含商業利用，須註明出處；
    只有「經機關特別聲明須經同意方可使用」的影音不在此限，本片沒有這種聲明。
  - 使用時請標示：**音訊與文字〔總統府提供〕**，來源同上。
  - 本節錄僅剪輯與轉檔，未變更內容；不代表總統府認可本專案。
- **參考逐字稿的修改**：以官方全文為準，再對照 YouTube 上總統府提供的人工字幕（zh-TW）與 Qwen3-ASR-1.7B、Whisper large-v3 的轉錄。
  只修改講者確實與講稿不同、而且字幕與兩個模型一致的地方，共 4 處：

  | 講稿 | 實際口述 |
  | --- | --- |
  | 不能只看產值與出口 | 不能**夠**只看產值與出口 |
  | 都能站得穩、過得好 | 都能**夠**站得穩、過得好 |
  | 都市更新分回及容積獎勵 | 都市更新分回**以**及容積獎勵 |
  | 安居也才能讓年輕人勇敢成家 | 安居也才能**夠**讓年輕人勇敢成家 |

  標點保留官方全文的寫法（計算字錯率時會去掉標點）。

## national_day_2026_zh (English)

- **What:** a real, read-from-script recording. It is a 4 min 29 s excerpt of President Lai's 2026 National Day address.
- **Source:**
  - Audio from the Office of the President's official YouTube channel: <https://www.youtube.com/watch?v=ozVNA0bl5uU>, 07:13.8–11:42.7.
  - Text from the official full text: <https://www.president.gov.tw/NEWS/40414>.
- **Licence:**
  - The video description says "Copyright belongs to the Office of the President; credit 〔總統府提供〕 when using it".
  - The site's open-data declaration (<https://www.president.gov.tw/Page/18>) puts site content under the Open Government Data License, version 1.0. That licence allows reuse, including commercial reuse, with attribution. The only exception is items specially marked as needing permission, and this video is not marked.
  - Attribution: **audio and text courtesy of the Office of the President (總統府提供)**.
- **Reference edits:** the reference follows the official text, with the four edits above where the speaker deviated from the script. Each edit is confirmed by the official captions and by two models.

## national_day_2026_en：官方英譯

- **文字**：總統府英文網站〈President Lai delivers 2026 National Day Address〉
  <https://english.president.gov.tw/NEWS/7226>（2026-10-10），取與 `national_day_2026_zh.txt` 同一段落
  （“My fellow citizens, this city upon a hill…” 到 “…but willing to stay.”），未修改。
- **授權**：英文網站〈Government Website Open Information Announcement〉（<https://english.president.gov.tw/Page/18>）同樣採
  政府資料開放授權條款第 1 版，須註明出處：**〔總統府提供〕**。
- **用途**：中文語音 → 英文翻譯的參考譯文（「語音模型比較」頁的翻譯模式）。官方英譯是書面翻譯，
  不是逐句口譯，有些句子重組或意譯，所以任何系統的 BLEU 都不會很高。
