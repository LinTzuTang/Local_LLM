# 範例檔 / Samples

| 檔案 | 內容 | 來源與授權 |
| --- | --- | --- |
| `meeting_zh.wav` / `meeting_zh.txt` | 88 秒虛構產銷會議，macOS 合成語音（兩位講者） | 本專案自製：`uv run python scripts/make_samples.py` 由 `meeting_zh.txt` 重建 |
| `supplier_contract.pdf` / `.html` | 虛構供應合約 | 本專案自製（同上） |
| `national_day_2026_zh.wav` / `national_day_2026_zh.txt` | 真人錄音：賴清德總統 115 年國慶演說節錄，4 分 29 秒，照稿宣讀 | 見下 |
| `national_day_2026_en.txt` | 上一列同一段落的**官方英譯**（翻譯評測的參考譯文） | 見下 |
| `alimeeting_r8001_zh.flac` / `alimeeting_r8001_zh.txt` | 真實 4 人會議（AliMeeting），5 分鐘，遠場麥克風，大陸口音；講者時間在 `data/lab/speech_refs/alimeeting_r8001_zh.rttm` | 見下（CC BY-SA 4.0） |
| `palau_hsiao_2026_en.wav` / `palau_hsiao_2026_en.txt` | 真人錄音：蕭美琴副總統在帛琉的英語致詞，4 分 59 秒（台灣口音英語） | 見下 |
| `librispeech_20_en.flac` / `librispeech_20_en.txt` | 英文有聲書：LibriSpeech test-clean 20 句接成一段，3 分 6 秒（句間 0.8 秒靜音） | 見下（CC BY 4.0） |
| （本機）`storage/speech_samples/hokkien_moe_20_nan.flac` | 台語：教育部台語辭典例句 20 句接成一段，1 分 47 秒；**不在 repo**，用 `scripts/make_speech_samples.py` 在本機產生 | 見下（CC BY-ND 3.0 TW） |

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
- **用途**：中文語音 → 英文翻譯的參考譯文（`scripts/speech_eval/run_translation.py`）。官方英譯是書面翻譯，
  不是逐句口譯，有些句子重組或意譯，所以任何系統的 BLEU 都不會很高。
- 重建：`uv run --group eval python scripts/speech_eval/prepare_data.py national_day_en`。

## alimeeting_r8001_zh：AliMeeting 真實會議節錄

- **來源**：AliMeeting 中文會議語料（阿里巴巴達摩院；M2MeT 挑戰賽），OpenSLR 119：<https://www.openslr.org/119/>。
  取評測集 `Eval_Ali` 的 `R8001_M8004` 遠場 8 聲道錄音的**第 1 聲道**、前 295 秒（切在沒人說話處），轉成 16 kHz 單聲道 FLAC。
- **參考答案**：資料集附的 TextGrid（逐人時間與文字），依開始時間串接成 `alimeeting_r8001_zh.txt`；
  講者時間轉成 RTTM（`data/lab/speech_refs/alimeeting_r8001_zh.rttm`，講者改名為 講者1–4）。多人同時說話時順序不唯一，整段 CER 會略為偏高。
- **授權**：CC BY-SA 4.0。使用請標示「AliMeeting corpus (Yu et al., ICASSP 2022), OpenSLR 119」。
  本節錄只剪輯與轉檔，依相同授權（CC BY-SA 4.0）提供。

## palau_hsiao_2026_en：蕭美琴副總統英語致詞

- **音訊**：總統府 YouTube 官方頻道〈20260607「帛榮專案」副總統出席帛琉惠恕仁總統歡迎晚宴〉
  <https://www.youtube.com/watch?v=2MkeuIZBhik>，取 0:00–4:58.8（整段致詞，略去最後未列入講稿的 “Thank you again.”），
  轉成 16 kHz 單聲道 WAV。影片說明同樣是「◎版權屬總統府所有，使用請加註〔總統府提供〕」。
- **文字**：總統府英文新聞〈Vice President Hsiao attends welcome banquet hosted by President Surangel S. Whipps Jr. of Palau〉
  <https://english.president.gov.tw/NEWS/7155>（2026-06-07）所附致詞逐字稿。
- **授權**：同上（政府資料開放授權條款第 1 版），標示 **〔總統府提供〕**；本節錄僅剪輯與轉檔，不代表總統府認可本專案。
- **參考逐字稿的修改**：官方逐字稿把口語稍微書面化。只修改 Whisper large-v3、Qwen3-ASR-1.7B、Breeze-ASR-25
  三個不同系列的模型**都**聽成同一說法、且與稿不同的地方，共 3 處（英文影片沒有英文字幕可以對照，這是比國慶演說弱的地方）：

  | 講稿 | 實際口述 |
  | --- | --- |
  | But I think it is important | But I think **it's** important |
  | the Pacific Ocean, does not separate us | the Pacific Ocean, **doesn't** separate us |
  | hospitality that continue to bring us together and demonstrate that | hospitality that **continues** to bring us together and **demonstrates** that |

  人名、地名維持官方拼法（Whipps、Peleliu、Palauan），模型拼錯（Wipps、Weeps、Palawan）就算錯。
- 重建：`uv run --group eval python scripts/speech_eval/prepare_data.py palau`（需要 `yt-dlp`）。

## librispeech_20_en：LibriSpeech 英文有聲書 20 句

- **來源**：LibriSpeech ASR corpus（Panayotov, Chen, Povey, Khudanpur, ICASSP 2015），OpenSLR 12：<https://www.openslr.org/12/>；
  錄音來自 LibriVox 公有領域有聲書。取 `test-clean` 裡 `docs/SPEECH_EVALUATION.md` 英文評測實際用的 **20 句**
  （`storage/speech_eval/results/asr_en_librispeech.json` 的順序），句間插入 0.8 秒靜音，轉成 16 kHz 單聲道 FLAC。
- **參考答案**：資料集附的逐句文字（全小寫、無標點），一句一行；開頭的 `#` 行列出每句的 ID 與在檔案中的起訖秒數。
- **授權**：CC BY 4.0（<https://creativecommons.org/licenses/by/4.0/>），可再散布，須標示出處：
  「LibriSpeech ASR corpus (Panayotov et al., ICASSP 2015), OpenSLR 12, CC BY 4.0」。本檔只接合與轉檔，文字未改。
- 重建：`PYTHONPATH=src .venv/bin/python scripts/make_speech_samples.py librispeech`。

## hokkien_moe_20_nan：台語辭典例句 20 句（只在本機）

- **來源**：教育部《臺灣台語常用詞辭典》<https://sutian.moe.edu.tw/> 的例句錄音與漢字，取台語評測實際用的 20 句
  （`storage/speech_eval/results/asr_hokkien_moe.json` 的順序），句間 0.8 秒靜音。參考答案是辭典的**漢字**一句一行。
- **授權**：CC BY-ND 3.0 TW（姓名標示－禁止改作）。把 20 句接成一個檔就是改作，所以**不放進 repo**：
  `scripts/make_speech_samples.py hokkien` 由 `storage/speech_eval/raw/moe_sutian/` 的原始錄音產生到
  `storage/speech_samples/`（不進版控），只供本機測試，請勿轉散布。檔案存在時，語音比較頁才會列出這個範例。
- **語言提示**：這些模型沒有台語選項，逐字稿模式用最接近的「中文（zh）」提示；模型多半寫成華語，所以字錯率會很高，
  這正是要看的地方（哪個模型最接近台語漢字）。

## librispeech_20_en / hokkien_moe_20_nan (English)

- **LibriSpeech**: the 20 test-clean utterances the English evaluation scored, joined with 0.8 s of silence (16 kHz mono FLAC). Source: <https://www.openslr.org/12/>; licence CC BY 4.0, attribution "LibriSpeech ASR corpus (Panayotov et al., ICASSP 2015), OpenSLR 12". Joined and resampled only.
- **Taiwanese Hokkien**: 20 example sentences of the MOE Dictionary of Frequently-Used Taiwan Taigi, with the 漢字 as reference. The licence is CC BY-ND 3.0 TW, and a joined file is a derivative, so it is **not in the repo**. `scripts/make_speech_samples.py` builds it locally in `storage/speech_samples/` (gitignored). Do not redistribute it.

## 評測用、但**不放進 repo** 的資料

`docs/SPEECH_EVALUATION.md` 另外用了下列資料，只下載到 `storage/speech_eval/`（不進版控），
用 `scripts/speech_eval/prepare_data.py` 重建：

| 資料 | 授權 | 不放進 repo 的原因 |
| --- | --- | --- |
| Common Voice nan-tw／en（經數位發展部 `adi-gov-tw/Taiwan-Tongues-ASR-CE-dataset-*` 測試集） | 錄音 CC0；打包 TRAIL-D | Mozilla Data Collective 的使用條件禁止重新散布、禁止辨識講者 |
| 教育部《臺灣台語常用詞辭典》例句與錄音 | CC BY-ND 3.0 TW | 禁止改作；只在本機評測，不轉散布（20 句的接合檔也只在本機，見上） |
| LibriSpeech test-clean | CC BY 4.0 | 太大，評測只抽 120 句（其中 20 句接成 `librispeech_20_en.flac` 放進 repo） |
| AliMeeting Eval（OpenSLR 119） | CC BY-SA 4.0 | 3.7 GB；中國口音，只用來測說話者分段 |

