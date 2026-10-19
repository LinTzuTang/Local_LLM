# 模型閘道（Model gateway）

Ollama 只跑文字／視覺 LLM 與 embedding（外加 Gemma 4 E2B／E4B 的音訊輸入）。語音轉文字、文件解析、reranker 這類模型改由本機的**模型閘道**（`src/local_llm/gateway/`，FastAPI）提供，使用方式比照 Ollama：

- 一鍵下載並顯示進度（`POST /api/pull`，NDJSON 串流，和 Ollama 相同格式）
- 第一次使用時才載入，閒置 `keep_alive` 秒（預設 300）後自動卸載並釋放記憶體
- `GET /api/tags` 看已下載的模型、`GET /api/ps` 看已載入的模型
- 推論 API 盡量沿用 OpenAI 的路徑

只監聽 `127.0.0.1:11500`，檔案不會離開這台 Mac。推論時伺服器處於離線模式（`HF_HUB_OFFLINE=1`），模型沒下載就回 404「請先 pull」，不會偷偷下載。

## 啟動

| 指令 | 用途 |
| --- | --- |
| `./scripts/tmux_up.sh` | 在 tmux session `local_llm` 開 `gateway` 視窗（11500 沒回應時才啟動） |
| `./scripts/start_gateway.sh` | 在目前的 terminal 執行（等同 `uv run python -m local_llm.gateway`） |
| `./scripts/restart_gateway.sh` | 改了 `src/local_llm/gateway/` 後重啟（已載入的模型會清空） |
| `./scripts/tmux_down.sh` | 連同平台、Ollama 一起停止 |

| 環境變數 | 預設 | 說明 |
| --- | --- | --- |
| `LOCAL_LLM_GATEWAY_PORT` | `11500` | 伺服器與 client 都會讀 |
| `LOCAL_LLM_GATEWAY_KEEP_ALIVE` | `300` | 秒；每個請求也可以帶 `keep_alive` 覆蓋（`0` 用完即卸載、`-1` 不卸載、也接受 `5m`） |
| `LOCAL_LLM_GATEWAY_MAX_UPLOAD_MB` | `200` | 上傳上限，超過回 413 |
| `LOCAL_LLM_GATEWAY_URL` | `http://127.0.0.1:11500` | 平台（Streamlit）連線用 |
| `LOCAL_LLM_OLLAMA_URL` | `http://127.0.0.1:11434` | 閘道轉送 Gemma 4 音訊用的 Ollama |

## 模型

| id | 任務 | 執行方式 | 磁碟 | 來源 |
| --- | --- | --- | --- | --- |
| `breeze-asr-25` | 語音轉文字 | mlx-whisper（Apple GPU） | 3.08 GB | 聯發科官方 `MediaTek-Research/Breeze-ASR-25` 的 safetensors，下載後在本機轉成 MLX（`gateway/convert.py`），轉完刪除原始檔 |
| `breeze-asr-26` | 語音轉文字（台語 → 華語漢字） | mlx-whisper | 3.1 GB（下載 6.2 GB） | 聯發科官方 `MediaTek-Research/Breeze-ASR-26`（BreezeASR-Taigi，Apache-2.0）的兩個 fp32 分片，下載後在本機轉成 MLX fp16，轉完刪除原始檔；輸出華語漢字，不是台語正字或台羅 |
| `taiwan-tongues-asr` | 語音轉文字（國語、台語、客語、英語） | mlx-whisper | 3.22 GB | 數位發展部 `adi-gov-tw/Taiwan-Tongues-ASR-CE-pretrained-v1.0` 的 safetensors，和 Breeze 一樣在本機轉 MLX |
| `whisper-large-v3-turbo` | 語音轉文字 | mlx-whisper | 1.61 GB | `mlx-community/whisper-large-v3-turbo` |
| `whisper-large-v3` | 語音轉文字 | mlx-whisper | 3.08 GB | OpenAI 官方 `openai/whisper-large-v3` 的 safetensors，和 Breeze 一樣在本機轉 MLX |
| `qwen3-asr-1.7b` | 語音轉文字 | mlx-audio（Apple GPU） | 4.70 GB | 通義千問官方 `Qwen/Qwen3-ASR-1.7B`（bf16，不轉檔） |
| `qwen3-asr-0.6b` | 語音轉文字 | mlx-audio | 1.88 GB | 官方 `Qwen/Qwen3-ASR-0.6B` |
| `sensevoice-small` | 語音轉文字 | mlx-audio | 0.94 GB | `mlx-community/SenseVoiceSmall`（官方 `FunAudioLLM/SenseVoiceSmall` 的 model.pt 轉 safetensors） |
| `fun-asr-nano` | 語音轉文字 | mlx-audio | 1.68 GB | `mlx-community/Fun-ASR-Nano-2512`（官方 `FunAudioLLM/Fun-ASR-Nano-2512` 轉檔） |
| `gemma4:e2b`、`gemma4:e4b` | 語音轉文字 | 轉送到 Ollama（MLX runner） | 7.48／9.46 GB | Ollama 的 `gemma4:e2b`／`gemma4:e4b`，就是「Ollama 模型」裡同一份權重 |
| `speaker-diarization` | 說話者分段（誰在什麼時候說話） | sherpa-onnx 1.13.8（onnxruntime，CPU，不佔 GPU） | 0.03 GB | pyannote segmentation-3.0 的 ONNX 轉檔 `csukuangfj/sherpa-onnx-pyannote-segmentation-3-0`（MIT，6 MB）＋ 3D-Speaker CAM++ 中英文進階版 `csukuangfj/speaker-embedding-models` 的 `3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx`（Apache-2.0，28 MB） |
| `zipformer-zh-en-stream` | 真串流語音辨識（中英） | sherpa-onnx OnlineRecognizer（CPU） | 0.2 GB | `csukuangfj/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20`（Apache-2.0），int8 編碼器／joiner＋fp32 解碼器 |
| `paraformer-zh-en-stream` | 真串流語音辨識（中英） | sherpa-onnx OnlineRecognizer（CPU） | 0.24 GB | `csukuangfj/sherpa-onnx-streaming-paraformer-bilingual-zh-en` int8（Apache-2.0；轉自 FunASR paraformer-zh-streaming，官方權重自 tag `apache-2.0-20260804` 起為 Apache-2.0） |
| `zipformer-zh-2025-stream` | 真串流語音辨識（中文） | sherpa-onnx OnlineRecognizer（CPU） | 0.17 GB | `csukuangfj/sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30`：**模型卡未標授權，只供本機測試，不建議商用** |
| `voxtral-mini-4b-realtime` | 真串流語音辨識（多語含中英） | mlx-audio `VoxtralStreamingSession`（Apple GPU） | 8.9 GB | `mlx-community/Voxtral-Mini-4B-Realtime-2602-fp16`（官方 Mistral 權重的 MLX 格式轉換，未量化，Apache-2.0） |
| `voxtral-mini-4b-realtime-4bit` | 真串流語音辨識（多語含中英） | 同上 | 3.1 GB | `mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit`（4-bit 量化，Apache-2.0）；本機 RTF 約 0.5，即時用途用這個 |
| `docling` | 文件解析 | Docling 標準流程：Heron 版面（MPS）＋TableFormer（CPU），OCR 用 macOS Vision（ocrmac，繁中） | 0.38 GB | `docling-project/docling-layout-heron`、`docling-project/docling-models`（v2.3.0） |
| `bge-reranker-v2-m3` | reranker | sentence-transformers CrossEncoder，MPS fp16 | 2.29 GB | `BAAI/bge-reranker-v2-m3` |

Hugging Face 來源都釘在固定 commit（`data/catalog.json` 的 `revision`）。

**說話者分段**：pyannote 官方的 `pyannote/segmentation-3.0` 要登入同意條款才能下載；sherpa-onnx 作者的 ONNX 轉檔不需要，授權同為 MIT（repo 內附 CNRS 的 LICENSE）。聲紋模型來自阿里巴巴通義實驗室 3D-Speaker（ModelScope `iic/speech_campplus_sv_zh_en_16k-common_advanced`，Apache-2.0）。分群門檻預設 0.7（catalog `options.threshold`）：0.5 在 88 秒會議錄音會多分出一位講者；知道人數時傳 `num_speakers` 最準。88 秒錄音在 CPU 上約 3 秒。

**Breeze-ASR-26**：模型卡與論文都沒有指定 Whisper 語言標記或提示（generation_config 的語言留空＝自動偵測）；平台和其他模型一樣傳 `language=zh`，評測另跑自動偵測對照（`scripts/speech_eval/run_breeze26.py`）。官方權重是兩個 fp32 分片，`convert.py` 依 `model.safetensors.index.json` 讀取。

**真串流**（`/v1/audio/stream`，`task: "stream"`）：`sherpa-online` 後端用 sherpa-onnx 的 OnlineRecognizer，每收到一小段音訊就解碼，句尾靜音 0.8 秒（`options.endpoint.rule2`）判定一句結束並定稿；`mlx-audio-realtime` 後端用 mlx-audio 的 Voxtral 串流 session（因果編碼器、解碼固定落後 480 ms），只往後加字。實作在 `gateway/stream_backends.py`，延遲與改字的計算在 `src/local_llm/asr_streaming.py`；麥克風即時字幕的 WebSocket 邏輯在 `gateway/live.py`（`LiveSession`），每個連線一條執行緒、整段期間佔用該模型。全精度 Voxtral（`voxtral-mini-4b-realtime`）在 M5 Pro 上 RTF 約 2.3、跟不上即時（記憶體頻寬瓶頸），catalog 標 `options.realtime: false`：「真串流」列表標示「跟不上即時」並顯示警告，「麥克風即時」不列出；即時請用 4-bit 版。mlx-audio 逐段解碼會讓 MLX 快取無限成長（88 秒錄音行程 footprint 達 40–48 GB），所以 Voxtral session 期間把 MLX 快取上限設為 `options.cache_limit_gb`（4 GB），結束後還原。Kyutai STT 需要 `moshi-mlx`，它要求 mlx 0.26 與 huggingface-hub 0.28，和本專案的 mlx 0.32／hub 1.x 衝突，所以沒有接進閘道。

**語音翻成英文**（`/v1/audio/translations`）：`whisper-large-v3` 用 Whisper 內建的 `task="translate"`；Gemma 4 E2B／E4B 改用「翻成英文」的指令。`whisper-large-v3-turbo` 訓練時拿掉了翻譯；Breeze-ASR-25 與 Taiwan-Tongues 實測收到翻譯指令仍輸出中文，三者在 catalog 標 `"translate": false`，呼叫時回 400。Qwen3-ASR、SenseVoice、Fun-ASR 也只會轉錄。

**Taiwan-Tongues-ASR-CE 的授權是 TRAIL – TWM Responsible AI License v0.1**（[LICENSE](https://huggingface.co/adi-gov-tw/Taiwan-Tongues-ASR-CE-pretrained-v1.0/blob/1180db44b24838d99f16ac8da093bd9238cf4f85/LICENSE)）：
- **可以：** 免費使用，包括商用。
- **不可以：** 違法、歧視、騷擾、冒充真人等用途。
- **標示：** 機器產生的內容須標示為機器產生。
- **散布轉換後的權重：** 修改或轉換後的模型屬衍生作品，若「散布」，就必須以 TRAIL-M 公開權重。這裡的「散布」包含提供下載、API 或遠端存取。
- **本機轉換的 MLX 權重：** `models/mlx/taiwan-tongues-asr` 自用沒問題，但不要散布或對外提供。

台語、客語也用 `zh` 語言代碼：詞表是標準 Whisper，沒有 `nan`／`hak` token，官方 API（`api/file_asr.py`）也一律用 `language="zh"`，輸出漢字。

另一個 repo `adi-gov-tw/Taiwan-Tongues-ASR-CE-v1.0` 是 CTranslate2（faster-whisper）格式，沒有加入閘道。實測條件：這台 Mac 只能用 CPU，int8、beam 5、VAD，和官方評估腳本相同。
- 速度：4 分 29 秒的錄音要 275 秒（約等於即時），MLX 轉檔只要約 20 秒。
- 準確率一樣差：CER 59%，錯字和重複迴圈與 MLX 轉檔相同。

MLX 轉檔正確：前 30 秒的輸出和 transformers（fp32、原始權重）逐字相同。也就是說，這個 pretrained checkpoint 用一般 Whisper 解碼時，本身就容易重複、幻覺。

授權：Breeze、Whisper（模型卡 Apache-2.0／MIT）、Qwen3-ASR、Fun-ASR-Nano、Gemma 4 都是 Apache-2.0 或 MIT；**SenseVoice-Small 的權重是 FunASR 模型授權 1.1（自訂、非 OSI）**：可使用、修改、散布，但須標示來源並保留模型名稱，阿里巴巴不負任何責任，條款可單方修改。平台在模型庫與比較頁都會標示，商用前請法務確認。

權重放在專案裡（同 `models/ollama`）：`models/hf`（`HF_HOME`）、`models/docling`、`models/mlx`；每個下載完成的模型在 `models/gateway/<id>.json` 有一份 manifest，列出檔案與來源 commit。manifest 存在且檔案齊全才算「已下載」，下載中斷不會被誤認。

實測（M5 Pro，88 秒會議錄音 `data/lab/samples/meeting_zh.wav`、3 頁合約 PDF）：

| 工作 | 時間 | 結果 |
| --- | --- | --- |
| Breeze-ASR-25 轉錄 | 約 6–8 秒（約 12× 即時） | 繁體中文，字錯率約 7%（多為同音字：伺服→四福、品保課→品寶客） |
| Whisper large-v3-turbo 轉錄 | 約 5–6 秒 | 輸出簡體中文；給繁體提示只改善一部分，結尾會幻覺出「請不吝點讚訂閱…」 |
| 十個語音模型並排比較 | 見下表 | 「模型比較 → 語音辨識模型」 |
| Docling 解析合約 | 第一次約 17 秒（含載入 13 秒），之後 1–4 秒 | 3 頁、4 個表格，Markdown 表格完整 |
| bge-reranker-v2-m3 | 第一次約 5 秒（含載入 4 秒），之後 < 1 秒 | 分數 0–1 |

### 語音辨識模型比較

兩段範例（`data/lab/samples/`，來源見該資料夾的 README.md）：
- **真人演說：** 總統 115 年國慶演說節錄，4 分 29 秒，照稿宣讀〔總統府提供〕。原稿是官方全文，依實際口述修改 4 處。
- **合成會議：** 88 秒，macOS 合成語音。

字錯率（CER）的算法：
- 對照原稿逐字計算（`src/local_llm/asr_metrics.py`）。
- 計算前先統一繁簡（OpenCC `s2tw`），把阿拉伯數字讀成中文（87% → 百分之八十七），並去掉標點與空白。

時間是模型已載入後的處理時間；即時倍率 = 錄音長度 ÷ 處理時間。

| 模型 | 演說 CER | 演說時間 | 會議 CER | 會議時間 | 輸出 |
| --- | --- | --- | --- | --- | --- |
| Qwen3-ASR-1.7B | **1.4%** | 8.7 秒（31×） | **3.2%** | 4.2 秒（21×） | 簡體；數字寫成中文 |
| Breeze-ASR-25 | 2.5% | 12.3 秒（22×） | 6.2% | 5.6 秒（16×） | **繁體**（台灣用字）；數字用阿拉伯數字 |
| Qwen3-ASR-0.6B | 2.7% | 4.3 秒（63×） | 5.4% | 1.9 秒（47×） | 簡體 |
| Whisper large-v3 | 3.1% | 12.0 秒（22×） | 5.4% | 5.5 秒（16×） | 簡體 |
| Whisper large-v3-turbo | 3.7% | 3.5 秒（76×） | 7.0% | 1.6 秒（56×） | 簡體 |
| Fun-ASR-Nano-2512 | 5.9% | 3.2 秒（83×） | 5.6% | 1.7 秒（53×） | 簡體 |
| SenseVoice-Small | 5.9% | 0.66 秒（407×） | 17.2% | 0.15 秒（589×） | 簡體；極快，合成語音漏字多 |
| Gemma 4 E4B（Ollama） | 11.0% | 6.7 秒（40×） | 12.6% | 3.0 秒（30×） | **繁體**（照指示）；會意譯，開頭「各位國人同胞」聽成「各位的領頭幫」 |
| Gemma 4 E2B（Ollama） | 21.2% | 6.4 秒（42×） | 13.9% | 2.5 秒（35×） | 繁體；同音字多 |
| Taiwan-Tongues-ASR-CE | 42.9% | 22.7 秒（12×） | 15.5% | 4.7 秒（19×） | 繁體；第一個 30 秒窗之後容易重複、離題（見上） |

真人錄音才看得出的問題，閘道已在 catalog 的 `options` 處理：

| 模型 | 問題 | 處理 | 演說 CER 前 → 後 |
| --- | --- | --- | --- |
| Whisper large-v3 | 長錄音會沿用前文而陷入重複，解碼失敗時提高 temperature 重試，所以每次結果不同 | `decode: {"condition_on_previous_text": false}` | 6–25% → 3.1%（也快 3–5 倍） |
| Whisper large-v3-turbo | 同上；在合成會議的結尾還會幻覺出多國語言亂碼 | 同上 | 7.2% → 3.7%（會議 9–60% → 7.0%） |
| Taiwan-Tongues-ASR-CE | 同上 | 同上 | 64% → 43% |
| Fun-ASR-Nano | 4 分半整段送入會一直重複 | `generate: {"chunk_duration": 60}` | 94% → 5.9%（會議 14.7% → 5.6%） |

Breeze-ASR-25 不受這些影響，沒有修改。

共同錯誤是同音字與專有名詞（伺服→四福／四伏、北辰→美神、品保課→品寶客），可在「上下文提示」填入人名與術語改善（Whisper、Qwen3-ASR、Fun-ASR、Gemma 會使用；SenseVoice 不支援）。

### Gemma 4 的音訊輸入（Ollama 0.40.3 實測）

| 模型 | Ollama 標示 | 實測 |
| --- | --- | --- |
| `gemma4:e2b`、`gemma4:e4b` | `audio` | 可用：原生 `/api/chat`（音訊 base64 放在 message 的 `images`）與 OpenAI 相容 `/v1/chat/completions`（`input_audio`）都接受；88 秒錄音一次送出約 3 秒、176 秒也可以 |
| `gemma4:12b` | `audio` | 接受音訊，但只聽懂第一句就開始重複（「目前进度符合计划…」不停）或寫出無關的文章，不適合當語音辨識，閘道不提供 |
| `gemma4:26b`、`gemma4:31b` | 無 `audio` | 回 400 `this model does not support audio input`（兩種 API 都一樣） |
| `nemotron3:33b` | 無 `audio`（ollama.com 只標 vision、tools、thinking） | 沒有下載 |

閘道的 `ollama-audio` 後端把錄音用 ffmpeg 轉成 16kHz 單聲道 WAV，附上「逐字轉錄、只輸出逐字稿」的指示（語言為 `zh` 時要求台灣繁體）送到 `/api/chat`，`think: false`、`temperature: 0`，並依錄音長度限制輸出 token 數（約每秒 12 個），避免模型重複時一路寫到 256K context；碰到上限會在回應標示 `truncated`。超過 4 分鐘的錄音在最安靜的地方切段。

## API

| 方法 | 路徑 | 內容 |
| --- | --- | --- |
| GET | `/api/version` | `{"version"}` |
| GET | `/api/tags` | `{"models": [{"id","task","installed","size_gb","loaded","name","license",...}]}` |
| GET | `/api/ps` | `{"models": [{"id","task","expires_at","device","loaded_at",...}]}` |
| POST | `/api/pull` | `{"model"}` → NDJSON：`{"status","total","completed"}`…，最後 `{"status":"success"}`；錯誤為 `{"error"}`。中途斷線即取消下載 |
| DELETE | `/api/delete` | `{"model"}`：卸載並刪除檔案 |
| POST | `/api/unload` | `{"model"}` |
| POST | `/v1/audio/transcriptions` | multipart：`file`、`model`、`language`（選填，`zh`／`en`…）、`prompt`（選填，上下文：人名、術語）、`keep_alive` → `{"text","segments":[{"start","end","text"}],"language","duration","model","seconds","load_seconds","queue_seconds"}`；Gemma 另有 `ollama_load_seconds`、`truncated` |
| POST | `/v1/audio/translations` | 同上的 multipart（`model` 預設 `whisper-large-v3`；`language` 是說話的語言）→ 同上的欄位，`text` 是英文 |
| POST | `/v1/audio/diarize` | multipart：`file`、`model=speaker-diarization`、`num_speakers`（選填，1–20；不填就估計）、`threshold`（選填，分群門檻，越低分出越多人）、`keep_alive` → `{"segments":[{"start","end","speaker"}],"speakers","duration","model","seconds","load_seconds"}`；講者依第一次出現的順序編號 0、1、… |
| POST | `/v1/audio/stream` | multipart：`file`、`model`（串流模型，預設 `zipformer-zh-en-stream`）、`frame_ms`（每次送出的音訊長度，20–2000，預設 160）、`pace`（`fast`＝以最快速度送、時間用量到的處理時間推算；`realtime`＝真的等每段音訊到齊）、`keep_alive` → NDJSON：文字每變一次一行 `{"t_audio","t_wall","text","final","seg"}`（`text` 是目前這句的假設；`final` 表示定稿），最後 `{"done":true,"duration","compute_s","frames","append_only","model","seconds","load_seconds"}`；中途斷線即停止解碼 |
| WebSocket | `/v1/audio/stream/ws?model=…&s2tw=0/1` | 麥克風即時字幕：用戶端送二進位 16 kHz 單聲道 PCM16（約 100 ms 一段），結束時送文字 `{"type":"stop"}`；伺服器先回 `{"type":"ready","append_only","load_seconds"}`，之後 `{"type":"partial"/"final","text","t_audio","t_wall","seg"[,"display"]}`、`{"type":"speech_start"/"speech_end","t_audio"}`（線上音量 VAD）、`{"type":"metric","kind":"first"/"final","utterance","latency"}`，最後 `{"type":"done","text","summary","first_latencies","final_latencies",…}` 後關閉；斷線即結束。只接受來自本機頁面（Origin 為 127.0.0.1／localhost）或沒有 Origin 的連線，其他回 403。`s2tw=1` 時另附繁體 `display`，`text` 保留模型原始輸出 |
| POST | `/v1/rerank` | `{"model","query","documents":[...],"top_n"}` → `{"model","results":[{"index","relevance_score","document"}],"seconds"}`（分數由高到低） |
| POST | `/v1/documents/parse` | multipart：`file`（PDF、DOCX、PPTX、HTML、MD、圖片）、`model=docling` → `{"markdown","pages","tables"（表格數）,"model","seconds"}` |

```sh
curl -N http://127.0.0.1:11500/api/pull -d '{"model": "breeze-asr-25"}'
curl http://127.0.0.1:11500/v1/audio/transcriptions -F file=@data/lab/samples/meeting_zh.wav \
  -F model=qwen3-asr-1.7b -F language=zh
curl http://127.0.0.1:11500/v1/audio/transcriptions -F file=@data/lab/samples/meeting_zh.wav \
  -F model=gemma4:e4b -F language=zh
curl http://127.0.0.1:11500/v1/audio/transcriptions -F file=@data/lab/samples/meeting_zh.wav -F language=zh
curl http://127.0.0.1:11500/v1/audio/translations -F file=@data/lab/samples/meeting_zh.wav \
  -F model=whisper-large-v3 -F language=zh
curl http://127.0.0.1:11500/v1/audio/diarize -F file=@data/lab/samples/meeting_zh.wav -F num_speakers=2
curl -N http://127.0.0.1:11500/v1/audio/stream -F file=@data/lab/samples/meeting_zh.wav \
  -F model=zipformer-zh-en-stream -F frame_ms=160
curl http://127.0.0.1:11500/v1/documents/parse -F file=@data/lab/samples/supplier_contract.pdf
curl http://127.0.0.1:11500/v1/rerank -H 'content-type: application/json' \
  -d '{"query": "逾期交貨罰多少？", "documents": ["每日 0.2% 違約金", "員工餐廳菜單"]}'
```

Python（`src/local_llm/gateway_client.py`）：

```python
from pathlib import Path
from local_llm import gateway_client as gw

gw.transcribe(Path("meeting.wav"), language="zh")["text"]
gw.translate(Path("meeting.wav"), "whisper-large-v3", "zh")["text"]  # 英文
gw.diarize(Path("meeting.wav"), num_speakers=2)["segments"]  # [{"start", "end", "speaker"}]
gw.stream_transcribe(Path("meeting.wav"), "zipformer-zh-en-stream")["events"]  # 真串流
gw.parse_document(Path("contract.pdf"))["markdown"]
gw.rerank("逾期罰款", ["段落一", "段落二"], top_n=1)  # [{"index", "relevance_score", "document"}]
for event in gw.pull("docling"):
    print(event)  # 錯誤時丟 RuntimeError
```

伺服器回錯誤時丟 `gw.GatewayError`（RuntimeError，含 `status`）；閘道沒啟動時丟 `httpx.ConnectError`，`gw.version()` 則回 `None`。

`queue_seconds`：MLX 的語音模型在同一個行程裡輪流用 GPU（`backends.MLX_LOCK`），場景平行跑多個語音模型時，等待別的模型的時間另外列出，不算進模型速度。管理 API（pull、delete、unload）不看 content type，`curl -d '{...}'` 直接可用。

## 平台與 agent

- 「語音模型比較」頁（`app/views/speech.py`，和語言模型的「模型比較」分開）有三種模式，都在背景執行、存成 `runs/asr_compare-*.json`（`mode` 欄位），可在「執行紀錄」查看：
  - **逐字稿**：同一段錄音（範例、上傳或麥克風錄音）依序交給選定的語音模型，並排顯示逐字稿、處理時間、即時倍率，有原稿時計算 CER 並標出差異。可加上**噪音與電話音質條件**（`src/local_llm/audio_aug.py`：白／粉紅／人聲嘈雜／辦公室噪音，指定 SNR；電話＝300–3400 Hz 帶通、8 kHz、μ-law），增強過的 WAV 快取在 `storage/speech_cache/`，結果是模型 × 條件的 CER 表與圖。
  - **翻成英文**：串接（語音模型 → Ollama LLM，TranslateGemma 用官方提示格式）、直接（Gemma 4 聽音訊寫英文）、Whisper 內建翻譯三種做法並排，列出首字時間、總時間，有參考譯文時算 chrF／BLEU（sacrebleu）；「模擬即時輸入」把錄音切成 N 秒片段依序處理，看英文落後講者多久。
  - **會議講者**：`/v1/audio/diarize` ＋語音辨識，合併成「講者 1：…」逐字稿與時間軸圖；有參考（RTTM 或「開始 結束 講者」）時算 DER。範例會議的參考時間軸與英文參考譯文在 `data/lab/speech_refs/`（本專案自製）。
  - **串流（即時）**：「切段模擬」把錄音切成 1–5 秒交給整段辨識模型；「真串流」用 `/v1/audio/stream` 把錄音以 100–320 ms 一段送給串流模型，畫面上看部分結果（灰字）變成定稿，結束後列出首字延遲、定稿延遲（中位數／p90）、RTF、改字比例與 CER／WER，可拖動重播；「麥克風即時」由瀏覽器元件（`app/static/live_mic/`，AudioWorklet 擷取、降頻成 16 kHz PCM16）經 WebSocket 直送閘道，即時顯示字幕與延遲，停止後連同錄音存成執行紀錄（`source: "mic"`，錄音在 `storage/speech_cache/mic/`）。
  - 很多模型輸出簡體字，頁面預設以 OpenCC `s2tw` 轉繁體顯示；存檔保留原始輸出。
- 場景「會議錄音 → 會議紀錄」（`meeting_minutes`）：先用 `diarize` 節點分出講者，`transcribe` 節點（`speakers` 參數）輸出標上講者的逐字稿，摘要與待辦事項因此能標出由誰提出。
- 場景「語音辨識模型比較」（`asr_model_comparison`）：四個語音模型平行轉錄，各自對照原稿算 CER，再由 LLM 比較錯誤類型並推薦。
- 「語音與文件」頁：語音轉文字、文件解析、rerank 試用（背景執行，顯示呼叫的 API 與 curl），以及 agent 的收件匣。
- 「模型庫 → 非 Ollama」：模型閘道的模型可下載、刪除、卸載；下載會出現在側邊欄的背景工作。
- Agent 工具（`src/local_llm/media_tools.py`）：`list_inbox_files`、`transcribe_audio`、`parse_document`，只能讀 `storage/inbox/` 裡的檔案（檔名檢查，防止路徑穿越）。「會議與文件助理」（`meeting_docs`）使用這些工具。第一次使用時收件匣會放入兩個範例檔；範例檔可用 `uv run python scripts/make_samples.py` 從原稿重建。

## 新增模型

**同一種執行方式**（另一個 MLX Whisper、另一個 CrossEncoder reranker）：在 `data/catalog.json` 對應的 `other_models` 項目加上 `gateway` 區塊，重啟閘道即可，平台頁面會自動列出。

```json
"gateway": {
  "id": "my-reranker",
  "task": "rerank",
  "backend": "cross-encoder",
  "weights": [{"repo": "org/name", "revision": "<commit>", "files": ["config.json", "model.safetensors", "tokenizer*"]}],
  "size_gb": 1.1,
  "options": {"max_length": 1024},
  "note": "說明（顯示在模型庫）"
}
```

| 欄位 | 說明 |
| --- | --- |
| `weights[].files` | 要下載的檔案（`allow_patterns`）；省略則整個 repo |
| `weights[].into` | 下載到 `models/` 下的資料夾；省略則放 Hugging Face 快取（`models/hf`） |
| `convert` | 下載後的轉檔步驟，例如 `hf-whisper-to-mlx`（見 `gateway/convert.py`） |
| `path` | 模型資料夾（`models/` 之下），轉檔輸出或多個 repo 的上層資料夾 |

**另一個 mlx-audio 支援的語音模型**（Parakeet、Voxtral、GLM-ASR、FireRedASR2…）：`backend` 用 `mlx-audio`，`options` 可設 `generate`（固定傳給 `generate()` 的參數）、`language_names`（語言傳 `Chinese` 而不是 `zh`）、`prompt_arg`（上下文提示要放進哪個參數）。

**Ollama 裡能聽音訊的 LLM**：`backend` 用 `ollama-audio`、加上 `"ollama": "<tag>"`，不需要 `weights`；下載走 `ollama pull`，刪除也會刪掉 Ollama 裡的模型。`options.chunk_seconds`（預設 240）與 `options.max_tokens_per_second`（預設 12）可調。

沒有接上的：Qwen3-ASR 官方的 `qwen-asr` 套件固定要 `transformers==4.57.6`，和專案的 transformers 5 衝突，所以改用 mlx-audio；NVIDIA Parakeet（mlx 可跑）不支援中文，沒有加入。

**新的執行方式或任務**：在 `gateway/backends.py` 寫一個 `Backend`（`load` 回傳 `(handle, device)`、`run` 回傳 JSON、必要時 `release`），加進 `BACKENDS`；新任務再到 `gateway/server.py` 加端點、`gateway_client.py` 加函式，並在 `tests/test_gateway.py` 用假 backend 測試。
