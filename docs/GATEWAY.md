# 模型閘道（Model gateway）

Ollama 只跑文字／視覺 LLM 與 embedding。語音轉文字、文件解析、reranker 這類模型改由本機的**模型閘道**（`src/local_llm/gateway/`，FastAPI）提供，使用方式比照 Ollama：

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

## 模型

| id | 任務 | 執行方式 | 磁碟 | 來源 |
| --- | --- | --- | --- | --- |
| `breeze-asr-25` | 語音轉文字 | mlx-whisper（Apple GPU） | 3.08 GB | 聯發科官方 `MediaTek-Research/Breeze-ASR-25` 的 safetensors，下載後在本機轉成 MLX（`gateway/convert.py`），轉完刪除原始檔 |
| `whisper-large-v3-turbo` | 語音轉文字 | mlx-whisper | 1.61 GB | `mlx-community/whisper-large-v3-turbo` |
| `docling` | 文件解析 | Docling 標準流程：Heron 版面（MPS）＋TableFormer（CPU），OCR 用 macOS Vision（ocrmac，繁中） | 0.38 GB | `docling-project/docling-layout-heron`、`docling-project/docling-models`（v2.3.0） |
| `bge-reranker-v2-m3` | reranker | sentence-transformers CrossEncoder，MPS fp16 | 2.29 GB | `BAAI/bge-reranker-v2-m3` |

權重放在專案裡（同 `models/ollama`）：`models/hf`（`HF_HOME`）、`models/docling`、`models/mlx`；每個下載完成的模型在 `models/gateway/<id>.json` 有一份 manifest，列出檔案與來源 commit。manifest 存在且檔案齊全才算「已下載」，下載中斷不會被誤認。

實測（M5 Pro，88 秒會議錄音 `data/lab/samples/meeting_zh.wav`、3 頁合約 PDF）：

| 工作 | 時間 | 結果 |
| --- | --- | --- |
| Breeze-ASR-25 轉錄 | 約 6–8 秒（約 12× 即時） | 繁體中文，字錯率約 7%（多為同音字：伺服→四福、品保課→品寶客） |
| Whisper large-v3-turbo 轉錄 | 約 5–6 秒 | 輸出簡體中文；給繁體提示只改善一部分，結尾會幻覺出「請不吝點讚訂閱…」 |
| Docling 解析合約 | 第一次約 17 秒（含載入 13 秒），之後 1–4 秒 | 3 頁、4 個表格，Markdown 表格完整 |
| bge-reranker-v2-m3 | 第一次約 5 秒（含載入 4 秒），之後 < 1 秒 | 分數 0–1 |

## API

| 方法 | 路徑 | 內容 |
| --- | --- | --- |
| GET | `/api/version` | `{"version"}` |
| GET | `/api/tags` | `{"models": [{"id","task","installed","size_gb","loaded","name","license",...}]}` |
| GET | `/api/ps` | `{"models": [{"id","task","expires_at","device","loaded_at",...}]}` |
| POST | `/api/pull` | `{"model"}` → NDJSON：`{"status","total","completed"}`…，最後 `{"status":"success"}`；錯誤為 `{"error"}`。中途斷線即取消下載 |
| DELETE | `/api/delete` | `{"model"}`：卸載並刪除檔案 |
| POST | `/api/unload` | `{"model"}` |
| POST | `/v1/audio/transcriptions` | multipart：`file`、`model`、`language`（選填，`zh`／`en`…）、`prompt`（選填）、`keep_alive` → `{"text","segments":[{"start","end","text"}],"language","duration","model","seconds"}` |
| POST | `/v1/rerank` | `{"model","query","documents":[...],"top_n"}` → `{"model","results":[{"index","relevance_score","document"}],"seconds"}`（分數由高到低） |
| POST | `/v1/documents/parse` | multipart：`file`（PDF、DOCX、PPTX、HTML、MD、圖片）、`model=docling` → `{"markdown","pages","tables"（表格數）,"model","seconds"}` |

```sh
curl -N http://127.0.0.1:11500/api/pull -d '{"model": "breeze-asr-25"}'
curl http://127.0.0.1:11500/v1/audio/transcriptions -F file=@data/lab/samples/meeting_zh.wav -F language=zh
curl http://127.0.0.1:11500/v1/documents/parse -F file=@data/lab/samples/supplier_contract.pdf
curl http://127.0.0.1:11500/v1/rerank -H 'content-type: application/json' \
  -d '{"query": "逾期交貨罰多少？", "documents": ["每日 0.2% 違約金", "員工餐廳菜單"]}'
```

Python（`src/local_llm/gateway_client.py`）：

```python
from pathlib import Path
from local_llm import gateway_client as gw

gw.transcribe(Path("meeting.wav"), language="zh")["text"]
gw.parse_document(Path("contract.pdf"))["markdown"]
gw.rerank("逾期罰款", ["段落一", "段落二"], top_n=1)  # [{"index", "relevance_score", "document"}]
for event in gw.pull("docling"):
    print(event)  # 錯誤時丟 RuntimeError
```

伺服器回錯誤時丟 `gw.GatewayError`（RuntimeError，含 `status`）；閘道沒啟動時丟 `httpx.ConnectError`，`gw.version()` 則回 `None`。

## 平台與 agent

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

**新的執行方式或任務**：在 `gateway/backends.py` 寫一個 `Backend`（`load` 回傳 `(handle, device)`、`run` 回傳 JSON、必要時 `release`），加進 `BACKENDS`；新任務再到 `gateway/server.py` 加端點、`gateway_client.py` 加函式，並在 `tests/test_gateway.py` 用假 backend 測試。
