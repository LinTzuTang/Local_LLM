# Local Agent Lab：地端 LLM／多 Agent 實驗平台（本機 Mac）

在這台 Mac 上用 Ollama 跑地端 LLM，搭配 LangChain 1.x／LangGraph agent、自建模型服務（語音、文件解析、reranker）與 PostgreSQL，提供 Streamlit 平台：**1. 測試各種地端 LLM 的表現　2. 展示多 Agent 協作場景**。所有服務只監聽 localhost，Ollama cloud 停用，資料不會送到雲端。介面可中英切換（右上角）。

[English README](README.md)

## 本機環境（2026-10-13 驗證）

| 項目 | 內容 |
| --- | --- |
| 硬體 | Apple M5 Pro、64GB 統一記憶體、macOS 27.0.1 |
| Python | 3.12（由 uv 管理，虛擬環境在 `.venv/`） |
| 套件 | `uv sync --locked` 依 `uv.lock` 安裝；LangChain 1.x、LangGraph、Streamlit、FastAPI（模型服務）、mlx-whisper、Docling、sentence-transformers |
| 推論引擎 | Ollama 0.40.3（`brew install ollama`，CLI 版，不會自動背景啟動）；模型服務（自建，port 11500） |
| 預設模型 | `qwen3:30b-a3b`（MoE，Q4_K_M，18GB，約 90 tokens/s）；`nimble:9b`（決策模型）、`bge-m3`（embedding）、`qwen3:1.7b`（smoke test）。都存在 `models/ollama/`；模型服務的權重在 `models/hf/` |
| 系統工具 | `brew install ollama postgresql@18 tmux ffmpeg` |

> **注意 `qwen3:30b-a3b` 一定會思考**：Ollama 上這個 tag 是 Thinking-2507 版本，回答前會先產生數百到上千 tokens 的思考。平台會自動偵測並把思考內容分開顯示。想要快速回應，可改用同大小的 `qwen3:30b-a3b-instruct-2507-q4_K_M`（不思考版）；也可以請「總控 Coding Agent」把場景裡的模型換掉。

## 每天開始開發

**一行啟動（建議）**：用 tmux 在背景執行，關掉 terminal 也不會停。

```sh
cd ~/Desktop/Local_LLM
./scripts/tmux_up.sh        # 資料庫＋Ollama（若尚未執行）＋模型服務＋平台 → http://127.0.0.1:8501
```

| 指令 | 用途 |
| --- | --- |
| `tmux attach -t local_llm` | 看平台、Ollama、模型服務的 log；按 `Ctrl-b` 再按 `d` 離開（不會停止） |
| `./scripts/restart_app.sh` | 改了程式後重新啟動平台 |
| `./scripts/restart_gateway.sh` | 重新啟動模型服務 |
| `./scripts/tmux_down.sh` | 停止平台、模型服務（以及在 tmux 裡啟動的 Ollama） |
| `./scripts/stop_db.sh` | 停止資料庫 |

`start_ollama.sh` 會設定 `OLLAMA_MODELS=models/ollama`、`OLLAMA_HOST=127.0.0.1:11434`、`OLLAMA_NO_CLOUD=1`，以及 `OLLAMA_NUM_PARALLEL=3`（讓三個部門 agent、多模型合議能同時跑）。如果 Ollama 已經在別的 terminal 執行，`tmux_up.sh` 會沿用它；要讓平行設定生效，需在那個 terminal 按 Ctrl+C 停掉，再執行一次 `./scripts/tmux_up.sh`。

**CLI 與程式開發（選用）**

```sh
code .                      # VS Code 已設定使用 .venv 與 ruff
uv run local-agent doctor   # 檢查 server 與已下載模型
./scripts/demo.sh           # chat + agent 工具測試，紀錄寫入 runs/
```

> 不要另外執行 `brew services start ollama` 或 Ollama.app：它們的模型存在 `~/.ollama/models`，與專案的 `models/ollama/` 不同，會造成「模型找不到」。

## 平台頁面

| 區塊 | 頁面 | 功能 |
| --- | --- | --- |
| 總覽 | 首頁 | 作品集首頁：兩大目的、精選場景流程圖、系統架構、即時狀態（模型、服務、記憶體） |
| | 模型庫 | 54 個 Ollama 模型，可依**級距**、**任務類型**、**系列**篩選（含台灣主權 LLM：TAIDE、Breeze、Taiwan-LLM、Twinkle；Cohere；決策模型 nimble）；「非 Ollama」分頁整理語音辨識、語音合成、文件解析／OCR、reranker，並可一鍵下載到模型服務 |
| | 導入路線圖 | 以平台示範情境為主的八個步驟，每步可直接前往該頁；完整版在 `docs/ADOPTION_ROADMAP.md` |
| ① 模型測試 | 對話 | 多輪對話、串流輸出、思考過程另外顯示、首字延遲與 tokens/s；視覺模型可附圖 |
| | 模型比較 | 同一題丟給最多 6 個模型並排比較；未下載的會先自動下載；有「台灣主權 LLM」「Cohere」快速組合；可選裁判模型評分 |
| | 語音與文件 | 模型服務的測試台：語音轉文字（Breeze-ASR-25／Whisper）、文件解析（Docling）、rerank、agent 用的檔案匣 |
| ② Agent 測試 | Agent | 對話、工具呼叫、公司資料庫（業務部／人資部／客服部）、研究助理（外部 API）、翻譯、會議與文件助理；逐步標示每次 LLM 呼叫與工具執行；外部工具可設定允許連網、人工核准與次數上限 |
| | 工具 | 所有工具是本機還是外部、送給模型的工具定義 JSON，並可直接試跑 |
| | 公司資料庫 | 預覽（含以某部門身分瀏覽）、上傳 CSV／Excel、權限設定（GRANT／REVOKE） |
| | 部門 Agent 比較 | 同一問題同時問三個部門 agent（同模型、只差資料庫權限），並排比較 |
| ③ 多 Agent 場景 | 場景展示 | 9 個場景（見下表），即時流程圖＋時間軸＋每一步細節；可直接編輯場景 JSON（即時驗證、diff） |
| | 知識庫（RAG） | 文件與部門權限、重建索引、**檢索比較**（BM25／向量／RRF／Rerank 並排） |
| | 總控 Coding Agent | 用自然語言修改所有 agent 與場景的設定與串接；每次寫入顯示 diff 與驗證結果，核准後才存檔，可一鍵復原 |
| 紀錄 | 執行紀錄 | 所有比較、agent、場景、總控的執行結果（`runs/*.json`） |

對話、Agent、比較、場景、下載都在背景執行：**切換頁面不會中斷**，側欄「背景工作」可看進度。

## 多 Agent 場景

| 場景 | 流程 | 模型 |
| --- | --- | --- |
| 公司問答路由 | nimble 路由 → 業務／人資／客服資料庫 agent 或一般 LLM | nimble:9b、qwen3:30b-a3b |
| 混合檢索 RAG | [BM25 ‖ 向量] → RRF → Rerank → 引用回答 | bge-m3、bge-reranker-v2-m3、qwen3 |
| 自我檢查 RAG | 改寫查詢 → 混合檢索 → 回答 → nimble 依據檢查 → 通過／保守回覆 | ＋nimble |
| 權限感知知識助理 | 依提問者部門過濾檢索（人資查得到薪資制度，業務查不到） | bge-m3、reranker、qwen3 |
| 會議錄音 → 會議紀錄 | Breeze-ASR-25 → [摘要 ‖ 待辦] → 紀錄 | breeze-asr-25、qwen3 |
| 合約審閱 | Docling 解析 PDF（含表格）→ [關鍵條款 ‖ 風險條款] → 報告 | docling、qwen3 |
| 客服工單分流 | [nimble 分類（團隊、緊急度、退款）‖ SOP 檢索] → 草稿 → 主管校稿 | nimble、bge-m3、reranker、qwen3 |
| 多模型合議 | [Qwen3 30B ‖ TAIDE 12B ‖ Qwen3 1.7B] → 彙整者 | 三個地端模型 |
| 業務週報 | 資料庫分析 agent → 撰寫 → 翻譯 | qwen3 |

場景格式、節點類型與如何新增見 [docs/SCENARIOS.md](docs/SCENARIOS.md)。

## 模型服務（非 Ollama 模型）

Ollama 只跑文字／視覺 LLM 與 embedding。語音辨識、文件解析、reranker 由自建的模型服務（FastAPI，`127.0.0.1:11500`）提供，用法比照 Ollama：一鍵下載（有進度）、用到才載入、閒置自動卸載、`/api/tags`、`/api/ps`，推論端點採 OpenAI 風格路徑（`/v1/audio/transcriptions`、`/v1/rerank`、`/v1/documents/parse`）。詳見 [docs/GATEWAY.md](docs/GATEWAY.md)。

## 外部工具與控制

研究助理 agent 有三個外部工具：`get_weather`（Open-Meteo）、`wikipedia_search`（中文維基百科）、`fetch_webpage`（抓網頁正文）。呼叫它們會把參數送出這台電腦，因此由程式強制控制：

| 控制 | 做法 |
| --- | --- |
| 允許連網（預設關閉） | 關閉時外部工具不會交給模型，模型根本看不到 |
| 人工核准（預設開啟） | `HumanInTheLoopMiddleware`：執行前暫停，顯示參數，可核准、修改或拒絕 |
| 次數上限（預設 3） | `ToolCallLimitMiddleware`：每個外部工具每次任務的上限 |
| 內網防護 | `fetch_webpage` 擋掉 localhost、私有網段、雲端 metadata 位址 |

總控 Coding Agent 的寫入工具也一律經 `HumanInTheLoopMiddleware` 核准，並由 `lab/store.py` 驗證與保留版本（`storage/lab/history/`）。

## 台灣主權 LLM 與 Cohere

台灣模型不在 Ollama 官方庫，但 Ollama 可以直接拉 Hugging Face 上的 GGUF（`ollama pull hf.co/<repo>:Q4_K_M`），平台的下載按鈕也用這個方式。

| 系列 | 代表模型 | 授權 | 注意 |
| --- | --- | --- | --- |
| TAIDE（國科會） | Gemma-3-TAIDE-12b-Chat-2602、Llama-3.1-TAIDE-LX-8B | TAIDE 自訂授權條款 | 社群 GGUF 轉檔；不支援工具呼叫 |
| Breeze（聯發科） | Llama-Breeze2-8B／3B、Breeze-7B；語音 Breeze-ASR-25 | Llama 3.2／Apache-2.0 | Breeze 2 為社群純文字轉檔；Breeze-ASR-25 在模型服務 |
| Taiwan-LLM | Llama-3.1-Taiwan-8B、Llama-3-Taiwan-8B／70B | Llama 3 系列 | 70B 只能用 Q3 量化，吃緊 |
| Twinkle AI | gemma-3-4B-T1、Llama-3.2-Taiwan-3B | Gemma／Llama 3.2 | 小模型 |
| Cohere | Command R（35B）、Command R7B、Aya Expanse、North Mini Code | **多為 CC-BY-NC-4.0（非商用）**；North Mini Code 為 Apache-2.0 | Command A（111B）放不下 |

## 模擬公司

`示範精密工業股份有限公司（虛構）`：

- **資料庫**：約 300 名員工、400 家客戶、54 項產品、4,200 張訂單、950 張客服工單、1,600 筆請假紀錄，分成 `hr`、`sales`、`inventory`、`support` 四個 schema（Faker zh_TW 產生，人名與公司都是虛構的）。
- **知識庫**：`data/lab/corpus/company_kb/`，8 份內部規章（請假、差旅、資安與生成式 AI、薪資（僅人資）、業務折扣（僅業務）、客服 SOP、產品 FAQ、8D 程序），`meta.json` 標示可讀部門。
- **範例檔**：`data/lab/samples/` 的會議錄音與供應商合約 PDF（附可重製的原始文字）。

| 資料庫角色 | 能讀 | 不能讀 |
| --- | --- | --- |
| `agent_sales`（業務部） | sales、inventory、support；員工基本資料 | 薪資、電話、請假紀錄 |
| `agent_hr`（人資部） | hr 全部（含薪資） | 業務、庫存、客服 |
| `agent_support`（客服部） | support；客戶、產品、庫存的部分欄位 | 產品成本、信用額度、訂單 |

所有角色都只有 SELECT 權限，交易預設唯讀，查詢 5 秒逾時。上傳的資料表放在 `uploads` schema，預設沒有任何部門可讀。密碼在第一次啟動時隨機產生並寫入 `.env`。重建資料：`uv run python -m local_llm.company_db init`。

## 範例

```sh
uv run local-agent chat --prompt '用三點說明地端 LLM 的優點'
uv run local-agent agent --prompt 'Use calculator to compute 37 * 19 + 8.' --require-tool --trace-file runs/my-test.json
```

```python
# 用程式跑一個場景（與平台相同的引擎）
from local_llm.lab import store
from local_llm.lab.engine import Run

run = Run(
    store.scenario("hybrid_rag"), {"question": "特休可以遞延到什麼時候？"}, "http://127.0.0.1:11434"
)
run.execute()
print(run.output())
for step in run.trace:
    print(step["node"], step["models"], round(step["end"] - step["start"], 1))
```

```sh
# nimble 決策模型
curl http://127.0.0.1:11434/v1/systemone -d '{"model": "nimble:9b",
  "state": {"text": "各部門的平均月薪"},
  "questions": {"route": {"type": "choice", "instructions": "Which department?",
                "criteria": {"sales": "Sales data", "hr": "HR data", "support": "Support data"}}}}'
```

## 設定（`.env`）

| 變數 | 預設 | 說明 |
| --- | --- | --- |
| `LOCAL_LLM_PROVIDER` | `ollama` | `ollama` 或 `vllm`（CLI） |
| `LOCAL_LLM_MODEL` | `qwen3:30b-a3b` | 模型名稱 |
| `LOCAL_LLM_BASE_URL` | `http://127.0.0.1:11434` | Endpoint |
| `LOCAL_LLM_CONTEXT` | `16384` | Context 長度 |
| `LOCAL_LLM_MAX_TOKENS` | `4096` | 單次輸出上限；會思考的模型需要較大值 |
| `LOCAL_LLM_TIMEOUT` | `180` | 秒 |
| `LOCAL_LLM_QWEN_NO_THINK` | `true` | 模型允許時關閉思考 |
| `LOCAL_LLM_GPU_BUDGET_GB` | `44` | 平台載入多個模型時的記憶體警告門檻 |
| `LOCAL_LLM_GATEWAY_KEEP_ALIVE` | `300` | 模型服務閒置多少秒後卸載模型 |
| `LOCAL_LLM_PG_*_PASSWORD` | 自動產生 | 資料庫密碼，由 `start_db.sh` 寫入 |

## 開發指令與架構

```sh
uv run pytest -q                # 測試（不需要模型）
uv run ruff check src tests app
uv run ruff format src tests app
```

```text
Streamlit 平台（app/）：main.py 導覽與語言切換、i18n.py、jobs.py 背景工作、flow_ui.py 流程圖與時間軸
    → 對話、比較：streaming.py → Ollama /api/chat
    → Agent：agents.py（讀 data/lab/agents.json）→ LangChain create_agent
    → 場景：lab/engine.py（讀 data/lab/scenarios.json）→ LangGraph StateGraph
        → 節點：lab/nodes.py（llm、agent、router、classify、BM25、向量、RRF、rerank、轉錄、文件解析、組合）
        → lab/rag.py 知識庫、lab/decide.py nimble、gateway_client.py 模型服務
    → 總控：lab/control.py（讀寫工具＋核准）→ lab/store.py（驗證、版本、復原）
    → 模型管理：ollama_admin.py、gateway/（FastAPI）
```

## 限制

- 單人、localhost 使用，沒有登入；權限依「選哪個 agent／部門」而非「誰登入」。
- PaddlePaddle 系列（PaddleOCR-VL、PP-StructureV3）在 Mac 沒有 GPU 加速，列在目錄中，建議放 DGX Spark（見 `docs/DGX_SPARK.md`）。
- 小模型（≤2B）適合展示機制，不適合評估答案品質。

## 疑難排解

| 狀況 | 處理 |
| --- | --- |
| 連不到 Ollama | `./scripts/tmux_up.sh` 或 `./scripts/start_ollama.sh` |
| 模型找不到 | 可能是別的 Ollama（app 或 brew services）佔用 11434 port；停掉後重新執行專案腳本 |
| 場景說模型服務沒有回應 | `./scripts/start_gateway.sh`（或 `restart_gateway.sh`） |
| 場景說知識庫沒有索引 | 到「知識庫」頁按「重建索引」 |
| 只有思考、沒有答案 | 輸出上限太小；調高「最大輸出 tokens」或 `LOCAL_LLM_MAX_TOKENS` |
| 改了 `src/` 但畫面沒變 | `./scripts/restart_app.sh` |
| 從零重建環境 | `rm -rf .venv && ./scripts/setup.sh`；模型遺失時執行 `./scripts/pull_model.sh` |

參考：[Streamlit](https://docs.streamlit.io/)、[LangGraph](https://docs.langchain.com/oss/python/langgraph/overview)、[Ollama API](https://github.com/ollama/ollama/blob/main/docs/api.md)、[Ollama decision models](https://ollama.com/blog/ollama-now-supports-jev-style-decision-models)。
