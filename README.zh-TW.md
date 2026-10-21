# Local Agent Lab：地端 LLM／多 Agent 實驗平台（本機 Mac）

在這台 Mac 上用 Ollama 跑地端 LLM，搭配 LangChain 1.x／LangGraph agent、自建模型服務（語音、文件解析、reranker）與 PostgreSQL，提供 Streamlit 平台：**1. 測試各種地端 LLM 的表現　2. 展示多 Agent 協作場景**。所有服務只監聽 localhost，Ollama cloud 停用，資料不會送到雲端。介面可中英切換（右上角）。

[English README](README.md)

## 最新更新（2026-10-21）

- **地端版 GPT-Live 進場景展示**：在「場景展示」選「地端版 GPT-Live（邊聊邊查）」就能直接執行。新增 7 段**測試腳本**（中英文問天氣與店家、明天會不會下雨、地標找咖啡廳、閒聊、插話改問題），也能自訂台詞或上傳錄音——不用麥克風也能完整測一輪。
- **前台模型改為 Gemma 4 12B**：用這 7 段對話比過 10 個模型，12B 判斷全對、不編造、最自然，記憶體（7.7 GB）還比 E4B 小；設定裡仍可換回 E4B（開口快約 0.2 秒）。
- **語音助理更聰明**：英文對話一律用英文回答；追問（「第一家開到幾點？」）會引用上次查到的資料，查不到就說沒有；問題裡有地名就直接查；天氣多了明天的預報；地名查詢優先台灣的行政區，查不到改用 OpenStreetMap（板橋、新竹、信義區、西門町都能正確定位）。
- **串流評測總表**：「語音模型比較 → 串流（即時）→ 評測總表」列出本機與雲端（Azure）串流模型在同四段錄音上的錯字率、延遲與記憶體。
- **聲音清單**：macOS 的 8 個 Eloquence 聲音（同一個合成器）只列 Flo；下載了加強版／高品質語音會自動採用。

## 2026-10-19

- **即時語音助理（地端版 GPT-Live）**：「多 Agent 場景 → 即時語音助理」。邊說邊辨識，前台模型（預設 Gemma 4 E4B）約 1 秒開口回應；需要查天氣或附近店家時先說「我幫你查一下」，工具在背景執行，查完再說出結果；可插話打斷。含示範對話播放、背景工作看板與延遲表。天氣（Open-Meteo）與店家（OpenStreetMap）查詢需在頁面上開啟連網。

- **真串流語音辨識**：新增 Voxtral 4B Realtime（4-bit）、sherpa-onnx Zipformer（中英、中文）、Paraformer 串流；「串流」模式可比較真串流與切段模擬的延遲，並可用**麥克風即時字幕**（本機 WebSocket，邊說邊出字）。
- **台語**：新增聯發科 Breeze-ASR-26（台語版），台語範例可切換用台語漢字或華語翻譯評分。

## 2026-10-16

- **語音模型比較**：新增「串流（即時字幕模擬）」「文字轉語音（macOS 內建語音，含聽寫回測）」「語音對話（語音 → 文字 → LLM → 語音）」三種模式；逐字稿可選台語、英語有聲書、台灣口音英語、真實多人會議（AliMeeting）範例；英文改用詞錯誤率（WER）評分。
- **模型庫**：非 Ollama 模型擴充到 107 個，涵蓋串流、文字轉語音、語音對話、語音翻譯、語音偵測、講者辨識（只列出，尚未接上下載）。

## 更早的更新（2026-10-15）

- **語音模型比較（新頁面）**：和文字模型的比較分開，有三種模式。
  - 辨識：比較錯字率，可加噪音或電話音質。
  - 翻譯：中翻英，比較「先轉文字再翻譯」「音訊模型直接翻」「Whisper 內建翻譯」三種方式，含邊講邊翻的模擬。
  - 會議：辨識誰在什麼時候說話，輸出標好講者的逐字稿。
- **語音模型**：模型閘道現在有 10 個語音辨識模型，以及 1 個講者分段模型：
  - 語音辨識：Breeze-ASR-25、Whisper large-v3／turbo、Qwen3-ASR 1.7B／0.6B、SenseVoice、Fun-ASR-Nano、數位部 Taiwan-Tongues、Gemma 4 E2B／E4B。
  - 講者分段：pyannote 加 3D-Speaker。
- **真人測試音檔**：總統府提供的國慶演說節錄（4 分 29 秒，照稿宣讀，附官方全文與英譯），可直接拿來算錯字率。
- **會議紀錄場景**：先分出講者再轉錄，待辦事項標示提出者。
- **總控 Coding Agent**：對話會存檔，可以接續；網頁和 `local-agent code --resume` 共用同一份紀錄。
- **介面**：
  - 模型庫改成條列。
  - 對話測試的設定移到頁面上方。
  - 場景展示改成精簡選單。
  - 全站換成 Material 圖示。
- **執行紀錄**：可以單筆或多筆刪除，刪掉的會先進垃圾桶，可以復原。

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

| 選單 | 頁面 | 功能 |
| --- | --- | --- |
| 首頁 | 首頁 | 即時狀態：服務是否啟動、記憶體使用量、載入中的模型（可卸載）、執行中的工作；9 個場景卡片可直接開啟 |
| 模型 | 模型庫 | 107 個地端模型（對照 ollama.com 的 local 清單），以**能力標籤**（工具、思考、視覺、程式、embedding、決策、語音、OCR、rerank、台灣、授權…）呈現，可搜尋、篩選、看詳情；「非 Ollama」分頁可一鍵下載到模型服務 |
| | 對話測試 | 多輪對話、串流、思考過程另外顯示、首字延遲與 tokens/s；視覺模型可附圖 |
| | 模型比較 | 同一題最多 6 個模型並排；可選裁判模型評分 |
| | 語音模型比較 | 逐字稿（CER、噪音與電話音質條件）、中翻英（串接／音訊 LLM／Whisper，含即時輸入模擬）、會議講者分段（DER）、串流（切段模擬、真串流、麥克風即時、評測總表）、文字轉語音、語音對話 |
| | 語音與文件 | 語音轉文字（Breeze-ASR-25／Whisper）、文件解析（Docling）、rerank |
| Agents | Agent 工作室 | 選模型、寫系統 prompt、選工具與技能、設定資料庫身分，邊測邊改，存成新 agent |
| | Agents 一覽 | 每個 agent 的模型、工具、技能、資料庫身分與用在哪些場景 |
| | 多 Agent 比較 | 同一題同時問最多 4 個 agent，可用同一模型或各自的模型 |
| | 工具與技能 | 25 個工具（含 12 個本機 MCP 工具：time、git、fetch、files）、11 個 Agent Skills（3 個取自開源專案）、外部生態整理 |
| | 公司資料庫 | 以任一身分預覽資料（業務、人資、客服、財務、管理層、實習生）、權限矩陣、上傳 CSV／Excel |
| 多 Agent 場景 | 場景展示 | 10 個場景（含地端版 GPT-Live），以模型類型標示；每一步可選相容模型（✅ 已下載者為預設）；定義改用流程圖＋步驟表＋表單編輯 |
| | 知識庫 RAG | RAG 流程圖；三個需要 rerank 才答得對的示範題；BM25／向量／RRF／Rerank 並排 |
| 總控 | 總控 Coding Agent | 全地端 coding agent：設定模式只改 agent／場景（經驗證），程式模式可改 Python 並跑 pytest／ruff；每次寫入顯示 diff、核准後才存，可復原 |
| 更多 | 導入路線圖 | 八步驟旅程圖＋12 週甘特圖，每步可前往對應頁面 |
| | 執行紀錄 | 活動時間軸，依類型顯示易讀的結果（不再是 JSON） |

對話、Agent、比較、場景、下載都在背景執行：**切換頁面不會中斷**，右上角「工作」可看進度。

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
| 地端版 GPT-Live（邊聊邊查） | 串流辨識 → 前台 LLM 決定回答／反問／查詢 → [背景工具：天氣、店家] → 語音；可插話打斷 | zipformer、gemma4:12b、macOS 語音 |

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

## 地端 Coding Agent

```sh
uv run local-agent code                 # 終端機互動（/undo、/changes、/model、/mode、/think）
uv run local-agent code --mode settings # 只改 agent／場景設定
```

預設模型 `qwen3.6:35b-a3b-coding`（35B MoE，約 3B active，24GB）。三個固定任務的實測：

| 模型 | 換場景模型 | 新增工具＋測試＋註冊 | 修壞掉的測試 | 速度 |
| --- | --- | --- | --- | --- |
| qwen3.6:35b-a3b-coding | ✅ 14 秒 | ✅ 113–156 秒 | ✅ 23 秒 | 約 80–90 tok/s |
| qwen3.8:27b | ✅ 59 秒 | ✅ 346–439 秒 | ✅ 67 秒 | 約 30 tok/s |
| qwen3:30b-a3b（非 coding 版） | – | ❌ 32 分鐘後失敗 | – | 約 25 tok/s |

與 OpenCode、Codex、Claude Code 的比較與建議見 [docs/LOCAL_CODING_AGENT.md](docs/LOCAL_CODING_AGENT.md)。注意：通用型 coding agent（如 OpenCode）請在沙箱或專案副本中執行，並在結束後停掉它的背景服務。

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
| `agent_finance`／`agent_management`／`agent_intern` | 見平台「公司資料庫 → 身分與權限」 | |

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
    → 總控：coder/（工作區工具、agent、終端機 REPL）＋ lab/control.py → lab/store.py（驗證、版本、復原）
    → 技能與 MCP：skills.py（data/lab/skills/）、mcp_tools.py
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
