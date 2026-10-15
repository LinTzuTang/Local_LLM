# 本機 Coding Agent（Local Coding Agent）

> 完全在公司內部（on-prem）執行的 coding agent：模型、程式碼、設定與紀錄都不離開這台電腦。
> 量測日期 2026-10-13，Apple M5 Pro 64 GB，Ollama 0.40.3。

## TL;DR

- **建議模型：`qwen3.6:35b-a3b-coding`**（Ollama MLX nvfp4，24 GB，MoE 35B／每 token 約 3B 啟用）。
  三個固定任務（T1 改設定、T2 新增工具＋測試、T3 修壞掉的測試）**全部成功**，
  每秒約 **74–90 tokens**，T2 只要 **113–156 秒**。
- 第二名 `qwen3.8:27b`（dense 27B，18 GB）：三個任務也全部成功，但每秒只有 **27–31 tokens**，
  T2 要 **346–439 秒**（慢 2.2–3.9 倍）。品質沒有明顯比較好，所以不當預設。
- 通用模型 `qwen3:30b-a3b`（不是 coding 調校版）在 T2 **失敗**：32 分鐘後留下語法錯誤的 `agents.py`。
  **一定要用 coding 調校過的模型。**
- **思考模式要開**：`qwen3.6` 關掉思考後 T2 仍成功，但工具呼叫從 27–39 次暴增到 104 次、時間 315 秒。
- 使用方式：terminal 執行 `uv run local-agent code`，或平台「總控 Coding Agent」頁。
- 和雲端 Claude Code 比：本機 agent 適合**範圍明確、有測試把關**的修改；跨很多檔案的大型重構、
  需求模糊或需要讀很多程式的工作，Claude Code（Opus 5.5／Sonnet 5.5，1M context）明顯更強更穩，
  但程式碼會送到雲端。

## What was built

### 程式碼位置

| 檔案 | 內容 |
|---|---|
| `src/local_llm/coder/workspace.py` | 專案範圍的檔案工具：路徑安全、精確取代、diff、備份與復原、允許清單的檢查 |
| `src/local_llm/coder/agent.py` | `CoderSession`：設定工具＋程式工具、核准流程、模型偏好清單、統計 |
| `src/local_llm/coder/repl.py` | terminal 介面 `local-agent code`（rich） |
| `src/local_llm/lab/control.py` | 設定工具（agents.json／scenarios.json）；可對另一份專案副本操作，程式改過後在新行程驗證 |
| `app/views/control.py` | 平台「總控 Coding Agent」頁：範圍切換、模型選擇、核准卡片（diff）、變更紀錄與復原 |
| `tests/test_coder.py` | 路徑安全、唯一取代、備份／復原、允許清單、核准流程（用腳本化的假模型，不需要 Ollama） |

### 工具（agent 看得到的）

| 工具 | 需要核准 | 說明 |
|---|---|---|
| `list_files(glob)` | – | 列出符合 glob 的檔案（`src/**/*.py`、`*.md`、`app/views`） |
| `read_file(path, start, end)` | – | 附行號讀檔，一次最多 400 行 |
| `search(pattern, glob)` | – | regex 搜尋（有 ripgrep 用 ripgrep，否則 Python） |
| `run_check(name, path?)` | – | 只能跑允許清單：`pytest`（pytest -q）、`ruff`（ruff check）、`ruff_format`（ruff format --check）、`compileall`、`config`（用目前程式驗證 agents.json＋scenarios.json）；有逾時、輸出截斷 |
| `edit_file(path, old, new)` | ✅ | `old` 必須**剛好出現一次**，否則回錯誤（0 次會提示是不是縮排不同或夾帶行號） |
| `write_file(path, content)` | ✅ | 只能**建立新檔**；已存在的檔案要用 `edit_file` |
| `list_agents` / `show_agent` / `list_scenarios` / `show_scenario` / `list_node_types` / `list_models` / `list_agent_tools` / `validate_scenario` | – | 讀設定 |
| `update_agent` / `create_agent` / `update_node` / `add_node` / `remove_node` / `set_edges` / `save_scenario` | ✅ | 改設定：先驗證、保留版本、可復原 |

### 安全設計

- **路徑**：所有路徑都解析成絕對路徑（包含 symlink）後必須在專案內；`.env`（及金鑰檔）、`models/`、
  `storage/pgdata`、`.venv`、`.git` **讀寫都拒絕**；`storage/`、`runs/` 只能讀。
  `data/lab/agents.json`、`scenarios.json` 不能用 `edit_file` 改，只能走設定工具（有驗證與版本）。
- **核准**：所有寫入都經過 LangChain `HumanInTheLoopMiddleware` 暫停，先顯示 unified diff
  （設定另外顯示驗證錯誤／警告）。**套不上的修改**（找不到 `old`、設定驗證失敗）不會打擾人，
  直接把錯誤回給模型，什麼都不會寫入。
- **備份與復原**：每次寫入前把修改前後內容存到 `storage/coder/backups/<id>/`（專案不是 git repo）。
  可以逐筆復原；如果檔案之後又被改過，會要求先復原較新的那筆，避免蓋掉別的修改。
  設定則沿用 `storage/lab/history/` 的版本紀錄。
- **只能執行允許清單**：沒有任意 shell。檢查以 `PYTHONPATH=<專案>/src` 在子行程執行，所以在專案副本裡跑也會測到副本的程式。
- **程式改過之後的設定驗證**：執行中的平台還是舊程式（例如新工具還沒載入），所以只要這次啟動後
  有 `.py` 被改過，設定的預覽、寫入與復原就改在**新的 Python 行程**用最新程式驗證。
  頁面也會提示執行 `./scripts/restart_app.sh` 載入新程式。
- **防失控**：每個使用者回合最多 80 次模型呼叫；上下文超過 60% 時自動清掉較舊的工具輸出
  （`ContextEditingMiddleware`）。

### Terminal：`uv run local-agent code`

```bash
uv run local-agent code                      # 互動模式，自動選最好的已安裝 coding 模型
uv run local-agent code --mode settings      # 只改設定（agents／scenarios）
uv run local-agent code --model qwen3.8:27b --ctx 65536 --think on
uv run local-agent code --root /path/to/copy # 在專案副本上實驗
uv run local-agent code -p "把 hybrid_rag 的 answer 節點改用 qwen3:1.7b"   # 單次執行
uv run local-agent code -p "..." --approve-all --log run.json  # 腳本化（請只在副本上用）
uv run local-agent code --resume             # 繼續最近一段對話（--resume <id> 指定）
```

畫面會即時串流模型的思考（淡色）與回答、每次工具呼叫（`→ read_file(...)`）與結果摘要；每個寫入顯示 diff 並詢問：

- `y` 核准、`n` 拒絕、`e` 拒絕並輸入理由／改成怎麼做（會告訴 agent）、`a` 核准這回合剩下的全部。

指令：`/undo`（復原這次 session 最新的一筆，程式或設定都可以）、`/undo list`、`/undo 2`、`/changes`、
`/model`（列出／切換模型，對話保留）、`/mode settings|code`、`/think on|off|auto`、
`/sessions`（列出存過的對話）、`/resume [n|id]`（接續某段對話）、`/clear`（開新對話，舊的仍保留）、`/exit`。

**對話紀錄**：每次執行後，整段對話（含工具呼叫與結果）會存到 `storage/coder/sessions/<id>.json`
（不進 git）。terminal 與平台頁面共用同一份紀錄：在 terminal 開始的對話可以在網頁接著聊，反之亦然。
接續時會沿用原本的模式與上下文設定；原本的模型若已移除，改用目前選的模型。
每回合結束顯示：時間、模型呼叫、工具呼叫、核准次數、輸出 tokens、tokens/s。

### 平台頁面：總控 Coding Agent

左側欄最上方是「對話」選單：新對話，或選一段存過的對話繼續（可刪除）。接著可切換「⚙️ 只改設定」與「💻 設定＋程式碼」、選模型（預設為已安裝中最好的 coding 模型，
沒安裝會提供下載按鈕）、思考模式、上下文長度。執行在背景工作中進行（切換頁面不會中斷）；
暫停時出現核准卡片（diff、驗證結果、核准／拒絕＋理由、全部核准）。右側「變更紀錄」分
「⚙️ 設定版本」與「💻 程式備份」兩個分頁，都可以一鍵復原。

## Choosing the model

### 候選（ollama.com，2026-10）

| 模型 | 大小 | 架構 | 公開的 coding 成績（廠商自報） | 判斷 |
|---|---|---|---|---|
| **qwen3.6:35b-a3b-coding** | 24 GB | MoE 35B / 3B 啟用，MLX nvfp4，256K | Qwen3.6-35B-A3B：SWE-bench Verified 73.4、Terminal-Bench 2.0 51.5 | ✅ 下載並量測 |
| **qwen3.8:27b** | 18 GB | dense 27B，MLX nvfp4，256K | Terminal-Bench 2.1 73.0、SWE-bench Pro 61.7（Claude Code harness） | ✅ 下載並量測 |
| qwen3-coder-next | 52 GB | MoE 80B / 3B | 被多篇評測推為 64 GB 機器的首選 | 64 GB Mac 太緊（GPU 預算約 44–48 GB），適合 DGX Spark |
| qwen3-coder:30b | 19 GB | MoE 30B / 3B | 上一代 | 被 qwen3.6 coding 取代 |
| devstral-small-2:24b | 15 GB | dense 24B | agentic coding | dense 速度與 qwen3.8 同級 |
| north-mini-code-1.0 | 19 GB | MoE 30B / 3B | 未公開文字成績 | 可做下一輪比較 |
| ornith:35b | 21 GB | – | 只有圖片成績，MIT 授權 | 太新、資訊不足 |
| qwen3:30b-a3b（已安裝） | 19 GB | MoE，一定思考 | 通用模型 | 當作對照組 |

選擇理由：64 GB Mac 上真正的瓶頸是**解碼速度**與**工具呼叫穩定度**。MoE（每 token 只算約 3B 參數）
在 M5 Pro 上比 dense 27B 快 3 倍左右，而 coding 調校版的工具呼叫格式很穩（11 次 T2 相關執行中，
coding 模型沒有出現一次格式錯誤的工具呼叫）。

### 三個固定任務

所有任務都在**專案副本**（scratchpad，排除 `.env`、`models/`、`storage/`、`.venv`）執行，
真正的 repo 沒有被改動；自動核准所有寫入（`--approve-all`），事後用獨立腳本驗證。

- **T1**：把 `hybrid_rag` 場景 `answer` 節點的模型改成 `qwen3:1.7b`，其他都不能動。
- **T2**：新增 `src/local_llm/unit_tools.py` 的 `unit_convert`（km↔mi、kg↔lb、m↔ft、C↔F），
  在 `agents.py` 以中英文 `ToolInfo` 註冊、加到 `tools` agent、寫 `tests/test_unit_tools.py`，
  ruff 與 pytest 都要過。驗證：新行程能載入並呼叫工具（10 km→6.21 mi、100 C→212 F）、agent 有這個工具、
  全部測試與 ruff 通過、設定驗證通過。
- **T3**：故意把 `textutil.format_value` 的 `enumerate(value, start=1)` 改成 `enumerate(value)`，
  請 agent 修好且**不能改測試**。驗證：pytest 通過、tests/ 雜湊不變、行為恢復。

### 結果（本專案的 coding agent）

| 模型 | 任務 | 成功 | 時間 (s) | 工具呼叫 | 核准 | 模型呼叫 | 輸出 tokens | tokens/s |
|---|---|---|---|---|---|---|---|---|
| qwen3.6:35b-a3b-coding | T1 | ✅ | 13.9 | 3 | 1 | 4 | 321 | 82.5 |
| qwen3.6:35b-a3b-coding | T2 | ✅ | 156.1 | 39 | 13 | 30 | 9,722 | 80.0 |
| qwen3.6:35b-a3b-coding | T2（重跑） | ✅ | 113.0 | 27 | 7 | 22 | 5,955 | 73.9 |
| qwen3.6:35b-a3b-coding | T3 | ✅ | 22.5 | 6 | 1 | 7 | 940 | 90.0 |
| qwen3.6:35b-a3b-coding | T3（重跑） | ✅ | 23.5 | 7 | 1 | 8 | 1,042 | 88.0 |
| qwen3.6:35b-a3b-coding，**關閉思考** | T2 | ✅ | 315.2 | 104 | 23 | 95 | 14,171 | 71.7 |
| qwen3.8:27b | T1 | ✅ | 58.6 | 5 | 1 | 5 | 755 | 27.0 |
| qwen3.8:27b | T2 | ✅ | 345.7 | 37 | 13 | 21 | 8,494 | 30.6 |
| qwen3.8:27b | T2（重跑） | ✅ | 439.4 | 39 | 20 | 25 | 11,210 | 31.4 |
| qwen3.8:27b | T3 | ✅ | 67.0 | 8 | 1 | 9 | 1,008 | 28.7 |
| qwen3:30b-a3b（通用） | T2 | ❌ | 1,921.4 | 19 | 9 | 20 | 39,992 | 24.5 |

觀察：

- T2 的典型流程（qwen3.6）：讀 `agents.py`、`tools.py`、`agents.json`、`media_tools.py` 找範例 →
  建新模組 → 改 `agents.py` 註冊 → 試著用 `edit_file` 改 `agents.json` 被路徑規則擋下、
  改用 `update_agent` → 寫測試 → ruff／pytest 失敗（未使用的 import、測試直接呼叫 `StructuredTool`、
  型別註記寫錯、C/F 大小寫）→ 逐一修正到全綠。**自我修正的能力是成功的關鍵**，而允許清單的
  `run_check` 給了它可靠的回饋。
- 思考模式關閉時模型會「邊改邊猜」：同一個錯誤反覆修改，prompt 最大到 49K tokens。
- `qwen3:30b-a3b` 的長思考加上 64K context 讓解碼掉到 24.5 tokens/s，且在 `agents.py` 留下
  `positional argument follows keyword argument` 的語法錯誤後放棄。
- 記憶體：qwen3.6 在 64K context 時約 24–27 GB（MLX；長對話時曾看到 39 GB），
  因為平台的 `OLLAMA_NUM_PARALLEL=3`，同時載入其他大模型時要注意 GPU 預算。

## Existing open-source local coding agents

| | 是什麼 | 安裝（隔離，不進專案 venv） | 接 Ollama |
|---|---|---|---|
| **OpenCode 2.0.25** | 開源 terminal coding agent（類 Claude Code 的工具：read/edit/bash/grep…），自動偵測 Ollama 模型 | `brew install opencode` | `opencode.json` 設 `provider.ollama`（OpenAI 相容 `/v1`）；`ollama launch opencode` 也可 |
| **Codex CLI 0.162.1** | OpenAI 開源的 terminal agent，`--oss --local-provider ollama` 可用本機模型，有 macOS sandbox | `brew install --cask codex` | `codex exec --oss --local-provider ollama -m <model>` |
| Aider | 以 diff 格式編輯的 pair-programming 工具（非 tool-calling agent），偏好 git repo | `uv tool install aider-chat` | `--model ollama_chat/<model>` |
| Goose / Qwen Code | Block 的通用 agent／Qwen 官方 CLI（需要 Node） | `brew install block-goose-cli` | 原生支援 Ollama |

On-prem 設定重點（OpenCode）：`"enabled_providers": ["ollama"]`（否則會列出 opencode 的免費**雲端**模型）、
`"share": "disabled"`、`"autoupdate": false`；它也會讀使用者家目錄的設定與技能資料夾，部署時要一併管理。
Ollama 的 OpenAI 相容端點預設用模型的最大 context，請建立固定 context 的別名：

```bash
curl http://127.0.0.1:11434/api/create -d '{"model":"qwen3.6-coding-64k","from":"qwen3.6:35b-a3b-coding","parameters":{"num_ctx":65536}}'
opencode run -m ollama/qwen3.6-coding-64k:latest "..." < /dev/null   # 非互動執行要關掉 stdin
```

（別名不佔額外空間；量測後已刪除，以免出現在平台的模型清單。Codex CLI 以
`brew install --cask codex` 安裝，不用時 `brew uninstall --cask codex`。）

### 量測結果（T2，同一個模型 qwen3.6 coding、64K context）

| agent | 結果 | 時間 | 工具呼叫 | 備註 |
|---|---|---|---|---|
| 本專案 coding agent | ✅ 2/2 | 113–156 s | 27–39 | 每個寫入都有 diff＋核准（測試時自動核准） |
| OpenCode 2.0.25 | ❌ 5 分鐘上限內未完成 | 300 s（被終止） | 38 | 見下方 |
| Codex CLI 0.162.1 | 未量測 | – | – | 已安裝，時間不足未跑 |

OpenCode 觀察（同模型、同一份專案說明放在 `AGENTS.md`）：

- 第一次執行**看起來卡住**（client 端沒有輸出）：`opencode run` 在 stdin 不是 TTY 時會等待 stdin 輸入；
  改成 `< /dev/null` 後才正常運作。更嚴重的是：client 被終止後，OpenCode 的**背景服務**
  （`opencode serve --service`，第一次使用時自動啟動）仍繼續執行那個 session，並且跑到**真正的專案目錄**
  新增了 `src/local_llm/unit_tools.py`、修改了 `src/local_llm/agents.py`。這兩個變更已比對量測前的快照後還原；
  教訓是一般用途的 agent 不只可能離開工作目錄，連「停掉」都要確認背景服務（`opencode service stop`）。
- 第二次它**離開了自己的工作目錄**：先把 session 移到一個猜測的路徑
  （從資料夾名稱推出 `/Users/tzutang/Desktop/Local-LLM`，不存在），再用 subagent 找專案，
  結果改到 scratchpad 裡**另一份**專案副本（用絕對路徑讀寫、用 shell 跑 `sed -i`）。真正的 repo 沒被動到，
  但這說明一般用途的 agent 預設權限很寬：要在 on-prem 使用，請放在容器／獨立帳號或 sandbox 裡，
  並設定 `permission`（不要用 `--auto`）。
- 5 分鐘時它寫了模組、改了 `agents.py` 與測試，但測試還有 17 個失敗（測試直接呼叫 `StructuredTool`）
  ——和我們的 agent 第一次遇到的錯誤一樣，只是還沒修完。以速度看，它的流程（多一個 explore subagent、
  大量 shell 指令）比我們的 agent 慢，結論是：**模型相同時，限制明確、工具少而精準的 harness 在小模型上更有效率**。

## Claude Code as the cloud reference

Claude Code 是 Anthropic 的雲端 coding agent，模型在 Anthropic（或 AWS Bedrock／Google Vertex AI／
Microsoft Foundry）上執行；**沒有可以自架的模型權重**，所以完全 on-prem 的環境不能用。

| | 本專案 coding agent（qwen3.6 coding） | OpenCode／Codex + 本機模型 | Claude Code（雲端） |
|---|---|---|---|
| 模型品質 | 範圍明確的修改＋測試把關可以穩定完成；大型重構、需求模糊時明顯較弱 | 同模型、同上限；harness 較通用 | Claude Opus 5.5／Sonnet 5.5，目前業界頂尖；多檔重構、讀大量程式、規劃與自我檢查都強很多 |
| Context | 實用 64K（可調 128K，但更慢更吃記憶體） | 同左 | 1M tokens |
| 速度 | 74–90 tokens/s；T2 約 2–2.5 分鐘 | 同模型速度 | 通常更快，且不佔用本機 GPU |
| 穩定度 | 3 個任務 7/7 成功（qwen3.6）；一般模型會失敗 | 視 harness 而定 | 很高 |
| 隱私 | 程式碼、設定、紀錄都不離開這台電腦 | 本機；需關閉雲端 provider／分享／自動更新 | 程式碼與 prompt 送到雲端（可選 Bedrock／Vertex 留在自己的雲端帳號，仍不是 on-prem） |
| 成本 | 硬體攤提＋電費；每次執行 0 元 | 同左 | 依 token 計費：Opus 5.5 每百萬 tokens 輸入 4 美元／輸出 20 美元；Sonnet 5.5 為 2／10 美元；Haiku 5.5 為 0.10／0.50 美元。agent 每個任務會反覆送整段對話，T2 這類任務約數十萬輸入 tokens |
| 平台整合 | 懂 agents.json／scenarios.json 的驗證與版本、核准卡片、變更紀錄、復原 | 一般檔案編輯（設定 JSON 直接改，沒有平台驗證） | 一般檔案編輯；可透過 MCP 或自訂指令整合 |

> 補充：Ollama 提供 Anthropic 相容 API（`ollama launch claude --model <本機模型>`），可以讓 Claude Code
> 的 **harness** 接本機模型，但品質取決於本機模型而不是 Claude；是否允許在公司內這樣使用，
> 要另外確認授權條款與遙測設定。本文沒有量測這個組合。

## Recommendation

1. **平台日常維運（改 agent／場景、加小工具、修測試）**：用本專案的 coding agent
   （`local-agent code` 或總控頁）＋ `qwen3.6:35b-a3b-coding`，思考開啟、64K context。
   它知道平台的設定規則，每個寫入都有 diff、核准、備份、可復原，最適合給非專職工程師使用。
2. **其他 repo 的一般開發（完全 on-prem）**：OpenCode 或 Codex CLI 接同一台 Ollama
   （或公司內的 GPU 伺服器），只開本機 provider、關閉分享與自動更新；工作要切小、要有測試。
3. **需要更強的本機模型**：把 Ollama 放到 DGX Spark（128 GB）跑 `qwen3-coder-next`（80B）或高精度版本，
   Mac 只當用戶端（`--base-url` 指向該伺服器）。
4. **允許雲端的非機密專案、或大型重構與架構設計**：Claude Code 仍是品質上限；可以先在雲端做設計，
   再把明確的小任務交給本機 agent 執行。
5. 不論哪一種，都要有**測試與 lint**：本機模型的成功幾乎完全來自「改 → 跑檢查 → 看錯 → 再改」的迴圈。

## What did not work / limits

- 通用模型（`qwen3:30b-a3b`）在 T2 失敗；不要用非 coding 模型做程式修改。
- 平台的 `ollama_admin.thinking_mode` 只看 chat template，新的 MLX 模型（qwen3.6／3.8）會被判成
  「一定思考」；coding agent 改讀 `/api/show` 的 `thinking` 欄位，可以正確開關。
- 程式改完後，執行中的平台要重新啟動才會載入新工具；在那之前設定驗證會在新行程進行。
- 實用 context 64K：讀很多大檔的任務要分段，否則舊的工具輸出會被清掉。
- 量測數量小（每組 1–2 次），只能看出趨勢；正式導入前建議用自己的 backlog 再跑兩週。

## Sources

- [Ollama model search](https://ollama.com/search?c=tools&q=code)、[qwen3.6 tags](https://ollama.com/library/qwen3.6/tags)、
  [qwen3.8](https://ollama.com/library/qwen3.8)、[qwen3-coder-next](https://ollama.com/library/qwen3-coder-next)、[ornith](https://ollama.com/library/ornith)
- [Qwen3.6-35B-A3B 成績整理](https://rits.shanghai.nyu.edu/ai/qwen3-6-35b-a3b-alibaba-open-sources-a-frontier-class-agentic-coder/)、
  [Qwen3.8-27B 成績整理](https://benchlm.ai/models/qwen3.8-27b)
- [Best local LLMs for agentic coding on Apple silicon](https://zachrattner.com/projects/ai-mac-cluster/coding-models)、
  [The Best Local Coding Models for Any Setup](https://blog.kilo.ai/p/the-best-local-coding-models-for)
- 量測原始資料：每次執行的 JSON 紀錄由 `local-agent code --log` 產生（本次放在 scratchpad，未放進 repo）。
