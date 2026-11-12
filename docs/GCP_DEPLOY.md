# 在 GCP GPU 機上跑 Local Agent Lab

目標機器：GCP 上的 NVIDIA GPU 機（目前用 RTX PRO 6000 96 GB，Linux x86_64，驅動 580／CUDA 13，spot）。從 Mac 一個指令開機、部署、啟動，再用 SSH tunnel 在 Mac 瀏覽器使用。

## 指令

```sh
./scripts/remote/up.sh          # 開機 → 複製程式碼 → 遠端 tmux_up → tunnel（Ctrl-C 只關 tunnel）
./scripts/remote/tunnel.sh      # 只重開 tunnel
./scripts/remote/deploy.sh      # 只同步程式碼
./scripts/remote/down.sh        # 停平台＋資料庫，關機，等到 TERMINATED
./scripts/remote/down.sh --keep # 只停平台，機器不關
```

第一次在遠端安裝（只需一次）：

```sh
./scripts/remote/deploy.sh
ssh ... 'cd ~/Local_LLM && ./scripts/remote/setup_linux.sh'
```

`setup_linux.sh` 會裝 ffmpeg、tmux、PostgreSQL、Ollama（固定成和 Mac 相同的 0.40.13），跑 `scripts/setup.sh`（`uv sync --locked`＋測試），並寫一份**不含雲端金鑰**的 `.env`（`LOCAL_LLM_GPU_BUDGET_GB=90`）。

機器資料（機器名稱、GCP 專案、zone、OS Login 帳號）寫在 `scripts/remote/config.local.sh`，這個檔不進 git（repo 是公開的），格式見 `scripts/remote/config.sh` 開頭；也可用環境變數覆寫。

## 和 Mac 版的差別

| 項目 | Mac | GCP（Linux＋CUDA） |
|---|---|---|
| Whisper 系辨識（Breeze-ASR-25／26、Taiwan-Tongues、Whisper large-v3／turbo） | mlx-whisper，權重轉成 MLX | `hf-whisper`：transformers 直接讀官方 repo 的 safetensors，fp16 在 CUDA 上跑，不轉檔 |
| sherpa 串流辨識、sherpa 講者分離、gemma4 音訊、Docling、reranker | 同左 | 同左；reranker 用 CUDA，Docling 用 AUTO（CUDA） |
| Qwen3-ASR、SenseVoice、Fun-ASR-Nano、Voxtral、Nemotron／Sortformer 講者分離 | mlx-audio | **不提供**（閘道在 Linux 上自動隱藏） |
| Docling 的 OCR | macOS Vision（ocrmac） | 關閉：有文字層的 PDF 照常解析，掃描檔讀不到字 |
| 語音合成（macOS `say`、Kokoro、Qwen3-TTS） | 有 | 沒有；語音助理頁面暫不能用 |
| 雲端模型（Azure／Bedrock／Vertex／ElevenLabs） | 有 `.env` 金鑰 | 預設沒有金鑰 |

平台判斷在 `catalog.for_platform()`：catalog 閘道模型的 `linux` 區塊在 Linux 上取代 backend／weights／path／convert；後端仍是 MLX 的模型在 Linux 上不出現。

## 注意

- **費用**：每小時約 1.5 美元。用完一定跑 `down.sh`，看到 TERMINATED 才算關好。平日 19:00 會自動關機。
- **Spot**：隨時可能被收回；平台狀態（資料庫、對話）在遠端 `storage/`，資料會留著，重新 `up.sh` 即可。
- **STOCKOUT**：開機時出現 `ZONE_RESOURCE_POOL_EXHAUSTED`／`STOCKOUT` 代表這個 zone 暫時沒機器，過一陣子再試。
- **IP 每次開機都會換**：腳本用 `HostKeyAlias=<機器名稱>` 記 host key，不受 IP 變動影響。
- **磁碟 194 GB**：模型在遠端下載（Ollama 放 `models/ollama`、閘道放 `models/`），不從 Mac 複製；只拉需要的。
- **不要直接對外開平台的 port**：Streamlit（8501）、閘道（11500）、登入服務（11600）都只聽 127.0.0.1；對外只能經過下面的 Caddy（443）。
- `deploy.sh` 不複製 `models/`、`storage/`、`runs/`、`.env`、`.git/`。

## 開放給同事：帳號密碼登入

```
同事瀏覽器 ──HTTPS 443──> Caddy ──forward_auth──> 登入服務 127.0.0.1:11600（cookie 有效？）
                            ├─ /auth/*                     → 登入服務（登入頁、登出）
                            ├─ /gateway/v1/{audio/stream,voice/agent}/ws → 閘道 127.0.0.1:11500
                            └─ 其他                         → Streamlit 127.0.0.1:8501
```

- 登入服務（`src/local_llm/auth_server.py`）檢查 cookie，通過後 Caddy 把帳號寫進 `X-Lab-User`／`X-Lab-Role` 再轉給平台；瀏覽器自己送來的同名 header 會先被刪掉。閘道只開放瀏覽器會直接連的兩個 WebSocket。
- 帳號存在遠端 `storage/auth/users.json`（scrypt 雜湊），cookie 用 `storage/auth/secret` 簽章，有效 7 天；改密碼、改角色、刪帳號都會讓該帳號的登入立即失效。同一帳號或同一 IP 10 分鐘內錯 5 次會鎖 10 分鐘。
- 角色：`admin` 看得到「總控 Coding Agent」、可以儲存場景 JSON；`user` 看不到總控頁（直接輸入網址也進不去）。從機器本身或 SSH tunnel 連進來（沒經過 Caddy）視為管理者。
- 沒開 `LOCAL_LLM_AUTH` 時（Mac）一切照舊。

**還沒有網域時先自己試登入版**：遠端跑 `./scripts/remote/setup_caddy.sh --local`（Caddy 只聽遠端 127.0.0.1:8080，HTTP），Mac 跑 `./scripts/remote/tunnel.sh --login`，打開 http://127.0.0.1:18080 。Mac 上自己的平台（8501／11500）不用關。

**要先請 IT 處理**（你的帳號沒有這些權限）：固定外部 IP、一個網域（DNS 指到該 IP）、防火牆開 TCP 80 和 443（80 是 Caddy 申請憑證用）。

之後在遠端執行一次：

```sh
./scripts/remote/setup_caddy.sh lab.example.com   # 裝 Caddy、寫 /etc/caddy/Caddyfile、開 LOCAL_LLM_AUTH、建第一個管理者
./scripts/tmux_down.sh && ./scripts/tmux_up.sh    # 重啟後 tmux 會多一個 auth 視窗
uv run local-agent users add <name>               # 新增同事（--admin 為管理者）
uv run local-agent users passwd|role|remove|list ...
```

開放後要注意：
- 機器是 spot，平日 19:00 自動關機，同事只能在開機時段使用；GPU 只有一張，多人同時跑大模型會互相排隊。
- 「模型庫」頁的下載／刪除模型、Agent 定義的修改目前沒有限制角色，所有登入的人都能操作。
