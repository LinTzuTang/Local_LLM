# DGX Spark 測試與遷移

研究日期 2026-10-06。此專案目前只在 Mac 實測，尚未取得 Spark 主機與 SSH 資訊。以下部署需要在 Spark 執行；不在 Mac 安裝 CUDA 版 vLLM。

## 平台與初始選型

Spark 是 Arm64 Linux + Grace Blackwell、128GB unified memory。[官方硬體資料](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)。CPU／GPU／系統共用記憶體，不是額外 128GB VRAM；TOPS 或「可跑 200B」不保證某個模型的延遲、長 context 或併發。

先以短 context 與單／少量併發開始：Qwen3-8B 可作較容易比較的 baseline；新模型使用官方 Qwen3.8-27B-NVFP4 recipe。coding 候選為 Qwen3-Coder-30B-A3B、Devstral Small 2；reasoning 可比較 gpt-oss-20b 或 Nemotron 3 Nano。模型相容、量化 kernel、tool parser 都要配對，不要直接用普通 x86 CUDA image。

## 複製專案

在 Mac：

```sh
rsync -av --exclude=.venv --exclude=.runtime --exclude=models --exclude=runs \
  ~/Desktop/Local_LLM/ your-user@your-spark:~/Local_LLM/
```

在 Spark：

```sh
uname -m                  # 預期 aarch64
nvidia-smi
docker ps
free -h
df -h
# 安裝 uv：若尚未安裝，依 https://docs.astral.sh/uv/getting-started/installation/
cd ~/Local_LLM
uv sync --locked
```

## vLLM 官方 recipe 路線

本專案腳本以 [NVIDIA Spark 單機 playbook](https://build.nvidia.com/spark/vllm/instructions) 與[官方 launch script](https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/playbook-vllm/assets/sync-vllm-single-spark.sh) 為來源，使用 `vllm/vllm-openai:qwen38` 及 `nvidia/Qwen3.8-27B-NVFP4`。啟動設定從 8K context、2 sequences、0.7 memory utilization 開始，是本專案為 smoke test 調整的值，沒有宣稱最優效能。

在 Spark：

```sh
./scripts/download_spark_model.sh
./scripts/start_vllm_spark.sh
```

首次會下載容器與模型，需要充分磁碟空間。模型本身至少數十 GB 級別的空間規劃，實際以 HF files／下載結果為準；HF open checkpoint 通常不需登入，遇 gated 或 rate limit 再以 `hf auth login` 處理。等待 log 顯示 API ready，再開另一個 Spark terminal：

```sh
curl -f http://127.0.0.1:8000/health
curl -f http://127.0.0.1:8000/v1/models
uv run local-agent doctor --provider vllm --model nvidia/Qwen3.8-27B-NVFP4
uv run local-agent agent --provider vllm --model nvidia/Qwen3.8-27B-NVFP4 \
  --require-tool --trace-file runs/spark-agent.json
```

此腳本只將 Docker host port bind 到 127.0.0.1。從 Mac 開 tunnel：

```sh
ssh -N -L 8000:127.0.0.1:8000 your-user@your-spark
```

然後在另一個 Mac terminal 使用相同的 `--provider vllm` 指令。若 8000 已佔用，可用 `-L 8001:127.0.0.1:8000`，CLI 加 `--base-url http://127.0.0.1:8001/v1`。

`start_vllm_spark.sh` 使用前景 Docker 與 `--rm`；Ctrl+C 停止並移除本次容器，模型 cache 保留。不會刪除其他容器。image tag 可變；部署成功後記錄 RepoDigest，再用 digest pin 固定復現。版本若不支援 flags 或模型，回到官方 recipe 檢查，不把它當成 Python client 問題。

## Ollama 備用路線

[NVIDIA Ollama playbook](https://build.nvidia.com/spark/open-webui)。在 Spark 安裝 Linux Arm64 Ollama，設置 `OLLAMA_MODELS` 後 `ollama serve`，以 `ollama pull qwen3:1.7b` 先重跑小模型 demo；接著才測試 8B／30B。Mac 的 `.runtime/Ollama.app` 是 macOS app，不可搬到 Linux 使用。

## 評測紀錄

每個候選同時記錄 checkpoint revision／quantization、runtime image digest、context、最大 token、併發數、GPU/系統記憶體、TTFT、decode tokens/s、p50/p95 總延遲、工具呼叫成功率、參數正確率、schema 合規與任務成功率。Mac 測速不能直接推估 Spark。

先跑 chat，再測工具（含錯誤參數與工具回傳錯誤），再測繁中／多步任務，最後增加 context 及併發。模型宣稱支援工具不表示某個 serving parser 已設定；`--enable-auto-tool-choice` 與 checkpoint 對應 parser 必須一致。[vLLM tool calling 官方說明](https://docs.vllm.ai/en/latest/features/tool_calling/)。
