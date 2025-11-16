# 訓練安全措施與自動恢復指南

**Training Safety Measures & Auto-Recovery Guide**

Created: 2025-11-15
Version: 1.0.0

---

## 📋 概述

本指南提供完整的訓練安全措施，防止類似 CUDA 錯誤、訓練中斷等問題，並提供自動恢復機制。

---

## 🔍 常見問題與根本原因

### 問題 1: CUDA Unknown Error

**症狀：**
```
RuntimeError: CUDA error: unknown error
CUDA kernel errors might be asynchronously reported...
```

**根本原因：**
1. **GPU 記憶體碎片化** — 長時間運行（>6小時）導致 VRAM 分配碎片化
2. **Gradient Checkpointing 不穩定** — 在某些 PyTorch 版本或 WSL2 環境下不穩定
3. **WSL2 + CUDA 長時間運行問題** — GPU 虛擬化層可能不如原生穩定
4. **驅動或硬體隨機問題** — 偶發性錯誤

**預防措施：**
- ✅ 關閉 `gradient_checkpointing`（使用穩定配置）
- ✅ 降低 `gradient_accumulation_steps`（減少記憶體壓力）
- ✅ 縮短訓練時長（12 epochs vs 20 epochs）
- ✅ 頻繁保存 checkpoint（每 2 epochs）
- ✅ 使用自動監控和重啟機制

---

### 問題 2: 訓練卡住（Hang）

**症狀：**
- GPU 利用率低但進程仍在運行
- 長時間沒有新的 checkpoint 產生
- 日誌輸出停止

**根本原因：**
1. 資料載入器死鎖
2. CUDA 同步問題
3. I/O 阻塞

**預防措施：**
- ✅ 啟用掛起檢測（30分鐘無進度自動重啟）
- ✅ 使用 `num_workers` 適當值（不要太高）
- ✅ 監控 checkpoint 更新時間

---

### 問題 3: OOM (Out of Memory)

**症狀：**
```
RuntimeError: CUDA out of memory
```

**根本原因：**
1. Batch size 太大
2. 模型或圖片解析度太高
3. 記憶體洩漏

**預防措施：**
- ✅ 使用 `train_batch_size=1`
- ✅ 啟用 `cache_latents=true`
- ✅ 使用 8-bit optimizer
- ✅ 監控 VRAM 使用率（警告閾值 95%）

---

## 🛡️ 安全措施系統架構

### Layer 1: 配置優化

**穩定版配置文件：** `configs/training/sdxl_16gb_stable.toml`

**關鍵設定：**
```toml
# 關閉 gradient checkpointing（避免 CUDA checkpoint 錯誤）
gradient_checkpointing = false

# 降低 accumulation steps（降低記憶體壓力）
gradient_accumulation_steps = 4  # 從 8 降到 4

# 縮短訓練時長（避免長時間運行問題）
max_train_epochs = 12  # 從 20 降到 12

# 頻繁保存（方便恢復）
save_every_n_epochs = 2
save_last_n_epochs = 3  # 保留最後 3 個 checkpoints

# 8-bit optimizer（省記憶體）
optimizer_type = "AdamW8bit"

# 完整 bf16（穩定性）
mixed_precision = "bf16"
full_bf16 = true
```

### Layer 2: 自動健康監控

**監控腳本：** `scripts/monitoring/training_health_monitor.sh`

**功能：**
- ✅ 每 5 分鐘檢查一次 GPU 狀態
- ✅ 監控溫度、VRAM 使用率、GPU 利用率
- ✅ 偵測訓練掛起（30分鐘無進度）
- ✅ 自動重啟失敗的訓練
- ✅ 發送桌面通知（如果可用）

**使用方式：**
```bash
# 基本監控（自動跟隨現有 session）
bash scripts/monitoring/training_health_monitor.sh

# 指定 session 監控
bash scripts/monitoring/training_health_monitor.sh \
  --session sdxl_luca_training_safe \
  --output-dir /mnt/data/ai_data/models/lora/luca/sdxl_trial1 \
  --interval 300 \
  --max-restarts 3
```

**監控指標：**
| 指標 | 閾值 | 動作 |
|------|------|------|
| GPU 溫度 | >85°C | 警告 |
| VRAM 使用率 | >95% | 警告 |
| GPU 利用率 | <5% | 檢查掛起 |
| Checkpoint 年齡 | >30分鐘 | 重啟訓練 |

### Layer 3: 自動重啟機制

**重啟腳本：** `scripts/training/safe_restart_training.sh`

**流程：**
1. **清理舊 session** — 殺掉卡住的訓練進程
2. **等待 GPU 清空** — 確保 VRAM 完全釋放
3. **檢查 checkpoint** — 找到最新的 checkpoint
4. **驗證配置** — 確認使用穩定版配置
5. **啟動訓練** — 在新 tmux session 中啟動
6. **啟動監控** — 自動啟動健康監控

**使用方式：**
```bash
# 自動化重啟（帶確認）
bash scripts/training/safe_restart_training.sh

# 查看重啟後的訓練
tmux attach -t sdxl_luca_training_safe
```

---

## 📊 監控與診斷工具

### 1. GPU 實時監控

```bash
# 每 5 秒更新一次
watch -n 5 nvidia-smi

# 簡化輸出
watch -n 5 "nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv"
```

### 2. 訓練進度檢查

```bash
# 查看最新 checkpoints
ls -lht /mnt/data/ai_data/models/lora/luca/sdxl_trial1/*.safetensors | head -5

# 查看最新 sample 圖片
ls -lt /mnt/data/ai_data/models/lora/luca/sdxl_trial1/sample/*.png | head -10

# 檢查 checkpoint 年齡
stat -c '%y' /mnt/data/ai_data/models/lora/luca/sdxl_trial1/luca_sdxl-*.safetensors | tail -1
```

### 3. Tmux Session 管理

```bash
# 列出所有 sessions
tmux ls

# 附加到訓練 session
tmux attach -t sdxl_luca_training_safe

# 離開 session（不終止）
# 按 Ctrl+B 然後按 D

# 殺掉 session
tmux kill-session -t sdxl_luca_training_safe
```

### 4. 查看訓練日誌

```bash
# 查看監控日誌
tail -f logs/training_monitor/monitor_*.log

# 查看最新的 sample 生成
ls -lt /mnt/data/ai_data/models/lora/luca/sdxl_trial1/sample/ | head -10
```

---

## 🚨 應急處理流程

### 情況 1: 訓練崩潰

**症狀：** Tmux session 仍在，但顯示錯誤訊息

**處理步驟：**
1. 檢查錯誤訊息類型（CUDA error, OOM, etc.）
2. 檢查最新 checkpoint 是否已保存
3. 使用安全重啟腳本重啟：
   ```bash
   bash scripts/training/safe_restart_training.sh
   ```

### 情況 2: 訓練卡住

**症狀：** GPU 利用率低，長時間無輸出

**處理步驟：**
1. 檢查 GPU 狀態：`nvidia-smi`
2. 檢查進程是否仍在運行：`ps aux | grep sdxl_train`
3. 檢查最後 checkpoint 時間
4. 如果超過 30 分鐘，手動殺掉並重啟：
   ```bash
   tmux kill-session -t sdxl_luca_training_safe
   bash scripts/training/safe_restart_training.sh
   ```

### 情況 3: OOM 錯誤

**症狀：** `CUDA out of memory`

**處理步驟：**
1. 進一步降低配置：
   ```toml
   gradient_accumulation_steps = 2  # 從 4 降到 2
   vae_batch_size = 1
   cache_latents_to_disk = true  # 使用磁碟快取
   ```
2. 重啟訓練

### 情況 4: 自動重啟失敗

**症狀：** 監控腳本達到最大重試次數

**處理步驟：**
1. 檢查根本問題（硬體？驅動？配置？）
2. 查看監控日誌：`cat logs/training_monitor/monitor_*.log`
3. 手動介入，調整配置後重試

---

## ✅ 最佳實踐清單

### 訓練前

- [ ] 使用穩定版配置（`sdxl_16gb_stable.toml`）
- [ ] 確認有足夠磁碟空間（每個 checkpoint ~870MB）
- [ ] 清理 GPU（無其他進程）
- [ ] 啟動健康監控

### 訓練中

- [ ] 每小時檢查一次 GPU 溫度和 VRAM
- [ ] 每 2-3 小時檢查 checkpoint 是否正常保存
- [ ] 監控日誌無異常錯誤
- [ ] GPU 利用率保持在 80-100%

### 訓練後

- [ ] 驗證所有 checkpoints 完整性
- [ ] 保留最後 3 個 checkpoints
- [ ] 測試 checkpoint 品質
- [ ] 備份最佳 checkpoint

---

## 📈 性能與穩定性對比

| 配置項目 | 優化版 | 穩定版 | 說明 |
|---------|--------|--------|------|
| `gradient_checkpointing` | ✅ true | ❌ false | 穩定版關閉避免 CUDA 錯誤 |
| `gradient_accumulation_steps` | 8 | 4 | 穩定版降低記憶體壓力 |
| `max_train_epochs` | 20 | 12 | 穩定版縮短避免長時間問題 |
| **VRAM 使用** | ~14GB | ~12GB | 穩定版更低 |
| **訓練速度** | 較快 | 較慢 | 穩定版慢 ~15% |
| **穩定性** | 中 | 高 | 穩定版更不易崩潰 |
| **推薦使用** | 短時訓練 | 長時訓練 | 穩定版適合overnight |

---

## 🔄 從舊 Checkpoint 恢復

如果需要從之前的 checkpoint 繼續訓練：

**方法 1: 手動指定 checkpoint（如果支持 resume）**
```toml
# 在配置文件中添加
network_weights = "/path/to/luca_sdxl-000004.safetensors"
```

**方法 2: 從 checkpoint 開始新訓練**
- 使用 checkpoint 作為基礎模型
- 降低學習率（避免破壞已訓練的權重）
- 縮短 epochs

---

## 📞 故障排除資源

**日誌位置：**
- 監控日誌：`logs/training_monitor/`
- 訓練輸出：tmux session 內
- Checkpoint：`/mnt/data/ai_data/models/lora/luca/sdxl_trial1/`

**常用命令：**
```bash
# 完整狀態檢查
nvidia-smi
tmux ls
ps aux | grep sdxl_train
ls -lht /mnt/data/ai_data/models/lora/luca/sdxl_trial1/*.safetensors

# 快速重啟
bash scripts/training/safe_restart_training.sh

# 啟動監控
bash scripts/monitoring/training_health_monitor.sh --session sdxl_luca_training_safe
```

---

## 🎯 未來改進計劃

- [ ] 實作 checkpoint 自動比較和品質評估
- [ ] 添加 Telegram/Discord 通知集成
- [ ] 實作訓練品質即時評估（FID/CLIP score）
- [ ] 自動調整超參數（動態 learning rate）
- [ ] 多 GPU 支援和負載平衡

---

**版本歷史：**
- v1.0.0 (2025-11-15): 初始版本，包含自動監控和重啟機制

**作者：** LLMProvider Tooling Assistant
**更新：** 2025-11-15
