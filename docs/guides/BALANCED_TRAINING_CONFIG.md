# 平衡訓練配置指南

**Balanced Training Configuration Guide**

Created: 2025-11-15
Version: 1.0.0

---

## 📋 概述

本指南說明如何平衡 **穩定性**、**速度** 和 **記憶體使用**，確保訓練可以長時間運行而不會崩潰或耗盡資源。

---

## 🎯 配置目標

### 平衡原則

1. **記憶體使用目標**: 60-70% RAM，95% VRAM
2. **訓練速度**: 不要太慢（避免浪費時間）
3. **穩定性**: 可連續運行 8-12 小時不崩潰
4. **恢復能力**: 每 2 epochs 保存 checkpoint

### 實測結果（30GB RAM，16GB VRAM 系統）

| 指標 | 目標範圍 | 實際表現 |
|------|---------|---------|
| RAM 使用 | 60-70% | 40% (12GB/30GB) ✅ |
| Available RAM | >8GB | 17GB ✅ |
| VRAM 使用 | 90-98% | 97% (15.8GB/16.3GB) ✅ |
| GPU 利用率 | >95% | 99% ✅ |
| GPU 溫度 | <80°C | 47°C ✅ |
| Swap 使用 | <1GB | 283MB ✅ |

**結論**: 當前配置非常健康，記憶體有充足餘裕，不會過載也不會浪費資源。

---

## ⚙️ 關鍵配置參數

### 1. Batch 與 Accumulation

```toml
train_batch_size = 1
gradient_accumulation_steps = 6  # ⭐ 平衡值
```

**解釋**:
- `train_batch_size = 1`: 必須為 1（16GB VRAM 限制）
- `gradient_accumulation_steps`:
  - **4** = 太保守，訓練很慢
  - **6** = 平衡，速度與穩定性兼顧 ✅
  - **8** = 較快，但記憶體壓力大，長時間運行可能出錯

**有效 batch size** = 1 × 6 = 6（足夠穩定訓練）

---

### 2. Gradient Checkpointing

```toml
gradient_checkpointing = false  # ⭐ 禁用
```

**原因**:
- 在 WSL2 + CUDA 環境下，長時間運行（>6 小時）會導致 `CUDA unknown error`
- 禁用後犧牲少量 VRAM（~2GB），換取穩定性

**如果你有更多 VRAM（24GB+）**: 可以啟用以進一步降低記憶體使用

---

### 3. VAE Batch Size

```toml
vae_batch_size = 2  # ⭐ 平衡值
```

**解釋**:
- **1** = 太慢，VAE 編碼是瓶頸
- **2** = 平衡，快 2 倍且安全 ✅
- **4+** = 可能導致 VRAM OOM

---

### 4. Data Loader Workers

```toml
persistent_data_loader_workers = true
max_data_loader_n_workers = 2  # ⭐ 平衡值
```

**解釋**:
- `persistent_data_loader_workers = true`: 保持 workers 在記憶體（加速）
- `max_data_loader_n_workers`:
  - **1** = 太慢，數據載入是瓶頸
  - **2** = 平衡，充分利用 CPU ✅
  - **4+** = RAM 消耗增加，但速度提升有限

**記憶體影響**: 每個 worker ~500MB，2 個 workers = 1GB（可接受）

---

### 5. Low RAM Mode

```toml
lowram = false  # ⭐ 禁用（我們有 30GB RAM）
```

**何時啟用**:
- 系統 RAM < 24GB
- Available RAM < 8GB
- 開始使用 swap (>500MB)

**我們的情況**: 有 17GB 可用 RAM，不需要 `lowram` 模式

---

### 6. Latents Caching

```toml
cache_latents = true
cache_latents_to_disk = false
```

**解釋**:
- `cache_latents = true`: 預先編碼所有圖片（大幅加速）
- `cache_latents_to_disk = false`: 保存在 RAM 而非硬碟
  - 需要額外 ~3-5GB RAM
  - 我們有 17GB 可用，完全足夠 ✅

**如果 RAM 不足**: 改為 `cache_latents_to_disk = true`（會變慢）

---

### 7. Training Duration

```toml
max_train_epochs = 12
save_every_n_epochs = 2
save_last_n_epochs = 3
```

**解釋**:
- `max_train_epochs = 12`: 縮短訓練避免長時間運行問題
  - 原本 20 epochs 在 epoch 4 (6.5 小時) 就崩潰
  - 12 epochs 預計 8-10 小時可完成
- `save_every_n_epochs = 2`: 頻繁保存，方便恢復
- `save_last_n_epochs = 3`: 保留最後 3 個 checkpoints（2.6GB 磁碟空間）

---

## 📊 配置對比表

| 配置項目 | 保守版 | **平衡版** ✅ | 激進版 |
|---------|-------|------------|--------|
| `gradient_accumulation_steps` | 4 | **6** | 8 |
| `vae_batch_size` | 1 | **2** | 4 |
| `max_data_loader_n_workers` | 1 | **2** | 4 |
| `persistent_data_loader_workers` | false | **true** | true |
| `lowram` | true | **false** | false |
| `gradient_checkpointing` | false | **false** | true |
| **預期 RAM 使用** | 8-10GB | **12-14GB** | 16-20GB |
| **預期 VRAM 使用** | 12-13GB | **15-16GB** | 16GB+ (OOM) |
| **訓練速度** | 慢 (-30%) | **標準** | 快 (+15%) |
| **穩定性** | 非常高 | **高** | 中 |

**推薦**: 平衡版（當前配置）

---

## 🔍 記憶體監控指標

### 安全範圍

| 指標 | 安全範圍 | 警告閾值 | 危險閾值 |
|------|---------|---------|---------|
| RAM 使用率 | <70% | 70-85% | >85% |
| Available RAM | >8GB | 4-8GB | <4GB |
| VRAM 使用率 | 90-98% | 85-90% 或 >98% | OOM 錯誤 |
| Swap 使用 | <500MB | 500MB-2GB | >2GB |
| GPU 溫度 | <75°C | 75-85°C | >85°C |

### 當前狀態 ✅

- RAM: 40% (12GB/30GB) - **非常健康**
- Available: 17GB - **充足餘裕**
- VRAM: 97% (15.8GB/16.3GB) - **最佳使用**
- Swap: 283MB - **正常**
- GPU Temp: 47°C - **冷卻良好**

---

## ⚠️ 常見問題與調整

### 問題 1: RAM 使用超過 80%

**症狀**: `free -h` 顯示 available < 6GB

**解決方案**:
```toml
# 方案 A: 降低 workers
max_data_loader_n_workers = 1
persistent_data_loader_workers = false

# 方案 B: 啟用 lowram
lowram = true

# 方案 C: Latents 存硬碟
cache_latents_to_disk = true
```

### 問題 2: VRAM OOM 錯誤

**症狀**: `RuntimeError: CUDA out of memory`

**解決方案**:
```toml
# 方案 A: 降低 VAE batch
vae_batch_size = 1

# 方案 B: 降低 accumulation
gradient_accumulation_steps = 4

# 方案 C: 啟用 gradient checkpointing（有風險）
gradient_checkpointing = true
```

### 問題 3: 訓練速度太慢

**症狀**: 每個 step 超過 3 秒

**可能原因與解決**:
1. **workers 太少**:
   ```toml
   max_data_loader_n_workers = 2  # 增加到 2
   persistent_data_loader_workers = true
   ```

2. **VAE batch 太小**:
   ```toml
   vae_batch_size = 2  # 從 1 增加到 2
   ```

3. **lowram 模式拖慢速度**:
   ```toml
   lowram = false  # 如果 RAM > 20GB
   ```

### 問題 4: 訓練卡住或掛起

**症狀**: GPU 利用率突然降到 0-5%，超過 30 分鐘無新 checkpoint

**檢查**:
```bash
# 查看最新 checkpoint 時間
stat /mnt/data/ai_data/models/lora/luca/sdxl_trial1/*.safetensors

# 查看訓練輸出
tmux attach -t sdxl_luca_training_safe
```

**解決**: 手動重啟或等待健康監控自動重啟

---

## 🚀 效能優化建議

### 如果你有更多資源

**如果 RAM > 48GB**:
```toml
max_data_loader_n_workers = 4
vae_batch_size = 4
```

**如果 VRAM > 20GB**:
```toml
gradient_checkpointing = true  # 可以啟用
train_batch_size = 2           # 可以增加
gradient_accumulation_steps = 4  # 相應降低
```

### 如果資源更少

**如果 RAM < 24GB**:
```toml
lowram = true
max_data_loader_n_workers = 1
persistent_data_loader_workers = false
cache_latents_to_disk = true
```

**如果 VRAM < 12GB**:
無法訓練 SDXL，請改用 SD 1.5

---

## 📈 預期訓練時長

### 以 Luca 數據集為例

**數據集大小**: ~400 圖片
**Epochs**: 12
**Total steps**: ~10,260 steps

**平衡配置預期時長**:
- 每 step: ~2.5 秒
- 每 epoch: ~35 分鐘
- **總計: 7-8 小時**

**與其他配置對比**:
- 保守配置 (accumulation=4): ~10 小時
- 激進配置 (accumulation=8): ~6 小時（但風險高）

---

## ✅ 最佳實踐

### 訓練前

1. ✅ 檢查 available RAM > 10GB
2. ✅ 確認沒有其他 GPU 進程
3. ✅ 清理舊的 tmux sessions
4. ✅ 啟動健康監控

### 訓練中

1. ✅ 每 1-2 小時檢查 GPU/RAM 狀態
2. ✅ 每 2 epochs 確認新 checkpoint 已保存
3. ✅ 監控 swap 使用（應保持 <500MB）
4. ✅ 如果 RAM 使用 >80%，考慮重啟並降低配置

### 訓練後

1. ✅ 測試所有 checkpoints（尤其是最後 3 個）
2. ✅ 保留最佳 checkpoint，刪除其他
3. ✅ 記錄訓練時長和配置供未來參考

---

## 🔄 動態調整策略

### 階段 1: 保守開始

首次訓練使用保守配置（確保成功）:
```toml
gradient_accumulation_steps = 4
vae_batch_size = 1
max_data_loader_n_workers = 1
```

### 階段 2: 逐步提升

如果訓練順利（2-3 epochs 無問題），逐步調高:
```toml
gradient_accumulation_steps = 6  # +50%
vae_batch_size = 2               # +100%
max_data_loader_n_workers = 2    # +100%
```

### 階段 3: 找到極限

繼續小幅提升直到遇到問題，然後回退一步。

---

## 📞 故障排除資源

**監控腳本**:
```bash
# 實時監控（推薦）
bash /mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/monitor_training_progress.sh

# 快速狀態
bash /tmp/quick_status.sh

# GPU 實時
watch -n 5 nvidia-smi
```

**健康監控**:
```bash
# 啟動自動監控
bash scripts/monitoring/training_health_monitor.sh \
  --session sdxl_luca_training_safe \
  --output-dir /mnt/data/ai_data/models/lora/luca/sdxl_trial1 \
  --interval 300 \
  --max-restarts 3 &
```

**安全重啟**:
```bash
# 如果訓練出問題
bash scripts/training/safe_restart_training.sh
```

---

## 📝 配置檔案位置

**當前使用**: `/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/configs/training/sdxl_16gb_stable.toml`

**完整配置**: 見 `configs/training/sdxl_16gb_stable.toml`

**相關文檔**:
- `docs/guides/TRAINING_SAFETY_AND_RECOVERY.md` - 安全措施與恢復
- `docs/guides/MONITORING_GUIDE.md` - 監控指南（本文檔）

---

**版本歷史**:
- v1.0.0 (2025-11-15): 初始版本，基於 30GB RAM / 16GB VRAM 系統實測

**作者**: LLMProvider Tooling Assistant
**更新**: 2025-11-15
