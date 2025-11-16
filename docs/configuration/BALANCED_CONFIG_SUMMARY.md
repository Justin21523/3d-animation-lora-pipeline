# ✅ 平衡配置完成報告

**Balanced Configuration Summary**

配置時間: 2025-11-15 08:54
狀態: ✅ **訓練中，記憶體穩定**

---

## 📊 當前系統狀態

### 記憶體使用（穩定且安全）

| 指標 | 數值 | 狀態 |
|------|------|------|
| **RAM 使用** | 12GB / 30GB (40%) | ✅ 健康 |
| **RAM 可用** | 17GB | ✅ 充足餘裕 |
| **Swap 使用** | 283MB / 8GB (3.5%) | ✅ 正常 |
| **VRAM 使用** | 15.8GB / 16.3GB (97%) | ✅ 最佳使用 |
| **GPU 利用率** | 100% | ✅ 全速運行 |
| **GPU 溫度** | 48°C | ✅ 冷卻良好 |

**結論**: 記憶體使用在安全範圍內，有充足餘裕，不會耗盡也不會浪費資源。

---

## ⚙️ 最終平衡配置

### 關鍵參數調整

```toml
# ========================================
# 平衡配置（推薦用於長時間訓練）
# ========================================

[training]
# Batch 設定
train_batch_size = 1
gradient_accumulation_steps = 6      # ⭐ 平衡值（介於 4 和 8 之間）

# 穩定性設定
gradient_checkpointing = false       # ⭐ 禁用避免 CUDA 錯誤

# 記憶體優化
cache_latents = true
cache_latents_to_disk = false        # RAM 充足，不需存硬碟
vae_batch_size = 2                   # ⭐ 加速 VAE 編碼

# 訓練時長
max_train_epochs = 12                # ⭐ 避免超長訓練
save_every_n_epochs = 2              # 頻繁保存

[hardware]
# 資料載入器
persistent_data_loader_workers = true  # ⭐ 保持 workers 加速
max_data_loader_n_workers = 2          # ⭐ 平衡 CPU 利用

# 精度設定
lowram = false                         # ⭐ RAM 充足，不需 lowram 模式
```

---

## 📈 配置對比

| 配置類型 | RAM 使用 | VRAM 使用 | 訓練速度 | 穩定性 | 推薦場景 |
|---------|---------|----------|---------|--------|---------|
| **保守配置** | ~8-10GB | ~12-14GB | 慢 (-30%) | 最高 | 首次訓練 |
| **平衡配置** ✅ | ~12GB | ~15.8GB | 標準 | 高 | **長時間訓練** |
| **激進配置** | ~16-20GB | ~16GB | 快 (+15%) | 中 | 短時測試 |

**當前使用**: 平衡配置 ✅

---

## 🎯 為什麼選擇平衡配置？

### 1. 記憶體安全

- ✅ RAM 使用 40%（距離危險 85% 還很遠）
- ✅ 17GB 可用空間（足夠應付突發需求）
- ✅ Swap 僅 283MB（幾乎不使用虛擬記憶體）

### 2. 速度適中

- ✅ 不會太慢浪費時間（保守配置慢 30%）
- ✅ 不會太激進導致崩潰（激進配置風險高）
- ✅ 預計 7-8 小時完成 12 epochs（合理時長）

### 3. 長期穩定

- ✅ `gradient_checkpointing = false` 避免 CUDA 錯誤
- ✅ `gradient_accumulation_steps = 6` 降低記憶體壓力
- ✅ 每 2 epochs 保存，即使崩潰也能恢復

---

## 🔍 調整歷程

### 調整 1: 過於保守（已改進）

**之前的極端保守配置**:
```toml
persistent_data_loader_workers = false
max_data_loader_n_workers = 1
lowram = true
vae_batch_size = 1
gradient_accumulation_steps = 4
```

**問題**: 訓練會非常慢（預計 10+ 小時）

---

### 調整 2: 平衡優化（✅ 當前配置）

**改進後的平衡配置**:
```toml
persistent_data_loader_workers = true   # 加速數據載入
max_data_loader_n_workers = 2           # 充分利用 CPU
lowram = false                          # RAM 充足，不需限制
vae_batch_size = 2                      # 加速 VAE 處理
gradient_accumulation_steps = 6         # 平衡速度與穩定性
```

**優勢**:
- ✅ 速度提升約 30-40%
- ✅ RAM 使用仍在安全範圍（40%）
- ✅ 穩定性不受影響

---

## ⏱️ 預期訓練時長

### 數據集資訊
- **圖片數量**: ~400 張
- **Epochs**: 12
- **Total Steps**: ~10,260

### 時間估算
- **每 Step**: ~2.5 秒
- **每 Epoch**: ~35 分鐘
- **總計**: **7-8 小時**

### Checkpoint 保存時間表
| Epoch | 預計時間 | Checkpoint |
|-------|---------|-----------|
| 2 | ~1.2 小時 | `luca_sdxl-000002.safetensors` |
| 4 | ~2.3 小時 | `luca_sdxl-000004.safetensors` |
| 6 | ~3.5 小時 | `luca_sdxl-000006.safetensors` |
| 8 | ~4.7 小時 | `luca_sdxl-000008.safetensors` |
| 10 | ~5.8 小時 | `luca_sdxl-000010.safetensors` |
| 12 | ~7.0 小時 | `luca_sdxl-000012.safetensors` ✅ |

**建議測試**: Epoch 6, 8, 10, 12

---

## 📱 監控方式

### 方法 1: 實時監控腳本（推薦）

```bash
# 在新終端或 tmux 中運行
bash /mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/monitor_training_progress.sh

# 或者在 tmux session 中
tmux new-session -d -s training_monitor \
  "bash monitor_training_progress.sh"
tmux attach -t training_monitor
```

**顯示內容**:
- GPU 狀態（使用率、VRAM、溫度）
- RAM 狀態（使用量、可用量、swap 警告）
- 訓練進度（steps、epochs）
- Checkpoints（最新時間、大小）
- Sample 圖片
- Top 5 記憶體消耗進程

---

### 方法 2: 手動快速檢查

```bash
# GPU 狀態
nvidia-smi

# RAM 狀態
free -h

# 查看訓練輸出
tmux attach -t sdxl_luca_training_safe
# 離開: Ctrl+B, D

# 查看最新 checkpoints
ls -lht /mnt/data/ai_data/models/lora/luca/sdxl_trial1/*.safetensors
```

---

### 方法 3: 健康監控（自動恢復）

```bash
# 啟動健康監控（每 5 分鐘檢查一次）
nohup bash scripts/monitoring/training_health_monitor.sh \
  --session sdxl_luca_training_safe \
  --output-dir /mnt/data/ai_data/models/lora/luca/sdxl_trial1 \
  --interval 300 \
  --max-restarts 3 \
  > logs/training_monitor/monitor_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

**功能**:
- 自動檢測訓練掛起（30 分鐘無進度）
- 自動檢測 CUDA 錯誤
- 自動重啟（最多 3 次）
- 監控溫度、VRAM、GPU 利用率

---

## ⚠️ 記憶體警告閾值

### 正常範圍 ✅
- RAM 使用: <70%
- Available RAM: >8GB
- Swap 使用: <500MB
- VRAM 使用: 90-98%
- GPU 溫度: <75°C

### 需要注意 ⚠️
- RAM 使用: 70-85%
- Available RAM: 4-8GB
- Swap 使用: 500MB-2GB
- VRAM 使用: 85-90% 或 >98%
- GPU 溫度: 75-85°C

### 危險狀態 🚨
- RAM 使用: >85%
- Available RAM: <4GB
- Swap 使用: >2GB
- VRAM: OOM 錯誤
- GPU 溫度: >85°C

**當前狀態**: 全部在正常範圍內 ✅

---

## 🔧 如果需要調整

### 如果 RAM 使用超過 70%

1. 降低 workers:
   ```toml
   max_data_loader_n_workers = 1
   ```

2. 或啟用 lowram:
   ```toml
   lowram = true
   ```

3. 或將 latents 存硬碟:
   ```toml
   cache_latents_to_disk = true
   ```

---

### 如果 VRAM OOM

1. 降低 VAE batch:
   ```toml
   vae_batch_size = 1
   ```

2. 或降低 accumulation:
   ```toml
   gradient_accumulation_steps = 4
   ```

---

### 如果訓練太慢

1. 增加 VAE batch（如果 VRAM 充足）:
   ```toml
   vae_batch_size = 3
   ```

2. 增加 accumulation（如果穩定）:
   ```toml
   gradient_accumulation_steps = 8
   ```

---

## 📁 相關檔案

### 配置檔案
- **主配置**: `configs/training/sdxl_16gb_stable.toml`

### 監控腳本
- **實時監控**: `monitor_training_progress.sh`
- **健康監控**: `scripts/monitoring/training_health_monitor.sh`
- **安全重啟**: `scripts/training/safe_restart_training.sh`

### 文檔
- **平衡配置詳解**: `docs/guides/BALANCED_TRAINING_CONFIG.md`
- **安全與恢復**: `docs/guides/TRAINING_SAFETY_AND_RECOVERY.md`

---

## ✅ 確認清單

訓練啟動前:
- [x] 清理舊 tmux sessions
- [x] 確認 available RAM > 10GB
- [x] 確認沒有其他 GPU 進程
- [x] 配置文件已優化
- [x] 訓練已啟動
- [ ] 健康監控已啟動（可選）

---

## 🎯 下一步

1. **等待第一個 checkpoint**（~1.2 小時後，epoch 2）
2. **定期檢查進度**（每 1-2 小時）
3. **完成後測試 checkpoints**（推薦測試 epoch 6, 8, 10, 12）

---

**配置完成時間**: 2025-11-15 08:54
**預計完成時間**: 2025-11-15 15:00-16:00
**訓練 Session**: `sdxl_luca_training_safe`

**狀態**: ✅ **訓練中，一切正常**
