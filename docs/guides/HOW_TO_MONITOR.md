# 如何監控訓練進度

**Quick Monitoring Guide**

更新: 2025-11-15

---

## 🎯 最簡單的監控方法

### 方法 1: 自動刷新監控（推薦）

```bash
bash /mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/monitor_training_progress.sh
```

**顯示內容**:
- GPU 狀態（使用率、VRAM、溫度、功耗）
- 系統 RAM（使用量、可用量）
- 訓練狀態（session 是否運行）
- 最新訓練輸出（steps、epochs）
- Checkpoints（最新 3 個，時間戳）
- Sample 圖片數量
- Top 5 記憶體消耗進程

**特點**: 每 10 秒自動刷新，按 Ctrl+C 退出

---

## 📱 其他監控方式

### 方法 2: 查看訓練實時輸出

```bash
# 進入訓練 session
tmux attach -t sdxl_luca_training_safe

# 離開但不終止訓練
# 按 Ctrl+B，然後按 D
```

---

### 方法 3: 快速狀態檢查

```bash
# GPU 狀態
nvidia-smi

# RAM 狀態
free -h

# 最新 checkpoints
ls -lht /mnt/data/ai_data/models/lora/luca/sdxl_trial1/*.safetensors | head -3

# 查看 tmux sessions
tmux ls
```

---

### 方法 4: 持續監控 GPU

```bash
# 每 5 秒自動刷新
watch -n 5 nvidia-smi
```

---

## 🔍 檢查訓練是否正常

### 健康指標

✅ **正常狀態**:
- GPU 使用率: 95-100%
- VRAM 使用: 15-16GB (90-98%)
- RAM 使用: <70% (目前 40%)
- GPU 溫度: <75°C (目前 48°C)
- 新 checkpoint: 每 2 epochs 保存一次

⚠️ **需要注意**:
- GPU 使用率 <5%（可能掛起）
- 超過 30 分鐘沒有新 checkpoint
- RAM 使用 >80%
- GPU 溫度 >80°C

---

## 📅 預期時間表

**總訓練時長**: ~7-8 小時（12 epochs）

| Epoch | 預計完成時間 | Checkpoint 名稱 |
|-------|------------|----------------|
| 2 | 啟動後 ~1.2 小時 | `luca_sdxl-000002.safetensors` |
| 4 | 啟動後 ~2.3 小時 | `luca_sdxl-000004.safetensors` |
| 6 | 啟動後 ~3.5 小時 | `luca_sdxl-000006.safetensors` |
| 8 | 啟動後 ~4.7 小時 | `luca_sdxl-000008.safetensors` |
| 10 | 啟動後 ~5.8 小時 | `luca_sdxl-000010.safetensors` |
| 12 | 啟動後 ~7.0 小時 | `luca_sdxl-000012.safetensors` ✅ |

**當前時間**: $(date '+%Y-%m-%d %H:%M:%S')
**訓練啟動**: 2025-11-15 08:54
**預計完成**: 2025-11-15 15:00-16:00

---

## 🚨 如果遇到問題

### 訓練卡住或崩潰

```bash
# 1. 檢查最新 checkpoint 時間
stat /mnt/data/ai_data/models/lora/luca/sdxl_trial1/*.safetensors

# 2. 查看訓練 session
tmux attach -t sdxl_luca_training_safe

# 3. 如果確認卡住，使用安全重啟
bash /mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/scripts/training/safe_restart_training.sh
```

---

### 記憶體不足

```bash
# 檢查記憶體狀態
free -h

# 如果 Available < 4GB，檢查並清理多餘進程
ps aux --sort=-%mem | head -15
```

---

## 💡 快捷命令（可添加到 ~/.bashrc）

```bash
# 添加到 ~/.bashrc
alias monitor-training='bash /mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/monitor_training_progress.sh'
alias check-gpu='nvidia-smi'
alias check-checkpoints='ls -lht /mnt/data/ai_data/models/lora/luca/sdxl_trial1/*.safetensors | head -5'
alias attach-training='tmux attach -t sdxl_luca_training_safe'

# 重新載入配置
source ~/.bashrc

# 之後可以直接使用
monitor-training
check-gpu
check-checkpoints
attach-training
```

---

## 📊 當前系統狀態

**已清理的進程**:
- ✅ 舊的 frame extraction sessions (已完成)
- ✅ 舊的 LaMa inpainting 進程 (已完成)

**當前運行**:
- ✅ SDXL LoRA 訓練 (session: `sdxl_luca_training_safe`)

**記憶體狀態**:
- RAM: 12GB / 30GB (40%) ✅
- Available: 17GB ✅
- VRAM: 15.8GB / 16.3GB (97%) ✅
- GPU Util: 100% ✅
- GPU Temp: 48°C ✅

**結論**: 系統健康，可以放心讓訓練繼續運行。

---

**更新**: 2025-11-15
**作者**: LLMProvider Tooling Assistant
