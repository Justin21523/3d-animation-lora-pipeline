# WSL 長時間運行任務完整指南

## 問題：WSL 可能會自動關閉

WSL 在以下情況下可能被終止：
1. Windows 更新/重啟
2. 長時間閒置後被 Windows 自動關閉
3. 電腦睡眠/休眠
4. WSL 預設的記憶體管理

## ✅ 解決方案：使用 tmux + WSL 配置

---

## 步驟 1: 配置 WSL 防止自動關閉

### 1.1 創建/編輯 `.wslconfig` 文件

在 Windows 用戶目錄創建配置文件：`C:\Users\YourUsername\.wslconfig`

```ini
[wsl2]
# 禁用記憶體回收
memory=16GB
# 禁用 swap
swap=0
# 禁用虛擬機閒置自動關閉
idleTimeout=-1
# 保持 VM 運行
vmIdleTimeout=-1
```

**創建方法（在 Windows PowerShell 中）：**
```powershell
# 打開記事本編輯配置
notepad $env:USERPROFILE\.wslconfig
```

貼上上面的配置，保存後重啟 WSL：
```powershell
wsl --shutdown
wsl
```

### 1.2 配置 Windows 電源選項

1. 打開 **設定** → **系統** → **電源**
2. 設定 **螢幕** 和 **睡眠** 為 **永不**（處理期間）
3. 高級電源設定中禁用 **USB 選擇性暫停**

---

## 步驟 2: 使用 tmux 運行任務

### 2.1 快速啟動（推薦）

```bash
cd /mnt/c/AI_LLM_projects/inazuma-eleven-lora
bash scripts/start_processing_tmux.sh
```

這會自動：
- ✅ 創建名為 `anime_processing` 的 tmux 會話
- ✅ 在分離的窗口中啟動 Yokai-Watch 處理
- ✅ 在分離的窗口中啟動 Inazuma-Eleven 特徵提取
- ✅ 提供監控窗口

### 2.2 手動使用 tmux

**創建新會話：**
```bash
tmux new -s anime_processing
```

**在 tmux 中運行任務：**
```bash
cd /mnt/c/AI_LLM_projects/inazuma-eleven-lora

# Yokai-Watch 處理
conda run -n blip2-env python3 scripts/tools/yokai_advanced_pipeline.py \
    --device cuda --resume 2>&1 | \
    tee /mnt/c/AI_LLM_projects/ai_warehouse/outputs/yokai-watch/full_processing.log
```

**分離會話（讓它在後台運行）：**
```
按 Ctrl+B，然後按 D
```

**重新連接：**
```bash
tmux attach -t anime_processing
```

---

## Tmux 常用命令

### 會話管理
```bash
# 列出所有會話
tmux ls

# 連接到會話
tmux attach -t anime_processing

# 創建新會話
tmux new -s session_name

# 殺死會話
tmux kill-session -t anime_processing

# 分離當前會話
Ctrl+B, then D
```

### 窗口管理（在 tmux 中）
```
Ctrl+B, 0-9     切換到窗口 0-9
Ctrl+B, C       創建新窗口
Ctrl+B, ,       重命名當前窗口
Ctrl+B, N       切換到下一個窗口
Ctrl+B, P       切換到上一個窗口
Ctrl+B, W       列出所有窗口
```

### 面板管理（在 tmux 中）
```
Ctrl+B, %       垂直分割
Ctrl+B, "       水平分割
Ctrl+B, ←→↑↓    切換面板
Ctrl+B, X       關閉面板
```

---

## 步驟 3: 監控任務

### 3.1 查看日誌（實時）

**Yokai-Watch：**
```bash
tail -f /mnt/c/AI_LLM_projects/ai_warehouse/outputs/yokai-watch/full_processing.log
```

**Inazuma-Eleven：**
```bash
# 檢查進度文件是否存在
ls -lh /mnt/c/AI_LLM_projects/ai_warehouse/outputs/inazuma-eleven_character_features.npz
```

### 3.2 檢查進度

**已處理的集數：**
```bash
ls /mnt/c/AI_LLM_projects/ai_warehouse/outputs/yokai-watch/multi_layer_segmentation/ | wc -l
```

**檢查點狀態：**
```bash
cat /mnt/c/AI_LLM_projects/ai_warehouse/outputs/yokai-watch/checkpoints/progress.json
```

### 3.3 檢查資源使用

**GPU：**
```bash
watch -n 1 nvidia-smi
```

**CPU/RAM：**
```bash
htop
```

**磁碟空間：**
```bash
df -h /mnt/c/AI_LLM_projects/ai_warehouse/outputs/
```

---

## 步驟 4: 處理中斷後的恢復

### 如果 WSL 被關閉或重啟：

1. **重新打開 WSL**
2. **重新連接到 tmux 會話：**
   ```bash
   tmux attach -t anime_processing
   ```

3. **如果會話不存在，重新啟動：**
   ```bash
   cd /mnt/c/AI_LLM_projects/inazuma-eleven-lora
   bash scripts/start_processing_tmux.sh
   ```

4. **系統會自動從檢查點恢復** ✅

---

## 故障排除

### 問題 1: tmux 會話消失了
**原因**: WSL 完全重啟，tmux 會話丟失

**解決**:
```bash
# 重新啟動處理（會從檢查點恢復）
bash scripts/start_processing_tmux.sh
```

### 問題 2: 處理似乎卡住了
**檢查**:
```bash
# 查看最後的日誌
tail -50 /mnt/c/AI_LLM_projects/ai_warehouse/outputs/yokai-watch/full_processing.log

# 檢查進程是否還在運行
ps aux | grep python3
```

### 問題 3: GPU 記憶體不足
**解決**:
```bash
# 在腳本中添加 --batch-size 參數
python3 scripts/tools/yokai_advanced_pipeline.py --device cuda --batch-size 4
```

---

## 最佳實踐

### ✅ DO（建議）
- ✅ 使用 tmux 運行所有長時間任務
- ✅ 定期檢查日誌和進度
- ✅ 配置 WSL 防止自動關閉
- ✅ 確保有足夠的磁碟空間（>500GB）
- ✅ 在處理期間保持 Windows 不休眠

### ❌ DON'T（避免）
- ❌ 直接在終端運行長任務（不用 tmux）
- ❌ 處理期間讓電腦睡眠
- ❌ 手動終止進程（會丟失進度）
- ❌ 同時運行多個 GPU 密集任務

---

## 預期時間線

### Yokai-Watch (1,042,599 幀)
- **總時間**: 12-24 小時
- **檢查點**: 每 100 幀自動保存
- **輸出大小**: ~200-500 GB

### Inazuma-Eleven (16,929 幀)
- **總時間**: 2-4 小時
- **輸出**: 單一 .npz 文件 (~500MB)

---

## 緊急操作

### 暫停所有處理
```bash
# 找到 tmux 會話
tmux ls

# 連接並暫停（Ctrl+Z）
tmux attach -t anime_processing
# 在窗口中按 Ctrl+Z
```

### 完全停止
```bash
# 殺死 tmux 會話
tmux kill-session -t anime_processing

# 或者手動殺死進程
pkill -f yokai_advanced_pipeline
pkill -f interactive_character_sorter
```

### 從檢查點恢復
```bash
# 系統會自動檢測並恢復，只需重新運行：
bash scripts/start_processing_tmux.sh
```

---

## 聯繫與支持

如果遇到問題：
1. 檢查日誌文件
2. 查看檢查點 JSON
3. 確認 GPU 狀態 (`nvidia-smi`)
4. 檢查磁碟空間 (`df -h`)

**所有任務都設計為可中斷和恢復！** 🛡️
