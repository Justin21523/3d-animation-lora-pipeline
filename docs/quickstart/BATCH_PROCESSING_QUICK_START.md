# 批次處理快速開始

## ⚡ 3 分鐘快速上手

### 1. 查看待處理電影

```bash
ls /mnt/data/ai_data/datasets/3d-anime/
# 應該看到: coco, elio, luca, onward, orion, turning-red, up
```

### 2. 測試配置（Dry Run）

```bash
# 測試 SAM2 + LaMa 配置（不實際執行）
bash scripts/batch/run_batch_processing.sh configs/batch/sam2_lama.yaml --dry-run
```

### 3. 開始批次處理

#### 方法 A：智能啟動（推薦，防止 GPU 競爭）

```bash
# 自動等待 GPU 空閒後才啟動批次處理
# 適用於有其他 GPU 任務正在運行的情況（例如 Luca SAM2）
nohup bash scripts/batch/smart_batch_launcher.sh configs/batch/sam2_lama.yaml > logs/smart_batch_launcher.log 2>&1 &

# 查看等待狀態
tail -f logs/smart_batch_launcher.log
```

#### 方法 B：直接啟動（確保 GPU 空閒時使用）

```bash
# 立即執行批次處理（需確保無其他 GPU 任務）
bash scripts/batch/run_batch_processing.sh configs/batch/sam2_lama.yaml
```

**⚠️ GPU 競爭警告**：

- 如果有其他 SAM2/LaMa 任務正在運行，使用**方法 A（智能啟動）**
- 智能啟動器會每 5 分鐘檢查 GPU 狀態，等待空閒後自動開始
- 條件：GPU 記憶體 < 5GB **且** 無 SAM2/LaMa 進程運行

---

## 📊 監控進度

### 查看當前狀態

```bash
# 查看進度文件（推薦）
cat logs/batch_processing/progress.json | jq '.jobs[] | {film, name, status}'

# 或使用簡單版本
tail -f logs/batch_processing/progress.json
```

### 查看處理日誌

```bash
# 查看最新的 SAM2 日誌
tail -f logs/batch_processing/sam2_segmentation_*.log

# 查看所有日誌列表
ls -lht logs/batch_processing/
```

---

## ⚙️ 常用配置

### 只執行 SAM2 分割

```bash
bash scripts/batch/run_batch_processing.sh configs/batch/sam2_only.yaml
```

### 排除特定電影

編輯 `configs/batch/sam2_lama.yaml`:

```yaml
discovery:
  exclude:
    - "luca"    # 已處理
    - "coco"    # 跳過
```

### 從頭開始（忽略進度）

```bash
bash scripts/batch/run_batch_processing.sh configs/batch/sam2_lama.yaml --no-resume
```

---

## 🎯 預期時間

| 操作 | 6 部電影 | 單部電影平均 |
|------|---------|-------------|
| SAM2 分割 | ~47 小時 | ~8 小時 |
| LaMa Inpainting | ~9 小時 | ~1.5 小時 |
| **總計** | **~56 小時** | **~9.5 小時** |

**建議：週末啟動，週一完成！**

---

## ✅ 完成後

處理完成後，每部電影都會有：

```
/mnt/data/ai_data/datasets/3d-anime/{film}/
├── {film}_instances_sam2_v2/
│   ├── instances/        # 角色實例
│   ├── masks/            # Masks
│   └── backgrounds/      # OpenCV 初步背景
└── backgrounds_lama_v2/  # 最終高品質背景
    └── *.jpg             # 用於訓練的背景圖片
```

---

## 📚 更多資訊

詳細指南: `docs/guides/BATCH_PROCESSING_GUIDE.md`
