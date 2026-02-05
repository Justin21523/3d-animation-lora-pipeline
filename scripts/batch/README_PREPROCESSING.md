# SDXL Image Preprocessing with Letterbox + LaMa Inpainting

## 概述

這個腳本將所有訓練圖片預處理為統一的 1024x1024 解析度，使用 **Letterbox 填充 + LaMa Inpainting** 方法：

1. **保留所有特徵**：不裁切，縮放圖片適應 1024x1024
2. **自然邊緣填充**：用 LaMa AI 模型自動填充邊緣區域，讓填充顏色與周圍融合
3. **加速訓練**：消除 bucketing overhead，預估提升 8-15 分/epoch

## 使用方式

### 基本用法（推薦）
```bash
conda run -n ai_env python scripts/batch/preprocess_images_for_sdxl.py \
  --base-dir /mnt/data/ai_data/datasets/3d-anime \
  --target-size square \
  --report logs/preprocessing_report.json
```

### 快速模式（僅黑色 letterbox，不用 LaMa）
如果想快速處理（30-60分鐘而非3-5小時），可以跳過 LaMa inpainting：
```bash
conda run -n ai_env python scripts/batch/preprocess_images_for_sdxl.py \
  --base-dir /mnt/data/ai_data/datasets/3d-anime \
  --target-size square \
  --no-lama \
  --report logs/preprocessing_report.json
```

### 不備份原始圖片
```bash
python scripts/batch/preprocess_images_for_sdxl.py \
  --no-backup \
  --report logs/preprocessing_report.json
```

## 參數說明

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--base-dir` | `/mnt/data/ai_data/datasets/3d-anime` | 包含所有電影數據集的根目錄 |
| `--target-size` | `square` | 目標解析度類型 (`square`=1024x1024, `landscape`=1152x896, `portrait`=896x1152) |
| `--no-backup` | False | 跳過備份原始圖片到 `_original/` 目錄 |
| `--no-lama` | False | 跳過 LaMa inpainting，僅使用黑色 letterbox（快速模式） |
| `--dry-run` | False | 預覽模式，不實際修改檔案 |
| `--report` | `logs/image_preprocessing_report.json` | 報告輸出路徑 |

## 處理流程

### 1. Letterbox 縮放
- 計算縮放比例，讓圖片適應 1024x1024
- 保持寬高比，不裁切
- 將縮放後的圖片居中放置在黑色 canvas 上

### 2. LaMa Inpainting 填充
- 檢測空白區域（letterbox padding）
- 使用 LaMa AI 模型自動生成自然的邊緣內容
- 讓填充區域與原始圖片邊緣融合

### 3. 儲存處理後圖片
- 覆寫原始圖片（如果 `--no-backup` 未設置，會先備份）
- 生成處理報告 JSON

## 處理時間預估

| 模式 | 圖片數量 | 預估時間 | 說明 |
|------|---------|---------|------|
| **LaMa Inpainting** | ~5000 張 | **3-5 小時** | GPU 推理，邊緣自然填充 |
| **僅 Letterbox** | ~5000 張 | **30-60 分鐘** | CPU 處理，黑色邊緣 |

*12 個角色，每個角色約 400 張圖片*

## 輸出報告範例

```json
{
  "summary": {
    "total_characters": 12,
    "total_images": 4892,
    "processed": 4651,
    "skipped": 241,
    "errors": 0,
    "success_rate": "95.1%"
  },
  "characters": [
    {
      "character_dir": ".../miguel_identity/10_miguel",
      "total_images": 449,
      "processed": 421,
      "skipped": 28,
      "errors": 0
    },
    ...
  ]
}
```

## 訓練速度改善

預處理後，訓練時的 bucketing overhead 將大幅降低：

| 指標 | 預處理前 | 預處理後 | 改善 |
|------|---------|---------|------|
| Bucketing 時間 | 每 step 多花 0.01-0.02s | 幾乎為 0 | -100% |
| 每 Epoch 時間 | 66 分鐘 | 47-51 分鐘 | **-8 to -15 分鐘** |
| 10 Epochs 總時長 | 11 小時 | 7.8-8.5 小時 | **-2.5 to -3.2 小時** |

## 自動化整合

此腳本已整合到 `monitor_epoch2_and_optimize.sh` 中，會在 Epoch 2 完成後自動執行。

## 技術細節

### Letterbox 算法
```python
def letterbox_resize(image, target_size=(1024, 1024)):
    # 1. 計算縮放比例（保持寬高比）
    scale = min(target_w / img_w, target_h / img_h)

    # 2. 縮放圖片
    new_size = (int(img_w * scale), int(img_h * scale))
    resized = image.resize(new_size, LANCZOS)

    # 3. 放置在黑色 canvas 中心
    canvas = Image.new("RGB", (1024, 1024), (0, 0, 0))
    paste_x = (1024 - new_w) // 2
    paste_y = (1024 - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))

    return canvas
```

### LaMa Inpainting
使用 `simple-lama-inpainting` 套件（已安裝）：
- 模型：LaMa (Large Mask Inpainting) - ECCV 2022
- 特點：生成式 inpainting，能自然填充大區域
- GPU 加速：RTX 4090 約 2-3 秒/張

## 疑難排解

### LaMa 模型載入失敗
如果看到警告訊息：
```
⚠️  Failed to load LaMa model. Falling back to black letterbox.
```

檢查安裝：
```bash
conda run -n ai_env pip install simple-lama-inpainting
```

### 記憶體不足
如果 GPU 記憶體不足，使用快速模式：
```bash
python scripts/batch/preprocess_images_for_sdxl.py --no-lama
```

## 參考資料

- **LaMa Paper**: "Resolution-robust Large Mask Inpainting with Fourier Convolutions" (ECCV 2022)
- **simple-lama-inpainting**: https://github.com/enesmsahin/simple-lama-inpainting
- **SDXL Bucketing**: https://github.com/kohya-ss/sd-scripts/wiki/SDXL-Bucketing
