# Luca Background Reprocessing Guide

## 🎯 目標

重新處理 Luca 的所有背景圖片，解決以下問題：
1. **SAM2 分割不完整** - 某些角色沒有被正確分割
2. **使用 OpenCV 而非 LaMa** - 導致幾何色塊和低品質填補
3. **Mask dilation 太小** - 角色邊緣沒有完全覆蓋

## 📋 問題根源分析

### 發現的問題：

1. **Background 使用 OpenCV TELEA inpainting**
   - 位置：`instance_segmentation.py` L518-520
   - 結果：簡單的顏色填充，出現幾何色塊
   - 應改為：LaMa inpainting

2. **Mask dilation 太小** (只有 ~10px)
   - 位置：`instance_segmentation.py` L512-513
   ```python
   kernel = np.ones((5, 5), np.uint8)
   combined_mask = cv2.dilate(combined_mask, kernel, iterations=2)
   ```
   - 應改為：20px dilation

3. **SAM2 參數偏保守**
   - 位置：`instance_segmentation.py` L78-86
   ```python
   points_per_side=20,  # 偏少
   pred_iou_thresh=0.76,  # 偏高
   stability_score_thresh=0.86,  # 偏高
   ```

## 🔧 解決方案

###方案 A: 修改現有腳本（推薦）

需要修改 `scripts/generic/segmentation/instance_segmentation.py`:

**修改 1: SAM2 參數 (L78-86)**
```python
self.mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2_model,
    points_per_side=32,  # 從 20 改為 32
    pred_iou_thresh=0.70,  # 從 0.76 降到 0.70
    stability_score_thresh=0.80,  # 從 0.86 降到 0.80
    crop_n_layers=0,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=self.min_mask_size,
    points_per_batch=192
)
```

**修改 2: Mask dilation (L512-513)**
```python
kernel = np.ones((20, 20), np.uint8)  # 從 (5,5) 改為 (20,20)
combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)  # 從 2 改為 1
```

**修改 3: 禁用 OpenCV inpainting (L504)**
- 將 `save_backgrounds=True` 改為 `save_backgrounds=False`
- 改用單獨的 LaMa inpainting 流程

### 方案 B: 兩階段處理（更簡單，推薦）

1. **階段 1: SAM2 分割（只保存 masks，不保存 backgrounds）**
   ```bash
   conda run -n ai_env python scripts/generic/segmentation/instance_segmentation.py \
       --input-dir /mnt/data/ai_data/datasets/3d-anime/luca/frames \
       --output-dir /mnt/data/ai_data/datasets/3d-anime/luca/luca_instances_sam2_v2 \
       --model-type sam2_hiera_large \
       --device cuda \
       --min-instance-size 4096 \
       --save-masks \
       --context-mode transparent
   ```
   **注意：不使用 `--save-backgrounds` flag**

2. **階段 2: LaMa Inpainting (20px dilation)**
   ```bash
   conda run -n ai_env python scripts/generic/inpainting/sam2_background_inpainting.py \
       --sam2-dir /mnt/data/ai_data/datasets/3d-anime/luca/luca_instances_sam2_v2 \
       --output-dir /mnt/data/ai_data/datasets/3d-anime/luca/backgrounds_lama_v2 \
       --method lama \
       --batch-size 8 \
       --device cuda \
       --mask-dilate 20
   ```

## ⚙️ 執行步驟

### 使用自動化腳本（最簡單）

```bash
# 執行完整的重新處理流程
bash scripts/pipelines/reprocess_luca_backgrounds.sh
```

這個腳本會：
1. 檢查環境和資料
2. 執行 SAM2 分割
3. 執行 LaMa inpainting
4. 驗證結果並生成報告

### 手動執行（分步驟）

#### 準備工作

1. **確認原始 frames 存在：**
   ```bash
   ls -lh /mnt/data/ai_data/datasets/3d-anime/luca/frames/ | head
   ```

2. **清理舊的輸出（可選）：**
   ```bash
   # 只在確認要重新處理時執行
   # rm -rf /mnt/data/ai_data/datasets/3d-anime/luca/luca_instances_sam2_v2
   ```

#### Step 1: 修改 SAM2 參數

編輯 `scripts/generic/segmentation/instance_segmentation.py`:

```bash
# 備份原檔案
cp scripts/generic/segmentation/instance_segmentation.py \
   scripts/generic/segmentation/instance_segmentation.py.backup

# 手動編輯或使用 sed
sed -i 's/points_per_side=20,/points_per_side=32,/' scripts/generic/segmentation/instance_segmentation.py
sed -i 's/pred_iou_thresh=0.76,/pred_iou_thresh=0.70,/' scripts/generic/segmentation/instance_segmentation.py
sed -i 's/stability_score_thresh=0.86,/stability_score_thresh=0.80,/' scripts/generic/segmentation/instance_segmentation.py
```

#### Step 2: 執行 SAM2 分割

```bash
conda run -n ai_env python scripts/generic/segmentation/instance_segmentation.py \
    --input-dir /mnt/data/ai_data/datasets/3d-anime/luca/frames \
    --output-dir /mnt/data/ai_data/datasets/3d-anime/luca/luca_instances_sam2_v2 \
    --model-type sam2_hiera_large \
    --device cuda \
    --min-instance-size 4096 \
    --save-masks \
    --context-mode transparent \
    --cache-clear-interval 10
```

**預估時間：** 2-4 小時（約 3-5 秒/frame）

#### Step 3: 執行 LaMa Inpainting

```bash
conda run -n ai_env python scripts/generic/inpainting/sam2_background_inpainting.py \
    --sam2-dir /mnt/data/ai_data/datasets/3d-anime/luca/luca_instances_sam2_v2 \
    --output-dir /mnt/data/ai_data/datasets/3d-anime/luca/backgrounds_lama_v2 \
    --method lama \
    --batch-size 8 \
    --device cuda \
    --mask-dilate 20
```

**預估時間：** 3-4 小時（約 2-3 秒/background）

## 📊 預期結果

### 輸出結構

```
/mnt/data/ai_data/datasets/3d-anime/luca/
├── luca_instances_sam2_v2/       # SAM2 輸出
│   ├── instances/                 # 角色 instances (透明背景)
│   ├── masks/                     # Instance masks (每個角色一個 mask)
│   ├── backgrounds/               # OpenCV inpainted backgrounds (不使用)
│   └── instances_metadata.json    # 分割統計
└── backgrounds_lama_v2/          # LaMa 處理後的最終背景
    ├── *.jpg                      # 清理後的背景圖片
    └── inpainting_metadata.json   # 處理統計
```

### 品質指標

- **SAM2 分割：**
  - 平均每 frame 檢測到 10-15 個 instances
  - Frames with multiple characters: > 70%
  - Failed frames: < 5%

- **LaMa Inpainting：**
  - Success rate: > 95%
  - Average mask coverage: 60-70%
  - Processing speed: 2-3 images/second

### 改進對比

| 指標 | 舊版 (OpenCV) | 新版 (LaMa + 20px) |
|------|--------------|-------------------|
| 分割完整度 | 中等 (漏掉部分角色) | 高 (更精細的參數) |
| Mask 覆蓋 | ~50% | ~70% |
| Inpainting 品質 | 低 (幾何色塊) | 高 (自然紋理) |
| 角色邊緣 | 有殘留 | 完全清除 |

## ⚠️ 注意事項

1. **磁碟空間：** 確保有至少 20GB 可用空間
   ```bash
   df -h /mnt/data/ai_data/
   ```

2. **GPU 記憶體：** SAM2 需要約 8-10GB VRAM
   ```bash
   nvidia-smi
   ```

3. **中斷恢復：** 兩個腳本都支援 resume，中斷後重新執行會自動跳過已處理的 frames

4. **備份：** 建議備份舊版輸出以便對比
   ```bash
   cp -r /mnt/data/ai_data/datasets/3d-anime/luca/luca_instances_sam2 \
         /mnt/data/ai_data/datasets/3d-anime/luca/luca_instances_sam2_backup
   ```

## 🔍 品質驗證

處理完成後，隨機抽查 10-20 張背景：

```bash
# 隨機抽樣
ls /mnt/data/ai_data/datasets/3d-anime/luca/backgrounds_lama_v2/*.jpg | shuf | head -10

# 對比新舊版本
feh /mnt/data/ai_data/datasets/3d-anime/luca/luca_instances_sam2/backgrounds/scene0535*.jpg \
    /mnt/data/ai_data/datasets/3d-anime/luca/backgrounds_lama_v2/scene0535*.jpg
```

檢查項目：
- ✅ 角色是否完全移除
- ✅ 背景紋理是否自然
- ✅ 沒有幾何色塊
- ✅ 邊緣沒有殘留

## 📝 配置記錄

所有優化參數已記錄在：
- `configs/stages/segmentation/sam2_luca_optimized.yaml`

供未來參考和複製使用。

## 🚀 下一步

處理完成後：
1. 組織背景圖片（按場景類型分類）
2. 創建 background LoRA 訓練配置
3. 訓練 background LoRA 模型
