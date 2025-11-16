# True LaMa AI Inpainting 使用指南

> **重要：本指南記錄正確使用真正的 LaMa AI 模型（`simple-lama-inpainting` 庫）進行背景填補的方法**

## 核心概念

### 什麼是 True LaMa AI Inpainting？

LaMa (Large Mask Inpainting) 是基於深度學習的 AI 填補技術，能夠：
- 智能填補透明區域，生成自然的背景場景
- 在角色邊緣產生自然的羽化（feathering）效果
- 比傳統 OpenCV 方法（TELEA/NS）質量更高

### 與其他方法的差異

| 方法 | 類型 | 質量 | 速度 | 適用場景 |
|------|------|------|------|---------|
| **SimpleLama (推薦)** | AI 深度學習 | ⭐⭐⭐⭐⭐ | 慢 | 最終訓練數據集 |
| OpenCV TELEA | 傳統算法 | ⭐⭐⭐ | 快 | 快速預覽 |
| 簡單背景合成 | 顏色填充 | ⭐ | 極快 | ❌ 不適合訓練 |

## 正確的腳本和庫

### ✅ 正確的方法

**腳本位置：**
```
scripts/generic/inpainting/lama_batch_optimized.py
```

**使用的庫：**
```python
from simple_lama_inpainting import SimpleLama
```

**環境：**
- 必須在 `ai_env` conda 環境中運行
- 已安裝 `simple-lama-inpainting` 庫

### ❌ 錯誤的方法（已廢棄）

以下方法**不應該使用**：

1. **簡單背景合成** (已刪除)
   - 檔案：`/tmp/alpha_composite_simple.py` ❌
   - 問題：只做灰色背景合成，沒有 AI 填補

2. **OpenCV fallback**
   - 檔案：`scripts/generic/enhancement/inpaint_occlusions.py`
   - 問題：當偵測不到 LaMa 時會退回到 OpenCV TELEA

## 使用方法

### 1. 基本用法（聚類結構）

處理已經聚類的角色實例：

```bash
# 啟用 ai_env 環境
conda activate ai_env

# 運行 LaMa AI inpainting
python scripts/generic/inpainting/lama_batch_optimized.py \
  --input-dir /path/to/clustered_characters \
  --output-dir /path/to/output \
  --batch-size 8 \
  --device cuda
```

**預期輸入結構：**
```
input_dir/
├── character_0/
│   ├── instance_001.png  (透明 PNG)
│   ├── instance_002.png
│   └── ...
├── character_1/
│   └── ...
```

### 2. 平面目錄模式（單層 PNG 檔案）

處理單一目錄中的透明 PNG 檔案：

```bash
# 啟用 ai_env 環境
conda activate ai_env

# 使用 --flat-input 選項
python scripts/generic/inpainting/lama_batch_optimized.py \
  --input-dir /path/to/transparent_pngs \
  --output-dir /path/to/output \
  --flat-input \
  --batch-size 8 \
  --device cuda
```

**預期輸入結構：**
```
input_dir/
├── instance_001.png  (透明 PNG)
├── instance_002.png
├── instance_003.png
└── ...
```

### 3. 使用 Wrapper Script（推薦）

為了確保正確的 conda 環境，建議使用 wrapper script：

```bash
#!/usr/bin/env bash
# save as: run_lama_inpainting.sh

set -e

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate ai_env

python scripts/generic/inpainting/lama_batch_optimized.py \
  --input-dir "$1" \
  --output-dir "$2" \
  --flat-input \
  --batch-size 8 \
  --device cuda
```

使用方式：
```bash
chmod +x run_lama_inpainting.sh
./run_lama_inpainting.sh /input/dir /output/dir
```

## 參數說明

### 必要參數

- `--input-dir`: 輸入目錄（聚類結構或平面目錄）
- `--output-dir`: 輸出目錄

### 選用參數

- `--flat-input`: 啟用平面目錄模式（處理單層 PNG 檔案）
- `--batch-size`: 批次大小（預設：16）
  - 建議值：8-16（取決於 GPU VRAM）
- `--device`: 運算設備（`cuda` 或 `cpu`）
- `--skip-existing`: 跳過已存在的檔案

## 處理流程

### 內部運作

1. **載入 LaMa 模型**
   ```python
   from simple_lama_inpainting import SimpleLama
   model = SimpleLama(device="cuda")
   ```

2. **創建填補遮罩**
   - 從 alpha 通道提取遮罩
   - 閾值：alpha < 240 的區域
   - 擴張遮罩以確保平滑邊緣

3. **AI 填補**
   - 使用 LaMa 模型填補透明區域
   - 生成自然的背景場景
   - 保留角色邊緣的羽化效果

4. **輸出**
   - PNG 格式，無透明通道
   - 背景已填補，角色邊緣自然融合

## 質量驗證

### 如何確認使用了真正的 LaMa AI 模型？

檢查日誌輸出：

✅ **正確的輸出：**
```
Loading LaMa model on cuda...
✓ LaMa model loaded successfully
```

❌ **錯誤的輸出（OpenCV fallback）：**
```
⚠️  LaMa not installed, falling back to OpenCV
✓ Using OpenCV Telea inpainting (fallback)
```

### 結果質量特徵

**True LaMa AI 的結果應該有：**
- ✅ AI 生成的背景場景（模糊的自然場景）
- ✅ 角色邊緣有羽化效果（漸變融合）
- ✅ 背景與角色自然銜接

**錯誤方法的結果：**
- ❌ 單純的灰色背景
- ❌ 邊緣生硬，沒有羽化
- ❌ 明顯的背景與角色界線

## 實際案例：Luca 362 張圖像處理

### 問題背景

- 有 362 張透明 PNG 角色實例
- 需要填補背景以用於 LoRA 訓練
- 之前使用簡單背景合成導致訓練失敗

### 解決方案

```bash
# 創建 wrapper script
cat > /tmp/run_lama_ai_inpainting.sh << 'EOF'
#!/usr/bin/env bash
set -e

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate ai_env

python scripts/generic/inpainting/lama_batch_optimized.py \
  --input-dir /mnt/data/ai_data/datasets/3d-anime/luca/luca_final_pure_instances \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/luca/luca_final_362_lama_ai \
  --flat-input \
  --batch-size 8 \
  --device cuda
EOF

# 執行
chmod +x /tmp/run_lama_ai_inpainting.sh
bash /tmp/run_lama_ai_inpainting.sh
```

### 處理結果

```
======================================================================
OPTIMIZED LAMA BATCH INPAINTING
======================================================================
Input: /mnt/data/ai_data/datasets/3d-anime/luca/luca_final_pure_instances
Output: /mnt/data/ai_data/datasets/3d-anime/luca/luca_final_362_lama_ai
Device: CUDA
Batch size: 8
======================================================================

Loading LaMa model on cuda...
✓ LaMa model loaded successfully
📂 Processing flat directory

Found 362 PNG images

Inpainting:  28%|██▊       | 13/46 [02:12<04:58,  9.04s/it]
```

- 總共 46 個批次（362 ÷ 8 = 45.25）
- 每批次約 9-15 秒
- 預計完成時間：約 10-12 分鐘

## 常見問題排解

### 問題 1：❌ simple-lama-inpainting not installed

**錯誤訊息：**
```
❌ simple-lama-inpainting not installed!

Install with:
  pip install simple-lama-inpainting
```

**解決方法：**
```bash
# 在 ai_env 環境中安裝
conda activate ai_env
pip install simple-lama-inpainting
```

### 問題 2：運行時使用了錯誤的 Python 環境

**症狀：**
- 直接用 `python3` 運行找不到 `simple-lama-inpainting`
- 因為系統 Python 環境沒有安裝此庫

**解決方法：**
- 使用 `conda activate ai_env` 啟用正確環境
- 或使用 wrapper script 確保環境正確

### 問題 3：conda run 參數解析錯誤

**症狀：**
```
lama_batch_optimized.py: error: unrecognized arguments:
```

**原因：**
- `conda run` 對多行參數的處理有問題

**解決方法：**
- 使用 wrapper script（推薦）
- 或直接在 activated 環境中運行

## 性能優化

### GPU 記憶體使用

- **batch_size=8**: 約 8-10 GB VRAM
- **batch_size=16**: 約 12-16 GB VRAM
- **batch_size=4**: 約 4-6 GB VRAM

### 處理速度

- 每張圖像：約 1-2 秒（batch processing）
- 300 張圖像：約 10-15 分鐘
- 1000 張圖像：約 30-45 分鐘

## 後續步驟

處理完成後：

1. **驗證結果**
   ```bash
   # 查看幾張輸出圖像
   ls -lh /output/dir/ | head -10

   # 比較與參考結果
   # 參考：clustered_v2_inpainted/luca_human/
   ```

2. **匹配 Captions**
   - 使用現有 captions 配對
   - 或重新生成 captions

3. **創建 Kohya 訓練資料集**
   ```
   output_dataset/
   └── 10_luca/
       ├── image_001.png
       ├── image_001.txt
       ├── image_002.png
       ├── image_002.txt
       └── ...
   ```

4. **開始訓練**
   - 使用正確填補的圖像
   - 預期訓練質量顯著提升

## 相關文件

- `scripts/generic/inpainting/lama_batch_optimized.py`: 主要腳本
- `docs/guides/INPAINTING_GUIDE.md`: 一般 inpainting 概述
- `configs/stages/inpainting/`: Inpainting 配置檔案

## 版本記錄

- **v1.0** (2025-11-14): 初始版本，記錄正確的 True LaMa AI inpainting 方法
- 新增 `--flat-input` 選項支援平面目錄結構
- 廢棄簡單背景合成方法

## 總結

使用 **True LaMa AI Inpainting** 的關鍵要點：

1. ✅ 使用 `lama_batch_optimized.py` 腳本
2. ✅ 在 `ai_env` conda 環境中運行
3. ✅ 確認使用 `simple-lama-inpainting` 庫
4. ✅ 檢查日誌確認模型成功載入
5. ✅ 驗證結果有 AI 填補的背景和羽化效果

**避免使用：**
- ❌ 簡單背景合成腳本
- ❌ OpenCV fallback 模式
- ❌ 非 ai_env 環境

通過遵循本指南，您將獲得高質量的訓練數據，顯著提升 LoRA 訓練效果。
