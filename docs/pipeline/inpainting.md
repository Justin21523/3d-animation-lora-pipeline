# 🎨 Inpainting 遮擋修復指南

> **⚠️ 重要更新（2025-11-14）**
> **對於訓練數據集的背景填補，請優先使用 [True LaMa AI Inpainting 指南](TRUE_LAMA_AI_INPAINTING_GUIDE.md)**
> 該方法使用 `simple-lama-inpainting` 庫，提供最高質量的 AI 背景填補，適合最終訓練數據。
> 本指南主要針對角色遮擋修復場景。

## 📋 目錄

1. [概述](#概述)
2. [安裝與設定](#安裝與設定)
3. [快速開始](#快速開始)
4. [三種方法詳解](#三種方法詳解)
5. [角色特定 Prompts (Luca)](#角色特定-prompts-luca)
6. [進階使用](#進階使用)
7. [參數調整指南](#參數調整指南)
8. [常見問題](#常見問題)

---

## 概述

### 什麼是 Inpainting（遮擋修復）？

在 SAM2 切割角色實例時，經常會遇到以下情況：
- **角色重疊：** A 角色的手臂被 B 角色遮住
- **物體遮擋：** 角色部分被前景物體擋住
- **幀邊緣裁切：** 角色部分超出畫面

這些情況會導致切出的實例有**缺口**（黑色或透明區域），影響訓練品質。

**Inpainting 技術**可以根據周圍像素和語義資訊，智慧填補這些缺口。

---

## 安裝與設定

### 步驟 1：安裝模型

```bash
# 執行安裝腳本
chmod +x scripts/setup/install_inpainting_models.sh
bash scripts/setup/install_inpainting_models.sh
```

這會安裝：
- ✅ **OpenCV** - 傳統快速方法
- ✅ **LaMa** (lama-cleaner) - 推薦預設
- ✅ **Stable Diffusion Inpainting** - 高品質選項

### 步驟 2：驗證安裝

```bash
conda run -n ai_env python scripts/generic/enhancement/inpaint_occlusions.py --help
```

應該顯示完整的幫助訊息。

---

## 快速開始

### 最簡單的使用方式（LaMa 快速處理）

```bash
conda run -n ai_env python scripts/generic/enhancement/inpaint_occlusions.py \
  --input-dir /mnt/data/ai_data/datasets/3d-anime/luca/instances_sampled/instances \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/luca/instances_inpainted \
  --method lama \
  --occlusion-threshold 0.15
```

**參數說明：**
- `--input-dir`: SAM2 切出的實例目錄
- `--output-dir`: 修復後的輸出目錄
- `--method lama`: 使用 LaMa 方法（推薦）
- `--occlusion-threshold 0.15`: 只處理遮擋比例 >15% 的實例

**預計時間：** 26,754 個實例約需 **1-2 小時**（GPU）

---

## 三種方法詳解

### 方法 1：LaMa（推薦） ⭐

**特點：**
- 速度快（1-2秒/張）
- 品質優秀
- 無需提示詞
- 適合自動批次處理

**適用場景：**
- 輕度至中度遮擋（<30%）
- 肢體部分缺失（手臂、腿部）
- 衣服紋理填補

**使用範例：**
```bash
python scripts/generic/enhancement/inpaint_occlusions.py \
  --input-dir .../instances \
  --output-dir .../inpainted \
  --method lama \
  --occlusion-threshold 0.15 \
  --device cuda
```

**優缺點：**
```
✅ 速度快
✅ 品質穩定
✅ 無需調整參數
⚠️ 大面積遮擋可能模糊
```

---

### 方法 2：Stable Diffusion Inpainting（高品質）

**特點：**
- 品質最高
- 可用提示詞控制生成內容
- 理解語義（如"3D 角色的手臂"）
- 速度較慢（5-10秒/張）

**適用場景：**
- 重度遮擋（>30%）
- 需要重建複雜結構（臉部、手部）
- 有明確的角色/風格需求

**基本使用：**
```bash
python scripts/generic/enhancement/inpaint_occlusions.py \
  --input-dir .../instances \
  --output-dir .../inpainted \
  --method sd \
  --prompt "a 3d animated character, pixar luca style, smooth shading, natural lighting" \
  --occlusion-threshold 0.2
```

**角色特定 Prompts（進階）：**
```bash
python scripts/generic/enhancement/inpaint_occlusions.py \
  --input-dir .../instances \
  --output-dir .../inpainted \
  --method sd \
  --config configs/inpainting/luca_prompts.json \
  --auto-detect-character \
  --occlusion-threshold 0.2
```

**優缺點：**
```
✅ 品質最佳
✅ 語義理解強
✅ 可控制性高
⚠️ 速度慢
⚠️ 需要 8GB+ VRAM
⚠️ 可能產生幻覺
```

---

### 方法 3：OpenCV（備用）

**特點：**
- 速度極快（<1秒/張）
- 無需深度學習模型
- 適合小遮擋

**適用場景：**
- 極小遮擋（<10%）
- 邊緣透明像素填補
- 快速預覽

**使用範例：**
```bash
python scripts/generic/enhancement/inpaint_occlusions.py \
  --input-dir .../instances \
  --output-dir .../inpainted \
  --method cv \
  --occlusion-threshold 0.05
```

**優缺點：**
```
✅ 速度極快
✅ 無需額外模型
⚠️ 品質較差
⚠️ 只適合小面積
```

---

## 角色特定 Prompts (Luca)

### 為什麼需要角色特定 Prompts？

不同角色有不同的外觀特徵：
- **Luca (人類):** 淺色皮膚、棕色捲髮、青色條紋衫
- **Alberto (人類):** 古銅色皮膚、金色捲髮、黃色背心
- **Giulia:** 紅色捲髮、天藍色毛帽

使用**通用 prompt** 可能產生錯誤的顏色或風格。

### 自動偵測 + 角色 Prompts

```bash
python scripts/generic/enhancement/inpaint_occlusions.py \
  --input-dir /mnt/data/ai_data/datasets/3d-anime/luca/instances_sampled/instances \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/luca/instances_inpainted_sd \
  --method sd \
  --config configs/inpainting/luca_prompts.json \
  --auto-detect-character \
  --occlusion-threshold 0.25 \
  --device cuda
```

**工作流程：**
1. 從檔名偵測角色（如 `scene0123_luca_inst0.png` → "luca_human"）
2. 從配置文件讀取 Luca 的特定 prompt
3. 使用該 prompt 進行 inpainting

**配置文件範例：**
```json
{
  "character_prompts": {
    "luca_human": {
      "full_body": {
        "prompt": "a 3d animated teenage boy, slender build, fair skin with rosy cheeks, wavy dark brown hair, light teal striped shirt, blue shorts, pixar luca style, smooth shading",
        "body_parts": {
          "arms": "fair skin, slender teenage arms, smooth shading",
          "face": "fair skin, rosy cheeks, brown eyes, wavy dark brown hair, curious expression",
          "hair": "wavy dark brown hair, soft texture, pixar hair shader"
        }
      }
    }
  }
}
```

---

## 進階使用

### 場景 1：處理特定幀

```bash
# 創建幀列表
cat > high_occlusion_frames.txt << 'EOF'
scene0123_pos1_frame001_inst0.png
scene0456_pos5_frame004_inst2.png
scene0789_pos8_frame007_inst1.png
EOF

# 處理
python scripts/generic/enhancement/inpaint_occlusions.py \
  --input-dir .../instances \
  --output-dir .../inpainted_high \
  --instance-list $(cat high_occlusion_frames.txt | tr '\n' ',') \
  --method sd \
  --occlusion-threshold 0.0
```

### 場景 2：分級處理策略

```bash
# 第一階段：輕度遮擋用 LaMa（快）
python scripts/generic/enhancement/inpaint_occlusions.py \
  --input-dir .../instances \
  --output-dir .../inpainted_lama \
  --method lama \
  --occlusion-threshold 0.10 \
  --max-occlusion 0.30

# 第二階段：重度遮擋用 SD（慢但高品質）
python scripts/generic/enhancement/inpaint_occlusions.py \
  --input-dir .../instances \
  --output-dir .../inpainted_sd \
  --method sd \
  --config configs/inpainting/luca_prompts.json \
  --auto-detect-character \
  --occlusion-threshold 0.30
```

### 場景 3：查看遮擋統計

```bash
# 執行後查看報告
cat /mnt/data/ai_data/datasets/3d-anime/luca/instances_inpainted/inpainting_report.json
```

報告內容：
```json
{
  "statistics": {
    "total_instances": 26754,
    "inpainted": 3245,
    "skipped_low_occlusion": 22834,
    "failed": 12
  },
  "parameters": {
    "method": "lama",
    "occlusion_threshold": 0.15
  }
}
```

---

## 參數調整指南

### occlusion-threshold（遮擋閾值）

決定**什麼時候需要修復**：

```
閾值     說明                    建議場景
─────────────────────────────────────────
0.05    極敏感，幾乎全處理        邊緣透明像素修復
0.10    處理小遮擋              輕微重疊
0.15    標準閾值（推薦）         一般情況
0.20    只處理中度遮擋          節省時間
0.30    只處理嚴重遮擋          高品質 SD 處理
```

**計算方式：**
```
遮擋比例 = (透明或黑色像素數) / (總像素數)
```

### method（方法選擇）

```
方法      速度    品質    VRAM   適合場景
────────────────────────────────────────
lama     ★★★    ★★★   4GB    推薦預設
sd       ★      ★★★★★ 8GB    高品質需求
cv       ★★★★★  ★     0GB    快速預覽
```

---

## 常見問題

### Q1: LaMa 安裝失敗怎麼辦？

```bash
# 手動安裝
pip install lama-cleaner

# 如果還是失敗，使用 OpenCV 備用
python ... --method cv
```

### Q2: SD 提示 VRAM 不足？

**解決方案：**
1. 降低處理批次（一次處理較少實例）
2. 使用 `--device cpu`（會很慢）
3. 改用 LaMa 方法

### Q3: 處理後顏色不對？

**原因：** 通用 prompt 不符合角色特徵

**解決方案：**
```bash
# 使用角色專用配置
--config configs/inpainting/luca_prompts.json \
--auto-detect-character
```

### Q4: 如何只處理特定角色？

**方法 1：** 從檔名篩選
```bash
# 只處理包含 "luca" 的實例
find .../instances -name "*luca*.png" > luca_only.txt
--instance-list-file luca_only.txt
```

**方法 2：** 在配置中只定義該角色

### Q5: 速度太慢？

**優化建議：**
1. 提高 `--occlusion-threshold`（處理更少實例）
2. 使用 LaMa 而非 SD
3. 不保存視覺化（未來功能）
4. 平行處理（未來功能）

### Q6: 如何評估修復品質？

**建議：**
1. 先小批次測試（10-20 張）
2. 手動查看輸出
3. 調整閾值和方法
4. 再批次處理全部

---

## 完整工作流程範例

### Luca 專案完整流程

```bash
# 1. 等待 SAM2 完成
bash scripts/utils/check_sam2.sh

# 2. 安裝 inpainting 模型（一次性）
bash scripts/setup/install_inpainting_models.sh

# 3. 第一階段：快速 LaMa 處理（推薦）
conda run -n ai_env python scripts/generic/enhancement/inpaint_occlusions.py \
  --input-dir /mnt/data/ai_data/datasets/3d-anime/luca/instances_sampled/instances \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/luca/instances_inpainted \
  --method lama \
  --occlusion-threshold 0.15 \
  --device cuda

# 4. 查看報告
cat /mnt/data/ai_data/datasets/3d-anime/luca/instances_inpainted/inpainting_report.json

# 5. （可選）第二階段：SD 處理嚴重遮擋
# 找出 occlusion > 30% 的實例
python scripts/generic/enhancement/inpaint_occlusions.py \
  --input-dir /mnt/data/ai_data/datasets/3d-anime/luca/instances_sampled/instances \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/luca/instances_inpainted_sd \
  --method sd \
  --config configs/inpainting/luca_prompts.json \
  --auto-detect-character \
  --occlusion-threshold 0.30

# 6. 合併結果
# 將 SD 處理的結果複製到主目錄（覆蓋）
cp -f .../instances_inpainted_sd/inpainted/*.png \
      .../instances_inpainted/inpainted/
```

---

## 總結

### 推薦策略（Luca 專案）

**大多數情況：**
```bash
--method lama \
--occlusion-threshold 0.15
```

**高品質需求：**
```bash
--method sd \
--config configs/inpainting/luca_prompts.json \
--auto-detect-character \
--occlusion-threshold 0.25
```

**快速預覽：**
```bash
--method cv \
--occlusion-threshold 0.10
```

**預計時間（26,754 實例）：**
- LaMa: 1-2 小時
- SD: 12-24 小時
- CV: 10-20 分鐘

**推薦硬體：**
- GPU: RTX 3080 或以上
- VRAM: 12GB+（SD 方法）/ 8GB（LaMa）
- RAM: 16GB+

---

## 下一步

完成 inpainting 後：

1. **身份聚類（Identity Clustering）**
   ```bash
   python scripts/generic/clustering/character_clustering.py
   ```

2. **人工審查與標註**
   ```bash
   python scripts/generic/clustering/interactive_character_selector.py
   ```

3. **Caption 生成**
   ```bash
   python scripts/generic/training/prepare_training_data.py
   ```

4. **LoRA 訓練**
   ```bash
   conda run -n ai_env python sd-scripts/train_network.py \
     --config_file configs/3d_characters/luca.toml
   ```

享受自動修復功能！🎨✨
