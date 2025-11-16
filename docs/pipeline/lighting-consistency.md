# Pixar 風格光照一致性問題與解決方案

## 📋 問題描述

訓練 Pixar 3D 角色 LoRA 時，生成的圖像出現以下問題：

1. **對比度過高** - 明暗差異過大
2. **光照不均勻** - 臉部、身體不同位置的亮度差異明顯
3. **與原始電影不符** - 原片光照柔和均勻，生成圖像則有明顯「攝影感」

## 🔍 根本原因

### 1. **Stable Diffusion 基礎模型偏好**
- SD 1.5 訓練數據主要來自攝影作品和藝術創作
- 這些作品通常強調「dramatic lighting」（戲劇性光照）
- Pixar 電影使用的是「uniform film lighting」（統一電影光照）
- 兩者的光照哲學完全不同

### 2. **Caption 描述不精確**
當前 caption 示例：
```
"soft, natural lighting, highlighting the character's features"
```

問題：
- ❌ 「soft lighting」太泛泛，SD 仍會添加對比度
- ❌ 「natural lighting」暗示有方向性光源（太陽等）
- ❌ 「highlighting」會造成局部過亮
- ❌ 缺少「uniform」、「low contrast」等關鍵詞

### 3. **訓練配置**
```toml
shuffle_caption = true
keep_tokens = 3
```

- 前 3 個 token 固定：`"a 3d animated"`
- 其他部分打亂順序
- 光照描述可能被推到後面，影響力降低

## ✅ 解決方案

### **方案 1：修正 Caption（推薦）** ⭐

#### 步驟 1：運行 caption 修正腳本

```bash
# 預覽效果（不實際修改）
python scripts/training/fix_lighting_captions.py \
  /path/to/dataset/images \
  --short \
  --dry-run

# 實際執行修改（Luca）
python scripts/training/fix_lighting_captions.py \
  /mnt/data/ai_data/datasets/3d-anime/luca/curated_dataset/luca_human/images \
  --short

# 實際執行修改（Alberto）
python scripts/training/fix_lighting_captions.py \
  /mnt/data/ai_data/datasets/3d-anime/luca/curated_dataset/alberto_human/images \
  --short
```

#### 修改效果：

**修改前：**
```
a 3d animated character, 12-year-old italian pre-teen boy, ...,
soft natural lighting, highlighting the character's features...
```

**修改後：**
```
a 3d animated character, pixar uniform lighting, even illumination,
low contrast, 12-year-old italian pre-teen boy, ...
```

#### 為什麼這樣有效？

1. **「pixar uniform lighting」** - 直接告訴模型這是 Pixar 風格
2. **「even illumination」** - 強調光照均勻性
3. **「low contrast」** - 明確要求低對比度
4. **位置提前** - 放在角色描述後第一位，確保不被打亂掉

---

### **方案 2：調整訓練配置** ⭐

#### 修改 TOML 配置：

```toml
[general]
shuffle_caption = true
keep_tokens = 10  # 改為 10，保護光照描述不被打亂

# 或者
shuffle_caption = false  # 完全不打亂（可能降低泛化性）
keep_tokens = 3
```

**權衡：**
- ✅ `keep_tokens = 10`：保護光照描述，但可能過度擬合前面的詞彙
- ✅ `shuffle_caption = false`：完整保留語義，但可能降低對 prompt 變化的泛化

**推薦：** 先試 `keep_tokens = 6-8`（保留角色+光照描述）

---

### **方案 3：Post-processing（生成後處理）**

如果已經訓練完成，可以在生成後進行調整：

#### A. Inference 時添加負面 Prompt
```python
negative_prompt = (
    "dramatic lighting, harsh shadows, high contrast, "
    "strong highlights, dark shadows, spotlight, "
    "theatrical lighting, moody lighting"
)
```

#### B. 後處理降低對比度
```python
from PIL import Image, ImageEnhance

def reduce_contrast(image, factor=0.85):
    """降低圖像對比度以匹配 Pixar 風格"""
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

def match_pixar_tone(image):
    """調整為 Pixar 色調"""
    # 1. 降低對比度
    image = reduce_contrast(image, factor=0.82)

    # 2. 輕微提升亮度（避免過暗）
    brightness = ImageEnhance.Brightness(image)
    image = brightness.enhance(1.08)

    # 3. 輕微提升飽和度（Pixar 風格）
    color = ImageEnhance.Color(image)
    image = color.enhance(1.05)

    return image
```

---

### **方案 4：重新訓練 LoRA（終極方案）** 🎯

如果當前訓練結果不理想：

#### 步驟：

1. **修正所有 caption**（方案 1）
2. **調整訓練配置**（方案 2）
3. **重新訓練**
   ```bash
   python scripts/training/launch_iterative_training.py
   ```

4. **測試時使用優化的 prompt**
   ```python
   prompt = (
       "luca human, pixar uniform lighting, even illumination, "
       "low contrast, soft shadows, consistent shading, "
       "12-year-old boy, curly brown hair, striped shirt"
   )
   ```

---

## 🎯 推薦執行順序

### **當前訓練尚未完成（正在進行）：**

**選項 A：等待完成 + 測試**
1. 讓當前訓練完成（已經用了新參數）
2. 測試生成結果
3. 如果光照問題仍存在 → 執行下面的「選項 B」

**選項 B：立即修正（推薦）** ⭐
1. **停止當前訓練**
   ```bash
   pkill -f "launch_iterative_training"
   ```

2. **修正 caption**
   ```bash
   # Luca
   python scripts/training/fix_lighting_captions.py \
     /mnt/data/ai_data/datasets/3d-anime/luca/curated_dataset/luca_human/images \
     --short

   # Alberto
   python scripts/training/fix_lighting_captions.py \
     /mnt/data/ai_data/datasets/3d-anime/luca/curated_dataset/alberto_human/images \
     --short
   ```

3. **重新啟動訓練**
   ```bash
   python scripts/training/launch_iterative_training.py
   ```

4. **訓練將自動：**
   - 使用新的優化 caption
   - 繼續從 iteration 3 最佳模型
   - 使用正確的新參數

---

### **訓練已經完成：**

1. **測試當前模型**
   ```bash
   python scripts/evaluation/test_lora_checkpoints.py \
     /path/to/lora_dir \
     --prompts-with-lighting  # 測試不同的光照描述
   ```

2. **如果不滿意 → 修正 caption + 重新訓練**

3. **或使用方案 3（後處理）作為臨時方案**

---

## 📊 預期改善效果

### 修正前：
- ❌ 明暗對比明顯
- ❌ 臉部高光過亮
- ❌ 陰影過深
- ❌ 類似攝影作品的「戲劇性」

### 修正後：
- ✅ 對比度降低 20-30%
- ✅ 膚色均勻一致
- ✅ 柔和陰影
- ✅ 更接近 Pixar 電影的視覺風格

---

## 🧪 驗證方法

### A. 定性比較
生成同一 prompt 的圖像：
```python
# 測試 prompt
test_prompts = [
    "luca human, standing, neutral expression, front view",
    "luca human, smiling, close-up, three-quarter view",
    "luca human, surprised expression, full body shot"
]
```

對比：
- 原片截圖
- 修正前模型生成
- 修正後模型生成

### B. 定量測量
```python
from PIL import Image
import numpy as np

def measure_contrast(image_path):
    """測量圖像對比度"""
    img = Image.open(image_path).convert('L')  # 轉灰階
    arr = np.array(img)

    # 計算標準差（對比度指標）
    contrast = arr.std()

    # 計算動態範圍
    dynamic_range = arr.max() - arr.min()

    return {
        'contrast': contrast,
        'dynamic_range': dynamic_range,
        'mean_brightness': arr.mean()
    }

# Pixar 原片通常：
# - contrast: 25-35
# - dynamic_range: 150-180
#
# SD 生成通常：
# - contrast: 45-60（過高！）
# - dynamic_range: 200-240（過高！）
```

---

## 📝 其他注意事項

### 1. **CFG Scale 影響**
生成時降低 CFG scale 也能減少對比度：
```python
cfg_scale = 5.5  # 預設通常是 7.5
# 降低 CFG → 更柔和，但可能失去細節
```

### 2. **Sampler 選擇**
某些 sampler 產生的對比度較低：
- ✅ DPM++ 2M Karras（推薦）
- ✅ Euler a
- ❌ DDIM（對比度較高）

### 3. **LoRA Weight**
降低 LoRA weight 可能改善：
```python
lora_weight = 0.75  # 預設 1.0
# 但可能削弱角色特徵
```

---

## 🎬 總結

### 核心問題：
Stable Diffusion 的「攝影偏好」vs Pixar 的「電影統一光照」

### 最有效解決方案：
1. **優化 caption**（加入 `pixar uniform lighting, even illumination, low contrast`）
2. **調整 keep_tokens**（保護光照描述不被打亂）
3. **重新訓練**（讓模型學習正確的光照特徵）

### 如果無法重新訓練：
- 使用負面 prompt 抑制高對比度
- 後處理降低對比度
- 降低 CFG scale

---

**作者備註：**
這是 3D 動畫 LoRA 訓練的常見問題。Pixar/Disney/DreamWorks 的光照風格與攝影和插畫完全不同，需要在 caption 和配置上做針對性優化。
