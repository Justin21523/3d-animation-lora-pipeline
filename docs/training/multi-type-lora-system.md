# 多類型 LoRA 生態系統 - 完整技術指南

## 概述

本指南說明如何利用現有技術棧（SAM2、CLIP、RTM-Pose 等）構建**多類型 LoRA 訓練數據集**，並通過 **LoRA 疊加（composition）**生成複雜場景。

---

## 🎯 LoRA 類型與應用場景

| LoRA 類型 | 訓練目標 | 數據來源 | Trigger Words | 應用場景 |
|----------|---------|---------|---------------|---------|
| **Character** | 角色外觀、服裝 | SAM2 character instances | `luca`, `boy with brown hair` | 生成特定角色 |
| **Style** | 視覺風格、渲染 | 完整幀 | `pixar style`, `3d animation` | 控制整體風格 |
| **Background** | 場景、環境 | SAM2 background layers | `portorosso`, `seaside town` | 生成特定場景 |
| **Pose/Action** | 動作、姿態 | Pose keypoints + instances | `running`, `jumping` | 控制角色動作 |
| **Expression** | 面部表情 | Face crops + emotion labels | `happy`, `surprised` | 控制表情 |
| **Lighting** | 光照氛圍 | Lighting analysis | `sunset`, `dramatic light` | 控制光影 |

---

## 📋 數據集準備流程（按類型）

### **1. 角色 LoRA（Character LoRA）** - ✅ 已實現

#### 數據來源
```bash
# 使用 SAM2 分割 character instances
python scripts/generic/segmentation/layered_segmentation.py \
  --input-dir frames/ \
  --output-dir segmented/ \
  --model sam2 \
  --extract-characters

# 使用 HDBSCAN 聚類相同角色
python scripts/generic/clustering/character_clustering.py \
  --input-dir segmented/characters/ \
  --output-dir clustered/
```

#### 訓練數據結構
```
training_data/luca_character/
├── images/
│   ├── luca_001.png  (isolated character, transparent bg)
│   ├── luca_002.png
│   └── ...
└── captions/
    ├── luca_001.txt  ("a 3d animated boy named luca, brown hair, blue eyes, striped shirt")
    └── ...
```

#### Caption 策略
- **重點描述**：角色特徵（髮色、眼睛、服裝、配飾）
- **固定前綴**：`a 3d animated character, pixar style`
- **觸發詞**：`luca`, `young boy`

---

### **2. 背景 LoRA（Background/Scene LoRA）** - 🆕 新增

#### 數據來源
```bash
# 步驟 1：使用 SAM2 分割後提取 BACKGROUND layers
python scripts/generic/segmentation/layered_segmentation.py \
  --input-dir frames/ \
  --output-dir segmented/ \
  --model sam2 \
  --extract-characters  # 會同時生成 background/

# 步驟 2：背景 inpainting（填補角色移除後的空洞）
python scripts/generic/inpainting/background_inpainting.py \
  --input-dir segmented/background/ \
  --output-dir backgrounds_clean/ \
  --model lama  # 或 powerpaint

# 步驟 3：場景聚類（按視覺相似度分組場景）
python scripts/generic/clustering/scene_clustering.py \
  --input-dir backgrounds_clean/ \
  --output-dir scene_clusters/ \
  --similarity-threshold 0.75
```

#### 訓練數據結構
```
training_data/portorosso_background/
├── images/
│   ├── scene_001.png  (clean background, no characters)
│   ├── scene_002.png
│   └── ...
└── captions/
    ├── scene_001.txt  ("italian seaside town, colorful buildings, blue sky, portorosso style")
    └── ...
```

#### Caption 策略
- **重點描述**：場景類型（室內/室外、建築風格、天氣、時間）
- **固定前綴**：`3d animated background, pixar style`
- **觸發詞**：`portorosso`, `italian seaside town`

#### 特殊處理
- **去除動態元素**：移除角色、車輛、動物
- **保持靜態環境**：建築、天空、地面、植物
- **統一分辨率**：1024×1024 或 512×512

---

### **3. 風格 LoRA（Style LoRA）** - 🆕 新增

#### 數據來源
```bash
# 使用完整幀（不分割），重點是整體視覺風格
python scripts/generic/video/universal_frame_extractor.py \
  --input video.mp4 \
  --output frames_for_style/ \
  --mode scene \
  --quality high

# 風格一致性過濾（移除異常幀、轉場效果）
python scripts/generic/quality/style_consistency_filter.py \
  --input-dir frames_for_style/ \
  --output-dir frames_style_filtered/ \
  --remove-transitions
```

#### 訓練數據結構
```
training_data/pixar_3d_style/
├── images/
│   ├── frame_001.png  (full frame, character + background)
│   ├── frame_002.png
│   └── ...
└── captions/
    ├── frame_001.txt  ("pixar style 3d animation, smooth shading, soft lighting, vibrant colors")
    └── ...
```

#### Caption 策略
- **重點描述**：渲染特性（材質、光照、色彩、細節層次）
- **固定前綴**：`pixar style`, `3d animation`, `photorealistic rendering`
- **觸發詞**：`pixar style`, `smooth shading`, `cinematic lighting`

#### 訓練建議
- **數據量**：300-500 張（涵蓋各種場景和光照）
- **多樣性**：包含不同時間、天氣、室內/室外
- **純淨度**：避免文字、UI、轉場特效

---

### **4. 動作/姿態 LoRA（Pose/Action LoRA）** - 🆕 新增

#### 數據來源
```bash
# 步驟 1：提取角色 instances（同 character LoRA）
python scripts/generic/segmentation/layered_segmentation.py \
  --input-dir frames/ \
  --output-dir segmented/ \
  --extract-characters

# 步驟 2：姿態估計（提取骨架 keypoints）
python scripts/generic/pose/pose_estimation.py \
  --input-dir segmented/characters/ \
  --output-dir pose_annotated/ \
  --model rtmpose-m \
  --save-keypoints

# 步驟 3：動作分類（基於骨架特徵聚類）
python scripts/generic/clustering/action_clustering.py \
  --input-dir pose_annotated/ \
  --output-dir action_clusters/ \
  --actions running,jumping,walking,standing
```

#### 訓練數據結構
```
training_data/luca_running_pose/
├── images/
│   ├── running_001.png  (character in running pose)
│   ├── running_002.png
│   └── ...
├── captions/
│   ├── running_001.txt  ("a boy running, dynamic pose, forward lean, arms swinging")
│   └── ...
└── poses/
    ├── running_001.json  (RTM-Pose keypoints, optional)
    └── ...
```

#### Caption 策略
- **重點描述**：動作類型、身體姿態、肢體位置
- **固定前綴**：`a 3d animated character`
- **觸發詞**：`running pose`, `jumping`, `walking`

#### 訓練建議
- **單一動作**：一個 LoRA 專注一種動作（更純粹）
- **數據量**：150-300 張（涵蓋動作的不同階段）
- **視角多樣**：包含側面、正面、斜角

---

### **5. 表情 LoRA（Expression LoRA）** - 🆕 新增

#### 數據來源
```bash
# 步驟 1：提取角色 instances
python scripts/generic/segmentation/layered_segmentation.py \
  --input-dir frames/ \
  --output-dir segmented/ \
  --extract-characters

# 步驟 2：人臉檢測與裁剪
python scripts/generic/face/face_detection.py \
  --input-dir segmented/characters/ \
  --output-dir face_crops/ \
  --model retinaface \
  --crop-margin 0.3

# 步驟 3：表情分類（使用預訓練模型或 VLM）
python scripts/generic/face/expression_classification.py \
  --input-dir face_crops/ \
  --output-dir expression_clusters/ \
  --model emotion_classifier  # 或 qwen2_vl
```

#### 訓練數據結構
```
training_data/luca_happy_expression/
├── images/
│   ├── happy_001.png  (close-up face or upper body)
│   ├── happy_002.png
│   └── ...
└── captions/
    ├── happy_001.txt  ("a boy with happy expression, wide smile, bright eyes")
    └── ...
```

#### Caption 策略
- **重點描述**：表情細節（嘴型、眼神、眉毛）
- **固定前綴**：`a 3d animated character`
- **觸發詞**：`happy expression`, `surprised face`, `sad look`

#### 訓練建議
- **臉部比例**：臉部佔圖片 40-60%（可包含上半身）
- **數據量**：100-200 張每種表情
- **純淨度**：避免遮擋、模糊、極端角度

---

### **6. 光照 LoRA（Lighting LoRA）** - 🆕 高級

#### 數據來源
```bash
# 步驟 1：提取完整幀
python scripts/generic/video/universal_frame_extractor.py \
  --input video.mp4 \
  --output frames/ \
  --mode scene

# 步驟 2：光照分析與分類
python scripts/generic/lighting/lighting_analysis.py \
  --input-dir frames/ \
  --output-dir lighting_clusters/ \
  --categories sunset,sunrise,midday,indoor,dramatic
```

#### 訓練數據結構
```
training_data/sunset_lighting/
├── images/
│   ├── sunset_001.png  (full scene with sunset lighting)
│   ├── sunset_002.png
│   └── ...
└── captions/
    ├── sunset_001.txt  ("warm sunset lighting, golden hour, soft rim light, long shadows")
    └── ...
```

#### Caption 策略
- **重點描述**：光源方向、色溫、陰影、高光
- **固定前綴**：`3d animation scene`
- **觸發詞**：`sunset lighting`, `dramatic rim light`, `soft diffused light`

---

## 🔄 LoRA 疊加技術（LoRA Composition）

### **核心原理**

Stable Diffusion 支持**同時加載多個 LoRA**，每個 LoRA 的權重獨立調整：

```python
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

# 加載多個 LoRA
pipe.load_lora_weights("luca_character.safetensors", adapter_name="character")
pipe.load_lora_weights("portorosso_background.safetensors", adapter_name="background")
pipe.load_lora_weights("running_pose.safetensors", adapter_name="pose")
pipe.load_lora_weights("happy_expression.safetensors", adapter_name="expression")

# 設置權重
pipe.set_adapters(
    ["character", "background", "pose", "expression"],
    adapter_weights=[1.0, 0.8, 0.7, 0.6]
)

# 生成圖片
prompt = "luca, running pose, happy expression, in portorosso town, sunset lighting"
image = pipe(prompt).images[0]
```

### **權重管理策略**

| LoRA 類型 | 推薦權重 | 原因 |
|----------|---------|------|
| **Character** | 1.0 | 核心要素，全權重 |
| **Background** | 0.7-0.9 | 避免過度影響角色 |
| **Pose** | 0.6-0.8 | 輔助控制，不搶主導 |
| **Expression** | 0.5-0.7 | 精細調整 |
| **Style** | 0.8-1.0 | 整體風格主導 |
| **Lighting** | 0.6-0.8 | 氛圍增強 |

### **衝突處理**

#### **1. Prompt 衝突**
❌ **錯誤示範**：
```
"luca standing in portorosso, luca running on the beach"
```
兩個矛盾的動作描述。

✅ **正確示範**：
```
"luca running on the beach in portorosso, happy expression"
```
清晰、單一的動作和場景。

#### **2. LoRA 權重衝突**
- **Character + Pose LoRA**：可能競爭身體姿態控制
  - **解決**：降低 Pose LoRA 權重（0.5-0.6）

- **Background + Style LoRA**：可能競爭整體色調
  - **解決**：Style LoRA 權重稍高（0.9），Background 降至 0.7

- **Expression + Character LoRA**：可能競爭面部細節
  - **解決**：Character LoRA 保持 1.0，Expression 降至 0.5-0.6

#### **3. 訓練數據交叉污染**
- **問題**：Character LoRA 訓練數據包含特定背景 → 難以分離
- **解決**：
  - Character LoRA：使用 **透明背景** 或 **純色背景** 圖片
  - Background LoRA：使用 **完全移除角色** 的乾淨背景

---

## 🛠️ 實戰工作流程

### **階段 1：基礎 LoRA 訓練（當前）**
✅ **Character LoRA** - 已在進行中
- 50 trials Optuna 優化
- 數據：SAM2 分割的 Luca instances
- 預期：1.5-2 天完成

### **階段 2：背景 LoRA 訓練**
```bash
# 1. 提取背景 layers（已完成分割時自動生成）
ls /mnt/data/ai_data/datasets/3d-anime/luca/segmented/background/

# 2. Background inpainting（移除角色殘留）
python scripts/generic/inpainting/background_inpainting.py \
  --input-dir segmented/background/ \
  --output-dir backgrounds_clean/ \
  --model lama

# 3. 場景聚類（按位置/風格分組）
python scripts/generic/clustering/scene_clustering.py \
  --input-dir backgrounds_clean/ \
  --output-dir scene_clusters/

# 4. 準備訓練數據
python scripts/generic/training/prepare_background_training_data.py \
  --scene-dirs scene_clusters/portorosso_town/ \
  --output-dir training_data/portorosso_background/ \
  --scene-name "portorosso" \
  --generate-captions

# 5. 訓練 Background LoRA（使用 Character LoRA 的最佳超參數）
cd /mnt/c/AI_LLM_projects/kohya_ss/sd-scripts
conda run -n kohya_ss python train_network.py \
  --dataset_config configs/training/portorosso_background.toml \
  --pretrained_model_name_or_path SD1.5.safetensors \
  --output_dir models/lora/luca/portorosso_background \
  --output_name portorosso_bg \
  --network_dim 64 \
  --learning_rate 0.0003 \
  --max_train_epochs 10 \
  # ... (使用 Character LoRA 的最佳參數)
```

### **階段 3：動作 LoRA 訓練**
```bash
# 1. 姿態估計（使用已有的 character instances）
python scripts/generic/pose/pose_estimation.py \
  --input-dir segmented/characters/ \
  --output-dir pose_annotated/ \
  --model rtmpose-m

# 2. 動作聚類
python scripts/generic/clustering/action_clustering.py \
  --input-dir pose_annotated/ \
  --output-dir action_clusters/ \
  --actions running,jumping,walking,standing

# 3. 準備訓練數據
python scripts/generic/training/prepare_pose_training_data.py \
  --action-dirs action_clusters/running/ \
  --output-dir training_data/luca_running_pose/ \
  --action-name "running"

# 4. 訓練 Pose LoRA
conda run -n kohya_ss python train_network.py \
  --dataset_config configs/training/luca_running_pose.toml \
  # ... (相同的最佳超參數)
```

### **階段 4：表情 LoRA 訓練**
```bash
# 1. 人臉檢測與裁剪
python scripts/generic/face/face_detection.py \
  --input-dir segmented/characters/ \
  --output-dir face_crops/ \
  --model retinaface

# 2. 表情分類
python scripts/generic/face/expression_classification.py \
  --input-dir face_crops/ \
  --output-dir expression_clusters/

# 3. 訓練 Expression LoRA
# ... (similar workflow)
```

### **階段 5：LoRA 組合測試**
```bash
# 使用 Python 腳本測試多 LoRA 組合
python scripts/evaluation/test_lora_composition.py \
  --character-lora models/lora/luca/luca_character.safetensors \
  --background-lora models/lora/luca/portorosso_background.safetensors \
  --pose-lora models/lora/luca/running_pose.safetensors \
  --expression-lora models/lora/luca/happy_expression.safetensors \
  --output-dir outputs/lora_composition_test/ \
  --prompts "luca running in portorosso, happy expression"
```

---

## 📊 超參數優化策略

### **選項 A：全局最佳參數（推薦）**
✅ **使用 Character LoRA 的最佳超參數訓練所有 LoRA**
- **優點**：節省時間，參數已被證明有效
- **適用**：Background, Pose, Expression LoRA
- **理由**：訓練數據結構相似（images + captions）

### **選項 B：分類型優化**
⚠️ **為每種 LoRA 類型單獨優化超參數**
- **優點**：可能獲得更優結果
- **缺點**：耗時（每種類型需 10-20 trials）
- **適用**：Style LoRA（數據特性不同）

### **建議流程**
1. ✅ Character LoRA 優化完成 → 提取全局最佳參數
2. ✅ 使用全局參數訓練 Background、Pose、Expression LoRA
3. ⚠️ 如果某種 LoRA 效果不佳 → 針對性優化（10-20 trials）

---

## 🎬 實戰案例：生成 "Luca 在 Portorosso 奔跑" 場景

### **準備工作**
```bash
# 訓練完成的 LoRA：
lora/luca/luca_character.safetensors         (weight: 1.0)
lora/luca/portorosso_background.safetensors  (weight: 0.8)
lora/luca/running_pose.safetensors           (weight: 0.7)
lora/luca/happy_expression.safetensors       (weight: 0.6)
```

### **生成腳本**
```python
import torch
from diffusers import StableDiffusionPipeline

# 加載 base model
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

# 加載多個 LoRA
pipe.load_lora_weights("lora/luca/luca_character.safetensors", adapter_name="character")
pipe.load_lora_weights("lora/luca/portorosso_background.safetensors", adapter_name="background")
pipe.load_lora_weights("lora/luca/running_pose.safetensors", adapter_name="pose")
pipe.load_lora_weights("lora/luca/happy_expression.safetensors", adapter_name="expression")

# 設置權重
pipe.set_adapters(
    ["character", "background", "pose", "expression"],
    adapter_weights=[1.0, 0.8, 0.7, 0.6]
)

# Prompt
prompt = """
a 3d animated boy named luca, brown hair, blue eyes, wearing blue striped shirt,
running pose, dynamic motion, happy expression with wide smile,
in italian seaside town portorosso, colorful buildings, blue sky,
pixar style, smooth shading, cinematic lighting, high detail
"""

negative_prompt = "blurry, low quality, distorted, ugly, bad anatomy"

# 生成圖片
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=30,
    guidance_scale=7.5,
    seed=42
).images[0]

image.save("luca_running_in_portorosso.png")
```

### **預期結果**
✅ Luca 角色特徵準確（Character LoRA）
✅ 奔跑姿態正確（Pose LoRA）
✅ 開心表情清晰（Expression LoRA）
✅ Portorosso 背景識別（Background LoRA）
✅ Pixar 風格統一（Base model + Style LoRA）

---

## 🚧 注意事項與限制

### **1. 數據分離純度**
❌ **錯誤**：Character LoRA 訓練數據包含固定背景
✅ **正確**：使用透明背景或純色背景

❌ **錯誤**：Background LoRA 訓練數據仍有角色殘留
✅ **正確**：使用 LaMa inpainting 完全移除角色

### **2. LoRA 權重平衡**
- **過高權重**：過度適應，失去泛化性
- **過低權重**：效果微弱，無法體現特徵
- **建議**：從推薦範圍開始，逐步調整

### **3. Prompt 工程**
- **過於簡單**：`luca running` → 缺乏細節，效果不佳
- **過於複雜**：250+ tokens → 超出 CLIP 限制，部分描述被忽略
- **最佳**：60-100 tokens，結構清晰

### **4. 訓練數據量**
| LoRA 類型 | 最少數據量 | 推薦數據量 | 最多數據量 |
|----------|-----------|-----------|-----------|
| Character | 150 | 300-500 | 1000 |
| Background | 100 | 200-400 | 800 |
| Pose | 100 | 150-300 | 600 |
| Expression | 80 | 100-200 | 400 |
| Style | 200 | 300-500 | 1000 |
| Lighting | 150 | 250-400 | 800 |

### **5. 模型相容性**
- **SD1.5 LoRA** ❌ 不能用於 SDXL
- **SDXL LoRA** ❌ 不能用於 SD1.5
- 需要為每種 base model 分別訓練

---

## 📂 完整目錄結構

```
/mnt/data/ai_data/
├── datasets/3d-anime/luca/
│   ├── frames/                     # 原始幀
│   ├── segmented/
│   │   ├── character/              # Character LoRA 數據源
│   │   ├── background/             # Background LoRA 數據源（需 inpainting）
│   │   └── masks/
│   ├── pose_annotated/             # Pose LoRA 數據源
│   ├── face_crops/                 # Expression LoRA 數據源
│   └── lighting_clusters/          # Lighting LoRA 數據源
│
├── training_data/luca/
│   ├── luca_character/             # Character LoRA 訓練集
│   ├── portorosso_background/      # Background LoRA 訓練集
│   ├── running_pose/               # Pose LoRA 訓練集
│   ├── happy_expression/           # Expression LoRA 訓練集
│   ├── pixar_style/                # Style LoRA 訓練集
│   └── sunset_lighting/            # Lighting LoRA 訓練集
│
└── models/lora/luca/
    ├── luca_character.safetensors          # SD1.5
    ├── luca_character_sdxl.safetensors     # SDXL
    ├── portorosso_background.safetensors
    ├── running_pose.safetensors
    ├── happy_expression.safetensors
    ├── pixar_style.safetensors
    └── sunset_lighting.safetensors
```

---

## 🎓 總結

### ✅ **可行性**
您的想法**完全可行**且**專業**！SAM2 和相關技術棧足以支撐多類型 LoRA 數據集準備。

### 🎯 **推薦優先級**
1. ✅ **Character LoRA**（進行中）→ 找到最佳超參數
2. 🔥 **Background LoRA**（高優先級）→ 場景控制最實用
3. 🔥 **Pose LoRA**（高優先級）→ 動作控制明顯提升質量
4. ⚠️ **Expression LoRA**（中優先級）→ 精細化表情
5. ⚠️ **Style LoRA**（可選）→ base model 本身已有 Pixar 風格傾向
6. ⚠️ **Lighting LoRA**（高級）→ 氛圍增強，但訓練難度較高

### 🚀 **下一步計劃**
1. **當前**：等待 Character LoRA 優化完成（1.5-2 天）
2. **第一批**：使用最佳參數訓練 Background + Pose LoRA（1-2 天）
3. **第二批**：訓練 Expression LoRA（1 天）
4. **測試**：組合 3-4 個 LoRA 生成測試圖片
5. **迭代**：根據效果調整權重和訓練策略

### 💡 **關鍵洞察**
- **LoRA 疊加** = 模組化生成控制
- **SAM2** = 多類型數據集的核心技術
- **超參數遷移** = 節省大量優化時間
- **權重管理** = 避免衝突的關鍵

---

**文檔版本**: 1.0
**最後更新**: 2025-11-12
**適用於**: Stable Diffusion 1.5 & SDXL, Kohya SS sd-scripts
