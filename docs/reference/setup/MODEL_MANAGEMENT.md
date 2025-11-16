# AI模型管理 - Warehouse統一結構

## 📂 AI Warehouse目錄結構

所有模型統一存放在：`/mnt/c/AI_LLM_projects/ai_warehouse/models/`

```
/mnt/c/AI_LLM_projects/ai_warehouse/models/
├── base/                          # 基礎生成模型
│   ├── stable-diffusion-v1-5/
│   ├── stable-diffusion-v2-1/
│   └── sd-xl-base-1.0/
│
├── lora/                          # LoRA模型輸出
│   ├── luca/                      # 按項目組織
│   │   ├── luca_human/
│   │   ├── alberto_human/
│   │   └── iterative_overnight/  # 迭代訓練輸出
│   ├── toy_story/
│   └── [other_projects]/
│
├── vlm/                           # 視覺語言模型
│   ├── Qwen2-VL-7B-Instruct/     # Caption生成
│   ├── Qwen2-VL-2B-Instruct/
│   ├── InternVL2-8B/              # 評估用（SOTA）
│   └── BLIP2/
│
├── evaluation/                    # 評估專用模型
│   ├── clip/
│   │   ├── ViT-B-32/
│   │   ├── ViT-L-14/
│   │   └── EVA-CLIP-L-14/         # SOTA CLIP
│   ├── aesthetics/
│   │   └── laion-aesthetics-v2/   # 美學評分
│   ├── face/
│   │   ├── arcface/
│   │   └── insightface/           # 角色一致性
│   └── quality/
│       └── musiq/                 # 圖像質量
│
├── segmentation/                  # 分割模型
│   ├── sam2/
│   │   ├── sam2_hiera_large/
│   │   └── sam2_hiera_base/
│   ├── isnet/
│   └── u2net/
│
├── inpainting/                    # 背景修復
│   └── lama/
│
├── embedding/                     # 特徵提取
│   ├── openclip/
│   └── siglip/
│
└── utility/                       # 輔助模型
    ├── depth/
    │   ├── zoedepth/
    │   └── midas/
    ├── pose/
    │   └── rtmpose/
    └── face_detection/
        └── retinaface/
```

---

## 🚀 模型下載腳本

### 完整下載腳本
```bash
#!/bin/bash
# 下載所有需要的模型到AI Warehouse

WAREHOUSE="/mnt/c/AI_LLM_projects/ai_warehouse/models"

echo "正在下載模型到 AI Warehouse..."

# ===== 基礎生成模型 =====
echo "1. 下載 Stable Diffusion 基礎模型..."
cd "$WAREHOUSE/base" || exit

# SD 1.5 (主要使用)
if [ ! -d "stable-diffusion-v1-5" ]; then
    git clone https://huggingface.co/runwayml/stable-diffusion-v1-5
fi

# ===== VLM模型 =====
echo "2. 下載 VLM模型 (Caption生成)..."
cd "$WAREHOUSE/vlm" || exit

# Qwen2-VL-7B (主要使用)
if [ ! -d "Qwen2-VL-7B-Instruct" ]; then
    huggingface-cli download Qwen/Qwen2-VL-7B-Instruct \
      --local-dir Qwen2-VL-7B-Instruct \
      --local-dir-use-symlinks False
fi

# InternVL2-8B (SOTA評估)
if [ ! -d "InternVL2-8B" ]; then
    huggingface-cli download OpenGVLab/InternVL2-8B \
      --local-dir InternVL2-8B \
      --local-dir-use-symlinks False
fi

# ===== 評估模型 =====
echo "3. 下載評估模型..."
cd "$WAREHOUSE/evaluation" || exit

# CLIP (基礎評估)
mkdir -p clip
cd clip
python -c "import clip; clip.load('ViT-L/14')"  # 會自動下載到cache

# LAION Aesthetics (美學評分)
cd "$WAREHOUSE/evaluation/aesthetics"
huggingface-cli download cafeai/cafe_aesthetic \
  --local-dir laion-aesthetics-v2 \
  --local-dir-use-symlinks False

# InsightFace (角色一致性)
cd "$WAREHOUSE/evaluation/face"
pip install insightface
python -c "import insightface; insightface.model_zoo.get_model('buffalo_l')"

# ===== 分割模型 =====
echo "4. 下載分割模型..."
cd "$WAREHOUSE/segmentation" || exit

# SAM2
if [ ! -d "sam2" ]; then
    git clone https://github.com/facebookresearch/segment-anything-2.git sam2
    cd sam2
    wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
fi

# ISNet (via rembg)
pip install rembg

# ===== Inpainting模型 =====
echo "5. 下載 Inpainting模型..."
cd "$WAREHOUSE/inpainting" || exit

if [ ! -d "lama" ]; then
    git clone https://github.com/advimman/lama.git
    cd lama
    curl -LJO https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip
    unzip big-lama.zip
fi

echo "✓ 所有模型下載完成！"
```

保存為：`scripts/setup/download_all_models.sh`

---

## 📝 各階段使用的模型

### Stage 1: Frame Extraction
**模型需求：** 無（使用OpenCV/ffmpeg）

---

### Stage 2: Character Segmentation
**位置：** `/mnt/c/AI_LLM_projects/ai_warehouse/models/segmentation/`

**使用模型：**
- **ISNet** (通過rembg) - 主要使用
- **SAM2** - 多人物場景
- **U²-Net** - 快速預覽

**配置示例：**
```python
# scripts/generic/segmentation/layered_segmentation.py

SEGMENTATION_MODELS = {
    'isnet': {
        'backend': 'rembg',
        'model': 'isnet-general-use',  # 自動下載到 ~/.u2net/
    },
    'sam2': {
        'checkpoint': '/mnt/c/AI_LLM_projects/ai_warehouse/models/segmentation/sam2/sam2_hiera_large.pt',
        'config': 'sam2_hiera_l.yaml'
    }
}
```

---

### Stage 3: Character Clustering
**位置：** `/mnt/c/AI_LLM_projects/ai_warehouse/models/embedding/`

**使用模型：**
- **CLIP ViT-L/14** - 視覺embedding
- **ArcFace** - 人臉識別
- **RTM-Pose** - 姿態估計（可選）

**配置示例：**
```python
# scripts/generic/clustering/character_clustering.py

CLIP_MODEL = "ViT-L/14"  # 會自動加載到 torch cache
# 或指定本地路徑：
# CLIP_MODEL_PATH = "/mnt/c/AI_LLM_projects/ai_warehouse/models/evaluation/clip/ViT-L-14"

ARCFACE_MODEL = "/mnt/c/AI_LLM_projects/ai_warehouse/models/evaluation/face/arcface"
```

---

### Stage 4: Caption Generation
**位置：** `/mnt/c/AI_LLM_projects/ai_warehouse/models/vlm/`

**使用模型：**
- **Qwen2-VL-7B-Instruct** (主要)
- **InternVL2-8B** (備選)

**配置示例：**
```python
# scripts/generic/training/qwen_caption_generator.py

MODEL_PATH = "/mnt/c/AI_LLM_projects/ai_warehouse/models/vlm/Qwen2-VL-7B-Instruct"

self.model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True  # 只使用本地文件
)
```

---

### Stage 5: LoRA Training
**位置：** `/mnt/c/AI_LLM_projects/ai_warehouse/models/base/`

**使用模型：**
- **Stable Diffusion v1.5** (主要)

**配置示例：**
```toml
# configs/projects/luca/luca_human.toml

pretrained_model_name_or_path = "/mnt/c/AI_LLM_projects/ai_warehouse/models/base/stable-diffusion-v1-5"
```

**輸出位置：**
```
/mnt/c/AI_LLM_projects/ai_warehouse/models/lora/luca/
├── luca_human/
│   ├── luca_human_v1-000015.safetensors
│   └── ...
└── iterative_overnight/
    ├── iteration_1/
    ├── iteration_2/
    └── ...
```

---

### Stage 6: LoRA Evaluation
**位置：** `/mnt/c/AI_LLM_projects/ai_warehouse/models/evaluation/`

**使用模型：**
- **CLIP ViT-L/14** - CLIP Score
- **InternVL2-8B** (SOTA升級)
- **InsightFace** - 角色一致性
- **LAION Aesthetics** - 美學評分
- **MUSIQ** - 圖像質量

**配置示例：**
```python
# scripts/evaluation/auto_lora_evaluator.py

EVALUATION_MODELS = {
    'clip': '/mnt/c/AI_LLM_projects/ai_warehouse/models/evaluation/clip/ViT-L-14',
    'internvl': '/mnt/c/AI_LLM_projects/ai_warehouse/models/vlm/InternVL2-8B',
    'aesthetics': '/mnt/c/AI_LLM_projects/ai_warehouse/models/evaluation/aesthetics/laion-aesthetics-v2',
    'insightface': '/mnt/c/AI_LLM_projects/ai_warehouse/models/evaluation/face/insightface',
}
```

---

## 🔧 路徑配置管理

### 創建全局配置

**位置：** `config/model_paths.yaml`

```yaml
# AI Warehouse模型路徑配置
# 所有腳本統一從這裡讀取路徑

warehouse_root: "/mnt/c/AI_LLM_projects/ai_warehouse/models"

base_models:
  sd_v1_5: "${warehouse_root}/base/stable-diffusion-v1-5"
  sd_v2_1: "${warehouse_root}/base/stable-diffusion-v2-1"
  sdxl: "${warehouse_root}/base/sd-xl-base-1.0"

vlm_models:
  qwen2_vl_7b: "${warehouse_root}/vlm/Qwen2-VL-7B-Instruct"
  qwen2_vl_2b: "${warehouse_root}/vlm/Qwen2-VL-2B-Instruct"
  internvl2_8b: "${warehouse_root}/vlm/InternVL2-8B"

evaluation_models:
  clip_vit_l: "${warehouse_root}/evaluation/clip/ViT-L-14"
  eva_clip: "${warehouse_root}/evaluation/clip/EVA-CLIP-L-14"
  aesthetics: "${warehouse_root}/evaluation/aesthetics/laion-aesthetics-v2"
  insightface: "${warehouse_root}/evaluation/face/insightface"
  musiq: "${warehouse_root}/evaluation/quality/musiq"

segmentation_models:
  sam2_large: "${warehouse_root}/segmentation/sam2/sam2_hiera_large.pt"
  sam2_base: "${warehouse_root}/segmentation/sam2/sam2_hiera_base.pt"
  u2net: "${warehouse_root}/segmentation/u2net"

inpainting_models:
  lama: "${warehouse_root}/inpainting/lama"

lora_output:
  base_dir: "${warehouse_root}/lora"
```

### Python讀取配置

```python
# scripts/core/utils/model_paths.py

import yaml
from pathlib import Path
from string import Template

def load_model_paths():
    """加載模型路徑配置"""
    config_path = Path(__file__).parent.parent.parent.parent / "config" / "model_paths.yaml"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 展開變量
    warehouse_root = config['warehouse_root']

    def expand_path(path_str):
        return Template(path_str).substitute(warehouse_root=warehouse_root)

    # 遞歸展開所有路徑
    def expand_dict(d):
        for key, value in d.items():
            if isinstance(value, str):
                d[key] = expand_path(value)
            elif isinstance(value, dict):
                expand_dict(value)

    expand_dict(config)

    return config

# 使用示例
MODEL_PATHS = load_model_paths()

# 在腳本中引用
from scripts.core.utils.model_paths import MODEL_PATHS

vlm_model_path = MODEL_PATHS['vlm_models']['qwen2_vl_7b']
base_model_path = MODEL_PATHS['base_models']['sd_v1_5']
```

---

## 📦 快速檢查模型

```bash
#!/bin/bash
# scripts/setup/verify_models.sh
# 驗證所有必需模型是否已下載

WAREHOUSE="/mnt/c/AI_LLM_projects/ai_warehouse/models"

echo "檢查AI Warehouse模型..."

check_model() {
    local path=$1
    local name=$2

    if [ -e "$path" ]; then
        echo "  ✓ $name"
    else
        echo "  ✗ $name (MISSING)"
    fi
}

echo ""
echo "基礎模型:"
check_model "$WAREHOUSE/base/stable-diffusion-v1-5" "SD v1.5"

echo ""
echo "VLM模型:"
check_model "$WAREHOUSE/vlm/Qwen2-VL-7B-Instruct" "Qwen2-VL-7B"
check_model "$WAREHOUSE/vlm/InternVL2-8B" "InternVL2-8B (SOTA)"

echo ""
echo "評估模型:"
check_model "$WAREHOUSE/evaluation/clip" "CLIP"
check_model "$WAREHOUSE/evaluation/aesthetics/laion-aesthetics-v2" "LAION Aesthetics"
check_model "$WAREHOUSE/evaluation/face/insightface" "InsightFace"

echo ""
echo "分割模型:"
check_model "$WAREHOUSE/segmentation/sam2" "SAM2"

echo ""
echo "Inpainting模型:"
check_model "$WAREHOUSE/inpainting/lama" "LaMa"

echo ""
echo "檢查完成！"
```

---

## 🎯 項目特定配置

### Luca項目快速啟動配置

**創建：** `configs/projects/luca/model_config.yaml`

```yaml
project: luca
style: pixar_3d

base_model: "/mnt/c/AI_LLM_projects/ai_warehouse/models/base/stable-diffusion-v1-5"

caption_model: "/mnt/c/AI_LLM_projects/ai_warehouse/models/vlm/Qwen2-VL-7B-Instruct"

evaluation_models:
  clip: "ViT-L/14"  # 會使用torch cache
  aesthetics: "/mnt/c/AI_LLM_projects/ai_warehouse/models/evaluation/aesthetics/laion-aesthetics-v2"
  insightface: "/mnt/c/AI_LLM_projects/ai_warehouse/models/evaluation/face/insightface"

segmentation_model: "isnet"  # 使用rembg (自動管理)

lora_output_dir: "/mnt/c/AI_LLM_projects/ai_warehouse/models/lora/luca"
```

---

## 💾 磁碟空間估算

| 模型類別 | 大小 | 必需 |
|---------|------|------|
| SD v1.5 | ~5GB | ✅ |
| Qwen2-VL-7B | ~15GB | ✅ |
| InternVL2-8B | ~16GB | ⚠️ SOTA升級 |
| CLIP ViT-L/14 | ~1GB | ✅ |
| SAM2 Large | ~3GB | ⚠️ 多人場景 |
| InsightFace | ~500MB | ⚠️ 高級評估 |
| LAION Aesthetics | ~300MB | ⚠️ 高級評估 |
| LaMa Inpainting | ~200MB | ⚠️ 背景修復 |

**總計：**
- **基礎配置 (必需):** ~22GB
- **完整配置 (含SOTA):** ~41GB

---

## 🚀 一鍵設置

```bash
# 1. 創建目錄結構
bash scripts/setup/create_warehouse_structure.sh

# 2. 下載基礎模型
bash scripts/setup/download_all_models.sh

# 3. 驗證安裝
bash scripts/setup/verify_models.sh

# 4. 運行測試
python scripts/setup/test_model_loading.py
```

---

## ✅ 最佳實踐

1. ✅ **統一路徑管理** - 所有模型在warehouse，通過配置文件引用
2. ✅ **版本控制** - 使用符號鏈接指向特定版本
3. ✅ **定期備份** - 重要的LoRA輸出定期備份
4. ✅ **清理cache** - 定期清理HuggingFace cache和torch cache
5. ✅ **文檔記錄** - 每個模型的用途和版本記錄在warehouse

---

**版本：** v1.0
**最後更新：** 2025-11-10
