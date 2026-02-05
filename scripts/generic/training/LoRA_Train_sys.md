# LoRA Training Data Preparation System

**完整的模組化 LoRA 訓練數據準備系統**

Version: 2.0 (Modular Architecture)

## 🎯 概述

這是一個完全模組化、可配置的 LoRA 訓練數據準備系統，支持：
- **5 種 LoRA 類型**: Character, Pose, Expression, Background, Style
- **5 種 Feature Extractors**: CLIP, EVA-CLIP, DINOv2, SigLIP, InternVL2
- **5 種 Clustering 算法**: HDBSCAN, KMeans, Spectral, Agglomerative, DBSCAN
- **4 種 Caption Engines**: Template, Qwen2-VL, InternVL2, LLMProvider API
- **3 種 Quality Filters**: Blur, Size, Perceptual Hash Deduplication
- **7 種預設配置**: character, pose, expression, background, style, high_quality, fast

## 📁 目錄結構

```
scripts/generic/training/
├── base/                      # 抽象基類
│   ├── feature_extractor.py  # BaseFeatureExtractor
│   ├── clusterer.py           # BaseClusterer
│   ├── caption_engine.py      # BaseCaptionEngine
│   ├── quality_filter.py      # BaseQualityFilter
│   └── processor.py           # BaseProcessor
│
├── feature_extractors/        # 特徵提取器實現
│   ├── clip_extractor.py
│   ├── eva_clip_extractor.py
│   ├── dinov2_extractor.py
│   ├── siglip_extractor.py
│   └── internvl2_extractor.py
│
├── clusterers/                # 聚類算法實現
│   ├── hdbscan_clusterer.py
│   ├── kmeans_clusterer.py
│   ├── spectral_clusterer.py
│   ├── agglomerative_clusterer.py
│   └── dbscan_clusterer.py
│
├── caption_engines/           # Caption 生成引擎
│   ├── template_engine.py
│   ├── qwen2_vl_engine.py
│   ├── internvl2_engine.py
│   └── llm_provider_api_engine.py
│
├── quality_filters/           # 質量過濾器
│   ├── blur_filter.py
│   ├── size_filter.py
│   └── perceptual_hash_deduplicator.py
│
├── preparers/                 # 端到端準備器
│   ├── character_lora_preparer.py
│   ├── pose_lora_preparer.py
│   ├── expression_lora_preparer.py
│   ├── background_lora_preparer.py
│   ├── style_lora_preparer.py
│   └── README.md             # 詳細使用文檔
│
├── config/                    # 配置系統
│   ├── schema.py             # Schema validation
│   ├── presets.py            # 預設配置
│   ├── loader.py             # Config I/O
│   └── README.md             # 配置文檔
│
└── README.md                  # 本文件
```

## 🚀 快速開始

### 1. 使用預設配置（推薦）

```python
from config import get_preset
from preparers import CharacterLoRAPreparer

# 獲取 character LoRA 預設配置
config = get_preset('character')

# 創建 preparer
preparer = CharacterLoRAPreparer(
    input_dir='/data/miguel_images',
    output_dir='/data/miguel_lora',
    character_name='miguel',
    config=config
)

# 執行準備流程
metadata = preparer.prepare()
```

### 2. 使用 CLI

```bash
python scripts/generic/training/preparers/character_lora_preparer.py \
  --input-dir /data/miguel_images \
  --output-dir /data/miguel_lora \
  --character-name miguel \
  --feature-extractor clip \
  --clusterer hdbscan \
  --caption-engine template \
  --device cuda
```

### 3. 自定義配置

```python
from config import get_preset, merge_configs
from preparers import CharacterLoRAPreparer

# 從預設開始
base_config = get_preset('character')

# 覆蓋特定設定
overrides = {
    'repeats': 15,
    'feature_extractor': {'type': 'internvl2'},
    'caption_engine': {'type': 'qwen2_vl'},
    'quality_filters': [
        {'type': 'blur', 'threshold': 120.0},
        {'type': 'size', 'min_width': 512, 'min_height': 512},
        {'type': 'dedup', 'threshold': 5}
    ]
}

config = merge_configs(base_config, overrides)

# 使用自定義配置
preparer = CharacterLoRAPreparer(
    input_dir='/data/miguel_images',
    output_dir='/data/miguel_lora',
    character_name='miguel',
    config=config
)
preparer.prepare()
```

## 📚 詳細文檔

### 主要文檔

- **[Preparers README](preparers/README.md)** - 所有 preparers 的詳細使用說明
- **[Config README](config/README.md)** - 配置系統完整指南

### 快速參考

#### 可用的 Preparers

| Preparer | 用途 | 默認設置 |
|----------|------|----------|
| `CharacterLoRAPreparer` | 角色身份 LoRA | HDBSCAN, min_cluster_size=12, repeats=10 |
| `PoseLoRAPreparer` | 姿勢 LoRA | HDBSCAN, min_cluster_size=10, repeats=10, 384x384 |
| `ExpressionLoRAPreparer` | 表情 LoRA | HDBSCAN, min_cluster_size=8, repeats=10, strict blur |
| `BackgroundLoRAPreparer` | 背景/場景 LoRA | HDBSCAN, min_cluster_size=15, repeats=5, 512x512 |
| `StyleLoRAPreparer` | 風格 LoRA | KMeans, n_clusters=5, repeats=8, 512x512 |

#### 可用的預設配置

| Preset | 描述 | 適用場景 |
|--------|------|----------|
| `character` | 角色身份 LoRA 默認配置 | 學習特定角色外觀 |
| `pose` | 姿勢 LoRA 默認配置 | 學習角色姿勢和身體位置 |
| `expression` | 表情 LoRA 默認配置 | 學習面部表情 |
| `background` | 背景 LoRA 默認配置 | 學習環境和場景 |
| `style` | 風格 LoRA 默認配置 | 學習渲染風格 |
| `high_quality` | 高質量配置（VLM captions） | 生產環境，追求最佳品質 |
| `fast` | 快速測試配置 | 快速迭代和原型開發 |

## 🎨 使用範例

### 範例 1: Character LoRA（基礎）

```python
from config import get_preset
from preparers import CharacterLoRAPreparer

config = get_preset('character')

preparer = CharacterLoRAPreparer(
    input_dir='/data/elio/bryce',
    output_dir='/data/elio/bryce_lora',
    character_name='bryce',
    config=config
)

metadata = preparer.prepare()
print(f"Dataset: {metadata['dataset_info']['dataset_dir']}")
print(f"Images: {metadata['dataset_info']['num_images']}")
print(f"Clusters: {metadata['dataset_info']['num_clusters']}")
```

### 範例 2: Expression LoRA（高質量）

```python
from config import get_preset, merge_configs
from preparers import ExpressionLoRAPreparer

# 使用高質量預設
config = get_preset('high_quality')

# 針對表情調整
overrides = {
    'repeats': 10,
    'clusterer': {'min_cluster_size': 8}  # 較小的簇用於表情
}

config = merge_configs(config, overrides)

preparer = ExpressionLoRAPreparer(
    input_dir='/data/elio/expressions',
    output_dir='/data/elio/expression_lora',
    character_name='elio',
    config=config
)

preparer.prepare()
```

### 範例 3: Background LoRA（從文件加載配置）

```python
from config import load_config, validate_config
from preparers import BackgroundLoRAPreparer

# 從 JSON 文件加載配置
config = load_config('configs/beach_background.json')

# 驗證配置
errors = validate_config(config)
if errors:
    raise ValueError(f"Config errors: {errors}")

preparer = BackgroundLoRAPreparer(
    input_dir='/data/beach_scenes',
    output_dir='/data/beach_lora',
    scene_name='tropical_beach',
    config=config
)

preparer.prepare()
```

### 範例 4: 批量處理多個角色

```python
from config import get_preset
from preparers import CharacterLoRAPreparer

config = get_preset('character')

characters = [
    ('bryce', '/data/elio/bryce'),
    ('caleb', '/data/elio/caleb'),
    ('elio', '/data/elio/elio_main')
]

for char_name, input_dir in characters:
    print(f"\n{'='*60}")
    print(f"Processing: {char_name}")
    print(f"{'='*60}")

    preparer = CharacterLoRAPreparer(
        input_dir=input_dir,
        output_dir=f'/data/elio/{char_name}_lora',
        character_name=char_name,
        config=config
    )

    metadata = preparer.prepare()
    print(f"✓ {char_name}: {metadata['dataset_info']['num_images']} images")
```

## 🔧 配置系統

### Schema Validation

所有配置會自動根據 schema 驗證：

```python
from config import validate_config

config = {
    'device': 'cuda',
    'batch_size': 32,
    'feature_extractor': {'type': 'clip'},
    'clusterer': {'type': 'hdbscan', 'min_cluster_size': 12}
}

errors = validate_config(config)
if errors:
    print("Config errors:", errors)
else:
    print("✓ Config valid!")
```

### 配置覆蓋（Merging）

```python
from config import get_preset, merge_configs

# 基礎配置
base = get_preset('character')

# 覆蓋特定參數
overrides = {
    'repeats': 15,
    'feature_extractor': {'type': 'internvl2'}
}

# 深度合併
config = merge_configs(base, overrides)
# config['repeats'] = 15  (覆蓋)
# config['device'] = 'cuda'  (保留 base 的值)
```

### 保存和加載配置

```python
from config import save_config, load_config

# 保存配置
save_config(config, 'my_config.json', format='json')

# 加載配置
loaded_config = load_config('my_config.json')
```

## 📊 輸出格式

所有 preparers 都生成 Kohya 兼容的訓練數據集：

```
output_dir/
├── {repeats}_{name}/
│   ├── image_001.png
│   ├── image_001.txt
│   ├── image_002.png
│   ├── image_002.txt
│   └── ...
└── preparation_metadata.json
```

Metadata JSON 包含完整的處理信息：
- 輸入/輸出路徑
- 時間戳和處理時長
- 使用的配置
- 數據集統計（圖片數量、簇數量、簇大小分布）
- 使用的組件（extractor, clusterer, caption engine）

## 🎯 最佳實踐

1. **從預設開始**: 使用 `get_preset()` 獲取適合的預設配置
2. **早期驗證**: 在傳給 preparer 前調用 `validate_config()`
3. **保存成功配置**: 將有效的配置保存到文件以便重現
4. **迭代使用 fast 預設**: 測試時使用 `fast`, 生產時升級到 `high_quality`
5. **使用合併而非替換**: 用 `merge_configs()` 保留預設的默認值
6. **版本控制配置**: 將配置文件與代碼一起納入版本控制

## 🆕 v2.0 新功能

### 完全模組化架構
- 所有組件（extractors, clusterers, caption engines, filters）都可通過配置切換
- 無需修改代碼即可實驗不同算法組合

### 配置系統
- Schema validation 確保配置正確性
- 7 種預設配置涵蓋常見場景
- 深度合併支持靈活的配置覆蓋

### 5 種專門化 Preparers
- 每種 LoRA 類型都有優化的默認設置
- 統一的接口，一致的使用體驗

### SOTA 模型支持
- InternVL2, Qwen2-VL 等最新 VLM
- EVA-CLIP, DINOv2, SigLIP 等先進視覺模型

## ⚠️ 已棄用的文件

以下文件已被新的模組化系統取代，**不應再使用**：

- `prepare_training_data.py` → 使用 `CharacterLoRAPreparer`
- `prepare_expression_lora_data.py` → 使用 `ExpressionLoRAPreparer`
- `prepare_pose_lora_data.py` → 使用 `PoseLoRAPreparer`
- `prepare_background_lora_data.py` → 使用 `BackgroundLoRAPreparer`
- `prepare_style_lora_data.py` → 使用 `StyleLoRAPreparer`

這些舊腳本將在下一個版本中移除。

## 📝 問題和支持

如遇到問題：
1. 檢查 [Preparers README](preparers/README.md) 的故障排除部分
2. 檢查 [Config README](config/README.md) 的配置範例
3. 確保配置通過 `validate_config()` 驗證
4. 查看 `preparation_metadata.json` 了解處理詳情

## 🔄 從 v1.0 遷移

如果你使用舊的單體腳本：

```bash
# 舊方式
python prepare_training_data.py \
  --input /data \
  --output /output \
  --character miguel

# 新方式
python preparers/character_lora_preparer.py \
  --input-dir /data \
  --output-dir /output \
  --character-name miguel
```

Python API:

```python
# 舊方式 (不再可用)
from prepare_training_data import prepare_character_lora
prepare_character_lora(...)

# 新方式
from preparers import CharacterLoRAPreparer
from config import get_preset

config = get_preset('character')
preparer = CharacterLoRAPreparer(
    input_dir='/data',
    output_dir='/output',
    character_name='miguel',
    config=config
)
preparer.prepare()
```

## 📈 性能基準

在 RTX 3090 上的典型處理時間（500張圖片）：

| 配置 | Feature Extraction | Clustering | Caption Gen | Total |
|------|-------------------|------------|-------------|-------|
| fast (CLIP + Template) | ~15s | ~2s | ~0.5s | ~18s |
| character (CLIP + Template) | ~15s | ~3s | ~0.5s | ~19s |
| high_quality (InternVL2 + Qwen2-VL) | ~45s | ~3s | ~120s | ~170s |

## 版本

- **v2.0** (2025-11): 完全模組化重構
  - 模組化架構
  - 配置系統
  - 5種 preparers
  - 27個可插拔組件

- **v1.0** (2024): 初始版本
  - 單體腳本
  - 硬編碼配置
