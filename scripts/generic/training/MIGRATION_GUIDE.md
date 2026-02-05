# Migration Guide: v1.0 → v2.0

從單體腳本遷移到模組化架構的完整指南。

## 概述

v2.0 將所有單體腳本（900-1300行）重構為模組化組件，提供：
- 可配置的算法選擇
- 預設配置系統
- 統一的 API
- 更好的可測試性和可維護性

## 主要變更

### 1. 文件重組

#### 已棄用的單體腳本 → 新的 Preparers

| 舊文件 | 新 Preparer | 狀態 |
|--------|-------------|------|
| `prepare_training_data.py` | `preparers/character_lora_preparer.py` | ⚠️ 棄用 |
| `prepare_expression_lora_data.py` | `preparers/expression_lora_preparer.py` | ⚠️ 棄用 |
| `prepare_pose_lora_data.py` | `preparers/pose_lora_preparer.py` | ⚠️ 棄用 |
| `prepare_background_lora_data.py` | `preparers/background_lora_preparer.py` | ⚠️ 棄用 |
| `prepare_style_lora_data.py` | `preparers/style_lora_preparer.py` | ⚠️ 棄用 |

#### 已提取的可重用組件

| 功能 | 舊位置（重複出現在多個腳本） | 新位置 |
|------|----------------------------|--------|
| Blur 檢測 | 內嵌在每個腳本 | `quality_filters/blur_filter.py` |
| Size 過濾 | 內嵌在每個腳本 | `quality_filters/size_filter.py` |
| 去重 | 內嵌在每個腳本 | `quality_filters/perceptual_hash_deduplicator.py` |
| CLIP 提取 | 內嵌在每個腳本 | `feature_extractors/clip_extractor.py` |
| HDBSCAN 聚類 | 內嵌在每個腳本 | `clusterers/hdbscan_clusterer.py` |

### 2. API 變更

#### 舊 API (v1.0)

```python
# 單體腳本，硬編碼參數
from prepare_training_data import prepare_character_lora

prepare_character_lora(
    input_dir='/data/miguel',
    output_dir='/output/miguel_lora',
    character_name='miguel',
    min_cluster_size=12,  # 每個參數都需要單獨指定
    min_samples=2,
    use_face_detection=True,
    blur_threshold=100.0,
    caption_engine='template',
    # ... 數十個參數
)
```

#### 新 API (v2.0)

```python
# 模組化，配置驅動
from preparers import CharacterLoRAPreparer
from config import get_preset, merge_configs

# 使用預設配置
config = get_preset('character')

# 可選：覆蓋特定參數
config = merge_configs(config, {
    'repeats': 15,
    'feature_extractor': {'type': 'internvl2'}
})

# 創建 preparer
preparer = CharacterLoRAPreparer(
    input_dir='/data/miguel',
    output_dir='/output/miguel_lora',
    character_name='miguel',
    config=config
)

# 執行
preparer.prepare()
```

### 3. CLI 變更

#### 舊 CLI (v1.0)

```bash
python prepare_training_data.py \
  --input /data/miguel \
  --output /output/miguel_lora \
  --character miguel \
  --min-cluster-size 12 \
  --min-samples 2 \
  --blur-threshold 100 \
  --use-face-detection \
  --caption-engine template
```

#### 新 CLI (v2.0)

```bash
# 使用預設
python preparers/character_lora_preparer.py \
  --input-dir /data/miguel \
  --output-dir /output/miguel_lora \
  --character-name miguel

# 或覆蓋特定參數
python preparers/character_lora_preparer.py \
  --input-dir /data/miguel \
  --output-dir /output/miguel_lora \
  --character-name miguel \
  --feature-extractor internvl2 \
  --clusterer hdbscan \
  --min-cluster-size 15 \
  --caption-engine qwen2_vl
```

## 遷移步驟

### 步驟 1: 識別你使用的腳本

檢查你的腳本/代碼中使用了哪些舊文件：

```bash
# 查找對舊腳本的引用
grep -r "prepare_training_data" scripts/
grep -r "prepare_expression_lora_data" scripts/
grep -r "prepare_pose_lora_data" scripts/
```

### 步驟 2: 映射到新 Preparers

| 如果你使用... | 遷移到... |
|--------------|----------|
| `prepare_training_data.py` | `CharacterLoRAPreparer` |
| `prepare_expression_lora_data.py` | `ExpressionLoRAPreparer` |
| `prepare_pose_lora_data.py` | `PoseLoRAPreparer` |
| `prepare_background_lora_data.py` | `BackgroundLoRAPreparer` |
| `prepare_style_lora_data.py` | `StyleLoRAPreparer` |

### 步驟 3: 轉換參數到配置

舊腳本的參數可以轉換為配置字典：

```python
# 舊參數
min_cluster_size = 12
min_samples = 2
blur_threshold = 100.0
use_clip = True
caption_engine = 'template'

# 轉換為配置
config = {
    'clusterer': {
        'type': 'hdbscan',
        'min_cluster_size': 12,
        'min_samples': 2
    },
    'feature_extractor': {
        'type': 'clip'
    },
    'caption_engine': {
        'type': 'template'
    },
    'quality_filters': [
        {'type': 'blur', 'threshold': 100.0}
    ]
}

# 或直接使用預設
from config import get_preset
config = get_preset('character')  # 已包含這些默認值
```

### 步驟 4: 更新導入

```python
# 舊導入
from prepare_training_data import prepare_character_lora

# 新導入
from preparers import CharacterLoRAPreparer
from config import get_preset
```

### 步驟 5: 更新函數調用

```python
# 舊調用
prepare_character_lora(
    input_dir='/data/miguel',
    output_dir='/output',
    character_name='miguel',
    min_cluster_size=12,
    # ... 許多參數
)

# 新調用
config = get_preset('character')
preparer = CharacterLoRAPreparer(
    input_dir='/data/miguel',
    output_dir='/output',
    character_name='miguel',
    config=config
)
result = preparer.prepare()
```

## 常見遷移場景

### 場景 1: 基本 Character LoRA

**Before (v1.0):**
```python
from prepare_training_data import prepare_character_lora

prepare_character_lora(
    input_dir='/data/miguel',
    output_dir='/output/miguel_lora',
    character_name='miguel'
)
```

**After (v2.0):**
```python
from preparers import CharacterLoRAPreparer
from config import get_preset

config = get_preset('character')
preparer = CharacterLoRAPreparer(
    input_dir='/data/miguel',
    output_dir='/output/miguel_lora',
    character_name='miguel',
    config=config
)
preparer.prepare()
```

### 場景 2: 自定義參數

**Before (v1.0):**
```python
from prepare_training_data import prepare_character_lora

prepare_character_lora(
    input_dir='/data/miguel',
    output_dir='/output',
    character_name='miguel',
    min_cluster_size=20,  # 自定義
    blur_threshold=120.0,  # 自定義
    caption_engine='qwen2_vl'  # 自定義
)
```

**After (v2.0):**
```python
from preparers import CharacterLoRAPreparer
from config import get_preset, merge_configs

base = get_preset('character')
overrides = {
    'clusterer': {'min_cluster_size': 20},
    'quality_filters': [
        {'type': 'blur', 'threshold': 120.0},
        {'type': 'size', 'min_width': 256, 'min_height': 256}
    ],
    'caption_engine': {'type': 'qwen2_vl'}
}

config = merge_configs(base, overrides)

preparer = CharacterLoRAPreparer(
    input_dir='/data/miguel',
    output_dir='/output',
    character_name='miguel',
    config=config
)
preparer.prepare()
```

### 場景 3: 批量處理

**Before (v1.0):**
```python
from prepare_training_data import prepare_character_lora

characters = ['bryce', 'caleb', 'elio']

for char in characters:
    prepare_character_lora(
        input_dir=f'/data/{char}',
        output_dir=f'/output/{char}_lora',
        character_name=char,
        min_cluster_size=12,
        # ... 重複參數
    )
```

**After (v2.0):**
```python
from preparers import CharacterLoRAPreparer
from config import get_preset

config = get_preset('character')
characters = ['bryce', 'caleb', 'elio']

for char in characters:
    preparer = CharacterLoRAPreparer(
        input_dir=f'/data/{char}',
        output_dir=f'/output/{char}_lora',
        character_name=char,
        config=config
    )
    preparer.prepare()
```

### 場景 4: 保存配置以供重用

**Before (v1.0):**
```python
# 無法保存配置，必須每次重複指定參數
```

**After (v2.0):**
```python
from config import get_preset, merge_configs, save_config, load_config

# 創建並保存配置
config = get_preset('character')
config = merge_configs(config, {'repeats': 15})
save_config(config, 'my_character_config.json')

# 稍後重用
config = load_config('my_character_config.json')
preparer = CharacterLoRAPreparer(..., config=config)
```

## 好處對比

### v1.0 的問題

❌ **代碼重複**: 相同的過濾/聚類邏輯在5個腳本中重複
❌ **硬編碼配置**: 更換算法需要修改代碼
❌ **難以測試**: 單體腳本難以進行單元測試
❌ **缺乏靈活性**: 添加新算法需要大量修改
❌ **配置管理**: 無法保存和重用成功的配置

### v2.0 的優勢

✅ **模組化**: 每個組件獨立、可重用
✅ **配置驅動**: 切換算法無需修改代碼
✅ **易於測試**: 小型、專注的類易於測試
✅ **可擴展**: 添加新算法只需繼承基類
✅ **配置管理**: 完整的配置系統（預設、驗證、I/O）
✅ **一致性**: 所有 preparers 使用相同的接口

## 向後兼容性

v2.0 **不**向後兼容 v1.0 的 API。所有使用舊腳本的代碼需要遷移。

舊腳本文件保留但標記為棄用，將在下一個主版本（v3.0）中移除。

## 需要刪除的文件

以下文件已被完全取代，建議在遷移完成後刪除：

```
scripts/generic/training/
├── prepare_training_data.py              ⚠️ → CharacterLoRAPreparer
├── prepare_expression_lora_data.py       ⚠️ → ExpressionLoRAPreparer
├── prepare_pose_lora_data.py             ⚠️ → PoseLoRAPreparer
├── prepare_background_lora_data.py       ⚠️ → BackgroundLoRAPreparer
├── prepare_style_lora_data.py            ⚠️ → StyleLoRAPreparer
├── assemble_expression_lora_dataset.py   ⚠️ → 已整合到 preparers
├── organize_lora_dataset.py              ⚠️ → 已整合到 preparers
├── background_dataset_organizer.py       ⚠️ → 已整合到 preparers
└── ... (其他單體腳本)
```

**注意**: 在刪除前，請確保：
1. 所有依賴這些文件的腳本都已更新
2. 已驗證新系統滿足你的需求
3. 已備份舊文件（如需要）

## 故障排除

### 問題 1: 找不到模組

**錯誤**: `ModuleNotFoundError: No module named 'preparers'`

**解決**:
```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from preparers import CharacterLoRAPreparer
```

### 問題 2: 配置驗證失敗

**錯誤**: `Config errors: ['device: invalid value...']`

**解決**:
```python
from config import validate_config

errors = validate_config(config)
if errors:
    print("Fix these errors:", errors)
```

### 問題 3: 輸出格式不同

v2.0 的輸出格式與 v1.0 相同（Kohya format），但 metadata JSON 的結構略有不同。

檢查 `preparation_metadata.json` 以了解新的結構。

## 獲取幫助

如果遇到遷移問題：

1. 查看 [README.md](README.md) 的快速開始指南
2. 查看 [preparers/README.md](preparers/README.md) 的詳細文檔
3. 查看 [config/README.md](config/README.md) 的配置範例
4. 檢查你的配置是否通過 `validate_config()`

## 遷移檢查清單

- [ ] 識別所有使用舊腳本的地方
- [ ] 映射舊參數到新配置
- [ ] 更新導入語句
- [ ] 更新函數調用
- [ ] 測試新代碼
- [ ] 驗證輸出與預期一致
- [ ] 更新文檔/註釋
- [ ] (可選) 刪除舊腳本文件
- [ ] 提交更改

## 總結

v2.0 的模組化架構提供了更好的：
- 可維護性
- 可擴展性
- 可測試性
- 配置管理

雖然需要一些遷移工作，但長期來看會節省時間並提高代碼質量。
