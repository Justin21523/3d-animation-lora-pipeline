# AI Warehouse 配置說明

本專案使用共用的 AI Warehouse 來節省硬碟空間並共享資源。

## Warehouse 結構

```
/mnt/c/AI_LLM_projects/ai_warehouse/
├── training_data/inazuma-eleven/         # 訓練數據
│   └── characters/
│       ├── endou_mamoru/
│       ├── gouenji_shuuya/
│       ├── fudou_akio/
│       └── utsunomiya_toramaru/
│
├── models/
│   ├── stable-diffusion/                 # 共用的 SD base models
│   │   ├── anything-v4.5-vae-swapped.safetensors
│   │   ├── AnythingXL_v50.safetensors
│   │   ├── dreamshaper_8.safetensors
│   │   └── v1-5-pruned-emaonly.safetensors
│   │
│   └── lora/character_loras/inazuma-eleven/  # 訓練完成的 LoRA
│
├── outputs/inazuma-eleven/               # 輸出和日誌
│   ├── logs/
│   ├── evaluations/
│   └── samples/
│
└── cache/inazuma-eleven/                 # 快取
    ├── clip/
    ├── wd14/
    └── hashes/
```

## 符號連結配置

專案目錄使用符號連結指向 warehouse：

```bash
# 專案目錄中的符號連結
inazuma-eleven-lora/
├── data -> /mnt/c/AI_LLM_projects/ai_warehouse/training_data/inazuma-eleven
├── models -> /mnt/c/AI_LLM_projects/ai_warehouse/models/lora/character_loras/inazuma-eleven
└── outputs -> /mnt/c/AI_LLM_projects/ai_warehouse/outputs/inazuma-eleven
```

## 好處

### 1. 節省空間
- Base models（SD 模型）在 warehouse 中共享，不需要每個專案都複製一份
- 單一 base model 可能 2-4GB，共享可以節省大量空間

### 2. 統一管理
- 所有 LoRA 模型統一存放在 `warehouse/models/lora/`
- 訓練數據統一管理
- 便於備份和遷移

### 3. 跨專案共享
- 同一角色的 LoRA 可以在不同專案中使用
- Cache 可以共享，加速重複操作

## 配置文件

### global_config.yaml

```yaml
paths:
  # 共用資料倉儲路徑
  warehouse_root: "/mnt/c/AI_LLM_projects/ai_warehouse"

  # 專案資料路徑（透過符號連結指向 warehouse）
  data_root: "data/"
  models_root: "models/"
  output_root: "outputs/"

  # Warehouse 中的實際路徑
  warehouse_training_data: "/mnt/c/AI_LLM_projects/ai_warehouse/training_data/inazuma-eleven"
  warehouse_models: "/mnt/c/AI_LLM_projects/ai_warehouse/models/lora/character_loras/inazuma-eleven"
  warehouse_outputs: "/mnt/c/AI_LLM_projects/ai_warehouse/outputs/inazuma-eleven"
  warehouse_cache: "/mnt/c/AI_LLM_projects/ai_warehouse/cache/inazuma-eleven"

  # Base models（共用的 Stable Diffusion 模型）
  base_models_root: "/mnt/c/AI_LLM_projects/ai_warehouse/models/stable-diffusion"
```

### endou_mamoru.yaml

```yaml
training:
  # 使用 warehouse 中的共用 base model
  base_model: "/mnt/c/AI_LLM_projects/ai_warehouse/models/stable-diffusion/anything-v4.5-vae-swapped.safetensors"
```

## 可用的 Base Models

目前 warehouse 中有以下 base models：

| 模型 | 大小 | 類型 | 適用場景 |
|------|------|------|---------|
| anything-v4.5-vae-swapped.safetensors | 4.0 GB | SD 1.5 | 動漫風格，高品質 |
| v1-5-pruned-emaonly.safetensors | 4.0 GB | SD 1.5 | 官方模型 |
| AnythingXL_v50.safetensors | 2.0 GB | SDXL | 動漫風格，SDXL |
| dreamshaper_8.safetensors | 2.0 GB | SDXL | 夢幻風格 |

**推薦用於閃電十一人：** `anything-v4.5-vae-swapped.safetensors`

### Pipeline 模組

所有模組已完成並配置正確：
- ✅ AutoCaptioner (WD14 Tagger v3)
- ✅ VideoProcessor (場景提取)
- ✅ ImageCleaner (去重品質過濾)
- ✅ CharacterFilter (兩階段識別)
- ✅ PipelineOrchestrator (完整流程)


## 使用方式

### 正常使用

由於有符號連結，腳本可以像往常一樣使用相對路徑：

```bash
# 這些命令會自動使用 warehouse 中的資料
python scripts/tools/prepare_training_data.py endou_mamoru

# 訪問資料
ls data/characters/endou_mamoru/
```

### 直接訪問 Warehouse

如果需要直接訪問 warehouse：

```bash
# 查看所有角色的訓練數據
ls /mnt/c/AI_LLM_projects/ai_warehouse/training_data/inazuma-eleven/characters/

# 查看所有訓練完成的 LoRA
ls /mnt/c/AI_LLM_projects/ai_warehouse/models/lora/character_loras/inazuma-eleven/

# 查看可用的 base models
ls /mnt/c/AI_LLM_projects/ai_warehouse/models/stable-diffusion/
```

## 添加新角色

當要訓練新角色時，目錄會自動在 warehouse 中創建：

```bash
# 目錄已預先建立
/mnt/c/AI_LLM_projects/ai_warehouse/training_data/inazuma-eleven/characters/
├── endou_mamoru/        # 已有資料
├── gouenji_shuuya/      # 準備就緒
├── fudou_akio/          # 準備就緒
└── utsunomiya_toramaru/ # 準備就緒
```

## 備份建議

只需要備份 warehouse 的相關目錄：

```bash
# 重要資料備份
/mnt/c/AI_LLM_projects/ai_warehouse/training_data/inazuma-eleven/
/mnt/c/AI_LLM_projects/ai_warehouse/models/lora/character_loras/inazuma-eleven/

# Cache 可以重新生成，不需要備份
# Base models 可以重新下載，但建議保留
```

## 驗證配置

運行以下命令驗證配置是否正確：

```bash
# 測試配置載入
python scripts/utils/config_loader.py

# 驗證路徑
python3 << 'EOF'
from pathlib import Path

paths = {
    "Gold standard": "data/characters/endou_mamoru/gold_standard/v1.0/images",
    "Base model": "/mnt/c/AI_LLM_projects/ai_warehouse/models/stable-diffusion/anything-v4.5-vae-swapped.safetensors",
}

for name, path_str in paths.items():
    exists = Path(path_str).exists()
    print(f"{'✓' if exists else '✗'} {name}: {exists}")
EOF
```

預期輸出：
```
✓ Gold standard: True
✓ Base model: True
```

## 疑難排解

### 符號連結失效

如果符號連結失效，重新建立：

```bash
cd /mnt/c/AI_LLM_projects/inazuma-eleven-lora

rm -f data models outputs

ln -s /mnt/c/AI_LLM_projects/ai_warehouse/training_data/inazuma-eleven data
ln -s /mnt/c/AI_LLM_projects/ai_warehouse/models/lora/character_loras/inazuma-eleven models
ln -s /mnt/c/AI_LLM_projects/ai_warehouse/outputs/inazuma-eleven outputs
```

### 路徑找不到

檢查 warehouse 目錄是否存在：

```bash
ls -la /mnt/c/AI_LLM_projects/ai_warehouse/
```

### 權限問題

確保有讀寫權限：

```bash
chmod -R u+rw /mnt/c/AI_LLM_projects/ai_warehouse/training_data/inazuma-eleven/
chmod -R u+rw /mnt/c/AI_LLM_projects/ai_warehouse/models/lora/character_loras/inazuma-eleven/
chmod -R u+rw /mnt/c/AI_LLM_projects/ai_warehouse/outputs/inazuma-eleven/
```

