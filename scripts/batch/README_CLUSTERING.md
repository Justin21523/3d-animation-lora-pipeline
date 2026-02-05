# Batch Identity & Scene Clustering 使用指南

## 概述

此批次處理腳本用於為多部影片自動執行：
1. **Identity Clustering** - 使用 ArcFace 將角色實例按身份分組
2. **Scene Clustering** - 使用 CLIP 將背景按場景類型分組

## 使用方式

### 處理所有可用影片
```bash
bash scripts/batch/batch_identity_scene_clustering.sh all
```

### 處理特定影片
```bash
bash scripts/batch/batch_identity_scene_clustering.sh luca onward
```

### 處理單一影片
```bash
bash scripts/batch/batch_identity_scene_clustering.sh luca
```

## 系統需求

- **CPU**: 多核心 (建議 8+ 核心以充分利用並行處理)
- **RAM**: 16GB+ (每個並行任務約需 2-4GB)
- **磁碟**: 足夠空間存放 clustering 結果
- **Python 環境**: `ai_env` conda 環境已安裝相依套件

## 並行處理

腳本會自動並行處理最多 **4 個任務**（可在腳本中調整 `PARALLEL_JOBS`）：

```bash
# 修改並行任務數量
PARALLEL_JOBS=4  # 改為你的 CPU 核心數
```

## 處理流程

### Phase 1: Identity Clustering (並行)
```
luca    ━━━━━━━━━━━━━━━━━━━━ 2h
onward  ━━━━━━━━━━━━━━━━━━━━ 1.5h
turning-red ━━━━━━━━━━━━━━━━ 1.5h
coco    ━━━━━━━━━━━━━━━━━━━━ 1h
elio    ━━━━━━━━━━━━━━━━━━━━ 1h

實際時間: ~2h (並行執行 4 個任務)
```

### Phase 2: Scene Clustering (並行)
```
luca backgrounds    ━━━━━━━━ 1h
onward backgrounds  ━━━━━━━━ 1h
turning-red bgs     ━━━━━━━━ 1h
coco backgrounds    ━━━━━━━━ 45m
elio backgrounds    ━━━━━━━━ 45m

實際時間: ~1h (並行執行 4 個任務)
```

**總預估時間**: ~3-4 小時 (處理所有 5 部影片)

## 輸出結構

### Identity Clusters
```
/mnt/data/ai_data/datasets/3d-anime/{film}/
└── identity_clusters/
    ├── character_0/           # 主角 1
    │   ├── *.png              # 角色實例圖片
    │   └── ...
    ├── character_1/           # 主角 2
    ├── character_2/           # 配角 1
    ├── ...
    ├── noise/                 # 無法分類的實例
    ├── cluster_report.json    # 聚類報告
    └── cluster_visualization.png
```

### Scene Clusters
```
/mnt/data/ai_data/datasets/3d-anime/{film}/
└── scene_clusters/
    ├── character_0/           # CLIP 自動命名的場景群 0
    │   ├── *.jpg/png          # 背景圖片
    │   └── ...
    ├── character_1/           # 場景群 1
    ├── character_2/           # 場景群 2
    ├── ...
    ├── noise/                 # 無法分類的背景
    ├── cluster_report.json
    └── cluster_visualization.png
```

**注意**: `clip_character_clustering.py` 預設輸出資料夾名稱為 `character_*`，即使用於背景聚類也是如此。未來可重命名為 `scene_*`。

## 日誌

所有日誌保存在 `logs/clustering/`:
- `identity_{film}_{timestamp}.log` - 每部影片的 identity clustering 日誌
- `scene_{film}_{timestamp}.log` - 每部影片的 scene clustering 日誌
- `clustering_summary_{timestamp}.txt` - 總結報告

## 監控進度

### 查看即時日誌
```bash
# Identity clustering
tail -f logs/clustering/identity_luca_*.log

# Scene clustering
tail -f logs/clustering/scene_luca_*.log
```

### 檢查運行中的任務
```bash
ps aux | grep -E "face_identity|clip_character"
```

### 查看輸出進度
```bash
# 檢查 identity clusters
ls -lh /mnt/data/ai_data/datasets/3d-anime/luca/identity_clusters/

# 檢查 scene clusters
ls -lh /mnt/data/ai_data/datasets/3d-anime/luca/scene_clusters/
```

## 故障排除

### 問題: 記憶體不足
```bash
# 降低並行任務數
PARALLEL_JOBS=2  # 從 4 改為 2
```

### 問題: ArcFace 模型找不到
```python
# 檢查模型路徑配置
python -c "from scripts.core.utils.config_loader import load_config; print(load_config('global_config'))"
```

### 問題: CLIP embedding 太慢
```bash
# 使用較小的 CLIP 模型
# 編輯 clip_character_clustering.py，改用 ViT-B/32 代替 ViT-L/14
```

## 後續步驟

### 1. 人工審查 (可選但推薦)
```bash
# 使用互動式工具檢視和修正聚類結果
python scripts/generic/clustering/interactive_cluster_review.py \
  --cluster-dir /mnt/data/ai_data/datasets/3d-anime/luca/identity_clusters
```

### 2. 準備訓練數據
```bash
# Character LoRA
python scripts/generic/training/prepare_training_data.py \
  --character-dirs /mnt/data/ai_data/datasets/3d-anime/luca/identity_clusters/character_0 \
  --output-dir /mnt/data/ai_data/training_data/luca/character_luca \
  --character-name "Luca" \
  --generate-captions \
  --target-size 400

# Background LoRA
python scripts/generic/training/prepare_background_lora_data.py \
  --sam2-backgrounds /mnt/data/ai_data/datasets/3d-anime/luca/luca_instances_sam2_v2/backgrounds \
  --lama-backgrounds /mnt/data/ai_data/datasets/3d-anime/luca/backgrounds_lama_v2 \
  --scene-clusters /mnt/data/ai_data/datasets/3d-anime/luca/scene_clusters \
  --output-dir /mnt/data/ai_data/training_data/luca/background_portorosso \
  --scene-name "Portorosso town" \
  --target-size 300
```

### 3. 開始 LoRA 訓練
```bash
# 使用 Kohya_ss sd-scripts
conda run -n ai_env python sd-scripts/train_network.py \
  --config_file configs/training/luca_character.toml
```

## 效能最佳化

### CPU 使用最佳化
```bash
# 設定 OpenMP 執行緒數
export OMP_NUM_THREADS=4

# 設定 PyTorch CPU 執行緒數
export MKL_NUM_THREADS=4
```

### 磁碟 I/O 最佳化
- 確保輸入/輸出目錄在快速 SSD 上
- 如果可能，將不同影片分散到不同磁碟

### 記憶體使用最佳化
```bash
# 如果記憶體不足，處理較小的批次
bash scripts/batch/batch_identity_scene_clustering.sh luca
# 等完成後再處理下一部
bash scripts/batch/batch_identity_scene_clustering.sh onward
```

## 預期結果

### Identity Clustering 質量指標
- ✅ **好**: 主要角色清楚分離，每個 cluster 有 100+ 圖片
- ⚠️ **普通**: 有些角色混在一起，需要人工分割
- ❌ **差**: 大部分圖片在 noise，需要調整參數

### Scene Clustering 質量指標
- ✅ **好**: 相似場景聚在一起（例如: 所有室內、所有海灘）
- ⚠️ **普通**: 一些場景分散在多個 clusters
- ❌ **差**: 聚類結果隨機，需要調整 `similarity-threshold`

## 參數調整指南

### Identity Clustering 參數

```bash
# 如果角色分得太細 (同一角色分成多個 clusters)
--min-cluster-size 8        # 降低最小 cluster 大小 (預設 12)
--min-samples 1             # 降低最小樣本數 (預設 2)

# 如果不同角色混在一起
--min-cluster-size 20       # 提高最小 cluster 大小
--min-samples 5             # 提高最小樣本數
```

### Scene Clustering 參數

```bash
# 如果場景分得太細
--min-cluster-size 10       # 降低最小 cluster 大小 (預設 15)
--similarity-threshold 0.70 # 降低相似度閾值 (預設 0.75)

# 如果不同場景混在一起
--min-cluster-size 25       # 提高最小 cluster 大小
--similarity-threshold 0.80 # 提高相似度閾值
```

## 常見問題

**Q: 處理需要多久時間？**
A: 5 部影片並行處理約 3-4 小時。單部影片約 2-3 小時。

**Q: 可以在 GPU 處理 SAM2 的同時運行嗎？**
A: 可以！此腳本使用 `--device cpu`，不會佔用 GPU。

**Q: 如何停止正在運行的批次處理？**
A: 使用 `Ctrl+C` 或 `pkill -f face_identity_clustering`

**Q: 可以恢復中斷的處理嗎？**
A: 目前不支援自動恢復。需要重新運行（已完成的影片會被跳過如果輸出目錄存在）。

**Q: 輸出的 character_* 資料夾可以重命名嗎？**
A: 可以！建議重命名為實際角色/場景名稱，例如 `Luca`, `Alberto`, `beach_scene` 等。

## 相關文檔

- `docs/guides/tools/character_clustering_guide.md` - Clustering 工具詳細說明
- `docs/pipeline/multi-character-clustering.md` - Multi-character 處理方法
- `docs/training/multi-type-lora-system.md` - Multi-LoRA 系統架構
- `LLM_PROVIDER.md` - 專案總覽和工作流程

## 聯絡與支援

如有問題，請查看日誌文件或參考上述文檔。
