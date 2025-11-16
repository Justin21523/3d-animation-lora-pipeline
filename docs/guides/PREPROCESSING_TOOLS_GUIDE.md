# 測試新開發的預處理工具

**測試日期**: 2025-11-15
**已完成 Frame 提取**:
- Coco: 19,755 frames ✅
- Elio: 17,304 frames ✅

---

## 📋 測試流程概覽

```
測試 1: Reference Face Manager
    ↓
測試 2: Frame Deduplication (Fast Mode)
    ↓
測試 3: Face-Driven Pre-Filter
```

---

## 🧪 測試 1: Reference Face Manager (參考臉部管理)

### 目標
為 Coco 電影的主要角色 (Miguel) 設置參考臉部圖像。

### 步驟

#### 1.1 準備參考臉部圖像

你需要手動挑選 3-5 張 Miguel 的清晰臉部圖像。建議從以下來源獲取：

**選項 A: 從已提取的 frames 中手動挑選**
```bash
# 瀏覽 Coco 的 frames，找出 Miguel 的清晰臉部特寫
cd /mnt/data/ai_data/datasets/3d-anime/coco/frames

# 你可以用圖片查看器打開這些 frames，選出 3-5 張 Miguel 的清晰臉部
# 建議選擇:
# - 正面照 1-2 張
# - 側面照 1-2 張
# - 不同表情/光照條件各 1 張
```

**選項 B: 從網路下載官方劇照**
```bash
# 創建臨時目錄存放參考圖
mkdir -p /tmp/coco_reference_faces/miguel

# 下載或複製 Miguel 的參考圖到這個目錄
# (你需要手動完成這個步驟)
```

#### 1.2 添加參考臉部到系統

假設你已經將 3 張 Miguel 的圖片準備好：

```bash
# 進入專案目錄
cd /mnt/c/AI_LLM_projects/3d-animation-lora-pipeline

# 添加參考臉部 (請替換成你實際的圖片路徑)
conda run -n ai_env python scripts/generic/preprocessing/reference_face_manager.py \
  --project coco \
  --character miguel \
  --add-references /tmp/coco_reference_faces/miguel/*.jpg
```

**預期輸出**:
```
Initialized InsightFace model: buffalo_l
Adding 3 reference faces for 'miguel'...
Processing faces: 100%|████████████| 3/3
Added reference face: .../miguel/miguel_001.jpg
Added reference face: .../miguel/miguel_002.jpg
Added reference face: .../miguel/miguel_003.jpg
Saved 3 embeddings to: .../miguel_embeddings.npy

Summary
========
Successful: 3
Failed: 0
Total: 3
```

#### 1.3 驗證參考臉部

```bash
# 驗證參考臉部是否正確儲存
conda run -n ai_env python scripts/generic/preprocessing/reference_face_manager.py \
  --project coco \
  --verify
```

**預期輸出**:
```
============================================================
Character: miguel
============================================================
Reference images: 3
Embeddings: 3 ✓
Match: ✓

Images:
  • miguel_001.jpg
  • miguel_002.jpg
  • miguel_003.jpg
```

#### 1.4 列出所有角色

```bash
# 列出專案中所有已設置參考臉部的角色
conda run -n ai_env python scripts/generic/preprocessing/reference_face_manager.py \
  --project coco \
  --list
```

**✅ 測試 1 成功標準**:
- [ ] 成功添加 3-5 張 Miguel 參考臉部
- [ ] `--verify` 顯示 embeddings 數量與圖片數量一致
- [ ] 沒有出現錯誤訊息

---

## 🧪 測試 2: Frame Deduplication (Fast Mode)

### 目標
使用 fast mode 移除 Coco frames 中的重複影格，預期減少 20-30%。

### 步驟

#### 2.1 執行快速去重

```bash
cd /mnt/c/AI_LLM_projects/3d-animation-lora-pipeline

# 使用 fast mode 去重 (aggressive, temporal window, parallel)
conda run -n ai_env python scripts/generic/preprocessing/deduplicate.py \
  --input-dir /mnt/data/ai_data/datasets/3d-anime/coco/frames \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/coco/frames_deduped \
  --mode fast \
  --project coco \
  --workers 8
```

**處理時間**: 約 10-20 分鐘（19,755 frames）

**預期輸出**:
```
📊 Found 19755 frames in /mnt/data/ai_data/datasets/3d-anime/coco/frames
   Mode: fast
   Temporal window: ±30 frames
   Parallel workers: 8

🔍 Computing hashes for 19755 frames...
Computing hashes (parallel): 100%|████████████| 19755/19755

🔍 Finding near-duplicates (threshold=15)...
   Using temporal window: 30 frames
Comparing hashes: 100%|████████████| XXXX/XXXX

📊 Found XXX duplicate groups

📁 Saving deduplicated frames...
Saving frames: 100%|████████████| ~14000/14000

✅ Deduplication complete!
   Input frames: 19755
   Duplicate groups: XXX
   Duplicates removed: ~5000-6000
   Unique frames kept: ~13000-14000
   Reduction: ~25-30%

📄 Report saved to: .../deduplication_report.json
```

#### 2.2 檢查去重結果

```bash
# 檢查輸出目錄的 frames 數量
ls -1 /mnt/data/ai_data/datasets/3d-anime/coco/frames_deduped/*.jpg | wc -l

# 查看詳細報告
cat /mnt/data/ai_data/datasets/3d-anime/coco/frames_deduped/deduplication_report.json
```

**✅ 測試 2 成功標準**:
- [ ] 去重後 frames 數量減少 20-30%（約剩 13,000-15,000 張）
- [ ] 產生 `deduplication_report.json`
- [ ] 產生 `duplicates_mapping.json`
- [ ] 沒有出現錯誤

---

## 🧪 測試 3: Character-Driven Pre-Filter

### 目標
使用 Miguel 的參考圖像過濾 frames，只保留包含 Miguel 的 frames。預期減少 60-80%。

**支援兩種模式**:
- **CLIP 模式 (推薦)**: 使用整體圖像 embedding 匹配，更robust，不需要完美的臉部檢測
- **Face 模式**: 使用臉部檢測 + ArcFace embedding，當臉部清晰時更精確

### 步驟

#### 3.1A 執行 CLIP 模式過濾 (推薦)

```bash
cd /mnt/c/AI_LLM_projects/3d-animation-lora-pipeline

# 使用 CLIP 模式 - 對遮擋、側面、光照變化更robust
conda run -n ai_env python scripts/generic/preprocessing/face_driven_prefilter.py \
  --input-dir /mnt/data/ai_data/datasets/3d-anime/coco/frames_deduped \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/coco/frames_filtered_clip \
  --project coco \
  --mode clip \
  --similarity-threshold 0.75 \
  --batch-size 16 \
  --device cuda
```

**優點**:
- ✅ 不需要完美的臉部檢測
- ✅ 對遮擋、側面、背影也有效
- ✅ 對光照、角度變化更robust
- ✅ 可以匹配全身、特殊姿勢

**處理時間**: 約 10-20 分鐘（~14,000 frames，取決於 GPU）

#### 3.1B 執行 Face 模式過濾 (精確但需要臉部可見)

```bash
cd /mnt/c/AI_LLM_projects/3d-animation-lora-pipeline

# 使用 Face 模式 - 當臉部清晰可見時更精確
conda run -n ai_env python scripts/generic/preprocessing/face_driven_prefilter.py \
  --input-dir /mnt/data/ai_data/datasets/3d-anime/coco/frames_deduped \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/coco/frames_filtered_face \
  --project coco \
  --mode face \
  --similarity-threshold 0.30 \
  --batch-size 16 \
  --device cuda
```

**優點**:
- ✅ 當臉部清晰時非常精確
- ✅ 較低的誤報率

**缺點**:
- ❌ 需要臉部清晰可見（>64x64 pixels）
- ❌ 側面、遮擋、背影會被拒絕

**處理時間**: 約 15-30 分鐘（~14,000 frames，取決於 GPU）

**預期輸出 (CLIP 模式)**:
```
Initializing CLIP model: openai/clip-vit-large-patch14...
Loaded 1 reference character(s)
  • miguel: 3 reference images

📊 Found 14000 frames in .../frames_deduped
   Mode: clip
   Similarity threshold: 0.75
   Batch size: 16

Batch 1/875: 100%|████████████| 16/16
Batch 2/875: 100%|████████████| 16/16
...
Batch 875/875: 100%|████████████| 8/8

✅ Character pre-filtering complete!
   Mode: clip
   Total input frames: ~14000
   Frames kept: ~4000-6000 (30-40%)
   Frames rejected: ~8000-10000 (60-70%)

   Characters detected:
     • miguel: ~4000-6000 frames

📄 Report saved to: .../prefilter_report.json
```

**預期輸出 (Face 模式)**:
```
Initializing InsightFace face detection and recognition...
Loaded 1 reference character(s)
  • miguel: 3 reference faces

📊 Found 14000 frames in .../frames_deduped
   Mode: face
   Similarity threshold: 0.30
   Min face size: 64x64
   Batch size: 16

Batch 1/875: 100%|████████████| 16/16
...

✅ Character pre-filtering complete!
   Mode: face
   Total input frames: ~14000
   Frames kept: ~3000-5000 (20-35%)
   Frames rejected: ~9000-11000 (65-80%)

   Characters detected:
     • miguel: ~3000-5000 frames

📄 Report saved to: .../prefilter_report.json
```

#### 3.2 檢查過濾結果

```bash
# CLIP 模式結果
ls -1 /mnt/data/ai_data/datasets/3d-anime/coco/frames_filtered_clip/*.jpg | wc -l
cat /mnt/data/ai_data/datasets/3d-anime/coco/frames_filtered_clip/prefilter_report.json

# Face 模式結果 (if tested)
ls -1 /mnt/data/ai_data/datasets/3d-anime/coco/frames_filtered_face/*.jpg | wc -l
cat /mnt/data/ai_data/datasets/3d-anime/coco/frames_filtered_face/prefilter_report.json
```

**✅ 測試 3 成功標準**:
- [ ] CLIP 模式：保留 30-40% frames（約 4,000-6,000 張）
- [ ] Face 模式：保留 20-35% frames（約 3,000-5,000 張）
- [ ] 產生 `prefilter_report.json` 和 `prefilter_detailed.json`
- [ ] Miguel 檢測數量合理
- [ ] 沒有 GPU OOM 錯誤

---

## 📊 完整流程測試結果摘要

執行完三個測試後，你應該看到以下數據流：

### CLIP 模式 (推薦)
```
原始 frames (Coco):              19,755 frames
    ↓ [Fast Deduplication]
去重後 frames:                   ~13,000-15,000 frames (-25-30%)
    ↓ [CLIP Character Pre-Filter]
最終保留 frames:                 ~4,000-6,000 frames (-70-80% total reduction)
```

**總減少率**: 70-80%（19,755 → 4,000-6,000）

### Face 模式 (精確但更嚴格)
```
原始 frames (Coco):              19,755 frames
    ↓ [Fast Deduplication]
去重後 frames:                   ~13,000-15,000 frames (-25-30%)
    ↓ [Face-Driven Pre-Filter]
最終保留 frames:                 ~3,000-5,000 frames (-75-85% total reduction)
```

**總減少率**: 75-85%（19,755 → 3,000-5,000）

### 推薦策略

**使用 CLIP 模式當**:
- ✅ 你想要更多樣化的角度和姿勢（包含側面、背影、遠景）
- ✅ 角色經常被部分遮擋
- ✅ 訓練數據需要涵蓋各種視角

**使用 Face 模式當**:
- ✅ 你只需要清晰的臉部特寫
- ✅ 想要最低的誤報率
- ✅ 優先考慮精確度而非召回率

這意味著後續的 SAM2 instance segmentation 只需處理 **3,000-6,000 frames**，而不是原本的 19,755 frames，**節省 70-85% 處理時間**！

---

## 🔧 故障排除

### 問題 1: InsightFace 未安裝

**錯誤訊息**:
```
ImportError: No module named 'insightface'
```

**解決方法**:
```bash
conda activate ai_env
pip install insightface onnxruntime-gpu
```

### 問題 2: 找不到參考臉部

**錯誤訊息**:
```
ValueError: No reference embeddings found for project 'coco'
```

**解決方法**:
確認你已完成測試 1，正確添加參考臉部：
```bash
ls -la /mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/configs/projects/coco/reference_faces/
```

### 問題 3: GPU OOM (Out of Memory)

**錯誤訊息**:
```
CUDA out of memory
```

**解決方法**:
降低 batch size：
```bash
--batch-size 8  # 或更小
```

或使用 CPU（較慢）:
```bash
--device cpu
```

### 問題 4: 過濾後 frames 太少

如果過濾後只剩非常少的 frames（<1000），可能是：
- 參考臉部品質不佳
- Threshold 太嚴格

**解決方法**:
```bash
# 使用更寬鬆的 threshold
--similarity-threshold 0.25  # 降低門檻

# 或重新添加更多/更好的參考臉部
```

---

## ✅ 測試完成檢查清單

完成所有測試後，確認以下檔案存在：

```bash
# Reference faces
configs/projects/coco/reference_faces/miguel/
  ├── miguel_001.jpg
  ├── miguel_002.jpg
  ├── miguel_003.jpg
  ├── miguel_embeddings.npy
  └── miguel_metadata.json

# Deduplication results
/mnt/data/ai_data/datasets/3d-anime/coco/frames_deduped/
  ├── (13,000-15,000 .jpg files)
  ├── deduplication_report.json
  └── duplicates_mapping.json

# Face-filtering results
/mnt/data/ai_data/datasets/3d-anime/coco/frames_filtered/
  ├── (3,000-5,000 .jpg files)
  ├── prefilter_report.json
  └── prefilter_detailed.json

/mnt/data/ai_data/datasets/3d-anime/coco/frames_filtered_rejected/
  └── (9,000-11,000 rejected .jpg files)
```

---

## 📊 CLIP 閾值校準指南 (重要！)

### 背景

CLIP 模式使用整體圖像 embedding 計算 cosine similarity，範圍 0-1：
- **1.0**: 完全相同
- **0.7-0.9**: 非常相似（相同角色、相似姿勢/光照）
- **0.5-0.7**: 中等相似（相同角色、不同角度或場景）
- **< 0.5**: 低相似度（可能是不同角色或完全不相關）

### 實證測試結果 (Coco - Miguel, 50 frames 樣本)

我們對 Miguel 進行了多個閾值的測試，以下是結果：

| 閾值 | 保留率 | 保留數量 | 最低分數 | 最高分數 | 備註 |
|------|--------|----------|----------|----------|------|
| **0.52** | 100% | 50/50 | 0.5382 | 0.7783 | 捕捉所有相似幀，無過濾 |
| **0.55** | 98% | 49/50 | 0.5382 (rejected) | 0.7783 | ⭐ **推薦：平衡召回率與精確度** |
| **0.60** | 88% | 44/50 | 0.5966 | 0.7783 | 過濾掉部分邊緣案例 |
| **0.70** | 22% | 11/50 | 0.7015 | 0.7783 | 過於嚴格，丟失大量有效幀 |
| **0.75** | 6% | 3/50 | 0.7486 | 0.7783 | 極度嚴格，只保留最相似幀 |

### 關鍵洞察

#### 1. 分數分布特性
- **Miguel 測試中** (50 frames):
  - 最低分數: 0.5382
  - 中位數: ~0.64
  - 最高分數: 0.7783

- **典型特徵**:
  - 正面、良好光照的特寫: 0.70-0.78
  - 側面、中景: 0.60-0.70
  - 遠景、部分遮擋、背影: 0.52-0.60
  - 完全不相關 (其他角色/場景): < 0.52

#### 2. 閾值選擇策略

**🎯 寬鬆模式 (threshold 0.50-0.55) - 推薦用於初次過濾**
- **目標**: 高召回率，確保不遺漏任何潛在有效幀
- **適用場景**:
  - 初次處理，不確定角色出現的場景類型
  - 角色經常處於遠景、側面、背影
  - 後續會有人工審核或第二階段過濾
- **預期保留率**: 95-100%
- **風險**: 可能包含少量誤報（相似但非目標角色）

**⚖️ 平衡模式 (threshold 0.55-0.65) - ⭐ 最推薦**
- **目標**: 平衡召回率與精確度
- **適用場景**:
  - 一般用途，角色有多種角度和場景
  - 希望過濾明顯不相關的幀，但保留大部分有效幀
  - **推薦為預設值**
- **預期保留率**: 70-98%
- **優勢**: 過濾掉明顯不相關場景，同時保留多樣化的角色幀

**🎯 嚴格模式 (threshold 0.65-0.75)**
- **目標**: 高精確度，只保留高置信度幀
- **適用場景**:
  - 只需要清晰、正面的角色幀
  - 參考圖像與目標幀視角/光照非常一致
  - 已有大量數據，希望進一步精煉
- **預期保留率**: 20-70%
- **風險**: 可能過度過濾，丟失有價值的多樣化幀

**🔬 極嚴格模式 (threshold > 0.75)**
- **不推薦用於一般情況**
- **僅適用於**:
  - 尋找幾乎相同的幀（例如：質量檢查、重複檢測）
  - 參考圖像與目標完全一致的場景
- **預期保留率**: < 20%

#### 3. 實務建議

**第一次處理新項目**:
```bash
# 步驟 1: 使用寬鬆閾值 0.55 進行初次過濾
--similarity-threshold 0.55

# 預期: 保留 95-98% 相關幀，過濾明顯不相關場景
```

**如果結果不理想**:

情況 A: **保留幀太少** (< 預期數量的 70%)
```bash
# 降低閾值到 0.50 或 0.52
--similarity-threshold 0.50
```

情況 B: **包含太多誤報** (目測 > 10% 非目標角色)
```bash
# 提高閾值到 0.60 或 0.65
--similarity-threshold 0.60
```

情況 C: **需要高質量子集** (已有足夠數據，想進一步精煉)
```bash
# 使用嚴格閾值 0.70
--similarity-threshold 0.70
```

#### 4. 參考圖像的影響 (關鍵！)

**參考圖像質量直接影響分數分布**:

**好的參考圖像組合**:
- ✅ 包含多種角度 (正面、3/4 側面、全側面)
- ✅ 包含多種光照條件 (明亮、柔和、背光)
- ✅ 包含多種距離 (特寫、半身、全身)
- ✅ 清晰、高分辨率
- ✅ 3-5 張即可（更多不一定更好）

**差的參考圖像組合**:
- ❌ 只有單一角度（例如：全是正面）
- ❌ 只有特寫或只有遠景
- ❌ 模糊、低分辨率
- ❌ 過多參考圖（> 10 張可能稀釋特徵）

**如果分數普遍偏低 (< 0.60)**:
1. 檢查參考圖像是否與目標幀視角差異過大
2. 增加更多樣化的參考圖像
3. 確保參考圖像清晰、高質量

#### 5. 實際案例：本專案的選擇

**Coco (Miguel) 和 Up (Russell)**:
- **最終選擇**: threshold **0.55**
- **理由**:
  - 保留 98% 相關幀 (49/50)
  - 只過濾明顯不相關的單幀 (score 0.5382)
  - 平衡了召回率與後續處理效率
  - 後續仍會有 instance segmentation 和人工審核

**預期結果**:
```
Coco:  7,566 frames (deduped) → ~7,400 frames (filtered, 98% kept)
Up:    6,138 frames (deduped) → ~6,015 frames (filtered, 98% kept)
```

### 監控與調整

**檢查過濾結果**:
```bash
# 查看詳細分數分布
jq '.[] | .best_similarity' prefilter_detailed.json | sort -n | head -20  # 最低分數
jq '.[] | .best_similarity' prefilter_detailed.json | sort -rn | head -20  # 最高分數

# 統計分數範圍
python -c "
import json
with open('prefilter_detailed.json') as f:
    data = json.load(f)
scores = [v['best_similarity'] for v in data.values()]
print(f'Min: {min(scores):.4f}')
print(f'Median: {sorted(scores)[len(scores)//2]:.4f}')
print(f'Max: {max(scores):.4f}')
"
```

**動態調整策略**:
1. 先用 0.55 處理小樣本 (100-200 frames)
2. 檢查分數分布和保留率
3. 根據結果調整閾值
4. 處理完整數據集

---

## 🎯 下一步

測試成功後，你可以：

1. **為其他電影重複此流程** (Elio, Turning Red, Up)
2. **繼續管道的下個階段**: Instance Segmentation (SAM2)
3. **報告任何問題或異常結果**

如有任何問題，請提供：
- 完整的錯誤訊息
- 使用的指令
- 相關的 log 檔案內容
