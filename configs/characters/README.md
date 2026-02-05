# Character Configuration Files

這個目錄包含角色批次訓練的配置文件。

## 文件結構

```yaml
characters:
  - movie: <電影/系列名稱>
    char_dir: <characters_inpainted或characters_augmented中的目錄名>
    char_name: <標準角色名稱,用於文件命名>
    display_name: <顯示名稱>
    source_type: inpainted 或 augmented
    image_count: <預期圖片數量>
    repeats: <Kohya SS repeats參數>
    epochs: <訓練epochs>
    batch_size: <batch size>
```

## 配置文件

- **current_batch.yaml**: 當前批次訓練配置
- **elio_characters.yaml**: Elio電影角色配置
- **luca_characters.yaml**: Luca電影角色配置
- **coco_characters.yaml**: Coco電影角色配置

## 使用方法

### 自動化批次訓練

```bash
# 使用current_batch.yaml (默認)
bash scripts/batch/train_character_loras_from_config.sh

# 使用指定配置文件
bash scripts/batch/train_character_loras_from_config.sh configs/characters/elio_characters.yaml

# 指定tmux session名稱
bash scripts/batch/train_character_loras_from_config.sh configs/characters/luca_characters.yaml luca_batch
```

### 創建新批次配置

1. 複製`current_batch.yaml`為新文件
2. 修改`characters`列表
3. 運行訓練腳本

## 配置參數說明

### source_type
- `inpainted`: 使用`characters_inpainted`目錄
- `augmented`: 使用`characters_augmented`目錄(資料增強後)

### image_count
用於驗證caption生成完成度。腳本會等待直到`.txt`文件數量 >= image_count

### repeats
Kohya SS的repeats參數:
- 小dataset (< 100張): 15-20
- 中dataset (100-300張): 10-12
- 大dataset (> 300張): 8-10

### epochs
建議值:
- 小dataset: 16-20
- 中dataset: 14-16
- 大dataset: 12-14

### batch_size
根據VRAM和dataset大小:
- < 100張: 4
- 100-300張: 6
- > 300張: 8

## 範例配置

### 單一角色

```yaml
characters:
  - movie: toy-story
    char_dir: woody
    char_name: woody
    display_name: Woody
    source_type: inpainted
    image_count: 450
    repeats: 10
    epochs: 14
    batch_size: 8
```

### 多角色批次

```yaml
characters:
  - movie: finding-nemo
    char_dir: nemo
    char_name: nemo
    display_name: Nemo
    source_type: inpainted
    image_count: 320
    repeats: 10
    epochs: 14
    batch_size: 8

  - movie: finding-nemo
    char_dir: dory
    char_name: dory
    display_name: Dory
    source_type: augmented
    image_count: 215
    repeats: 12
    epochs: 16
    batch_size: 6
```

## 自動化工作流程

腳本會自動執行:

1. ✅ 等待caption generation完成
2. ✅ 組織Kohya SS格式訓練資料
3. ✅ 在tmux中依序訓練所有角色
4. ✅ 每個LoRA訓練後自動評估
5. ✅ 生成evaluation reports

## 監控訓練

```bash
# 附加到tmux session
tmux attach -t <session_name>

# 查看所有sessions
tmux list-sessions

# Detach (不中斷訓練)
Ctrl+B, then D
```

## 日誌文件

- 訓練日誌: `logs/train_<char_name>_<timestamp>.log`
- 評估日誌: `logs/eval_<char_name>_<timestamp>.log`
