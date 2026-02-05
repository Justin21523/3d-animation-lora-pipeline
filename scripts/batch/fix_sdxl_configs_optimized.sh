#!/bin/bash
set -e

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║     修正所有SDXL配置 - 完整優化版（Trial1 + 速度優化）          ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

CONFIG_DIR="/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/configs/training/character_loras_sdxl"

CONFIGS=(
    "coco_miguel_identity_sdxl.toml"
    "elio_bryce_identity_sdxl.toml"
    "elio_caleb_identity_sdxl.toml"
    "elio_elio_identity_sdxl.toml"
    "elio_glordon_identity_sdxl.toml"
    "luca_alberto_identity_sdxl.toml"
    "luca_giulia_identity_sdxl.toml"
    "onward_barley_lightfoot_identity_sdxl.toml"
    "onward_ian_lightfoot_identity_sdxl.toml"
    "orion_orion_identity_sdxl.toml"
    "turning-red_tyler_identity_sdxl.toml"
    "up_russell_identity_sdxl.toml"
)

echo "將修改以下配置以匹配Trial1並加入速度優化:"
echo ""
echo "【核心修復】（與Trial1相同）"
echo "  cache_latents_to_disk: true → false"
echo "  persistent_data_loader_workers: 確保為 true"
echo "  lowram: 確保為 false"
echo ""
echo "【速度優化】（新增）"
echo "  max_data_loader_n_workers: 2 → 6"
echo "  min_bucket_reso: 640 → 768"
echo "  max_bucket_reso: 1536 → 1280"
echo "  bucket_reso_steps: 64 → 128"
echo ""

BACKUP_DIR="${CONFIG_DIR}/_backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

for config in "${CONFIGS[@]}"; do
    CONFIG_PATH="${CONFIG_DIR}/${config}"

    if [ ! -f "$CONFIG_PATH" ]; then
        echo "⚠️  找不到: $config"
        continue
    fi

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "處理: $config"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # 備份原始文件
    cp "$CONFIG_PATH" "${BACKUP_DIR}/${config}"
    echo "  ✅ 已備份到: ${BACKUP_DIR}/${config}"

    # 1. 修復 cache_latents_to_disk = true → false
    sed -i 's/^cache_latents_to_disk = true/cache_latents_to_disk = false/' "$CONFIG_PATH"
    echo "  ✅ cache_latents_to_disk = false"

    # 2. 確保 persistent_data_loader_workers = true
    sed -i 's/^persistent_data_loader_workers = false/persistent_data_loader_workers = true/' "$CONFIG_PATH"
    echo "  ✅ persistent_data_loader_workers = true"

    # 3. 確保 lowram = false
    sed -i 's/^lowram = true/lowram = false/' "$CONFIG_PATH"
    echo "  ✅ lowram = false"

    # 4. 優化 max_data_loader_n_workers: 2 → 6
    sed -i 's/^max_data_loader_n_workers = 2/max_data_loader_n_workers = 6/' "$CONFIG_PATH"
    echo "  ✅ max_data_loader_n_workers = 6"

    # 5. 優化 min_bucket_reso: 640 → 768
    sed -i 's/^min_bucket_reso = 640/min_bucket_reso = 768/' "$CONFIG_PATH"
    echo "  ✅ min_bucket_reso = 768"

    # 6. 優化 max_bucket_reso: 1536 → 1280
    sed -i 's/^max_bucket_reso = 1536/max_bucket_reso = 1280/' "$CONFIG_PATH"
    echo "  ✅ max_bucket_reso = 1280"

    # 7. 優化 bucket_reso_steps: 64 → 128
    sed -i 's/^bucket_reso_steps = 64/bucket_reso_steps = 128/' "$CONFIG_PATH"
    echo "  ✅ bucket_reso_steps = 128"

    # 8. 確保 bucket_no_upscale = true（如果不存在則添加）
    if ! grep -q "^bucket_no_upscale" "$CONFIG_PATH"; then
        # 在 bucket_reso_steps 後添加
        sed -i '/^bucket_reso_steps/a bucket_no_upscale = true' "$CONFIG_PATH"
        echo "  ✅ bucket_no_upscale = true （新增）"
    else
        sed -i 's/^bucket_no_upscale = false/bucket_no_upscale = true/' "$CONFIG_PATH"
        echo "  ✅ bucket_no_upscale = true"
    fi

    echo "  ✓ 完成"
    echo ""
done

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                      配置修改完成                                ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "備份文件位置: ${BACKUP_DIR}/"
echo ""

# 驗證修改
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "驗證關鍵配置參數:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for config in "${CONFIGS[@]}"; do
    CONFIG_PATH="${CONFIG_DIR}/${config}"
    if [ -f "$CONFIG_PATH" ]; then
        echo ""
        echo "=== $(basename $config .toml) ==="

        # 核心修復驗證
        echo "【核心修復】"
        grep -E "^cache_latents_to_disk|^persistent_data_loader_workers|^lowram" "$CONFIG_PATH" || true

        # 速度優化驗證
        echo "【速度優化】"
        grep -E "^max_data_loader_n_workers|^min_bucket_reso|^max_bucket_reso|^bucket_reso_steps|^bucket_no_upscale" "$CONFIG_PATH" || true
    fi
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "配置優化總結:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "✅ Trial1核心配置已應用"
echo "   - cache_latents_to_disk = false（RAM快取，預計+6-11分/epoch）"
echo "   - persistent_data_loader_workers = true"
echo "   - lowram = false"
echo ""
echo "✅ 速度優化已應用"
echo "   - max_data_loader_n_workers = 6（預計+3-8分/epoch）"
echo "   - Bucketing優化（768-1280，步長128）（預計+2-5分/epoch）"
echo "   - bucket_no_upscale = true"
echo ""
echo "🎯 預期總改善: 11-24分/epoch"
echo "🎯 配合圖片預處理（1024x1024）: 額外+8-15分/epoch"
echo "🎯 總計預期: 19-39分/epoch 提升"
echo ""
echo "💾 所有原始配置已備份到:"
echo "   ${BACKUP_DIR}/"
echo ""
