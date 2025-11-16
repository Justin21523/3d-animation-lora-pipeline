#!/usr/bin/env python3
"""
修正 caption 以改善 Pixar 風格的統一光照問題

問題：生成的圖像對比度過高，光照不均勻
解決：在 caption 中明確描述 Pixar 的光照特徵
"""

import os
from pathlib import Path
from typing import List
import argparse

# Pixar 風格光照描述（將被添加到 caption 前面）
PIXAR_LIGHTING_PREFIX = (
    "pixar film lighting, uniform soft illumination, "
    "consistent material shading, low contrast, "
    "even skin tone, subtle subsurface scattering, "
    "no harsh shadows, "
)

# 替代版本（更短）
PIXAR_LIGHTING_SHORT = (
    "pixar uniform lighting, even illumination, low contrast, "
)

# 要移除的可能造成高對比度的詞彙
HIGH_CONTRAST_WORDS = [
    "dramatic lighting",
    "high contrast",
    "harsh shadows",
    "strong shadows",
    "dark shadows",
    "bright highlights",
    "vivid",
]


def fix_caption(caption: str, use_short: bool = False) -> str:
    """修正單個 caption"""

    # 移除可能造成高對比度的描述
    for word in HIGH_CONTRAST_WORDS:
        caption = caption.replace(word, "")

    # 清理多餘空格
    caption = " ".join(caption.split())

    # 檢查是否已經有 lighting 描述
    has_lighting = any(keyword in caption.lower() for keyword in [
        "pixar film lighting",
        "uniform soft illumination",
        "pixar uniform lighting"
    ])

    if has_lighting:
        return caption  # 已經修正過

    # 找到第一個逗號後的位置（保留角色描述前綴）
    # 例如: "a 3d animated character, ..." -> 在第一個逗號後插入
    if ", " in caption:
        parts = caption.split(", ", 1)
        prefix = parts[0]  # "a 3d animated character"
        rest = parts[1]

        lighting_desc = PIXAR_LIGHTING_SHORT if use_short else PIXAR_LIGHTING_PREFIX
        new_caption = f"{prefix}, {lighting_desc}{rest}"
    else:
        # 沒有逗號，直接添加到前面
        lighting_desc = PIXAR_LIGHTING_SHORT if use_short else PIXAR_LIGHTING_PREFIX
        new_caption = f"{lighting_desc}{caption}"

    return new_caption


def process_directory(
    image_dir: Path,
    output_dir: Path = None,
    use_short: bool = False,
    dry_run: bool = False
):
    """處理整個目錄的 captions"""

    if output_dir is None:
        output_dir = image_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    # 找到所有 .txt 文件
    txt_files = list(image_dir.glob("*.txt"))

    print(f"找到 {len(txt_files)} 個 caption 文件")
    print(f"輸出目錄: {output_dir}")
    print(f"使用短版光照描述: {use_short}")
    print(f"Dry run (不寫入): {dry_run}")
    print()

    modified_count = 0
    skipped_count = 0

    for txt_file in txt_files:
        with open(txt_file, 'r', encoding='utf-8') as f:
            original_caption = f.read().strip()

        new_caption = fix_caption(original_caption, use_short=use_short)

        if new_caption != original_caption:
            modified_count += 1

            if modified_count <= 3:  # 顯示前3個例子
                print(f"\n修改 {txt_file.name}:")
                print(f"  原始: {original_caption[:100]}...")
                print(f"  新版: {new_caption[:100]}...")

            if not dry_run:
                output_file = output_dir / txt_file.name
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(new_caption)
        else:
            skipped_count += 1

    print(f"\n完成！")
    print(f"  修改: {modified_count} 個文件")
    print(f"  跳過: {skipped_count} 個文件 (已包含光照描述)")

    if dry_run:
        print(f"\n⚠️  這是 dry run，實際上沒有寫入任何文件")
        print(f"   移除 --dry-run 參數來實際執行修改")


def main():
    parser = argparse.ArgumentParser(
        description="修正 caption 以改善 Pixar 風格的統一光照"
    )
    parser.add_argument(
        "image_dir",
        type=Path,
        help="包含圖片和 caption (.txt) 的目錄"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="輸出目錄（預設：覆蓋原始文件）"
    )
    parser.add_argument(
        "--short",
        action="store_true",
        help="使用短版光照描述（減少 token 數量）"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="預覽修改但不實際寫入文件"
    )

    args = parser.parse_args()

    if not args.image_dir.exists():
        print(f"錯誤：目錄不存在: {args.image_dir}")
        return 1

    process_directory(
        args.image_dir,
        output_dir=args.output_dir,
        use_short=args.short,
        dry_run=args.dry_run
    )

    return 0


if __name__ == "__main__":
    exit(main())
