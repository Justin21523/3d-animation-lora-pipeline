#!/usr/bin/env python3
"""
U2Net 參數優化測試
測試不同的後處理參數組合，找到最佳配置
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
import cv2
import json
from datetime import datetime
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


class U2NetOptimizer:
    """U2Net 優化測試器"""

    def __init__(self, device: str = "cuda"):
        self.device = device
        from rembg import new_session
        self.session = new_session("u2net")

    def segment_basic(self, image: Image.Image) -> np.ndarray:
        """基礎分割 (當前方法)"""
        from rembg import remove
        mask = remove(image, session=self.session, only_mask=True)
        return np.array(mask)

    def segment_with_alpha_matting(self, image: Image.Image) -> np.ndarray:
        """使用 alpha matting 優化"""
        from rembg import remove
        mask = remove(
            image,
            session=self.session,
            only_mask=True,
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=10
        )
        return np.array(mask)

    def segment_with_post_processing(self, image: Image.Image, config: Dict) -> np.ndarray:
        """自定義後處理優化"""
        from rembg import remove

        # 基礎分割
        mask = remove(image, session=self.session, only_mask=True)
        mask = np.array(mask)

        # 後處理
        if config.get("morphology", False):
            kernel_size = config.get("kernel_size", 5)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

            # Close operation - 填充小洞
            if config.get("close", True):
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=config.get("close_iter", 2))

            # Open operation - 移除小雜點
            if config.get("open", True):
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=config.get("open_iter", 1))

        # Gaussian blur - 平滑邊緣
        if config.get("blur", True):
            blur_size = config.get("blur_size", 5)
            mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)

        # Threshold
        threshold = config.get("threshold", 127)
        _, mask = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)

        # Dilate - 擴張邊緣
        if config.get("dilate", False):
            dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.get("dilate_size", 3), config.get("dilate_size", 3)))
            mask = cv2.dilate(mask, dilate_kernel, iterations=config.get("dilate_iter", 1))

        return mask

    def test_configuration(self, image_path: Path, output_dir: Path, config_name: str, config: Dict) -> Dict:
        """測試單一配置"""
        print(f"\n測試: {config_name}")

        try:
            start_time = datetime.now()

            # 載入圖片
            image = Image.open(image_path).convert("RGB")
            image_np = np.array(image)

            # 根據配置類型分割
            if config_name == "基礎版本":
                mask = self.segment_basic(image)
            elif config_name == "Alpha Matting":
                mask = self.segment_with_alpha_matting(image)
            else:
                mask = self.segment_with_post_processing(image, config)

            elapsed = (datetime.now() - start_time).total_seconds()

            # 分析結果
            coverage = np.sum(mask > 127) / mask.size * 100

            # 計算邊界框
            coords = np.column_stack(np.where(mask > 127))
            if len(coords) > 0:
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                bbox_area = (x_max - x_min) * (y_max - y_min)
                bbox_ratio = bbox_area / (image_np.shape[0] * image_np.shape[1]) * 100
            else:
                bbox_ratio = 0

            # 保存結果
            config_output = output_dir / config_name.replace(" ", "_")
            config_output.mkdir(parents=True, exist_ok=True)

            # Mask
            Image.fromarray(mask).save(config_output / "mask.png")

            # Character (RGBA)
            rgba = np.zeros((*image_np.shape[:2], 4), dtype=np.uint8)
            rgba[:, :, :3] = image_np
            rgba[:, :, 3] = mask
            Image.fromarray(rgba).save(config_output / "character.png")

            # 對比圖 (原圖 + mask overlay)
            overlay = image_np.copy()
            overlay[mask > 127] = overlay[mask > 127] * 0.5 + np.array([0, 255, 0]) * 0.5
            Image.fromarray(overlay.astype(np.uint8)).save(config_output / "overlay.jpg")

            result = {
                "config_name": config_name,
                "success": True,
                "time_seconds": elapsed,
                "coverage_percent": float(coverage),
                "bbox_area_percent": float(bbox_ratio),
                "config": config
            }

            print(f"  ✓ 完成 - {elapsed:.2f}s")
            print(f"  覆蓋率: {coverage:.1f}%")
            print(f"  邊界框: {bbox_ratio:.1f}%")

            return result

        except Exception as e:
            print(f"  ✗ 失敗: {e}")
            return {
                "config_name": config_name,
                "success": False,
                "error": str(e)
            }


def get_optimization_configs() -> Dict[str, Dict]:
    """定義各種優化配置"""
    return {
        "基礎版本": {},  # 當前使用的方法

        "Alpha Matting": {},  # rembg 內建的 alpha matting

        "強化邊緣": {
            "morphology": True,
            "kernel_size": 3,
            "close": True,
            "close_iter": 3,
            "open": True,
            "open_iter": 2,
            "blur": True,
            "blur_size": 3,
            "threshold": 127,
            "dilate": False
        },

        "擴張版本": {
            "morphology": True,
            "kernel_size": 5,
            "close": True,
            "close_iter": 2,
            "open": True,
            "open_iter": 1,
            "blur": True,
            "blur_size": 5,
            "threshold": 127,
            "dilate": True,
            "dilate_size": 5,
            "dilate_iter": 2
        },

        "保守版本": {
            "morphology": True,
            "kernel_size": 7,
            "close": True,
            "close_iter": 1,
            "open": True,
            "open_iter": 1,
            "blur": True,
            "blur_size": 7,
            "threshold": 140,
            "dilate": False
        },

        "激進版本": {
            "morphology": True,
            "kernel_size": 3,
            "close": True,
            "close_iter": 5,
            "open": False,
            "blur": True,
            "blur_size": 3,
            "threshold": 100,
            "dilate": True,
            "dilate_size": 7,
            "dilate_iter": 3
        },

        "平滑優先": {
            "morphology": True,
            "kernel_size": 5,
            "close": True,
            "close_iter": 2,
            "open": True,
            "open_iter": 1,
            "blur": True,
            "blur_size": 9,
            "threshold": 127,
            "dilate": False
        },

        "精細模式": {
            "morphology": True,
            "kernel_size": 3,
            "close": True,
            "close_iter": 1,
            "open": True,
            "open_iter": 1,
            "blur": True,
            "blur_size": 3,
            "threshold": 150,
            "dilate": False
        }
    }


def main():
    parser = argparse.ArgumentParser(description="U2Net 參數優化測試")

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="測試圖片目錄"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="輸出目錄"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="測試圖片數量"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda/cpu)"
    )

    args = parser.parse_args()

    # 找測試圖片
    image_files = sorted(list(args.input_dir.glob("*.jpg")) + list(args.input_dir.glob("*.png")))[:args.num_samples]

    if not image_files:
        print(f"❌ 沒有找到圖片: {args.input_dir}")
        return

    print(f"\n{'='*80}")
    print(f"U2Net 參數優化測試")
    print(f"{'='*80}\n")
    print(f"測試圖片: {len(image_files)} 張")
    print(f"Device: {args.device}")

    # 創建測試器
    optimizer = U2NetOptimizer(device=args.device)

    # 獲取配置
    configs = get_optimization_configs()
    print(f"\n測試配置: {len(configs)} 種\n")

    # 測試所有圖片 × 所有配置
    all_results = {}

    for idx, image_path in enumerate(image_files, 1):
        print(f"\n{'#'*80}")
        print(f"圖片 {idx}/{len(image_files)}: {image_path.name}")
        print(f"{'#'*80}")

        image_results = {}
        output_dir = args.output_dir / f"image_{idx:02d}_{image_path.stem}"

        for config_name, config in configs.items():
            result = optimizer.test_configuration(image_path, output_dir, config_name, config)
            image_results[config_name] = result

        all_results[image_path.name] = image_results

    # 生成總結
    print(f"\n{'='*80}")
    print("生成總結報告...")
    print(f"{'='*80}\n")

    summary = {}
    for config_name in configs.keys():
        config_results = []
        for image_results in all_results.values():
            if config_name in image_results:
                config_results.append(image_results[config_name])

        success_count = sum(1 for r in config_results if r.get("success", False))
        successful_results = [r for r in config_results if r.get("success", False)]

        if successful_results:
            avg_time = np.mean([r["time_seconds"] for r in successful_results])
            avg_coverage = np.mean([r["coverage_percent"] for r in successful_results])
            avg_bbox = np.mean([r["bbox_area_percent"] for r in successful_results])
        else:
            avg_time = 0
            avg_coverage = 0
            avg_bbox = 0

        summary[config_name] = {
            "success_rate": success_count / len(config_results) * 100 if config_results else 0,
            "avg_time": avg_time,
            "avg_coverage": avg_coverage,
            "avg_bbox": avg_bbox,
            "total_tests": len(config_results)
        }

    # 保存結果
    report = {
        "timestamp": datetime.now().isoformat(),
        "device": args.device,
        "num_images": len(image_files),
        "num_configs": len(configs),
        "configs": configs,
        "results": all_results,
        "summary": summary
    }

    report_file = args.output_dir / "optimization_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"✓ 報告已保存: {report_file}\n")

    # 打印總結
    print(f"{'='*80}")
    print("總結")
    print(f"{'='*80}\n")

    # 按覆蓋率排序
    sorted_configs = sorted(summary.items(), key=lambda x: x[1]["avg_coverage"], reverse=True)

    for config_name, stats in sorted_configs:
        print(f"{config_name}:")
        print(f"  成功率: {stats['success_rate']:.1f}%")
        print(f"  平均時間: {stats['avg_time']:.3f}s")
        print(f"  平均覆蓋率: {stats['avg_coverage']:.1f}%")
        print(f"  平均邊界框: {stats['avg_bbox']:.1f}%")
        print()

    print(f"\n查看視覺結果: {args.output_dir}/image_*/")


if __name__ == "__main__":
    main()
