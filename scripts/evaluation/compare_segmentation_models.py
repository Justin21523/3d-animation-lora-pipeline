#!/usr/bin/env python3
"""
分割/檢測模型對比測試
測試不同方法在 Yokai Watch 數據上的效果：
1. U2Net (rembg) - 當前使用
2. YOLO 檢測
3. anime-segmentation (如果可用)
4. 其他可用模型

生成詳細對比報告
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
from typing import Dict, List
import cv2
import json
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class SegmentationTester:
    """測試和對比不同分割/檢測方法"""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.results = {}

    def test_u2net_rembg(self, image_path: Path, output_dir: Path) -> Dict:
        """測試 U2Net (當前方法)"""
        print("\n1️⃣ 測試 U2Net/Rembg...")

        try:
            from rembg import new_session, remove

            start_time = datetime.now()

            # 載入圖片
            image = Image.open(image_path).convert("RGB")
            image_np = np.array(image)

            # 分割
            session = new_session("u2net")
            mask = remove(image, session=session, only_mask=True)
            mask_np = np.array(mask)

            elapsed = (datetime.now() - start_time).total_seconds()

            # 分析結果
            coverage = np.sum(mask_np > 127) / mask_np.size * 100

            # 保存結果
            output_dir.mkdir(parents=True, exist_ok=True)

            # 保存 mask
            Image.fromarray(mask_np).save(output_dir / "u2net_mask.png")

            # 保存提取的角色
            rgba = np.zeros((*image_np.shape[:2], 4), dtype=np.uint8)
            rgba[:, :, :3] = image_np
            rgba[:, :, 3] = mask_np
            Image.fromarray(rgba).save(output_dir / "u2net_character.png")

            # 邊界框
            coords = np.column_stack(np.where(mask_np > 127))
            if len(coords) > 0:
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                bbox_area = (x_max - x_min) * (y_max - y_min)
                bbox_ratio = bbox_area / (image_np.shape[0] * image_np.shape[1]) * 100
            else:
                bbox_ratio = 0

            result = {
                "method": "U2Net/Rembg",
                "success": True,
                "time_seconds": elapsed,
                "coverage_percent": float(coverage),
                "bbox_area_percent": float(bbox_ratio),
                "notes": "當前使用的方法 - 可能只抓到頭部/顯著部位"
            }

            print(f"  ✓ 完成 - {elapsed:.2f}s")
            print(f"  覆蓋率: {coverage:.1f}%")
            print(f"  邊界框: {bbox_ratio:.1f}% 畫面")

            return result

        except Exception as e:
            print(f"  ✗ 失敗: {e}")
            return {"method": "U2Net/Rembg", "success": False, "error": str(e)}

    def test_yolo_detection(self, image_path: Path, output_dir: Path) -> Dict:
        """測試 YOLO 檢測"""
        print("\n2️⃣ 測試 YOLO 檢測...")

        try:
            from ultralytics import YOLO

            start_time = datetime.now()

            # 載入模型
            model = YOLO("yolov8x.pt")

            # 載入圖片
            image = cv2.imread(str(image_path))
            h, w = image.shape[:2]

            # 檢測
            results = model(image, conf=0.25, verbose=False)[0]

            elapsed = (datetime.now() - start_time).total_seconds()

            # 分析結果
            num_detections = len(results.boxes)
            total_bbox_area = 0

            # 創建可視化
            vis_image = image.copy()
            all_characters = []

            for idx, box in enumerate(results.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                # 計算面積
                bbox_area = (x2 - x1) * (y2 - y1)
                total_bbox_area += bbox_area

                # 繪製邊界框
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis_image, f"{conf:.2f}", (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # 裁切角色
                character_crop = image[y1:y2, x1:x2]
                all_characters.append(character_crop)

            # 保存結果
            output_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_dir / "yolo_detections.png"), vis_image)

            # 保存裁切的角色
            for idx, char in enumerate(all_characters):
                cv2.imwrite(str(output_dir / f"yolo_character_{idx}.png"), char)

            # 計算覆蓋率
            coverage = total_bbox_area / (h * w) * 100

            result = {
                "method": "YOLO Detection",
                "success": True,
                "time_seconds": elapsed,
                "num_detections": num_detections,
                "coverage_percent": float(coverage),
                "notes": "檢測完整物體邊界框 - 獲得完整角色"
            }

            print(f"  ✓ 完成 - {elapsed:.2f}s")
            print(f"  檢測到: {num_detections} 個物體")
            print(f"  覆蓋率: {coverage:.1f}%")

            return result

        except Exception as e:
            print(f"  ✗ 失敗: {e}")
            return {"method": "YOLO Detection", "success": False, "error": str(e)}

    def test_animeseg(self, image_path: Path, output_dir: Path) -> Dict:
        """測試 anime-segmentation"""
        print("\n3️⃣ 測試 anime-segmentation...")

        try:
            # 從 HuggingFace 載入模型
            import sys
            sys.path.insert(0, '/mnt/c/AI_LLM_projects/anime-segmentation')
            from train import AnimeSegmentation
            import torch
            from torchvision import transforms

            start_time = datetime.now()

            # 載入預訓練模型
            model = AnimeSegmentation.from_pretrained("skytnt/anime-seg")
            if self.device == "cuda" and torch.cuda.is_available():
                model = model.cuda()
            model.eval()

            # 載入圖片
            image = Image.open(image_path).convert("RGB")
            image_np = np.array(image)
            h, w = image_np.shape[:2]

            # 轉換為 tensor
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

            img_tensor = transform(image).unsqueeze(0)
            if self.device == "cuda" and torch.cuda.is_available():
                img_tensor = img_tensor.cuda()

            # 分割
            with torch.no_grad():
                pred = model(img_tensor)
                pred = pred.squeeze().cpu().numpy()

            # 轉換為 mask (0-255)
            mask = (pred * 255).astype(np.uint8)

            elapsed = (datetime.now() - start_time).total_seconds()

            # 分析結果
            coverage = np.sum(mask > 127) / mask.size * 100

            # 保存結果
            output_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask).save(output_dir / "animeseg_mask.png")

            # 提取角色
            rgba = np.zeros((*image_np.shape[:2], 4), dtype=np.uint8)
            rgba[:, :, :3] = image_np
            rgba[:, :, 3] = mask
            Image.fromarray(rgba).save(output_dir / "animeseg_character.png")

            # 邊界框
            coords = np.column_stack(np.where(mask > 127))
            if len(coords) > 0:
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                bbox_area = (x_max - x_min) * (y_max - y_min)
                bbox_ratio = bbox_area / (image_np.shape[0] * image_np.shape[1]) * 100
            else:
                bbox_ratio = 0

            result = {
                "method": "anime-segmentation",
                "success": True,
                "time_seconds": elapsed,
                "coverage_percent": float(coverage),
                "bbox_area_percent": float(bbox_ratio),
                "notes": "專為動畫設計 - 應該獲得完整角色"
            }

            print(f"  ✓ 完成 - {elapsed:.2f}s")
            print(f"  覆蓋率: {coverage:.1f}%")
            print(f"  邊界框: {bbox_ratio:.1f}% 畫面")

            return result

        except Exception as e:
            print(f"  ✗ 失敗: {e}")
            return {"method": "anime-segmentation", "success": False, "error": str(e)}

    def compare_methods(self, image_path: Path, output_dir: Path) -> Dict:
        """對比所有方法"""
        print(f"\n{'='*80}")
        print(f"測試圖片: {image_path.name}")
        print(f"{'='*80}")

        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        # 測試各種方法
        results["u2net"] = self.test_u2net_rembg(image_path, output_dir / "u2net")
        results["yolo"] = self.test_yolo_detection(image_path, output_dir / "yolo")
        results["animeseg"] = self.test_animeseg(image_path, output_dir / "animeseg")

        return results

    def generate_report(self, all_results: Dict, output_file: Path):
        """生成對比報告"""
        print(f"\n{'='*80}")
        print("生成對比報告...")
        print(f"{'='*80}\n")

        report = {
            "timestamp": datetime.now().isoformat(),
            "device": self.device,
            "test_images": list(all_results.keys()),
            "results": all_results,
            "summary": self.summarize_results(all_results)
        }

        # 保存 JSON
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"✓ 報告已保存: {output_file}")

        # 打印摘要
        print(f"\n{'='*80}")
        print("摘要")
        print(f"{'='*80}\n")

        summary = report["summary"]
        for method, stats in summary.items():
            print(f"{method}:")
            print(f"  成功率: {stats['success_rate']:.1f}%")
            if stats['success_rate'] > 0:
                print(f"  平均時間: {stats['avg_time']:.2f}s")
                print(f"  平均覆蓋率: {stats['avg_coverage']:.1f}%")
            print()

    def summarize_results(self, all_results: Dict) -> Dict:
        """統計摘要"""
        summary = {}

        # 收集每個方法的統計
        methods = set()
        for image_results in all_results.values():
            methods.update(image_results.keys())

        for method in methods:
            method_results = []
            for image_results in all_results.values():
                if method in image_results:
                    method_results.append(image_results[method])

            # 計算統計
            success_count = sum(1 for r in method_results if r.get("success", False))
            success_rate = success_count / len(method_results) * 100 if method_results else 0

            successful_results = [r for r in method_results if r.get("success", False)]

            if successful_results:
                avg_time = np.mean([r["time_seconds"] for r in successful_results])
                avg_coverage = np.mean([r.get("coverage_percent", 0) for r in successful_results])
            else:
                avg_time = 0
                avg_coverage = 0

            summary[method] = {
                "success_rate": success_rate,
                "avg_time": avg_time,
                "avg_coverage": avg_coverage,
                "total_tests": len(method_results)
            }

        return summary


def main():
    parser = argparse.ArgumentParser(description="對比分割/檢測模型")

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
    image_files = list(args.input_dir.glob("*.jpg")) + list(args.input_dir.glob("*.png"))
    image_files = sorted(image_files)[:args.num_samples]

    if not image_files:
        print(f"❌ 沒有找到圖片: {args.input_dir}")
        return

    print(f"\n找到 {len(image_files)} 張測試圖片")

    # 創建測試器
    tester = SegmentationTester(device=args.device)

    # 測試所有圖片
    all_results = {}

    for idx, image_path in enumerate(image_files, 1):
        print(f"\n{'#'*80}")
        print(f"測試 {idx}/{len(image_files)}")
        print(f"{'#'*80}")

        output_dir = args.output_dir / f"test_{idx:03d}"
        results = tester.compare_methods(image_path, output_dir)
        all_results[str(image_path.name)] = results

    # 生成報告
    report_file = args.output_dir / "comparison_report.json"
    tester.generate_report(all_results, report_file)

    print(f"\n{'='*80}")
    print("✅ 完成！")
    print(f"{'='*80}")
    print(f"\n查看結果:")
    print(f"  報告: {report_file}")
    print(f"  圖片: {args.output_dir}/test_*/")


if __name__ == "__main__":
    main()
