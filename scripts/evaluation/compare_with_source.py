#!/usr/bin/env python3
"""
自動化分析工具：比較生成圖像 vs 原始電影截圖

量化指標：
1. 對比度 (Contrast)
2. 動態範圍 (Dynamic Range)
3. 色彩飽和度 (Saturation)
4. 亮度分佈 (Brightness Distribution)
5. 光照均勻性 (Lighting Uniformity)
6. 膚色一致性 (Skin Tone Consistency)
7. SSIM 結構相似度
8. 色彩直方圖距離
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image, ImageStat
import cv2
from scipy import stats
from collections import defaultdict


class ImageAnalyzer:
    """圖像分析器"""

    def __init__(self):
        self.metrics = {}

    def analyze_contrast(self, img_array: np.ndarray) -> Dict:
        """
        分析對比度

        Pixar 電影特徵：
        - 標準差: 25-35
        - 動態範圍: 150-180

        高對比度問題：
        - 標準差: 45-60
        - 動態範圍: 200-240
        """
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        return {
            'std_dev': float(np.std(gray)),
            'dynamic_range': int(gray.max() - gray.min()),
            'mean_brightness': float(np.mean(gray)),
            'contrast_level': 'high' if np.std(gray) > 40 else ('medium' if np.std(gray) > 30 else 'low')
        }

    def analyze_color_saturation(self, img_array: np.ndarray) -> Dict:
        """
        分析色彩飽和度

        Pixar 特徵：
        - 平衡飽和度，不過度鮮豔
        - 色彩協調
        """
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1]

        return {
            'mean_saturation': float(np.mean(saturation)),
            'saturation_std': float(np.std(saturation)),
            'max_saturation': int(saturation.max()),
            'saturation_level': 'high' if np.mean(saturation) > 100 else ('balanced' if np.mean(saturation) > 60 else 'low')
        }

    def analyze_brightness_distribution(self, img_array: np.ndarray) -> Dict:
        """
        分析亮度分佈

        檢查是否有過暗或過亮區域
        """
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        hist, _ = np.histogram(gray, bins=256, range=(0, 256))

        # 計算各區域的像素比例
        shadows = np.sum(hist[:85]) / gray.size  # 0-85: 陰影
        midtones = np.sum(hist[85:170]) / gray.size  # 85-170: 中間調
        highlights = np.sum(hist[170:]) / gray.size  # 170-255: 高光

        return {
            'shadows_ratio': float(shadows),
            'midtones_ratio': float(midtones),
            'highlights_ratio': float(highlights),
            'skewness': float(stats.skew(hist)),
            'kurtosis': float(stats.kurtosis(hist))
        }

    def analyze_lighting_uniformity(self, img_array: np.ndarray) -> Dict:
        """
        分析光照均勻性

        Pixar 特徵：
        - 全局照明均勻
        - 少量方向性光源
        - 沒有強烈熱點
        """
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # 分割成多個區域並計算標準差
        h, w = gray.shape
        grid_size = 4
        regions = []

        for i in range(grid_size):
            for j in range(grid_size):
                y1 = i * h // grid_size
                y2 = (i + 1) * h // grid_size
                x1 = j * w // grid_size
                x2 = (j + 1) * w // grid_size
                region = gray[y1:y2, x1:x2]
                regions.append(np.mean(region))

        # 區域間亮度差異
        region_std = np.std(regions)

        return {
            'region_brightness_std': float(region_std),
            'uniformity_score': float(100 - min(region_std, 100)),  # 0-100
            'uniformity_level': 'high' if region_std < 20 else ('medium' if region_std < 40 else 'low')
        }

    def analyze_skin_tone_consistency(self, img_array: np.ndarray, mask: np.ndarray = None) -> Dict:
        """
        分析膚色一致性

        Pixar 特徵：
        - 膚色均勻
        - 沒有過度高光或陰影
        """
        # 簡化版：檢測膚色範圍的一致性
        # 真實應用需要人臉檢測

        # YCrCb 色彩空間對膚色檢測更有效
        ycrcb = cv2.cvtColor(img_array, cv2.COLOR_RGB2YCrCb)

        # 膚色範圍（粗略）
        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([255, 173, 127], dtype=np.uint8)

        skin_mask = cv2.inRange(ycrcb, lower, upper)
        skin_pixels = img_array[skin_mask > 0]

        if len(skin_pixels) == 0:
            return {
                'detected': False,
                'error': 'No skin tone detected'
            }

        return {
            'detected': True,
            'mean_color': [int(x) for x in np.mean(skin_pixels, axis=0)],
            'color_std': [float(x) for x in np.std(skin_pixels, axis=0)],
            'consistency_score': float(100 - min(np.mean(np.std(skin_pixels, axis=0)), 100))
        }

    def compare_images(self, img1: np.ndarray, img2: np.ndarray) -> Dict:
        """
        比較兩張圖像的結構相似度和色彩差異
        """
        from skimage.metrics import structural_similarity as ssim

        # 確保兩張圖像尺寸相同
        if img1.shape != img2.shape:
            h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
            img1 = cv2.resize(img1, (w, h))
            img2 = cv2.resize(img2, (w, h))

        # 轉灰階計算 SSIM
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

        ssim_score = ssim(gray1, gray2)

        # 色彩直方圖距離
        hist1 = [cv2.calcHist([img1], [i], None, [256], [0, 256]) for i in range(3)]
        hist2 = [cv2.calcHist([img2], [i], None, [256], [0, 256]) for i in range(3)]

        hist_distances = [
            cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)
            for h1, h2 in zip(hist1, hist2)
        ]

        return {
            'ssim': float(ssim_score),
            'hist_distance_r': float(hist_distances[0]),
            'hist_distance_g': float(hist_distances[1]),
            'hist_distance_b': float(hist_distances[2]),
            'avg_hist_distance': float(np.mean(hist_distances))
        }

    def full_analysis(self, image_path: Path) -> Dict:
        """完整分析單張圖像"""
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)

        return {
            'image_path': str(image_path),
            'resolution': f"{img.width}x{img.height}",
            'contrast': self.analyze_contrast(img_array),
            'saturation': self.analyze_color_saturation(img_array),
            'brightness': self.analyze_brightness_distribution(img_array),
            'lighting': self.analyze_lighting_uniformity(img_array),
            'skin_tone': self.analyze_skin_tone_consistency(img_array)
        }


def analyze_dataset(
    source_dir: Path,
    generated_dir: Path = None,
    output_path: Path = None,
    sample_size: int = 50
) -> Dict:
    """
    分析整個數據集

    Args:
        source_dir: 原始電影截圖目錄
        generated_dir: 生成圖像目錄（可選）
        output_path: 輸出 JSON 報告路徑
        sample_size: 分析的樣本數量
    """
    analyzer = ImageAnalyzer()

    # 分析原始圖像
    print(f"📊 分析原始電影截圖...")
    source_images = list(source_dir.glob("*.png"))[:sample_size]

    source_results = []
    for img_path in source_images:
        print(f"  分析: {img_path.name}")
        result = analyzer.full_analysis(img_path)
        source_results.append(result)

    # 統計原始圖像的平均值
    source_stats = aggregate_statistics(source_results)

    report = {
        'source_images': {
            'count': len(source_results),
            'directory': str(source_dir),
            'statistics': source_stats,
            'samples': source_results[:5]  # 保存前5個樣本
        }
    }

    # 如果有生成圖像，也分析並對比
    if generated_dir and generated_dir.exists():
        print(f"\n📊 分析生成圖像...")
        generated_images = list(generated_dir.glob("*.png"))[:sample_size]

        generated_results = []
        for img_path in generated_images:
            print(f"  分析: {img_path.name}")
            result = analyzer.full_analysis(img_path)
            generated_results.append(result)

        generated_stats = aggregate_statistics(generated_results)

        report['generated_images'] = {
            'count': len(generated_results),
            'directory': str(generated_dir),
            'statistics': generated_stats,
            'samples': generated_results[:5]
        }

        # 對比分析
        print(f"\n📊 對比分析...")
        report['comparison'] = compare_statistics(source_stats, generated_stats)

    # 生成診斷建議
    report['diagnosis'] = generate_diagnosis(report)

    # 保存報告
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n✅ 報告已保存: {output_path}")

    return report


def aggregate_statistics(results: List[Dict]) -> Dict:
    """聚合多個圖像的統計數據"""
    stats = defaultdict(list)

    for result in results:
        # 對比度
        stats['contrast_std'].append(result['contrast']['std_dev'])
        stats['dynamic_range'].append(result['contrast']['dynamic_range'])

        # 飽和度
        stats['saturation_mean'].append(result['saturation']['mean_saturation'])

        # 光照均勻性
        stats['lighting_uniformity'].append(result['lighting']['uniformity_score'])

        # 亮度分佈
        stats['shadows_ratio'].append(result['brightness']['shadows_ratio'])
        stats['midtones_ratio'].append(result['brightness']['midtones_ratio'])
        stats['highlights_ratio'].append(result['brightness']['highlights_ratio'])

    # 計算平均值和標準差
    aggregated = {}
    for key, values in stats.items():
        aggregated[key] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values))
        }

    return aggregated


def compare_statistics(source_stats: Dict, generated_stats: Dict) -> Dict:
    """對比原始和生成圖像的統計數據"""
    comparison = {}

    for key in source_stats:
        if key in generated_stats:
            source_mean = source_stats[key]['mean']
            generated_mean = generated_stats[key]['mean']

            diff = generated_mean - source_mean
            diff_percent = (diff / source_mean * 100) if source_mean != 0 else 0

            comparison[key] = {
                'source': source_mean,
                'generated': generated_mean,
                'difference': float(diff),
                'difference_percent': float(diff_percent)
            }

    return comparison


def generate_diagnosis(report: Dict) -> Dict:
    """根據數據生成診斷建議"""
    diagnosis = {
        'issues': [],
        'recommendations': []
    }

    if 'comparison' in report:
        comp = report['comparison']

        # 檢查對比度
        if 'contrast_std' in comp:
            diff = comp['contrast_std']['difference_percent']
            if diff > 20:
                diagnosis['issues'].append({
                    'severity': 'high',
                    'category': 'contrast',
                    'description': f'對比度過高（高出原片 {diff:.1f}%）',
                    'metric': f"原片: {comp['contrast_std']['source']:.1f}, 生成: {comp['contrast_std']['generated']:.1f}"
                })
                diagnosis['recommendations'].append({
                    'category': 'caption',
                    'action': '添加 "low contrast, even illumination" 到 caption'
                })
                diagnosis['recommendations'].append({
                    'category': 'training',
                    'action': '降低學習率或增加 text encoder 權重'
                })

        # 檢查光照均勻性
        if 'lighting_uniformity' in comp:
            diff = comp['lighting_uniformity']['difference_percent']
            if diff < -20:  # 生成圖像的均勻性更差
                diagnosis['issues'].append({
                    'severity': 'high',
                    'category': 'lighting',
                    'description': f'光照不均勻（差於原片 {abs(diff):.1f}%）',
                    'metric': f"原片: {comp['lighting_uniformity']['source']:.1f}, 生成: {comp['lighting_uniformity']['generated']:.1f}"
                })
                diagnosis['recommendations'].append({
                    'category': 'caption',
                    'action': '添加 "pixar uniform lighting, no harsh shadows"'
                })

        # 檢查飽和度
        if 'saturation_mean' in comp:
            diff = comp['saturation_mean']['difference_percent']
            if diff > 15:
                diagnosis['issues'].append({
                    'severity': 'medium',
                    'category': 'saturation',
                    'description': f'色彩過度飽和（高出原片 {diff:.1f}%）',
                    'metric': f"原片: {comp['saturation_mean']['source']:.1f}, 生成: {comp['saturation_mean']['generated']:.1f}"
                })
                diagnosis['recommendations'].append({
                    'category': 'caption',
                    'action': '添加 "balanced saturation, film color grading"'
                })

    return diagnosis


def print_report_summary(report: Dict):
    """打印報告摘要"""
    print("\n" + "="*70)
    print("📊 圖像質量分析報告")
    print("="*70)

    if 'source_images' in report:
        print(f"\n原始電影截圖統計 ({report['source_images']['count']} 張):")
        stats = report['source_images']['statistics']
        print(f"  對比度標準差: {stats['contrast_std']['mean']:.1f} ± {stats['contrast_std']['std']:.1f}")
        print(f"  光照均勻性: {stats['lighting_uniformity']['mean']:.1f}/100")
        print(f"  色彩飽和度: {stats['saturation_mean']['mean']:.1f}/255")

    if 'generated_images' in report:
        print(f"\n生成圖像統計 ({report['generated_images']['count']} 張):")
        stats = report['generated_images']['statistics']
        print(f"  對比度標準差: {stats['contrast_std']['mean']:.1f} ± {stats['contrast_std']['std']:.1f}")
        print(f"  光照均勻性: {stats['lighting_uniformity']['mean']:.1f}/100")
        print(f"  色彩飽和度: {stats['saturation_mean']['mean']:.1f}/255")

    if 'comparison' in report:
        print(f"\n📈 對比分析:")
        comp = report['comparison']
        if 'contrast_std' in comp:
            print(f"  對比度差異: {comp['contrast_std']['difference_percent']:+.1f}%")
        if 'lighting_uniformity' in comp:
            print(f"  光照均勻性差異: {comp['lighting_uniformity']['difference_percent']:+.1f}%")
        if 'saturation_mean' in comp:
            print(f"  飽和度差異: {comp['saturation_mean']['difference_percent']:+.1f}%")

    if 'diagnosis' in report:
        diagnosis = report['diagnosis']

        if diagnosis['issues']:
            print(f"\n⚠️  發現 {len(diagnosis['issues'])} 個問題:")
            for issue in diagnosis['issues']:
                print(f"  [{issue['severity'].upper()}] {issue['description']}")
                print(f"    數據: {issue['metric']}")

        if diagnosis['recommendations']:
            print(f"\n💡 改善建議 ({len(diagnosis['recommendations'])} 條):")
            for rec in diagnosis['recommendations']:
                print(f"  [{rec['category']}] {rec['action']}")

    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description="自動化分析工具：比較生成圖像 vs 原始電影截圖"
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="原始電影截圖目錄"
    )
    parser.add_argument(
        "--generated",
        type=Path,
        default=None,
        help="生成圖像目錄（可選）"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="輸出 JSON 報告路徑"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="分析的樣本數量"
    )

    args = parser.parse_args()

    if not args.source.exists():
        print(f"錯誤：原始圖像目錄不存在: {args.source}")
        return 1

    # 運行分析
    report = analyze_dataset(
        source_dir=args.source,
        generated_dir=args.generated,
        output_path=args.output,
        sample_size=args.sample_size
    )

    # 打印摘要
    print_report_summary(report)

    return 0


if __name__ == "__main__":
    exit(main())
