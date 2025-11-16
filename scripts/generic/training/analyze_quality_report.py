#!/usr/bin/env python3
"""
Analyze Quality Filter Report

Generates visualizations and statistics from quality filtering results.
"""

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List


def load_report(report_path: Path) -> Dict:
    """Load quality filter report"""
    with open(report_path, 'r') as f:
        return json.load(f)


def plot_cluster_statistics(report: Dict, output_dir: Path):
    """Plot cluster-level statistics"""
    clusters = report["clusters"]

    cluster_names = [c["cluster"] for c in clusters]
    total_counts = [c["total"] for c in clusters]
    passed_counts = [c["passed_quality"] for c in clusters]
    selected_counts = [c["selected"] for c in clusters]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Quality Filtering Statistics', fontsize=16)

    # 1. Total vs Passed vs Selected
    x = np.arange(len(cluster_names))
    width = 0.25

    axes[0, 0].bar(x - width, total_counts, width, label='Total', alpha=0.8)
    axes[0, 0].bar(x, passed_counts, width, label='Passed Quality', alpha=0.8)
    axes[0, 0].bar(x + width, selected_counts, width, label='Selected', alpha=0.8)
    axes[0, 0].set_xlabel('Cluster')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Image Counts per Cluster')
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis='x', rotation=90)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(cluster_names, rotation=90, ha='right')

    # 2. Pass rates
    pass_rates = [100 * c["passed_quality"] / c["total"] if c["total"] > 0 else 0
                  for c in clusters]

    axes[0, 1].bar(x, pass_rates, color='green', alpha=0.7)
    axes[0, 1].set_xlabel('Cluster')
    axes[0, 1].set_ylabel('Pass Rate (%)')
    axes[0, 1].set_title('Quality Check Pass Rate')
    axes[0, 1].tick_params(axis='x', rotation=90)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(cluster_names, rotation=90, ha='right')
    axes[0, 1].axhline(y=85, color='r', linestyle='--', label='Target (85%)')
    axes[0, 1].legend()

    # 3. Selection rates
    selection_rates = [100 * c["selected"] / c["total"] if c["total"] > 0 else 0
                       for c in clusters]

    axes[1, 0].bar(x, selection_rates, color='blue', alpha=0.7)
    axes[1, 0].set_xlabel('Cluster')
    axes[1, 0].set_ylabel('Selection Rate (%)')
    axes[1, 0].set_title('Final Selection Rate')
    axes[1, 0].tick_params(axis='x', rotation=90)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(cluster_names, rotation=90, ha='right')

    # 4. Rejection reasons
    rejected_sharpness = [c["rejected_sharpness"] for c in clusters]
    rejected_completeness = [c["rejected_completeness"] for c in clusters]

    axes[1, 1].bar(x, rejected_sharpness, width, label='Low Sharpness', alpha=0.8)
    axes[1, 1].bar(x, rejected_completeness, width, bottom=rejected_sharpness,
                   label='Low Completeness', alpha=0.8)
    axes[1, 1].set_xlabel('Cluster')
    axes[1, 1].set_ylabel('Rejected Count')
    axes[1, 1].set_title('Rejection Reasons')
    axes[1, 1].legend()
    axes[1, 1].tick_params(axis='x', rotation=90)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(cluster_names, rotation=90, ha='right')

    plt.tight_layout()

    output_path = output_dir / "quality_statistics.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved statistics plot: {output_path}")
    plt.close()


def plot_quality_distributions(filtered_dir: Path, output_dir: Path):
    """Plot quality metric distributions across all clusters"""

    all_sharpness = []
    all_completeness = []
    all_overall_scores = []

    # Load all quality metrics
    for cluster_dir in filtered_dir.iterdir():
        if not cluster_dir.is_dir():
            continue

        metrics_file = cluster_dir / "quality_metrics.json"
        if not metrics_file.exists():
            continue

        with open(metrics_file, 'r') as f:
            metrics_list = json.load(f)

        for m in metrics_list:
            all_sharpness.append(m["sharpness"])
            all_completeness.append(m["completeness"])
            all_overall_scores.append(m["overall_score"])

    if not all_sharpness:
        print("⚠️  No quality metrics found")
        return

    # Create distributions plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Quality Metric Distributions', fontsize=16)

    # Sharpness
    axes[0].hist(all_sharpness, bins=50, color='blue', alpha=0.7)
    axes[0].set_xlabel('Sharpness (Laplacian Variance)')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'Sharpness Distribution\n(mean: {np.mean(all_sharpness):.1f})')
    axes[0].axvline(x=100, color='r', linestyle='--', label='Min Threshold')
    axes[0].legend()

    # Completeness
    axes[1].hist(all_completeness, bins=50, color='green', alpha=0.7)
    axes[1].set_xlabel('Completeness Ratio')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'Completeness Distribution\n(mean: {np.mean(all_completeness):.3f})')
    axes[1].axvline(x=0.85, color='r', linestyle='--', label='Min Threshold')
    axes[1].legend()

    # Overall Score
    axes[2].hist(all_overall_scores, bins=50, color='purple', alpha=0.7)
    axes[2].set_xlabel('Overall Quality Score')
    axes[2].set_ylabel('Count')
    axes[2].set_title(f'Overall Score Distribution\n(mean: {np.mean(all_overall_scores):.3f})')
    axes[2].legend()

    plt.tight_layout()

    output_path = output_dir / "quality_distributions.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved distributions plot: {output_path}")
    plt.close()


def print_summary(report: Dict):
    """Print text summary"""
    clusters = report["clusters"]
    params = report["parameters"]

    total_input = sum(c["total"] for c in clusters)
    total_passed = sum(c["passed_quality"] for c in clusters)
    total_selected = sum(c["selected"] for c in clusters)

    print("\n" + "="*70)
    print("QUALITY FILTERING SUMMARY")
    print("="*70)
    print(f"\nParameters:")
    print(f"  Target per cluster: {params['target_per_cluster']}")
    print(f"  Min sharpness: {params['min_sharpness']}")
    print(f"  Min completeness: {params['min_completeness']}")
    print(f"  Diversity method: {params['diversity_method']}")

    print(f"\nOverall Statistics:")
    print(f"  Total clusters: {len(clusters)}")
    print(f"  Total input images: {total_input}")
    print(f"  Passed quality check: {total_passed} ({100*total_passed/total_input:.1f}%)")
    print(f"  Final selection: {total_selected} ({100*total_selected/total_input:.1f}%)")

    print(f"\nPer-Cluster Breakdown:")
    print(f"  {'Cluster':<20} {'Total':>8} {'Passed':>8} {'Selected':>8} {'Pass %':>8} {'Select %':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for c in clusters:
        pass_pct = 100 * c["passed_quality"] / c["total"] if c["total"] > 0 else 0
        select_pct = 100 * c["selected"] / c["total"] if c["total"] > 0 else 0
        print(f"  {c['cluster']:<20} {c['total']:>8} {c['passed_quality']:>8} "
              f"{c['selected']:>8} {pass_pct:>7.1f}% {select_pct:>7.1f}%")

    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze Quality Filter Report")
    parser.add_argument(
        "--report",
        type=str,
        required=True,
        help="Path to quality_filter_report.json"
    )
    parser.add_argument(
        "--filtered-dir",
        type=str,
        help="Path to filtered instances directory (for distributions)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for plots (default: same as report)"
    )

    args = parser.parse_args()

    report_path = Path(args.report)

    if not report_path.exists():
        print(f"❌ Report not found: {report_path}")
        return

    # Load report
    report = load_report(report_path)

    # Print summary
    print_summary(report)

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = report_path.parent

    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot cluster statistics
    plot_cluster_statistics(report, output_dir)

    # Plot quality distributions (if filtered-dir provided)
    if args.filtered_dir:
        filtered_dir = Path(args.filtered_dir)
        if filtered_dir.exists():
            plot_quality_distributions(filtered_dir, output_dir)
        else:
            print(f"⚠️  Filtered directory not found: {filtered_dir}")

    print(f"\n✓ Analysis complete! Plots saved to: {output_dir}\n")


if __name__ == "__main__":
    main()
