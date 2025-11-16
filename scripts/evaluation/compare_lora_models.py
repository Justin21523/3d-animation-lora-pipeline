#!/usr/bin/env python3
"""
Compare Multiple LoRA Models
Side-by-side comparison of quality metrics across different LoRA versions
"""

import json
from pathlib import Path
import argparse
from typing import List, Dict
import matplotlib.pyplot as plt
import pandas as pd


class LoRAComparator:
    """Compare multiple LoRA evaluation results"""

    def __init__(self, evaluation_jsons: List[Path]):
        """
        Initialize comparator with evaluation results

        Args:
            evaluation_jsons: List of paths to quality_evaluation.json files
        """
        self.results = []
        self.model_names = []

        for json_path in evaluation_jsons:
            if not json_path.exists():
                print(f"⚠️  Warning: {json_path} not found, skipping")
                continue

            with open(json_path, 'r') as f:
                data = json.load(f)
                self.results.append(data)
                # Extract model name from output_dir
                output_dir = Path(data.get("output_dir", ""))
                self.model_names.append(output_dir.name)

        if not self.results:
            raise ValueError("No valid evaluation results found")

        print(f"Loaded {len(self.results)} LoRA evaluation results")

    def generate_comparison_report(self) -> Dict:
        """Generate comprehensive comparison report"""

        comparison = {
            "models": self.model_names,
            "summary": {},
            "detailed_comparison": []
        }

        # Compare overall metrics
        for model_name, result in zip(self.model_names, self.results):
            overall = result.get("overall_metrics", {})

            model_summary = {
                "model_name": model_name,
                "total_images": result.get("total_images", 0),
                "num_prompts": result.get("num_prompts", 0),
                "clip_score_mean": overall.get("clip_score", {}).get("mean", 0),
                "clip_score_std": overall.get("clip_score", {}).get("std", 0),
                "consistency_mean": overall.get("character_consistency", {}).get("mean", 0),
                "consistency_std": overall.get("character_consistency", {}).get("std", 0),
            }

            comparison["detailed_comparison"].append(model_summary)

        return comparison

    def print_comparison_table(self, comparison: Dict):
        """Print comparison table"""

        print(f"\n{'='*100}")
        print("LORA MODEL COMPARISON")
        print(f"{'='*100}\n")

        # Create DataFrame for easy comparison
        df = pd.DataFrame(comparison["detailed_comparison"])

        print("📊 Overall Metrics Comparison:\n")
        print(df.to_string(index=False))
        print()

        # Find best and worst performers
        best_clip_idx = df["clip_score_mean"].idxmax()
        best_consistency_idx = df["consistency_mean"].idxmax()

        print(f"\n🏆 Best Performers:")
        print(f"  CLIP Score:    {df.loc[best_clip_idx, 'model_name']} ({df.loc[best_clip_idx, 'clip_score_mean']:.2f})")
        print(f"  Consistency:   {df.loc[best_consistency_idx, 'model_name']} ({df.loc[best_consistency_idx, 'consistency_mean']:.2f}%)")

        # Statistical comparison
        if len(df) > 1:
            print(f"\n📈 Differences:")
            clip_diff = df["clip_score_mean"].max() - df["clip_score_mean"].min()
            cons_diff = df["consistency_mean"].max() - df["consistency_mean"].min()
            print(f"  CLIP Score Range:    {clip_diff:.2f} points")
            print(f"  Consistency Range:   {cons_diff:.2f}%")

    def create_visualization(self, comparison: Dict, output_path: Path):
        """Create visualization comparing models"""

        df = pd.DataFrame(comparison["detailed_comparison"])

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # CLIP Score comparison
        axes[0].bar(df["model_name"], df["clip_score_mean"],
                   yerr=df["clip_score_std"], capsize=5,
                   color='steelblue', alpha=0.8)
        axes[0].set_xlabel("LoRA Model", fontsize=12)
        axes[0].set_ylabel("CLIP Score (0-100)", fontsize=12)
        axes[0].set_title("Prompt Adherence (CLIP Score)", fontsize=14, fontweight='bold')
        axes[0].axhline(y=30, color='green', linestyle='--', label='Excellent (≥30)')
        axes[0].axhline(y=25, color='orange', linestyle='--', label='Good (≥25)')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        axes[0].tick_params(axis='x', rotation=15)

        # Consistency comparison
        axes[1].bar(df["model_name"], df["consistency_mean"],
                   yerr=df["consistency_std"], capsize=5,
                   color='coral', alpha=0.8)
        axes[1].set_xlabel("LoRA Model", fontsize=12)
        axes[1].set_ylabel("Consistency (%)", fontsize=12)
        axes[1].set_title("Character Consistency", fontsize=14, fontweight='bold')
        axes[1].axhline(y=85, color='green', linestyle='--', label='Excellent (≥85%)')
        axes[1].axhline(y=75, color='orange', linestyle='--', label='Good (≥75%)')
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)
        axes[1].tick_params(axis='x', rotation=15)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple LoRA model evaluations"
    )
    parser.add_argument(
        "evaluation_dirs",
        type=Path,
        nargs="+",
        help="Directories containing quality_evaluation.json files"
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Save comparison to JSON file"
    )
    parser.add_argument(
        "--output-viz",
        type=Path,
        help="Save visualization to file (PNG)"
    )

    args = parser.parse_args()

    # Find evaluation JSON files
    evaluation_jsons = []
    for eval_dir in args.evaluation_dirs:
        json_path = eval_dir / "quality_evaluation.json"
        if json_path.exists():
            evaluation_jsons.append(json_path)
        else:
            print(f"⚠️  Warning: {json_path} not found")

    if not evaluation_jsons:
        print("❌ No evaluation results found")
        return

    # Create comparator
    comparator = LoRAComparator(evaluation_jsons)

    # Generate comparison
    comparison = comparator.generate_comparison_report()

    # Print comparison table
    comparator.print_comparison_table(comparison)

    # Save to JSON
    if args.output_json:
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Comparison saved to: {args.output_json}")

    # Create visualization
    if args.output_viz:
        try:
            comparator.create_visualization(comparison, args.output_viz)
        except ImportError:
            print("⚠️  Matplotlib/Pandas not available, skipping visualization")

    print(f"\n{'='*100}\n")


if __name__ == "__main__":
    main()
