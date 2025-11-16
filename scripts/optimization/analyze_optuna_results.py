#!/usr/bin/env python3
"""
Analyze and Visualize Optuna Hyperparameter Optimization Results
Generates comprehensive reports and comparison visualizations
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

# Optional imports for visualization
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available. Install with: pip install matplotlib seaborn")


class OptunaResultsAnalyzer:
    """
    Analyze Optuna optimization results

    Features:
    - Load and parse trial data
    - Generate summary statistics
    - Create comparison visualizations
    - Export best configuration
    - Generate training configuration files
    """

    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.trials_data = None
        self.best_trial = None

        # Load data
        self.load_trials()

    def load_trials(self):
        """Load trials data from JSON file"""
        trials_file = self.results_dir / "all_trials.json"

        if not trials_file.exists():
            raise FileNotFoundError(f"Trials data not found: {trials_file}")

        with open(trials_file, 'r') as f:
            self.trials_data = json.load(f)

        # Load best parameters
        best_params_file = self.results_dir / "best_parameters.json"
        if best_params_file.exists():
            with open(best_params_file, 'r') as f:
                self.best_trial = json.load(f)

        print(f"✅ Loaded {len(self.trials_data)} trials")

    def get_completed_trials(self) -> List[Dict]:
        """Get only completed trials"""
        return [t for t in self.trials_data if t["state"] == "COMPLETE"]

    def generate_summary_statistics(self) -> Dict:
        """Generate summary statistics for all trials"""
        completed = self.get_completed_trials()

        if not completed:
            return {"error": "No completed trials"}

        # Extract scores
        scores = [t["value"] for t in completed]

        # Extract metrics
        brightness_values = [t["user_attrs"].get("mean_brightness") for t in completed
                            if "mean_brightness" in t["user_attrs"]]
        contrast_values = [t["user_attrs"].get("mean_contrast") for t in completed
                          if "mean_contrast" in t["user_attrs"]]

        summary = {
            "total_trials": len(self.trials_data),
            "completed_trials": len(completed),
            "pruned_trials": len([t for t in self.trials_data if t["state"] == "PRUNED"]),
            "failed_trials": len([t for t in self.trials_data if t["state"] == "FAIL"]),

            "scores": {
                "best": float(np.min(scores)),
                "worst": float(np.max(scores)),
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "median": float(np.median(scores)),
            },

            "brightness": {
                "mean": float(np.mean(brightness_values)) if brightness_values else None,
                "std": float(np.std(brightness_values)) if brightness_values else None,
                "min": float(np.min(brightness_values)) if brightness_values else None,
                "max": float(np.max(brightness_values)) if brightness_values else None,
            },

            "contrast": {
                "mean": float(np.mean(contrast_values)) if contrast_values else None,
                "std": float(np.std(contrast_values)) if contrast_values else None,
                "min": float(np.min(contrast_values)) if contrast_values else None,
                "max": float(np.max(contrast_values)) if contrast_values else None,
            },
        }

        return summary

    def create_trials_dataframe(self) -> pd.DataFrame:
        """Create pandas DataFrame from trials data"""
        completed = self.get_completed_trials()

        if not completed:
            return pd.DataFrame()

        # Flatten trial data
        records = []
        for trial in completed:
            record = {
                "trial_number": trial["number"],
                "score": trial["value"],
            }

            # Add parameters
            for param, value in trial["params"].items():
                record[f"param_{param}"] = value

            # Add metrics
            for metric, value in trial["user_attrs"].items():
                record[f"metric_{metric}"] = value

            records.append(record)

        df = pd.DataFrame(records)
        return df

    def create_comparison_table(self, top_n: int = 10) -> pd.DataFrame:
        """Create comparison table for top N trials"""
        df = self.create_trials_dataframe()

        if df.empty:
            return df

        # Sort by score (lower is better)
        df_sorted = df.sort_values("score").head(top_n)

        # Select key columns
        key_columns = [
            "trial_number",
            "score",
            "metric_mean_brightness",
            "metric_mean_contrast",
            "param_learning_rate",
            "param_network_dim",
            "param_network_alpha",
            "param_optimizer_type",
        ]

        available_columns = [col for col in key_columns if col in df_sorted.columns]
        comparison_table = df_sorted[available_columns]

        return comparison_table

    def plot_score_evolution(self, save_path: str = None):
        """Plot optimization score evolution over trials"""
        if not MATPLOTLIB_AVAILABLE:
            print("⚠️  Matplotlib not available, skipping plot")
            return

        completed = self.get_completed_trials()
        if not completed:
            print("⚠️  No completed trials to plot")
            return

        trial_numbers = [t["number"] for t in completed]
        scores = [t["value"] for t in completed]

        # Calculate cumulative best
        cumulative_best = []
        current_best = float('inf')
        for score in scores:
            current_best = min(current_best, score)
            cumulative_best.append(current_best)

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot individual trial scores
        ax.scatter(trial_numbers, scores, alpha=0.6, label="Trial Score", s=50)

        # Plot cumulative best
        ax.plot(trial_numbers, cumulative_best, color='red', linewidth=2,
                label="Best Score (cumulative)", alpha=0.8)

        ax.set_xlabel("Trial Number", fontsize=12)
        ax.set_ylabel("Combined Score (lower = better)", fontsize=12)
        ax.set_title("Optimization Score Evolution", fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Score evolution plot saved: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_metrics_comparison(self, save_path: str = None):
        """Plot brightness and contrast distributions"""
        if not MATPLOTLIB_AVAILABLE:
            print("⚠️  Matplotlib not available, skipping plot")
            return

        completed = self.get_completed_trials()
        if not completed:
            print("⚠️  No completed trials to plot")
            return

        brightness = [t["user_attrs"].get("mean_brightness") for t in completed
                     if "mean_brightness" in t["user_attrs"]]
        contrast = [t["user_attrs"].get("mean_contrast") for t in completed
                   if "mean_contrast" in t["user_attrs"]]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Brightness distribution
        axes[0].hist(brightness, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0].axvline(0.50, color='red', linestyle='--', linewidth=2, label='Target (0.50)')
        axes[0].axvspan(0.4, 0.6, alpha=0.2, color='green', label='Optimal Range')
        axes[0].set_xlabel("Mean Brightness", fontsize=12)
        axes[0].set_ylabel("Frequency", fontsize=12)
        axes[0].set_title("Brightness Distribution", fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Contrast distribution
        axes[1].hist(contrast, bins=20, alpha=0.7, color='coral', edgecolor='black')
        axes[1].axvline(0.20, color='red', linestyle='--', linewidth=2, label='Target (0.20)')
        axes[1].axvspan(0.15, 0.25, alpha=0.2, color='green', label='Optimal Range')
        axes[1].set_xlabel("Mean Contrast", fontsize=12)
        axes[1].set_ylabel("Frequency", fontsize=12)
        axes[1].set_title("Contrast Distribution (Pixar Style)", fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Metrics comparison plot saved: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_parameter_correlation(self, save_path: str = None):
        """Plot correlation between parameters and score"""
        if not MATPLOTLIB_AVAILABLE:
            print("⚠️  Matplotlib not available, skipping plot")
            return

        df = self.create_trials_dataframe()
        if df.empty:
            print("⚠️  No trial data to plot")
            return

        # Select numeric parameters
        param_cols = [col for col in df.columns if col.startswith("param_")]
        numeric_params = []

        for col in param_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_params.append(col)

        if not numeric_params:
            print("⚠️  No numeric parameters to plot")
            return

        n_params = len(numeric_params)
        fig, axes = plt.subplots(1, n_params, figsize=(5 * n_params, 4))

        if n_params == 1:
            axes = [axes]

        for ax, param_col in zip(axes, numeric_params):
            param_name = param_col.replace("param_", "")

            ax.scatter(df[param_col], df["score"], alpha=0.6, s=50)
            ax.set_xlabel(param_name, fontsize=11)
            ax.set_ylabel("Score", fontsize=11)
            ax.set_title(f"{param_name} vs Score", fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Parameter correlation plot saved: {save_path}")
        else:
            plt.show()

        plt.close()

    def generate_training_config(self, output_path: str):
        """Generate training configuration file with best parameters"""
        if not self.best_trial:
            print("❌ No best trial found")
            return

        params = self.best_trial["parameters"]

        # Create TOML-like config (simple key-value format)
        config_content = f"""# Best LoRA Training Configuration
# Generated from Optuna optimization
# Trial: {self.best_trial['trial_number']}
# Score: {self.best_trial['combined_score']:.4f}

# Network Architecture
network_dim = {params['network_dim']}
network_alpha = {params['network_alpha']}

# Learning Rates
learning_rate = {params['learning_rate']}
text_encoder_lr = {params['text_encoder_lr']}

# Optimizer
optimizer_type = "{params['optimizer_type']}"

# Learning Rate Scheduler
lr_scheduler = "{params['lr_scheduler']}"
lr_warmup_steps = {params['lr_warmup_steps']}
"""

        if "lr_scheduler_num_cycles" in params:
            config_content += f"lr_scheduler_num_cycles = {params['lr_scheduler_num_cycles']}\n"

        config_content += f"""
# Training Settings
max_train_epochs = {params['max_train_epochs']}
gradient_accumulation_steps = {params['gradient_accumulation_steps']}

# Performance Metrics
# Mean Brightness: {self.best_trial['metrics']['mean_brightness']:.3f}
# Mean Contrast: {self.best_trial['metrics']['mean_contrast']:.3f}
"""

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write(config_content)

        print(f"✅ Training config saved: {output_path}")

    def generate_markdown_report(self, output_path: str):
        """Generate comprehensive markdown report"""
        summary = self.generate_summary_statistics()
        comparison_table = self.create_comparison_table(top_n=10)

        report = f"""# Optuna Hyperparameter Optimization Report

**Generated:** {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}

## Summary Statistics

- **Total Trials:** {summary['total_trials']}
- **Completed Trials:** {summary['completed_trials']}
- **Pruned Trials:** {summary['pruned_trials']}
- **Failed Trials:** {summary['failed_trials']}

## Optimization Scores

- **Best Score:** {summary['scores']['best']:.4f}
- **Worst Score:** {summary['scores']['worst']:.4f}
- **Mean Score:** {summary['scores']['mean']:.4f} ± {summary['scores']['std']:.4f}
- **Median Score:** {summary['scores']['median']:.4f}

## Quality Metrics

### Brightness
- **Mean:** {summary['brightness']['mean']:.3f}
- **Std:** {summary['brightness']['std']:.3f}
- **Range:** [{summary['brightness']['min']:.3f}, {summary['brightness']['max']:.3f}]
- **Target:** 0.50 (Pixar optimal: 0.4-0.6)

### Contrast
- **Mean:** {summary['contrast']['mean']:.3f}
- **Std:** {summary['contrast']['std']:.3f}
- **Range:** [{summary['contrast']['min']:.3f}, {summary['contrast']['max']:.3f}]
- **Target:** 0.20 (Pixar optimal: 0.15-0.25)

"""

        if self.best_trial:
            report += f"""## Best Trial

**Trial Number:** {self.best_trial['trial_number']}
**Combined Score:** {self.best_trial['combined_score']:.4f}

### Hyperparameters

| Parameter | Value |
|-----------|-------|
"""
            for param, value in self.best_trial['parameters'].items():
                report += f"| {param} | {value} |\n"

            report += f"""
### Performance Metrics

| Metric | Value |
|--------|-------|
| Mean Brightness | {self.best_trial['metrics']['mean_brightness']:.3f} ± {self.best_trial['metrics']['std_brightness']:.3f} |
| Mean Contrast | {self.best_trial['metrics']['mean_contrast']:.3f} ± {self.best_trial['metrics']['std_contrast']:.3f} |

"""

        # Add top trials comparison
        if not comparison_table.empty:
            report += "\n## Top 10 Trials Comparison\n\n"
            report += comparison_table.to_markdown(index=False)
            report += "\n"

        # Save report
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write(report)

        print(f"✅ Markdown report saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Optuna optimization results")
    parser.add_argument("--results-dir", required=True, help="Directory containing Optuna results")
    parser.add_argument("--output-dir", help="Output directory for analysis (default: results-dir/analysis)")
    parser.add_argument("--top-n", type=int, default=10, help="Number of top trials to compare")

    args = parser.parse_args()

    results_dir = Path(args.results_dir) / "results"
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"📊 ANALYZING OPTUNA RESULTS")
    print(f"{'='*60}")
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")

    # Create analyzer
    analyzer = OptunaResultsAnalyzer(results_dir)

    # Generate summary statistics
    summary = analyzer.generate_summary_statistics()
    summary_file = output_dir / "summary_statistics.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Summary statistics saved: {summary_file}")

    # Generate comparison table
    comparison_table = analyzer.create_comparison_table(top_n=args.top_n)
    if not comparison_table.empty:
        comparison_file = output_dir / f"top_{args.top_n}_trials.csv"
        comparison_table.to_csv(comparison_file, index=False)
        print(f"✅ Comparison table saved: {comparison_file}")

    # Generate visualizations
    print("\n📈 Generating visualizations...")
    analyzer.plot_score_evolution(save_path=str(output_dir / "score_evolution.png"))
    analyzer.plot_metrics_comparison(save_path=str(output_dir / "metrics_comparison.png"))
    analyzer.plot_parameter_correlation(save_path=str(output_dir / "parameter_correlation.png"))

    # Generate training config
    if analyzer.best_trial:
        config_file = output_dir / "best_training_config.txt"
        analyzer.generate_training_config(config_file)

    # Generate markdown report
    report_file = output_dir / "OPTIMIZATION_REPORT.md"
    analyzer.generate_markdown_report(report_file)

    print(f"\n{'='*60}")
    print(f"✅ ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"📁 All results saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
