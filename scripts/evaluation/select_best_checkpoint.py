#!/usr/bin/env python3
"""
Best Checkpoint Selection - Multi-Metric Aggregation
====================================================

Selects the best LoRA checkpoint based on balanced quality metrics:
- Control Precision (pose/action/expression accuracy)
- Identity Consistency (face similarity across variations)
- Generalization (performance on unseen prompts)

Weighted Scoring Formula:
    Final Score = 0.35 × Control Precision
                + 0.35 × Identity Consistency
                + 0.30 × Generalization

Usage:
    python scripts/evaluation/select_best_checkpoint.py \\
        --checkpoint-dir /path/to/lora_checkpoints \\
        --lora-type pose \\
        --character-name "luca" \\
        --output-dir /path/to/selection_results \\
        --device cuda

Author: LLMProvider Tooling
Date: 2025-12-04
"""

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

import numpy as np
from PIL import Image
import torch

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Metric Weights and Thresholds
# ============================================================================

# Weighted scoring formula
METRIC_WEIGHTS = {
    'control_precision': 0.35,
    'identity_consistency': 0.35,
    'generalization': 0.30
}

# Type-specific minimum thresholds
MINIMUM_THRESHOLDS = {
    'pose': {
        'clip_similarity': 0.30,
        'pose_accuracy': 0.80,
        'identity_consistency': 0.85
    },
    'action': {
        'clip_similarity': 0.28,
        'motion_clarity': 0.60,
        'action_recognition': 0.70,
        'identity_consistency': 0.85
    },
    'expression': {
        'clip_similarity': 0.32,
        'expression_accuracy': 0.70,
        'facial_quality': 0.80,
        'identity_consistency': 0.85
    }
}


# ============================================================================
# Checkpoint Metadata Loading
# ============================================================================

def find_checkpoints(checkpoint_dir: Path) -> List[Path]:
    """
    Find all checkpoint files in directory.

    Args:
        checkpoint_dir: Directory containing checkpoints

    Returns:
        List of checkpoint paths (sorted by epoch)
    """
    checkpoints = sorted(checkpoint_dir.glob('*.safetensors'))

    if not checkpoints:
        logger.warning(f"No checkpoints found in {checkpoint_dir}")

    return checkpoints


def load_evaluation_results(
    checkpoint_dir: Path,
    lora_type: str
) -> Dict[str, Any]:
    """
    Load evaluation results for all checkpoints.

    Args:
        checkpoint_dir: Directory containing checkpoints and evaluation results
        lora_type: LoRA type (pose/action/expression)

    Returns:
        Dictionary mapping checkpoint name to evaluation metrics
    """
    eval_results = {}

    # Look for evaluation JSON files
    eval_dir = checkpoint_dir / 'evaluation'

    if not eval_dir.exists():
        logger.warning(f"Evaluation directory not found: {eval_dir}")
        return {}

    # Load per-checkpoint evaluation results
    for eval_file in eval_dir.glob('*.json'):
        try:
            with open(eval_file, 'r') as f:
                data = json.load(f)

            checkpoint_name = eval_file.stem
            eval_results[checkpoint_name] = data

        except Exception as e:
            logger.warning(f"Failed to load {eval_file}: {e}")

    return eval_results


# ============================================================================
# Metric Computation
# ============================================================================

def compute_control_precision(
    eval_data: Dict[str, Any],
    lora_type: str
) -> float:
    """
    Compute control precision score.

    Args:
        eval_data: Evaluation results for a checkpoint
        lora_type: LoRA type (pose/action/expression)

    Returns:
        Control precision score (0-1)
    """
    if lora_type == 'pose':
        # Pose: RTM-Pose accuracy + CLIP similarity
        pose_acc = eval_data.get('pose_accuracy', {}).get('avg_pose_confidence', 0.0)
        clip_sim = eval_data.get('clip_similarity', {}).get('avg_score', 0.0)
        return (pose_acc + clip_sim) / 2.0

    elif lora_type == 'action':
        # Action: Action recognition + Motion clarity + CLIP similarity
        action_rec = eval_data.get('action_recognition', {}).get('action_match_rate', 0.0)
        motion_clarity = eval_data.get('motion_clarity', {}).get('avg_motion_magnitude', 0.0)
        clip_sim = eval_data.get('clip_similarity', {}).get('avg_score', 0.0)

        # Normalize motion clarity (0-50 pixels → 0-1)
        motion_clarity_norm = min(motion_clarity / 50.0, 1.0)

        return (action_rec + motion_clarity_norm + clip_sim) / 3.0

    elif lora_type == 'expression':
        # Expression: FER accuracy + CLIP similarity
        fer_acc = eval_data.get('expression_accuracy', {}).get('exact_match_rate', 0.0)
        clip_sim = eval_data.get('clip_similarity', {}).get('avg_score', 0.0)
        return (fer_acc + clip_sim) / 2.0

    else:
        logger.warning(f"Unknown LoRA type: {lora_type}")
        return 0.0


def compute_identity_consistency(
    eval_data: Dict[str, Any]
) -> float:
    """
    Compute identity consistency score.

    Args:
        eval_data: Evaluation results for a checkpoint

    Returns:
        Identity consistency score (0-1)
    """
    # Face similarity across variations (using ArcFace embeddings)
    face_similarity = eval_data.get('face_similarity', {})

    # Average pairwise similarity
    avg_similarity = face_similarity.get('avg_pairwise_similarity', 0.0)

    # Consistency = high similarity, low variance
    similarity_std = face_similarity.get('similarity_std', 0.0)
    consistency = avg_similarity * (1.0 - min(similarity_std, 0.3) / 0.3)

    return consistency


def compute_generalization(
    eval_data: Dict[str, Any]
) -> float:
    """
    Compute generalization score.

    Args:
        eval_data: Evaluation results for a checkpoint

    Returns:
        Generalization score (0-1)
    """
    # Performance on unseen prompts
    unseen_prompts = eval_data.get('unseen_prompts', {})

    # CLIP score on unseen prompts
    unseen_clip = unseen_prompts.get('avg_clip_score', 0.0)

    # Cross-lighting robustness
    cross_lighting = eval_data.get('cross_lighting', {})
    lighting_robustness = cross_lighting.get('consistency_score', 1.0)

    # Cross-angle robustness
    cross_angle = eval_data.get('cross_angle', {})
    angle_robustness = cross_angle.get('consistency_score', 1.0)

    # Average robustness metrics
    generalization = (unseen_clip + lighting_robustness + angle_robustness) / 3.0

    return generalization


def compute_final_score(
    eval_data: Dict[str, Any],
    lora_type: str
) -> Dict[str, float]:
    """
    Compute final weighted score for a checkpoint.

    Args:
        eval_data: Evaluation results
        lora_type: LoRA type

    Returns:
        Dictionary with component scores and final score
    """
    # Compute component scores
    control = compute_control_precision(eval_data, lora_type)
    identity = compute_identity_consistency(eval_data)
    generalization = compute_generalization(eval_data)

    # Weighted final score
    final_score = (
        METRIC_WEIGHTS['control_precision'] * control +
        METRIC_WEIGHTS['identity_consistency'] * identity +
        METRIC_WEIGHTS['generalization'] * generalization
    )

    return {
        'control_precision': control,
        'identity_consistency': identity,
        'generalization': generalization,
        'final_score': final_score
    }


def check_minimum_thresholds(
    eval_data: Dict[str, Any],
    lora_type: str
) -> Tuple[bool, List[str]]:
    """
    Check if checkpoint meets minimum quality thresholds.

    Args:
        eval_data: Evaluation results
        lora_type: LoRA type

    Returns:
        (passes_thresholds, list_of_failures)
    """
    thresholds = MINIMUM_THRESHOLDS.get(lora_type, {})
    failures = []

    for metric_name, min_value in thresholds.items():
        # Navigate nested dictionary structure
        if metric_name == 'clip_similarity':
            actual_value = eval_data.get('clip_similarity', {}).get('avg_score', 0.0)
        elif metric_name == 'pose_accuracy':
            actual_value = eval_data.get('pose_accuracy', {}).get('avg_pose_confidence', 0.0)
        elif metric_name == 'motion_clarity':
            actual_value = eval_data.get('motion_clarity', {}).get('avg_motion_magnitude', 0.0) / 50.0
        elif metric_name == 'action_recognition':
            actual_value = eval_data.get('action_recognition', {}).get('action_match_rate', 0.0)
        elif metric_name == 'expression_accuracy':
            actual_value = eval_data.get('expression_accuracy', {}).get('exact_match_rate', 0.0)
        elif metric_name == 'facial_quality':
            actual_value = eval_data.get('facial_quality', {}).get('avg_quality', 0.0)
        elif metric_name == 'identity_consistency':
            actual_value = compute_identity_consistency(eval_data)
        else:
            actual_value = 0.0

        if actual_value < min_value:
            failures.append(f"{metric_name}: {actual_value:.3f} < {min_value:.3f}")

    passes = len(failures) == 0
    return passes, failures


# ============================================================================
# Checkpoint Selection
# ============================================================================

class CheckpointSelector:
    """Select best checkpoint from evaluation results."""

    def __init__(
        self,
        lora_type: str,
        character_name: str,
        min_checkpoints: int = 3
    ):
        """
        Initialize checkpoint selector.

        Args:
            lora_type: LoRA type (pose/action/expression)
            character_name: Character name
            min_checkpoints: Minimum number of valid checkpoints required
        """
        self.lora_type = lora_type
        self.character_name = character_name
        self.min_checkpoints = min_checkpoints

    def select_best(
        self,
        checkpoint_dir: Path,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Select the best checkpoint based on evaluation metrics.

        Args:
            checkpoint_dir: Directory containing checkpoints
            output_dir: Optional output directory for selection report

        Returns:
            Selection results with best checkpoint info
        """
        # Find checkpoints
        checkpoints = find_checkpoints(checkpoint_dir)

        if len(checkpoints) < self.min_checkpoints:
            logger.warning(f"Only {len(checkpoints)} checkpoints found (minimum {self.min_checkpoints})")

        # Load evaluation results
        eval_results = load_evaluation_results(checkpoint_dir, self.lora_type)

        if not eval_results:
            logger.error("No evaluation results found!")
            return {'error': 'no_evaluation_results'}

        # Score each checkpoint
        checkpoint_scores = []

        for checkpoint_path in checkpoints:
            ckpt_name = checkpoint_path.stem

            # Check if evaluation exists
            if ckpt_name not in eval_results:
                logger.warning(f"No evaluation for {ckpt_name}")
                continue

            eval_data = eval_results[ckpt_name]

            # Check minimum thresholds
            passes, failures = check_minimum_thresholds(eval_data, self.lora_type)

            if not passes:
                logger.info(f"{ckpt_name} failed thresholds: {failures}")
                continue

            # Compute scores
            scores = compute_final_score(eval_data, self.lora_type)

            checkpoint_scores.append({
                'checkpoint_path': checkpoint_path,
                'checkpoint_name': ckpt_name,
                'scores': scores,
                'eval_data': eval_data
            })

        # Sort by final score
        checkpoint_scores.sort(key=lambda x: x['scores']['final_score'], reverse=True)

        if not checkpoint_scores:
            logger.error("No checkpoints passed minimum thresholds!")
            return {'error': 'no_valid_checkpoints'}

        # Select best checkpoint
        best = checkpoint_scores[0]
        best_checkpoint = best['checkpoint_path']
        best_scores = best['scores']

        logger.info(f"Best checkpoint: {best['checkpoint_name']}")
        logger.info(f"Final score: {best_scores['final_score']:.3f}")

        # Prepare results
        results = {
            'character': self.character_name,
            'lora_type': self.lora_type,
            'best_checkpoint': str(best_checkpoint),
            'best_checkpoint_name': best['checkpoint_name'],
            'best_scores': best_scores,
            'total_checkpoints': len(checkpoints),
            'valid_checkpoints': len(checkpoint_scores),
            'all_checkpoint_scores': [
                {
                    'name': c['checkpoint_name'],
                    'scores': c['scores']
                }
                for c in checkpoint_scores
            ]
        }

        # Save results
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save selection report
            report_file = output_dir / f'{self.character_name}_{self.lora_type}_selection.json'
            with open(report_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved selection report to {report_file}")

            # Copy best checkpoint to production directory
            production_name = f'BEST_{self.character_name}_{self.lora_type}_lora.safetensors'
            production_path = output_dir / production_name

            shutil.copy2(best_checkpoint, production_path)
            logger.info(f"Copied best checkpoint to {production_path}")

        return results

    def generate_comparison_report(
        self,
        results: Dict[str, Any],
        output_dir: Path
    ) -> None:
        """
        Generate detailed comparison report.

        Args:
            results: Selection results
            output_dir: Output directory
        """
        report_lines = []
        report_lines.append("="*80)
        report_lines.append(f"CHECKPOINT SELECTION REPORT")
        report_lines.append("="*80)
        report_lines.append(f"Character: {results['character']}")
        report_lines.append(f"LoRA Type: {results['lora_type']}")
        report_lines.append(f"Total Checkpoints: {results['total_checkpoints']}")
        report_lines.append(f"Valid Checkpoints: {results['valid_checkpoints']}")
        report_lines.append("")
        report_lines.append("BEST CHECKPOINT:")
        report_lines.append(f"  Name: {results['best_checkpoint_name']}")
        report_lines.append(f"  Path: {results['best_checkpoint']}")
        report_lines.append("")
        report_lines.append("SCORES:")
        best_scores = results['best_scores']
        report_lines.append(f"  Control Precision:     {best_scores['control_precision']:.3f}")
        report_lines.append(f"  Identity Consistency:  {best_scores['identity_consistency']:.3f}")
        report_lines.append(f"  Generalization:        {best_scores['generalization']:.3f}")
        report_lines.append(f"  FINAL SCORE:           {best_scores['final_score']:.3f}")
        report_lines.append("")
        report_lines.append("ALL CHECKPOINTS (sorted by score):")
        report_lines.append("-"*80)
        report_lines.append(f"{'Rank':<6} {'Checkpoint':<30} {'Final':<8} {'Control':<8} {'Identity':<8} {'General':<8}")
        report_lines.append("-"*80)

        for rank, ckpt in enumerate(results['all_checkpoint_scores'], 1):
            scores = ckpt['scores']
            report_lines.append(
                f"{rank:<6} {ckpt['name']:<30} "
                f"{scores['final_score']:<8.3f} "
                f"{scores['control_precision']:<8.3f} "
                f"{scores['identity_consistency']:<8.3f} "
                f"{scores['generalization']:<8.3f}"
            )

        report_lines.append("="*80)

        # Save text report
        report_file = output_dir / f'{results["character"]}_{results["lora_type"]}_comparison.txt'
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))

        logger.info(f"Saved comparison report to {report_file}")

        # Print to console
        print('\n'.join(report_lines))


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Select best LoRA checkpoint based on evaluation metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--checkpoint-dir',
        type=Path,
        required=True,
        help='Directory containing checkpoints and evaluation results'
    )

    parser.add_argument(
        '--lora-type',
        type=str,
        required=True,
        choices=['pose', 'action', 'expression'],
        help='LoRA type'
    )

    parser.add_argument(
        '--character-name',
        type=str,
        required=True,
        help='Character name'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Output directory for selection results (default: checkpoint_dir/selection)'
    )

    parser.add_argument(
        '--min-checkpoints',
        type=int,
        default=3,
        help='Minimum number of valid checkpoints required (default: 3)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device for inference (default: cuda) - currently unused'
    )

    args = parser.parse_args()

    # Default output directory
    if args.output_dir is None:
        args.output_dir = args.checkpoint_dir / 'selection'

    # Initialize selector
    selector = CheckpointSelector(
        lora_type=args.lora_type,
        character_name=args.character_name,
        min_checkpoints=args.min_checkpoints
    )

    # Select best checkpoint
    results = selector.select_best(
        args.checkpoint_dir,
        args.output_dir
    )

    # Generate comparison report
    if 'error' not in results:
        selector.generate_comparison_report(results, args.output_dir)
    else:
        print(f"\n❌ Error: {results['error']}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
