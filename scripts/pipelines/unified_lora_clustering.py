#!/usr/bin/env python3
"""
Unified LoRA Data Preparation Clustering Pipeline

Integrates all SOTA clustering methods for LoRA training data:
- Character Identity: InsightFace ArcFace (face recognition)
- Scene: DINOv2 + k-means (fixed-k scene separation)
- Expression: HSEmotion (supervised emotion classification)
- Action/Pose: CLIP + HDBSCAN with checkpointing (visual clustering)

Usage:
    # Run all clustering types
    python scripts/pipelines/unified_lora_clustering.py \
        --project luca \
        --instances-dir /path/to/instances \
        --backgrounds-dir /path/to/backgrounds \
        --output-dir /path/to/output \
        --all

    # Run specific clustering types
    python scripts/pipelines/unified_lora_clustering.py \
        --project luca \
        --instances-dir /path/to/instances \
        --backgrounds-dir /path/to/backgrounds \
        --output-dir /path/to/output \
        --character --scene --expression

    # Resume from checkpoint
    python scripts/pipelines/unified_lora_clustering.py \
        --project luca \
        --instances-dir /path/to/instances \
        --output-dir /path/to/output \
        --action \
        --resume
"""

import argparse
import json
import logging
import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


def setup_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """Setup logger with console and file handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class UnifiedLoRAClustering:
    """Unified pipeline for LoRA data clustering."""

    def __init__(
        self,
        project: str,
        output_dir: str,
        device: str = "cpu",
        resume: bool = False
    ):
        """Initialize unified clustering pipeline.

        Args:
            project: Project/film name
            output_dir: Base output directory
            device: 'cpu' or 'cuda'
            resume: Resume from checkpoints if available
        """
        self.project = project
        self.output_dir = Path(output_dir)
        self.device = device
        self.resume = resume

        # Setup logging
        log_file = self.output_dir / "logs" / f"unified_clustering_{project}.log"
        self.logger = setup_logger(f"unified_clustering_{project}", str(log_file))

        # Clustering directories
        self.character_dir = self.output_dir / "character_clusters"
        self.scene_dir = self.output_dir / "scene_clusters"
        self.expression_dir = self.output_dir / "expression_clusters"
        self.action_dir = self.output_dir / "action_clusters"

        # Results tracking
        self.results = {
            'project': project,
            'timestamp': datetime.now().isoformat(),
            'device': device,
            'clustering_results': {}
        }

    def run_character_clustering(
        self,
        instances_dir: str,
        min_cluster_size: int = 10
    ) -> bool:
        """Run character identity clustering with InsightFace.

        Args:
            instances_dir: Directory with character instance images
            min_cluster_size: Minimum faces per identity

        Returns:
            True if successful
        """
        self.logger.info("="*80)
        self.logger.info("STEP 1: CHARACTER IDENTITY CLUSTERING (InsightFace ArcFace)")
        self.logger.info("="*80)

        cmd = [
            sys.executable,
            "scripts/generic/clustering/face_identity_clustering.py",
            str(instances_dir),
            "--output-dir", str(self.character_dir),
            "--min-cluster-size", str(min_cluster_size),
            "--device", self.device,
        ]

        self.logger.info(f"Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            self.logger.info("✅ Character clustering complete")

            # Load results
            metadata_path = self.character_dir / "identity_clustering.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    self.results['clustering_results']['character'] = json.load(f)

            return True

        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ Character clustering failed: {e}")
            self.logger.error(f"Stdout: {e.stdout}")
            self.logger.error(f"Stderr: {e.stderr}")
            return False

    def run_scene_clustering(
        self,
        backgrounds_dir: str,
        model: str = "dinov2-base",
        k_range: List[int] = [30, 150]
    ) -> bool:
        """Run scene clustering with DINOv2 + k-means.

        Args:
            backgrounds_dir: Directory with background images
            model: DINOv2 model variant
            k_range: K range for k-means [min, max]

        Returns:
            True if successful
        """
        self.logger.info("="*80)
        self.logger.info("STEP 2: SCENE CLUSTERING (DINOv2 + k-means)")
        self.logger.info("="*80)

        cmd = [
            sys.executable,
            "scripts/generic/clustering/scene_clustering.py",
            str(backgrounds_dir),
            "--output-dir", str(self.scene_dir),
            "--model", model,
            "--method", "kmeans",
            "--hierarchical",
            "--k-range", str(k_range[0]), str(k_range[1]),
            "--device", self.device,
        ]

        self.logger.info(f"Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            self.logger.info("✅ Scene clustering complete")

            # Load results
            metadata_path = self.scene_dir / "scene_clustering_report.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    self.results['clustering_results']['scene'] = json.load(f)

            return True

        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ Scene clustering failed: {e}")
            self.logger.error(f"Stdout: {e.stdout}")
            self.logger.error(f"Stderr: {e.stderr}")
            return False

    def run_expression_clustering(
        self,
        instances_dir: str,
        method: str = "hsemotion"
    ) -> bool:
        """Run expression clustering with HSEmotion.

        Args:
            instances_dir: Directory with character instance images
            method: Classification method (hsemotion recommended)

        Returns:
            True if successful
        """
        self.logger.info("="*80)
        self.logger.info("STEP 3: EXPRESSION CLUSTERING (HSEmotion)")
        self.logger.info("="*80)

        cmd = [
            sys.executable,
            "scripts/generic/face/expression_classification.py",
            str(instances_dir),
            "--output-dir", str(self.expression_dir),
            "--method", method,
            "--device", self.device,
        ]

        self.logger.info(f"Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            self.logger.info("✅ Expression clustering complete")

            # Load results
            metadata_path = self.expression_dir / "expression_classification.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    self.results['clustering_results']['expression'] = json.load(f)

            return True

        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ Expression clustering failed: {e}")
            self.logger.error(f"Stdout: {e.stdout}")
            self.logger.error(f"Stderr: {e.stderr}")
            return False

    def run_action_clustering(
        self,
        instances_dir: str,
        min_cluster_size: int = 15,
        checkpoint_interval: int = 1000
    ) -> bool:
        """Run action/pose clustering with CLIP + checkpointing.

        Args:
            instances_dir: Directory with character instance images
            min_cluster_size: Minimum instances per action cluster
            checkpoint_interval: Save checkpoint every N images

        Returns:
            True if successful
        """
        self.logger.info("="*80)
        self.logger.info("STEP 4: ACTION/POSE CLUSTERING (CLIP + HDBSCAN)")
        self.logger.info("="*80)

        cmd = [
            sys.executable,
            "scripts/generic/clustering/action_clustering.py",
            str(instances_dir),
            "--output-dir", str(self.action_dir),
            "--method", "visual",
            "--device", self.device,
            "--min-cluster-size", str(min_cluster_size),
            "--checkpoint-interval", str(checkpoint_interval),
        ]

        if self.resume:
            cmd.append("--resume")

        self.logger.info(f"Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            self.logger.info("✅ Action clustering complete")

            # Load results
            metadata_path = self.action_dir / "action_clustering.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    self.results['clustering_results']['action'] = json.load(f)

            return True

        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ Action clustering failed: {e}")
            self.logger.error(f"Stdout: {e.stdout}")
            self.logger.error(f"Stderr: {e.stderr}")
            return False

    def generate_report(self) -> str:
        """Generate comprehensive clustering report.

        Returns:
            Report path
        """
        self.logger.info("="*80)
        self.logger.info("GENERATING COMPREHENSIVE REPORT")
        self.logger.info("="*80)

        report_path = self.output_dir / f"unified_clustering_report_{self.project}.json"

        # Add summary
        summary = {
            'total_clustering_types': len(self.results['clustering_results']),
            'completed_types': list(self.results['clustering_results'].keys()),
            'output_directory': str(self.output_dir),
        }

        # Extract key metrics
        if 'character' in self.results['clustering_results']:
            char_data = self.results['clustering_results']['character']
            summary['character_identities'] = char_data.get('clustering_info', {}).get('n_identities', 0)

        if 'scene' in self.results['clustering_results']:
            scene_data = self.results['clustering_results']['scene']
            summary['scene_clusters'] = scene_data.get('n_clusters', 0)

        if 'expression' in self.results['clustering_results']:
            expr_data = self.results['clustering_results']['expression']
            if expr_data.get('method') == 'hsemotion':
                summary['expression_distribution'] = expr_data.get('emotion_distribution', {})
            else:
                summary['expression_clusters'] = expr_data.get('expression_clusters', 0)

        if 'action' in self.results['clustering_results']:
            action_data = self.results['clustering_results']['action']
            summary['action_clusters'] = action_data.get('n_action_clusters', 0)

        self.results['summary'] = summary

        # Save report
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        self.logger.info(f"📊 Report saved: {report_path}")

        # Print summary
        self.logger.info("\n" + "="*80)
        self.logger.info("CLUSTERING SUMMARY")
        self.logger.info("="*80)
        for key, value in summary.items():
            if isinstance(value, dict):
                self.logger.info(f"{key}:")
                for k, v in value.items():
                    self.logger.info(f"  {k}: {v}")
            else:
                self.logger.info(f"{key}: {value}")
        self.logger.info("="*80)

        return str(report_path)


def main():
    parser = argparse.ArgumentParser(
        description="Unified LoRA Data Preparation Clustering Pipeline"
    )

    # Required arguments
    parser.add_argument(
        "--project",
        required=True,
        help="Project/film name (e.g., luca, coco, elio)"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Base output directory for all clustering results"
    )

    # Input directories
    parser.add_argument(
        "--instances-dir",
        help="Directory with character instance images (required for character/expression/action)"
    )
    parser.add_argument(
        "--backgrounds-dir",
        help="Directory with background images (required for scene)"
    )

    # Clustering type selection
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all clustering types"
    )
    parser.add_argument(
        "--character",
        action="store_true",
        help="Run character identity clustering"
    )
    parser.add_argument(
        "--scene",
        action="store_true",
        help="Run scene clustering"
    )
    parser.add_argument(
        "--expression",
        action="store_true",
        help="Run expression clustering"
    )
    parser.add_argument(
        "--action",
        action="store_true",
        help="Run action/pose clustering"
    )

    # Common parameters
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for processing (default: cpu)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoints if available"
    )

    # Character clustering parameters
    parser.add_argument(
        "--min-identity-size",
        type=int,
        default=10,
        help="Minimum faces per identity (default: 10)"
    )

    # Scene clustering parameters
    parser.add_argument(
        "--scene-model",
        default="dinov2-base",
        choices=["dinov2-small", "dinov2-base", "dinov2-large", "dinov2-giant"],
        help="DINOv2 model for scene clustering (default: dinov2-base)"
    )
    parser.add_argument(
        "--scene-k-range",
        type=int,
        nargs=2,
        default=[30, 150],
        help="K range for scene k-means (default: 30 150)"
    )

    # Expression clustering parameters
    parser.add_argument(
        "--expression-method",
        default="hsemotion",
        choices=["hsemotion", "clip"],
        help="Expression classification method (default: hsemotion)"
    )

    # Action clustering parameters
    parser.add_argument(
        "--min-action-size",
        type=int,
        default=15,
        help="Minimum instances per action cluster (default: 15)"
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1000,
        help="Checkpoint interval for action clustering (default: 1000)"
    )

    args = parser.parse_args()

    # Determine which clustering types to run
    run_types = []
    if args.all:
        run_types = ['character', 'scene', 'expression', 'action']
    else:
        if args.character:
            run_types.append('character')
        if args.scene:
            run_types.append('scene')
        if args.expression:
            run_types.append('expression')
        if args.action:
            run_types.append('action')

    if not run_types:
        print("Error: No clustering types selected. Use --all or specify individual types.")
        return 1

    # Validate input directories
    if 'character' in run_types or 'expression' in run_types or 'action' in run_types:
        if not args.instances_dir:
            print("Error: --instances-dir required for character/expression/action clustering")
            return 1
        if not os.path.isdir(args.instances_dir):
            print(f"Error: Instances directory not found: {args.instances_dir}")
            return 1

    if 'scene' in run_types:
        if not args.backgrounds_dir:
            print("Error: --backgrounds-dir required for scene clustering")
            return 1
        if not os.path.isdir(args.backgrounds_dir):
            print(f"Error: Backgrounds directory not found: {args.backgrounds_dir}")
            return 1

    # Initialize pipeline
    pipeline = UnifiedLoRAClustering(
        project=args.project,
        output_dir=args.output_dir,
        device=args.device,
        resume=args.resume
    )

    pipeline.logger.info(f"🚀 Starting unified LoRA clustering pipeline for project: {args.project}")
    pipeline.logger.info(f"   Clustering types: {', '.join(run_types)}")
    pipeline.logger.info(f"   Device: {args.device}")
    pipeline.logger.info(f"   Output: {args.output_dir}")

    # Run clustering types
    success_count = 0
    failed_types = []

    if 'character' in run_types:
        if pipeline.run_character_clustering(
            args.instances_dir,
            min_cluster_size=args.min_identity_size
        ):
            success_count += 1
        else:
            failed_types.append('character')

    if 'scene' in run_types:
        if pipeline.run_scene_clustering(
            args.backgrounds_dir,
            model=args.scene_model,
            k_range=args.scene_k_range
        ):
            success_count += 1
        else:
            failed_types.append('scene')

    if 'expression' in run_types:
        if pipeline.run_expression_clustering(
            args.instances_dir,
            method=args.expression_method
        ):
            success_count += 1
        else:
            failed_types.append('expression')

    if 'action' in run_types:
        if pipeline.run_action_clustering(
            args.instances_dir,
            min_cluster_size=args.min_action_size,
            checkpoint_interval=args.checkpoint_interval
        ):
            success_count += 1
        else:
            failed_types.append('action')

    # Generate report
    report_path = pipeline.generate_report()

    # Final summary
    pipeline.logger.info("\n" + "="*80)
    pipeline.logger.info("PIPELINE COMPLETE")
    pipeline.logger.info("="*80)
    pipeline.logger.info(f"Successful: {success_count}/{len(run_types)}")
    if failed_types:
        pipeline.logger.info(f"Failed: {', '.join(failed_types)}")
    pipeline.logger.info(f"Report: {report_path}")
    pipeline.logger.info("="*80)

    return 0 if success_count == len(run_types) else 1


if __name__ == "__main__":
    sys.exit(main())
