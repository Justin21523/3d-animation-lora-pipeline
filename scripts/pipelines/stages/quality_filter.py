"""
Stage 2: Quality Filter

Wrapper stage that uses generic quality filtering tools.
This stage integrates the standalone quality filtering tools into the pipeline.
"""

import sys
from pathlib import Path
from typing import Dict, Any
import subprocess
import json

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from scripts.pipelines.stages.base_stage import BaseStage


class QualityFilterStage(BaseStage):
    """Stage 2: Filter images by quality metrics using generic tools"""

    def validate_config(self) -> bool:
        """Validate configuration"""
        required_keys = ['input_dir', 'output_dir']

        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

        input_dir = Path(self.config['input_dir'])
        if not input_dir.exists():
            raise ValueError(f"Input directory not found: {input_dir}")

        return True

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute quality filter stage using generic training_quality_filter.py"""

        self.logger.info("="*60)
        self.logger.info("STAGE 2: Quality Filter")
        self.logger.info("="*60)

        # Setup paths
        input_dir = Path(self.config['input_dir'])
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build command to call generic training quality filter
        script_path = project_root / "scripts/generic/training/training_quality_filter.py"

        cmd = [
            "python", str(script_path),
            "--input-dir", str(input_dir),
            "--output-dir", str(output_dir),
            "--target-per-cluster", str(self.config.get('target_per_cluster', 200)),
            "--min-sharpness", str(self.config.get('sharpness_min', 100)),
            "--min-completeness", str(self.config.get('completeness_min', 0.85)),
            "--diversity-method", self.config.get('diversity_method', 'clip'),
            "--device", self.config.get('device', 'cuda')
        ]

        if self.config.get('use_face_detection', False):
            cmd.append("--use-face-detection")

        self.logger.info(f"Executing: {' '.join(cmd)}")

        # Execute the generic tool
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            self.logger.error(f"Quality filtering failed: {result.stderr}")
            raise RuntimeError(f"Quality filtering failed: {result.stderr}")

        self.logger.info(result.stdout)

        # Load results from quality_filter_report.json
        report_path = output_dir / "quality_filter_report.json"

        if report_path.exists():
            with open(report_path, 'r') as f:
                metadata = json.load(f)
        else:
            # Fallback metadata if report doesn't exist
            selected_images = list(output_dir.glob("*.png")) + list(output_dir.glob("*.jpg"))
            metadata = {
                'output_dir': str(output_dir),
                'num_selected': len(selected_images)
            }

        self.logger.info(f"\n✅ Quality filtering complete!")
        self.logger.info(f"   Output directory: {output_dir}")
        if 'num_selected' in metadata:
            self.logger.info(f"   Selected images: {metadata['num_selected']}")

        return {
            'output_dir': output_dir,
            'metadata': metadata,
            'num_selected': metadata.get('num_selected', 0)
        }
