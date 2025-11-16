#!/usr/bin/env python3
"""
Intelligent Frame Processor - AI-driven strategy selection and execution

Analyzes each frame and applies optimal processing strategy:
- keep_full: Keep complete frame with background
- segment: Segment character and inpaint background
- create_occlusion: Generate synthetic occlusions
- enhance_segment: Enhance quality then segment

Integrates:
- Frame Decision Engine (analysis & strategy selection)
- SAM2 (instance segmentation)
- LaMa/PowerPaint (inpainting)
- RealESRGAN/CodeFormer (enhancement)
- Qwen2-VL (caption generation)
"""

import argparse
import cv2
import json
import numpy as np
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from tqdm import tqdm
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.frame_decision_engine import FrameDecisionEngine, FrameAnalysis
from scripts.core.models.model_loader import get_model_loader


@dataclass
class ProcessingResult:
    """Result from processing a single frame"""
    frame_path: Path
    strategy: str
    confidence: float
    reasoning: str
    outputs: List[Path]  # Generated output files
    metadata: Dict
    success: bool
    error: Optional[str] = None


class StrategyExecutor:
    """
    Executes the 4 processing strategies
    Integrates existing pipeline tools
    """

    def __init__(self, config: Dict, output_root: Path, device: str = "cuda"):
        """
        Initialize strategy executor

        Args:
            config: Strategy configuration from YAML
            output_root: Root output directory
            device: GPU device
        """
        self.config = config
        self.output_root = output_root
        self.device = device

        # Create strategy output directories
        self.strategy_dirs = {
            'keep_full': output_root / 'keep_full',
            'segment': output_root / 'segment',
            'create_occlusion': output_root / 'create_occlusion',
            'enhance_segment': output_root / 'enhance_segment'
        }

        for dir_path in self.strategy_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
            (dir_path / 'images').mkdir(exist_ok=True)
            (dir_path / 'captions').mkdir(exist_ok=True)

        # Initialize unified model loader (lazy loading)
        self.model_loader = get_model_loader(device=device)

        print(f"✓ Strategy Executor initialized")
        print(f"   Output: {output_root}")

    def execute_keep_full(
        self,
        frame_path: Path,
        analysis: FrameAnalysis,
        output_name: str
    ) -> ProcessingResult:
        """
        Strategy A: Keep complete frame with background

        Steps:
        1. Copy original frame
        2. Generate caption with VLM
        3. Save metadata

        Args:
            frame_path: Input frame path
            analysis: Frame analysis results
            output_name: Base name for outputs

        Returns:
            ProcessingResult with outputs
        """
        try:
            strategy_dir = self.strategy_dirs['keep_full']
            outputs = []

            # 1. Copy frame
            img_out = strategy_dir / 'images' / f"{output_name}.png"
            shutil.copy2(frame_path, img_out)
            outputs.append(img_out)

            # 2. Generate caption (if enabled)
            caption = ""
            if self.config['strategies']['keep_full']['generate_caption']:
                caption_prefix = self.config['strategies']['keep_full'].get(
                    'caption_prefix',
                    'a 3d animated character, pixar style'
                )

                # TODO: Integrate VLM caption generation
                # For now, use prefix + basic analysis
                caption = self._generate_basic_caption(
                    frame_path, analysis, caption_prefix
                )

                # Save caption
                caption_out = strategy_dir / 'captions' / f"{output_name}.txt"
                caption_out.write_text(caption, encoding='utf-8')
                outputs.append(caption_out)

            # 3. Save metadata
            metadata = {
                'strategy': 'keep_full',
                'original_frame': str(frame_path),
                'analysis': asdict(analysis),
                'caption': caption,
                'timestamp': datetime.now().isoformat()
            }

            return ProcessingResult(
                frame_path=frame_path,
                strategy='keep_full',
                confidence=1.0,
                reasoning="Kept full frame with background intact",
                outputs=outputs,
                metadata=metadata,
                success=True
            )

        except Exception as e:
            return ProcessingResult(
                frame_path=frame_path,
                strategy='keep_full',
                confidence=0.0,
                reasoning="",
                outputs=[],
                metadata={},
                success=False,
                error=str(e)
            )

    def execute_segment(
        self,
        frame_path: Path,
        analysis: FrameAnalysis,
        output_name: str
    ) -> ProcessingResult:
        """
        Strategy B: Segment character and inpaint background

        Steps:
        1. Run SAM2 instance segmentation
        2. Identify Luca instance (largest person or face detection)
        3. Inpaint background with LaMa
        4. Save character, background, composite variants
        5. Generate captions for each variant

        Args:
            frame_path: Input frame path
            analysis: Frame analysis results
            output_name: Base name for outputs

        Returns:
            ProcessingResult with multiple outputs
        """
        try:
            strategy_dir = self.strategy_dirs['segment']
            outputs = []

            # Load image
            img = cv2.imread(str(frame_path))
            if img is None:
                raise ValueError(f"Could not read image: {frame_path}")

            # 1. SAM2 Segmentation
            cfg = self.config['strategies']['segment']
            character_mask = self._segment_character(img, cfg)

            # 2. Extract character & inpaint background
            character_img = self._extract_character(img, character_mask)
            background_img = self._inpaint_background(img, character_mask)

            # 3. Save outputs based on config
            cfg = self.config['strategies']['segment']

            if cfg.get('save_character', True):
                char_out = strategy_dir / 'images' / f"{output_name}_character.png"
                cv2.imwrite(str(char_out), character_img)
                outputs.append(char_out)

                # Generate character caption
                if cfg.get('generate_captions', True):
                    suffix = cfg.get('character_caption_suffix', ', isolated character')
                    caption = self._generate_basic_caption(
                        frame_path, analysis,
                        "a 3d animated character" + suffix
                    )
                    caption_out = strategy_dir / 'captions' / f"{output_name}_character.txt"
                    caption_out.write_text(caption, encoding='utf-8')
                    outputs.append(caption_out)

            if cfg.get('save_background', True):
                bg_out = strategy_dir / 'images' / f"{output_name}_background.png"
                cv2.imwrite(str(bg_out), background_img)
                outputs.append(bg_out)

                # Generate background caption
                if cfg.get('generate_captions', True):
                    suffix = cfg.get('background_caption_suffix', ', environment only')
                    caption = self._generate_basic_caption(
                        frame_path, analysis,
                        "3d animated scene" + suffix
                    )
                    caption_out = strategy_dir / 'captions' / f"{output_name}_background.txt"
                    caption_out.write_text(caption, encoding='utf-8')
                    outputs.append(caption_out)

            if cfg.get('save_composite', True):
                # Save original for comparison
                comp_out = strategy_dir / 'images' / f"{output_name}_composite.png"
                shutil.copy2(frame_path, comp_out)
                outputs.append(comp_out)

            # 4. Metadata
            metadata = {
                'strategy': 'segment',
                'original_frame': str(frame_path),
                'analysis': asdict(analysis),
                'segmentation_model': cfg.get('segmentation_model', 'sam2_hiera_large'),
                'inpainting_model': cfg.get('inpainting_model', 'lama'),
                'timestamp': datetime.now().isoformat()
            }

            return ProcessingResult(
                frame_path=frame_path,
                strategy='segment',
                confidence=0.9,
                reasoning="Segmented and inpainted successfully",
                outputs=outputs,
                metadata=metadata,
                success=True
            )

        except Exception as e:
            return ProcessingResult(
                frame_path=frame_path,
                strategy='segment',
                confidence=0.0,
                reasoning="",
                outputs=[],
                metadata={},
                success=False,
                error=str(e)
            )

    def execute_create_occlusion(
        self,
        frame_path: Path,
        analysis: FrameAnalysis,
        output_name: str
    ) -> ProcessingResult:
        """
        Strategy C: Create synthetic occlusions for data augmentation

        Steps:
        1. Generate N variations with different occlusion types
        2. Apply edge blur / semi-transparent overlays
        3. Adjust captions to include occlusion descriptions

        Args:
            frame_path: Input frame path
            analysis: Frame analysis results
            output_name: Base name for outputs

        Returns:
            ProcessingResult with multiple occlusion variants
        """
        try:
            strategy_dir = self.strategy_dirs['create_occlusion']
            outputs = []
            cfg = self.config['strategies']['create_occlusion']

            # Load image
            img = cv2.imread(str(frame_path))
            if img is None:
                raise ValueError(f"Could not read image: {frame_path}")

            # Generate variations
            variations_per_image = cfg.get('variations_per_image', 3)
            occlusion_types = cfg.get('occlusion_types', ['edge_blur'])
            occlusion_descs = cfg.get('occlusion_descriptions', ['partially obscured'])

            for i in range(variations_per_image):
                # Rotate through occlusion types
                occ_type = occlusion_types[i % len(occlusion_types)]
                occ_desc = occlusion_descs[i % len(occlusion_descs)]

                # Apply occlusion
                if occ_type == 'edge_blur':
                    occluded_img = self._apply_edge_blur(
                        img,
                        cfg.get('blur_radius', 15),
                        cfg.get('blur_positions', ['left', 'right'])
                    )
                elif occ_type == 'overlay':
                    occluded_img = self._apply_overlay(
                        img,
                        cfg.get('overlay_opacity', 0.3)
                    )
                else:
                    occluded_img = img.copy()

                # Save image
                img_out = strategy_dir / 'images' / f"{output_name}_occ{i}.png"
                cv2.imwrite(str(img_out), occluded_img)
                outputs.append(img_out)

                # Generate caption with occlusion tag
                if cfg.get('add_occlusion_tags', True):
                    caption = self._generate_basic_caption(
                        frame_path, analysis,
                        f"a 3d animated character, {occ_desc}"
                    )
                    caption_out = strategy_dir / 'captions' / f"{output_name}_occ{i}.txt"
                    caption_out.write_text(caption, encoding='utf-8')
                    outputs.append(caption_out)

            # Metadata
            metadata = {
                'strategy': 'create_occlusion',
                'original_frame': str(frame_path),
                'analysis': asdict(analysis),
                'variations_created': variations_per_image,
                'occlusion_types': occlusion_types,
                'timestamp': datetime.now().isoformat()
            }

            return ProcessingResult(
                frame_path=frame_path,
                strategy='create_occlusion',
                confidence=0.8,
                reasoning=f"Created {variations_per_image} occlusion variants",
                outputs=outputs,
                metadata=metadata,
                success=True
            )

        except Exception as e:
            return ProcessingResult(
                frame_path=frame_path,
                strategy='create_occlusion',
                confidence=0.0,
                reasoning="",
                outputs=[],
                metadata={},
                success=False,
                error=str(e)
            )

    def execute_enhance_segment(
        self,
        frame_path: Path,
        analysis: FrameAnalysis,
        output_name: str
    ) -> ProcessingResult:
        """
        Strategy D: Enhance quality then segment

        Steps:
        1. Apply enhancement pipeline (upscale, denoise, face enhance)
        2. Then run segmentation (reuse execute_segment)

        Args:
            frame_path: Input frame path
            analysis: Frame analysis results
            output_name: Base name for outputs

        Returns:
            ProcessingResult with enhanced + segmented outputs
        """
        try:
            strategy_dir = self.strategy_dirs['enhance_segment']
            outputs = []
            cfg = self.config['strategies']['enhance_segment']

            # Load image
            img = cv2.imread(str(frame_path))
            if img is None:
                raise ValueError(f"Could not read image: {frame_path}")

            # 1. Enhancement pipeline
            enhanced_img = self._enhance_image(img, cfg)

            # Save enhanced frame temporarily
            temp_enhanced = strategy_dir / f"temp_{output_name}_enhanced.png"
            cv2.imwrite(str(temp_enhanced), enhanced_img)

            # 2. Apply segmentation on enhanced image
            # Update analysis for enhanced image
            enhanced_analysis = analysis  # TODO: re-analyze enhanced image

            # Run segmentation (reuse code from execute_segment)
            seg_cfg = self.config['strategies']['segment']
            character_mask = self._segment_character(enhanced_img, seg_cfg)
            character_img = self._extract_character(enhanced_img, character_mask)

            # Save enhanced character
            char_out = strategy_dir / 'images' / f"{output_name}_enhanced_character.png"
            cv2.imwrite(str(char_out), character_img)
            outputs.append(char_out)

            # Generate caption
            if cfg.get('apply_segmentation', True):
                caption = self._generate_basic_caption(
                    frame_path, analysis,
                    "a 3d animated character, high quality, enhanced"
                )
                caption_out = strategy_dir / 'captions' / f"{output_name}_enhanced_character.txt"
                caption_out.write_text(caption, encoding='utf-8')
                outputs.append(caption_out)

            # Clean up temp file
            temp_enhanced.unlink()

            # Metadata
            metadata = {
                'strategy': 'enhance_segment',
                'original_frame': str(frame_path),
                'analysis': asdict(analysis),
                'enhancement_steps': cfg.get('enhancement_steps', []),
                'timestamp': datetime.now().isoformat()
            }

            return ProcessingResult(
                frame_path=frame_path,
                strategy='enhance_segment',
                confidence=0.85,
                reasoning="Enhanced then segmented successfully",
                outputs=outputs,
                metadata=metadata,
                success=True
            )

        except Exception as e:
            return ProcessingResult(
                frame_path=frame_path,
                strategy='enhance_segment',
                confidence=0.0,
                reasoning="",
                outputs=[],
                metadata={},
                success=False,
                error=str(e)
            )

    # ==================== Helper Methods ====================

    def _generate_basic_caption(
        self,
        frame_path: Path,
        analysis: FrameAnalysis,
        prefix: str
    ) -> str:
        """Generate basic caption based on analysis (placeholder for VLM)"""
        # TODO: Replace with actual VLM caption generation

        # Add quality descriptors
        descriptors = []
        if analysis.lighting_quality > 0.7:
            descriptors.append("well-lit")
        if analysis.sharpness > 0.6:
            descriptors.append("sharp focus")
        if analysis.complexity < 0.3:
            descriptors.append("simple background")
        else:
            descriptors.append("detailed environment")

        caption = prefix
        if descriptors:
            caption += ", " + ", ".join(descriptors)

        return caption

    def _segment_character(self, img: np.ndarray, config: Optional[Dict] = None) -> np.ndarray:
        """
        Segment character using SAM2 (or fallback)

        Args:
            img: Input image (BGR)
            config: Optional SAM2 config

        Returns:
            Binary mask (255 = character, 0 = background)
        """
        # Try SAM2 first
        sam2_generator = self.model_loader.get_sam2_model(config=config)

        if sam2_generator is not None:
            try:
                # Convert BGR to RGB for SAM2
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Generate masks
                masks = sam2_generator.generate(img_rgb)

                if len(masks) > 0:
                    # Get largest mask (likely main character)
                    largest_mask = max(masks, key=lambda x: x['area'])
                    mask = (largest_mask['segmentation'] * 255).astype(np.uint8)
                    return mask

            except Exception as e:
                print(f"⚠️ SAM2 segmentation failed: {e}, using fallback")

        # Fallback: simple center-focused mask
        h, w = img.shape[:2]
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        mask = ((y - center_y)**2 + (x - center_x)**2) < (min(h, w) // 3)**2

        return mask.astype(np.uint8) * 255

    def _extract_character(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Extract character using mask"""
        # Create RGBA image
        character = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        character[:, :, 3] = mask

        return character

    def _inpaint_background(self, img: np.ndarray, mask: np.ndarray, config: Optional[Dict] = None) -> np.ndarray:
        """
        Inpaint background using LaMa (or OpenCV fallback)

        Args:
            img: Input image (BGR)
            mask: Binary mask (255 = inpaint, 0 = keep)
            config: Optional inpainting config

        Returns:
            Inpainted image
        """
        # Use model loader's inpainting method (handles LaMa or fallback)
        return self.model_loader.inpaint_with_lama(img, mask, config)

    def _apply_edge_blur(
        self,
        img: np.ndarray,
        blur_radius: int,
        positions: List[str]
    ) -> np.ndarray:
        """Apply edge blur to simulate occlusion"""
        result = img.copy()
        h, w = img.shape[:2]
        edge_width = w // 4

        for pos in positions:
            if pos == 'left':
                region = result[:, :edge_width]
                blurred = cv2.GaussianBlur(region, (blur_radius*2+1, blur_radius*2+1), 0)
                result[:, :edge_width] = blurred
            elif pos == 'right':
                region = result[:, -edge_width:]
                blurred = cv2.GaussianBlur(region, (blur_radius*2+1, blur_radius*2+1), 0)
                result[:, -edge_width:] = blurred
            elif pos == 'bottom':
                edge_height = h // 4
                region = result[-edge_height:, :]
                blurred = cv2.GaussianBlur(region, (blur_radius*2+1, blur_radius*2+1), 0)
                result[-edge_height:, :] = blurred

        return result

    def _apply_overlay(self, img: np.ndarray, opacity: float) -> np.ndarray:
        """Apply semi-transparent overlay to simulate occlusion"""
        overlay = np.zeros_like(img)
        h, w = img.shape[:2]

        # Create random overlay pattern
        cv2.rectangle(overlay, (0, 0), (w//3, h), (50, 50, 50), -1)

        # Blend with original
        result = cv2.addWeighted(img, 1.0, overlay, opacity, 0)

        return result

    def _enhance_image(self, img: np.ndarray, config: Dict) -> np.ndarray:
        """
        Enhance image using RealESRGAN (or fallback)

        Args:
            img: Input image (BGR)
            config: Enhancement configuration

        Returns:
            Enhanced image
        """
        enhancement_steps = config.get('enhancement_steps', [])

        enhanced = img.copy()

        # Apply enhancement steps
        for step in enhancement_steps:
            if step == 'upscale':
                # RealESRGAN upscaling
                enhanced = self.model_loader.enhance_with_realesrgan(
                    enhanced,
                    config={'upscale': config.get('upscale', 2)}
                )
            elif step == 'denoise':
                # Simple denoising (OpenCV)
                enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
            elif step == 'sharpen':
                # Unsharp mask sharpening
                kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
                enhanced = cv2.filter2D(enhanced, -1, kernel)
            elif step == 'face_enhance':
                # CodeFormer would go here (not implemented yet)
                # For now, skip this step
                pass

        return enhanced


class IntelligentFrameProcessor:
    """
    Main orchestrator for intelligent frame processing
    Analyzes frames and applies optimal strategies
    """

    def __init__(
        self,
        decision_config_path: Path,
        strategy_config_path: Path,
        output_dir: Path,
        device: str = "cuda"
    ):
        """
        Initialize intelligent processor

        Args:
            decision_config_path: Path to decision thresholds YAML
            strategy_config_path: Path to strategy configs YAML
            output_dir: Root output directory
            device: GPU device
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

        # Load decision engine
        self.decision_engine = FrameDecisionEngine(decision_config_path)

        # Load strategy config
        with open(strategy_config_path, 'r') as f:
            self.strategy_config = yaml.safe_load(f)

        # Initialize strategy executor
        self.executor = StrategyExecutor(
            self.strategy_config,
            self.output_dir,
            device
        )

        # Statistics
        self.stats = {
            'total_processed': 0,
            'strategies': {
                'keep_full': 0,
                'segment': 0,
                'create_occlusion': 0,
                'enhance_segment': 0
            },
            'errors': 0
        }

    def process_frame(
        self,
        frame_path: Path,
        dataset_needs: Optional[Dict[str, float]] = None
    ) -> ProcessingResult:
        """
        Process a single frame with intelligent strategy selection

        Args:
            frame_path: Path to input frame
            dataset_needs: Optional dataset augmentation needs

        Returns:
            ProcessingResult
        """
        # 1. Analyze frame
        analysis = self.decision_engine.analyze_frame(frame_path)

        # 2. Decide strategy
        strategy, confidence, reasoning = self.decision_engine.decide_strategy(
            analysis, dataset_needs
        )

        # 3. Execute strategy
        output_name = frame_path.stem

        if strategy == 'keep_full':
            result = self.executor.execute_keep_full(frame_path, analysis, output_name)
        elif strategy == 'segment':
            result = self.executor.execute_segment(frame_path, analysis, output_name)
        elif strategy == 'create_occlusion':
            result = self.executor.execute_create_occlusion(frame_path, analysis, output_name)
        elif strategy == 'enhance_segment':
            result = self.executor.execute_enhance_segment(frame_path, analysis, output_name)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # 4. Update statistics
        self.stats['total_processed'] += 1
        if result.success:
            self.stats['strategies'][strategy] += 1
        else:
            self.stats['errors'] += 1

        return result

    def process_batch(
        self,
        frame_paths: List[Path],
        dataset_needs: Optional[Dict[str, float]] = None,
        save_report: bool = True
    ) -> Dict:
        """
        Process multiple frames in batch

        Args:
            frame_paths: List of frame paths
            dataset_needs: Optional dataset augmentation needs
            save_report: Whether to save processing report

        Returns:
            Processing report dictionary
        """
        results = []

        print(f"\n🚀 Processing {len(frame_paths)} frames with intelligent strategy selection...\n")

        # Process each frame with progress bar
        for frame_path in tqdm(frame_paths, desc="Processing frames"):
            result = self.process_frame(frame_path, dataset_needs)
            results.append(result)

        # Generate report
        report = self._generate_report(results)

        # Save report
        if save_report:
            report_path = self.output_dir / 'processing_report.json'
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"\n📊 Report saved: {report_path}")

        # Print summary
        self._print_summary(report)

        return report

    def _generate_report(self, results: List[ProcessingResult]) -> Dict:
        """Generate processing report"""
        report = {
            'summary': {
                'total_frames': len(results),
                'successful': sum(1 for r in results if r.success),
                'failed': sum(1 for r in results if not r.success),
                'strategies': dict(self.stats['strategies'])
            },
            'results': []
        }

        for result in results:
            report['results'].append({
                'frame': str(result.frame_path),
                'strategy': result.strategy,
                'confidence': result.confidence,
                'reasoning': result.reasoning,
                'outputs': [str(p) for p in result.outputs],
                'success': result.success,
                'error': result.error
            })

        return report

    def _print_summary(self, report: Dict):
        """Print processing summary"""
        summary = report['summary']

        print("\n" + "="*60)
        print("  📊 INTELLIGENT PROCESSING SUMMARY")
        print("="*60)
        print(f"\n✅ Total Processed: {summary['total_frames']}")
        print(f"   Successful: {summary['successful']}")
        print(f"   Failed: {summary['failed']}")
        print(f"\n📈 Strategy Distribution:")
        for strategy, count in summary['strategies'].items():
            percentage = (count / summary['total_frames'] * 100) if summary['total_frames'] > 0 else 0
            bar = "█" * int(percentage / 5)
            print(f"   {strategy:18s}: {count:4d} ({percentage:5.1f}%) {bar}")
        print("\n" + "="*60 + "\n")


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Intelligent Frame Processor - AI-driven strategy selection"
    )

    parser.add_argument(
        'input_dir',
        type=Path,
        help='Directory containing input frames'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Output directory for processed frames'
    )
    parser.add_argument(
        '--decision-config',
        type=Path,
        default=PROJECT_ROOT / 'configs/stages/intelligent_processing/decision_thresholds.yaml',
        help='Path to decision thresholds config'
    )
    parser.add_argument(
        '--strategy-config',
        type=Path,
        default=PROJECT_ROOT / 'configs/stages/intelligent_processing/strategy_configs.yaml',
        help='Path to strategy config'
    )
    parser.add_argument(
        '--dataset-needs',
        type=Path,
        help='JSON file with dataset augmentation needs (optional)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of frames to process (for testing)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Processing device'
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.input_dir.exists():
        print(f"❌ Input directory not found: {args.input_dir}")
        sys.exit(1)

    if not args.decision_config.exists():
        print(f"❌ Decision config not found: {args.decision_config}")
        sys.exit(1)

    if not args.strategy_config.exists():
        print(f"❌ Strategy config not found: {args.strategy_config}")
        sys.exit(1)

    # Load dataset needs if provided
    dataset_needs = None
    if args.dataset_needs and args.dataset_needs.exists():
        with open(args.dataset_needs, 'r') as f:
            dataset_needs = json.load(f)
        print(f"📊 Loaded dataset needs from {args.dataset_needs}")

    # Find all frames
    frame_extensions = {'.png', '.jpg', '.jpeg'}
    frame_paths = [
        p for p in args.input_dir.iterdir()
        if p.suffix.lower() in frame_extensions
    ]

    if not frame_paths:
        print(f"❌ No frames found in {args.input_dir}")
        sys.exit(1)

    # Apply limit if specified
    if args.limit:
        frame_paths = frame_paths[:args.limit]
        print(f"⚠️  Processing limited to {args.limit} frames")

    print(f"📁 Found {len(frame_paths)} frames")

    # Initialize processor
    processor = IntelligentFrameProcessor(
        decision_config_path=args.decision_config,
        strategy_config_path=args.strategy_config,
        output_dir=args.output_dir,
        device=args.device
    )

    # Process batch
    report = processor.process_batch(frame_paths, dataset_needs)

    print(f"\n✅ Processing complete!")
    print(f"📁 Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
