#!/usr/bin/env python3
"""
Batch Orchestration Layer for Synthetic Data Generation Pipeline

Coordinates the complete pipeline:
1. Vocabulary Generation (Module 1)
2. Image Generation (Module 2)
3. Quality Filtering (Module 3)
4. Dataset Organization (Module 5 - future)
5. Training Integration (Module 6 - future)

Features:
- Stage-based execution with skip capability
- Multi-job batch processing
- Unified progress tracking and reporting
- Checkpoint/resume at orchestrator level
- Comprehensive error handling and recovery

Author: LLMProvider Tooling
Date: 2025-11-30
"""

import json
import sys
import traceback
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import our modules
from scripts.generic.training.vocabulary_generator import VocabularyGenerator
from scripts.generic.training.batch_image_generator import BatchImageGenerator, GenerationConfig
from scripts.generic.quality.image_quality_filter import ImageQualityFilter, FilterConfig
from scripts.generic.training.dataset_organizer import DatasetOrganizer, DatasetConfig
from scripts.generic.training.training_config_generator import TrainingConfigGenerator, TrainingConfig
from scripts.generic.training.training_launcher import TrainingLauncher, TrainingStatus
from scripts.core.utils.checkpoint_manager import IndexCheckpointManager


# ============================================================================
# Enums and Data Classes
# ============================================================================

class PipelineStage(Enum):
    """Pipeline execution stages"""
    VOCABULARY_GENERATION = "vocabulary_generation"
    IMAGE_GENERATION = "image_generation"
    QUALITY_FILTERING = "quality_filtering"
    DATASET_ORGANIZATION = "dataset_organization"  # Module 5
    TRAINING_INTEGRATION = "training_integration"  # Module 6


@dataclass
class JobConfig:
    """Configuration for a single batch job"""
    job_id: str
    job_name: str

    # Vocabulary config
    character_name: str
    character_description: str
    num_prompts: int
    vocabulary_config: Dict[str, Any]

    # Generation config
    base_model_path: str
    lora_paths: Optional[List[str]]
    num_images_per_prompt: int
    generation_config: Dict[str, Any]

    # Filtering config
    filtering_config: Dict[str, Any]

    # Execution control
    stages_to_run: List[str]  # List of stage names

    # Stage-specific configurations (optional dict for future modules)
    stage_configs: Optional[Dict[str, Dict[str, Any]]] = None

    # Optional execution control
    skip_existing: bool = True

    # Resources
    device: str = "cuda"
    batch_size: int = 4

    def __post_init__(self):
        """Initialize optional fields"""
        if self.stage_configs is None:
            self.stage_configs = {}


@dataclass
class StageResult:
    """Result from executing a pipeline stage"""
    stage: str
    status: str  # 'completed', 'failed', 'skipped'
    start_time: str
    end_time: str
    duration_seconds: float
    output_dir: str
    metrics: Dict[str, Any]
    error: Optional[str] = None


@dataclass
class JobResult:
    """Result from executing a complete job"""
    job_id: str
    job_name: str
    status: str  # 'completed', 'partial', 'failed'
    start_time: str
    end_time: str
    duration_seconds: float
    stage_results: List[Dict[str, Any]]  # List of StageResult dicts
    final_output_dir: str
    summary_metrics: Dict[str, Any]


# ============================================================================
# Main Orchestrator
# ============================================================================

class BatchOrchestrator:
    """
    Main orchestrator for batch synthetic data generation

    Coordinates execution of multiple jobs through the complete pipeline
    """

    def __init__(
        self,
        workspace_dir: Path,
        log_dir: Path,
        checkpoint_dir: Path
    ):
        """
        Initialize batch orchestrator

        Args:
            workspace_dir: Root workspace for all jobs
            log_dir: Directory for logs
            checkpoint_dir: Directory for checkpoints
        """
        self.workspace_dir = Path(workspace_dir)
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)

        # Create directories
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize checkpoint manager
        self.checkpoint_mgr = IndexCheckpointManager(
            self.checkpoint_dir,
            filename="batch_orchestrator_checkpoint.json"
        )

        print("\n" + "="*80)
        print("🎯 BATCH ORCHESTRATOR - Initialized")
        print("="*80)
        print(f"  Workspace:   {self.workspace_dir}")
        print(f"  Logs:        {self.log_dir}")
        print(f"  Checkpoints: {self.checkpoint_dir}")
        print("="*80 + "\n")

    def execute_batch(
        self,
        job_configs: List[JobConfig],
        resume: bool = True
    ) -> List[JobResult]:
        """
        Execute multiple jobs sequentially

        Args:
            job_configs: List of job configurations
            resume: If True, resume from checkpoint

        Returns:
            List of JobResult objects
        """
        print(f"\n{'='*80}")
        print(f"🚀 BATCH EXECUTION - Starting {len(job_configs)} jobs")
        print(f"{'='*80}\n")

        # Check for checkpoint
        start_idx = 0
        completed_jobs = []

        if resume:
            checkpoint = self.checkpoint_mgr.load()
            if checkpoint:
                start_idx = checkpoint['last_completed_index'] + 1
                completed_jobs = checkpoint.get('completed_jobs', [])
                print(f"📂 Resuming from checkpoint: job {start_idx}/{len(job_configs)}\n")

        # Execute jobs
        all_results = []

        for idx in range(start_idx, len(job_configs)):
            job_config = job_configs[idx]

            print(f"\n{'='*80}")
            print(f"📦 JOB {idx + 1}/{len(job_configs)}: {job_config.job_name}")
            print(f"{'='*80}\n")

            try:
                result = self.execute_job(job_config)
                all_results.append(result)
                completed_jobs.append(asdict(result))

                # Checkpoint after each job
                self.checkpoint_mgr.save(
                    last_completed_index=idx,
                    total_items=len(job_configs),
                    completed_jobs=completed_jobs
                )

            except Exception as e:
                print(f"\n❌ Job {job_config.job_name} failed: {e}")
                print(f"Traceback: {traceback.format_exc()}")

                # Create failed result
                failed_result = JobResult(
                    job_id=job_config.job_id,
                    job_name=job_config.job_name,
                    status='failed',
                    start_time=datetime.now().isoformat(),
                    end_time=datetime.now().isoformat(),
                    duration_seconds=0.0,
                    stage_results=[],
                    final_output_dir='',
                    summary_metrics={'error': str(e)}
                )
                all_results.append(failed_result)
                completed_jobs.append(asdict(failed_result))

                # Checkpoint and continue
                self.checkpoint_mgr.save(
                    last_completed_index=idx,
                    total_items=len(job_configs),
                    completed_jobs=completed_jobs
                )

        # Clear checkpoint after completion
        self.checkpoint_mgr.clear()

        print(f"\n{'='*80}")
        print(f"✅ BATCH EXECUTION COMPLETE")
        print(f"{'='*80}")
        print(f"  Total jobs:      {len(job_configs)}")
        print(f"  Completed:       {sum(1 for r in all_results if r.status == 'completed')}")
        print(f"  Partial:         {sum(1 for r in all_results if r.status == 'partial')}")
        print(f"  Failed:          {sum(1 for r in all_results if r.status == 'failed')}")
        print(f"{'='*80}\n")

        return all_results

    def execute_job(self, job_config: JobConfig) -> JobResult:
        """
        Execute a single batch job through all configured stages

        Args:
            job_config: Job configuration

        Returns:
            JobResult object
        """
        start_time = datetime.now()
        job_dir = self.workspace_dir / f"jobs/{job_config.job_id}_{job_config.job_name}"
        job_dir.mkdir(parents=True, exist_ok=True)

        stage_results = []
        previous_output = None

        # Execute stages in order
        for stage_name in job_config.stages_to_run:
            try:
                stage = PipelineStage(stage_name)

                print(f"\n{'─'*80}")
                print(f"⚙️  STAGE: {stage.value}")
                print(f"{'─'*80}\n")

                result = self.execute_stage(stage, job_config, job_dir, previous_output)
                stage_results.append(asdict(result))

                if result.status == 'completed':
                    previous_output = Path(result.output_dir)
                elif result.status == 'failed':
                    print(f"⚠️  Stage {stage.value} failed, stopping job execution")
                    break

            except Exception as e:
                print(f"❌ Stage {stage_name} error: {e}")
                print(f"Traceback: {traceback.format_exc()}")

                # Create failed stage result
                failed_stage = StageResult(
                    stage=stage_name,
                    status='failed',
                    start_time=datetime.now().isoformat(),
                    end_time=datetime.now().isoformat(),
                    duration_seconds=0.0,
                    output_dir='',
                    metrics={},
                    error=str(e)
                )
                stage_results.append(asdict(failed_stage))
                break

        # Determine overall status
        if all(r['status'] == 'completed' for r in stage_results):
            status = 'completed'
        elif any(r['status'] == 'completed' for r in stage_results):
            status = 'partial'
        else:
            status = 'failed'

        # Calculate duration
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Collect summary metrics
        summary_metrics = {
            'total_stages': len(stage_results),
            'completed_stages': sum(1 for r in stage_results if r['status'] == 'completed'),
            'failed_stages': sum(1 for r in stage_results if r['status'] == 'failed'),
            'skipped_stages': sum(1 for r in stage_results if r['status'] == 'skipped'),
        }

        # Create job result
        job_result = JobResult(
            job_id=job_config.job_id,
            job_name=job_config.job_name,
            status=status,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            duration_seconds=duration,
            stage_results=stage_results,
            final_output_dir=str(previous_output) if previous_output else '',
            summary_metrics=summary_metrics
        )

        # Save job result
        result_file = job_dir / "job_result.json"
        with open(result_file, 'w') as f:
            json.dump(asdict(job_result), f, indent=2)

        print(f"\n{'='*80}")
        print(f"📊 JOB SUMMARY: {job_config.job_name}")
        print(f"{'='*80}")
        print(f"  Status:           {status}")
        print(f"  Duration:         {duration/60:.1f} minutes")
        print(f"  Completed stages: {summary_metrics['completed_stages']}/{summary_metrics['total_stages']}")
        print(f"  Output:           {previous_output or 'N/A'}")
        print(f"{'='*80}\n")

        return job_result

    def execute_stage(
        self,
        stage: PipelineStage,
        job_config: JobConfig,
        job_dir: Path,
        previous_output: Optional[Path] = None
    ) -> StageResult:
        """
        Execute a single pipeline stage

        Args:
            stage: Pipeline stage to execute
            job_config: Job configuration
            job_dir: Job directory
            previous_output: Output from previous stage

        Returns:
            StageResult object
        """
        if stage == PipelineStage.VOCABULARY_GENERATION:
            return self._execute_vocabulary_generation(job_config, job_dir)
        elif stage == PipelineStage.IMAGE_GENERATION:
            return self._execute_image_generation(job_config, job_dir, previous_output)
        elif stage == PipelineStage.QUALITY_FILTERING:
            return self._execute_quality_filtering(job_config, job_dir, previous_output)
        elif stage == PipelineStage.DATASET_ORGANIZATION:
            return self._execute_dataset_organization(job_config, job_dir, previous_output)
        elif stage == PipelineStage.TRAINING_INTEGRATION:
            return self._execute_training_integration(job_config, job_dir, previous_output)
        else:
            raise ValueError(f"Unknown stage: {stage}")

    def _execute_vocabulary_generation(
        self,
        job_config: JobConfig,
        job_dir: Path
    ) -> StageResult:
        """
        Stage 1: Generate vocabulary and prompts

        Args:
            job_config: Job configuration
            job_dir: Job directory

        Returns:
            StageResult object
        """
        start_time = datetime.now()
        output_dir = job_dir / "01_vocabulary"
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Initialize vocabulary generator
            vocab_gen = VocabularyGenerator()

            # Generate vocabulary
            print(f"📝 Generating vocabulary for {job_config.character_name}...")
            vocabulary = vocab_gen.generate_character_vocabulary(
                character_name=job_config.character_name,
                character_description=job_config.character_description,
                **job_config.vocabulary_config
            )

            # Save vocabulary
            vocab_file = output_dir / "vocabulary.json"
            with open(vocab_file, 'w') as f:
                json.dump(vocabulary, f, indent=2)

            # Generate prompts
            print(f"📝 Generating {job_config.num_prompts} prompts...")
            prompts = vocab_gen.generate_prompts_batch(
                vocabulary=vocabulary,
                num_prompts=job_config.num_prompts
            )

            # Save prompts
            prompts_file = output_dir / "prompts.json"
            with open(prompts_file, 'w') as f:
                json.dump(prompts, f, indent=2)

            # Calculate duration
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            # Create result
            result = StageResult(
                stage=PipelineStage.VOCABULARY_GENERATION.value,
                status='completed',
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                duration_seconds=duration,
                output_dir=str(output_dir),
                metrics={
                    'vocabulary_size': len(vocabulary.get('vocabulary', [])),
                    'num_prompts': len(prompts),
                    'vocab_file': str(vocab_file),
                    'prompts_file': str(prompts_file)
                }
            )

            print(f"✅ Vocabulary generation complete")
            print(f"   - Vocabulary: {len(vocabulary.get('vocabulary', []))} terms")
            print(f"   - Prompts: {len(prompts)} generated")
            print(f"   - Output: {output_dir}\n")

            return result

        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            print(f"❌ Vocabulary generation failed: {e}")

            return StageResult(
                stage=PipelineStage.VOCABULARY_GENERATION.value,
                status='failed',
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                duration_seconds=duration,
                output_dir=str(output_dir),
                metrics={},
                error=str(e)
            )

    def _execute_image_generation(
        self,
        job_config: JobConfig,
        job_dir: Path,
        prompts_dir: Optional[Path] = None
    ) -> StageResult:
        """
        Stage 2: Generate images from prompts

        Args:
            job_config: Job configuration
            job_dir: Job directory
            prompts_dir: Directory containing prompts.json from stage 1

        Returns:
            StageResult object
        """
        start_time = datetime.now()
        output_dir = job_dir / "02_generation"
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Load prompts
            if prompts_dir is None:
                prompts_dir = job_dir / "01_vocabulary"

            prompts_file = prompts_dir / "prompts.json"
            if not prompts_file.exists():
                raise FileNotFoundError(f"Prompts file not found: {prompts_file}")

            with open(prompts_file, 'r') as f:
                prompts = json.load(f)

            # Create generation config
            gen_config = GenerationConfig(**job_config.generation_config)

            # Initialize generator
            print(f"🎨 Loading SDXL model...")
            generator = BatchImageGenerator(
                base_model_path=job_config.base_model_path,
                lora_paths=job_config.lora_paths,
                output_dir=output_dir,
                config=gen_config,
                device=job_config.device
            )

            # Generate images
            print(f"🎨 Generating images ({len(prompts)} prompts × {job_config.num_images_per_prompt} images)...")
            report = generator.generate_batch(
                prompts=prompts,
                num_images_per_prompt=job_config.num_images_per_prompt,
                checkpoint_interval=100
            )

            # Calculate duration
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            # Create result
            result = StageResult(
                stage=PipelineStage.IMAGE_GENERATION.value,
                status='completed',
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                duration_seconds=duration,
                output_dir=str(output_dir / "images"),
                metrics={
                    'total_prompts': report.total_prompts,
                    'images_generated': report.images_generated,
                    'images_failed': report.images_failed,
                    'avg_time_per_image': report.avg_time_per_image,
                    'images_dir': str(output_dir / "images"),
                    'report_file': str(output_dir / "generation_report.json")
                }
            )

            print(f"✅ Image generation complete")
            print(f"   - Prompts: {report.total_prompts}")
            print(f"   - Images generated: {report.images_generated}")
            print(f"   - Images failed: {report.images_failed}")
            print(f"   - Avg time/image: {report.avg_time_per_image:.2f}s")
            print(f"   - Output: {output_dir / 'images'}\n")

            return result

        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            print(f"❌ Image generation failed: {e}")

            return StageResult(
                stage=PipelineStage.IMAGE_GENERATION.value,
                status='failed',
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                duration_seconds=duration,
                output_dir=str(output_dir),
                metrics={},
                error=str(e)
            )

    def _execute_quality_filtering(
        self,
        job_config: JobConfig,
        job_dir: Path,
        images_dir: Optional[Path] = None
    ) -> StageResult:
        """
        Stage 3: Filter images by quality

        Args:
            job_config: Job configuration
            job_dir: Job directory
            images_dir: Directory containing images from stage 2

        Returns:
            StageResult object
        """
        start_time = datetime.now()
        output_dir = job_dir / "03_filtering"
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Determine input directory
            if images_dir is None:
                images_dir = job_dir / "02_generation" / "images"

            if not images_dir.exists():
                raise FileNotFoundError(f"Images directory not found: {images_dir}")

            # Create filter config
            filter_config = FilterConfig(**job_config.filtering_config)

            # Initialize filter
            print(f"🔍 Loading quality filters...")
            quality_filter = ImageQualityFilter(
                config=filter_config,
                checkpoint_dir=output_dir / "checkpoints",
                device=job_config.device
            )

            # Filter images
            print(f"🔍 Filtering images from {images_dir}...")
            report = quality_filter.filter_batch(
                input_dir=images_dir,
                output_dir=output_dir
            )

            # Calculate duration
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            # Create result
            result = StageResult(
                stage=PipelineStage.QUALITY_FILTERING.value,
                status='completed',
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                duration_seconds=duration,
                output_dir=str(output_dir),
                metrics={
                    'total_images': report.total_images,
                    'images_passed': report.images_passed,
                    'images_rejected': report.images_rejected,
                    'rejection_breakdown': report.rejection_breakdown,
                    'quality_distribution': report.quality_distribution,
                    'duplicates_found': report.duplicates_found,
                    'excellent_dir': str(output_dir / "excellent"),
                    'good_dir': str(output_dir / "good"),
                    'acceptable_dir': str(output_dir / "acceptable"),
                    'rejected_dir': str(output_dir / "rejected"),
                    'report_file': str(output_dir / "filtering_report.json")
                }
            )

            print(f"✅ Quality filtering complete")
            print(f"   - Total images: {report.total_images}")
            print(f"   - Passed: {report.images_passed} ({report.images_passed/report.total_images*100:.1f}%)")
            print(f"   - Rejected: {report.images_rejected} ({report.images_rejected/report.total_images*100:.1f}%)")
            print(f"   - Duplicates: {report.duplicates_found}")
            print(f"   - Output: {output_dir}\n")

            return result

        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            print(f"❌ Quality filtering failed: {e}")

            return StageResult(
                stage=PipelineStage.QUALITY_FILTERING.value,
                status='failed',
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                duration_seconds=duration,
                output_dir=str(output_dir),
                metrics={},
                error=str(e)
            )

    def _execute_dataset_organization(
        self,
        job_config: JobConfig,
        job_dir: Path,
        previous_output: Optional[StageResult]
    ) -> StageResult:
        """
        Execute dataset organization stage

        Organizes filtered images into Kohya_ss training format:
        - {repeat_count}_{concept_name}/ directory structure
        - image.png + image.txt (caption) pairs
        - dataset_metadata.json

        Args:
            job_config: Job configuration
            job_dir: Job working directory
            previous_output: Output from quality filtering stage

        Returns:
            StageResult with organization metrics
        """
        start_time = datetime.now()
        output_dir = job_dir / "04_dataset"
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"📦 DATASET ORGANIZATION")
        print(f"{'='*80}")

        try:
            # Get input directory from previous stage
            if previous_output is None:
                raise ValueError("No previous stage output (filtering required)")

            input_dir = Path(previous_output.output_dir)
            if not input_dir.exists():
                raise FileNotFoundError(f"Input directory not found: {input_dir}")

            # Get dataset organization config from job config
            org_config_dict = job_config.stage_configs.get('dataset_organization', {})

            # Create dataset configuration
            dataset_config = DatasetConfig(
                repeat_count=org_config_dict.get('repeat_count', 12),
                concept_name=org_config_dict.get('concept_name', job_config.job_id),
                min_resolution=org_config_dict.get('min_resolution', 512),
                max_resolution=org_config_dict.get('max_resolution', 2048),
                target_resolution=org_config_dict.get('target_resolution', 1024),
                resize_if_needed=org_config_dict.get('resize_if_needed', True),
                copy_images=org_config_dict.get('copy_images', True),
                generate_captions_if_missing=org_config_dict.get('generate_captions_if_missing', False),
                caption_template=org_config_dict.get('caption_template', None)
            )

            print(f"  Concept name:    {dataset_config.concept_name}")
            print(f"  Repeat count:    {dataset_config.repeat_count}")
            print(f"  Input dir:       {input_dir}")
            print(f"  Output dir:      {output_dir}")
            print(f"  Target res:      {dataset_config.target_resolution}px")

            # Initialize dataset organizer
            organizer = DatasetOrganizer(
                output_dir=output_dir,
                config=dataset_config
            )

            # Look for prompts file from generation stage
            prompts_file = None
            generation_dir = job_dir / "01_vocabulary"
            if generation_dir.exists():
                prompts_json = generation_dir / "prompts.json"
                if prompts_json.exists():
                    prompts_file = prompts_json

            # Run organization
            print(f"\n  Organizing dataset...")
            results = organizer.organize_from_directory(
                source_dir=input_dir,
                prompts_file=prompts_file,
                metadata_file=None
            )

            print(f"\n  ✅ Organization complete!")
            print(f"  Images organized: {results['stats']['images_copied']}")
            print(f"  Captions created: {results['stats']['captions_created']}")
            print(f"  Images skipped:   {results['stats']['images_skipped']}")
            print(f"  Concept dir:      {results['concept_dir']}")

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            return StageResult(
                stage=PipelineStage.DATASET_ORGANIZATION.value,
                status='completed',
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                duration_seconds=duration,
                output_dir=str(output_dir),
                metrics={
                    'images_organized': results['stats']['images_copied'],
                    'captions_created': results['stats']['captions_created'],
                    'images_skipped': results['stats']['images_skipped'],
                    'captions_from_source': results['stats']['captions_from_source'],
                    'concept_directory': results['concept_dir'],
                    'metadata_file': results['metadata_file'],
                    'repeat_count': dataset_config.repeat_count,
                    'concept_name': dataset_config.concept_name
                }
            )

        except Exception as e:
            print(f"\n  ❌ Dataset organization failed: {e}")
            logging.error(f"Dataset organization error: {e}", exc_info=True)

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            return StageResult(
                stage=PipelineStage.DATASET_ORGANIZATION.value,
                status='failed',
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                duration_seconds=duration,
                output_dir=str(output_dir),
                metrics={},
                error=str(e)
            )

    def _execute_training_integration(
        self,
        job_config: JobConfig,
        job_dir: Path,
        previous_output: Optional[StageResult]
    ) -> StageResult:
        """
        Execute training integration stage - generate config and launch training

        Args:
            job_config: Job configuration
            job_dir: Job directory
            previous_output: Output from dataset organization stage

        Returns:
            StageResult with training status and metrics
        """
        start_time = datetime.now()
        print(f"\n{'='*80}")
        print(f"🎯 Stage: Training Integration")
        print(f"{'='*80}")

        # Get training config from stage_configs
        training_config = job_config.stage_configs.get('training_integration', {})

        # Setup directories
        output_dir = job_dir / "06_training"
        output_dir.mkdir(parents=True, exist_ok=True)

        config_output_dir = output_dir / "configs"
        config_output_dir.mkdir(parents=True, exist_ok=True)

        lora_output_dir = Path(training_config.get('output_lora_dir', str(output_dir / "lora")))
        lora_output_dir.mkdir(parents=True, exist_ok=True)

        training_log_dir = Path(training_config.get('training_log_dir', str(output_dir / "logs")))
        training_log_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Get dataset directory from previous stage
            if previous_output is None or previous_output.status != 'completed':
                raise ValueError("Dataset organization stage did not complete successfully")

            dataset_dir = Path(previous_output.output_dir)

            print(f"\n  Dataset directory: {dataset_dir}")
            print(f"  LoRA output:      {lora_output_dir}")
            print(f"  Training logs:    {training_log_dir}")

            # ===================================================================
            # Step 1: Generate Training Config
            # ===================================================================
            print(f"\n  📝 Generating training configuration...")

            base_model_path = training_config.get(
                'base_model_path',
                job_config.base_model_path
            )

            # Initialize config generator
            config_generator = TrainingConfigGenerator()

            # Generate config with overrides from training_config
            overrides = {
                'network_dim': training_config.get('network_dim', 64),
                'network_alpha': training_config.get('network_alpha', 32),
                'learning_rate': training_config.get('learning_rate', 0.0001),
                'max_train_epochs': training_config.get('max_train_epochs', 4),
                'save_every_n_epochs': training_config.get('save_every_n_epochs', 2),
                'train_batch_size': training_config.get('train_batch_size', 1),
                'gradient_accumulation_steps': training_config.get('gradient_accumulation_steps', 2),
                'mixed_precision': training_config.get('mixed_precision', 'bf16'),
                'optimizer_type': training_config.get('optimizer_type', 'AdamW8bit'),
                'lr_scheduler': training_config.get('lr_scheduler', 'cosine_with_restarts'),
                'min_snr_gamma': training_config.get('min_snr_gamma', 5.0),
                'noise_offset': training_config.get('noise_offset', 0.05),
                'resolution': training_config.get('resolution', '1024,1024'),
                'enable_bucket': training_config.get('enable_bucket', True),
            }

            generated_config = config_generator.generate_config(
                base_model_path=base_model_path,
                dataset_dir=dataset_dir,
                output_lora_dir=lora_output_dir,
                character_name=job_config.character_name,
                **overrides
            )

            # Save config
            config_file = config_output_dir / f"{job_config.character_name}_training.toml"
            saved_config_path = config_generator.save_config(generated_config, config_file)

            print(f"  ✅ Config saved: {saved_config_path}")

            # Validate config
            if not config_generator.validate_config(generated_config):
                raise ValueError("Generated training config failed validation")

            # ===================================================================
            # Step 2: Launch Training (optional - controlled by run_in_background)
            # ===================================================================
            run_in_background = training_config.get('run_in_background', True)
            blocking = not run_in_background

            if not training_config.get('skip_training', False):
                print(f"\n  🚀 Launching training...")
                print(f"  Mode: {'Background' if run_in_background else 'Blocking'}")

                # Initialize launcher
                launcher = TrainingLauncher(
                    kohya_scripts_path=training_config.get(
                        'kohya_scripts_path',
                        '/mnt/c/ai_projects/kohya_ss/sd-scripts'
                    ),
                    conda_env=training_config.get('kohya_conda_env', 'kohya_ss'),
                    device=job_config.device,
                    use_tmux=training_config.get('use_tmux', False)
                )

                # Launch training
                training_process = launcher.launch_training(
                    config_path=saved_config_path,
                    job_id=job_config.job_id,
                    output_dir=lora_output_dir,
                    log_dir=training_log_dir,
                    blocking=blocking,
                    check_gpu_memory=training_config.get('check_gpu_memory', True),
                    required_vram_mb=training_config.get('required_vram_mb', 8000)
                )

                print(f"  ✅ Training launched: PID {training_process.pid}")
                print(f"  Log file: {training_process.log_file}")

                # Save process state
                process_state_file = output_dir / "training_process.json"
                launcher.save_process_state(job_config.job_id, process_state_file)

                training_status = training_process.status.value
                training_pid = training_process.pid

                # If blocking, wait for completion
                if blocking:
                    print(f"\n  ⏳ Waiting for training to complete...")
                    final_process = launcher.get_status(job_config.job_id)
                    training_status = final_process.status.value

                    if final_process.status == TrainingStatus.FAILED:
                        raise RuntimeError(f"Training failed: {final_process.error_message}")

                    print(f"  ✅ Training completed successfully")
            else:
                print(f"\n  ⏭️  Skipping training launch (skip_training=True)")
                training_status = 'skipped'
                training_pid = None

            # ===================================================================
            # Step 3: Prepare result
            # ===================================================================
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            metrics = {
                'config_file': str(saved_config_path),
                'lora_output_dir': str(lora_output_dir),
                'training_log_dir': str(training_log_dir),
                'training_status': training_status,
                'training_pid': training_pid,
                'blocking_mode': blocking,
                'base_model': base_model_path,
                'network_dim': generated_config.network_dim,
                'network_alpha': generated_config.network_alpha,
                'max_train_epochs': generated_config.max_train_epochs,
                'learning_rate': generated_config.learning_rate,
            }

            print(f"\n  ✅ Training integration complete")
            print(f"  Config: {saved_config_path}")
            if not training_config.get('skip_training', False):
                print(f"  Status: {training_status}")
                if run_in_background:
                    print(f"  PID: {training_pid}")
                    print(f"  Note: Training running in background")

            return StageResult(
                stage=PipelineStage.TRAINING_INTEGRATION.value,
                status='completed',
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                duration_seconds=duration,
                output_dir=str(output_dir),
                metrics=metrics
            )

        except Exception as e:
            print(f"\n  ❌ Training integration failed: {e}")
            logging.error(f"Training integration error: {e}", exc_info=True)

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            return StageResult(
                stage=PipelineStage.TRAINING_INTEGRATION.value,
                status='failed',
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                duration_seconds=duration,
                output_dir=str(output_dir),
                metrics={},
                error=str(e)
            )

    def generate_report(
        self,
        job_results: List[JobResult],
        output_file: Path
    ):
        """
        Generate comprehensive batch report

        Args:
            job_results: List of job results
            output_file: Output file path
        """
        report = {
            'batch_summary': {
                'total_jobs': len(job_results),
                'completed_jobs': sum(1 for r in job_results if r.status == 'completed'),
                'partial_jobs': sum(1 for r in job_results if r.status == 'partial'),
                'failed_jobs': sum(1 for r in job_results if r.status == 'failed'),
                'total_duration_seconds': sum(r.duration_seconds for r in job_results),
            },
            'job_results': [asdict(r) for r in job_results],
            'generated_at': datetime.now().isoformat()
        }

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n{'='*80}")
        print(f"📊 BATCH REPORT")
        print(f"{'='*80}")
        print(f"  Total jobs:      {report['batch_summary']['total_jobs']}")
        print(f"  Completed:       {report['batch_summary']['completed_jobs']}")
        print(f"  Partial:         {report['batch_summary']['partial_jobs']}")
        print(f"  Failed:          {report['batch_summary']['failed_jobs']}")
        print(f"  Total duration:  {report['batch_summary']['total_duration_seconds']/60:.1f} minutes")
        print(f"  Report saved:    {output_file}")
        print(f"{'='*80}\n")


def main():
    """CLI for batch orchestrator"""
    import argparse

    parser = argparse.ArgumentParser(description="Batch Orchestration Layer")
    parser.add_argument("--config", type=str, required=True, help="Batch configuration JSON file")
    parser.add_argument("--workspace", type=str, required=True, help="Workspace directory")
    parser.add_argument("--log-dir", type=str, required=True, help="Log directory")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="Checkpoint directory")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume from checkpoint")

    args = parser.parse_args()

    # Load batch config
    with open(args.config, 'r') as f:
        batch_config = json.load(f)

    # Create job configs
    job_configs = [JobConfig(**jc) for jc in batch_config['jobs']]

    # Initialize orchestrator
    orchestrator = BatchOrchestrator(
        workspace_dir=Path(args.workspace),
        log_dir=Path(args.log_dir),
        checkpoint_dir=Path(args.checkpoint_dir)
    )

    # Execute batch
    results = orchestrator.execute_batch(
        job_configs=job_configs,
        resume=not args.no_resume
    )

    # Generate report
    report_file = Path(args.workspace) / f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    orchestrator.generate_report(results, report_file)

    print(f"✅ Batch orchestration complete!")
    print(f"   Report: {report_file}")


if __name__ == "__main__":
    main()
