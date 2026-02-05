#!/usr/bin/env python3
"""
Synthetic Data Generation Pipeline Orchestrator (v2.0)

This Python script coordinates the entire synthetic data generation pipeline,
reading configuration from YAML and executing the workflow phases.

v2.0 Changes (2025-12-06):
- Added Phase 2.5: Caption generation from prompts
- Integrated prompt_to_caption_converter for identity removal
- Updated to support 100 prompts per type
- Added caption conversion statistics

Author: LLMProvider Tooling
Date: 2025-11-30 (v1), 2025-12-06 (v2)
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional

import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import caption converter
try:
    from scripts.generic.training.caption_engines.prompt_to_caption_converter import (
        PromptToCaptionConverter
    )
    CAPTION_CONVERTER_AVAILABLE = True
except ImportError:
    CAPTION_CONVERTER_AVAILABLE = False
    logging.warning("Caption converter not available - caption generation will be skipped")


@dataclass
class PipelineConfig:
    """Pipeline configuration loaded from YAML (v2.0)"""
    workspace_root: Path
    workspace_logs_dir: Path
    workspace_checkpoints_dir: Path
    workspace_generated_data_dir: Path
    workspace_filtered_data_dir: Path
    workspace_datasets_dir: Path
    base_model: Path
    identity_loras_dir: Path
    characters: List[str]
    lora_types: List[str]
    character_descriptions: Dict[str, str]

    # Vocabulary settings
    num_prompts_per_type: int
    use_templates: bool
    template_variations: int
    ensure_template_coverage: bool
    vocab_seed: Optional[int]

    # Caption generation settings (v2.0)
    caption_enabled: bool
    caption_min_tokens: int
    caption_max_tokens: int
    caption_target_tokens: int
    caption_generic_subject: str

    # Image generation settings
    num_images_per_prompt: int
    num_inference_steps: int
    guidance_scale: float
    height: int
    width: int
    use_random_seeds: bool
    negative_prompt: str
    lora_scale: float
    device: str

    # Resilience settings
    max_retries: int
    retry_delay_seconds: int
    gpu_recovery_delay_seconds: int
    enable_checkpointing: bool
    checkpoint_filename: str

    # Logging
    log_level: str
    conda_env: str

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "PipelineConfig":
        """Load configuration from YAML file (v2.0)"""
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)

        workspace = config['workspace']
        workspace_root = Path(workspace["root"])
        subdirs = workspace.get("subdirs", {}) or {}
        logs_dir = workspace_root / str(subdirs.get("logs", "logs"))
        checkpoints_dir = workspace_root / str(subdirs.get("checkpoints", "checkpoints"))
        generated_data_dir = workspace_root / str(subdirs.get("generated_data", "generated_data"))
        filtered_data_dir = workspace_root / str(subdirs.get("filtered_data", "filtered_data"))
        datasets_dir = workspace_root / str(subdirs.get("datasets", "datasets"))

        models = config['models']
        vocab = config['vocabulary_generation']
        image_gen = config['image_generation']
        resilience = config['resilience']
        logging_cfg = config['logging']
        conda = config['conda']

        # v2.0: Load caption generation settings (with defaults for backward compatibility)
        caption_cfg = config.get('caption_generation', {})

        return cls(
            workspace_root=workspace_root,
            workspace_logs_dir=logs_dir,
            workspace_checkpoints_dir=checkpoints_dir,
            workspace_generated_data_dir=generated_data_dir,
            workspace_filtered_data_dir=filtered_data_dir,
            workspace_datasets_dir=datasets_dir,
            base_model=Path(models['base_model']),
            identity_loras_dir=Path(models['identity_loras_dir']),
            characters=config['characters'],
            lora_types=config['lora_types'],
            character_descriptions=config.get('character_descriptions', {}),

            num_prompts_per_type=vocab['num_prompts_per_type'],
            use_templates=vocab['use_templates'],
            template_variations=vocab['template_variations'],
            ensure_template_coverage=vocab.get('ensure_template_coverage', True),
            vocab_seed=vocab.get('seed'),

            # v2.0: Caption generation settings
            caption_enabled=caption_cfg.get('enabled', True),
            caption_min_tokens=caption_cfg.get('min_tokens', 30),
            caption_max_tokens=caption_cfg.get('max_tokens', 225),
            caption_target_tokens=caption_cfg.get('target_tokens', 100),
            caption_generic_subject=caption_cfg.get('generic_subject', 'a 3d animated character'),

            num_images_per_prompt=image_gen['num_images_per_prompt'],
            num_inference_steps=image_gen['num_inference_steps'],
            guidance_scale=image_gen['guidance_scale'],
            height=image_gen['height'],
            width=image_gen['width'],
            use_random_seeds=image_gen['use_random_seeds'],
            negative_prompt=image_gen.get('negative_prompt', ''),
            lora_scale=image_gen['lora_scale'],
            device=image_gen['device'],

            max_retries=resilience['max_retries'],
            retry_delay_seconds=resilience['retry_delay_seconds'],
            gpu_recovery_delay_seconds=resilience['gpu_recovery_delay_seconds'],
            enable_checkpointing=resilience['enable_checkpointing'],
            checkpoint_filename=resilience['checkpoint_filename'],

            log_level=logging_cfg['log_level'],
            conda_env=conda['env_name']
        )


class CheckpointManager:
    """Manages pipeline checkpoint for resume capability"""

    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> Dict[str, Any]:
        """Load checkpoint data"""
        if self.checkpoint_path.exists():
            with open(self.checkpoint_path, 'r') as f:
                return json.load(f)
        return {}

    def is_completed(self, task_key: str) -> bool:
        """Check if a task is marked as completed"""
        data = self.load()
        return data.get(task_key) == "completed"

    def mark_completed(self, task_key: str):
        """Mark a task as completed"""
        data = self.load()
        data[task_key] = "completed"
        data[f"{task_key}_timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        with open(self.checkpoint_path, 'w') as f:
            json.dump(data, f, indent=2)

        logging.info(f"✓ Marked completed: {task_key}")


class SyntheticDataOrchestrator:
    """Main orchestrator for synthetic data generation pipeline"""

    def __init__(self, config: PipelineConfig, resume: bool = True, dry_run: bool = False):
        self.config = config
        self.resume = resume
        self.dry_run = dry_run

        # Setup directories
        self.config.workspace_root.mkdir(parents=True, exist_ok=True)
        self.config.workspace_logs_dir.mkdir(parents=True, exist_ok=True)
        self.config.workspace_checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.config.workspace_generated_data_dir.mkdir(parents=True, exist_ok=True)
        self.config.workspace_filtered_data_dir.mkdir(parents=True, exist_ok=True)
        self.config.workspace_datasets_dir.mkdir(parents=True, exist_ok=True)

        # Setup checkpoint manager
        checkpoint_path = self.config.workspace_checkpoints_dir / self.config.checkpoint_filename
        self.checkpoint_mgr = CheckpointManager(checkpoint_path)

        # Setup logging
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        log_level = getattr(logging, self.config.log_level.upper())
        logging.basicConfig(level=log_level, format=log_format)

        self.stats = {
            "total_vocabularies": 0,
            "total_images": 0,
            "failed_tasks": []
        }

    def run_command(self, cmd: List[str], task_name: str, retries: int = 0) -> bool:
        """Execute a command with optional retry logic"""
        max_attempts = retries + 1

        # Set PYTHONPATH to project root for module imports
        import os
        env = os.environ.copy()
        env['PYTHONPATH'] = str(PROJECT_ROOT)

        for attempt in range(1, max_attempts + 1):
            logging.info(f"[{attempt}/{max_attempts}] Executing: {task_name}")
            logging.debug(f"Command: {' '.join(cmd)}")

            if self.dry_run:
                logging.info("[DRY RUN] Command prepared but not executed")
                return True

            try:
                result = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                    env=env
                )
                logging.info(f"✓ Success: {task_name}")
                if result.stdout:
                    logging.debug(f"Output: {result.stdout[:500]}")
                return True

            except subprocess.CalledProcessError as e:
                logging.error(f"✗ Failed: {task_name} (exit code: {e.returncode})")
                if e.stderr:
                    logging.error(f"Error output: {e.stderr[:500]}")

                if attempt < max_attempts:
                    delay = self.config.retry_delay_seconds
                    logging.info(f"Retrying in {delay}s... ({max_attempts - attempt} attempts left)")
                    time.sleep(delay)
                else:
                    logging.error(f"All {max_attempts} attempts exhausted for: {task_name}")
                    self.stats["failed_tasks"].append(task_name)
                    return False

        return False

    def phase_1_vocabulary_generation(self, characters: Optional[List[str]] = None,
                                     lora_types: Optional[List[str]] = None):
        """Phase 1: Generate prompt vocabularies"""
        logging.info("=" * 72)
        logging.info("PHASE 1: VOCABULARY GENERATION")
        logging.info("=" * 72)

        characters = characters or self.config.characters
        lora_types = lora_types or self.config.lora_types

        total_tasks = len(characters) * len(lora_types)
        completed = 0

        for char in characters:
            for lora_type in lora_types:
                task_key = f"vocab_{char}_{lora_type}"

                if self.resume and self.checkpoint_mgr.is_completed(task_key):
                    logging.info(f"⏭️  Skipping completed: {task_key}")
                    completed += 1
                    self.stats["total_vocabularies"] += 1
                    continue

                # Get character description
                char_desc = self.config.character_descriptions.get(
                    char,
                    self.config.character_descriptions.get("default", "A 3D animated character")
                )

                # Build output path
                output_file = self.config.workspace_generated_data_dir / char / lora_type / "prompts.json"
                output_file.parent.mkdir(parents=True, exist_ok=True)

                # Build command
                cmd = [
                    "conda", "run", "-n", self.config.conda_env, "python",
                    str(PROJECT_ROOT / "scripts/generic/training/orchestration/vocabulary_generator.py"),
                    "--character-name", char,
                    "--character-description", char_desc,
                    "--lora-type", lora_type,
                    "--num-prompts", str(self.config.num_prompts_per_type),
                    "--output-file", str(output_file)
                ]

                if self.config.use_templates:
                    cmd.append("--use-templates")
                    cmd.extend(["--template-variations", str(self.config.template_variations)])

                if self.config.vocab_seed is not None:
                    cmd.extend(["--seed", str(self.config.vocab_seed)])

                # Execute
                success = self.run_command(cmd, task_key, retries=self.config.max_retries)

                if success:
                    self.checkpoint_mgr.mark_completed(task_key)
                    completed += 1
                    self.stats["total_vocabularies"] += 1
                    if self.config.caption_enabled and CAPTION_CONVERTER_AVAILABLE:
                        self._generate_captions(output_file, character_name=char)

        logging.info(f"Phase 1 complete: {completed}/{total_tasks} vocabularies generated")

    def phase_2_image_generation(self, characters: Optional[List[str]] = None,
                                lora_types: Optional[List[str]] = None):
        """Phase 2: Generate synthetic images"""
        logging.info("=" * 72)
        logging.info("PHASE 2: IMAGE GENERATION")
        logging.info("=" * 72)

        characters = characters or self.config.characters
        lora_types = lora_types or self.config.lora_types

        total_tasks = len(characters) * len(lora_types)
        completed = 0

        for char in characters:
            # Find identity LoRA - use exact filename to avoid matching wrong characters
            # (e.g. "alberto" should not match "alberto_seamonster")
            lora_filename = f"BEST_{char}_lora_sdxl.safetensors"
            lora_path = self.config.identity_loras_dir / lora_filename

            if not lora_path.exists():
                logging.warning(f"⚠️  Identity LoRA not found: {lora_path}, skipping {char}")
                continue

            logging.info(f"Using identity LoRA: {lora_path.name}")

            for lora_type in lora_types:
                task_key = f"generation_{char}_{lora_type}"

                if self.resume and self.checkpoint_mgr.is_completed(task_key):
                    logging.info(f"⏭️  Skipping completed: {task_key}")
                    completed += 1
                    continue

                # Build paths
                prompts_file = self.config.workspace_generated_data_dir / char / lora_type / "prompts.json"
                prompts_converted = self.config.workspace_generated_data_dir / char / lora_type / "prompts_converted.json"
                output_dir = self.config.workspace_generated_data_dir / char / lora_type / "generated"

                if not prompts_file.exists():
                    logging.error(f"✗ Prompts file not found: {prompts_file}")
                    continue

                # Convert prompts format
                self._convert_prompts_format(prompts_file, prompts_converted)

                # Build command
                output_dir.mkdir(parents=True, exist_ok=True)

                cmd = [
                    "conda", "run", "-n", self.config.conda_env, "python",
                    str(PROJECT_ROOT / "scripts/generic/training/batch_image_generator.py"),
                    "--prompts-file", str(prompts_converted),
                    "--base-model", str(self.config.base_model),
                    "--lora-paths", str(lora_path),
                    "--output-dir", str(output_dir),
                    "--num-images-per-prompt", str(self.config.num_images_per_prompt),
                    "--steps", str(self.config.num_inference_steps),
                    "--guidance-scale", str(self.config.guidance_scale),
                    "--height", str(self.config.height),
                    "--width", str(self.config.width),
                    "--device", self.config.device
                ]

                # Add lora_scales only if configured
                if self.config.lora_scale:
                    cmd.extend(["--lora-scales", str(self.config.lora_scale)])

                if self.config.use_random_seeds:
                    cmd.append("--use-random-seeds")

                if self.config.negative_prompt:
                    cmd.extend(["--negative-prompt", self.config.negative_prompt])

                # Execute
                success = self.run_command(cmd, task_key, retries=self.config.max_retries)

                if success:
                    self.checkpoint_mgr.mark_completed(task_key)
                    completed += 1

                    # Count generated images
                    image_count = len(list(output_dir.glob("*.png")))
                    self.stats["total_images"] += image_count
                    logging.info(f"Generated {image_count} images for {char}/{lora_type}")

        logging.info(f"Phase 2 complete: {completed}/{total_tasks} generation tasks completed")
        logging.info(f"Total images generated: {self.stats['total_images']}")

    def _convert_prompts_format(self, input_file: Path, output_file: Path,
                                character_name: Optional[str] = None):
        """Convert vocabulary JSON format to batch_image_generator format (v2.0)"""
        with open(input_file, 'r') as f:
            data = json.load(f)

        prompts = [p["prompt"] for p in data["prompts"]]

        # Include negative_prompt in the converted file
        output = {
            "prompts": prompts,
            "negative_prompt": self.config.negative_prompt
        }

        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)

        logging.debug(f"Converted {len(prompts)} prompts: {input_file.name} → {output_file.name}")

        # v2.0: Generate captions if enabled
        if self.config.caption_enabled and CAPTION_CONVERTER_AVAILABLE:
            self._generate_captions(input_file, character_name or data.get("character", "unknown"))

    def _generate_captions(self, prompts_file: Path, character_name: str):
        """Generate training captions from prompts by removing character identity (v2.0)"""
        caption_file = prompts_file.parent / "captions.json"

        # Check if already generated
        if caption_file.exists():
            logging.debug(f"Captions already exist: {caption_file.name}")
            return

        logging.info(f"Generating captions for {character_name}...")

        # Load prompts
        with open(prompts_file, 'r') as f:
            data = json.load(f)

        # Initialize converter
        converter = PromptToCaptionConverter(
            min_tokens=self.config.caption_min_tokens,
            max_tokens=self.config.caption_max_tokens,
            target_tokens=self.config.caption_target_tokens,
            generic_subject=self.config.caption_generic_subject,
        )

        # Convert prompts to captions
        results = converter.convert_batch(data["prompts"], character_name=character_name)

        # Calculate statistics
        valid_count = sum(1 for r in results if r["is_valid"])
        avg_tokens = sum(r["token_count"] for r in results) / len(results) if results else 0

        logging.info(f"  Generated {len(results)} captions ({valid_count} valid, avg {avg_tokens:.1f} tokens)")

        # Save captions
        caption_data = {
            "character": character_name,
            "lora_type": data.get("lora_type", "unknown"),
            "num_captions": len(results),
            "stats": {
                "valid_count": valid_count,
                "total_count": len(results),
                "avg_token_count": avg_tokens,
            },
            "captions": results
        }

        with open(caption_file, 'w', encoding='utf-8') as f:
            json.dump(caption_data, f, indent=2, ensure_ascii=False)

        # Also create individual .txt caption files for Kohya format
        txt_captions_dir = prompts_file.parent / "txt_captions"
        txt_captions_dir.mkdir(exist_ok=True)

        for i, result in enumerate(results):
            txt_file = txt_captions_dir / f"prompt_{i:04d}.txt"
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(result["caption"])

        logging.info(f"  Saved captions to: {caption_file.name} and {txt_captions_dir.name}/")


def main():
    parser = argparse.ArgumentParser(description="Synthetic Data Generation Pipeline Orchestrator")

    parser.add_argument("--config", type=str, required=True,
                       help="Path to YAML configuration file")
    parser.add_argument("--phase", type=str, choices=["1", "2", "all"], default="all",
                       help="Which phase to run (1=vocab, 2=generation, all=both)")
    parser.add_argument("--resume", action="store_true", default=True,
                       help="Resume from checkpoint")
    parser.add_argument("--no-resume", action="store_false", dest="resume",
                       help="Start fresh, ignoring checkpoints")
    parser.add_argument("--characters", type=str,
                       help="Comma-separated list of characters to process (overrides config)")
    parser.add_argument("--lora-types", type=str,
                       help="Comma-separated list of LoRA types (overrides config)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be done without executing")

    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)

    config = PipelineConfig.from_yaml(config_path)

    # Override characters/lora_types if specified
    characters = args.characters.split(',') if args.characters else None
    lora_types = args.lora_types.split(',') if args.lora_types else None

    # Create orchestrator
    orchestrator = SyntheticDataOrchestrator(config, resume=args.resume, dry_run=args.dry_run)

    # Run phases
    start_time = time.time()

    if args.phase in ["1", "all"]:
        orchestrator.phase_1_vocabulary_generation(characters, lora_types)

    if args.phase in ["2", "all"]:
        orchestrator.phase_2_image_generation(characters, lora_types)

    # Report
    duration = time.time() - start_time

    logging.info("=" * 72)
    logging.info("PIPELINE COMPLETE")
    logging.info("=" * 72)
    logging.info(f"Duration: {duration:.1f}s ({duration/60:.1f}min)")
    logging.info(f"Vocabularies generated: {orchestrator.stats['total_vocabularies']}")
    logging.info(f"Images generated: {orchestrator.stats['total_images']}")

    if orchestrator.stats["failed_tasks"]:
        logging.warning(f"Failed tasks: {len(orchestrator.stats['failed_tasks'])}")
        for task in orchestrator.stats["failed_tasks"]:
            logging.warning(f"  - {task}")


if __name__ == "__main__":
    main()
