#!/usr/bin/env python3
"""
Launch iteration_v6 training with optimized dataset and captions

Key improvements in v6:
1. Smart-curated dataset (372 diverse images, inpaint-only, no enhancement)
2. Enhanced captions with detailed expression and action descriptions
3. Optimized training parameters for low-contrast Pixar style
4. Checkpoint-level evaluation for immediate feedback

Usage:
    python scripts/training/launch_iteration_v6.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.training.iterative_lora_optimizer import IterativeTrainingOrchestrator

# Configuration for iteration_v6
characters = ['luca_human']  # Focus on luca_human first
base_dataset_dir = Path('/mnt/data/ai_data/datasets/3d-anime/luca/curated_dataset_v2_smart')
base_model_path = '/mnt/c/AI_LLM_projects/ai_warehouse/tool-caches/huggingface/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14'
output_base_dir = Path('/mnt/data/ai_data/models/lora/luca/iterative_overnight_v6')
sd_scripts_dir = Path('/mnt/c/AI_LLM_projects/ai_warehouse/sd-scripts')

# 🔄 CONTINUATION FROM ITERATION 3 BEST MODELS
# Use iteration3 best model as starting point (proven to work well)
pretrained_lora_dir = Path('/mnt/data/ai_data/models/lora/luca/BEST_MODELS')

max_iterations = 15  # Reasonable limit for focused training
time_limit_hours = 12.0  # Overnight training window

print("="*80)
print("ITERATION V6 - OPTIMIZED DATASET + ENHANCED CAPTIONS")
print("="*80)
print()
print("🎯 Key Improvements:")
print("  1. Smart-curated dataset: 372 diverse images (vs 191 previous)")
print("  2. Inpaint-only data: preserves low-contrast Pixar style")
print("  3. Enhanced captions: detailed expression + action descriptions")
print("  4. From iteration3 best weights: proven baseline")
print()
print(f"Characters: {', '.join(characters)}")
print(f"Dataset: {base_dataset_dir}")
print(f"Base Model: SD v1.5")
print(f"Output: {output_base_dir}")
print(f"Max Iterations: {max_iterations}")
print(f"Time Budget: {time_limit_hours} hours")
print("="*80)
print()

# Verify dataset directories exist
for char in characters:
    char_images = base_dataset_dir / char / 'images'
    char_captions = base_dataset_dir / char / 'captions'

    if not char_images.exists():
        print(f"ERROR: Images not found for {char}: {char_images}")
        sys.exit(1)

    if not char_captions.exists():
        print(f"ERROR: Captions not found for {char}: {char_captions}")
        sys.exit(1)

    # Count images and captions
    images = list(char_images.glob('*.png'))
    captions = list(char_captions.glob('*.txt'))
    print(f"✓ {char}:")
    print(f"    Images: {len(images)}")
    print(f"    Captions: {len(captions)}")

    if len(images) != len(captions):
        print(f"  ⚠️  Warning: Image count ({len(images)}) != Caption count ({len(captions)})")

print()

# Check if caption generation is complete
char = characters[0]
expected_captions = 372
actual_captions = len(list((base_dataset_dir / char / 'captions').glob('*.txt')))

if actual_captions < expected_captions:
    print(f"⚠️  WARNING: Caption generation incomplete!")
    print(f"    Expected: {expected_captions} captions")
    print(f"    Found: {actual_captions} captions ({actual_captions/expected_captions*100:.1f}%)")
    print()
    response = input("    Continue anyway? (y/N): ")
    if response.lower() != 'y':
        print("    Aborted. Please wait for caption generation to complete.")
        sys.exit(0)

# Create orchestrator
orchestrator = IterativeTrainingOrchestrator(
    characters=characters,
    base_dataset_dir=base_dataset_dir,
    base_model_path=base_model_path,
    output_base_dir=output_base_dir,
    sd_scripts_dir=sd_scripts_dir,
    max_iterations=max_iterations,
    time_limit_hours=time_limit_hours,
    pretrained_lora_dir=pretrained_lora_dir  # Continue from iteration3 best
)

# Start training
print("\n" + "="*80)
print("STARTING ITERATION V6 TRAINING")
print("="*80)
print()

try:
    orchestrator.run_optimization()
    print("\n" + "="*80)
    print("✓ ITERATION V6 TRAINING COMPLETED SUCCESSFULLY")
    print("="*80)
except KeyboardInterrupt:
    print("\n" + "="*80)
    print("! TRAINING INTERRUPTED BY USER")
    print("="*80)
except Exception as e:
    print("\n" + "="*80)
    print(f"✗ TRAINING FAILED: {e}")
    print("="*80)
    import traceback
    traceback.print_exc()
    sys.exit(1)
