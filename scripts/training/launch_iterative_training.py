#!/usr/bin/env python3
"""
Launch 14-hour iterative LoRA training for luca_human and alberto_human

Usage:
    python scripts/training/launch_iterative_training.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.training.iterative_lora_optimizer import IterativeTrainingOrchestrator

# Configuration
characters = ['luca_human', 'alberto_human']
base_dataset_dir = Path('/mnt/data/ai_data/datasets/3d-anime/luca/curated_dataset')
base_model_path = '/mnt/c/AI_LLM_projects/ai_warehouse/tool-caches/huggingface/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14'
output_base_dir = Path('/mnt/data/ai_data/models/lora/luca/iterative_overnight_v5')
sd_scripts_dir = Path('/mnt/c/AI_LLM_projects/ai_warehouse/sd-scripts')

# 🔄 CONTINUATION FROM ITERATION 3 BEST MODELS
# Load the best models from iteration 3 as starting point
pretrained_lora_dir = Path('/mnt/data/ai_data/models/lora/luca/BEST_MODELS')

max_iterations = 999  # High limit - time budget will be the real constraint
time_limit_hours = 14.0

print("="*70)
print("ITERATIVE LORA TRAINING - TIME-DRIVEN SYSTEM")
print("="*70)
print(f"Characters: {', '.join(characters)}")
print(f"Dataset: {base_dataset_dir}")
print(f"Base Model: SD v1.5")
print(f"Output: {output_base_dir}")
print(f"Max Iterations: {max_iterations} (time-limited)")
print(f"Time Budget: {time_limit_hours} hours (PRIMARY CONSTRAINT)")
print(f"Expected: ~16-18 iterations within time budget")
print("="*70)
print()

# Verify dataset directories exist
for char in characters:
    char_dir = base_dataset_dir / char / 'images'
    if not char_dir.exists():
        print(f"ERROR: Dataset not found for {char}: {char_dir}")
        sys.exit(1)

    # Count images
    images = list(char_dir.glob('*.png'))
    captions = list(char_dir.glob('*.txt'))
    print(f"✓ {char}: {len(images)} images, {len(captions)} captions")

print()

# Create orchestrator
orchestrator = IterativeTrainingOrchestrator(
    characters=characters,
    base_dataset_dir=base_dataset_dir,
    base_model_path=base_model_path,
    output_base_dir=output_base_dir,
    sd_scripts_dir=sd_scripts_dir,
    max_iterations=max_iterations,
    time_limit_hours=time_limit_hours,
    pretrained_lora_dir=pretrained_lora_dir  # Continue from iteration 3 best models
)

# Start training
print("\n" + "="*70)
print("STARTING TRAINING - This will take approximately 14 hours")
print("="*70)
print()

try:
    orchestrator.run_optimization()
    print("\n" + "="*70)
    print("✓ TRAINING COMPLETED SUCCESSFULLY")
    print("="*70)
except KeyboardInterrupt:
    print("\n" + "="*70)
    print("! TRAINING INTERRUPTED BY USER")
    print("="*70)
except Exception as e:
    print("\n" + "="*70)
    print(f"✗ TRAINING FAILED: {e}")
    print("="*70)
    import traceback
    traceback.print_exc()
    sys.exit(1)
