#!/usr/bin/env python3
"""
Recalculate SDXL Configs with Optimal Epochs/Repeats
Strategy: Keep higher epochs (8/10/12), adjust repeats to stay under 35,000 steps
"""

from pathlib import Path

# Target configuration: (film, char_id, char_name, image_count, target_epochs)
# We'll calculate optimal repeats to stay under 35,000 steps
CHARACTERS = [
    ("luca", "alberto", "Alberto Scorfano", 509, 10),
    ("luca", "giulia", "Giulia Marcovaldo", 546, 8),
    ("coco", "miguel", "Miguel Rivera", 449, 10),
    ("elio", "elio", "Elio Solis", 538, 8),
    ("elio", "bryce", "Bryce Markwell", 201, 12),
    ("elio", "caleb", "Caleb", 195, 12),
    ("elio", "glordon", "Glordon", 201, 12),
    ("onward", "ian_lightfoot", "Ian Lightfoot", 343, 10),
    ("onward", "barley_lightfoot", "Barley Lightfoot", 254, 12),
    ("up", "russell", "Russell", 243, 12),
    ("orion", "orion", "Orion", 261, 10),
    ("turning-red", "tyler", "Tyler", 276, 10),
]

MAX_TOTAL_STEPS = 35000

def calculate_optimal_repeats(image_count: int, epochs: int, max_steps: int = MAX_TOTAL_STEPS):
    """Calculate optimal repeats to stay under max_steps"""
    max_repeats = max_steps // (image_count * epochs)
    return max_repeats

print("=" * 100)
print("SDXL CONFIG RECALCULATION (High Epochs + Adjusted Repeats)")
print("=" * 100)
print(f"{'Character':<25} {'Images':<8} {'Epochs':<8} {'Repeats':<10} {'Steps/Epoch':<12} {'Total Steps':<12}")
print("=" * 100)

results = []
for film, char_id, char_name, image_count, target_epochs in CHARACTERS:
    optimal_repeats = calculate_optimal_repeats(image_count, target_epochs)
    steps_per_epoch = image_count * optimal_repeats
    total_steps = steps_per_epoch * target_epochs

    results.append((film, char_id, char_name, image_count, optimal_repeats, target_epochs, steps_per_epoch, total_steps))

    status = "✅" if total_steps <= MAX_TOTAL_STEPS else "❌"
    print(f"{status} {char_name:<23} {image_count:<8} {target_epochs:<8} {optimal_repeats:<10} {steps_per_epoch:<12} {total_steps:<12}")

print("=" * 100)
print(f"\n✅ All configs stay under {MAX_TOTAL_STEPS:,} total steps\n")

# Generate Python tuples for generate_sdxl_configs.py
print("=" * 100)
print("CHARACTERS list for generate_sdxl_configs.py:")
print("=" * 100)
for film, char_id, char_name, image_count, repeats, epochs, _, _ in results:
    print(f'    ("{film}", "{char_id}", "{char_name}", {image_count}, {repeats}, {epochs}),')
print("=" * 100)
