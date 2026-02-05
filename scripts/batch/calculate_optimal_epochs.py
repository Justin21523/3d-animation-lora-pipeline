#!/usr/bin/env python3
"""Calculate optimal epochs to keep total steps ≤ 35000"""

# Character data: (film, char_id, char_name, image_count, repeats)
CHARACTERS = [
    ("luca", "alberto", "Alberto Scorfano", 509, 10),
    ("luca", "giulia", "Giulia Marcovaldo", 546, 15),
    ("coco", "miguel", "Miguel Rivera", 449, 10),
    ("elio", "elio", "Elio Solis", 538, 7),
    ("elio", "bryce", "Bryce Markwell", 201, 20),
    ("elio", "caleb", "Caleb", 195, 20),
    ("elio", "glordon", "Glordon", 201, 20),
    ("onward", "ian_lightfoot", "Ian Lightfoot", 343, 10),
    ("onward", "barley_lightfoot", "Barley Lightfoot", 254, 10),
    ("up", "russell", "Russell", 243, 15),
    ("orion", "orion", "Orion", 261, 15),
    ("turning-red", "tyler", "Tyler", 276, 15),
]

MAX_TOTAL_STEPS = 35000

print("=" * 80)
print("Optimal Epochs Calculation (Max Total Steps: 35,000)")
print("=" * 80 + "\n")

results = []

for film, char_id, char_name, image_count, repeats in CHARACTERS:
    steps_per_epoch = image_count * repeats

    # Calculate max epochs to stay under 35000 steps
    max_epochs = MAX_TOTAL_STEPS // steps_per_epoch

    # Prefer epochs that are multiples of 2 for save checkpoints
    optimal_epochs = max(6, (max_epochs // 2) * 2)

    total_steps = steps_per_epoch * optimal_epochs

    results.append((film, char_id, char_name, image_count, repeats, optimal_epochs, steps_per_epoch, total_steps))

    print(f"{char_name:25s} {image_count:3d} × {repeats:2d} = {steps_per_epoch:5d} steps/epoch")
    print(f"  → {optimal_epochs:2d} epochs = {total_steps:6d} total steps")
    print()

print("=" * 80)
print("Summary for generate_sdxl_configs.py:")
print("=" * 80 + "\n")

for film, char_id, char_name, image_count, repeats, epochs, _, _ in results:
    print(f'    ("{film}", "{char_id}", "{char_name}", {image_count}, {repeats}, {epochs}),')
