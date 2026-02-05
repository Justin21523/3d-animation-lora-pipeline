#!/usr/bin/env python3
"""Check actual dataset repeats from directory structure"""

from pathlib import Path

def check_repeats():
    base_dir = Path("/mnt/data/ai_data/datasets/3d-anime")

    print("=" * 70)
    print("Dataset Repeats Check (from actual directories)")
    print("=" * 70 + "\n")

    results = []

    for film_dir in sorted(base_dir.iterdir()):
        if not film_dir.is_dir():
            continue

        training_dir = film_dir / "lora_data/training_data"
        if not training_dir.exists():
            continue

        for char_dir in sorted(training_dir.iterdir()):
            if not char_dir.is_dir() or not char_dir.name.endswith("_identity"):
                continue

            # Check subdirectories for repeat count
            subdirs = list(char_dir.iterdir())
            if subdirs:
                # Get first subdir and extract repeat number
                first_subdir = subdirs[0].name
                if "_" in first_subdir:
                    try:
                        repeats = int(first_subdir.split("_")[0])
                        char_name = char_dir.name.replace("_identity", "")

                        # Count images
                        image_count = len(list(char_dir.rglob("*.png"))) + len(list(char_dir.rglob("*.jpg")))

                        results.append((film_dir.name, char_name, repeats, image_count))
                    except ValueError:
                        pass

    # Print results
    for film, char, repeats, count in sorted(results):
        steps_per_epoch = count * repeats
        print(f"{film:15s} {char:25s} {count:4d} images × {repeats:2d} repeats = {steps_per_epoch:5d} steps/epoch")

    print("\n" + "=" * 70)
    print(f"Total characters: {len(results)}")
    print("=" * 70)

    return results

if __name__ == "__main__":
    check_repeats()
