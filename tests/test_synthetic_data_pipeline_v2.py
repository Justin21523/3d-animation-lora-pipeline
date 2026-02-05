#!/usr/bin/env python3
"""
Test Suite for Synthetic Data Generation Pipeline v2.0
=======================================================

Tests:
1. Template counts (pose=66, expression=67, action=143)
2. Caption conversion (identity removal, token validation)
3. Balanced sampling (template coverage)
4. End-to-end integration

Usage:
    python tests/test_synthetic_data_pipeline_v2.py
"""

import sys
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "generic" / "training" / "orchestration"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "generic" / "training" / "caption_engines"))


def test_template_counts():
    """Test 1: Verify template counts match expectations."""
    print("\n" + "="*60)
    print("TEST 1: Template Counts")
    print("="*60)

    from vocabulary_generator import (
        POSE_TEMPLATES, EXPRESSION_TEMPLATES, ACTION_TEMPLATES,
        STYLE_VARIATIONS, get_template_stats
    )

    stats = get_template_stats()

    expected = {
        "pose": 66,  # Expanded from 27
        "expression": 67,  # Expanded from 35
        "action": 143,  # Expanded from 35
        "style_variations": 14,
    }

    passed = True
    for key, expected_val in expected.items():
        actual_val = stats.get(key, 0)
        status = "✓" if actual_val == expected_val else "✗"
        if actual_val != expected_val:
            passed = False
        print(f"  {status} {key}: {actual_val} (expected: {expected_val})")

    total = stats["pose"] + stats["expression"] + stats["action"]
    expected_total = 276
    print(f"\n  Total templates: {total} (expected: {expected_total})")

    return passed


def test_caption_conversion_identity_removal():
    """Test 2: Verify caption conversion removes all character identity."""
    print("\n" + "="*60)
    print("TEST 2: Caption Conversion - Identity Removal")
    print("="*60)

    from prompt_to_caption_converter import PromptToCaptionConverter

    converter = PromptToCaptionConverter(
        min_tokens=60,
        max_tokens=225,
        target_tokens=100,
        auto_enhance=True,
    )

    # Test prompts with various identity patterns
    test_cases = [
        {
            "name": "Alberto with parenthetical",
            "prompt": "alberto (Alberto from Pixar Luca, Italian teen, curly brown hair), 3d animation, pixar style, high quality, detailed, full body view standing straight with neutral expression, clean studio lighting, pixar style 3d animated character",
            "character": "alberto",
            "should_not_contain": ["alberto", "italian", "pixar luca", "curly brown hair"],
        },
        {
            "name": "Luca sea monster",
            "prompt": "luca_seamonster, sea monster with vibrant green-purple iridescent scaly skin, turquoise fin-like hair crest, large expressive eyes, webbed hands, 3d animated character, swimming action, underwater lighting",
            "character": "luca_seamonster",
            "should_not_contain": ["luca", "sea monster with", "scaly skin", "webbed hands"],
        },
        {
            "name": "Miguel with film reference",
            "prompt": "miguel (Miguel from Pixar Coco, Mexican boy, dark spiky hair), pixar style 3d animation, playing guitar, warm lighting",
            "character": "miguel",
            "should_not_contain": ["miguel", "mexican", "pixar coco", "dark spiky hair"],
        },
        {
            "name": "Barley Lightfoot elf",
            "prompt": "barley_lightfoot (Barley from Pixar Onward, blue elf teen, purple hair), 3d animation, confident stance with arms crossed",
            "character": "barley_lightfoot",
            "should_not_contain": ["barley", "lightfoot", "elf", "blue skin", "purple hair", "pixar onward"],
        },
        {
            "name": "Giulia with freckles",
            "prompt": "giulia (Giulia from Pixar Luca, Italian girl, red curly hair), 3d render, happy expression with bright smile and freckles",
            "character": "giulia",
            "should_not_contain": ["giulia", "italian girl", "red curly hair", "freckles"],
        },
    ]

    all_passed = True
    for tc in test_cases:
        result = converter.convert(tc["prompt"], character_name=tc["character"])
        caption_lower = result.caption.lower()

        print(f"\n  Test: {tc['name']}")
        print(f"    Original: {tc['prompt'][:80]}...")
        print(f"    Caption:  {result.caption[:80]}...")

        failures = []
        for term in tc["should_not_contain"]:
            if term.lower() in caption_lower:
                failures.append(term)

        if failures:
            print(f"    ✗ FAILED: Still contains: {failures}")
            all_passed = False
        else:
            print(f"    ✓ PASSED: All identity terms removed")

        print(f"    Tokens: {result.token_count_estimate}, Valid: {result.is_valid}")
        print(f"    Removed: {len(result.removed_terms)} terms")

    return all_passed


def test_caption_conversion_preserves_content():
    """Test 3: Verify caption conversion preserves action/pose/expression content."""
    print("\n" + "="*60)
    print("TEST 3: Caption Conversion - Content Preservation")
    print("="*60)

    from prompt_to_caption_converter import PromptToCaptionConverter

    converter = PromptToCaptionConverter(min_tokens=60, max_tokens=225, auto_enhance=True)

    test_cases = [
        {
            "prompt": "alberto, 3d animation, basketball dribbling with athletic stance, court environment, cinematic sports lighting",
            "character": "alberto",
            "should_contain": ["basketball", "dribbling", "athletic", "court", "lighting"],
        },
        {
            "prompt": "luca, pixar style, happy expression with bright smile and joyful eyes, warm studio lighting",
            "character": "luca",
            "should_contain": ["happy", "smile", "joyful", "warm", "lighting"],
        },
        {
            "prompt": "miguel, 3d render, playing guitar with musical expression, performance setting, stage lighting",
            "character": "miguel",
            "should_contain": ["playing guitar", "musical", "performance", "stage", "lighting"],
        },
    ]

    all_passed = True
    for tc in test_cases:
        result = converter.convert(tc["prompt"], character_name=tc["character"])
        caption_lower = result.caption.lower()

        print(f"\n  Prompt: {tc['prompt'][:60]}...")

        missing = []
        for term in tc["should_contain"]:
            if term.lower() not in caption_lower:
                missing.append(term)

        if missing:
            print(f"    ✗ FAILED: Missing content: {missing}")
            print(f"    Caption: {result.caption}")
            all_passed = False
        else:
            print(f"    ✓ PASSED: All content preserved")

    return all_passed


def test_balanced_sampling():
    """Test 4: Verify balanced sampling ensures template coverage."""
    print("\n" + "="*60)
    print("TEST 4: Balanced Sampling")
    print("="*60)

    from vocabulary_generator import (
        sample_balanced_from_templates,
        POSE_TEMPLATES, generate_template_prompts
    )

    # Test with exact coverage
    templates = ["A", "B", "C", "D", "E"]
    samples = sample_balanced_from_templates(templates, num_samples=10, ensure_coverage=True)

    # Count occurrences
    counts = {t: samples.count(t) for t in templates}

    print(f"\n  Test: 5 templates, 10 samples")
    print(f"  Counts: {counts}")

    # Each template should appear at least once (with 10 samples from 5 templates)
    all_covered = all(c >= 1 for c in counts.values())
    print(f"  All templates covered at least once: {all_covered}")

    # Test with real templates
    print(f"\n  Test: 100 prompts from {len(POSE_TEMPLATES)} pose templates")
    prompts = generate_template_prompts(
        character_name="alberto",
        character_description="test",
        lora_type="pose",
        num_prompts=100,
        ensure_template_coverage=True
    )

    # Check unique templates used
    templates_used = set()
    for p in prompts:
        templates_used.add(p["metadata"]["template"])

    coverage_pct = len(templates_used) / len(POSE_TEMPLATES) * 100
    print(f"  Unique templates used: {len(templates_used)}/{len(POSE_TEMPLATES)} ({coverage_pct:.1f}%)")

    passed = all_covered and coverage_pct > 90  # At least 90% coverage
    print(f"  {'✓' if passed else '✗'} Balanced sampling working correctly")

    return passed


def test_token_estimation():
    """Test 5: Verify token estimation is reasonable."""
    print("\n" + "="*60)
    print("TEST 5: Token Estimation")
    print("="*60)

    from prompt_to_caption_converter import PromptToCaptionConverter

    converter = PromptToCaptionConverter()

    # Token estimation uses: words * 1.3 + punctuation + hyphens * 0.5
    # Test expectations based on this formula
    test_cases = [
        # "short text" = 2 words -> 2*1.3 = 2.6 -> 2
        ("short text", 1, 5),
        # 8 words + 2 commas -> 8*1.3 + 2 = 12.4 -> 12
        ("a 3d animated character, standing pose, studio lighting", 10, 20),
        # 22 words + 5 commas -> 22*1.3 + 5 = 33.6 -> 33
        ("full body view standing straight with neutral expression, clean studio lighting, "
         "3d animation, pixar style, high quality, detailed rendering with smooth shading", 30, 45),
    ]

    all_passed = True
    for text, min_expected, max_expected in test_cases:
        tokens = converter._estimate_tokens(text)
        in_range = min_expected <= tokens <= max_expected
        status = "✓" if in_range else "✗"
        if not in_range:
            all_passed = False
        print(f"  {status} '{text[:40]}...' -> {tokens} tokens (expected: {min_expected}-{max_expected})")

    return all_passed


def test_config_loading():
    """Test 6: Verify configuration loads correctly."""
    print("\n" + "="*60)
    print("TEST 6: Configuration Loading")
    print("="*60)

    import yaml

    config_path = PROJECT_ROOT / "configs" / "batch" / "synthetic_data_generation.yaml"

    if not config_path.exists():
        print(f"  ✗ Config file not found: {config_path}")
        return False

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Check key settings
    checks = [
        ("vocabulary_generation.num_prompts_per_type", 100),
        ("vocabulary_generation.ensure_template_coverage", True),
        ("caption_generation.enabled", True),
        ("caption_generation.min_tokens", 60),
        ("caption_generation.max_tokens", 225),
        ("expected_output.total_images_all", 42000),
    ]

    all_passed = True
    for path, expected in checks:
        keys = path.split(".")
        value = config
        for key in keys:
            value = value.get(key)
            if value is None:
                break

        match = value == expected
        status = "✓" if match else "✗"
        if not match:
            all_passed = False
        print(f"  {status} {path}: {value} (expected: {expected})")

    return all_passed


def test_full_prompt_generation():
    """Test 7: End-to-end prompt generation."""
    print("\n" + "="*60)
    print("TEST 7: Full Prompt Generation")
    print("="*60)

    from vocabulary_generator import generate_template_prompts

    test_cases = [
        ("alberto", "pose", 100),
        ("luca", "expression", 100),
        ("miguel", "action", 100),
    ]

    all_passed = True
    for character, lora_type, num_prompts in test_cases:
        prompts = generate_template_prompts(
            character_name=character,
            character_description="test",
            lora_type=lora_type,
            num_prompts=num_prompts,
        )

        correct_count = len(prompts) == num_prompts
        has_metadata = all("metadata" in p for p in prompts)
        has_prompts = all("prompt" in p and len(p["prompt"]) > 50 for p in prompts)

        status = "✓" if (correct_count and has_metadata and has_prompts) else "✗"
        if not (correct_count and has_metadata and has_prompts):
            all_passed = False

        print(f"  {status} {character}/{lora_type}: {len(prompts)} prompts generated")
        print(f"      Sample: {prompts[0]['prompt'][:70]}...")

    return all_passed


def test_consistent_templates_across_characters():
    """Test 7.5: Verify all characters get the same templates for each lora_type."""
    print("\n" + "="*60)
    print("TEST 7.5: Consistent Templates Across Characters")
    print("="*60)

    from vocabulary_generator import generate_template_prompts

    characters = ['alberto', 'luca', 'miguel', 'giulia']
    lora_types = ['pose', 'expression', 'action']

    all_passed = True
    for lora_type in lora_types:
        template_sets = {}
        for char in characters:
            prompts = generate_template_prompts(
                character_name=char,
                character_description="test",
                lora_type=lora_type,
                num_prompts=100,
                consistent_across_characters=True
            )
            template_sets[char] = set(p['metadata']['template'] for p in prompts)

        # Check if all characters have same template set
        reference = template_sets['alberto']
        same = all(template_sets[c] == reference for c in characters)
        status = "✓" if same else "✗"
        if not same:
            all_passed = False
        print(f"  {status} {lora_type}: All {len(characters)} characters have identical templates")

    return all_passed


def test_caption_batch_conversion():
    """Test 8: Batch caption conversion."""
    print("\n" + "="*60)
    print("TEST 8: Batch Caption Conversion")
    print("="*60)

    from vocabulary_generator import generate_template_prompts
    from prompt_to_caption_converter import PromptToCaptionConverter

    converter = PromptToCaptionConverter(min_tokens=60, max_tokens=225, auto_enhance=True)

    # Generate prompts
    prompts = generate_template_prompts(
        character_name="alberto",
        character_description="test",
        lora_type="action",
        num_prompts=20,
    )

    # Convert to captions
    results = converter.convert_batch(prompts, character_name="alberto")

    # Check results
    valid_count = sum(1 for r in results if r["is_valid"])
    avg_tokens = sum(r["token_count"] for r in results) / len(results)

    print(f"  Converted {len(results)} prompts")
    print(f"  Valid captions: {valid_count}/{len(results)} ({100*valid_count/len(results):.1f}%)")
    print(f"  Average tokens: {avg_tokens:.1f}")

    # Show sample
    print(f"\n  Sample conversion:")
    print(f"    Original: {results[0]['original_prompt'][:70]}...")
    print(f"    Caption:  {results[0]['caption'][:70]}...")

    passed = valid_count >= len(results) * 0.9  # At least 90% valid
    print(f"\n  {'✓' if passed else '✗'} Batch conversion working correctly")

    return passed


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*70)
    print("SYNTHETIC DATA PIPELINE v2.0 - TEST SUITE")
    print("="*70)

    tests = [
        ("Template Counts", test_template_counts),
        ("Identity Removal", test_caption_conversion_identity_removal),
        ("Content Preservation", test_caption_conversion_preserves_content),
        ("Balanced Sampling", test_balanced_sampling),
        ("Token Estimation", test_token_estimation),
        ("Configuration Loading", test_config_loading),
        ("Full Prompt Generation", test_full_prompt_generation),
        ("Consistent Templates", test_consistent_templates_across_characters),
        ("Batch Caption Conversion", test_caption_batch_conversion),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed, None))
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"\n  ✗ EXCEPTION: {e}")

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed_count = 0
    for name, passed, error in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        if error:
            status += f" ({error[:50]})"
        print(f"  {status}: {name}")
        if passed:
            passed_count += 1

    print(f"\n  Total: {passed_count}/{len(results)} tests passed")
    print("="*70)

    return passed_count == len(results)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
