#!/usr/bin/env python3
"""
Analyze and validate captions across training datasets.

CPU-ONLY TOOL - No GPU required, uses 32-thread parallel processing.

This tool analyzes caption files for:
- Token count distribution
- Trigger word verification
- Duplicate caption detection
- Style consistency analysis
- Quality scoring

Usage:
    python scripts/utils/caption_analyzer.py
    python scripts/utils/caption_analyzer.py --dataset-dir /path/to/training_data
    python scripts/utils/caption_analyzer.py --check-trigger "character_name" --output report.json

Author: Justin Lu
Date: 2025-12-02
"""

import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
import argparse
import json
import time
import re
import hashlib
from collections import Counter, defaultdict
from datetime import datetime

# Add parent directory for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# Common style terms for 2D/3D animation
STYLE_TERMS_2D = {
    "2d", "animated", "cartoon", "cel", "shaded", "western", "animation",
    "drawn", "illustrated", "anime", "lineart", "flat"
}

STYLE_TERMS_3D = {
    "3d", "pixar", "rendered", "cgi", "smooth", "shading", "lighting",
    "studio", "realistic", "digital", "pbr"
}

# Common caption quality issues
QUALITY_PATTERNS = {
    "too_short": r"^.{0,20}$",
    "no_description": r"^[a-z_]+$",
    "placeholder": r"(placeholder|todo|fixme|xxx)",
    "url_present": r"https?://",
    "excessive_punctuation": r"[!?]{3,}",
    "all_caps": r"^[A-Z\s,]+$",
}


@dataclass
class CaptionInfo:
    """Information about a single caption."""
    path: str
    text: str
    token_count: int = 0
    word_count: int = 0
    char_count: int = 0
    has_trigger: bool = False
    style_type: str = "unknown"  # "2d", "3d", or "unknown"
    quality_issues: List[str] = field(default_factory=list)
    hash: str = ""


@dataclass
class CharacterCaptionStats:
    """Caption statistics for a character."""
    name: str
    caption_count: int = 0
    avg_tokens: float = 0.0
    min_tokens: int = 0
    max_tokens: int = 0
    trigger_present_count: int = 0
    trigger_rate: float = 0.0
    style_consistency: str = "unknown"
    duplicate_count: int = 0
    quality_score: float = 0.0
    issues: List[str] = field(default_factory=list)
    token_distribution: Dict[str, int] = field(default_factory=dict)
    word_frequency: Dict[str, int] = field(default_factory=dict)


def tokenize_caption(text: str) -> List[str]:
    """
    Tokenize caption text (simple whitespace + punctuation).

    Returns:
        List of tokens
    """
    # Replace common punctuation with spaces
    for char in '.,!?;:()[]{}"\'-':
        text = text.replace(char, ' ')
    return text.split()


def estimate_sd_tokens(text: str) -> int:
    """
    Estimate Stable Diffusion token count.
    SD uses CLIP tokenizer, roughly 1 token per 4 characters + 1 per word.

    Returns:
        Estimated token count
    """
    words = tokenize_caption(text)
    # Rough estimate: each word is 1-2 tokens, plus punctuation
    return len(words) + len(text) // 8


def detect_style(text: str) -> str:
    """
    Detect animation style from caption text.

    Returns:
        "2d", "3d", or "mixed"
    """
    text_lower = text.lower()

    has_2d = any(term in text_lower for term in STYLE_TERMS_2D)
    has_3d = any(term in text_lower for term in STYLE_TERMS_3D)

    if has_2d and has_3d:
        return "mixed"
    elif has_3d:
        return "3d"
    elif has_2d:
        return "2d"
    else:
        return "unknown"


def check_quality_issues(text: str) -> List[str]:
    """
    Check for quality issues in caption text.

    Returns:
        List of issue descriptions
    """
    issues = []

    for issue_name, pattern in QUALITY_PATTERNS.items():
        if re.search(pattern, text, re.IGNORECASE):
            issues.append(issue_name)

    # Check for very long captions (>100 tokens is problematic)
    token_count = estimate_sd_tokens(text)
    if token_count > 100:
        issues.append("too_long")
    elif token_count < 10:
        issues.append("too_short")

    # Check for repeated words
    words = tokenize_caption(text.lower())
    word_counts = Counter(words)
    for word, count in word_counts.items():
        if count > 3 and len(word) > 3:
            issues.append(f"word_repeated:{word}")
            break

    return issues


def analyze_caption_file(
    caption_path: Path,
    trigger_word: Optional[str] = None
) -> CaptionInfo:
    """
    Analyze a single caption file.

    Args:
        caption_path: Path to caption file
        trigger_word: Expected trigger word

    Returns:
        CaptionInfo with analysis results
    """
    info = CaptionInfo(
        path=str(caption_path),
        text=""
    )

    try:
        text = caption_path.read_text(encoding='utf-8', errors='ignore').strip()
        info.text = text
        info.char_count = len(text)
        info.word_count = len(tokenize_caption(text))
        info.token_count = estimate_sd_tokens(text)
        info.style_type = detect_style(text)
        info.quality_issues = check_quality_issues(text)

        # Check trigger word
        if trigger_word:
            info.has_trigger = trigger_word.lower() in text.lower()

        # Compute hash for duplicate detection
        info.hash = hashlib.md5(text.lower().encode()).hexdigest()

    except Exception as e:
        info.quality_issues.append(f"read_error:{str(e)[:30]}")

    return info


def analyze_character_captions(
    char_dir: Path,
    trigger_word: Optional[str] = None,
    num_threads: int = 8
) -> CharacterCaptionStats:
    """
    Analyze all captions for a character.

    Args:
        char_dir: Character directory
        trigger_word: Expected trigger word
        num_threads: Threads for parallel processing

    Returns:
        CharacterCaptionStats
    """
    stats = CharacterCaptionStats(name=char_dir.name)

    # Find caption files
    caption_files = list(char_dir.glob("*.txt"))
    stats.caption_count = len(caption_files)

    if not caption_files:
        stats.issues.append("No caption files found")
        return stats

    # Analyze in parallel (within character)
    captions: List[CaptionInfo] = []

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {
            executor.submit(analyze_caption_file, f, trigger_word): f
            for f in caption_files
        }

        for future in as_completed(futures):
            try:
                info = future.result()
                captions.append(info)
            except Exception:
                pass

    if not captions:
        stats.issues.append("Failed to analyze captions")
        return stats

    # Compute statistics
    token_counts = [c.token_count for c in captions]
    stats.avg_tokens = sum(token_counts) / len(token_counts)
    stats.min_tokens = min(token_counts)
    stats.max_tokens = max(token_counts)

    # Token distribution buckets
    token_buckets = Counter()
    for tc in token_counts:
        if tc < 20:
            token_buckets["<20"] += 1
        elif tc < 40:
            token_buckets["20-40"] += 1
        elif tc < 60:
            token_buckets["40-60"] += 1
        elif tc < 77:
            token_buckets["60-77"] += 1
        else:
            token_buckets[">77"] += 1
    stats.token_distribution = dict(token_buckets)

    # Trigger word analysis
    if trigger_word:
        trigger_count = sum(1 for c in captions if c.has_trigger)
        stats.trigger_present_count = trigger_count
        stats.trigger_rate = trigger_count / len(captions) if captions else 0

    # Style consistency
    style_counts = Counter(c.style_type for c in captions)
    if len(style_counts) == 1:
        stats.style_consistency = list(style_counts.keys())[0]
    elif style_counts:
        dominant = style_counts.most_common(1)[0]
        if dominant[1] / len(captions) > 0.8:
            stats.style_consistency = f"mostly_{dominant[0]}"
        else:
            stats.style_consistency = "mixed"

    # Duplicate detection
    hash_counts = Counter(c.hash for c in captions)
    stats.duplicate_count = sum(1 for h, c in hash_counts.items() if c > 1)

    # Word frequency (top 30)
    all_words: List[str] = []
    for c in captions:
        all_words.extend(tokenize_caption(c.text.lower()))
    word_freq = Counter(all_words)
    # Filter stopwords
    stopwords = {'a', 'an', 'the', 'is', 'are', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'and', 'or'}
    stats.word_frequency = {
        w: f for w, f in word_freq.most_common(50)
        if w not in stopwords and len(w) > 2
    }

    # Quality score (0-100)
    score = 100.0

    # Penalize for issues
    issue_counts = Counter()
    for c in captions:
        issue_counts.update(c.quality_issues)

    total_issues = sum(issue_counts.values())
    issue_rate = total_issues / len(captions) if captions else 1
    score -= min(30, issue_rate * 10)

    # Penalize for trigger word missing
    if trigger_word and stats.trigger_rate < 0.9:
        score -= (1 - stats.trigger_rate) * 20

    # Penalize for duplicates
    dup_rate = stats.duplicate_count / len(captions) if captions else 0
    score -= min(20, dup_rate * 50)

    # Penalize for inconsistent token length
    if token_counts:
        token_std = (sum((t - stats.avg_tokens) ** 2 for t in token_counts) / len(token_counts)) ** 0.5
        if token_std > 20:
            score -= 10

    stats.quality_score = max(0, min(100, score))

    # Collect issues
    if stats.avg_tokens > 77:
        stats.issues.append(f"Average tokens too high ({stats.avg_tokens:.1f} > 77)")
    if stats.avg_tokens < 20:
        stats.issues.append(f"Average tokens too low ({stats.avg_tokens:.1f} < 20)")
    if trigger_word and stats.trigger_rate < 0.9:
        stats.issues.append(f"Trigger word missing in {(1-stats.trigger_rate)*100:.0f}% of captions")
    if stats.duplicate_count > 0:
        stats.issues.append(f"{stats.duplicate_count} duplicate captions")
    if issue_counts:
        for issue, count in issue_counts.most_common(3):
            if count > len(captions) * 0.1:
                stats.issues.append(f"{issue}: {count} occurrences")

    return stats


def analyze_all_captions(
    dataset_root: Path,
    trigger_words: Optional[Dict[str, str]] = None,
    animation_type: str = "2d",
    num_threads: int = 32,
    verbose: bool = False
) -> Dict:
    """
    Analyze captions across all character datasets.

    Args:
        dataset_root: Root directory for training data
        trigger_words: Dict mapping character name to trigger word
        animation_type: "2d" or "3d"
        num_threads: Number of threads
        verbose: Print detailed output

    Returns:
        Analysis report dictionary
    """
    print(f"Analyzing captions in: {dataset_root}")
    start_time = time.time()

    # Find character directories
    type_dir = dataset_root / f"{animation_type}_characters"
    if not type_dir.exists():
        type_dir = dataset_root

    char_dirs = [d for d in type_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]

    print(f"Found {len(char_dirs)} character directories")

    # Prepare trigger words
    trigger_words = trigger_words or {}

    # Analyze characters in parallel
    all_stats: List[CharacterCaptionStats] = []

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {}
        for char_dir in char_dirs:
            trigger = trigger_words.get(char_dir.name, char_dir.name.replace('_', ' '))
            futures[executor.submit(
                analyze_character_captions,
                char_dir,
                trigger,
                max(4, num_threads // 8)
            )] = char_dir

        for future in as_completed(futures):
            try:
                stats = future.result()
                all_stats.append(stats)
                if verbose:
                    print(f"  Analyzed: {stats.name} ({stats.caption_count} captions, score: {stats.quality_score:.0f})")
            except Exception as e:
                if verbose:
                    char_dir = futures[future]
                    print(f"  Error: {char_dir.name}: {e}")

    elapsed = time.time() - start_time

    # Compile report
    total_captions = sum(s.caption_count for s in all_stats)
    avg_score = sum(s.quality_score for s in all_stats) / len(all_stats) if all_stats else 0

    # Aggregate token distribution
    global_token_dist = Counter()
    for s in all_stats:
        global_token_dist.update(s.token_distribution)

    # Aggregate word frequency
    global_word_freq = Counter()
    for s in all_stats:
        global_word_freq.update(s.word_frequency)

    report = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "dataset_root": str(dataset_root),
            "animation_type": animation_type,
            "elapsed_seconds": round(elapsed, 2)
        },
        "summary": {
            "total_characters": len(all_stats),
            "total_captions": total_captions,
            "average_quality_score": round(avg_score, 1),
            "characters_with_issues": len([s for s in all_stats if s.issues]),
            "total_duplicates": sum(s.duplicate_count for s in all_stats),
            "global_token_distribution": dict(global_token_dist),
            "top_words": dict(global_word_freq.most_common(50))
        },
        "characters": {
            s.name: {
                "caption_count": s.caption_count,
                "avg_tokens": round(s.avg_tokens, 1),
                "min_tokens": s.min_tokens,
                "max_tokens": s.max_tokens,
                "trigger_rate": round(s.trigger_rate, 2) if s.trigger_present_count > 0 else None,
                "style_consistency": s.style_consistency,
                "quality_score": round(s.quality_score, 1),
                "duplicate_count": s.duplicate_count,
                "issues": s.issues,
                "token_distribution": s.token_distribution
            }
            for s in sorted(all_stats, key=lambda x: x.quality_score, reverse=True)
        },
        "rankings": {
            "by_quality": [(s.name, round(s.quality_score, 1)) for s in sorted(all_stats, key=lambda x: x.quality_score, reverse=True)[:20]],
            "worst_quality": [(s.name, round(s.quality_score, 1)) for s in sorted(all_stats, key=lambda x: x.quality_score)[:10]],
            "most_issues": [(s.name, len(s.issues)) for s in sorted(all_stats, key=lambda x: len(x.issues), reverse=True) if s.issues][:10]
        }
    }

    return report


def print_summary(report: Dict) -> None:
    """Print summary to console."""
    print("\n" + "=" * 70)
    print("Caption Analysis Summary")
    print("=" * 70)

    s = report['summary']
    print(f"\nTotal Characters: {s['total_characters']}")
    print(f"Total Captions: {s['total_captions']:,}")
    print(f"Average Quality Score: {s['average_quality_score']:.1f}/100")
    print(f"Characters with Issues: {s['characters_with_issues']}")
    print(f"Total Duplicates: {s['total_duplicates']}")

    print("\n" + "-" * 70)
    print("Token Distribution:")
    for bucket, count in sorted(s['global_token_distribution'].items()):
        pct = count / s['total_captions'] * 100 if s['total_captions'] > 0 else 0
        bar = "#" * int(pct / 2)
        print(f"  {bucket:>6}: {count:>6} ({pct:>5.1f}%) {bar}")

    print("\n" + "-" * 70)
    print("Top 10 Quality Scores:")
    print(f"{'Rank':<6} {'Character':<30} {'Score':>10}")
    print("-" * 46)
    for i, (name, score) in enumerate(report['rankings']['by_quality'][:10], 1):
        print(f"{i:<6} {name:<30} {score:>10.1f}")

    if report['rankings']['worst_quality']:
        print("\n" + "-" * 70)
        print("Characters Needing Attention:")
        for name, score in report['rankings']['worst_quality'][:5]:
            char = report['characters'].get(name, {})
            issues = char.get('issues', [])[:2]
            print(f"  {name}: score {score:.1f}")
            for issue in issues:
                print(f"    - {issue}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and validate captions across training datasets"
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="/mnt/data/training/lora",
        help="Root directory for training data"
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["2d", "3d"],
        default="2d",
        help="Animation type (default: 2d)"
    )
    parser.add_argument(
        "--check-trigger",
        type=str,
        help="Check for specific trigger word across all characters"
    )
    parser.add_argument(
        "--trigger-config",
        type=str,
        help="JSON file mapping character names to trigger words"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=32,
        help="Number of threads (default: 32)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Save JSON report to file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output"
    )

    args = parser.parse_args()

    # Load trigger words if provided
    trigger_words = {}
    if args.trigger_config:
        try:
            with open(args.trigger_config) as f:
                trigger_words = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load trigger config: {e}")

    if args.check_trigger:
        # Apply same trigger to all
        pass  # Will use default (character name)

    # Run analysis
    report = analyze_all_captions(
        Path(args.dataset_root),
        trigger_words,
        args.type,
        args.threads,
        args.verbose
    )

    # Print summary
    print_summary(report)

    # Save if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
