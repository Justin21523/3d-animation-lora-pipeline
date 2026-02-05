#!/usr/bin/env python3
"""
Caption Quality Validator for LoRA Training.

Validates and scores captions for training quality:
- Token length validation (40-77 optimal range)
- Trigger word presence and position
- Grammar and formatting checks
- Duplicate content detection
- Quality scoring and recommendations

CPU-only tool with 32-thread parallel processing.
AI_WAREHOUSE 3.0 compliant paths.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CaptionValidation:
    """Validation result for a single caption."""
    file_path: str
    caption: str
    token_count: int
    word_count: int
    has_trigger: bool
    trigger_position: int  # -1 if not found
    quality_score: float  # 0-100
    issues: List[str]
    suggestions: List[str]


@dataclass
class ValidationStats:
    """Overall validation statistics."""
    total_captions: int
    valid_count: int
    invalid_count: int
    avg_token_count: float
    avg_quality_score: float
    common_issues: Dict[str, int]
    missing_trigger_count: int
    too_short_count: int
    too_long_count: int


@dataclass
class ValidationReport:
    """Complete validation report."""
    directory: str
    trigger_word: Optional[str]
    stats: ValidationStats
    validations: List[CaptionValidation]
    recommendations: List[str]

    def to_dict(self) -> Dict:
        return {
            "directory": self.directory,
            "trigger_word": self.trigger_word,
            "stats": asdict(self.stats),
            "validations": [asdict(v) for v in self.validations],
            "recommendations": self.recommendations,
        }


class CaptionQualityValidator:
    """
    Validate caption quality for LoRA training.

    Checks:
    - Token length (optimal: 40-77)
    - Trigger word presence and position
    - Grammar and formatting
    - Duplicate content
    - Common mistakes
    """

    # Quality thresholds
    MIN_TOKENS = 10
    OPTIMAL_MIN_TOKENS = 40
    OPTIMAL_MAX_TOKENS = 77
    MAX_TOKENS = 150

    # Common issues patterns
    ISSUE_PATTERNS = [
        (r'\b(very|really|extremely|absolutely)\s+\1\b', "Repeated intensifier"),
        (r'[.!?]{2,}', "Multiple punctuation"),
        (r'\s{2,}', "Multiple spaces"),
        (r'^[a-z]', "Doesn't start with capital"),
        (r'\b[A-Z]{3,}\b', "ALL CAPS words"),
        (r'(?i)\b(beautiful|gorgeous|stunning|amazing)\b', "Overused adjective"),
        (r'\d{4,}', "Long number sequence"),
    ]

    # Words that shouldn't appear in captions
    FORBIDDEN_WORDS = {
        "anime", "manga", "cartoon",  # Style confusion
        "image", "picture", "photo", "screenshot",  # Meta descriptions
        "hd", "4k", "8k", "high resolution",  # Quality tags
        "masterpiece", "best quality",  # Booru tags
    }

    def __init__(
        self,
        trigger_word: Optional[str] = None,
        num_workers: int = 32,
    ):
        self.trigger_word = trigger_word
        self.num_workers = num_workers

    def validate_caption(
        self,
        caption: str,
        file_path: Optional[str] = None,
    ) -> CaptionValidation:
        """
        Validate a single caption.

        Args:
            caption: Caption text to validate
            file_path: Optional source file path

        Returns:
            CaptionValidation with scores and issues
        """
        issues = []
        suggestions = []

        # Tokenize (simple whitespace + punctuation split)
        words = caption.split()
        word_count = len(words)

        # Estimate token count (rough approximation)
        # SD tokenizer typically splits on spaces and some punctuation
        token_count = self._estimate_tokens(caption)

        # Check trigger word
        has_trigger = False
        trigger_position = -1

        if self.trigger_word:
            trigger_lower = self.trigger_word.lower()
            caption_lower = caption.lower()

            if trigger_lower in caption_lower:
                has_trigger = True
                trigger_position = caption_lower.find(trigger_lower)

                # Check position (should be at start)
                if trigger_position > 20:
                    issues.append("Trigger word not at beginning")
                    suggestions.append(f"Move '{self.trigger_word}' to the start")
            else:
                issues.append("Missing trigger word")
                suggestions.append(f"Add '{self.trigger_word}' at the beginning")

        # Check length
        if token_count < self.MIN_TOKENS:
            issues.append(f"Too short ({token_count} tokens)")
            suggestions.append("Add more descriptive details")
        elif token_count < self.OPTIMAL_MIN_TOKENS:
            issues.append(f"Below optimal length ({token_count} tokens)")
            suggestions.append("Consider adding more context")
        elif token_count > self.MAX_TOKENS:
            issues.append(f"Too long ({token_count} tokens)")
            suggestions.append("Trim to under 77 tokens for best results")
        elif token_count > self.OPTIMAL_MAX_TOKENS:
            issues.append(f"Above optimal length ({token_count} tokens)")

        # Check patterns
        for pattern, issue_name in self.ISSUE_PATTERNS:
            if re.search(pattern, caption):
                issues.append(issue_name)

        # Check forbidden words
        for word in self.FORBIDDEN_WORDS:
            if word.lower() in caption.lower():
                issues.append(f"Contains '{word}' (avoid style/quality tags)")

        # Check for common formatting issues
        if caption != caption.strip():
            issues.append("Leading/trailing whitespace")

        if "  " in caption:
            issues.append("Double spaces")
            suggestions.append("Remove extra spaces")

        if caption.endswith(","):
            issues.append("Ends with comma")

        # Check for repetitive phrases
        words_lower = [w.lower() for w in words]
        word_counts = Counter(words_lower)
        repeated = [w for w, c in word_counts.items() if c > 2 and len(w) > 3]
        if repeated:
            issues.append(f"Repeated words: {', '.join(repeated[:3])}")

        # Calculate quality score
        quality_score = self._calculate_quality_score(
            token_count, has_trigger, trigger_position, issues
        )

        return CaptionValidation(
            file_path=file_path or "",
            caption=caption,
            token_count=token_count,
            word_count=word_count,
            has_trigger=has_trigger,
            trigger_position=trigger_position,
            quality_score=quality_score,
            issues=issues,
            suggestions=suggestions,
        )

    def validate_directory(
        self,
        directory: str | Path,
        caption_extension: str = ".txt",
    ) -> ValidationReport:
        """
        Validate all captions in a directory.

        Args:
            directory: Directory containing caption files
            caption_extension: Caption file extension

        Returns:
            ValidationReport with all results
        """
        directory = Path(directory)
        caption_files = list(directory.glob(f"*{caption_extension}"))

        logger.info(f"Validating {len(caption_files)} captions in {directory}")

        validations = []

        # Parallel validation
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {}
            for caption_file in caption_files:
                try:
                    caption = caption_file.read_text(encoding="utf-8").strip()
                    future = executor.submit(
                        self.validate_caption,
                        caption,
                        str(caption_file),
                    )
                    futures[future] = caption_file
                except Exception as e:
                    logger.warning(f"Could not read {caption_file}: {e}")

            for future in as_completed(futures):
                try:
                    validation = future.result()
                    validations.append(validation)
                except Exception as e:
                    logger.warning(f"Validation failed: {e}")

        # Calculate statistics
        stats = self._calculate_stats(validations)

        # Generate recommendations
        recommendations = self._generate_recommendations(stats, validations)

        return ValidationReport(
            directory=str(directory),
            trigger_word=self.trigger_word,
            stats=stats,
            validations=validations,
            recommendations=recommendations,
        )

    def fix_common_issues(
        self,
        caption: str,
    ) -> Tuple[str, List[str]]:
        """
        Auto-fix common caption issues.

        Args:
            caption: Original caption

        Returns:
            Tuple of (fixed caption, list of changes made)
        """
        changes = []
        fixed = caption

        # Fix whitespace
        if fixed != fixed.strip():
            fixed = fixed.strip()
            changes.append("Removed leading/trailing whitespace")

        # Fix double spaces
        if "  " in fixed:
            fixed = re.sub(r'\s+', ' ', fixed)
            changes.append("Normalized spaces")

        # Fix multiple punctuation
        fixed = re.sub(r'([.!?])\1+', r'\1', fixed)
        if fixed != caption:
            changes.append("Fixed multiple punctuation")

        # Add trigger word if missing
        if self.trigger_word:
            if self.trigger_word.lower() not in fixed.lower():
                fixed = f"{self.trigger_word}, {fixed}"
                changes.append(f"Added trigger word '{self.trigger_word}'")

        # Fix capitalization
        if fixed and fixed[0].islower():
            fixed = fixed[0].upper() + fixed[1:]
            changes.append("Capitalized first letter")

        # Remove trailing comma
        if fixed.endswith(","):
            fixed = fixed[:-1]
            changes.append("Removed trailing comma")

        return fixed, changes

    def batch_fix(
        self,
        directory: str | Path,
        caption_extension: str = ".txt",
        dry_run: bool = True,
    ) -> Dict[str, List[str]]:
        """
        Fix common issues in all captions.

        Args:
            directory: Directory containing caption files
            caption_extension: Caption file extension
            dry_run: If True, don't write changes

        Returns:
            Dict mapping file paths to changes made
        """
        directory = Path(directory)
        caption_files = list(directory.glob(f"*{caption_extension}"))

        all_changes = {}

        for caption_file in caption_files:
            try:
                original = caption_file.read_text(encoding="utf-8").strip()
                fixed, changes = self.fix_common_issues(original)

                if changes:
                    all_changes[str(caption_file)] = changes

                    if not dry_run:
                        caption_file.write_text(fixed, encoding="utf-8")
                        logger.info(f"Fixed {caption_file.name}: {', '.join(changes)}")

            except Exception as e:
                logger.warning(f"Could not process {caption_file}: {e}")

        return all_changes

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        # CLIP tokenizer roughly splits on spaces and punctuation
        # This is a rough estimate; actual count may vary
        tokens = re.findall(r'\w+|[^\w\s]', text.lower())
        return len(tokens)

    def _calculate_quality_score(
        self,
        token_count: int,
        has_trigger: bool,
        trigger_position: int,
        issues: List[str],
    ) -> float:
        """Calculate quality score 0-100."""
        score = 100.0

        # Token count scoring
        if token_count < self.MIN_TOKENS:
            score -= 30
        elif token_count < self.OPTIMAL_MIN_TOKENS:
            score -= 15
        elif token_count > self.MAX_TOKENS:
            score -= 25
        elif token_count > self.OPTIMAL_MAX_TOKENS:
            score -= 10

        # Trigger word scoring
        if not has_trigger and self.trigger_word:
            score -= 20
        elif trigger_position > 20:
            score -= 10

        # Issue penalties
        issue_penalty = min(40, len(issues) * 5)
        score -= issue_penalty

        return max(0, min(100, score))

    def _calculate_stats(
        self,
        validations: List[CaptionValidation],
    ) -> ValidationStats:
        """Calculate overall statistics."""
        if not validations:
            return ValidationStats(
                total_captions=0,
                valid_count=0,
                invalid_count=0,
                avg_token_count=0,
                avg_quality_score=0,
                common_issues={},
                missing_trigger_count=0,
                too_short_count=0,
                too_long_count=0,
            )

        # Count issues
        issue_counter = Counter()
        for v in validations:
            for issue in v.issues:
                issue_counter[issue] += 1

        # Calculate averages
        avg_tokens = sum(v.token_count for v in validations) / len(validations)
        avg_score = sum(v.quality_score for v in validations) / len(validations)

        # Count categories
        valid_count = sum(1 for v in validations if v.quality_score >= 70)
        missing_trigger = sum(1 for v in validations if not v.has_trigger and self.trigger_word)
        too_short = sum(1 for v in validations if v.token_count < self.OPTIMAL_MIN_TOKENS)
        too_long = sum(1 for v in validations if v.token_count > self.OPTIMAL_MAX_TOKENS)

        return ValidationStats(
            total_captions=len(validations),
            valid_count=valid_count,
            invalid_count=len(validations) - valid_count,
            avg_token_count=avg_tokens,
            avg_quality_score=avg_score,
            common_issues=dict(issue_counter.most_common(10)),
            missing_trigger_count=missing_trigger,
            too_short_count=too_short,
            too_long_count=too_long,
        )

    def _generate_recommendations(
        self,
        stats: ValidationStats,
        validations: List[CaptionValidation],
    ) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []

        if stats.missing_trigger_count > stats.total_captions * 0.1:
            recommendations.append(
                f"Add trigger word to {stats.missing_trigger_count} captions"
            )

        if stats.too_short_count > stats.total_captions * 0.2:
            recommendations.append(
                f"Expand {stats.too_short_count} short captions with more detail"
            )

        if stats.too_long_count > stats.total_captions * 0.1:
            recommendations.append(
                f"Trim {stats.too_long_count} long captions to under 77 tokens"
            )

        if stats.avg_quality_score < 70:
            recommendations.append(
                "Overall caption quality is low - review and improve captions"
            )

        for issue, count in stats.common_issues.items():
            if count > stats.total_captions * 0.2:
                recommendations.append(f"Fix '{issue}' in {count} captions")

        return recommendations


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate caption quality for LoRA training"
    )
    parser.add_argument(
        "directory",
        help="Directory containing caption files"
    )
    parser.add_argument(
        "--trigger", "-t",
        help="Trigger word to check for"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Auto-fix common issues"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show fixes without applying (use with --fix)"
    )
    parser.add_argument(
        "--extension", "-e",
        default=".txt",
        help="Caption file extension (default: .txt)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output JSON report path"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    validator = CaptionQualityValidator(
        trigger_word=args.trigger,
    )

    if args.fix:
        changes = validator.batch_fix(
            args.directory,
            caption_extension=args.extension,
            dry_run=args.dry_run,
        )

        print(f"\n{'DRY RUN: ' if args.dry_run else ''}Changes to make:")
        for file_path, file_changes in changes.items():
            print(f"  {Path(file_path).name}: {', '.join(file_changes)}")

        print(f"\nTotal files to {'fix' if args.dry_run else 'fixed'}: {len(changes)}")

    else:
        report = validator.validate_directory(
            args.directory,
            caption_extension=args.extension,
        )

        print(f"\nCaption Validation Report")
        print(f"=" * 60)
        print(f"Directory: {report.directory}")
        print(f"Trigger word: {report.trigger_word or 'None'}")
        print(f"\nStatistics:")
        print(f"  Total captions: {report.stats.total_captions}")
        print(f"  Valid (score >= 70): {report.stats.valid_count}")
        print(f"  Invalid: {report.stats.invalid_count}")
        print(f"  Average token count: {report.stats.avg_token_count:.1f}")
        print(f"  Average quality score: {report.stats.avg_quality_score:.1f}")

        if report.stats.common_issues:
            print(f"\nCommon Issues:")
            for issue, count in report.stats.common_issues.items():
                print(f"  {issue}: {count}")

        if report.recommendations:
            print(f"\nRecommendations:")
            for rec in report.recommendations:
                print(f"  - {rec}")

        if args.output:
            output_path = Path(args.output)
            with open(output_path, "w") as f:
                json.dump(report.to_dict(), f, indent=2)
            print(f"\nReport saved to: {output_path}")
