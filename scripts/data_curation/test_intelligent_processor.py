#!/usr/bin/env python3
"""
Quick test script for Intelligent Frame Processor

Tests:
1. Decision engine loads correctly
2. Strategy configuration loads
3. Frame analysis works
4. Strategy selection logic works
5. Basic execution (with placeholders)
"""

import sys
from pathlib import Path
import numpy as np
import cv2

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.frame_decision_engine import FrameDecisionEngine
from scripts.data_curation.intelligent_frame_processor import IntelligentFrameProcessor


def create_test_frames(output_dir: Path, num_frames: int = 5) -> list:
    """Create synthetic test frames with different characteristics"""
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = []

    for i in range(num_frames):
        # Create different test patterns
        if i == 0:
            # Simple background, good lighting (should trigger keep_full)
            img = np.full((512, 512, 3), (200, 210, 220), dtype=np.uint8)
            cv2.circle(img, (256, 256), 100, (100, 100, 100), -1)
        elif i == 1:
            # Complex background (should trigger segment)
            img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        elif i == 2:
            # Low quality / blurry (should trigger enhance_segment)
            img = np.full((512, 512, 3), (150, 150, 150), dtype=np.uint8)
            img = cv2.GaussianBlur(img, (51, 51), 0)
        elif i == 3:
            # Good for occlusion creation
            img = np.full((512, 512, 3), (180, 190, 200), dtype=np.uint8)
            cv2.rectangle(img, (100, 100), (400, 400), (120, 120, 120), -1)
        else:
            # Another simple background
            img = np.full((512, 512, 3), (210, 220, 230), dtype=np.uint8)

        # Save test frame
        frame_path = output_dir / f"test_frame_{i:04d}.png"
        cv2.imwrite(str(frame_path), img)
        frame_paths.append(frame_path)

    return frame_paths


def test_decision_engine():
    """Test decision engine initialization and analysis"""
    print("\n" + "="*60)
    print("TEST 1: Decision Engine")
    print("="*60)

    config_path = PROJECT_ROOT / 'configs/stages/intelligent_processing/decision_thresholds.yaml'

    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        return False

    try:
        engine = FrameDecisionEngine(config_path)
        print("✅ Decision engine initialized successfully")

        # Check thresholds loaded
        print(f"   - Simple background threshold: {engine.thresholds.simple_background}")
        print(f"   - Good lighting threshold: {engine.thresholds.good_lighting}")
        print(f"   - Multi-character threshold: {engine.thresholds.multi_character}")

        return True
    except Exception as e:
        print(f"❌ Failed to initialize decision engine: {e}")
        return False


def test_frame_analysis(frame_paths: list):
    """Test frame analysis on synthetic frames"""
    print("\n" + "="*60)
    print("TEST 2: Frame Analysis")
    print("="*60)

    config_path = PROJECT_ROOT / 'configs/stages/intelligent_processing/decision_thresholds.yaml'
    engine = FrameDecisionEngine(config_path)

    for frame_path in frame_paths[:3]:  # Test first 3 frames
        try:
            analysis = engine.analyze_frame(frame_path)

            print(f"\n📸 {frame_path.name}:")
            print(f"   Complexity:    {analysis.complexity:.2f}")
            print(f"   Lighting:      {analysis.lighting_quality:.2f}")
            print(f"   Occlusion:     {analysis.occlusion_level:.2f}")
            print(f"   Quality:       {analysis.quality_score:.2f}")
            print(f"   Sharpness:     {analysis.sharpness:.2f}")

            # Test strategy decision
            strategy, confidence, reasoning = engine.decide_strategy(analysis)
            print(f"   → Strategy:    {strategy} (confidence: {confidence:.2f})")
            print(f"   → Reasoning:   {reasoning}")

        except Exception as e:
            print(f"❌ Failed to analyze {frame_path.name}: {e}")
            return False

    print("\n✅ Frame analysis working correctly")
    return True


def test_intelligent_processor(frame_paths: list, output_dir: Path):
    """Test full intelligent processor"""
    print("\n" + "="*60)
    print("TEST 3: Intelligent Processor")
    print("="*60)

    decision_config = PROJECT_ROOT / 'configs/stages/intelligent_processing/decision_thresholds.yaml'
    strategy_config = PROJECT_ROOT / 'configs/stages/intelligent_processing/strategy_configs.yaml'

    if not strategy_config.exists():
        print(f"❌ Strategy config not found: {strategy_config}")
        return False

    try:
        processor = IntelligentFrameProcessor(
            decision_config_path=decision_config,
            strategy_config_path=strategy_config,
            output_dir=output_dir,
            device='cpu'  # Use CPU for testing
        )
        print("✅ Processor initialized successfully")

        # Process first frame
        print(f"\n📸 Processing test frame: {frame_paths[0].name}")
        result = processor.process_frame(frame_paths[0])

        print(f"   Strategy:      {result.strategy}")
        print(f"   Confidence:    {result.confidence:.2f}")
        print(f"   Success:       {result.success}")
        print(f"   Outputs:       {len(result.outputs)} files")

        if result.error:
            print(f"   Error:         {result.error}")

        # Process batch (limited to 3 frames)
        print(f"\n📦 Processing batch of {len(frame_paths[:3])} frames...")
        report = processor.process_batch(frame_paths[:3], save_report=True)

        print(f"\n✅ Batch processing completed")
        print(f"   Total:         {report['summary']['total_frames']}")
        print(f"   Successful:    {report['summary']['successful']}")
        print(f"   Failed:        {report['summary']['failed']}")

        return True

    except Exception as e:
        print(f"❌ Processor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "█"*60)
    print("  🧪 INTELLIGENT FRAME PROCESSOR - TEST SUITE")
    print("█"*60)

    # Setup test directories
    test_dir = Path("/tmp/intelligent_processor_test")
    test_dir.mkdir(parents=True, exist_ok=True)

    frames_dir = test_dir / "frames"
    output_dir = test_dir / "output"

    print(f"\n📁 Test directory: {test_dir}")

    # Create test frames
    print(f"\n🎨 Creating synthetic test frames...")
    frame_paths = create_test_frames(frames_dir, num_frames=5)
    print(f"✅ Created {len(frame_paths)} test frames")

    # Run tests
    tests = [
        ("Decision Engine", lambda: test_decision_engine()),
        ("Frame Analysis", lambda: test_frame_analysis(frame_paths)),
        ("Intelligent Processor", lambda: test_intelligent_processor(frame_paths, output_dir))
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    print("\n" + "█"*60)
    print("  📊 TEST SUMMARY")
    print("█"*60)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}  {test_name}")

    all_passed = all(success for _, success in results)

    if all_passed:
        print("\n✅ ALL TESTS PASSED!")
        print(f"\n📁 Test outputs: {output_dir}")
        print("\nYou can inspect:")
        print(f"  - Decision engine outputs: {output_dir}/keep_full")
        print(f"  - Processing report: {output_dir}/processing_report.json")
    else:
        print("\n❌ SOME TESTS FAILED")
        print("Please review errors above")

    print("\n" + "█"*60 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
