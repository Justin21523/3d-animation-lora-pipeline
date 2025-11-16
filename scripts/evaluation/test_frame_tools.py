#!/usr/bin/env python3
"""
Frame Tools Testing Script
Tests ControlNet, Background Organizer, and Multi-Character Detector
"""

import subprocess
import json
from pathlib import Path
import time
from typing import Dict
import sys


class FrameToolsTester:
    """Test suite for frame analysis tools"""

    def __init__(
        self,
        layered_frames_dir: Path,
        test_output_dir: Path,
        test_episodes: int = 3
    ):
        """
        Initialize tester

        Args:
            layered_frames_dir: Directory with layered frames
            test_output_dir: Output directory for test results
            test_episodes: Number of episodes to test
        """
        self.layered_frames_dir = Path(layered_frames_dir)
        self.test_output_dir = Path(test_output_dir)
        self.test_episodes = test_episodes

        self.test_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"🧪 Frame Tools Testing Suite")
        print(f"  Input: {layered_frames_dir}")
        print(f"  Output: {test_output_dir}")
        print(f"  Test Episodes: {test_episodes}")

    def run_command(self, cmd: list, name: str) -> Dict:
        """Run a command and measure performance"""
        print(f"\n{'='*80}")
        print(f"🔧 Testing: {name}")
        print(f"{'='*80}")
        print(f"Command: {' '.join(cmd)}")

        start_time = time.time()

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout
            )

            elapsed = time.time() - start_time

            if result.returncode == 0:
                print(f"✅ {name} completed successfully in {elapsed:.1f}s")
                return {
                    'success': True,
                    'elapsed_time': elapsed,
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
            else:
                print(f"❌ {name} failed with return code {result.returncode}")
                print(f"Error output:\n{result.stderr}")
                return {
                    'success': False,
                    'elapsed_time': elapsed,
                    'error': result.stderr
                }

        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            print(f"⏱️  {name} timed out after {elapsed:.1f}s")
            return {
                'success': False,
                'elapsed_time': elapsed,
                'error': 'Timeout'
            }
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ {name} failed with exception: {e}")
            return {
                'success': False,
                'elapsed_time': elapsed,
                'error': str(e)
            }

    def test_controlnet_preparer(self) -> Dict:
        """Test ControlNet dataset preparer"""
        output_dir = self.test_output_dir / "controlnet_test"

        # Test on first N episodes
        episodes = sorted([d for d in self.layered_frames_dir.iterdir()
                          if d.is_dir() and d.name.startswith('episode_')])[:self.test_episodes]

        results = []

        for episode_dir in episodes:
            episode_output = output_dir / episode_dir.name

            cmd = [
                'python3',
                'scripts/tools/controlnet_dataset_preparer.py',
                str(episode_dir),
                '--output-dir', str(episode_output),
                '--control-types', 'canny', 'depth'
            ]

            result = self.run_command(cmd, f"ControlNet - {episode_dir.name}")
            result['episode'] = episode_dir.name
            results.append(result)

        # Collect statistics
        successful = sum(1 for r in results if r['success'])
        total_time = sum(r['elapsed_time'] for r in results)

        return {
            'tool': 'ControlNet Preparer',
            'episodes_tested': len(episodes),
            'successful': successful,
            'failed': len(results) - successful,
            'total_time': total_time,
            'avg_time_per_episode': total_time / len(results) if results else 0,
            'results': results,
            'output_dir': str(output_dir)
        }

    def test_background_organizer(self) -> Dict:
        """Test background organizer"""
        output_dir = self.test_output_dir / "background_test"

        # Create test input with first N episodes
        test_input = self.test_output_dir / "test_layered_frames"
        test_input.mkdir(parents=True, exist_ok=True)

        episodes = sorted([d for d in self.layered_frames_dir.iterdir()
                          if d.is_dir() and d.name.startswith('episode_')])[:self.test_episodes]

        # Create symlinks to episodes
        for episode_dir in episodes:
            link_path = test_input / episode_dir.name
            if not link_path.exists():
                link_path.symlink_to(episode_dir)

        cmd = [
            'python3',
            'scripts/tools/background_dataset_organizer.py',
            str(test_input),
            '--output-dir', str(output_dir),
            '--min-cluster-size', '5'
        ]

        result = self.run_command(cmd, "Background Organizer")

        # Load metadata if successful
        metadata_path = output_dir / "background_organization.json"
        metadata = None
        if result['success'] and metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

        return {
            'tool': 'Background Organizer',
            'episodes_tested': len(episodes),
            'successful': result['success'],
            'total_time': result['elapsed_time'],
            'metadata': metadata,
            'output_dir': str(output_dir)
        }

    def test_multi_character_detector(self) -> Dict:
        """Test multi-character scene detector"""
        output_dir = self.test_output_dir / "multi_character_test"

        # Use same test input as background organizer
        test_input = self.test_output_dir / "test_layered_frames"

        cmd = [
            'python3',
            'scripts/tools/multi_character_scene_detector.py',
            str(test_input),
            '--output-dir', str(output_dir),
            '--min-characters', '2',
            '--max-characters', '5',
            '--min-character-size', '5000'
        ]

        result = self.run_command(cmd, "Multi-Character Detector")

        # Load metadata if successful
        metadata_path = output_dir / "multi_character_detection.json"
        metadata = None
        if result['success'] and metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

        return {
            'tool': 'Multi-Character Detector',
            'episodes_tested': self.test_episodes,
            'successful': result['success'],
            'total_time': result['elapsed_time'],
            'metadata': metadata,
            'output_dir': str(output_dir)
        }

    def evaluate_results(self, all_results: Dict) -> Dict:
        """Evaluate and summarize all test results"""
        print(f"\n{'='*80}")
        print(f"📊 Test Results Summary")
        print(f"{'='*80}\n")

        summary = {
            'total_tests': 3,
            'passed': 0,
            'failed': 0,
            'total_time': 0,
            'tools': {}
        }

        for tool_name, results in all_results.items():
            success = results.get('successful', False)
            if isinstance(success, bool):
                if success:
                    summary['passed'] += 1
                else:
                    summary['failed'] += 1
            else:
                # For ControlNet which has multiple episodes
                if success == results.get('episodes_tested', 0):
                    summary['passed'] += 1
                else:
                    summary['failed'] += 1

            summary['total_time'] += results.get('total_time', 0)

            # Print tool results
            print(f"{'='*80}")
            print(f"Tool: {results['tool']}")
            print(f"{'='*80}")

            if tool_name == 'controlnet':
                print(f"  Episodes Tested: {results['episodes_tested']}")
                print(f"  Successful: {results['successful']}/{results['episodes_tested']}")
                print(f"  Failed: {results['failed']}")
                print(f"  Total Time: {results['total_time']:.1f}s")
                print(f"  Avg Time/Episode: {results['avg_time_per_episode']:.1f}s")
                print(f"  Output: {results['output_dir']}")

                # Check output structure
                output_dir = Path(results['output_dir'])
                for episode_result in results['results']:
                    if episode_result['success']:
                        episode_dir = output_dir / episode_result['episode']
                        if episode_dir.exists():
                            canny_count = len(list((episode_dir / 'canny').glob('*.png'))) if (episode_dir / 'canny').exists() else 0
                            depth_count = len(list((episode_dir / 'depth').glob('*.png'))) if (episode_dir / 'depth').exists() else 0
                            source_count = len(list((episode_dir / 'source').glob('*.png'))) if (episode_dir / 'source').exists() else 0
                            print(f"    {episode_result['episode']}: {source_count} source, {canny_count} canny, {depth_count} depth")

            elif tool_name == 'background':
                print(f"  Episodes Tested: {results['episodes_tested']}")
                print(f"  Success: {'✅' if results['successful'] else '❌'}")
                print(f"  Total Time: {results['total_time']:.1f}s")
                print(f"  Output: {results['output_dir']}")

                if results['metadata']:
                    print(f"\n  Background Classification:")
                    print(f"    Total Backgrounds: {results['metadata'].get('total_backgrounds', 0)}")
                    print(f"    Organized: {results['metadata'].get('organized_count', 0)}")
                    if 'valid_types' in results['metadata']:
                        for scene_type, count in sorted(results['metadata']['valid_types'].items()):
                            print(f"      {scene_type}: {count}")

            elif tool_name == 'multi_character':
                print(f"  Episodes Tested: {results['episodes_tested']}")
                print(f"  Success: {'✅' if results['successful'] else '❌'}")
                print(f"  Total Time: {results['total_time']:.1f}s")
                print(f"  Output: {results['output_dir']}")

                if results['metadata']:
                    print(f"\n  Multi-Character Detection:")
                    print(f"    Total Frames: {results['metadata'].get('total_frames', 0)}")
                    print(f"    Multi-Char Frames: {results['metadata'].get('multi_char_frames', 0)}")
                    print(f"    Detection Rate: {results['metadata'].get('detection_rate', 0) * 100:.1f}%")
                    if 'by_count' in results['metadata']:
                        print(f"    By Character Count:")
                        for count, num in sorted(results['metadata']['by_count'].items()):
                            print(f"      {count} characters: {num} scenes")

            print()

            summary['tools'][tool_name] = {
                'success': success if isinstance(success, bool) else (success == results.get('episodes_tested', 0)),
                'time': results.get('total_time', 0)
            }

        # Overall summary
        print(f"{'='*80}")
        print(f"Overall Summary")
        print(f"{'='*80}")
        print(f"  Tests Passed: {summary['passed']}/{summary['total_tests']}")
        print(f"  Tests Failed: {summary['failed']}/{summary['total_tests']}")
        print(f"  Total Time: {summary['total_time']:.1f}s")
        print(f"  Success Rate: {summary['passed'] / summary['total_tests'] * 100:.1f}%")
        print()

        return summary

    def run_all_tests(self):
        """Run all tool tests"""
        print(f"\n{'='*80}")
        print(f"🚀 Starting Frame Tools Test Suite")
        print(f"{'='*80}\n")

        all_results = {}

        # Test 1: ControlNet Preparer
        all_results['controlnet'] = self.test_controlnet_preparer()

        # Test 2: Background Organizer
        all_results['background'] = self.test_background_organizer()

        # Test 3: Multi-Character Detector
        all_results['multi_character'] = self.test_multi_character_detector()

        # Evaluate results
        summary = self.evaluate_results(all_results)

        # Save results to JSON
        results_file = self.test_output_dir / "test_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'all_results': all_results,
                'summary': summary
            }, f, indent=2)

        print(f"📝 Test results saved to: {results_file}")

        return all_results, summary


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Test frame analysis tools"
    )
    parser.add_argument(
        "layered_frames_dir",
        type=Path,
        help="Directory with layered frames"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for test results"
    )
    parser.add_argument(
        "--test-episodes",
        type=int,
        default=3,
        help="Number of episodes to test (default: 3)"
    )

    args = parser.parse_args()

    tester = FrameToolsTester(
        layered_frames_dir=args.layered_frames_dir,
        test_output_dir=args.output_dir,
        test_episodes=args.test_episodes
    )

    results, summary = tester.run_all_tests()

    # Exit with appropriate code
    if summary['failed'] == 0:
        print(f"\n✅ All tests passed!")
        sys.exit(0)
    else:
        print(f"\n❌ {summary['failed']} test(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
