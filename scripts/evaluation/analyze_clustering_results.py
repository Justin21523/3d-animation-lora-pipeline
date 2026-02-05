#!/usr/bin/env python3
"""
Analyze and Compare Clustering Results

Evaluates the effectiveness of different clustering methods by comparing:
- Scene Clustering: HDBSCAN vs k-means
- Character Clustering: InsightFace results
- Expression Clustering: HSEmotion results
- Action Clustering: CLIP + HDBSCAN results

Generates comprehensive comparison reports and visualizations.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List
from datetime import datetime


class ClusteringAnalyzer:
    """Analyze and compare clustering results."""

    def __init__(self, project: str):
        """Initialize clustering analyzer.

        Args:
            project: Project/film name
        """
        self.project = project
        self.results = {
            'project': project,
            'timestamp': datetime.now().isoformat(),
            'comparisons': {}
        }

    def analyze_scene_clustering(
        self,
        hdbscan_report: str,
        kmeans_report: str
    ) -> Dict:
        """Compare HDBSCAN vs k-means for scene clustering.

        Args:
            hdbscan_report: Path to HDBSCAN clustering report
            kmeans_report: Path to k-means clustering report

        Returns:
            Comparison results
        """
        print("\n" + "="*80)
        print("SCENE CLUSTERING COMPARISON: HDBSCAN vs K-means")
        print("="*80)

        comparison = {}

        # Load HDBSCAN results
        if os.path.exists(hdbscan_report):
            with open(hdbscan_report) as f:
                hdbscan_data = json.load(f)
                comparison['hdbscan'] = {
                    'n_clusters': hdbscan_data['n_clusters'],
                    'n_noise': hdbscan_data['n_noise'],
                    'noise_ratio': hdbscan_data['noise_ratio'],
                    'silhouette_score': hdbscan_data.get('silhouette_score'),
                    'davies_bouldin_score': hdbscan_data.get('davies_bouldin_score'),
                    'cluster_sizes': hdbscan_data['cluster_sizes']
                }
        else:
            print(f"⚠️  HDBSCAN report not found: {hdbscan_report}")
            comparison['hdbscan'] = None

        # Load k-means results
        if os.path.exists(kmeans_report):
            with open(kmeans_report) as f:
                kmeans_data = json.load(f)
                comparison['kmeans'] = {
                    'n_clusters': kmeans_data['n_clusters'],
                    'n_noise': kmeans_data.get('n_noise', 0),
                    'noise_ratio': kmeans_data.get('noise_ratio', 0.0),
                    'silhouette_score': kmeans_data.get('silhouette_score'),
                    'davies_bouldin_score': kmeans_data.get('davies_bouldin_score'),
                    'cluster_sizes': kmeans_data['cluster_sizes'],
                    'optimal_k': kmeans_data.get('optimal_k'),
                    'method': kmeans_data.get('method')
                }
        else:
            print(f"⚠️  K-means report not found: {kmeans_report}")
            comparison['kmeans'] = None

        # Print comparison
        if comparison['hdbscan'] and comparison['kmeans']:
            print("\n📊 COMPARISON:")
            print(f"\nNumber of Clusters:")
            print(f"  HDBSCAN: {comparison['hdbscan']['n_clusters']} clusters")
            print(f"  K-means: {comparison['kmeans']['n_clusters']} clusters (optimal k={comparison['kmeans'].get('optimal_k', 'N/A')})")

            print(f"\nNoise:")
            print(f"  HDBSCAN: {comparison['hdbscan']['n_noise']} ({comparison['hdbscan']['noise_ratio']*100:.1f}%)")
            print(f"  K-means: {comparison['kmeans']['n_noise']} ({comparison['kmeans']['noise_ratio']*100:.1f}%)")

            print(f"\nQuality Metrics:")
            print(f"  Silhouette Score (higher = better):")
            print(f"    HDBSCAN: {comparison['hdbscan']['silhouette_score']:.4f}")
            print(f"    K-means: {comparison['kmeans']['silhouette_score']:.4f}")
            print(f"  Davies-Bouldin Index (lower = better):")
            print(f"    HDBSCAN: {comparison['hdbscan']['davies_bouldin_score']:.4f}")
            print(f"    K-means: {comparison['kmeans']['davies_bouldin_score']:.4f}")

            print(f"\nCluster Size Distribution:")
            print(f"  HDBSCAN: mean={comparison['hdbscan']['cluster_sizes']['mean']:.1f}, median={comparison['hdbscan']['cluster_sizes']['median']}")
            print(f"  K-means: mean={comparison['kmeans']['cluster_sizes']['mean']:.1f}, median={comparison['kmeans']['cluster_sizes']['median']}")

            # Determine winner
            print(f"\n🏆 RECOMMENDATION:")
            if comparison['kmeans']['n_clusters'] > comparison['hdbscan']['n_clusters'] * 3:
                print(f"  ✅ K-means is MORE EFFECTIVE:")
                print(f"     - Produces {comparison['kmeans']['n_clusters']} clusters vs HDBSCAN's {comparison['hdbscan']['n_clusters']}")
                print(f"     - Better scene separation (avoids merging different scenes)")
                print(f"     - Lower noise ratio ({comparison['kmeans']['noise_ratio']*100:.1f}% vs {comparison['hdbscan']['noise_ratio']*100:.1f}%)")
            elif comparison['kmeans']['silhouette_score'] > comparison['hdbscan']['silhouette_score']:
                print(f"  ✅ K-means has BETTER cluster quality")
            else:
                print(f"  ⚖️  Both methods have trade-offs")

        return comparison

    def analyze_character_clustering(
        self,
        report_path: str
    ) -> Dict:
        """Analyze character identity clustering results.

        Args:
            report_path: Path to identity clustering report

        Returns:
            Analysis results
        """
        print("\n" + "="*80)
        print("CHARACTER IDENTITY CLUSTERING (InsightFace ArcFace)")
        print("="*80)

        if not os.path.exists(report_path):
            print(f"⚠️  Report not found: {report_path}")
            return None

        with open(report_path) as f:
            data = json.load(f)

        analysis = {
            'total_instances': data['total_instances'],
            'faces_detected': data['faces_detected'],
            'no_face': data['no_face'],
            'detection_rate': data['faces_detected'] / data['total_instances'],
            'n_identities': data['clustering_info']['n_identities'],
            'n_noise': data['clustering_info']['n_noise'],
            'identity_sizes': data['clustering_info']['identity_sizes']
        }

        print(f"\n📊 RESULTS:")
        print(f"  Total instances: {analysis['total_instances']}")
        print(f"  Faces detected: {analysis['faces_detected']} ({analysis['detection_rate']*100:.1f}%)")
        print(f"  No face: {analysis['no_face']}")
        print(f"  Identities found: {analysis['n_identities']}")
        print(f"  Noise: {analysis['n_noise']}")

        print(f"\n🏆 ASSESSMENT:")
        if analysis['detection_rate'] > 0.8:
            print(f"  ✅ EXCELLENT face detection rate ({analysis['detection_rate']*100:.1f}%)")
        elif analysis['detection_rate'] > 0.6:
            print(f"  ✓ GOOD face detection rate ({analysis['detection_rate']*100:.1f}%)")
        else:
            print(f"  ⚠️  LOW face detection rate ({analysis['detection_rate']*100:.1f}%)")

        if analysis['n_identities'] > 0:
            print(f"  ✅ Successfully identified {analysis['n_identities']} distinct characters")

        return analysis

    def analyze_expression_clustering(
        self,
        report_path: str
    ) -> Dict:
        """Analyze expression clustering results.

        Args:
            report_path: Path to expression clustering report

        Returns:
            Analysis results
        """
        print("\n" + "="*80)
        print("EXPRESSION CLUSTERING (HSEmotion)")
        print("="*80)

        if not os.path.exists(report_path):
            print(f"⚠️  Report not found: {report_path}")
            return None

        with open(report_path) as f:
            data = json.load(f)

        analysis = {
            'method': data['method'],
            'total_images': data['total_images'],
            'classified': data.get('classified', 0),
            'failed': data.get('failed', 0)
        }

        if data['method'] == 'hsemotion':
            analysis['emotion_distribution'] = data['emotion_distribution']
            analysis['classification_rate'] = analysis['classified'] / analysis['total_images']

        print(f"\n📊 RESULTS:")
        print(f"  Method: {analysis['method']}")
        print(f"  Total images: {analysis['total_images']}")

        if analysis['method'] == 'hsemotion':
            print(f"  Classified: {analysis['classified']} ({analysis['classification_rate']*100:.1f}%)")
            print(f"  Failed: {analysis['failed']}")
            print(f"\n  Emotion Distribution:")
            for emotion, stats in sorted(analysis['emotion_distribution'].items()):
                count = stats['count']
                avg_conf = stats['avg_confidence']
                print(f"    {emotion:12s}: {count:4d} images (avg conf: {avg_conf:.3f})")

            print(f"\n🏆 ASSESSMENT:")
            if analysis['classification_rate'] > 0.9:
                print(f"  ✅ EXCELLENT classification rate ({analysis['classification_rate']*100:.1f}%)")
            elif analysis['classification_rate'] > 0.7:
                print(f"  ✓ GOOD classification rate ({analysis['classification_rate']*100:.1f}%)")
            else:
                print(f"  ⚠️  LOW classification rate ({analysis['classification_rate']*100:.1f}%)")

        return analysis

    def analyze_action_clustering(
        self,
        report_path: str
    ) -> Dict:
        """Analyze action/pose clustering results.

        Args:
            report_path: Path to action clustering report

        Returns:
            Analysis results
        """
        print("\n" + "="*80)
        print("ACTION/POSE CLUSTERING (CLIP + HDBSCAN)")
        print("="*80)

        if not os.path.exists(report_path):
            print(f"⚠️  Report not found: {report_path}")
            return None

        with open(report_path) as f:
            data = json.load(f)

        analysis = {
            'method': data['method'],
            'total_images': data['total_images'],
            'n_action_clusters': data['n_action_clusters'],
            'n_noise': data['n_noise'],
            'noise_ratio': data['n_noise'] / data['total_images'],
            'feature_dim': data['feature_dim'],
            'reduced_dim': data['reduced_dim']
        }

        print(f"\n📊 RESULTS:")
        print(f"  Total images: {analysis['total_images']}")
        print(f"  Action clusters: {analysis['n_action_clusters']}")
        print(f"  Noise: {analysis['n_noise']} ({analysis['noise_ratio']*100:.1f}%)")
        print(f"  Feature reduction: {analysis['feature_dim']}D → {analysis['reduced_dim']}D")

        print(f"\n🏆 ASSESSMENT:")
        if analysis['n_action_clusters'] > 5:
            print(f"  ✅ Good pose variety ({analysis['n_action_clusters']} pose clusters)")
        else:
            print(f"  ⚠️  Limited pose variety ({analysis['n_action_clusters']} pose clusters)")

        if analysis['noise_ratio'] < 0.2:
            print(f"  ✅ Low noise ratio ({analysis['noise_ratio']*100:.1f}%)")
        elif analysis['noise_ratio'] < 0.4:
            print(f"  ✓ Acceptable noise ratio ({analysis['noise_ratio']*100:.1f}%)")
        else:
            print(f"  ⚠️  High noise ratio ({analysis['noise_ratio']*100:.1f}%)")

        return analysis

    def generate_report(self, output_path: str):
        """Generate comprehensive analysis report.

        Args:
            output_path: Path to save report
        """
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        print("\n" + "="*80)
        print(f"📄 COMPREHENSIVE REPORT SAVED: {output_path}")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and compare clustering results"
    )
    parser.add_argument(
        "--project",
        required=True,
        help="Project/film name"
    )
    parser.add_argument(
        "--base-dir",
        required=True,
        help="Base directory with clustering results"
    )
    parser.add_argument(
        "--scene-hdbscan",
        help="Path to HDBSCAN scene clustering report"
    )
    parser.add_argument(
        "--scene-kmeans",
        help="Path to k-means scene clustering report"
    )
    parser.add_argument(
        "--character",
        help="Path to character identity clustering report"
    )
    parser.add_argument(
        "--expression",
        help="Path to expression clustering report"
    )
    parser.add_argument(
        "--action",
        help="Path to action clustering report"
    )
    parser.add_argument(
        "--output",
        default="clustering_analysis_report.json",
        help="Output report path"
    )

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = ClusteringAnalyzer(args.project)

    # Run analyses
    if args.scene_hdbscan and args.scene_kmeans:
        scene_comparison = analyzer.analyze_scene_clustering(
            args.scene_hdbscan,
            args.scene_kmeans
        )
        analyzer.results['comparisons']['scene'] = scene_comparison

    if args.character:
        character_analysis = analyzer.analyze_character_clustering(args.character)
        analyzer.results['comparisons']['character'] = character_analysis

    if args.expression:
        expression_analysis = analyzer.analyze_expression_clustering(args.expression)
        analyzer.results['comparisons']['expression'] = expression_analysis

    if args.action:
        action_analysis = analyzer.analyze_action_clustering(args.action)
        analyzer.results['comparisons']['action'] = action_analysis

    # Generate report
    analyzer.generate_report(args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
