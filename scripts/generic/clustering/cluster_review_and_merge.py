#!/usr/bin/env python3
"""
Interactive Cluster Review and Merge Tool

Purpose: Review CLIP clustering results, merge similar clusters, assign character names
Features: Web UI with image previews, drag-and-drop merging, renaming
Film-Agnostic: Works with any 3D animation project

Usage:
    python cluster_review_and_merge.py \
      --cluster-dir /path/to/clustered \
      --output-dir /path/to/merged \
      --project luca \
      --port 5566
"""

import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from collections import defaultdict
import base64
from io import BytesIO

from flask import Flask, render_template_string, request, jsonify
from PIL import Image


class ClusterReviewer:
    """Interactive cluster review and merge tool"""

    def __init__(self, cluster_dir: Path, output_dir: Path, port: int = 5566):
        """Initialize reviewer"""
        self.cluster_dir = Path(cluster_dir)
        self.output_dir = Path(output_dir)
        self.port = port

        # Load cluster metadata
        self.clusters = self._load_clusters()
        self.merges = {}  # Map old cluster -> new cluster name
        self.deleted = set()

        # Flask app
        self.app = Flask(__name__)
        self._setup_routes()

    def _load_clusters(self) -> Dict:
        """Load all clusters with stats"""
        clusters = {}

        for cluster_dir in sorted(self.cluster_dir.glob("character_*")):
            cluster_id = cluster_dir.name
            images = list(cluster_dir.glob("*.png"))

            # Get sample images (up to 12)
            samples = sorted(images)[:12]

            clusters[cluster_id] = {
                "id": cluster_id,
                "name": cluster_id,  # Default name
                "count": len(images),
                "sample_paths": [str(p) for p in samples],
                "directory": str(cluster_dir)
            }

        return dict(sorted(clusters.items(), key=lambda x: x[1]["count"], reverse=True))

    def _setup_routes(self):
        """Setup Flask routes"""

        @self.app.route("/")
        def index():
            """Main page"""
            return render_template_string(HTML_TEMPLATE)

        @self.app.route("/api/clusters")
        def get_clusters():
            """Get all clusters with metadata"""
            # Filter out deleted
            active_clusters = {
                cid: data for cid, data in self.clusters.items()
                if cid not in self.deleted
            }

            return jsonify({
                "clusters": list(active_clusters.values()),
                "total": len(active_clusters),
                "merges": self.merges,
                "deleted": list(self.deleted)
            })

        @self.app.route("/api/image/<path:filepath>")
        def get_image(filepath):
            """Get image as base64"""
            try:
                img = Image.open(filepath)
                # Resize for preview
                img.thumbnail((300, 300))
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                return jsonify({"image": f"data:image/png;base64,{img_str}"})
            except Exception as e:
                return jsonify({"error": str(e)}), 404

        @self.app.route("/api/merge", methods=["POST"])
        def merge_clusters():
            """Merge multiple clusters into one"""
            data = request.json
            source_ids = data.get("source_clusters", [])
            target_name = data.get("target_name", "")

            if not source_ids or not target_name:
                return jsonify({"error": "Invalid merge request"}), 400

            for cid in source_ids:
                self.merges[cid] = target_name

            return jsonify({"success": True, "merged": source_ids, "into": target_name})

        @self.app.route("/api/rename", methods=["POST"])
        def rename_cluster():
            """Rename a cluster"""
            data = request.json
            cluster_id = data.get("cluster_id")
            new_name = data.get("new_name")

            if cluster_id not in self.clusters:
                return jsonify({"error": "Cluster not found"}), 404

            self.clusters[cluster_id]["name"] = new_name

            return jsonify({"success": True})

        @self.app.route("/api/delete", methods=["POST"])
        def delete_cluster():
            """Mark cluster for deletion"""
            data = request.json
            cluster_id = data.get("cluster_id")

            if cluster_id not in self.clusters:
                return jsonify({"error": "Cluster not found"}), 404

            self.deleted.add(cluster_id)

            return jsonify({"success": True})

        @self.app.route("/api/export", methods=["POST"])
        def export_merged():
            """Export merged and renamed clusters"""
            try:
                self._export_clusters()
                return jsonify({
                    "success": True,
                    "output_dir": str(self.output_dir),
                    "message": "Clusters exported successfully"
                })
            except Exception as e:
                return jsonify({"error": str(e)}), 500

    def _export_clusters(self):
        """Export merged clusters to output directory"""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Group by final name
        final_clusters = defaultdict(list)

        for cluster_id, data in self.clusters.items():
            # Skip deleted
            if cluster_id in self.deleted:
                continue

            # Get final name (after merges and renames)
            final_name = self.merges.get(cluster_id, data["name"])
            final_clusters[final_name].append(cluster_id)

        # Copy images to final clusters
        for final_name, source_clusters in final_clusters.items():
            final_dir = self.output_dir / final_name
            final_dir.mkdir(exist_ok=True)

            for cluster_id in source_clusters:
                source_dir = Path(self.clusters[cluster_id]["directory"])

                for img_path in source_dir.glob("*.png"):
                    # Copy with unique name (cluster_id prefix to avoid conflicts)
                    dst_path = final_dir / f"{cluster_id}_{img_path.name}"
                    shutil.copy2(img_path, dst_path)

        # Save metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "source_dir": str(self.cluster_dir),
            "output_dir": str(self.output_dir),
            "original_clusters": len(self.clusters),
            "deleted_clusters": len(self.deleted),
            "final_clusters": len(final_clusters),
            "merges": self.merges,
            "deleted": list(self.deleted),
            "final_cluster_names": list(final_clusters.keys()),
            "cluster_stats": {
                name: {
                    "instance_count": sum(self.clusters[cid]["count"] for cid in sources),
                    "source_clusters": sources
                }
                for name, sources in final_clusters.items()
            }
        }

        with open(self.output_dir / "merge_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\n✓ Exported {len(final_clusters)} final clusters to {self.output_dir}")

    def run(self):
        """Start the web server"""
        print(f"\n{'='*60}")
        print(f"CLUSTER REVIEW AND MERGE TOOL")
        print(f"{'='*60}")
        print(f"Cluster directory: {self.cluster_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Total clusters: {len(self.clusters)}")
        print(f"\n🌐 Web UI: http://localhost:{self.port}")
        print(f"{'='*60}\n")

        self.app.run(host="0.0.0.0", port=self.port, debug=False)


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Cluster Review & Merge</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            background: #1a1a1a;
            color: #e0e0e0;
            padding: 20px;
        }
        .header {
            background: #2d2d2d;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        h1 { color: #4fc3f7; margin-bottom: 10px; }
        .stats {
            display: flex;
            gap: 30px;
            margin-top: 15px;
            font-size: 14px;
        }
        .stat { color: #aaa; }
        .stat strong { color: #fff; margin-left: 5px; }

        .controls {
            background: #2d2d2d;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: flex;
            gap: 15px;
            align-items: center;
            flex-wrap: wrap;
        }
        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: all 0.2s;
        }
        .btn-primary {
            background: #4fc3f7;
            color: #000;
        }
        .btn-primary:hover { background: #29b6f6; }
        .btn-success {
            background: #66bb6a;
            color: #000;
        }
        .btn-success:hover { background: #4caf50; }
        .btn-danger {
            background: #ef5350;
            color: #fff;
        }
        .btn-danger:hover { background: #e53935; }

        input[type="text"] {
            padding: 8px 12px;
            border: 1px solid #444;
            border-radius: 4px;
            background: #1a1a1a;
            color: #fff;
            font-size: 14px;
        }

        .cluster-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 20px;
        }
        .cluster-card {
            background: #2d2d2d;
            border-radius: 8px;
            padding: 15px;
            transition: all 0.2s;
        }
        .cluster-card.selected {
            border: 2px solid #4fc3f7;
            box-shadow: 0 0 20px rgba(79, 195, 247, 0.3);
        }
        .cluster-card.merged {
            opacity: 0.5;
            border: 2px solid #66bb6a;
        }
        .cluster-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .cluster-title {
            font-size: 16px;
            font-weight: bold;
            color: #4fc3f7;
        }
        .cluster-count {
            background: #444;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
        }
        .cluster-images {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 5px;
            margin: 10px 0;
        }
        .cluster-img {
            width: 100%;
            height: 80px;
            object-fit: cover;
            border-radius: 4px;
            background: #1a1a1a;
            cursor: pointer;
        }
        .cluster-actions {
            display: flex;
            gap: 8px;
            margin-top: 10px;
        }
        .cluster-actions button {
            flex: 1;
            padding: 6px;
            font-size: 12px;
        }
        .status-bar {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: #2d2d2d;
            padding: 15px 20px;
            border-top: 2px solid #444;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .loading {
            text-align: center;
            padding: 50px;
            color: #aaa;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎬 Cluster Review & Merge Tool</h1>
        <div class="stats">
            <div class="stat">Total Clusters: <strong id="totalClusters">-</strong></div>
            <div class="stat">Selected: <strong id="selectedCount">0</strong></div>
            <div class="stat">Merged: <strong id="mergedCount">0</strong></div>
        </div>
    </div>

    <div class="controls">
        <button class="btn btn-primary" onclick="selectAll()">Select All</button>
        <button class="btn btn-primary" onclick="deselectAll()">Deselect All</button>
        <input type="text" id="mergeNameInput" placeholder="Character name (e.g., luca)" style="flex: 1; max-width: 300px;">
        <button class="btn btn-success" onclick="mergeSelected()">Merge Selected</button>
        <button class="btn btn-danger" onclick="deleteSelected()">Delete Selected</button>
        <button class="btn btn-success" onclick="exportClusters()" style="margin-left: auto;">💾 Export Final Clusters</button>
    </div>

    <div id="clusterGrid" class="cluster-grid"></div>

    <div class="status-bar">
        <div id="statusMessage">Ready</div>
        <div id="mergeInfo"></div>
    </div>

    <script>
        let clusters = [];
        let selected = new Set();
        let merges = {};

        async function loadClusters() {
            const response = await fetch('/api/clusters');
            const data = await response.json();
            clusters = data.clusters;
            merges = data.merges;

            document.getElementById('totalClusters').textContent = clusters.length;
            document.getElementById('mergedCount').textContent = Object.keys(merges).length;

            renderClusters();
        }

        function renderClusters() {
            const grid = document.getElementById('clusterGrid');
            grid.innerHTML = '';

            clusters.forEach(cluster => {
                const card = document.createElement('div');
                card.className = 'cluster-card';
                card.id = `cluster-${cluster.id}`;

                if (selected.has(cluster.id)) {
                    card.classList.add('selected');
                }
                if (merges[cluster.id]) {
                    card.classList.add('merged');
                }

                card.innerHTML = `
                    <div class="cluster-header">
                        <div class="cluster-title">${cluster.name}</div>
                        <div class="cluster-count">${cluster.count} 實例</div>
                    </div>
                    <div class="cluster-images">
                        ${cluster.sample_paths.slice(0, 8).map(path =>
                            `<img class="cluster-img" data-path="${path}" loading="lazy">`
                        ).join('')}
                    </div>
                    <div class="cluster-actions">
                        <button class="btn btn-primary" onclick="toggleSelect('${cluster.id}')">Select</button>
                        <button class="btn btn-primary" onclick="renameCluster('${cluster.id}')">Rename</button>
                    </div>
                    ${merges[cluster.id] ? `<div style="margin-top:8px;color:#66bb6a;font-size:12px;">→ Merged into: ${merges[cluster.id]}</div>` : ''}
                `;

                grid.appendChild(card);
            });

            // Load images lazily
            document.querySelectorAll('.cluster-img').forEach(img => {
                const path = img.dataset.path;
                img.src = '/api/image/' + path;
            });
        }

        function toggleSelect(clusterId) {
            if (selected.has(clusterId)) {
                selected.delete(clusterId);
            } else {
                selected.add(clusterId);
            }
            document.getElementById('selectedCount').textContent = selected.size;
            renderClusters();
        }

        function selectAll() {
            clusters.forEach(c => selected.add(c.id));
            document.getElementById('selectedCount').textContent = selected.size;
            renderClusters();
        }

        function deselectAll() {
            selected.clear();
            document.getElementById('selectedCount').textContent = 0;
            renderClusters();
        }

        async function mergeSelected() {
            if (selected.size === 0) {
                alert('Please select clusters to merge');
                return;
            }

            const targetName = document.getElementById('mergeNameInput').value.trim();
            if (!targetName) {
                alert('Please enter a character name');
                return;
            }

            const response = await fetch('/api/merge', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    source_clusters: Array.from(selected),
                    target_name: targetName
                })
            });

            if (response.ok) {
                alert(`Merged ${selected.size} clusters into "${targetName}"`);
                selected.clear();
                document.getElementById('mergeNameInput').value = '';
                loadClusters();
            }
        }

        async function deleteSelected() {
            if (selected.size === 0) {
                alert('Please select clusters to delete');
                return;
            }

            if (!confirm(`Delete ${selected.size} clusters?`)) {
                return;
            }

            for (const clusterId of selected) {
                await fetch('/api/delete', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({cluster_id: clusterId})
                });
            }

            selected.clear();
            loadClusters();
        }

        async function renameCluster(clusterId) {
            const newName = prompt('Enter new name:');
            if (!newName) return;

            await fetch('/api/rename', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({cluster_id: clusterId, new_name: newName})
            });

            loadClusters();
        }

        async function exportClusters() {
            if (!confirm('Export final merged clusters?')) return;

            const response = await fetch('/api/export', {method: 'POST'});
            const result = await response.json();

            if (result.success) {
                alert(`✓ Clusters exported to:\\n${result.output_dir}`);
            } else {
                alert('Error: ' + result.error);
            }
        }

        // Initial load
        loadClusters();
    </script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(
        description="Interactive cluster review and merge tool (Film-Agnostic)"
    )
    parser.add_argument(
        "--cluster-dir",
        type=str,
        required=True,
        help="Directory with clustered instances"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for merged clusters"
    )
    parser.add_argument(
        "--project",
        type=str,
        help="Project/film name (auto-constructs output path)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5566,
        help="Web UI port (default: 5566)"
    )

    args = parser.parse_args()

    # Determine output directory
    output_dir = args.output_dir
    if args.project and not output_dir:
        base_dir = Path("/mnt/data/ai_data/datasets/3d-anime")
        output_dir = str(base_dir / args.project / "clustered_merged")
        print(f"✓ Using project: {args.project}")
        print(f"   Auto output: {output_dir}")
    elif not output_dir:
        parser.error("Either --output-dir or --project must be specified")

    reviewer = ClusterReviewer(
        cluster_dir=Path(args.cluster_dir),
        output_dir=Path(output_dir),
        port=args.port
    )

    reviewer.run()


if __name__ == "__main__":
    main()
