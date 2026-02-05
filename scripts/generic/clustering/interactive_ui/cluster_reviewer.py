#!/usr/bin/env python3
"""
Interactive Cluster Review UI - Flask Backend.

Web-based interface for reviewing and correcting character clusters:
- Visual grid of clustered images
- Drag-and-drop to move images between clusters
- Merge/split cluster operations
- Rename clusters with character names
- Export corrected results
- Keyboard shortcuts (J/K navigate, M merge, S split, R rename)
- Batch selection (Shift+click)
- Undo/redo support
- Cluster statistics and diversity scores
- Automatic merge suggestions

Usage:
    python cluster_reviewer.py --cluster-dir /path/to/clustered --port 5000
    # Open http://localhost:5000 in browser
"""

from __future__ import annotations

import base64
import json
import logging
import os
import shutil
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from flask import Flask, jsonify, request, send_file, send_from_directory, render_template_string

logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static')

# Global state
cluster_state: Dict = {
    "cluster_dir": None,
    "clusters": {},
    "changes": [],
    "undo_stack": [],
    "redo_stack": [],
    "embeddings_cache": {},
    "merge_suggestions": [],
}


@dataclass
class ClusterInfo:
    """Information about a cluster."""
    id: str
    name: str
    image_count: int
    images: List[str]
    diversity_score: float = 0.0
    is_outlier_cluster: bool = False


def calculate_diversity_score(images: List[Path]) -> float:
    """
    Calculate diversity score for a cluster based on image characteristics.

    Higher score = more diverse (good for training data).
    Score range: 0.0 - 1.0
    """
    if len(images) < 2:
        return 0.0

    try:
        from PIL import Image
        import numpy as np

        # Sample up to 20 images for efficiency
        sample_size = min(20, len(images))
        sampled = images[:sample_size] if len(images) <= sample_size else \
                  [images[i] for i in range(0, len(images), len(images) // sample_size)][:sample_size]

        # Calculate basic statistics from images
        brightnesses = []
        aspect_ratios = []

        for img_path in sampled:
            try:
                with Image.open(img_path) as img:
                    # Convert to grayscale for brightness
                    gray = img.convert('L')
                    arr = np.array(gray)
                    brightnesses.append(np.mean(arr))

                    # Aspect ratio
                    w, h = img.size
                    aspect_ratios.append(w / h if h > 0 else 1.0)
            except Exception:
                continue

        if len(brightnesses) < 2:
            return 0.0

        # Diversity = normalized standard deviation
        brightness_std = np.std(brightnesses) / 128.0  # Normalize to 0-1 range
        aspect_std = np.std(aspect_ratios) / 0.5  # Normalize

        # Combined score (capped at 1.0)
        score = min(1.0, (brightness_std + aspect_std) / 2.0)
        return round(score, 3)

    except ImportError:
        return 0.5  # Default if PIL/numpy not available


def detect_outlier_cluster(cluster_name: str, image_count: int,
                           all_counts: List[int]) -> bool:
    """
    Detect if a cluster is likely an outlier based on size.

    Outliers are clusters significantly smaller than average.
    """
    if cluster_name == "noise":
        return True

    if len(all_counts) < 2 or image_count == 0:
        return False

    avg_count = sum(all_counts) / len(all_counts)

    # Cluster is outlier if less than 20% of average size
    return image_count < avg_count * 0.2


def load_clusters(cluster_dir: Path) -> Dict[str, ClusterInfo]:
    """Load clusters from directory with statistics."""
    clusters = {}
    cluster_sizes = []

    # First pass: collect basic info
    cluster_data = []
    for subdir in sorted(cluster_dir.iterdir()):
        if not subdir.is_dir():
            continue

        if subdir.name == "noise":
            cluster_id = "noise"
        elif subdir.name.startswith("character_"):
            cluster_id = subdir.name
        else:
            # Also allow custom-named clusters
            cluster_id = subdir.name

        # Get image files
        images = []
        for ext in [".png", ".jpg", ".jpeg", ".webp"]:
            images.extend(list(subdir.glob(f"*{ext}")))

        if images:
            cluster_data.append({
                "id": cluster_id,
                "name": subdir.name,
                "images": images,
                "count": len(images),
            })
            cluster_sizes.append(len(images))

    # Second pass: create ClusterInfo with statistics
    for data in cluster_data:
        diversity = calculate_diversity_score(data["images"])
        is_outlier = detect_outlier_cluster(data["name"], data["count"], cluster_sizes)

        clusters[data["id"]] = ClusterInfo(
            id=data["id"],
            name=data["name"],
            image_count=data["count"],
            images=[str(img.relative_to(cluster_dir)) for img in sorted(data["images"])],
            diversity_score=diversity,
            is_outlier_cluster=is_outlier,
        )

    return clusters


def get_image_thumbnail(image_path: Path, size: int = 150) -> str:
    """Get base64 encoded thumbnail."""
    try:
        from PIL import Image
        import io

        with Image.open(image_path) as img:
            img.thumbnail((size, size))
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=80)
            return base64.b64encode(buffer.getvalue()).decode()
    except Exception as e:
        logger.warning(f"Could not create thumbnail for {image_path}: {e}")
        return ""


# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cluster Review - {{ cluster_dir }}</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            min-height: 100vh;
        }
        .header {
            background: #16213e;
            padding: 15px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #0f3460;
        }
        .header h1 { font-size: 1.4em; }
        .header .actions button {
            background: #0f3460;
            color: #fff;
            border: none;
            padding: 8px 16px;
            margin-left: 10px;
            border-radius: 4px;
            cursor: pointer;
        }
        .header .actions button:hover { background: #1a4a7a; }
        .header .actions button.primary { background: #e94560; }
        .header .actions button.primary:hover { background: #ff6b6b; }

        .container { display: flex; height: calc(100vh - 60px); }

        .sidebar {
            width: 250px;
            background: #16213e;
            padding: 15px;
            overflow-y: auto;
            border-right: 1px solid #0f3460;
        }
        .sidebar h3 { margin-bottom: 15px; font-size: 1.1em; }
        .cluster-list { list-style: none; }
        .cluster-item {
            padding: 10px;
            margin-bottom: 8px;
            background: #0f3460;
            border-radius: 6px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .cluster-item:hover { background: #1a4a7a; }
        .cluster-item.selected { background: #e94560; }
        .cluster-item .count {
            background: #1a1a2e;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 0.85em;
        }

        .main-content {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
        }
        .cluster-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        .cluster-header input {
            background: #0f3460;
            border: 1px solid #1a4a7a;
            color: #fff;
            padding: 8px 12px;
            font-size: 1.2em;
            border-radius: 4px;
            width: 300px;
        }
        .cluster-actions button {
            background: #0f3460;
            color: #fff;
            border: none;
            padding: 6px 12px;
            margin-left: 8px;
            border-radius: 4px;
            cursor: pointer;
        }

        .image-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
            gap: 15px;
        }
        .image-item {
            position: relative;
            background: #0f3460;
            border-radius: 8px;
            overflow: hidden;
            cursor: grab;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .image-item:hover {
            transform: translateY(-4px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.3);
        }
        .image-item.selected {
            outline: 3px solid #e94560;
        }
        .image-item.dragging { opacity: 0.5; }
        .image-item img {
            width: 100%;
            height: 150px;
            object-fit: cover;
        }
        .image-item .filename {
            padding: 8px;
            font-size: 0.75em;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .image-item .checkbox {
            position: absolute;
            top: 8px;
            left: 8px;
            width: 20px;
            height: 20px;
            background: rgba(0,0,0,0.5);
            border: 2px solid #fff;
            border-radius: 4px;
            cursor: pointer;
        }
        .image-item.selected .checkbox { background: #e94560; }
        .image-item.focused {
            outline: 3px solid #00ff88;
            outline-offset: 2px;
        }
        .image-item .index {
            position: absolute;
            top: 8px;
            right: 8px;
            background: rgba(0,0,0,0.7);
            color: #aaa;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.75em;
            font-family: monospace;
        }
        .image-item.focused .index {
            background: #00ff88;
            color: #000;
        }

        /* Cluster list improvements */
        .cluster-item.outlier {
            border-left: 3px solid #ffaa00;
        }
        .cluster-item .info-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            width: 100%;
        }
        .cluster-item .diversity {
            font-size: 0.75em;
            color: #888;
            margin-top: 4px;
        }

        /* Suggestions panel */
        .suggestions-panel {
            margin-top: 20px;
            padding: 10px;
            background: #0f3460;
            border-radius: 6px;
        }
        .suggestions-panel h4 {
            margin: 0 0 10px 0;
            font-size: 0.9em;
            color: #e94560;
        }
        .suggestion-item {
            padding: 8px;
            background: #1a1a2e;
            border-radius: 4px;
            margin-bottom: 6px;
            cursor: pointer;
            font-size: 0.85em;
        }
        .suggestion-item:hover {
            background: #1a4a7a;
        }
        .suggestion-item .reason {
            font-size: 0.75em;
            color: #888;
            margin-top: 4px;
            text-transform: capitalize;
        }

        .drop-zone {
            min-height: 200px;
            border: 2px dashed #0f3460;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #666;
            margin-top: 20px;
        }
        .drop-zone.drag-over {
            border-color: #e94560;
            background: rgba(233, 69, 96, 0.1);
        }

        .modal {
            display: none;
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.8);
            z-index: 1000;
            align-items: center;
            justify-content: center;
        }
        .modal.active { display: flex; }
        .modal-content {
            background: #16213e;
            padding: 30px;
            border-radius: 12px;
            max-width: 500px;
            width: 90%;
        }
        .modal-content h2 { margin-bottom: 20px; }
        .modal-content input {
            width: 100%;
            padding: 12px;
            margin-bottom: 15px;
            background: #0f3460;
            border: 1px solid #1a4a7a;
            color: #fff;
            border-radius: 4px;
        }
        .modal-buttons {
            display: flex;
            justify-content: flex-end;
            gap: 10px;
        }
        .modal-buttons button {
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        .modal-buttons .cancel { background: #0f3460; color: #fff; }
        .modal-buttons .confirm { background: #e94560; color: #fff; }

        .toast {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: #0f3460;
            color: #fff;
            padding: 15px 25px;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            transform: translateY(100px);
            opacity: 0;
            transition: all 0.3s;
        }
        .toast.show { transform: translateY(0); opacity: 1; }
        .toast.success { background: #2ecc71; }
        .toast.error { background: #e94560; }

        .stats {
            background: #0f3460;
            padding: 10px 15px;
            border-radius: 6px;
            margin-bottom: 15px;
            font-size: 0.9em;
        }

        /* Cluster item enhancements */
        .cluster-item.outlier {
            border-left: 3px solid #f39c12;
        }
        .cluster-item .diversity {
            font-size: 0.7em;
            color: #888;
            display: block;
        }
        .cluster-item .info-row {
            display: flex;
            flex-direction: column;
            gap: 2px;
        }

        /* Cluster details panel */
        .cluster-details {
            background: #0f3460;
            padding: 10px;
            border-radius: 6px;
            margin-bottom: 15px;
            font-size: 0.85em;
        }
        .cluster-details .detail-row {
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
        }
        .cluster-details .label { color: #888; }

        /* Merge suggestions panel */
        .suggestions-panel {
            background: #0f3460;
            padding: 10px;
            border-radius: 6px;
            margin-top: 15px;
            font-size: 0.85em;
        }
        .suggestions-panel h4 {
            margin-bottom: 10px;
            color: #f39c12;
        }
        .suggestion-item {
            padding: 8px;
            margin-bottom: 5px;
            background: #1a1a2e;
            border-radius: 4px;
            cursor: pointer;
        }
        .suggestion-item:hover { background: #1a4a7a; }
        .suggestion-item .reason {
            font-size: 0.8em;
            color: #888;
        }

        /* Image index for navigation */
        .image-item .index {
            position: absolute;
            top: 8px;
            right: 8px;
            background: rgba(0,0,0,0.7);
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 0.7em;
        }
        .image-item.focused {
            outline: 3px solid #2ecc71;
        }

        /* Keyboard shortcut hints */
        .shortcuts {
            position: fixed;
            bottom: 20px;
            left: 20px;
            background: #0f3460;
            padding: 10px 15px;
            border-radius: 6px;
            font-size: 0.75em;
            opacity: 0.7;
            max-width: 300px;
        }
        .shortcuts:hover { opacity: 1; }
        .shortcuts kbd {
            background: #1a4a7a;
            padding: 2px 5px;
            border-radius: 3px;
            margin-right: 3px;
            font-size: 0.9em;
        }
        .shortcuts .shortcut-row {
            margin-bottom: 4px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎨 Cluster Review</h1>
        <div class="actions">
            <button onclick="selectAll()">Select All</button>
            <button onclick="deselectAll()">Deselect All</button>
            <button onclick="showMergeModal()">Merge</button>
            <button onclick="showSplitModal()">Split</button>
            <button onclick="undoAction()">Undo</button>
            <button onclick="redoAction()">Redo</button>
            <button onclick="exportReport()">📊 Export</button>
            <button class="primary" onclick="saveChanges()">💾 Save</button>
        </div>
    </div>

    <div class="container">
        <div class="sidebar">
            <h3>Clusters</h3>
            <div class="stats" id="stats"></div>
            <ul class="cluster-list" id="clusterList"></ul>
            <button onclick="createNewCluster()" style="width:100%; margin-top:15px; padding:10px; background:#0f3460; color:#fff; border:none; border-radius:4px; cursor:pointer;">
                + New Cluster
            </button>
            <div class="suggestions-panel" id="suggestionsPanel" style="display:none;">
                <h4>Merge Suggestions</h4>
                <div id="suggestionsList"></div>
            </div>
        </div>

        <div class="main-content">
            <div class="cluster-header">
                <input type="text" id="clusterName" placeholder="Cluster name..." onchange="renameCluster()">
                <div class="cluster-actions">
                    <button onclick="deleteSelected()">🗑️ Delete Selected</button>
                    <button onclick="moveToNoise()">Move to Noise</button>
                </div>
            </div>
            <div class="image-grid" id="imageGrid"></div>
            <div class="drop-zone" id="dropZone">
                Drop images here to add to this cluster
            </div>
        </div>
    </div>

    <div class="modal" id="mergeModal">
        <div class="modal-content">
            <h2>Merge Clusters</h2>
            <p>Enter name for merged cluster:</p>
            <input type="text" id="mergedClusterName" placeholder="character_merged">
            <div class="modal-buttons">
                <button class="cancel" onclick="closeModal()">Cancel</button>
                <button class="confirm" onclick="confirmMerge()">Merge</button>
            </div>
        </div>
    </div>

    <div class="modal" id="newClusterModal">
        <div class="modal-content">
            <h2>Create New Cluster</h2>
            <input type="text" id="newClusterName" placeholder="character_name">
            <div class="modal-buttons">
                <button class="cancel" onclick="closeModal()">Cancel</button>
                <button class="confirm" onclick="confirmNewCluster()">Create</button>
            </div>
        </div>
    </div>

    <div class="modal" id="splitModal">
        <div class="modal-content">
            <h2>Split Cluster</h2>
            <p>Move <span id="splitCount">0</span> selected images to a new cluster:</p>
            <input type="text" id="splitClusterName" placeholder="character_split">
            <div class="modal-buttons">
                <button class="cancel" onclick="closeModal()">Cancel</button>
                <button class="confirm" onclick="confirmSplit()">Split</button>
            </div>
        </div>
    </div>

    <div class="toast" id="toast"></div>

    <div class="shortcuts">
        <div class="shortcut-row"><kbd>J</kbd>/<kbd>K</kbd> Navigate images</div>
        <div class="shortcut-row"><kbd>A</kbd> Select all <kbd>D</kbd> Deselect</div>
        <div class="shortcut-row"><kbd>Space</kbd> Toggle select <kbd>Shift+Click</kbd> Range</div>
        <div class="shortcut-row"><kbd>M</kbd> Merge <kbd>S</kbd> Split <kbd>R</kbd> Rename</div>
        <div class="shortcut-row"><kbd>N</kbd> Move to Noise <kbd>Del</kbd> Delete</div>
        <div class="shortcut-row"><kbd>Ctrl+S</kbd> Save <kbd>Ctrl+Z</kbd> Undo <kbd>Ctrl+Y</kbd> Redo</div>
    </div>

    <script>
        let clusters = {};
        let selectedCluster = null;
        let selectedImages = new Set();
        let changes = [];
        let undoStack = [];
        let redoStack = [];
        let focusedIndex = -1;
        let lastSelectedIndex = -1;

        // Save current state to undo stack
        function saveState() {
            undoStack.push(JSON.stringify({
                clusters: clusters,
                selectedCluster: selectedCluster,
                selectedImages: Array.from(selectedImages),
            }));
            redoStack = []; // Clear redo stack on new action
            if (undoStack.length > 50) undoStack.shift(); // Limit stack size
        }

        // Load initial data
        async function loadClusters() {
            const response = await fetch('/api/clusters');
            clusters = await response.json();
            renderClusterList();
            updateStats();
            loadMergeSuggestions();

            // Select first cluster
            const clusterIds = Object.keys(clusters);
            if (clusterIds.length > 0) {
                selectCluster(clusterIds[0]);
            }
        }

        // Load merge suggestions
        async function loadMergeSuggestions() {
            try {
                const response = await fetch('/api/merge-suggestions');
                const data = await response.json();
                if (data.success && data.suggestions.length > 0) {
                    renderSuggestions(data.suggestions);
                }
            } catch (e) {
                console.log('Could not load merge suggestions:', e);
            }
        }

        function renderSuggestions(suggestions) {
            const panel = document.getElementById('suggestionsPanel');
            const list = document.getElementById('suggestionsList');

            if (suggestions.length === 0) {
                panel.style.display = 'none';
                return;
            }

            panel.style.display = 'block';
            list.innerHTML = '';

            for (const s of suggestions) {
                const item = document.createElement('div');
                item.className = 'suggestion-item';
                item.innerHTML = `
                    <div>${clusters[s.cluster1]?.name || s.cluster1} + ${clusters[s.cluster2]?.name || s.cluster2}</div>
                    <div class="reason">${s.reason.replace(/_/g, ' ')}</div>
                `;
                item.onclick = () => applySuggestion(s);
                list.appendChild(item);
            }
        }

        function applySuggestion(s) {
            // Select both clusters' images and trigger merge
            const c1 = clusters[s.cluster1];
            const c2 = clusters[s.cluster2];
            if (!c1 || !c2) return;

            selectedImages.clear();
            c1.images.forEach(img => selectedImages.add(img));
            c2.images.forEach(img => selectedImages.add(img));

            document.getElementById('mergedClusterName').value =
                c1.name.replace('character_', '') + '_' + c2.name.replace('character_', '');
            showMergeModal();
        }

        function renderClusterList() {
            const list = document.getElementById('clusterList');
            list.innerHTML = '';

            for (const [id, cluster] of Object.entries(clusters)) {
                const item = document.createElement('li');
                let className = 'cluster-item' + (id === selectedCluster ? ' selected' : '');
                if (cluster.is_outlier_cluster) className += ' outlier';
                item.className = className;

                // Show diversity score and outlier indicator
                const diversityPct = Math.round((cluster.diversity_score || 0) * 100);
                const outlierBadge = cluster.is_outlier_cluster ? ' ⚠️' : '';

                item.innerHTML = `
                    <div class="info-row">
                        <span>${cluster.name}${outlierBadge}</span>
                        <span class="count">${cluster.image_count}</span>
                    </div>
                    <div class="diversity" title="Diversity score">📊 ${diversityPct}%</div>
                `;
                item.onclick = () => selectCluster(id);
                item.setAttribute('draggable', false);

                // Drop target for moving images
                item.ondragover = (e) => { e.preventDefault(); item.style.background = '#e94560'; };
                item.ondragleave = () => { item.style.background = ''; };
                item.ondrop = (e) => {
                    e.preventDefault();
                    item.style.background = '';
                    moveSelectedToCluster(id);
                };

                list.appendChild(item);
            }
        }

        function selectCluster(clusterId) {
            selectedCluster = clusterId;
            selectedImages.clear();
            renderClusterList();
            renderImageGrid();

            document.getElementById('clusterName').value = clusters[clusterId]?.name || '';
        }

        function renderImageGrid() {
            const grid = document.getElementById('imageGrid');
            grid.innerHTML = '';
            focusedIndex = -1;

            if (!selectedCluster || !clusters[selectedCluster]) return;

            const cluster = clusters[selectedCluster];

            cluster.images.forEach((imagePath, index) => {
                const item = document.createElement('div');
                let className = 'image-item' + (selectedImages.has(imagePath) ? ' selected' : '');
                if (index === focusedIndex) className += ' focused';
                item.className = className;
                item.setAttribute('draggable', true);
                item.dataset.path = imagePath;
                item.dataset.index = index;

                const filename = imagePath.split('/').pop();

                item.innerHTML = `
                    <div class="checkbox"></div>
                    <div class="index">${index + 1}</div>
                    <img src="/api/image/${encodeURIComponent(imagePath)}" loading="lazy">
                    <div class="filename" title="${filename}">${filename}</div>
                `;

                item.onclick = (e) => {
                    if (e.shiftKey && lastSelectedIndex >= 0) {
                        // Range select between lastSelectedIndex and current
                        const start = Math.min(lastSelectedIndex, index);
                        const end = Math.max(lastSelectedIndex, index);
                        for (let i = start; i <= end; i++) {
                            selectedImages.add(cluster.images[i]);
                        }
                        renderImageGrid();
                    } else if (e.ctrlKey || e.metaKey) {
                        toggleImageSelection(imagePath);
                        lastSelectedIndex = index;
                    } else {
                        toggleImageSelection(imagePath);
                        lastSelectedIndex = index;
                    }
                };

                item.ondragstart = (e) => {
                    item.classList.add('dragging');
                    if (!selectedImages.has(imagePath)) {
                        selectedImages.add(imagePath);
                        item.classList.add('selected');
                    }
                    e.dataTransfer.setData('text/plain', imagePath);
                };

                item.ondragend = () => item.classList.remove('dragging');

                grid.appendChild(item);
            });
        }

        function toggleImageSelection(imagePath) {
            if (selectedImages.has(imagePath)) {
                selectedImages.delete(imagePath);
            } else {
                selectedImages.add(imagePath);
            }
            renderImageGrid();
        }

        function selectAll() {
            if (!selectedCluster || !clusters[selectedCluster]) return;
            clusters[selectedCluster].images.forEach(img => selectedImages.add(img));
            renderImageGrid();
        }

        function deselectAll() {
            selectedImages.clear();
            renderImageGrid();
        }

        function updateStats() {
            const stats = document.getElementById('stats');
            const totalClusters = Object.keys(clusters).length;
            const totalImages = Object.values(clusters).reduce((sum, c) => sum + c.image_count, 0);
            stats.innerHTML = `${totalClusters} clusters · ${totalImages} images`;
        }

        async function moveSelectedToCluster(targetClusterId) {
            if (selectedImages.size === 0) return;
            if (targetClusterId === selectedCluster) return;

            const imagesToMove = Array.from(selectedImages);

            // Update local state
            const sourceCluster = clusters[selectedCluster];
            const targetCluster = clusters[targetClusterId];

            sourceCluster.images = sourceCluster.images.filter(img => !selectedImages.has(img));
            sourceCluster.image_count = sourceCluster.images.length;

            targetCluster.images.push(...imagesToMove);
            targetCluster.image_count = targetCluster.images.length;

            // Record change
            changes.push({
                type: 'move',
                images: imagesToMove,
                from: selectedCluster,
                to: targetClusterId,
            });

            selectedImages.clear();
            renderClusterList();
            renderImageGrid();
            updateStats();

            showToast(`Moved ${imagesToMove.length} images`, 'success');
        }

        function moveToNoise() {
            moveSelectedToCluster('noise');
        }

        function deleteSelected() {
            if (selectedImages.size === 0) return;
            if (!confirm(`Delete ${selectedImages.size} images?`)) return;

            const imagesToDelete = Array.from(selectedImages);
            const cluster = clusters[selectedCluster];

            cluster.images = cluster.images.filter(img => !selectedImages.has(img));
            cluster.image_count = cluster.images.length;

            changes.push({
                type: 'delete',
                images: imagesToDelete,
                from: selectedCluster,
            });

            selectedImages.clear();
            renderImageGrid();
            renderClusterList();
            updateStats();

            showToast(`Deleted ${imagesToDelete.length} images`, 'success');
        }

        function renameCluster() {
            const newName = document.getElementById('clusterName').value.trim();
            if (!newName || !selectedCluster) return;

            const oldName = clusters[selectedCluster].name;
            clusters[selectedCluster].name = newName;

            changes.push({
                type: 'rename',
                cluster: selectedCluster,
                oldName: oldName,
                newName: newName,
            });

            renderClusterList();
            showToast(`Renamed to ${newName}`, 'success');
        }

        function createNewCluster() {
            document.getElementById('newClusterModal').classList.add('active');
            document.getElementById('newClusterName').focus();
        }

        function confirmNewCluster() {
            const name = document.getElementById('newClusterName').value.trim();
            if (!name) return;

            const newId = 'character_' + Date.now();
            clusters[newId] = {
                id: newId,
                name: name,
                image_count: 0,
                images: [],
            };

            changes.push({
                type: 'create',
                cluster: newId,
                name: name,
            });

            closeModal();
            renderClusterList();
            selectCluster(newId);
            showToast(`Created cluster: ${name}`, 'success');
        }

        function showMergeModal() {
            // Need at least 2 images from different source clusters selected
            document.getElementById('mergeModal').classList.add('active');
            document.getElementById('mergedClusterName').focus();
        }

        function confirmMerge() {
            const name = document.getElementById('mergedClusterName').value.trim();
            if (!name) return;

            // Merge all selected images into new cluster
            const newId = 'character_' + Date.now();
            const imagesToMerge = Array.from(selectedImages);

            // Remove from current clusters
            for (const [id, cluster] of Object.entries(clusters)) {
                cluster.images = cluster.images.filter(img => !selectedImages.has(img));
                cluster.image_count = cluster.images.length;
            }

            // Create new merged cluster
            clusters[newId] = {
                id: newId,
                name: name,
                image_count: imagesToMerge.length,
                images: imagesToMerge,
            };

            changes.push({
                type: 'merge',
                cluster: newId,
                name: name,
                images: imagesToMerge,
            });

            selectedImages.clear();
            closeModal();
            renderClusterList();
            selectCluster(newId);
            updateStats();
            showToast(`Created merged cluster: ${name}`, 'success');
        }

        function closeModal() {
            document.querySelectorAll('.modal').forEach(m => m.classList.remove('active'));
        }

        function undoLast() {
            if (changes.length === 0) {
                showToast('Nothing to undo', 'error');
                return;
            }

            // Reload clusters to reset state
            changes.pop();
            loadClusters();
            showToast('Undone', 'success');
        }

        function undoAction() {
            if (undoStack.length === 0) {
                showToast('Nothing to undo', 'error');
                return;
            }

            // Save current state to redo stack
            redoStack.push(JSON.stringify({
                clusters: clusters,
                selectedCluster: selectedCluster,
                selectedImages: Array.from(selectedImages),
            }));

            // Restore previous state
            const previousState = JSON.parse(undoStack.pop());
            clusters = previousState.clusters;
            selectedCluster = previousState.selectedCluster;
            selectedImages = new Set(previousState.selectedImages);

            renderClusterList();
            renderImageGrid();
            updateStats();
            showToast('Undone', 'success');
        }

        function redoAction() {
            if (redoStack.length === 0) {
                showToast('Nothing to redo', 'error');
                return;
            }

            // Save current state to undo stack
            undoStack.push(JSON.stringify({
                clusters: clusters,
                selectedCluster: selectedCluster,
                selectedImages: Array.from(selectedImages),
            }));

            // Restore redo state
            const redoState = JSON.parse(redoStack.pop());
            clusters = redoState.clusters;
            selectedCluster = redoState.selectedCluster;
            selectedImages = new Set(redoState.selectedImages);

            renderClusterList();
            renderImageGrid();
            updateStats();
            showToast('Redone', 'success');
        }

        function showSplitModal() {
            if (selectedImages.size === 0) {
                showToast('Select images to split first', 'error');
                return;
            }
            document.getElementById('splitCount').textContent = selectedImages.size;
            document.getElementById('splitModal').classList.add('active');
            document.getElementById('splitClusterName').focus();
        }

        async function confirmSplit() {
            const name = document.getElementById('splitClusterName').value.trim();
            if (!name) return;

            saveState(); // Save for undo

            try {
                const response = await fetch('/api/split-cluster', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        source_cluster: selectedCluster,
                        images: Array.from(selectedImages),
                        new_cluster_name: name,
                    }),
                });

                const result = await response.json();
                if (result.success) {
                    // Update local state
                    const newId = result.new_cluster_id;
                    const imagesToMove = Array.from(selectedImages);

                    // Remove from source
                    clusters[selectedCluster].images = clusters[selectedCluster].images.filter(
                        img => !selectedImages.has(img)
                    );
                    clusters[selectedCluster].image_count = clusters[selectedCluster].images.length;

                    // Create new cluster
                    clusters[newId] = {
                        id: newId,
                        name: name,
                        image_count: imagesToMove.length,
                        images: imagesToMove,
                        diversity_score: 0,
                        is_outlier_cluster: false,
                    };

                    changes.push({
                        type: 'split',
                        source: selectedCluster,
                        newCluster: newId,
                        images: imagesToMove,
                    });

                    selectedImages.clear();
                    closeModal();
                    renderClusterList();
                    selectCluster(newId);
                    updateStats();
                    showToast(`Split ${imagesToMove.length} images to ${name}`, 'success');
                } else {
                    showToast('Split failed: ' + result.error, 'error');
                }
            } catch (e) {
                showToast('Split failed: ' + e.message, 'error');
            }
        }

        async function exportReport() {
            try {
                const response = await fetch('/api/export', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                });

                const result = await response.json();
                if (result.success) {
                    showToast(`Report saved to ${result.report_path}`, 'success');
                } else {
                    showToast('Export failed: ' + result.error, 'error');
                }
            } catch (e) {
                showToast('Export failed: ' + e.message, 'error');
            }
        }

        // Navigate to focused image
        function navigateImages(direction) {
            if (!selectedCluster || !clusters[selectedCluster]) return;
            const cluster = clusters[selectedCluster];
            if (cluster.images.length === 0) return;

            if (focusedIndex === -1) {
                focusedIndex = 0;
            } else {
                focusedIndex = Math.max(0, Math.min(cluster.images.length - 1, focusedIndex + direction));
            }

            // Update visual focus
            const items = document.querySelectorAll('.image-item');
            items.forEach((item, idx) => {
                item.classList.toggle('focused', idx === focusedIndex);
            });

            // Scroll into view
            if (items[focusedIndex]) {
                items[focusedIndex].scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
        }

        function toggleFocusedSelection() {
            if (!selectedCluster || !clusters[selectedCluster] || focusedIndex === -1) return;
            const cluster = clusters[selectedCluster];
            if (focusedIndex < cluster.images.length) {
                toggleImageSelection(cluster.images[focusedIndex]);
                lastSelectedIndex = focusedIndex;
            }
        }

        async function saveChanges() {
            if (changes.length === 0) {
                showToast('No changes to save', 'error');
                return;
            }

            try {
                const response = await fetch('/api/save', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ changes, clusters }),
                });

                const result = await response.json();

                if (result.success) {
                    changes = [];
                    showToast('Changes saved!', 'success');
                } else {
                    showToast('Save failed: ' + result.error, 'error');
                }
            } catch (e) {
                showToast('Save failed: ' + e.message, 'error');
            }
        }

        function showToast(message, type = '') {
            const toast = document.getElementById('toast');
            toast.textContent = message;
            toast.className = 'toast show ' + type;
            setTimeout(() => toast.classList.remove('show'), 3000);
        }

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'INPUT') return;

            // Navigation: J/K for image navigation
            if (e.key === 'j' || e.key === 'J') {
                e.preventDefault();
                navigateImages(1); // Next
            }
            if (e.key === 'k' || e.key === 'K') {
                e.preventDefault();
                navigateImages(-1); // Previous
            }

            // Selection
            if (e.key === 'a' || e.key === 'A') selectAll();
            if (e.key === 'd' || e.key === 'D') deselectAll();
            if (e.key === ' ') {
                e.preventDefault();
                toggleFocusedSelection(); // Space to toggle focused
            }

            // Actions
            if (e.key === 'm' || e.key === 'M') showMergeModal();
            if (e.key === 's' && !e.ctrlKey && !e.metaKey) showSplitModal();
            if (e.key === 'r' || e.key === 'R') {
                e.preventDefault();
                document.getElementById('clusterName').focus();
            }
            if (e.key === 'n' || e.key === 'N') moveToNoise();
            if (e.key === 'Delete' || e.key === 'Backspace') deleteSelected();

            // System shortcuts
            if ((e.ctrlKey || e.metaKey) && e.key === 's') {
                e.preventDefault();
                saveChanges();
            }
            if ((e.ctrlKey || e.metaKey) && e.key === 'z') {
                e.preventDefault();
                undoAction();
            }
            if ((e.ctrlKey || e.metaKey) && e.key === 'y') {
                e.preventDefault();
                redoAction();
            }

            // Escape to close modals
            if (e.key === 'Escape') closeModal();
        });

        // Initialize
        loadClusters();
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    """Serve main page."""
    return render_template_string(
        HTML_TEMPLATE,
        cluster_dir=cluster_state["cluster_dir"],
    )


@app.route('/api/clusters')
def get_clusters():
    """Get all clusters."""
    return jsonify({
        cluster_id: asdict(info)
        for cluster_id, info in cluster_state["clusters"].items()
    })


@app.route('/api/image/<path:image_path>')
def get_image(image_path: str):
    """Serve an image file."""
    cluster_dir = Path(cluster_state["cluster_dir"])
    full_path = cluster_dir / image_path

    if full_path.exists():
        return send_file(full_path)
    else:
        return "Image not found", 404


@app.route('/api/save', methods=['POST'])
def save_changes():
    """Save all changes to disk."""
    try:
        data = request.json
        changes = data.get("changes", [])
        new_clusters = data.get("clusters", {})

        cluster_dir = Path(cluster_state["cluster_dir"])

        # Process each change
        for change in changes:
            change_type = change.get("type")

            if change_type == "move":
                # Move images between clusters
                images = change.get("images", [])
                from_cluster = change.get("from")
                to_cluster = change.get("to")

                for image_path in images:
                    src = cluster_dir / image_path
                    dst_dir = cluster_dir / to_cluster
                    dst_dir.mkdir(exist_ok=True)
                    dst = dst_dir / Path(image_path).name

                    if src.exists():
                        shutil.move(str(src), str(dst))

            elif change_type == "delete":
                # Delete images
                images = change.get("images", [])
                for image_path in images:
                    src = cluster_dir / image_path
                    if src.exists():
                        src.unlink()

            elif change_type == "rename":
                # Rename cluster directory
                old_name = change.get("cluster")
                new_name = change.get("newName")

                old_dir = cluster_dir / old_name
                new_dir = cluster_dir / new_name

                if old_dir.exists() and old_dir != new_dir:
                    old_dir.rename(new_dir)

            elif change_type == "create":
                # Create new cluster directory
                name = change.get("name")
                new_dir = cluster_dir / name
                new_dir.mkdir(exist_ok=True)

            elif change_type == "merge":
                # Create merged cluster (images already moved via 'move' changes)
                name = change.get("name")
                new_dir = cluster_dir / name
                new_dir.mkdir(exist_ok=True)

        # Reload clusters
        cluster_state["clusters"] = load_clusters(cluster_dir)

        return jsonify({"success": True})

    except Exception as e:
        logger.error(f"Save failed: {e}")
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/export', methods=['POST'])
def export_report():
    """Export cluster review report to JSON."""
    try:
        cluster_dir = Path(cluster_state["cluster_dir"])
        clusters = cluster_state["clusters"]

        # Build comprehensive report
        report = {
            "metadata": {
                "export_time": datetime.now().isoformat(),
                "cluster_directory": str(cluster_dir),
                "total_clusters": len(clusters),
                "total_images": sum(c.image_count for c in clusters.values()),
            },
            "clusters": {},
            "statistics": {
                "average_cluster_size": 0,
                "largest_cluster": None,
                "smallest_cluster": None,
                "outlier_clusters": [],
                "high_diversity_clusters": [],
            }
        }

        sizes = []
        for cluster_id, info in clusters.items():
            report["clusters"][cluster_id] = {
                "name": info.name,
                "image_count": info.image_count,
                "diversity_score": info.diversity_score,
                "is_outlier": info.is_outlier_cluster,
                "images": info.images,
            }
            sizes.append((cluster_id, info.image_count))

            if info.is_outlier_cluster:
                report["statistics"]["outlier_clusters"].append(cluster_id)
            if info.diversity_score > 0.5:
                report["statistics"]["high_diversity_clusters"].append(cluster_id)

        if sizes:
            report["statistics"]["average_cluster_size"] = sum(s[1] for s in sizes) / len(sizes)
            sizes.sort(key=lambda x: x[1], reverse=True)
            report["statistics"]["largest_cluster"] = sizes[0][0]
            report["statistics"]["smallest_cluster"] = sizes[-1][0]

        # Save to file
        report_path = cluster_dir / "cluster_review_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return jsonify({
            "success": True,
            "report_path": str(report_path),
            "report": report,
        })

    except Exception as e:
        logger.error(f"Export failed: {e}")
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/merge-suggestions')
def get_merge_suggestions():
    """
    Get automatic merge suggestions based on cluster similarity.

    Uses image filename patterns and cluster sizes to suggest merges.
    """
    try:
        clusters = cluster_state["clusters"]
        suggestions = []

        cluster_list = list(clusters.values())

        # Simple heuristic: suggest merging small clusters with similar names
        for i, c1 in enumerate(cluster_list):
            if c1.id == "noise":
                continue

            for c2 in cluster_list[i+1:]:
                if c2.id == "noise":
                    continue

                # Check name similarity (e.g., "character_0" and "character_0_split")
                if c1.name in c2.name or c2.name in c1.name:
                    suggestions.append({
                        "cluster1": c1.id,
                        "cluster2": c2.id,
                        "reason": "name_similarity",
                        "confidence": 0.7,
                    })
                    continue

                # Check if both are small outlier clusters
                if c1.is_outlier_cluster and c2.is_outlier_cluster:
                    suggestions.append({
                        "cluster1": c1.id,
                        "cluster2": c2.id,
                        "reason": "both_outliers",
                        "confidence": 0.5,
                    })

        # Sort by confidence
        suggestions.sort(key=lambda x: x["confidence"], reverse=True)

        return jsonify({
            "success": True,
            "suggestions": suggestions[:10],  # Top 10 suggestions
        })

    except Exception as e:
        logger.error(f"Merge suggestions failed: {e}")
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/split-cluster', methods=['POST'])
def split_cluster():
    """
    Split a cluster into two based on selected images.

    Moves selected images to a new cluster.
    """
    try:
        data = request.json
        source_cluster = data.get("source_cluster")
        selected_images = data.get("selected_images", [])
        new_cluster_name = data.get("new_cluster_name")

        if not source_cluster or not selected_images or not new_cluster_name:
            return jsonify({"success": False, "error": "Missing required parameters"})

        cluster_dir = Path(cluster_state["cluster_dir"])

        # Create new cluster directory
        new_dir = cluster_dir / new_cluster_name
        new_dir.mkdir(exist_ok=True)

        # Move selected images
        moved_count = 0
        for image_path in selected_images:
            src = cluster_dir / image_path
            dst = new_dir / Path(image_path).name

            if src.exists():
                shutil.move(str(src), str(dst))
                moved_count += 1

        # Reload clusters
        cluster_state["clusters"] = load_clusters(cluster_dir)

        return jsonify({
            "success": True,
            "moved_count": moved_count,
            "new_cluster": new_cluster_name,
        })

    except Exception as e:
        logger.error(f"Split failed: {e}")
        return jsonify({"success": False, "error": str(e)})


def run_server(cluster_dir: str, port: int = 5000, host: str = "0.0.0.0"):
    """Run the Flask server."""
    cluster_path = Path(cluster_dir)

    if not cluster_path.exists():
        raise ValueError(f"Cluster directory not found: {cluster_dir}")

    # Initialize state
    cluster_state["cluster_dir"] = str(cluster_path)
    cluster_state["clusters"] = load_clusters(cluster_path)

    logger.info(f"Loaded {len(cluster_state['clusters'])} clusters from {cluster_dir}")
    logger.info(f"Starting server at http://{host}:{port}")

    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Interactive cluster review UI"
    )
    parser.add_argument(
        "--cluster-dir", "-d",
        required=True,
        help="Directory containing cluster subdirectories"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=5000,
        help="Port to run server on (default: 5000)"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    run_server(args.cluster_dir, args.port, args.host)
