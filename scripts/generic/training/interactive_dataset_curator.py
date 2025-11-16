#!/usr/bin/env python3
"""
Interactive Dataset Curator for LoRA Training

Web-based UI to review and select final training images after caption generation.
Users can view images with captions, mark for keep/remove, and export curated dataset.
"""

import json
import shutil
import argparse
import webbrowser
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from flask import Flask, render_template_string, jsonify, request, send_from_directory

app = Flask(__name__)

# Global config
CONFIG = {
    'training_data_dir': None,
    'output_dir': None,
    'current_data': None
}


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dataset Curator - Luca LoRA Training</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 0;
            margin: 0;
        }

        .container {
            max-width: 100%;
            margin: 0;
            background: white;
        }

        .sticky-header {
            position: sticky;
            top: 0;
            z-index: 100;
            background: white;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px 30px;
            text-align: center;
        }

        .header h1 {
            font-size: 2em;
            margin-bottom: 5px;
        }

        .header p {
            font-size: 0.9em;
            opacity: 0.9;
        }

        .stats-bar {
            display: flex;
            justify-content: space-around;
            padding: 20px;
            background: #f8f9fa;
            border-bottom: 2px solid #e9ecef;
        }

        .stat-item {
            text-align: center;
        }

        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }

        .stat-label {
            color: #6c757d;
            margin-top: 5px;
        }

        .character-tabs {
            display: flex;
            overflow-x: auto;
            background: #f8f9fa;
            border-bottom: 2px solid #dee2e6;
            padding: 10px 20px;
        }

        .character-tab {
            padding: 12px 24px;
            margin-right: 10px;
            background: white;
            border: 2px solid #dee2e6;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s;
            white-space: nowrap;
            font-weight: 500;
        }

        .character-tab:hover {
            background: #e9ecef;
        }

        .character-tab.active {
            background: #667eea;
            color: white;
            border-color: #667eea;
        }

        .character-tab .count {
            font-size: 0.9em;
            opacity: 0.8;
        }

        .controls {
            padding: 20px;
            background: #f8f9fa;
            border-bottom: 2px solid #dee2e6;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 15px;
        }

        .filter-controls {
            display: flex;
            gap: 10px;
        }

        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.3s;
            font-size: 14px;
        }

        .btn-primary {
            background: #667eea;
            color: white;
        }

        .btn-primary:hover {
            background: #5568d3;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
        }

        .btn-success {
            background: #28a745;
            color: white;
        }

        .btn-success:hover {
            background: #218838;
        }

        .btn-danger {
            background: #dc3545;
            color: white;
        }

        .btn-danger:hover {
            background: #c82333;
        }

        .btn-secondary {
            background: #6c757d;
            color: white;
        }

        .btn-secondary:hover {
            background: #5a6268;
        }

        .action-buttons {
            display: flex;
            gap: 10px;
        }

        .image-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 20px;
            padding: 30px;
            background: #f8f9fa;
            min-height: 100vh;
        }

        .image-card {
            background: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            transition: all 0.3s;
            position: relative;
        }

        .image-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 24px rgba(0,0,0,0.15);
        }

        .image-card.removed {
            opacity: 0.4;
            filter: grayscale(100%);
        }

        .image-wrapper {
            position: relative;
            width: 100%;
            padding-top: 100%;
            background: #f8f9fa;
            overflow: hidden;
        }

        .image-wrapper img {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: contain;
        }

        .status-badge {
            position: absolute;
            top: 10px;
            right: 10px;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 12px;
            z-index: 10;
        }

        .status-badge.kept {
            background: #28a745;
            color: white;
        }

        .status-badge.removed {
            background: #dc3545;
            color: white;
        }

        .image-info {
            padding: 15px;
        }

        .caption {
            font-size: 13px;
            color: #495057;
            line-height: 1.6;
            margin-bottom: 10px;
            max-height: 80px;
            overflow: hidden;
            position: relative;
            transition: max-height 0.3s ease;
            cursor: pointer;
            background: #f8f9fa;
            padding: 10px;
            border-radius: 6px;
            border-left: 3px solid #667eea;
        }

        .caption:hover {
            background: #e9ecef;
        }

        .caption.expanded {
            max-height: 1000px;
            overflow: visible;
        }

        .caption::after {
            content: '...click to expand';
            position: absolute;
            bottom: 0;
            right: 0;
            background: linear-gradient(to right, transparent, #f8f9fa 50%);
            padding: 0 10px;
            font-size: 11px;
            color: #667eea;
            font-style: italic;
        }

        .caption.expanded::after {
            content: 'click to collapse';
            position: static;
            display: block;
            margin-top: 5px;
            background: none;
            text-align: right;
        }

        .filename {
            font-size: 11px;
            color: #6c757d;
            margin-bottom: 10px;
            font-family: monospace;
        }

        .card-actions {
            display: flex;
            gap: 10px;
        }

        .card-actions .btn {
            flex: 1;
            padding: 8px 0;
            font-size: 13px;
        }

        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.8);
            z-index: 1000;
            justify-content: center;
            align-items: center;
        }

        .modal.active {
            display: flex;
        }

        .modal-content {
            background: white;
            padding: 30px;
            border-radius: 15px;
            max-width: 500px;
            width: 90%;
            text-align: center;
        }

        .modal-content h2 {
            margin-bottom: 20px;
            color: #667eea;
        }

        .modal-actions {
            display: flex;
            gap: 15px;
            margin-top: 25px;
            justify-content: center;
        }

        .progress-bar {
            width: 100%;
            height: 30px;
            background: #e9ecef;
            border-radius: 15px;
            overflow: hidden;
            margin: 20px 0;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            transition: width 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }

        .loading {
            text-align: center;
            padding: 50px;
            font-size: 1.5em;
            color: #667eea;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="sticky-header">
            <div class="header">
                <h1>📷 Dataset Curator</h1>
                <p>Review and select final training images for Luca LoRA</p>
            </div>

            <div class="stats-bar">
                <div class="stat-item">
                    <div class="stat-value" id="totalCount">0</div>
                    <div class="stat-label">Total Images</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="keptCount">0</div>
                    <div class="stat-label">Kept</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="removedCount">0</div>
                    <div class="stat-label">Removed</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="characterCount">0</div>
                    <div class="stat-label">Characters</div>
                </div>
            </div>

            <div class="character-tabs" id="characterTabs"></div>

            <div class="controls">
                <div class="filter-controls">
                    <button class="btn btn-secondary" onclick="filterImages('all')">All</button>
                    <button class="btn btn-success" onclick="filterImages('kept')">Kept Only</button>
                    <button class="btn btn-danger" onclick="filterImages('removed')">Removed Only</button>
                </div>

                <div class="action-buttons">
                    <button class="btn btn-success" onclick="keepAllVisible()">✓ Keep All Visible</button>
                    <button class="btn btn-danger" onclick="removeAllVisible()">✗ Remove All Visible</button>
                    <button class="btn btn-primary" onclick="exportDataset()">💾 Export Curated Dataset</button>
                </div>
            </div>
        </div>

        <div id="imageGrid" class="image-grid">
            <div class="loading">Loading images...</div>
        </div>
    </div>

    <div class="modal" id="exportModal">
        <div class="modal-content">
            <h2>Export Curated Dataset?</h2>
            <p>This will copy all kept images and captions to the output directory.</p>
            <div class="progress-bar" id="exportProgress" style="display: none;">
                <div class="progress-fill" id="exportProgressFill">0%</div>
            </div>
            <div class="modal-actions">
                <button class="btn btn-success" onclick="confirmExport()">Confirm Export</button>
                <button class="btn btn-secondary" onclick="closeModal()">Cancel</button>
            </div>
        </div>
    </div>

    <script>
        let allData = {};
        let currentCharacter = 'all';
        let currentFilter = 'all';

        // Load data on page load
        async function loadData() {
            try {
                const response = await fetch('/api/load-dataset');
                const data = await response.json();
                allData = data;
                renderCharacterTabs();
                renderImages();
                updateStats();
            } catch (error) {
                console.error('Error loading data:', error);
                document.getElementById('imageGrid').innerHTML = '<div class="loading">Error loading images</div>';
            }
        }

        function renderCharacterTabs() {
            const tabs = document.getElementById('characterTabs');
            tabs.innerHTML = '';

            // All tab
            const allTab = document.createElement('div');
            allTab.className = 'character-tab active';
            allTab.innerHTML = `All Characters <span class="count">(${getTotalImageCount()})</span>`;
            allTab.onclick = () => switchCharacter('all', allTab);
            tabs.appendChild(allTab);

            // Individual character tabs
            Object.keys(allData).forEach(char => {
                const tab = document.createElement('div');
                tab.className = 'character-tab';
                tab.innerHTML = `${formatCharacterName(char)} <span class="count">(${allData[char].length})</span>`;
                tab.onclick = () => switchCharacter(char, tab);
                tabs.appendChild(tab);
            });

            document.getElementById('characterCount').textContent = Object.keys(allData).length;
        }

        function switchCharacter(char, tabElement) {
            currentCharacter = char;
            document.querySelectorAll('.character-tab').forEach(t => t.classList.remove('active'));
            tabElement.classList.add('active');
            renderImages();
        }

        function formatCharacterName(name) {
            return name.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
        }

        function getTotalImageCount() {
            return Object.values(allData).reduce((sum, images) => sum + images.length, 0);
        }

        function renderImages() {
            const grid = document.getElementById('imageGrid');
            grid.innerHTML = '';

            let imagesToShow = [];

            if (currentCharacter === 'all') {
                Object.entries(allData).forEach(([char, images]) => {
                    images.forEach(img => {
                        img.character = char;
                        imagesToShow.push(img);
                    });
                });
            } else {
                imagesToShow = allData[currentCharacter] || [];
            }

            // Apply filter
            if (currentFilter === 'kept') {
                imagesToShow = imagesToShow.filter(img => img.status === 'kept');
            } else if (currentFilter === 'removed') {
                imagesToShow = imagesToShow.filter(img => img.status === 'removed');
            }

            if (imagesToShow.length === 0) {
                grid.innerHTML = '<div class="loading">No images to display</div>';
                return;
            }

            imagesToShow.forEach((img, idx) => {
                const card = createImageCard(img, idx);
                grid.appendChild(card);
            });

            updateStats();
        }

        function createImageCard(img, idx) {
            const card = document.createElement('div');
            card.className = `image-card ${img.status || 'kept'}`;
            card.dataset.id = img.id;

            const status = img.status || 'kept';
            const badge = status === 'kept' ?
                '<div class="status-badge kept">✓ KEPT</div>' :
                '<div class="status-badge removed">✗ REMOVED</div>';

            card.innerHTML = `
                ${badge}
                <div class="image-wrapper">
                    <img src="/api/image/${img.character}/${img.filename}" alt="${img.filename}" loading="lazy">
                </div>
                <div class="image-info">
                    <div class="caption" onclick="toggleCaption(this)">${img.caption}</div>
                    <div class="filename">${img.filename}</div>
                    <div class="card-actions">
                        <button class="btn btn-success" onclick="toggleStatus('${img.id}', 'kept')">✓ Keep</button>
                        <button class="btn btn-danger" onclick="toggleStatus('${img.id}', 'removed')">✗ Remove</button>
                    </div>
                </div>
            `;

            return card;
        }

        function toggleCaption(element) {
            element.classList.toggle('expanded');
        }

        function toggleStatus(imageId, newStatus) {
            // Find and update in allData
            for (let char in allData) {
                const img = allData[char].find(i => i.id === imageId);
                if (img) {
                    img.status = newStatus;
                    break;
                }
            }

            renderImages();
        }

        function filterImages(filter) {
            currentFilter = filter;
            renderImages();
        }

        function keepAllVisible() {
            const visibleIds = Array.from(document.querySelectorAll('.image-card')).map(card => card.dataset.id);

            for (let char in allData) {
                allData[char].forEach(img => {
                    if (visibleIds.includes(img.id)) {
                        img.status = 'kept';
                    }
                });
            }

            renderImages();
        }

        function removeAllVisible() {
            const visibleIds = Array.from(document.querySelectorAll('.image-card')).map(card => card.dataset.id);

            for (let char in allData) {
                allData[char].forEach(img => {
                    if (visibleIds.includes(img.id)) {
                        img.status = 'removed';
                    }
                });
            }

            renderImages();
        }

        function updateStats() {
            let total = 0, kept = 0, removed = 0;

            Object.values(allData).forEach(images => {
                images.forEach(img => {
                    total++;
                    if (img.status === 'removed') {
                        removed++;
                    } else {
                        kept++;
                    }
                });
            });

            document.getElementById('totalCount').textContent = total;
            document.getElementById('keptCount').textContent = kept;
            document.getElementById('removedCount').textContent = removed;
        }

        function exportDataset() {
            document.getElementById('exportModal').classList.add('active');
        }

        function closeModal() {
            document.getElementById('exportModal').classList.remove('active');
        }

        async function confirmExport() {
            document.getElementById('exportProgress').style.display = 'block';

            try {
                const response = await fetch('/api/export-dataset', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(allData)
                });

                const result = await response.json();

                if (result.success) {
                    alert(`✓ Export successful!\\n\\nKept: ${result.kept} images\\nRemoved: ${result.removed} images\\n\\nOutput: ${result.output_dir}`);
                    closeModal();
                } else {
                    alert('Export failed: ' + result.error);
                }
            } catch (error) {
                alert('Export error: ' + error);
            }

            document.getElementById('exportProgress').style.display = 'none';
        }

        // Initialize
        loadData();
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    """Serve main UI"""
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/load-dataset')
def load_dataset():
    """Load all images and captions from training_data directory"""
    training_dir = Path(CONFIG['training_data_dir'])
    dataset = {}

    for char_dir in training_dir.iterdir():
        if not char_dir.is_dir():
            continue

        images_dir = char_dir / 'images'
        captions_dir = char_dir / 'captions'

        if not images_dir.exists() or not captions_dir.exists():
            continue

        character_images = []

        for img_path in sorted(images_dir.glob('*')):
            if img_path.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
                continue

            caption_path = captions_dir / f"{img_path.stem}.txt"

            if caption_path.exists():
                with open(caption_path, 'r', encoding='utf-8') as f:
                    caption = f.read().strip()
            else:
                caption = "No caption available"

            character_images.append({
                'id': f"{char_dir.name}_{img_path.stem}",
                'filename': img_path.name,
                'character': char_dir.name,
                'caption': caption,
                'status': 'kept'  # Default to kept
            })

        if character_images:
            dataset[char_dir.name] = character_images

    CONFIG['current_data'] = dataset
    return jsonify(dataset)


@app.route('/api/image/<character>/<filename>')
def serve_image(character, filename):
    """Serve individual image"""
    images_dir = Path(CONFIG['training_data_dir']) / character / 'images'
    return send_from_directory(images_dir, filename)


@app.route('/api/export-dataset', methods=['POST'])
def export_dataset():
    """Export curated dataset (only kept images)"""
    try:
        data = request.get_json()
        output_dir = Path(CONFIG['output_dir'])
        training_dir = Path(CONFIG['training_data_dir'])

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        kept_count = 0
        removed_count = 0

        # Process each character
        for character, images in data.items():
            char_output_dir = output_dir / character
            char_images_dir = char_output_dir / 'images'
            char_captions_dir = char_output_dir / 'captions'

            char_images_dir.mkdir(parents=True, exist_ok=True)
            char_captions_dir.mkdir(parents=True, exist_ok=True)

            for img in images:
                if img['status'] == 'removed':
                    removed_count += 1
                    continue

                # Copy image
                src_img = training_dir / character / 'images' / img['filename']
                dst_img = char_images_dir / img['filename']
                shutil.copy2(src_img, dst_img)

                # Copy caption
                caption_filename = f"{Path(img['filename']).stem}.txt"
                src_caption = training_dir / character / 'captions' / caption_filename
                dst_caption = char_captions_dir / caption_filename

                if src_caption.exists():
                    shutil.copy2(src_caption, dst_caption)

                kept_count += 1

        # Save curation report
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_characters': len(data),
            'kept_images': kept_count,
            'removed_images': removed_count,
            'characters': {
                char: {
                    'total': len(images),
                    'kept': sum(1 for img in images if img['status'] != 'removed'),
                    'removed': sum(1 for img in images if img['status'] == 'removed')
                }
                for char, images in data.items()
            }
        }

        report_path = output_dir / 'curation_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*60}")
        print(f"DATASET EXPORT COMPLETE")
        print(f"{'='*60}")
        print(f"Output: {output_dir}")
        print(f"Kept:   {kept_count} images")
        print(f"Removed: {removed_count} images")
        print(f"{'='*60}\n")

        return jsonify({
            'success': True,
            'kept': kept_count,
            'removed': removed_count,
            'output_dir': str(output_dir)
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


def main():
    parser = argparse.ArgumentParser(description="Interactive Dataset Curator for LoRA Training")
    parser.add_argument(
        '--training-data-dir',
        type=str,
        required=True,
        help='Directory with captioned training data'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for curated dataset'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Server port (default: 5000)'
    )
    parser.add_argument(
        '--no-browser',
        action='store_true',
        help='Do not open browser automatically'
    )

    args = parser.parse_args()

    training_dir = Path(args.training_data_dir)
    if not training_dir.exists():
        print(f"Error: Training data directory not found: {training_dir}")
        return 1

    output_dir = Path(args.output_dir)

    CONFIG['training_data_dir'] = str(training_dir)
    CONFIG['output_dir'] = str(output_dir)

    print(f"\n{'='*60}")
    print(f"Interactive Dataset Curator")
    print(f"{'='*60}")
    print(f"Training Data: {training_dir}")
    print(f"Output:        {output_dir}")
    print(f"Server:        http://localhost:{args.port}")
    print(f"{'='*60}\n")
    print("Opening browser...\n")
    print("Press Ctrl+C to stop the server\n")

    if not args.no_browser:
        import threading
        threading.Timer(1.0, lambda: webbrowser.open(f'http://localhost:{args.port}')).start()

    app.run(host='0.0.0.0', port=args.port, debug=False)


if __name__ == '__main__':
    main()
