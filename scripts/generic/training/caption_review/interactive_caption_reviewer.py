#!/usr/bin/env python3
"""
Interactive Caption Reviewer (Web UI)

Web-based interface for manual review of generated captions.
Sampling-based review (30-50 images per character/type).

Features:
- Side-by-side image + caption display
- Accept/Edit/Reject actions
- Keyboard shortcuts for efficiency
- Progress tracking
- Bulk corrections

Author: LLMProvider Tooling
Date: 2025-12-04
"""

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from collections import defaultdict

from flask import Flask, render_template_string, jsonify, request, send_from_directory
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)


class CaptionReviewSession:
    """Manages a caption review session."""

    def __init__(
        self,
        data_root: Path,
        characters: List[str],
        lora_types: List[str],
        samples_per_type: int = 40,
        output_dir: Path = None
    ):
        """
        Initialize review session.

        Args:
            data_root: Root directory with filtered images
            characters: Characters to review
            lora_types: LoRA types to review
            samples_per_type: Samples per character/type
            output_dir: Output directory for review results
        """
        self.data_root = data_root
        self.characters = characters
        self.lora_types = lora_types
        self.samples_per_type = samples_per_type
        self.output_dir = output_dir or (data_root.parent / "review_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Sample images for review
        self.review_items = []
        self.current_index = 0

        # Statistics
        self.stats = {
            'total_reviewed': 0,
            'accepted': 0,
            'edited': 0,
            'rejected': 0,
            'issues_found': defaultdict(int)
        }

        # Load samples
        self._load_samples()

        logger.info(f"✓ Review session initialized")
        logger.info(f"  Total samples: {len(self.review_items)}")

    def _load_samples(self):
        """Load sampled images for review."""
        logger.info("Loading review samples...")

        for character in self.characters:
            for lora_type in self.lora_types:
                # Get tier A images
                tier_a_dir = self.data_root / character / lora_type / "tier_a"

                if not tier_a_dir.exists():
                    logger.warning(f"Tier A directory not found: {tier_a_dir}")
                    continue

                # Get all images
                all_images = list(tier_a_dir.glob("*.png"))

                if len(all_images) == 0:
                    continue

                # Sample randomly
                sample_size = min(self.samples_per_type, len(all_images))
                sampled = random.sample(all_images, sample_size)

                # Add to review items
                for img_path in sampled:
                    caption_path = img_path.with_suffix('.txt')

                    if not caption_path.exists():
                        logger.warning(f"Caption not found: {caption_path}")
                        continue

                    # Load caption
                    with open(caption_path, 'r', encoding='utf-8') as f:
                        caption = f.read().strip()

                    self.review_items.append({
                        'id': len(self.review_items),
                        'image_path': str(img_path),
                        'caption_path': str(caption_path),
                        'original_caption': caption,
                        'current_caption': caption,
                        'character': character,
                        'lora_type': lora_type,
                        'status': 'pending',  # pending/accepted/edited/rejected
                        'issues': [],
                        'notes': ''
                    })

                logger.info(f"  ✓ Loaded {len(sampled)} samples for {character} {lora_type}")

        # Shuffle for variety
        random.shuffle(self.review_items)

        logger.info(f"✓ Loaded {len(self.review_items)} total samples for review")

    def get_current_item(self) -> Dict:
        """Get current review item."""
        if self.current_index < len(self.review_items):
            return self.review_items[self.current_index]
        return None

    def submit_review(
        self,
        item_id: int,
        status: str,
        caption: str,
        issues: List[str],
        notes: str = ''
    ):
        """
        Submit review for an item.

        Args:
            item_id: Item ID
            status: Status (accepted/edited/rejected)
            caption: Current caption (may be edited)
            issues: List of issue types
            notes: Optional notes
        """
        item = self.review_items[item_id]

        # Update item
        item['status'] = status
        item['current_caption'] = caption
        item['issues'] = issues
        item['notes'] = notes
        item['reviewed_at'] = datetime.now().isoformat()

        # Update statistics
        self.stats['total_reviewed'] += 1
        self.stats[status] += 1

        for issue in issues:
            self.stats['issues_found'][issue] += 1

        # If edited, save corrected caption
        if status == 'edited' and caption != item['original_caption']:
            caption_path = Path(item['caption_path'])
            with open(caption_path, 'w', encoding='utf-8') as f:
                f.write(caption)

            logger.info(f"✓ Updated caption: {caption_path.name}")

        # Move to next item
        self.current_index += 1

    def get_progress(self) -> Dict:
        """Get review progress."""
        return {
            'current': self.current_index,
            'total': len(self.review_items),
            'percent': (self.current_index / len(self.review_items) * 100) if self.review_items else 0,
            'stats': dict(self.stats)
        }

    def save_results(self):
        """Save review results."""
        results_file = self.output_dir / f"review_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        results = {
            'session_info': {
                'characters': self.characters,
                'lora_types': self.lora_types,
                'samples_per_type': self.samples_per_type,
                'total_items': len(self.review_items)
            },
            'statistics': dict(self.stats),
            'review_items': self.review_items
        }

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"✓ Review results saved to {results_file}")

        return results_file


# Global session
review_session = None


# HTML Template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Caption Review - Interactive</title>
    <meta charset="UTF-8">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #1a1a1a;
            color: #e0e0e0;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        .header {
            background: #2a2a2a;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .header h1 {
            color: #4CAF50;
            margin-bottom: 10px;
        }
        .progress-bar {
            background: #333;
            height: 30px;
            border-radius: 15px;
            overflow: hidden;
            margin-top: 10px;
        }
        .progress-fill {
            background: linear-gradient(90deg, #4CAF50, #66BB6A);
            height: 100%;
            transition: width 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }
        .main-content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        .panel {
            background: #2a2a2a;
            padding: 20px;
            border-radius: 8px;
        }
        .panel h2 {
            color: #4CAF50;
            margin-bottom: 15px;
            font-size: 18px;
        }
        .image-container {
            text-align: center;
            background: #1a1a1a;
            padding: 10px;
            border-radius: 8px;
        }
        .image-container img {
            max-width: 100%;
            max-height: 500px;
            border-radius: 4px;
        }
        .caption-box {
            background: #1a1a1a;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
        }
        .caption-text {
            width: 100%;
            min-height: 150px;
            background: #333;
            border: 1px solid #555;
            color: #e0e0e0;
            padding: 10px;
            border-radius: 4px;
            font-size: 14px;
            line-height: 1.6;
            resize: vertical;
        }
        .metadata {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            margin-bottom: 15px;
            font-size: 13px;
        }
        .metadata-item {
            background: #1a1a1a;
            padding: 8px;
            border-radius: 4px;
        }
        .metadata-label {
            color: #888;
            font-size: 11px;
            text-transform: uppercase;
        }
        .checklist {
            margin: 15px 0;
        }
        .checklist label {
            display: block;
            padding: 8px;
            margin-bottom: 5px;
            background: #1a1a1a;
            border-radius: 4px;
            cursor: pointer;
            transition: background 0.2s;
        }
        .checklist label:hover {
            background: #333;
        }
        .checklist input[type="checkbox"] {
            margin-right: 10px;
        }
        .actions {
            display: flex;
            gap: 10px;
            margin-top: 20px;
        }
        .btn {
            flex: 1;
            padding: 15px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
        }
        .btn-accept {
            background: #4CAF50;
            color: white;
        }
        .btn-accept:hover {
            background: #45a049;
            transform: translateY(-2px);
        }
        .btn-edit {
            background: #2196F3;
            color: white;
        }
        .btn-edit:hover {
            background: #0b7dda;
            transform: translateY(-2px);
        }
        .btn-reject {
            background: #f44336;
            color: white;
        }
        .btn-reject:hover {
            background: #da190b;
            transform: translateY(-2px);
        }
        .shortcuts {
            background: #2a2a2a;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
            font-size: 13px;
        }
        .shortcuts h3 {
            color: #4CAF50;
            margin-bottom: 10px;
            font-size: 14px;
        }
        .shortcut-item {
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
        }
        .key {
            background: #444;
            padding: 2px 8px;
            border-radius: 3px;
            font-family: monospace;
        }
        .complete-screen {
            text-align: center;
            padding: 60px 20px;
            background: #2a2a2a;
            border-radius: 8px;
        }
        .complete-screen h2 {
            color: #4CAF50;
            font-size: 32px;
            margin-bottom: 20px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin: 30px 0;
        }
        .stat-card {
            background: #1a1a1a;
            padding: 20px;
            border-radius: 8px;
        }
        .stat-number {
            font-size: 36px;
            font-weight: bold;
            color: #4CAF50;
        }
        .stat-label {
            color: #888;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎨 Caption Review Interface</h1>
            <div id="progress-info"></div>
            <div class="progress-bar">
                <div class="progress-fill" id="progress-fill"></div>
            </div>
        </div>

        <div id="review-content"></div>

        <div class="shortcuts">
            <h3>⌨️ Keyboard Shortcuts</h3>
            <div class="shortcut-item">
                <span>Accept caption</span>
                <span class="key">A</span>
            </div>
            <div class="shortcut-item">
                <span>Save edits</span>
                <span class="key">E</span>
            </div>
            <div class="shortcut-item">
                <span>Reject caption</span>
                <span class="key">R</span>
            </div>
            <div class="shortcut-item">
                <span>Save & download results</span>
                <span class="key">S</span>
            </div>
        </div>
    </div>

    <script>
        let currentItem = null;

        // Load current item
        async function loadCurrentItem() {
            const response = await fetch('/api/current');
            const data = await response.json();

            if (data.item) {
                currentItem = data.item;
                renderReviewUI(data.item);
            } else {
                renderComplete(data.progress);
            }

            updateProgress(data.progress);
        }

        // Render review UI
        function renderReviewUI(item) {
            const content = `
                <div class="main-content">
                    <div class="panel">
                        <h2>📷 Image Preview</h2>
                        <div class="metadata">
                            <div class="metadata-item">
                                <div class="metadata-label">Character</div>
                                <div>${item.character}</div>
                            </div>
                            <div class="metadata-item">
                                <div class="metadata-label">LoRA Type</div>
                                <div>${item.lora_type}</div>
                            </div>
                            <div class="metadata-item">
                                <div class="metadata-label">Image</div>
                                <div>${item.image_path.split('/').pop()}</div>
                            </div>
                            <div class="metadata-item">
                                <div class="metadata-label">Token Count</div>
                                <div id="token-count">${item.current_caption.split(' ').length} words</div>
                            </div>
                        </div>
                        <div class="image-container">
                            <img src="/api/image?path=${encodeURIComponent(item.image_path)}" alt="Review image">
                        </div>
                    </div>

                    <div class="panel">
                        <h2>📝 Caption</h2>
                        <div class="caption-box">
                            <textarea class="caption-text" id="caption-text" oninput="updateTokenCount()">${item.current_caption}</textarea>
                        </div>

                        <h2>✅ Review Checklist</h2>
                        <div class="checklist">
                            <label><input type="checkbox" class="issue-check" value="identity_leakage"> Contains character identity info</label>
                            <label><input type="checkbox" class="issue-check" value="wrong_focus"> Wrong LoRA type focus</label>
                            <label><input type="checkbox" class="issue-check" value="length_issue"> Caption too short/long</label>
                            <label><input type="checkbox" class="issue-check" value="grammar_issue"> Grammar or coherence issues</label>
                            <label><input type="checkbox" class="issue-check" value="technical_terms"> Inappropriate technical terms</label>
                        </div>

                        <div class="actions">
                            <button class="btn btn-accept" onclick="submitReview('accepted')">✓ Accept</button>
                            <button class="btn btn-edit" onclick="submitReview('edited')">✎ Save Edits</button>
                            <button class="btn btn-reject" onclick="submitReview('rejected')">✗ Reject</button>
                        </div>
                    </div>
                </div>
            `;

            document.getElementById('review-content').innerHTML = content;
        }

        // Render completion screen
        function renderComplete(progress) {
            const stats = progress.stats;
            const content = `
                <div class="complete-screen">
                    <h2>🎉 Review Complete!</h2>
                    <p>You've reviewed all ${progress.total} samples.</p>

                    <div class="stats-grid">
                        <div class="stat-card">
                            <div class="stat-number">${stats.accepted}</div>
                            <div class="stat-label">Accepted</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">${stats.edited}</div>
                            <div class="stat-label">Edited</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">${stats.rejected}</div>
                            <div class="stat-label">Rejected</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">${progress.total}</div>
                            <div class="stat-label">Total</div>
                        </div>
                    </div>

                    <button class="btn btn-accept" onclick="saveResults()" style="max-width: 300px; margin: 20px auto;">
                        💾 Download Results
                    </button>
                </div>
            `;

            document.getElementById('review-content').innerHTML = content;
        }

        // Update progress bar
        function updateProgress(progress) {
            const fill = document.getElementById('progress-fill');
            const info = document.getElementById('progress-info');

            fill.style.width = progress.percent + '%';
            fill.textContent = Math.round(progress.percent) + '%';

            info.innerHTML = `
                <div style="display: flex; justify-content: space-between; margin-top: 10px; font-size: 14px;">
                    <span>Progress: ${progress.current} / ${progress.total}</span>
                    <span>Accepted: ${progress.stats.accepted} | Edited: ${progress.stats.edited} | Rejected: ${progress.stats.rejected}</span>
                </div>
            `;
        }

        // Update token count
        function updateTokenCount() {
            const text = document.getElementById('caption-text').value;
            const count = text.trim().split(/\\s+/).length;
            document.getElementById('token-count').textContent = count + ' words';
        }

        // Submit review
        async function submitReview(status) {
            const caption = document.getElementById('caption-text').value;
            const issues = Array.from(document.querySelectorAll('.issue-check:checked')).map(cb => cb.value);

            const response = await fetch('/api/review', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    item_id: currentItem.id,
                    status: status,
                    caption: caption,
                    issues: issues
                })
            });

            if (response.ok) {
                loadCurrentItem();
            }
        }

        // Save results
        async function saveResults() {
            const response = await fetch('/api/save');
            const data = await response.json();
            alert('Results saved to: ' + data.results_file);
        }

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'TEXTAREA') return;

            if (e.key === 'a' || e.key === 'A') {
                submitReview('accepted');
            } else if (e.key === 'e' || e.key === 'E') {
                submitReview('edited');
            } else if (e.key === 'r' || e.key === 'R') {
                submitReview('rejected');
            } else if (e.key === 's' || e.key === 'S') {
                saveResults();
            }
        });

        // Initial load
        loadCurrentItem();
    </script>
</body>
</html>
'''


@app.route('/')
def index():
    """Render main page."""
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/current')
def get_current():
    """Get current review item."""
    item = review_session.get_current_item()
    progress = review_session.get_progress()

    return jsonify({
        'item': item,
        'progress': progress
    })


@app.route('/api/review', methods=['POST'])
def submit_review():
    """Submit review for current item."""
    data = request.json

    review_session.submit_review(
        item_id=data['item_id'],
        status=data['status'],
        caption=data['caption'],
        issues=data['issues']
    )

    return jsonify({'success': True})


@app.route('/api/save')
def save_results():
    """Save review results."""
    results_file = review_session.save_results()
    return jsonify({'success': True, 'results_file': str(results_file)})


@app.route('/api/image')
def get_image():
    """Serve image file."""
    image_path = request.args.get('path')
    image_path = Path(image_path)

    return send_from_directory(
        str(image_path.parent),
        image_path.name
    )


def main():
    parser = argparse.ArgumentParser(
        description="Interactive caption reviewer (Web UI)"
    )
    parser.add_argument(
        '--data-root',
        type=Path,
        default='/mnt/data/ai_data/synthetic_lora_data/filtered_data',
        help='Root directory with filtered images'
    )
    parser.add_argument(
        '--characters',
        nargs='+',
        default=['alberto', 'bryce', 'caleb', 'elio', 'giulia', 'ian_lightfoot',
                 'luca', 'miguel', 'orion', 'russell', 'tyler',
                 'alberto_seamonster', 'luca_seamonster', 'barley_lightfoot'],
        help='Characters to review'
    )
    parser.add_argument(
        '--lora-types',
        nargs='+',
        default=['pose', 'action', 'expression'],
        help='LoRA types to review'
    )
    parser.add_argument(
        '--samples-per-type',
        type=int,
        default=40,
        help='Number of samples per character/type (default: 40)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Web server port'
    )

    args = parser.parse_args()

    # Initialize session
    global review_session
    review_session = CaptionReviewSession(
        data_root=args.data_root,
        characters=args.characters,
        lora_types=args.lora_types,
        samples_per_type=args.samples_per_type
    )

    # Start web server
    logger.info(f"\n{'='*60}")
    logger.info("CAPTION REVIEW WEB UI")
    logger.info(f"{'='*60}")
    logger.info(f"\n🌐 Open your browser to: http://localhost:{args.port}")
    logger.info(f"📊 Total samples to review: {len(review_session.review_items)}")
    logger.info(f"⌨️  Use keyboard shortcuts for faster review\n")

    app.run(host='0.0.0.0', port=args.port, debug=False)


if __name__ == '__main__':
    main()
