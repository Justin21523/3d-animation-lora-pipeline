#!/usr/bin/env python3
"""
Flask Server for Interactive Image Reviewer
============================================

Provides API endpoints for the web-based image review tool.

Author: Claude Code
Date: 2025-11-14
"""

from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS
from pathlib import Path
import random
import json
import logging

app = Flask(__name__)
CORS(app)  # Enable CORS for local development

# Configuration
IMAGE_DIR = Path("/mnt/data/ai_data/datasets/3d-anime/luca/luca_face_matched")
SAMPLE_SIZE = 200
RESULTS_DIR = Path("/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/logs/review_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.route('/')
def index():
    """Serve the main HTML page"""
    html_path = Path(__file__).parent / "interactive_image_reviewer.html"
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    return html_content


@app.route('/api/images')
def get_images():
    """Get list of random sample images"""
    try:
        # Find all images
        all_images = list(IMAGE_DIR.glob("*.png")) + list(IMAGE_DIR.glob("*.jpg"))
        logger.info(f"Found {len(all_images)} total images")

        # Random sample
        if len(all_images) > SAMPLE_SIZE:
            sampled = random.sample(all_images, SAMPLE_SIZE)
        else:
            sampled = all_images

        # Prepare response
        images = [
            {
                "name": img.name,
                "path": f"/images/{img.name}",
                "index": i
            }
            for i, img in enumerate(sampled)
        ]

        logger.info(f"Returning {len(images)} sampled images")
        return jsonify({
            "success": True,
            "total": len(all_images),
            "sample_size": len(images),
            "images": images
        })

    except Exception as e:
        logger.error(f"Error getting images: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/images/<path:filename>')
def serve_image(filename):
    """Serve individual images"""
    return send_from_directory(IMAGE_DIR, filename)


@app.route('/api/save-results', methods=['POST'])
def save_results():
    """Save review results"""
    try:
        data = request.json
        timestamp = data.get('timestamp', 'unknown')

        # Save to JSON file
        result_file = RESULTS_DIR / f"review_{timestamp.replace(':', '-')}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved results to {result_file}")
        logger.info(f"Approved: {len(data['approved'])}, Rejected: {len(data['rejected'])}")

        # Create approved images list file
        approved_list = RESULTS_DIR / f"approved_{timestamp.replace(':', '-')}.txt"
        with open(approved_list, 'w', encoding='utf-8') as f:
            for img_name in data['approved']:
                f.write(f"{img_name}\n")

        return jsonify({
            "success": True,
            "saved_to": str(result_file),
            "approved_count": len(data['approved']),
            "rejected_count": len(data['rejected'])
        })

    except Exception as e:
        logger.error(f"Error saving results: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/config')
def get_config():
    """Get current configuration"""
    return jsonify({
        "image_dir": str(IMAGE_DIR),
        "sample_size": SAMPLE_SIZE,
        "results_dir": str(RESULTS_DIR)
    })


if __name__ == '__main__':
    print("=" * 80)
    print("Luca 圖片審查工具 - Flask 服務器")
    print("=" * 80)
    print(f"圖片目錄: {IMAGE_DIR}")
    print(f"樣本數量: {SAMPLE_SIZE}")
    print(f"結果保存: {RESULTS_DIR}")
    print()
    print("啟動服務器...")
    print("請在瀏覽器開啟: http://localhost:5000")
    print("=" * 80)

    app.run(host='0.0.0.0', port=5000, debug=True)
