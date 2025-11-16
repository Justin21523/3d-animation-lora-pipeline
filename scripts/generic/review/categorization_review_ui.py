#!/usr/bin/env python3
"""
Categorization Review UI (Performance Optimized)

Purpose: Review and correct CLIP auto-categorization results
Features: Paginated view, lazy loading, efficient rendering
Use Cases: Manual review of instance categorization to fix CLIP errors

Usage:
    python categorization_review_ui.py \
        --categorized-dir /path/to/instances_categorized \
        --output-dir /path/to/reviewed \
        --project luca \
        --port 5555

Then open browser to http://localhost:5555
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from flask import Flask, render_template_string, request, jsonify, send_from_directory


class CategorizationReviewApp:
    """Flask app for reviewing CLIP categorization results (optimized)"""

    def __init__(
        self,
        categorized_dir: Path,
        output_dir: Path,
        project: Optional[str] = None,
        port: int = 5555
    ):
        """
        Initialize categorization review app

        Args:
            categorized_dir: Directory with categorized instances
            output_dir: Directory to save reviewed instances
            project: Project name
            port: Server port
        """
        self.categorized_dir = Path(categorized_dir)
        self.output_dir = Path(output_dir)
        self.project = project
        self.port = port

        # Load categorization results
        self.results_file = self.categorized_dir / "categorization_results.json"
        self.original_results = self.load_results()

        # Available categories
        self.categories = [
            "character",
            "human person",
            "object",
            "furniture",
            "background",
            "vehicle",
            "prop",
            "accessory",
            "uncertain",
        ]

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load or initialize review state
        self.state_file = self.output_dir / "review_state.json"
        self.state = self.load_state()

        # Scan all instances (cached)
        print("📂 Scanning instances...")
        self.instances = self.scan_instances()
        self.instances_cache = self.instances  # Cache for performance
        print(f"✅ Loaded {len(self.instances)} instances")

        # Create Flask app
        self.app = Flask(__name__)
        self.setup_routes()

    def load_results(self) -> Dict:
        """Load original categorization results"""
        if self.results_file.exists():
            with open(self.results_file) as f:
                return json.load(f)
        else:
            return {"instances": {}, "statistics": {}}

    def scan_instances(self) -> List[Dict]:
        """
        Scan all categorized instances (with caching)

        Returns:
            List of instance metadata with categorization info
        """
        instances = []

        # Scan all category folders
        for category in self.categories:
            category_dir = self.categorized_dir / category
            if not category_dir.exists():
                continue

            image_files = sorted(
                list(category_dir.glob("*.png")) +
                list(category_dir.glob("*.jpg"))
            )

            for img_path in image_files:
                filename = img_path.name

                # Get original categorization info
                original_info = self.original_results.get("instances", {}).get(filename, {})

                # Check review status
                review_info = self.state["reviews"].get(filename)

                if review_info:
                    status = "reviewed"
                    current_category = review_info["new_category"]
                    action = review_info["action"]  # keep, discard, reclassify
                else:
                    status = "pending"
                    current_category = category
                    action = None

                instances.append({
                    "filename": filename,
                    "path": str(img_path),
                    "original_category": category,
                    "current_category": current_category,
                    "confidence": original_info.get("confidence", 0),
                    "status": status,
                    "action": action,
                    "raw_category": original_info.get("raw_category"),
                })

        return instances

    def load_state(self) -> Dict:
        """Load saved review state"""
        if self.state_file.exists():
            with open(self.state_file) as f:
                return json.load(f)
        else:
            return {
                "reviews": {},  # filename -> review info
                "kept": [],
                "discarded": [],
                "reclassified": [],
                "last_updated": None,
            }

    def save_state(self):
        """Save current state to file"""
        self.state["last_updated"] = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def setup_routes(self):
        """Setup Flask routes"""

        @self.app.route('/')
        def index():
            """Main UI page"""
            return render_template_string(HTML_TEMPLATE, categories=self.categories)

        @self.app.route('/api/instances')
        def get_instances():
            """Get paginated instances with filtering"""
            # Get pagination params
            page = int(request.args.get('page', 1))
            per_page = int(request.args.get('per_page', 100))

            # Get filter params
            category_filter = request.args.get('category', 'all')
            status_filter = request.args.get('status', 'all')

            # Apply filters (use cache)
            filtered = [
                inst for inst in self.instances_cache
                if (category_filter == 'all' or inst['current_category'] == category_filter)
                and (status_filter == 'all' or inst['status'] == status_filter)
            ]

            # Compute stats
            stats = {
                "total": len(self.instances_cache),
                "filtered": len(filtered),
                "reviewed": len([i for i in self.instances_cache if i["status"] == "reviewed"]),
                "pending": len([i for i in self.instances_cache if i["status"] == "pending"]),
                "kept": len(self.state["kept"]),
                "discarded": len(self.state["discarded"]),
                "reclassified": len(self.state["reclassified"]),
            }

            # Category distribution
            category_counts = {}
            for inst in self.instances_cache:
                cat = inst["current_category"]
                category_counts[cat] = category_counts.get(cat, 0) + 1

            # Pagination
            total_pages = (len(filtered) + per_page - 1) // per_page
            start_idx = (page - 1) * per_page
            end_idx = start_idx + per_page
            page_instances = filtered[start_idx:end_idx]

            return jsonify({
                "instances": page_instances,
                "stats": stats,
                "category_counts": category_counts,
                "categories": self.categories,
                "project": self.project,
                "pagination": {
                    "page": page,
                    "per_page": per_page,
                    "total": len(filtered),
                    "total_pages": total_pages,
                }
            })

        @self.app.route('/api/image/<category>/<path:filename>')
        def get_image(category, filename):
            """Serve instance image"""
            category_dir = self.categorized_dir / category
            return send_from_directory(category_dir, filename)

        @self.app.route('/api/review', methods=['POST'])
        def review_instance():
            """Review an instance (keep, discard, reclassify)"""
            data = request.json
            filename = data.get("filename")
            action = data.get("action")  # keep, discard, reclassify
            new_category = data.get("new_category")

            if not filename:
                return jsonify({"error": "Missing filename"}), 400

            # Find the instance
            inst = next((i for i in self.instances_cache if i["filename"] == filename), None)
            if not inst:
                return jsonify({"error": "Instance not found"}), 404

            # Update state
            review_info = {
                "action": action,
                "original_category": inst["original_category"],
                "new_category": new_category if new_category else inst["original_category"],
                "timestamp": datetime.now().isoformat(),
            }

            self.state["reviews"][filename] = review_info

            # Track action
            if action == "keep":
                if filename not in self.state["kept"]:
                    self.state["kept"].append(filename)
                if filename in self.state["discarded"]:
                    self.state["discarded"].remove(filename)

            elif action == "discard":
                if filename not in self.state["discarded"]:
                    self.state["discarded"].append(filename)
                if filename in self.state["kept"]:
                    self.state["kept"].remove(filename)

            elif action == "reclassify":
                if filename not in self.state["reclassified"]:
                    self.state["reclassified"].append(filename)

            # Update cache
            inst["status"] = "reviewed"
            inst["action"] = action
            inst["current_category"] = new_category if new_category else inst["original_category"]

            self.save_state()

            return jsonify({"success": True})

        @self.app.route('/api/batch', methods=['POST'])
        def batch_review():
            """Batch review multiple instances"""
            data = request.json
            filenames = data.get("filenames", [])
            action = data.get("action")
            new_category = data.get("new_category")

            for filename in filenames:
                inst = next((i for i in self.instances_cache if i["filename"] == filename), None)
                if not inst:
                    continue

                review_info = {
                    "action": action,
                    "original_category": inst["original_category"],
                    "new_category": new_category if new_category else inst["original_category"],
                    "timestamp": datetime.now().isoformat(),
                }

                self.state["reviews"][filename] = review_info

                if action == "keep":
                    if filename not in self.state["kept"]:
                        self.state["kept"].append(filename)
                    if filename in self.state["discarded"]:
                        self.state["discarded"].remove(filename)

                elif action == "discard":
                    if filename not in self.state["discarded"]:
                        self.state["discarded"].append(filename)
                    if filename in self.state["kept"]:
                        self.state["kept"].remove(filename)

                elif action == "reclassify":
                    if filename not in self.state["reclassified"]:
                        self.state["reclassified"].append(filename)

                # Update cache
                inst["status"] = "reviewed"
                inst["action"] = action
                inst["current_category"] = new_category if new_category else inst["original_category"]

            self.save_state()

            return jsonify({"success": True, "count": len(filenames)})

        @self.app.route('/api/refresh', methods=['POST'])
        def refresh_instances():
            """Manually refresh instances list"""
            print("🔄 Refreshing instances...")
            self.instances_cache = self.scan_instances()
            print(f"✅ Refreshed {len(self.instances_cache)} instances")
            return jsonify({"success": True, "count": len(self.instances_cache)})

        @self.app.route('/api/export', methods=['POST'])
        def export_results():
            """Export reviewed results to output directory"""
            print("\n" + "="*70)
            print("EXPORTING REVIEWED RESULTS")
            print("="*70)

            # Create category folders in output
            for category in self.categories:
                (self.output_dir / category).mkdir(exist_ok=True)

            kept_dir = self.output_dir / "kept"
            discarded_dir = self.output_dir / "discarded"
            kept_dir.mkdir(exist_ok=True)
            discarded_dir.mkdir(exist_ok=True)

            export_stats = {
                "kept": 0,
                "discarded": 0,
                "reclassified": 0,
                "unchanged": 0,
            }

            # Process all instances
            for inst in self.instances_cache:
                filename = inst["filename"]
                src_path = Path(inst["path"])

                review_info = self.state["reviews"].get(filename)

                if not review_info:
                    # No review - keep in original category
                    category = inst["original_category"]
                    if category != "uncertain":  # Don't export uncertain by default
                        dst_path = self.output_dir / category / filename
                        shutil.copy2(src_path, dst_path)
                        export_stats["unchanged"] += 1
                    continue

                action = review_info["action"]
                new_category = review_info["new_category"]

                if action == "keep":
                    # Copy to kept folder
                    dst_path = kept_dir / filename
                    shutil.copy2(src_path, dst_path)
                    export_stats["kept"] += 1

                elif action == "discard":
                    # Copy to discarded folder (for record)
                    dst_path = discarded_dir / filename
                    shutil.copy2(src_path, dst_path)
                    export_stats["discarded"] += 1

                elif action == "reclassify":
                    # Copy to new category folder
                    dst_path = self.output_dir / new_category / filename
                    shutil.copy2(src_path, dst_path)
                    export_stats["reclassified"] += 1

            # Save final report
            report = {
                "project": self.project,
                "categorized_dir": str(self.categorized_dir),
                "output_dir": str(self.output_dir),
                "total_instances": len(self.instances_cache),
                "reviewed_instances": len(self.state["reviews"]),
                "export_stats": export_stats,
                "review_state": self.state,
                "timestamp": datetime.now().isoformat(),
            }

            report_path = self.output_dir / "review_report.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)

            print(f"\n✅ Export complete!")
            print(f"   Kept: {export_stats['kept']}")
            print(f"   Discarded: {export_stats['discarded']}")
            print(f"   Reclassified: {export_stats['reclassified']}")
            print(f"   Unchanged: {export_stats['unchanged']}")
            print(f"\n📄 Report: {report_path}")
            print("="*70 + "\n")

            return jsonify({
                "success": True,
                "report_path": str(report_path),
                "stats": export_stats,
            })

    def run(self):
        """Start the Flask server"""
        print("\n" + "="*70)
        print("CATEGORIZATION REVIEW UI (OPTIMIZED)")
        print("="*70)
        print(f"Categorized: {self.categorized_dir}")
        print(f"Output:      {self.output_dir}")
        if self.project:
            print(f"Project:     {self.project}")
        print(f"\nTotal instances: {len(self.instances)}")
        print(f"Already reviewed: {len(self.state['reviews'])}")
        print(f"Pending: {len(self.instances) - len(self.state['reviews'])}")

        # Show category distribution
        category_counts = {}
        for inst in self.instances:
            cat = inst["original_category"]
            category_counts[cat] = category_counts.get(cat, 0) + 1

        print(f"\nCategory distribution:")
        for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  - {cat}: {count}")
        if len(category_counts) > 5:
            print(f"  ... and {len(category_counts) - 5} more")

        print("="*70)
        print(f"\n🌐 Open in browser: http://localhost:{self.port}")
        print("\n⚡ Performance optimizations:")
        print("  - Paginated loading (100 instances/page)")
        print("  - Lazy image loading")
        print("  - Cached instance list")
        print("  - Manual refresh (click button to update)")
        print("\nKeyboard shortcuts:")
        print("  K = Keep | D = Discard | C = Re-classify")
        print("  → = Next | ← = Previous | Shift+Click = Multi-select")
        print("\nPress Ctrl+C to stop server")
        print("="*70 + "\n")

        self.app.run(host='0.0.0.0', port=self.port, debug=False)


# HTML Template (Optimized with Pagination + Lazy Loading)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Categorization Review UI</title>
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
        .header h1 {
            font-size: 24px;
            margin-bottom: 10px;
        }
        .stats {
            display: flex;
            gap: 15px;
            margin-top: 15px;
            flex-wrap: wrap;
        }
        .stat {
            background: #3d3d3d;
            padding: 10px 20px;
            border-radius: 4px;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #4CAF50;
        }
        .stat-label {
            font-size: 12px;
            color: #999;
            margin-top: 5px;
        }
        .controls {
            background: #2d2d2d;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .control-row {
            display: flex;
            gap: 10px;
            align-items: center;
            flex-wrap: wrap;
            margin-bottom: 10px;
        }
        .control-row:last-child {
            margin-bottom: 0;
        }
        .btn {
            background: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }
        .btn:hover { background: #45a049; }
        .btn-danger { background: #f44336; }
        .btn-danger:hover { background: #da190b; }
        .btn-warning { background: #FF9800; }
        .btn-warning:hover { background: #F57C00; }
        .btn-secondary { background: #555; }
        .btn-secondary:hover { background: #666; }
        .btn-info { background: #2196F3; }
        .btn-info:hover { background: #0b7dda; }

        .filter-tabs {
            display: flex;
            gap: 5px;
            flex-wrap: wrap;
        }
        .filter-tab {
            background: #3d3d3d;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 13px;
        }
        .filter-tab.active {
            background: #4CAF50;
        }

        select, input {
            background: #3d3d3d;
            color: #e0e0e0;
            border: 1px solid #555;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 14px;
        }

        .pagination {
            display: flex;
            gap: 10px;
            align-items: center;
            margin-bottom: 20px;
            justify-content: center;
            background: #2d2d2d;
            padding: 15px;
            border-radius: 8px;
        }
        .page-info {
            color: #e0e0e0;
            font-weight: bold;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }
        .instance-card {
            background: #2d2d2d;
            border-radius: 8px;
            overflow: hidden;
            cursor: pointer;
            transition: transform 0.2s;
            position: relative;
        }
        .instance-card:hover {
            transform: scale(1.05);
        }
        .instance-card.selected {
            outline: 3px solid #4CAF50;
        }
        .instance-card.reviewed {
            outline: 2px solid #2196F3;
        }
        .instance-img {
            width: 100%;
            height: 200px;
            object-fit: contain;
            background: #1a1a1a;
        }
        .instance-img[data-src] {
            background: #1a1a1a url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"><text x="50%" y="50%" text-anchor="middle" fill="%23666" font-size="14">Loading...</text></svg>') center no-repeat;
        }
        .instance-info {
            padding: 10px;
            font-size: 11px;
        }
        .instance-category {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 5px;
        }
        .category-badge {
            background: #555;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 10px;
        }
        .confidence {
            color: #999;
            font-size: 10px;
        }
        .instance-status {
            position: absolute;
            top: 5px;
            right: 5px;
            background: rgba(0,0,0,0.7);
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: bold;
        }
        .status-reviewed { color: #2196F3; }
        .status-pending { color: #FFC107; }

        .action-indicator {
            position: absolute;
            top: 5px;
            left: 5px;
            background: rgba(0,0,0,0.7);
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 10px;
            font-weight: bold;
        }
        .action-keep { color: #4CAF50; }
        .action-discard { color: #f44336; }
        .action-reclassify { color: #FF9800; }

        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.9);
            z-index: 1000;
            justify-content: center;
            align-items: center;
            overflow-y: auto;
            padding: 20px 0;
        }
        .modal.active {
            display: flex;
        }
        .modal-content {
            max-width: 90%;
            max-height: 90vh;
            position: relative;
            display: flex;
            flex-direction: column;
            overflow-y: auto;
        }
        .modal-img {
            max-width: 100%;
            max-height: 50vh;
            object-fit: contain;
            flex-shrink: 0;
        }
        .modal-controls {
            background: #2d2d2d;
            padding: 20px;
            margin-top: 10px;
            border-radius: 8px;
            max-height: 40vh;
            overflow-y: auto;
        }
        .modal-info {
            margin-bottom: 15px;
            padding: 10px;
            background: #3d3d3d;
            border-radius: 4px;
        }
        .modal-buttons {
            display: flex;
            gap: 10px;
            justify-content: center;
            flex-wrap: wrap;
        }
        .reclassify-section {
            margin-top: 15px;
            padding: 15px;
            background: #3d3d3d;
            border-radius: 4px;
            display: none;
        }
        .reclassify-section.active {
            display: block;
        }

        .loading {
            text-align: center;
            padding: 20px;
            font-size: 18px;
            color: #999;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🏷️  Categorization Review UI (Optimized)</h1>
        <div id="project-name"></div>
        <div class="stats">
            <div class="stat">
                <div class="stat-value" id="stat-total">0</div>
                <div class="stat-label">Total</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="stat-filtered">0</div>
                <div class="stat-label">Filtered</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="stat-reviewed" style="color: #2196F3;">0</div>
                <div class="stat-label">Reviewed</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="stat-pending" style="color: #FFC107;">0</div>
                <div class="stat-label">Pending</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="stat-kept" style="color: #4CAF50;">0</div>
                <div class="stat-label">Kept</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="stat-discarded" style="color: #f44336;">0</div>
                <div class="stat-label">Discarded</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="stat-reclassified" style="color: #FF9800;">0</div>
                <div class="stat-label">Reclassified</div>
            </div>
        </div>
    </div>

    <div class="controls">
        <div class="control-row">
            <strong>Filter by Category:</strong>
            <div class="filter-tabs" id="category-filters"></div>
        </div>
        <div class="control-row">
            <strong>Filter by Status:</strong>
            <div class="filter-tabs">
                <div class="filter-tab active" data-filter="all">All</div>
                <div class="filter-tab" data-filter="pending">Pending</div>
                <div class="filter-tab" data-filter="reviewed">Reviewed</div>
            </div>
        </div>
        <div class="control-row">
            <button class="btn btn-info" onclick="refreshInstances()">🔄 Refresh</button>
            <button class="btn btn-secondary" onclick="selectAll()">Select All (Page)</button>
            <button class="btn btn-secondary" onclick="deselectAll()">Deselect All</button>
            <button class="btn" onclick="batchKeep()">Keep Selected</button>
            <button class="btn btn-danger" onclick="batchDiscard()">Discard Selected</button>
            <select id="batch-category">
                {% for cat in categories %}
                <option value="{{ cat }}">{{ cat }}</option>
                {% endfor %}
            </select>
            <button class="btn btn-warning" onclick="batchReclassify()">Reclassify Selected</button>
            <button class="btn btn-secondary" onclick="exportResults()">💾 Export Results</button>
        </div>
    </div>

    <div class="pagination">
        <button class="btn btn-secondary" onclick="goToPage(1)" id="btn-first">First</button>
        <button class="btn btn-secondary" onclick="previousPage()" id="btn-prev">Previous</button>
        <span class="page-info">Page <span id="current-page">1</span> of <span id="total-pages">1</span> (<span id="page-count">0</span> items)</span>
        <button class="btn btn-secondary" onclick="nextPage()" id="btn-next">Next</button>
        <button class="btn btn-secondary" onclick="goToPage(totalPages)" id="btn-last">Last</button>
    </div>

    <div class="grid" id="instances-grid"></div>

    <div class="pagination">
        <button class="btn btn-secondary" onclick="goToPage(1)">First</button>
        <button class="btn btn-secondary" onclick="previousPage()">Previous</button>
        <span class="page-info">Page <span id="current-page-bottom">1</span> of <span id="total-pages-bottom">1</span></span>
        <button class="btn btn-secondary" onclick="nextPage()">Next</button>
        <button class="btn btn-secondary" onclick="goToPage(totalPages)">Last</button>
    </div>

    <div class="modal" id="modal">
        <div class="modal-content">
            <img class="modal-img" id="modal-img" src="">
            <div class="modal-controls">
                <div class="modal-info">
                    <div style="margin-bottom: 5px;"><strong>Filename:</strong> <span id="modal-filename"></span></div>
                    <div style="margin-bottom: 5px;"><strong>Original Category:</strong> <span id="modal-original-cat"></span></div>
                    <div style="margin-bottom: 5px;"><strong>Confidence:</strong> <span id="modal-confidence"></span></div>
                    <div><strong>Current Category:</strong> <span id="modal-current-cat"></span></div>
                </div>
                <div class="modal-buttons">
                    <button class="btn" onclick="modalKeep()">✓ Keep (K)</button>
                    <button class="btn btn-danger" onclick="modalDiscard()">✗ Discard (D)</button>
                    <button class="btn btn-warning" onclick="toggleReclassify()">↻ Re-classify (C)</button>
                    <button class="btn btn-secondary" onclick="closeModal()">Close (Esc)</button>
                </div>
                <div class="reclassify-section" id="reclassify-section">
                    <strong>Choose new category:</strong>
                    <div style="display: flex; gap: 10px; margin-top: 10px; flex-wrap: wrap;">
                        {% for cat in categories %}
                        <button class="btn btn-secondary" onclick="modalReclassify('{{ cat }}')">{{ cat }}</button>
                        {% endfor %}
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let allInstances = [];  // All instances on current page
        let categories = [];
        let selectedInstances = new Set();
        let currentCategoryFilter = 'all';
        let currentStatusFilter = 'all';
        let currentPage = 1;
        let totalPages = 1;
        let perPage = 100;
        let currentModalIndex = -1;

        // Lazy loading observer
        let imageObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const img = entry.target;
                    const src = img.getAttribute('data-src');
                    if (src) {
                        img.src = src;
                        img.removeAttribute('data-src');
                        imageObserver.unobserve(img);
                    }
                }
            });
        }, {
            rootMargin: '50px'
        });

        async function loadInstances() {
            const params = new URLSearchParams({
                page: currentPage,
                per_page: perPage,
                category: currentCategoryFilter,
                status: currentStatusFilter
            });

            const response = await fetch('/api/instances?' + params);
            const data = await response.json();

            allInstances = data.instances;
            categories = data.categories;
            totalPages = data.pagination.total_pages;

            if (data.project) {
                document.getElementById('project-name').textContent = 'Project: ' + data.project;
            }

            updateStats(data.stats);
            updatePagination(data.pagination);
            renderCategoryFilters(data.category_counts);
            renderGrid();
        }

        function updatePagination(pagination) {
            currentPage = pagination.page;
            totalPages = pagination.total_pages;

            document.getElementById('current-page').textContent = currentPage;
            document.getElementById('total-pages').textContent = totalPages;
            document.getElementById('current-page-bottom').textContent = currentPage;
            document.getElementById('total-pages-bottom').textContent = totalPages;
            document.getElementById('page-count').textContent = pagination.total;

            // Disable buttons
            document.getElementById('btn-first').disabled = currentPage <= 1;
            document.getElementById('btn-prev').disabled = currentPage <= 1;
            document.getElementById('btn-next').disabled = currentPage >= totalPages;
            document.getElementById('btn-last').disabled = currentPage >= totalPages;
        }

        function goToPage(page) {
            if (page < 1 || page > totalPages) return;
            currentPage = page;
            selectedInstances.clear();
            loadInstances();
        }

        function nextPage() {
            if (currentPage < totalPages) {
                currentPage++;
                selectedInstances.clear();
                loadInstances();
            }
        }

        function previousPage() {
            if (currentPage > 1) {
                currentPage--;
                selectedInstances.clear();
                loadInstances();
            }
        }

        function renderCategoryFilters(categoryCounts) {
            const container = document.getElementById('category-filters');
            container.innerHTML = '<div class="filter-tab active" data-category="all">All</div>';

            categories.forEach(cat => {
                const count = categoryCounts[cat] || 0;
                const tab = document.createElement('div');
                tab.className = 'filter-tab';
                if (cat === currentCategoryFilter) {
                    tab.classList.add('active');
                }
                tab.dataset.category = cat;
                tab.textContent = `${cat} (${count})`;
                tab.onclick = () => {
                    document.querySelectorAll('[data-category]').forEach(t => t.classList.remove('active'));
                    tab.classList.add('active');
                    currentCategoryFilter = cat;
                    currentPage = 1;
                    selectedInstances.clear();
                    loadInstances();
                };
                container.appendChild(tab);
            });
        }

        function updateStats(stats) {
            document.getElementById('stat-total').textContent = stats.total;
            document.getElementById('stat-filtered').textContent = stats.filtered;
            document.getElementById('stat-reviewed').textContent = stats.reviewed;
            document.getElementById('stat-pending').textContent = stats.pending;
            document.getElementById('stat-kept').textContent = stats.kept;
            document.getElementById('stat-discarded').textContent = stats.discarded;
            document.getElementById('stat-reclassified').textContent = stats.reclassified;
        }

        function renderGrid() {
            const grid = document.getElementById('instances-grid');
            grid.innerHTML = '';

            if (allInstances.length === 0) {
                grid.innerHTML = '<div class="loading">No instances found with current filters.</div>';
                return;
            }

            allInstances.forEach((inst, index) => {
                const card = document.createElement('div');
                card.className = 'instance-card ' + inst.status;
                if (selectedInstances.has(inst.filename)) {
                    card.classList.add('selected');
                }

                let actionBadge = '';
                if (inst.action) {
                    actionBadge = `<div class="action-indicator action-${inst.action}">${inst.action.toUpperCase()}</div>`;
                }

                const imgSrc = `/api/image/${inst.original_category}/${inst.filename}`;

                card.innerHTML = `
                    ${actionBadge}
                    <div class="instance-status status-${inst.status}">${inst.status.toUpperCase()}</div>
                    <img class="instance-img" data-src="${imgSrc}" alt="${inst.filename}">
                    <div class="instance-info">
                        <div class="instance-category">
                            <span class="category-badge">${inst.current_category}</span>
                            <span class="confidence">${(inst.confidence * 100).toFixed(0)}%</span>
                        </div>
                        <div style="font-size: 10px; color: #666; overflow: hidden; text-overflow: ellipsis;">${inst.filename}</div>
                    </div>
                `;

                card.onclick = (e) => {
                    if (e.shiftKey) {
                        toggleSelect(inst.filename);
                        renderGrid();
                    } else {
                        openModal(index);
                    }
                };

                grid.appendChild(card);

                // Lazy load image
                const img = card.querySelector('.instance-img');
                imageObserver.observe(img);
            });
        }

        function toggleSelect(filename) {
            if (selectedInstances.has(filename)) {
                selectedInstances.delete(filename);
            } else {
                selectedInstances.add(filename);
            }
        }

        function selectAll() {
            allInstances.forEach(inst => selectedInstances.add(inst.filename));
            renderGrid();
        }

        function deselectAll() {
            selectedInstances.clear();
            renderGrid();
        }

        async function reviewInstance(filename, action, newCategory = null) {
            await fetch('/api/review', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    filename: filename,
                    action: action,
                    new_category: newCategory
                })
            });
            await loadInstances();
        }

        async function batchKeep() {
            if (selectedInstances.size === 0) return;
            await fetch('/api/batch', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    filenames: Array.from(selectedInstances),
                    action: 'keep'
                })
            });
            selectedInstances.clear();
            await loadInstances();
        }

        async function batchDiscard() {
            if (selectedInstances.size === 0) return;
            await fetch('/api/batch', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    filenames: Array.from(selectedInstances),
                    action: 'discard'
                })
            });
            selectedInstances.clear();
            await loadInstances();
        }

        async function batchReclassify() {
            if (selectedInstances.size === 0) return;
            const newCategory = document.getElementById('batch-category').value;
            await fetch('/api/batch', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    filenames: Array.from(selectedInstances),
                    action: 'reclassify',
                    new_category: newCategory
                })
            });
            selectedInstances.clear();
            await loadInstances();
        }

        async function refreshInstances() {
            await fetch('/api/refresh', {method: 'POST'});
            await loadInstances();
        }

        function openModal(index) {
            currentModalIndex = index;
            const inst = allInstances[index];
            document.getElementById('modal-img').src = `/api/image/${inst.original_category}/${inst.filename}`;
            document.getElementById('modal-filename').textContent = inst.filename;
            document.getElementById('modal-original-cat').textContent = inst.original_category;
            document.getElementById('modal-current-cat').textContent = inst.current_category;
            document.getElementById('modal-confidence').textContent = (inst.confidence * 100).toFixed(1) + '%';
            document.getElementById('modal').classList.add('active');
            document.getElementById('reclassify-section').classList.remove('active');
        }

        function closeModal() {
            document.getElementById('modal').classList.remove('active');
        }

        async function modalKeep() {
            const inst = allInstances[currentModalIndex];
            await reviewInstance(inst.filename, 'keep');
            nextImage();
        }

        async function modalDiscard() {
            const inst = allInstances[currentModalIndex];
            await reviewInstance(inst.filename, 'discard');
            nextImage();
        }

        function toggleReclassify() {
            const section = document.getElementById('reclassify-section');
            section.classList.toggle('active');
        }

        async function modalReclassify(newCategory) {
            const inst = allInstances[currentModalIndex];
            await reviewInstance(inst.filename, 'reclassify', newCategory);
            document.getElementById('reclassify-section').classList.remove('active');
            nextImage();
        }

        function nextImage() {
            if (currentModalIndex < allInstances.length - 1) {
                openModal(currentModalIndex + 1);
            } else if (currentPage < totalPages) {
                // Go to next page
                closeModal();
                nextPage();
            } else {
                closeModal();
            }
        }

        function previousImage() {
            if (currentModalIndex > 0) {
                openModal(currentModalIndex - 1);
            } else if (currentPage > 1) {
                // Go to previous page
                closeModal();
                previousPage();
            }
        }

        async function exportResults() {
            if (!confirm('Export all reviewed results? This will copy files to output directory.')) {
                return;
            }
            const response = await fetch('/api/export', {method: 'POST'});
            const data = await response.json();
            alert(`✅ Export complete!\\n\\nKept: ${data.stats.kept}\\nDiscarded: ${data.stats.discarded}\\nReclassified: ${data.stats.reclassified}\\n\\nReport: ${data.report_path}`);
        }

        // Status filter tabs
        document.querySelectorAll('[data-filter]').forEach(tab => {
            tab.onclick = () => {
                document.querySelectorAll('[data-filter]').forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                currentStatusFilter = tab.dataset.filter;
                currentPage = 1;
                selectedInstances.clear();
                loadInstances();
            };
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (document.getElementById('modal').classList.contains('active')) {
                if (e.key === 'k') modalKeep();
                else if (e.key === 'd') modalDiscard();
                else if (e.key === 'c') toggleReclassify();
                else if (e.key === 'ArrowRight') nextImage();
                else if (e.key === 'ArrowLeft') previousImage();
                else if (e.key === 'Escape') closeModal();
            }
        });

        // Load on start
        loadInstances();
    </script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(
        description="Categorization Review UI (Optimized for Performance)"
    )
    parser.add_argument(
        "--categorized-dir",
        type=str,
        required=True,
        help="Directory with categorized instances"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save reviewed instances"
    )
    parser.add_argument(
        "--project",
        type=str,
        help="Project/film name"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5555,
        help="Web server port (default: 5555)"
    )

    args = parser.parse_args()

    # Create and run app
    app = CategorizationReviewApp(
        categorized_dir=Path(args.categorized_dir),
        output_dir=Path(args.output_dir),
        project=args.project,
        port=args.port
    )

    app.run()


if __name__ == "__main__":
    main()
