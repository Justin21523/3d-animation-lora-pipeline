#!/usr/bin/env python3
"""
Instance Filter UI

Purpose: Interactive web UI for filtering SAM2 instances before inpainting
Features: Grid view, keyboard shortcuts, batch operations, auto-save, statistics
Use Cases: Manual review to keep only characters, discard objects/furniture/background

Usage:
    python instance_filter_ui.py \
        --instances-dir /path/to/instances \
        --output-dir /path/to/filtered \
        --project luca \
        --port 5000

Then open browser to http://localhost:5000
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from flask import Flask, render_template_string, request, jsonify, send_from_directory
import base64


class InstanceFilterApp:
    """Flask app for interactive instance filtering"""

    def __init__(
        self,
        instances_dir: Path,
        output_dir: Path,
        project: Optional[str] = None,
        port: int = 5000
    ):
        """
        Initialize instance filter app

        Args:
            instances_dir: Directory with SAM2 instances
            output_dir: Directory to save filtered instances
            project: Project name
            port: Server port
        """
        self.instances_dir = Path(instances_dir)
        self.output_dir = Path(output_dir)
        self.project = project
        self.port = port

        # Create output directories
        self.kept_dir = self.output_dir / "kept"
        self.discarded_dir = self.output_dir / "discarded"
        self.kept_dir.mkdir(parents=True, exist_ok=True)
        self.discarded_dir.mkdir(parents=True, exist_ok=True)

        # Load or initialize state
        self.state_file = self.output_dir / "filter_state.json"
        self.state = self.load_state()

        # Find all instances
        self.instances = self.scan_instances()

        # Create Flask app
        self.app = Flask(__name__)
        self.setup_routes()

    def scan_instances(self) -> List[Dict]:
        """
        Scan instances directory

        Returns:
            List of instance metadata
        """
        instances = []
        image_files = sorted(
            list(self.instances_dir.glob("*.png")) +
            list(self.instances_dir.glob("*.jpg"))
        )

        for img_path in image_files:
            filename = img_path.name

            # Check current status
            if filename in self.state["kept"]:
                status = "kept"
                category = self.state["categories"].get(filename, "character")
            elif filename in self.state["discarded"]:
                status = "discarded"
                category = None
            else:
                status = "pending"
                category = None

            instances.append({
                "filename": filename,
                "path": str(img_path),
                "status": status,
                "category": category,
            })

        return instances

    def load_state(self) -> Dict:
        """
        Load saved state

        Returns:
            State dictionary
        """
        if self.state_file.exists():
            with open(self.state_file) as f:
                return json.load(f)
        else:
            return {
                "kept": [],
                "discarded": [],
                "categories": {},  # filename -> category mapping
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
            return render_template_string(HTML_TEMPLATE)

        @self.app.route('/api/instances')
        def get_instances():
            """Get all instances with status"""
            # Rescan to get fresh data
            self.instances = self.scan_instances()

            stats = {
                "total": len(self.instances),
                "kept": len([i for i in self.instances if i["status"] == "kept"]),
                "discarded": len([i for i in self.instances if i["status"] == "discarded"]),
                "pending": len([i for i in self.instances if i["status"] == "pending"]),
            }

            return jsonify({
                "instances": self.instances,
                "stats": stats,
                "project": self.project,
            })

        @self.app.route('/api/image/<path:filename>')
        def get_image(filename):
            """Serve instance image"""
            return send_from_directory(self.instances_dir, filename)

        @self.app.route('/api/mark', methods=['POST'])
        def mark_instance():
            """Mark instance as kept/discarded"""
            data = request.json
            filename = data.get("filename")
            action = data.get("action")  # keep, discard
            category = data.get("category", "character")

            if not filename:
                return jsonify({"error": "Missing filename"}), 400

            src_path = self.instances_dir / filename

            if not src_path.exists():
                return jsonify({"error": "File not found"}), 404

            # Update state
            if action == "keep":
                if filename not in self.state["kept"]:
                    self.state["kept"].append(filename)
                if filename in self.state["discarded"]:
                    self.state["discarded"].remove(filename)
                self.state["categories"][filename] = category

                # Copy to kept directory
                dst_path = self.kept_dir / filename
                shutil.copy2(src_path, dst_path)

            elif action == "discard":
                if filename not in self.state["discarded"]:
                    self.state["discarded"].append(filename)
                if filename in self.state["kept"]:
                    self.state["kept"].remove(filename)
                if filename in self.state["categories"]:
                    del self.state["categories"][filename]

                # Copy to discarded directory
                dst_path = self.discarded_dir / filename
                shutil.copy2(src_path, dst_path)

            elif action == "reset":
                if filename in self.state["kept"]:
                    self.state["kept"].remove(filename)
                if filename in self.state["discarded"]:
                    self.state["discarded"].remove(filename)
                if filename in self.state["categories"]:
                    del self.state["categories"][filename]

            self.save_state()

            return jsonify({"success": True})

        @self.app.route('/api/batch', methods=['POST'])
        def batch_operation():
            """Batch mark multiple instances"""
            data = request.json
            filenames = data.get("filenames", [])
            action = data.get("action")
            category = data.get("category", "character")

            for filename in filenames:
                src_path = self.instances_dir / filename

                if not src_path.exists():
                    continue

                # Update state
                if action == "keep":
                    if filename not in self.state["kept"]:
                        self.state["kept"].append(filename)
                    if filename in self.state["discarded"]:
                        self.state["discarded"].remove(filename)
                    self.state["categories"][filename] = category

                    dst_path = self.kept_dir / filename
                    shutil.copy2(src_path, dst_path)

                elif action == "discard":
                    if filename not in self.state["discarded"]:
                        self.state["discarded"].append(filename)
                    if filename in self.state["kept"]:
                        self.state["kept"].remove(filename)
                    if filename in self.state["categories"]:
                        del self.state["categories"][filename]

                    dst_path = self.discarded_dir / filename
                    shutil.copy2(src_path, dst_path)

            self.save_state()

            return jsonify({"success": True, "count": len(filenames)})

        @self.app.route('/api/export', methods=['POST'])
        def export_results():
            """Export final filtered results"""
            # Save final report
            report = {
                "project": self.project,
                "instances_dir": str(self.instances_dir),
                "output_dir": str(self.output_dir),
                "total_instances": len(self.instances),
                "kept": len(self.state["kept"]),
                "discarded": len(self.state["discarded"]),
                "kept_files": self.state["kept"],
                "discarded_files": self.state["discarded"],
                "categories": self.state["categories"],
                "timestamp": datetime.now().isoformat(),
            }

            report_path = self.output_dir / "filtering_report.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)

            return jsonify({
                "success": True,
                "report_path": str(report_path),
                "stats": {
                    "total": report["total_instances"],
                    "kept": report["kept"],
                    "discarded": report["discarded"],
                }
            })

    def run(self):
        """Start the Flask server"""
        print("\n" + "="*70)
        print("INSTANCE FILTER UI")
        print("="*70)
        print(f"Instances: {self.instances_dir}")
        print(f"Output:    {self.output_dir}")
        if self.project:
            print(f"Project:   {self.project}")
        print(f"\nTotal instances: {len(self.instances)}")
        print(f"Already reviewed: {len(self.state['kept']) + len(self.state['discarded'])}")
        print(f"Pending: {len(self.instances) - len(self.state['kept']) - len(self.state['discarded'])}")
        print("="*70)
        print(f"\n🌐 Open in browser: http://localhost:{self.port}")
        print("\nKeyboard shortcuts:")
        print("  K = Keep instance")
        print("  D = Discard instance")
        print("  R = Reset/Undo")
        print("  → = Next image")
        print("  ← = Previous image")
        print("\nPress Ctrl+C to stop server")
        print("="*70 + "\n")

        self.app.run(host='0.0.0.0', port=self.port, debug=False)


# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Instance Filter UI</title>
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
            gap: 20px;
            margin-top: 15px;
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
            display: flex;
            gap: 10px;
            align-items: center;
            flex-wrap: wrap;
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
        .btn-secondary { background: #555; }
        .btn-secondary:hover { background: #666; }

        .filter-tabs {
            display: flex;
            gap: 5px;
        }
        .filter-tab {
            background: #3d3d3d;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
        }
        .filter-tab.active {
            background: #4CAF50;
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
        .instance-card.kept {
            outline: 2px solid #4CAF50;
        }
        .instance-card.discarded {
            outline: 2px solid #f44336;
            opacity: 0.5;
        }
        .instance-img {
            width: 100%;
            height: 200px;
            object-fit: contain;
            background: #1a1a1a;
        }
        .instance-info {
            padding: 10px;
            font-size: 12px;
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
        .status-kept { color: #4CAF50; }
        .status-discarded { color: #f44336; }
        .status-pending { color: #FFC107; }

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
        }
        .modal.active {
            display: flex;
        }
        .modal-content {
            max-width: 90%;
            max-height: 90%;
            position: relative;
        }
        .modal-img {
            max-width: 100%;
            max-height: 80vh;
            object-fit: contain;
        }
        .modal-controls {
            background: #2d2d2d;
            padding: 20px;
            margin-top: 10px;
            border-radius: 8px;
            display: flex;
            gap: 10px;
            justify-content: center;
        }
        .keyboard-hint {
            background: #3d3d3d;
            padding: 10px;
            border-radius: 4px;
            font-size: 11px;
            color: #999;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎬 Instance Filter UI</h1>
        <div id="project-name"></div>
        <div class="stats">
            <div class="stat">
                <div class="stat-value" id="stat-total">0</div>
                <div class="stat-label">Total</div>
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
                <div class="stat-value" id="stat-pending" style="color: #FFC107;">0</div>
                <div class="stat-label">Pending</div>
            </div>
        </div>
    </div>

    <div class="controls">
        <div class="filter-tabs">
            <div class="filter-tab active" data-filter="all">All</div>
            <div class="filter-tab" data-filter="pending">Pending</div>
            <div class="filter-tab" data-filter="kept">Kept</div>
            <div class="filter-tab" data-filter="discarded">Discarded</div>
        </div>
        <button class="btn btn-secondary" onclick="selectAll()">Select All</button>
        <button class="btn btn-secondary" onclick="deselectAll()">Deselect All</button>
        <button class="btn" onclick="batchKeep()">Keep Selected</button>
        <button class="btn btn-danger" onclick="batchDiscard()">Discard Selected</button>
        <button class="btn btn-secondary" onclick="exportResults()">Export Results</button>
        <div class="keyboard-hint">
            Keyboard: K=Keep | D=Discard | R=Reset | ←→=Navigate
        </div>
    </div>

    <div class="grid" id="instances-grid"></div>

    <div class="modal" id="modal">
        <div class="modal-content">
            <img class="modal-img" id="modal-img" src="">
            <div class="modal-controls">
                <button class="btn" onclick="modalKeep()">Keep (K)</button>
                <button class="btn btn-danger" onclick="modalDiscard()">Discard (D)</button>
                <button class="btn btn-secondary" onclick="modalReset()">Reset (R)</button>
                <button class="btn btn-secondary" onclick="closeModal()">Close (Esc)</button>
            </div>
            <div style="text-align: center; margin-top: 10px; color: #999;" id="modal-filename"></div>
        </div>
    </div>

    <script>
        let instances = [];
        let selectedInstances = new Set();
        let currentFilter = 'all';
        let currentModalIndex = -1;

        async function loadInstances() {
            const response = await fetch('/api/instances');
            const data = await response.json();
            instances = data.instances;

            if (data.project) {
                document.getElementById('project-name').textContent = 'Project: ' + data.project;
            }

            updateStats(data.stats);
            renderGrid();
        }

        function updateStats(stats) {
            document.getElementById('stat-total').textContent = stats.total;
            document.getElementById('stat-kept').textContent = stats.kept;
            document.getElementById('stat-discarded').textContent = stats.discarded;
            document.getElementById('stat-pending').textContent = stats.pending;
        }

        function renderGrid() {
            const grid = document.getElementById('instances-grid');
            grid.innerHTML = '';

            const filtered = instances.filter(inst => {
                if (currentFilter === 'all') return true;
                return inst.status === currentFilter;
            });

            filtered.forEach((inst, index) => {
                const card = document.createElement('div');
                card.className = 'instance-card ' + inst.status;
                if (selectedInstances.has(inst.filename)) {
                    card.classList.add('selected');
                }

                card.innerHTML = `
                    <div class="instance-status status-${inst.status}">${inst.status.toUpperCase()}</div>
                    <img class="instance-img" src="/api/image/${inst.filename}" alt="${inst.filename}">
                    <div class="instance-info">${inst.filename}</div>
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
            const filtered = instances.filter(inst => {
                if (currentFilter === 'all') return true;
                return inst.status === currentFilter;
            });
            filtered.forEach(inst => selectedInstances.add(inst.filename));
            renderGrid();
        }

        function deselectAll() {
            selectedInstances.clear();
            renderGrid();
        }

        async function markInstance(filename, action) {
            await fetch('/api/mark', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({filename, action})
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

        function openModal(index) {
            currentModalIndex = index;
            const inst = instances[index];
            document.getElementById('modal-img').src = '/api/image/' + inst.filename;
            document.getElementById('modal-filename').textContent = inst.filename;
            document.getElementById('modal').classList.add('active');
        }

        function closeModal() {
            document.getElementById('modal').classList.remove('active');
        }

        async function modalKeep() {
            const inst = instances[currentModalIndex];
            await markInstance(inst.filename, 'keep');
            nextImage();
        }

        async function modalDiscard() {
            const inst = instances[currentModalIndex];
            await markInstance(inst.filename, 'discard');
            nextImage();
        }

        async function modalReset() {
            const inst = instances[currentModalIndex];
            await markInstance(inst.filename, 'reset');
        }

        function nextImage() {
            if (currentModalIndex < instances.length - 1) {
                openModal(currentModalIndex + 1);
            } else {
                closeModal();
            }
        }

        function previousImage() {
            if (currentModalIndex > 0) {
                openModal(currentModalIndex - 1);
            }
        }

        async function exportResults() {
            const response = await fetch('/api/export', {method: 'POST'});
            const data = await response.json();
            alert(`Exported results!\\nKept: ${data.stats.kept}\\nDiscarded: ${data.stats.discarded}\\nReport: ${data.report_path}`);
        }

        // Filter tabs
        document.querySelectorAll('.filter-tab').forEach(tab => {
            tab.onclick = () => {
                document.querySelectorAll('.filter-tab').forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                currentFilter = tab.dataset.filter;
                renderGrid();
            };
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (document.getElementById('modal').classList.contains('active')) {
                if (e.key === 'k') modalKeep();
                else if (e.key === 'd') modalDiscard();
                else if (e.key === 'r') modalReset();
                else if (e.key === 'ArrowRight') nextImage();
                else if (e.key === 'ArrowLeft') previousImage();
                else if (e.key === 'Escape') closeModal();
            }
        });

        // Load on start
        loadInstances();
        setInterval(loadInstances, 5000); // Auto-refresh every 5 seconds
    </script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(
        description="Instance Filter UI (Film-Agnostic)"
    )
    parser.add_argument(
        "--instances-dir",
        type=str,
        required=True,
        help="Directory with SAM2 instances"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save filtered instances"
    )
    parser.add_argument(
        "--project",
        type=str,
        help="Project/film name"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Web server port (default: 5000)"
    )

    args = parser.parse_args()

    # Create and run app
    app = InstanceFilterApp(
        instances_dir=Path(args.instances_dir),
        output_dir=Path(args.output_dir),
        project=args.project,
        port=args.port
    )

    app.run()


if __name__ == "__main__":
    main()
