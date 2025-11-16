#!/usr/bin/env python3
"""
Launch Interactive Clustering Review Tool

Starts a local web server to review and refine clustering results.
"""

import os
import sys
import json
import shutil
import argparse
import webbrowser
from pathlib import Path
from typing import Dict, List, Optional
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
import threading


class ClusterReviewHandler(SimpleHTTPRequestHandler):
    """Custom HTTP handler for cluster review tool"""

    def __init__(self, *args, cluster_dir=None, **kwargs):
        self.cluster_dir = cluster_dir
        super().__init__(*args, **kwargs)

    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urlparse(self.path)

        if parsed_path.path == '/api/load-clusters':
            self.handle_load_clusters(parsed_path)
        elif parsed_path.path.endswith('.html') or parsed_path.path == '/':
            self.handle_html()
        else:
            super().do_GET()

    def do_POST(self):
        """Handle POST requests"""
        if self.path == '/api/save-clusters':
            self.handle_save_clusters()
        else:
            self.send_error(404)

    def handle_html(self):
        """Serve the HTML file"""
        html_path = Path(__file__).parent / 'interactive_ui' / 'cluster_review.html'
        if html_path.exists():
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            with open(html_path, 'rb') as f:
                self.wfile.write(f.read())
        else:
            self.send_error(404, "HTML file not found")

    def handle_load_clusters(self, parsed_path):
        """Load cluster data from directory"""
        try:
            # Parse query parameters
            query = parse_qs(parsed_path.query)
            cluster_dir = query.get('path', [self.cluster_dir])[0]

            if not cluster_dir:
                self.send_error(400, "No cluster directory specified")
                return

            cluster_path = Path(cluster_dir)
            if not cluster_path.exists():
                self.send_error(404, f"Cluster directory not found: {cluster_dir}")
                return

            # Load cluster data
            clusters = self.load_cluster_structure(cluster_path)

            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            response = {
                'clusters': clusters,
                'path': str(cluster_path)
            }

            self.wfile.write(json.dumps(response).encode())

        except Exception as e:
            self.send_error(500, f"Error loading clusters: {str(e)}")

    def load_cluster_structure(self, cluster_dir: Path) -> List[Dict]:
        """Load cluster directory structure and image metadata"""
        clusters = []

        # Read cluster_report.json if exists
        report_path = cluster_dir / 'cluster_report.json'
        cluster_metadata = {}

        if report_path.exists():
            with open(report_path, 'r') as f:
                report = json.load(f)
                cluster_metadata = report.get('clusters', {})

        # Scan directories
        for subdir in sorted(cluster_dir.iterdir()):
            if not subdir.is_dir():
                continue

            cluster_id = subdir.name
            images = []

            # Load all images in cluster
            for img_path in sorted(subdir.glob('*.jpg')):
                images.append({
                    'id': f"{cluster_id}_{img_path.stem}",
                    'path': f"/images/{cluster_id}/{img_path.name}",
                    'filename': img_path.name,
                    'metadata': self.parse_filename_metadata(img_path.name)
                })

            # Get cluster metadata
            meta = cluster_metadata.get(cluster_id, {})

            cluster = {
                'id': cluster_id,
                'name': meta.get('name', cluster_id),
                'description': meta.get('description', ''),
                'images': images,
                'metadata': meta
            }

            clusters.append(cluster)

        return clusters

    def parse_filename_metadata(self, filename: str) -> Dict:
        """Parse metadata from frame filename"""
        # Example: scene0001_pos0_frame000123_t45.67s.jpg
        metadata = {}

        try:
            parts = filename.replace('.jpg', '').split('_')
            for part in parts:
                if part.startswith('scene'):
                    metadata['scene'] = int(part[5:])
                elif part.startswith('pos'):
                    metadata['position'] = int(part[3:])
                elif part.startswith('frame'):
                    metadata['frame'] = int(part[5:])
                elif part.startswith('t') and part.endswith('s'):
                    metadata['timestamp'] = part[1:-1] + 's'
        except:
            pass

        return metadata

    def handle_save_clusters(self):
        """Save cluster modifications"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode())

            cluster_path = Path(data['path'])
            clusters = data['clusters']
            changes = data['changes']

            # Save changes log
            changes_log_path = cluster_path / f'changes_{int(time.time())}.json'
            with open(changes_log_path, 'w') as f:
                json.dump(changes, f, indent=2)

            # Apply changes to directory structure
            self.apply_cluster_changes(cluster_path, clusters, changes)

            # Send success response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            response = {'status': 'success', 'changes_saved': len(changes)}
            self.wfile.write(json.dumps(response).encode())

        except Exception as e:
            self.send_error(500, f"Error saving clusters: {str(e)}")

    def apply_cluster_changes(self, cluster_dir: Path, clusters: List[Dict], changes: List[Dict]):
        """Apply changes to actual cluster directories"""
        # Create backup
        backup_dir = cluster_dir.parent / f"{cluster_dir.name}_backup_{int(time.time())}"
        shutil.copytree(cluster_dir, backup_dir)

        # Apply each change
        for change in changes:
            change_type = change['type']

            if change_type == 'rename_cluster':
                old_path = cluster_dir / change['clusterId']
                new_name = change['newName'].replace(' ', '_').lower()
                new_path = cluster_dir / new_name
                if old_path.exists():
                    old_path.rename(new_path)

            elif change_type == 'move_images':
                source_dir = cluster_dir / change['sourceClusterId']
                target_dir = cluster_dir / change['targetClusterId']
                target_dir.mkdir(exist_ok=True)

                for image_id in change['imageIds']:
                    # Find image file
                    for img_path in source_dir.glob('*.jpg'):
                        if image_id in img_path.stem:
                            shutil.move(str(img_path), str(target_dir / img_path.name))
                            break

            elif change_type == 'create_cluster':
                new_cluster = change['cluster']
                new_dir = cluster_dir / new_cluster['id']
                new_dir.mkdir(exist_ok=True)

            elif change_type == 'merge_clusters':
                merged_cluster = change['mergedCluster']
                merged_dir = cluster_dir / merged_cluster['id']
                merged_dir.mkdir(exist_ok=True)

                # Move all images to merged cluster
                for cluster_id in change['clusterIds']:
                    old_dir = cluster_dir / cluster_id
                    if old_dir.exists():
                        for img_path in old_dir.glob('*.jpg'):
                            shutil.move(str(img_path), str(merged_dir / img_path.name))
                        old_dir.rmdir()

            elif change_type == 'split_cluster':
                new_cluster = change['newCluster']
                new_dir = cluster_dir / new_cluster['id']
                new_dir.mkdir(exist_ok=True)

                source_dir = cluster_dir / change['sourceClusterId']

                for image_id in change['imageIds']:
                    for img_path in source_dir.glob('*.jpg'):
                        if image_id in img_path.stem:
                            shutil.move(str(img_path), str(new_dir / img_path.name))
                            break

        # Update cluster_report.json
        report_path = cluster_dir / 'cluster_report.json'
        report = {}

        for cluster in clusters:
            report[cluster['id']] = {
                'name': cluster['name'],
                'description': cluster['description'],
                'image_count': len(cluster['images']),
                'metadata': cluster['metadata']
            }

        with open(report_path, 'w') as f:
            json.dump({'clusters': report, 'updated': str(datetime.now())}, f, indent=2)


def start_server(cluster_dir: Path, port: int = 8000, open_browser: bool = True):
    """Start the web server"""
    # Change to UI directory to serve static files
    ui_dir = Path(__file__).parent / 'interactive_ui'
    os.chdir(ui_dir)

    # Create handler with cluster directory
    handler = lambda *args, **kwargs: ClusterReviewHandler(
        *args, cluster_dir=str(cluster_dir), **kwargs
    )

    server = HTTPServer(('localhost', port), handler)

    print(f"\n{'='*60}")
    print(f"Interactive Clustering Review Server")
    print(f"{'='*60}")
    print(f"Cluster Directory: {cluster_dir}")
    print(f"Server Address: http://localhost:{port}")
    print(f"{'='*60}\n")
    print("Press Ctrl+C to stop the server\n")

    # Open browser
    if open_browser:
        url = f"http://localhost:{port}/?path={cluster_dir}"
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
        server.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description="Launch interactive clustering review tool"
    )
    parser.add_argument(
        'cluster_dir',
        type=str,
        help='Path to clustering results directory'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Server port (default: 8000)'
    )
    parser.add_argument(
        '--no-browser',
        action='store_true',
        help='Do not open browser automatically'
    )

    args = parser.parse_args()

    cluster_path = Path(args.cluster_dir)
    if not cluster_path.exists():
        print(f"Error: Cluster directory not found: {cluster_path}")
        sys.exit(1)

    start_server(cluster_path, args.port, not args.no_browser)


if __name__ == '__main__':
    import time
    from datetime import datetime
    main()
