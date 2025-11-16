# Interactive Clustering Review Tool

A powerful web-based tool for reviewing, refining, and managing character clustering results from 3D animation frame extraction.

---

## 📖 Table of Contents

1. [Features](#features)
2. [Technical Architecture](#technical-architecture)
3. [How It Works](#how-it-works)
4. [Why We Need a Web Server](#why-we-need-a-web-server)
5. [Quick Start](#quick-start)
6. [Advanced Usage](#advanced-usage)
7. [Troubleshooting](#troubleshooting)

---

## 🎯 Technical Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────┐
│  Frontend (Browser)                                      │
│  ├─ cluster_review.html    (UI Structure)               │
│  ├─ cluster_review.js      (Interaction Logic)          │
│  └─ CSS (inline)           (Styling)                    │
└─────────────────────────────────────────────────────────┘
                          ↕ HTTP
┌─────────────────────────────────────────────────────────┐
│  Backend (Python HTTP Server)                            │
│  launch_interactive_review.py                           │
│  ├─ Serve static files (HTML/JS/CSS)                   │
│  ├─ API: /api/load-clusters                            │
│  └─ API: /api/save-clusters                            │
└─────────────────────────────────────────────────────────┘
                          ↕ File I/O
┌─────────────────────────────────────────────────────────┐
│  File System (Cluster Data)                             │
│  identity_clusters/                                     │
│  ├─ identity_000/                                       │
│  │   ├─ image1.png                                     │
│  │   └─ faces/                                         │
│  └─ identity_clustering.json                            │
└─────────────────────────────────────────────────────────┘
```

### Why We Need a Web Server

**Problem: Browser Security Restrictions**

If we use only HTML/CSS/JavaScript without a server:

```javascript
// ❌ Cannot read local file system
fetch('file:///mnt/data/clusters/')  // Blocked by CORS

// ❌ Cannot display local images
<img src="file:///mnt/data/image.png">  // May be blocked

// ❌ Cannot modify files
// JavaScript in browser cannot directly write to file system
```

**Solution: Python Web Server as Legal Bridge**

```
Browser (JavaScript)
    ↓ HTTP Request
Web Server (Python)
    ↓ Legal file system access
Local File System
    ↓
✓ Success
```

**Server Functions:**
1. 🔓 **Bypass browser security** - Legal access to local files
2. 📂 **Read directory structure** - Auto-scan cluster data
3. 🖼️ **Serve images** - Convert local images to HTTP resources
4. 💾 **Save changes** - Actually modify file system
5. 🌐 **HTTP protocol** - Enable normal browser operation

### Complete Request Flow

**Loading Clusters:**

```
1. User opens browser
   ↓
2. Visit http://localhost:8000/?path=/path/to/clusters
   ↓
3. [GET /]
   Browser → Python Server
   ↓
4. Server returns cluster_review.html
   ↓
5. Browser loads HTML + JavaScript
   ↓
6. [GET /api/load-clusters?path=...]
   JavaScript → Python Server
   ↓
7. Server reads file system
   - Scan directory structure
   - Read image list
   - Read metadata.json
   ↓
8. Server returns JSON data
   ↓
9. JavaScript renders UI (sidebar + grid)
```

**Loading Images:**

```
10. Frontend renders <img src="http://localhost:8000/images/identity_000/img1.png">
    ↓
11. [GET /images/identity_000/img1.png]
    Browser → Python Server
    ↓
12. Server reads actual file
    /mnt/data/.../identity_000/img1.png
    ↓
13. Server returns image binary data
    ↓
14. Browser displays image ✓
```

**Saving Changes:**

```
15. User operations (move images, rename clusters)
    ↓
16. JavaScript records all changes
    changes = [
      {type: 'move_images', ...},
      {type: 'rename_cluster', ...}
    ]
    ↓
17. User clicks "Save"
    ↓
18. [POST /api/save-clusters]
    JavaScript → Python Server
    ↓
19. Server creates backup
    ↓
20. Server executes file operations
    - shutil.move() to move images
    - Path.rename() to rename directories
    ↓
21. Server returns {status: 'success'}
    ↓
22. JavaScript shows "✓ Saved!"
```

---

## Features

### 🎯 Core Functionality
- **Visual Cluster Browser**: Browse all clusters with image thumbnails
- **Real-time Image Selection**: Multi-select images with checkboxes
- **Drag & Move**: Move images between clusters
- **Cluster Operations**:
  - ✏️ Rename clusters
  - 🔀 Merge multiple clusters
  - ✂️ Split clusters
  - ➕ Create new clusters
  - 🗑️ Delete/move to noise cluster

### 🔍 Advanced Features
- **Search & Filter**: Search clusters by name or description
- **Keyboard Shortcuts**:
  - `Ctrl+A`: Select all images in current cluster
  - `Ctrl+Z`: Undo last change
  - `Ctrl+Y`: Redo
  - `Ctrl+S`: Save changes
  - `Delete`: Move selected images to noise cluster
- **Change Tracking**: Track all modifications with undo/redo
- **Statistics Dashboard**: Live stats on clusters, images, and changes

### 💾 Data Management
- **Auto-save**: Export changes as JSON
- **Backup**: Automatic backup before applying changes
- **Change Log**: Detailed log of all modifications

## Quick Start

### 1. Launch the Interactive Review Tool

```bash
# Start the web server with your clustering results
conda run -n ai_env python scripts/generic/clustering/launch_interactive_review.py \
  /mnt/data/ai_data/datasets/3d-anime/luca/clustered \
  --port 8000
```

This will:
- Start a local web server on port 8000
- Automatically open your browser to the review interface
- Load all clusters from the specified directory

### 2. Review and Refine Clusters

#### Select a Cluster
- Click on a cluster in the left sidebar to view its images
- Images will display in a responsive grid

#### Select Images
- Click on images to select/deselect them
- Click the checkbox to toggle selection without viewing
- Use `Ctrl+A` to select all images in the current cluster

#### Move Images Between Clusters
1. Select images you want to move
2. Click "Move Selected" button
3. Choose target cluster from the modal
4. Click "Move" to confirm

#### Merge Clusters
1. `Ctrl+Click` on 2 or more clusters in the sidebar to multi-select
2. Click "Merge Clusters" button in toolbar
3. Enter a name for the merged cluster
4. Confirm to merge all selected clusters

#### Split a Cluster
1. Select images you want to split off
2. Click "Split Cluster" button
3. Enter a name for the new cluster
4. The selected images will be moved to the new cluster

#### Rename a Cluster
1. Select the cluster
2. Click "Rename" button
3. Enter new name
4. Press Enter to confirm

#### Delete Images
1. Select unwanted images
2. Press `Delete` key or click "Delete" button
3. Images will be moved to the "Noise / Unclassified" cluster

### 3. Save Your Changes

When you're done reviewing:

1. Click the "Save Changes" button in the toolbar
2. Changes will be applied to the directory structure
3. A backup will be created automatically
4. A change log JSON will be saved

## Directory Structure

### Input Structure (Clustering Results)

```
clustered/
├── character_0/           # First cluster
│   ├── frame0001.jpg
│   ├── frame0023.jpg
│   └── ...
├── character_1/           # Second cluster
│   └── ...
├── noise/                 # Unclassified images
│   └── ...
└── cluster_report.json    # Cluster metadata
```

### After Review (Modified Structure)

```
clustered/
├── luca_human_form/       # Renamed from character_0
│   └── ...
├── alberto_sea_monster/   # Renamed from character_1
│   └── ...
├── noise/
│   └── ...
├── cluster_report.json    # Updated metadata
├── changes_1699999999.json  # Change log
└── clustered_backup_1699999999/  # Backup before changes
```

---

## 🔧 How It Works - Detailed Technical Explanation

### Frontend State Management

The entire application state is managed in JavaScript:

```javascript
class ClusterReviewApp {
    constructor() {
        // Application state
        this.clusters = [];              // All cluster data
        this.currentCluster = null;      // Currently displayed cluster
        this.selectedImages = new Set(); // Selected image IDs
        this.selectedClustersForMerge = new Set(); // Clusters to merge
        this.changes = [];               // Change history
        this.undoStack = [];             // Undo stack
        this.redoStack = [];             // Redo stack
    }
}
```

### Key Operations Explained

#### 1. Image Selection

```javascript
// When user clicks an image
toggleImageSelection(imageId) {
    if (this.selectedImages.has(imageId)) {
        // Deselect
        this.selectedImages.delete(imageId);
    } else {
        // Select
        this.selectedImages.add(imageId);
    }

    // Re-render (update visual state)
    this.renderImageGrid();

    // Update button states
    this.updateButtons();
}
```

**Visual Feedback:**
```css
/* Selected image styling */
.image-card.selected {
    border-color: #667eea;  /* Blue border */
    transform: scale(1.08);  /* Slightly larger */
}

.image-checkbox.checked {
    background: #667eea;     /* Blue background */
    color: white;
}
```

#### 2. Moving Images Between Clusters

```javascript
// Show modal with target cluster list
showMoveImagesModal() {
    const targetList = document.getElementById('targetClusterList');

    // List all other clusters
    this.clusters.forEach(cluster => {
        if (cluster.id === this.currentCluster.id) return;

        const item = document.createElement('div');
        item.innerHTML = `
            <div>${cluster.name}</div>
            <div>${cluster.images.length} images</div>
        `;
        item.onclick = () => selectTarget(cluster.id);
        targetList.appendChild(item);
    });
}

// Execute move operation
moveImages() {
    const targetClusterId = getSelectedTarget();
    const imagesToMove = Array.from(this.selectedImages);

    // Record this change (for undo)
    this.recordChange({
        type: 'move_images',
        sourceClusterId: this.currentCluster.id,
        targetClusterId: targetClusterId,
        imageIds: imagesToMove
    });

    // Move images in memory
    imagesToMove.forEach(imageId => {
        // Remove from source
        const idx = this.currentCluster.images.findIndex(
            img => img.id === imageId
        );
        const image = this.currentCluster.images.splice(idx, 1)[0];

        // Add to target
        const targetCluster = this.clusters.find(c => c.id === targetClusterId);
        targetCluster.images.push(image);
    });

    // Re-render UI
    this.renderClusterList();
    this.renderImageGrid();
}
```

#### 3. Merging Clusters

```javascript
// Ctrl+Click to multi-select clusters
toggleClusterForMerge(cluster) {
    if (this.selectedClustersForMerge.has(cluster.id)) {
        this.selectedClustersForMerge.delete(cluster.id);
    } else {
        this.selectedClustersForMerge.add(cluster.id);
    }
    this.renderClusterList();  // Mark visually (yellow)
}

// Merge operation
mergeClusters() {
    const clusterIds = Array.from(this.selectedClustersForMerge);
    const clustersToMerge = this.clusters.filter(c => clusterIds.includes(c.id));

    // Create merged cluster
    const mergedCluster = {
        id: `cluster_${Date.now()}`,
        name: prompt('Enter merged cluster name:'),
        images: [],
        metadata: {}
    };

    // Collect all images
    clustersToMerge.forEach(cluster => {
        mergedCluster.images.push(...cluster.images);
    });

    // Remove old clusters
    this.clusters = this.clusters.filter(c => !clusterIds.includes(c.id));

    // Add new cluster
    this.clusters.push(mergedCluster);

    // Record change
    this.recordChange({
        type: 'merge_clusters',
        clusterIds: clusterIds,
        mergedCluster: mergedCluster
    });
}
```

### Backend File Operations

When user clicks "Save", the backend performs actual file system operations:

```python
# launch_interactive_review.py
def handle_save_clusters(self):
    # 1. Receive data from frontend
    data = json.loads(request_body)
    cluster_path = Path(data['path'])
    changes = data['changes']

    # 2. Create backup
    backup_dir = cluster_path.parent / f"{cluster_path.name}_backup_{timestamp}"
    shutil.copytree(cluster_path, backup_dir)

    # 3. Apply each change to actual file system
    for change in changes:
        if change['type'] == 'move_images':
            # Actually move image files
            source_dir = cluster_path / change['sourceClusterId']
            target_dir = cluster_path / change['targetClusterId']

            for image_id in change['imageIds']:
                # Find corresponding image file
                for img_file in source_dir.glob('*.png'):
                    if image_id in img_file.stem:
                        # Move file
                        shutil.move(str(img_file), str(target_dir / img_file.name))

        elif change['type'] == 'rename_cluster':
            # Rename actual directory
            old_dir = cluster_path / change['clusterId']
            new_name = change['newName'].replace(' ', '_').lower()
            new_dir = cluster_path / new_name
            old_dir.rename(new_dir)

        elif change['type'] == 'merge_clusters':
            # Merge: move all images to new cluster
            merged_dir = cluster_path / change['mergedCluster']['id']
            merged_dir.mkdir(exist_ok=True)

            for cluster_id in change['clusterIds']:
                old_dir = cluster_path / cluster_id
                for img_file in old_dir.glob('*.png'):
                    shutil.move(str(img_file), str(merged_dir / img_file.name))
                old_dir.rmdir()  # Remove empty directory

    # 4. Update metadata JSON
    metadata = {
        'clusters': {
            cluster['id']: {
                'name': cluster['name'],
                'image_count': len(cluster['images'])
            }
            for cluster in clusters
        },
        'updated': datetime.now().isoformat()
    }

    with open(cluster_path / 'cluster_report.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    # 5. Return success
    return {'status': 'success', 'changes_saved': len(changes)}
```

### Change Tracking System

Every operation is recorded for undo/redo and audit trail:

```javascript
recordChange(change) {
    // Add timestamp
    change.timestamp = new Date().toISOString();

    // Add to changes list (for saving)
    this.changes.push(change);

    // Add to undo stack (for undo functionality)
    this.undoStack.push(change);

    // Clear redo stack
    this.redoStack = [];

    // Update statistics display
    this.updateStats();
}
```

**Change Format:**
```javascript
{
    type: 'move_images',
    sourceClusterId: 'identity_000',
    targetClusterId: 'identity_001',
    imageIds: ['img1', 'img2', 'img3'],
    timestamp: '2025-11-09T02:30:00.000Z'
}
```

### Keyboard Shortcuts Implementation

```javascript
document.addEventListener('keydown', (e) => {
    if (e.ctrlKey || e.metaKey) {
        switch(e.key) {
            case 'a':  // Ctrl+A: Select all
                e.preventDefault();
                this.selectAllImages();
                break;
            case 'z':  // Ctrl+Z: Undo
                e.preventDefault();
                this.undo();
                break;
            case 'y':  // Ctrl+Y: Redo
                e.preventDefault();
                this.redo();
                break;
            case 's':  // Ctrl+S: Save
                e.preventDefault();
                this.saveChanges();
                break;
        }
    } else if (e.key === 'Delete') {
        this.deleteSelected();  // Delete: Move to noise
    }
});
```

### Example: Complete User Flow

**Scenario: Move misclassified images to correct cluster**

```
1. User: Click "identity_000" cluster in sidebar
   ↓
   Frontend: Call selectCluster(identity_000)
   ↓
   Frontend: Render all 45 images in grid

2. User: See 3 images are Alberto, not Luca
   ↓
   User: Ctrl+Click these 3 images
   ↓
   Frontend: selectedImages.add(img1, img2, img3)
   ↓
   Frontend: Images get blue border (visual feedback)

3. User: Click "Move Selected" button
   ↓
   Frontend: Show modal with other clusters
   ↓
   User: Click "identity_001" (Alberto's cluster)
   ↓
   Frontend: Mark as selected (green background)

4. User: Click "Move" to confirm
   ↓
   Frontend: Call moveImages()
   ↓
   Frontend: Record change recordChange({type: 'move_images', ...})
   ↓
   Frontend: Move images in memory (update data structures)
   ↓
   Frontend: Re-render UI
   ↓
   User: See images disappeared from identity_000

5. User: Continue reviewing other clusters...

6. User: Click "Save Changes" when done
   ↓
   Frontend: fetch('/api/save-clusters', {changes: [...]})
   ↓
   Backend: Receive save request
   ↓
   Backend: Create backup directory
   ↓
   Backend: Actually move image files (shutil.move)
   ↓
   Backend: Update metadata JSON
   ↓
   Backend: Return {status: 'success'}
   ↓
   Frontend: Show notification "✓ Changes saved!"
```

---

## 🎯 Why This Approach Works

### Problem: Automated Clustering is Not Perfect

```
Automated clustering results:
identity_000: [45 Luca images] + [3 Alberto misclassified] ❌
identity_001: [38 Alberto images] + [2 Luca misclassified] ❌
identity_002: [52 Giulia images]                          ✓
```

### Solution: Human Review + Quick Correction

```
After interactive review:
identity_000 (renamed→luca_human_form): [45 Luca] ✓
identity_001 (renamed→alberto_human_form): [38 Alberto] ✓
identity_002 (renamed→giulia): [52 Giulia] ✓

Correction time: 5-10 minutes (vs manual sorting: hours)
```

**Interactive Review = Automation + Human Intelligence**

1. **Automation** (Face clustering) does 90% of work
2. **Human review** (Interactive UI) fixes last 10% errors
3. **Quick operations** (drag, shortcuts) improve efficiency
4. **Real-time feedback** (visual updates) ensure correctness
5. **Safe saving** (backup + change log) prevent mistakes

Result: **95%+ accuracy** in final character clusters!

---

## Advanced Usage

### Running Without Auto-opening Browser

```bash
conda run -n ai_env python scripts/generic/clustering/launch_interactive_review.py \
  /path/to/clusters \
  --port 8000 \
  --no-browser
```

Then manually navigate to: `http://localhost:8000/?path=/path/to/clusters`

### Custom Port

```bash
conda run -n ai_env python scripts/generic/clustering/launch_interactive_review.py \
  /path/to/clusters \
  --port 8888
```

### Development Mode (Mock Data)

Open `cluster_review.html` directly in a browser without the server:

```bash
# Navigate to UI directory
cd scripts/generic/clustering/interactive_ui

# Open in browser (will load mock data)
open cluster_review.html
# or
python -m http.server 8000
```

## Workflow Integration

### Complete Pipeline

```bash
# 1. Extract frames
conda run -n ai_env python scripts/generic/video/universal_frame_extractor.py \
  /mnt/data/ai_data/raw_videos/luca \
  --mode scene --scene-threshold 30.0

# 2. Segment characters
conda run -n ai_env python scripts/generic/segmentation/layered_segmentation.py \
  --input-dir /mnt/data/ai_data/datasets/3d-anime/luca/frames \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/luca/segmented \
  --model isnet

# 3. Cluster characters
conda run -n ai_env python scripts/generic/clustering/character_clustering.py \
  --input-dir /mnt/data/ai_data/datasets/3d-anime/luca/segmented/characters \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/luca/clustered \
  --min-cluster-size 12

# 4. REVIEW & REFINE (Interactive UI) ⭐
conda run -n ai_env python scripts/generic/clustering/launch_interactive_review.py \
  /mnt/data/ai_data/datasets/3d-anime/luca/clustered

# 5. Generate captions
conda run -n ai_env python scripts/generic/training/prepare_training_data.py \
  --character-dirs /mnt/data/ai_data/datasets/3d-anime/luca/clustered/luca_human_form \
  --output-dir /mnt/data/ai_data/training_data/luca \
  --character-name "Luca" \
  --generate-captions

# 6. Train LoRA
# ... training scripts ...
```

## Keyboard Shortcuts Reference

| Shortcut | Action |
|----------|--------|
| `Ctrl+A` | Select all images in current cluster |
| `Ctrl+Z` | Undo last change |
| `Ctrl+Y` | Redo last undone change |
| `Ctrl+S` | Save all changes |
| `Delete` | Move selected images to noise cluster |
| `Ctrl+Click` (on cluster) | Multi-select clusters for merging |

## Tips & Best Practices

### 1. Start with the Largest Clusters
- Review high-count clusters first
- These often contain the main characters

### 2. Create Pose/View Subclusters
- For the same character, create subclusters by:
  - Viewing angle (front, side, back)
  - Pose (standing, sitting, running)
  - Expression (happy, sad, neutral)

### 3. Use the Noise Cluster Wisely
- Keep obvious misclassifications in noise
- Review noise periodically for missed good images

### 4. Meaningful Cluster Names
- Use descriptive names: `luca_human_closeup_happy` instead of `cluster_0`
- Include: character name, form, view angle, expression

### 5. Regular Saves
- Save frequently using `Ctrl+S`
- Backups are created automatically

### 6. Review in Sessions
- Don't try to review everything at once
- Focus on one character at a time

## Troubleshooting

### Server Won't Start
- Check if port 8000 is already in use
- Try a different port: `--port 8888`

### Images Not Loading
- Make sure image paths are accessible
- Check that clustering directory structure is correct

### Changes Not Saving
- Ensure you have write permissions to the cluster directory
- Check that backup directory can be created

### Browser Not Opening Automatically
- Use `--no-browser` flag and open manually
- Navigate to `http://localhost:8000`

## API Endpoints

If you want to integrate with custom tools:

### GET `/api/load-clusters?path=/path/to/clusters`
Returns cluster structure and image metadata as JSON.

**Response:**
```json
{
  "clusters": [
    {
      "id": "character_0",
      "name": "Luca Human Form",
      "description": "Luca in human form",
      "images": [...],
      "metadata": {...}
    }
  ],
  "path": "/path/to/clusters"
}
```

### POST `/api/save-clusters`
Saves cluster modifications.

**Request:**
```json
{
  "path": "/path/to/clusters",
  "clusters": [...],
  "changes": [...]
}
```

**Response:**
```json
{
  "status": "success",
  "changes_saved": 15
}
```

## Future Enhancements

- [ ] Batch operations (select across multiple clusters)
- [ ] Image comparison side-by-side
- [ ] Automatic character naming (VLM-based)
- [ ] Export to training format directly
- [ ] Thumbnail quality settings
- [ ] Dark mode
- [ ] Mobile responsive improvements
- [ ] Collaborative review (multi-user)

## Support

For issues or questions:
- Check the main project documentation
- Review the clustering guide: `docs/guides/CHARACTER_CLUSTERING.md`
- Open an issue on GitHub

## License

Part of the 3D Animation LoRA Pipeline project.
