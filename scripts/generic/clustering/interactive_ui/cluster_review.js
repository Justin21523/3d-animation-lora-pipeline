// cluster_review.js - Interactive Clustering Review Tool

class ClusterReviewApp {
    constructor() {
        this.clusters = [];
        this.currentCluster = null;
        this.selectedImages = new Set();
        this.selectedClustersForMerge = new Set();
        this.changes = [];
        this.undoStack = [];
        this.redoStack = [];
        this.dataPath = null;

        this.init();
    }

    init() {
        this.bindEvents();
        this.loadClusterData();
    }

    bindEvents() {
        // Search
        document.getElementById('searchBox').addEventListener('input', (e) => {
            this.searchClusters(e.target.value);
        });

        // Toolbar buttons
        document.getElementById('createClusterBtn').addEventListener('click', () => {
            this.showCreateClusterModal();
        });

        document.getElementById('mergeBtn').addEventListener('click', () => {
            this.mergeClusters();
        });

        document.getElementById('splitBtn').addEventListener('click', () => {
            this.splitCluster();
        });

        document.getElementById('deleteBtn').addEventListener('click', () => {
            this.deleteSelected();
        });

        document.getElementById('undoBtn').addEventListener('click', () => {
            this.undo();
        });

        document.getElementById('redoBtn').addEventListener('click', () => {
            this.redo();
        });

        document.getElementById('saveBtn').addEventListener('click', () => {
            this.saveChanges();
        });

        // Cluster actions
        document.getElementById('renameBtn').addEventListener('click', () => {
            this.renameCluster();
        });

        document.getElementById('moveImagesBtn').addEventListener('click', () => {
            this.showMoveImagesModal();
        });

        // Modal buttons
        document.getElementById('confirmCreateBtn').addEventListener('click', () => {
            this.createCluster();
        });

        document.getElementById('cancelCreateBtn').addEventListener('click', () => {
            this.hideModal('createClusterModal');
        });

        document.getElementById('confirmMoveBtn').addEventListener('click', () => {
            this.moveImages();
        });

        document.getElementById('cancelMoveBtn').addEventListener('click', () => {
            this.hideModal('moveImagesModal');
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey || e.metaKey) {
                if (e.key === 'z') {
                    e.preventDefault();
                    this.undo();
                } else if (e.key === 'y') {
                    e.preventDefault();
                    this.redo();
                } else if (e.key === 's') {
                    e.preventDefault();
                    this.saveChanges();
                } else if (e.key === 'a') {
                    e.preventDefault();
                    this.selectAllImages();
                }
            } else if (e.key === 'Delete') {
                this.deleteSelected();
            }
        });
    }

    async loadClusterData() {
        // Check if running with local server or need to load from API
        const urlParams = new URLSearchParams(window.location.search);
        const dataPath = urlParams.get('path');

        if (dataPath) {
            this.dataPath = dataPath;
            try {
                const response = await fetch(`/api/load-clusters?path=${encodeURIComponent(dataPath)}`);
                const data = await response.json();
                this.clusters = data.clusters;
                this.renderClusterList();
                this.updateStats();
            } catch (error) {
                console.error('Error loading cluster data:', error);
                this.showNotification('Failed to load cluster data. Make sure the server is running.', 'error');
                // Load mock data for development
                this.loadMockData();
            }
        } else {
            this.loadMockData();
        }
    }

    loadMockData() {
        // Mock data for development/testing
        this.clusters = [
            {
                id: 'cluster_0',
                name: 'Luca_Human_Form',
                description: 'Luca in human form with green shirt',
                images: this.generateMockImages('cluster_0', 45),
                metadata: { character: 'Luca', form: 'human' }
            },
            {
                id: 'cluster_1',
                name: 'Alberto_Sea_Monster',
                description: 'Alberto in sea monster form',
                images: this.generateMockImages('cluster_1', 38),
                metadata: { character: 'Alberto', form: 'sea_monster' }
            },
            {
                id: 'cluster_2',
                name: 'Giulia_Bicycle',
                description: 'Giulia riding bicycle',
                images: this.generateMockImages('cluster_2', 52),
                metadata: { character: 'Giulia', action: 'bicycle' }
            },
            {
                id: 'cluster_3',
                name: 'Ercole_Vespa',
                description: 'Ercole with his Vespa',
                images: this.generateMockImages('cluster_3', 29),
                metadata: { character: 'Ercole', object: 'vespa' }
            },
            {
                id: 'noise',
                name: 'Noise / Unclassified',
                description: 'Images that don\'t fit other clusters',
                images: this.generateMockImages('noise', 15),
                metadata: { type: 'noise' }
            }
        ];

        this.renderClusterList();
        this.updateStats();
        this.showNotification('Loaded mock data for development', 'warning');
    }

    generateMockImages(clusterId, count) {
        const images = [];
        for (let i = 0; i < count; i++) {
            images.push({
                id: `${clusterId}_img_${i}`,
                path: `https://via.placeholder.com/200x200.png?text=${clusterId}_${i}`,
                filename: `scene00${Math.floor(i/10)}_pos${i%10}_frame${String(i).padStart(6, '0')}.jpg`,
                metadata: {
                    scene: Math.floor(i / 10),
                    frame: i,
                    timestamp: `${Math.floor(i * 2.5)}s`
                }
            });
        }
        return images;
    }

    renderClusterList() {
        const clusterList = document.getElementById('clusterList');
        clusterList.innerHTML = '';

        this.clusters.forEach(cluster => {
            const item = document.createElement('div');
            item.className = 'cluster-list-item';
            item.dataset.clusterId = cluster.id;

            if (this.currentCluster && this.currentCluster.id === cluster.id) {
                item.classList.add('active');
            }

            if (this.selectedClustersForMerge.has(cluster.id)) {
                item.classList.add('selected-for-merge');
            }

            item.innerHTML = `
                <div class="cluster-name">${cluster.name}</div>
                <div class="cluster-count">${cluster.images.length} images</div>
            `;

            // Single click to select cluster
            item.addEventListener('click', (e) => {
                if (!e.ctrlKey && !e.metaKey) {
                    this.selectCluster(cluster);
                } else {
                    // Ctrl+click to multi-select for merging
                    this.toggleClusterForMerge(cluster);
                }
            });

            clusterList.appendChild(item);
        });
    }

    selectCluster(cluster) {
        this.currentCluster = cluster;
        this.selectedImages.clear();
        this.renderClusterList();
        this.renderImageGrid();
        this.updateButtons();
        this.updateStats();
    }

    toggleClusterForMerge(cluster) {
        if (this.selectedClustersForMerge.has(cluster.id)) {
            this.selectedClustersForMerge.delete(cluster.id);
        } else {
            this.selectedClustersForMerge.add(cluster.id);
        }
        this.renderClusterList();
        this.updateButtons();
    }

    renderImageGrid() {
        const imageGrid = document.getElementById('imageGrid');
        const clusterTitle = document.getElementById('clusterTitle');
        const clusterInfo = document.getElementById('clusterInfo');

        if (!this.currentCluster) {
            imageGrid.innerHTML = `
                <div class="empty-state">
                    <div class="empty-state-icon">📁</div>
                    <div class="empty-state-text">Select a cluster to view images</div>
                </div>
            `;
            clusterTitle.textContent = 'Select a cluster';
            clusterInfo.textContent = 'Click on a cluster from the left panel';
            return;
        }

        clusterTitle.textContent = this.currentCluster.name;
        clusterInfo.textContent = `${this.currentCluster.images.length} images · ${this.currentCluster.description || 'No description'}`;

        imageGrid.innerHTML = '';

        this.currentCluster.images.forEach(image => {
            const card = document.createElement('div');
            card.className = 'image-card';
            card.dataset.imageId = image.id;

            if (this.selectedImages.has(image.id)) {
                card.classList.add('selected');
            }

            card.innerHTML = `
                <div class="image-checkbox ${this.selectedImages.has(image.id) ? 'checked' : ''}">
                    ${this.selectedImages.has(image.id) ? '✓' : ''}
                </div>
                <img src="${image.path}" alt="${image.filename}" loading="lazy">
                <div class="image-info">
                    <div>${image.filename}</div>
                    <div style="font-size: 0.8em; opacity: 0.7;">${image.metadata.timestamp || ''}</div>
                </div>
            `;

            // Click on checkbox to toggle selection
            card.querySelector('.image-checkbox').addEventListener('click', (e) => {
                e.stopPropagation();
                this.toggleImageSelection(image.id);
            });

            // Click on card to view full size (or toggle selection)
            card.addEventListener('click', () => {
                this.toggleImageSelection(image.id);
            });

            imageGrid.appendChild(card);
        });
    }

    toggleImageSelection(imageId) {
        if (this.selectedImages.has(imageId)) {
            this.selectedImages.delete(imageId);
        } else {
            this.selectedImages.add(imageId);
        }
        this.renderImageGrid();
        this.updateButtons();
        this.updateStats();
    }

    selectAllImages() {
        if (!this.currentCluster) return;

        if (this.selectedImages.size === this.currentCluster.images.length) {
            // Deselect all
            this.selectedImages.clear();
        } else {
            // Select all
            this.currentCluster.images.forEach(img => {
                this.selectedImages.add(img.id);
            });
        }

        this.renderImageGrid();
        this.updateButtons();
        this.updateStats();
    }

    updateButtons() {
        const hasSelection = this.selectedImages.size > 0;
        const hasCluster = this.currentCluster !== null;
        const hasMergeSelection = this.selectedClustersForMerge.size >= 2;

        document.getElementById('mergeBtn').disabled = !hasMergeSelection;
        document.getElementById('splitBtn').disabled = !hasSelection || !hasCluster;
        document.getElementById('deleteBtn').disabled = !hasSelection;
        document.getElementById('renameBtn').disabled = !hasCluster;
        document.getElementById('moveImagesBtn').disabled = !hasSelection;
        document.getElementById('undoBtn').disabled = this.undoStack.length === 0;
        document.getElementById('redoBtn').disabled = this.redoStack.length === 0;
    }

    updateStats() {
        const totalClusters = this.clusters.length;
        const totalImages = this.clusters.reduce((sum, c) => sum + c.images.length, 0);
        const selectedImages = this.selectedImages.size;
        const changesCount = this.changes.length;

        document.getElementById('totalClusters').textContent = totalClusters;
        document.getElementById('totalImages').textContent = totalImages;
        document.getElementById('selectedImages').textContent = selectedImages;
        document.getElementById('changesCount').textContent = changesCount;
    }

    searchClusters(query) {
        const lowerQuery = query.toLowerCase();
        const clusterListItems = document.querySelectorAll('.cluster-list-item');

        clusterListItems.forEach(item => {
            const clusterId = item.dataset.clusterId;
            const cluster = this.clusters.find(c => c.id === clusterId);

            const nameMatch = cluster.name.toLowerCase().includes(lowerQuery);
            const descMatch = (cluster.description || '').toLowerCase().includes(lowerQuery);

            if (nameMatch || descMatch || query === '') {
                item.style.display = 'block';
            } else {
                item.style.display = 'none';
            }
        });
    }

    // Operations

    showCreateClusterModal() {
        document.getElementById('newClusterName').value = '';
        document.getElementById('newClusterDesc').value = '';
        this.showModal('createClusterModal');
    }

    createCluster() {
        const name = document.getElementById('newClusterName').value.trim();
        const description = document.getElementById('newClusterDesc').value.trim();

        if (!name) {
            this.showNotification('Please enter a cluster name', 'error');
            return;
        }

        const newCluster = {
            id: `cluster_${Date.now()}`,
            name: name,
            description: description,
            images: [],
            metadata: {}
        };

        this.recordChange({
            type: 'create_cluster',
            cluster: newCluster
        });

        this.clusters.push(newCluster);
        this.renderClusterList();
        this.updateStats();
        this.hideModal('createClusterModal');
        this.showNotification(`Created cluster: ${name}`, 'success');
    }

    renameCluster() {
        if (!this.currentCluster) return;

        const newName = prompt('Enter new cluster name:', this.currentCluster.name);
        if (!newName || newName === this.currentCluster.name) return;

        this.recordChange({
            type: 'rename_cluster',
            clusterId: this.currentCluster.id,
            oldName: this.currentCluster.name,
            newName: newName
        });

        this.currentCluster.name = newName;
        this.renderClusterList();
        this.renderImageGrid();
        this.showNotification(`Renamed to: ${newName}`, 'success');
    }

    showMoveImagesModal() {
        if (this.selectedImages.size === 0) return;

        const targetList = document.getElementById('targetClusterList');
        targetList.innerHTML = '';

        this.clusters.forEach(cluster => {
            if (cluster.id === this.currentCluster.id) return; // Skip current cluster

            const item = document.createElement('div');
            item.className = 'cluster-target-item';
            item.dataset.clusterId = cluster.id;
            item.innerHTML = `
                <div style="font-weight: bold;">${cluster.name}</div>
                <div style="font-size: 0.9em; opacity: 0.7;">${cluster.images.length} images</div>
            `;

            item.addEventListener('click', () => {
                document.querySelectorAll('.cluster-target-item').forEach(i => i.classList.remove('selected'));
                item.classList.add('selected');
            });

            targetList.appendChild(item);
        });

        this.showModal('moveImagesModal');
    }

    moveImages() {
        const selectedItem = document.querySelector('.cluster-target-item.selected');
        if (!selectedItem) {
            this.showNotification('Please select a target cluster', 'error');
            return;
        }

        const targetClusterId = selectedItem.dataset.clusterId;
        const targetCluster = this.clusters.find(c => c.id === targetClusterId);

        const imagesToMove = Array.from(this.selectedImages);
        const sourceCluster = this.currentCluster;

        this.recordChange({
            type: 'move_images',
            sourceClusterId: sourceCluster.id,
            targetClusterId: targetClusterId,
            imageIds: imagesToMove
        });

        // Move images
        imagesToMove.forEach(imageId => {
            const imageIndex = sourceCluster.images.findIndex(img => img.id === imageId);
            if (imageIndex !== -1) {
                const image = sourceCluster.images.splice(imageIndex, 1)[0];
                targetCluster.images.push(image);
            }
        });

        this.selectedImages.clear();
        this.renderClusterList();
        this.renderImageGrid();
        this.hideModal('moveImagesModal');
        this.updateStats();
        this.showNotification(`Moved ${imagesToMove.length} images to ${targetCluster.name}`, 'success');
    }

    mergeClusters() {
        if (this.selectedClustersForMerge.size < 2) {
            this.showNotification('Select at least 2 clusters to merge (Ctrl+Click)', 'error');
            return;
        }

        const clusterIds = Array.from(this.selectedClustersForMerge);
        const clustersToMerge = this.clusters.filter(c => clusterIds.includes(c.id));

        const newName = prompt('Enter name for merged cluster:', clustersToMerge[0].name);
        if (!newName) return;

        const mergedCluster = {
            id: `cluster_${Date.now()}`,
            name: newName,
            description: `Merged from: ${clustersToMerge.map(c => c.name).join(', ')}`,
            images: [],
            metadata: {}
        };

        // Collect all images
        clustersToMerge.forEach(cluster => {
            mergedCluster.images.push(...cluster.images);
        });

        this.recordChange({
            type: 'merge_clusters',
            clusterIds: clusterIds,
            mergedCluster: mergedCluster
        });

        // Remove old clusters
        this.clusters = this.clusters.filter(c => !clusterIds.includes(c.id));

        // Add merged cluster
        this.clusters.push(mergedCluster);

        this.selectedClustersForMerge.clear();
        this.currentCluster = mergedCluster;
        this.renderClusterList();
        this.renderImageGrid();
        this.updateStats();
        this.showNotification(`Merged ${clustersToMerge.length} clusters into: ${newName}`, 'success');
    }

    splitCluster() {
        if (this.selectedImages.size === 0 || !this.currentCluster) {
            this.showNotification('Select images to split into new cluster', 'error');
            return;
        }

        const newName = prompt('Enter name for new cluster:');
        if (!newName) return;

        const imagesToSplit = Array.from(this.selectedImages);

        const newCluster = {
            id: `cluster_${Date.now()}`,
            name: newName,
            description: `Split from: ${this.currentCluster.name}`,
            images: [],
            metadata: {}
        };

        this.recordChange({
            type: 'split_cluster',
            sourceClusterId: this.currentCluster.id,
            newCluster: newCluster,
            imageIds: imagesToSplit
        });

        // Move images to new cluster
        imagesToSplit.forEach(imageId => {
            const imageIndex = this.currentCluster.images.findIndex(img => img.id === imageId);
            if (imageIndex !== -1) {
                const image = this.currentCluster.images.splice(imageIndex, 1)[0];
                newCluster.images.push(image);
            }
        });

        this.clusters.push(newCluster);
        this.selectedImages.clear();
        this.renderClusterList();
        this.renderImageGrid();
        this.updateStats();
        this.showNotification(`Created new cluster: ${newName} with ${imagesToSplit.length} images`, 'success');
    }

    deleteSelected() {
        if (this.selectedImages.size === 0) return;

        if (!confirm(`Delete ${this.selectedImages.size} images? They will be moved to 'noise' cluster.`)) {
            return;
        }

        let noiseCluster = this.clusters.find(c => c.id === 'noise');
        if (!noiseCluster) {
            noiseCluster = {
                id: 'noise',
                name: 'Noise / Unclassified',
                description: 'Deleted or unclassified images',
                images: [],
                metadata: { type: 'noise' }
            };
            this.clusters.push(noiseCluster);
        }

        const imagesToDelete = Array.from(this.selectedImages);

        this.recordChange({
            type: 'delete_images',
            sourceClusterId: this.currentCluster.id,
            imageIds: imagesToDelete
        });

        // Move to noise
        imagesToDelete.forEach(imageId => {
            const imageIndex = this.currentCluster.images.findIndex(img => img.id === imageId);
            if (imageIndex !== -1) {
                const image = this.currentCluster.images.splice(imageIndex, 1)[0];
                noiseCluster.images.push(image);
            }
        });

        this.selectedImages.clear();
        this.renderClusterList();
        this.renderImageGrid();
        this.updateStats();
        this.showNotification(`Moved ${imagesToDelete.length} images to noise`, 'success');
    }

    // Undo/Redo

    recordChange(change) {
        this.changes.push(change);
        this.undoStack.push(change);
        this.redoStack = []; // Clear redo stack
        this.updateStats();
        this.updateButtons();
    }

    undo() {
        if (this.undoStack.length === 0) return;

        const change = this.undoStack.pop();
        this.redoStack.push(change);

        // Revert change (simplified - would need full implementation)
        this.showNotification('Undo not fully implemented in demo', 'warning');

        this.updateButtons();
        this.updateStats();
    }

    redo() {
        if (this.redoStack.length === 0) return;

        const change = this.redoStack.pop();
        this.undoStack.push(change);

        // Reapply change (simplified - would need full implementation)
        this.showNotification('Redo not fully implemented in demo', 'warning');

        this.updateButtons();
        this.updateStats();
    }

    // Save

    async saveChanges() {
        if (this.changes.length === 0) {
            this.showNotification('No changes to save', 'warning');
            return;
        }

        try {
            this.showLoading();

            if (this.dataPath) {
                // Save to server
                const response = await fetch('/api/save-clusters', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        path: this.dataPath,
                        clusters: this.clusters,
                        changes: this.changes
                    })
                });

                if (response.ok) {
                    this.changes = [];
                    this.showNotification('Changes saved successfully!', 'success');
                } else {
                    throw new Error('Failed to save');
                }
            } else {
                // Demo mode - just download JSON
                const data = {
                    clusters: this.clusters,
                    changes: this.changes,
                    timestamp: new Date().toISOString()
                };

                const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `cluster_changes_${Date.now()}.json`;
                a.click();

                this.showNotification('Changes exported to JSON file', 'success');
            }

            this.hideLoading();
            this.updateStats();

        } catch (error) {
            this.hideLoading();
            console.error('Error saving changes:', error);
            this.showNotification('Failed to save changes', 'error');
        }
    }

    // UI Helpers

    showModal(modalId) {
        document.getElementById(modalId).classList.add('active');
    }

    hideModal(modalId) {
        document.getElementById(modalId).classList.remove('active');
    }

    showNotification(message, type = 'success') {
        const notification = document.getElementById('notification');
        const notificationText = document.getElementById('notificationText');

        notification.className = `notification ${type}`;
        notificationText.textContent = message;
        notification.style.display = 'block';

        setTimeout(() => {
            notification.style.display = 'none';
        }, 3000);
    }

    showLoading() {
        document.getElementById('loading').style.display = 'block';
    }

    hideLoading() {
        document.getElementById('loading').style.display = 'none';
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new ClusterReviewApp();
});
