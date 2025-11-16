# Batch Frame Extraction Guide

**批次影格提取工具使用指南**

Created: 2025-11-15
Version: 1.0.0

---

## 📖 Overview

The batch frame extraction system allows you to process multiple 3D animation films in parallel using tmux sessions. Each film runs in its own isolated session with configurable resource allocation, automatic retry on failure, and resume capability.

**Key Features:**
- ✅ **Parallel Processing** — Multiple films processed simultaneously in separate tmux sessions
- ✅ **Smart Resource Management** — Automatic CPU worker allocation based on system resources
- ✅ **Resume Support** — Skip already-processed films automatically
- ✅ **Automatic Retry** — Configurable retry attempts on failure
- ✅ **Real-time Monitoring** — Live progress tracking with dedicated monitoring script
- ✅ **Comprehensive Logging** — Structured logs for each project with timestamps

---

## 🚀 Quick Start

### Basic Usage

Process two films with default settings:

```bash
cd /mnt/c/AI_LLM_projects/3d-animation-lora-pipeline

bash scripts/batch/batch_frame_extraction.sh \
  --projects "onward,orion"
```

This will:
1. Create two tmux sessions: `frame_extraction_onward` and `frame_extraction_orion`
2. Allocate 12 CPU workers per project (configurable)
3. Use hybrid extraction mode (scene detection + interval sampling)
4. Save logs to `logs/frame_extraction/`
5. Output frames to `/mnt/data/ai_data/datasets/3d-anime/{project}/frames/`

### With Monitoring

Launch extraction and start real-time monitoring:

```bash
bash scripts/batch/batch_frame_extraction.sh \
  --projects "onward,orion" \
  --workers-per-project 12 \
  --monitor
```

---

## 📋 Prerequisites

### 1. Project Configuration Files

Each project must have a configuration file in `configs/projects/{project}.yaml`

**Example:** `configs/projects/onward.yaml`

```yaml
project:
  name: "onward"
  studio: "Pixar"
  year: 2020

frame_extraction:
  mode: "hybrid"
  scene_threshold: 30.0
  frames_per_scene: 10
  quality: "high"

# ... (see full config for details)
```

**Available configurations:**
- ✅ `configs/projects/luca.yaml`
- ✅ `configs/projects/onward.yaml`
- ✅ `configs/projects/orion.yaml`

### 2. Video Files

Video files must be located in `/mnt/c/raw_videos/{project}/`

**Supported formats:**
- `.mp4`, `.mkv`, `.avi`, `.ts`, `.m2ts`, `.mov`, `.wmv`, `.flv`

**Example structure:**
```
/mnt/c/raw_videos/
├── onward/
│   └── onward.ts
└── orion/
    └── orion.ts
```

### 3. Environment

- Python environment: `ai_env` (conda)
- tmux installed
- Sufficient disk space in `/mnt/data/ai_data/`

---

## 🎯 Command-Line Options

### Required Arguments

| Option | Description | Example |
|--------|-------------|---------|
| `--projects` | Comma-separated list of project names | `"onward,orion"` |

### Extraction Parameters

| Option | Default | Description |
|--------|---------|-------------|
| `--mode` | `hybrid` | Extraction mode: `scene`, `interval`, or `hybrid` |
| `--scene-threshold` | `30.0` | Scene detection sensitivity (lower = more scenes) |
| `--frames-per-scene` | `10` | Number of frames to extract per scene |
| `--interval-seconds` | `2.0` | Interval for interval/hybrid modes |
| `--quality` | `high` | JPEG quality: `low` (85), `medium` (90), `high` (95) |

### Resource Management

| Option | Default | Description |
|--------|---------|-------------|
| `--workers-per-project` | `12` | CPU workers allocated per project |

**Auto-calculation logic:**
- System detects total CPU cores
- Calculates 75% of cores (leaves 25% for system + GPU tasks)
- Divides by number of projects
- Uses minimum of requested and calculated value

**Example with 32 threads:**
- Available: 32 × 0.75 = 24 threads
- 2 projects → 12 workers per project ✅
- 3 projects → 8 workers per project

### Reliability Features

| Option | Default | Description |
|--------|---------|-------------|
| `--retry` | `3` | Number of retry attempts on failure |
| `--force` | `false` | Force re-processing even if results exist |

### Monitoring & Output

| Option | Default | Description |
|--------|---------|-------------|
| `--monitor` | `false` | Launch monitoring script after starting |
| `--dry-run` | `false` | Print configuration without executing |

---

## 📂 Output Structure

### Frame Outputs

```
/mnt/data/ai_data/datasets/3d-anime/{project}/
├── frames/
│   ├── scene0000_pos0_frame000000_t0.50s.jpg
│   ├── scene0000_pos1_frame000042_t1.75s.jpg
│   ├── scene0001_pos0_frame000125_t5.20s.jpg
│   └── ...
├── extraction_results.json
└── temp/                  # Temporary format conversion directory
```

### Log Files

```
logs/frame_extraction/
├── onward_20251115_143000.log
├── orion_20251115_143000.log
└── completion_notifications.log
```

### extraction_results.json Format

```json
{
  "total_episodes": 1,
  "successful_episodes": 1,
  "total_frames": 4523,
  "output_dir": "/mnt/data/ai_data/datasets/3d-anime/onward/frames",
  "config": {
    "mode": "hybrid",
    "scene_threshold": 30.0,
    "frames_per_scene": 10,
    "interval_seconds": 2.0,
    "quality": "high",
    "workers": 12
  },
  "timestamp": "2025-11-15T14:30:00"
}
```

---

## 🎬 Usage Examples

### Example 1: Standard Batch Processing

Process two films with default balanced settings:

```bash
bash scripts/batch/batch_frame_extraction.sh \
  --projects "onward,orion" \
  --workers-per-project 12
```

### Example 2: Aggressive CPU Usage

Use more CPU resources (28 workers total):

```bash
bash scripts/batch/batch_frame_extraction.sh \
  --projects "onward,orion" \
  --workers-per-project 14
```

### Example 3: Scene-Based Extraction Only

Extract only from scene boundaries (no interval sampling):

```bash
bash scripts/batch/batch_frame_extraction.sh \
  --projects "onward" \
  --mode scene \
  --scene-threshold 25.0 \
  --frames-per-scene 15
```

### Example 4: Interval-Based Extraction

Extract every 3 seconds:

```bash
bash scripts/batch/batch_frame_extraction.sh \
  --projects "orion" \
  --mode interval \
  --interval-seconds 3.0
```

### Example 5: Force Re-processing

Re-extract frames even if results already exist:

```bash
bash scripts/batch/batch_frame_extraction.sh \
  --projects "onward,orion" \
  --force \
  --retry 5
```

### Example 6: Dry Run (Test Configuration)

Preview what would be executed without actually running:

```bash
bash scripts/batch/batch_frame_extraction.sh \
  --projects "onward,orion" \
  --workers-per-project 12 \
  --dry-run
```

---

## 🖥️ Tmux Session Management

### List Active Sessions

```bash
tmux ls
```

**Example output:**
```
frame_extraction_onward: 1 windows (created Fri Nov 15 14:30:00 2025)
frame_extraction_orion: 1 windows (created Fri Nov 15 14:30:02 2025)
```

### Attach to Session

View real-time output for a specific project:

```bash
tmux attach -t frame_extraction_onward
```

**Detach from session:** Press `Ctrl+B` then `D`

### Kill a Session

Stop processing for a specific project:

```bash
tmux kill-session -t frame_extraction_onward
```

### Kill All Extraction Sessions

```bash
tmux ls | grep "frame_extraction_" | cut -d: -f1 | xargs -I {} tmux kill-session -t {}
```

---

## 📊 Progress Monitoring

### Real-time Monitoring

Launch interactive monitoring dashboard:

```bash
bash scripts/monitoring/monitor_frame_extraction.sh \
  --projects "onward,orion"
```

**Features:**
- System resource usage (CPU, memory)
- Per-project status (RUNNING, COMPLETE, IDLE)
- Total frames extracted
- Elapsed time
- Recent log output
- Auto-refresh every 10 seconds

### One-Time Status Check

Display current status without continuous monitoring:

```bash
bash scripts/monitoring/monitor_frame_extraction.sh \
  --projects "onward,orion" \
  --once
```

### Auto-Detect Sessions

Automatically find and monitor all active extraction sessions:

```bash
bash scripts/monitoring/monitor_frame_extraction.sh
```

---

## 🔧 Troubleshooting

### Issue: "No video file found"

**Cause:** Video file not in expected location or unsupported format

**Solution:**
1. Check video file exists: `ls /mnt/c/raw_videos/{project}/`
2. Ensure file extension is supported (`.ts`, `.mp4`, `.mkv`, etc.)
3. If using unusual extension, convert to `.mp4`:
   ```bash
   ffmpeg -i input.ext -c:v copy -c:a copy output.mp4
   ```

### Issue: "Session already exists"

**Cause:** Previous extraction session is still running

**Solution:**
1. Check session status: `tmux ls`
2. Attach to see if it's actually running: `tmux attach -t frame_extraction_{project}`
3. Kill if stuck: `tmux kill-session -t frame_extraction_{project}`
4. Re-run batch script

### Issue: High CPU usage affecting other tasks

**Cause:** Too many workers allocated

**Solution:**
Reduce workers per project:
```bash
bash scripts/batch/batch_frame_extraction.sh \
  --projects "onward,orion" \
  --workers-per-project 8
```

### Issue: Out of memory (OOM) errors

**Cause:** Insufficient RAM for number of workers

**Solution:**
1. Check available memory: `free -h`
2. Reduce workers:
   ```bash
   --workers-per-project 6
   ```
3. Process projects sequentially instead of parallel:
   ```bash
   bash scripts/batch/batch_frame_extraction.sh --projects "onward"
   # Wait for completion
   bash scripts/batch/batch_frame_extraction.sh --projects "orion"
   ```

### Issue: Extraction results incomplete

**Cause:** Processing failed mid-way

**Solution:**
1. Check logs: `tail -100 logs/frame_extraction/{project}_*.log`
2. Re-run with `--force` to restart:
   ```bash
   bash scripts/batch/batch_frame_extraction.sh \
     --projects "{project}" \
     --force \
     --retry 5
   ```

### Issue: Frames look blurry or low quality

**Cause:** Quality settings too aggressive

**Solution:**
1. Check blur threshold in project config (`configs/projects/{project}.yaml`)
2. Adjust `quality.blur_threshold` (default 80 for 3D animation)
3. Increase if too many frames rejected: `blur_threshold: 100`
4. Or use `--quality high` to ensure 95% JPEG quality

---

## 🎨 Extraction Mode Comparison

### Scene-Based Mode

**Best for:** Films with clear scene transitions

**Pros:**
- Captures key moments and shot changes
- Reduces redundancy from static scenes
- Good for narrative-driven content

**Cons:**
- May miss gradual changes
- Depends on scene detection sensitivity

**Usage:**
```bash
--mode scene --scene-threshold 30.0 --frames-per-scene 10
```

### Interval-Based Mode

**Best for:** Uniform temporal coverage

**Pros:**
- Predictable output size
- Ensures coverage of all time periods
- Simple and fast

**Cons:**
- May include redundant frames
- Ignores scene semantics

**Usage:**
```bash
--mode interval --interval-seconds 2.0
```

### Hybrid Mode (Recommended)

**Best for:** 3D animation films (Pixar, DreamWorks)

**Pros:**
- Combines benefits of both approaches
- Scene detection for key moments
- Interval sampling ensures complete coverage
- Optimal for character LoRA training

**Cons:**
- Slightly larger output size
- Longer processing time

**Usage:**
```bash
--mode hybrid --scene-threshold 30.0 --frames-per-scene 10 --interval-seconds 2.0
```

---

## 📈 Performance Guidelines

### CPU Allocation Strategy

| System | Projects | Recommended Workers/Project | Total CPU Usage |
|--------|----------|----------------------------|-----------------|
| 16 cores | 1 | 12 | 75% |
| 16 cores | 2 | 8-12 | 75-100% |
| 32 cores | 1 | 24 | 75% |
| 32 cores | 2 | 12 | 75% |
| 32 cores | 3 | 8 | 75% |

### Processing Time Estimates

Based on 1080p video with hybrid mode:

| Film Length | Workers | Estimated Time |
|-------------|---------|----------------|
| 90 min | 12 | 1.5-2.5 hours |
| 100 min | 12 | 2.0-3.0 hours |
| 120 min | 12 | 2.5-3.5 hours |
| 90 min | 16 | 1.0-2.0 hours |

**Factors affecting speed:**
- Video encoding (H.264 faster than H.265)
- Scene complexity (more scenes = more processing)
- Disk I/O speed
- Background processes

### Expected Output Sizes

| Mode | Film Length | Approx. Frames | Disk Space |
|------|-------------|----------------|------------|
| Scene | 90 min | 2,000-3,000 | 0.6-1.0 GB |
| Interval (2s) | 90 min | 2,700 | 0.8-1.2 GB |
| Hybrid | 90 min | 3,500-5,000 | 1.0-1.8 GB |
| Scene | 100 min | 2,500-3,500 | 0.8-1.2 GB |
| Hybrid | 100 min | 4,000-5,500 | 1.2-2.0 GB |

---

## 🔄 Resume & Checkpoint Behavior

### Automatic Skip Logic

The script automatically detects completed extractions:

```bash
if [[ -f "$OUTPUT_DIR/extraction_results.json" ]] && [[ "$total_frames" -gt 0 ]]; then
    echo "⏭️  Results already exist for $PROJECT, skipping"
    exit 0
fi
```

**To bypass and re-process:**
```bash
--force
```

### Manual Resume

If you killed a session mid-processing:

1. **Check what was completed:**
   ```bash
   ls /mnt/data/ai_data/datasets/3d-anime/{project}/frames/ | wc -l
   ```

2. **Remove incomplete results:**
   ```bash
   rm -rf /mnt/data/ai_data/datasets/3d-anime/{project}/frames/
   rm /mnt/data/ai_data/datasets/3d-anime/{project}/extraction_results.json
   ```

3. **Re-run with retry:**
   ```bash
   bash scripts/batch/batch_frame_extraction.sh \
     --projects "{project}" \
     --retry 5
   ```

---

## 🛠️ Advanced Usage

### Custom Configuration Per Project

Edit project config before extraction:

```bash
nano configs/projects/onward.yaml
```

**Key settings:**
```yaml
frame_extraction:
  mode: "hybrid"
  scene_threshold: 28.0      # Lower = more scenes detected
  frames_per_scene: 12       # More frames per scene
  min_scene_length: 20       # Longer minimum scene duration
  interval_seconds: 1.5      # More frequent interval samples
  quality: "high"
```

### Parallel Processing Multiple Sets

Process different film sets simultaneously:

**Terminal 1:**
```bash
bash scripts/batch/batch_frame_extraction.sh --projects "onward,orion"
```

**Terminal 2:**
```bash
bash scripts/batch/batch_frame_extraction.sh --projects "luca"
```

Each set uses separate tmux sessions with no conflict.

### Integration with Full Pipeline

After frame extraction completes, continue with next stages:

```bash
# 1. Extract frames (this guide)
bash scripts/batch/batch_frame_extraction.sh --projects "onward"

# 2. Instance segmentation (SAM2)
bash scripts/pipelines/run_multi_character_clustering.sh onward

# 3. Identity clustering
# (handled by pipeline script)

# 4. LoRA training
conda run -n ai_env python sd-scripts/train_network.py \
  --config_file configs/training/onward_character.toml
```

---

## 📚 Related Documentation

- **Universal Frame Extractor:** `docs/guides/tools/universal_frame_extraction_guide.md`
- **Multi-Character Clustering:** `docs/guides/MULTI_CHARACTER_CLUSTERING.md`
- **Project Configuration Template:** `configs/projects/template.yaml`
- **Pipeline Overview:** `CLAUDE.md`

---

## 🐛 Reporting Issues

If you encounter problems:

1. **Check logs:**
   ```bash
   tail -100 logs/frame_extraction/{project}_*.log
   ```

2. **Run dry-run to validate config:**
   ```bash
   bash scripts/batch/batch_frame_extraction.sh \
     --projects "{project}" \
     --dry-run
   ```

3. **Test with single project first:**
   ```bash
   bash scripts/batch/batch_frame_extraction.sh --projects "onward"
   ```

4. **Report with details:**
   - Command used
   - Error message from log
   - System specs (CPU, RAM)
   - Video file format and size

---

**Version History:**
- v1.0.0 (2025-11-15): Initial release with batch processing, tmux sessions, and monitoring
