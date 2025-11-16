# Universal Frame Extraction Guide

## Overview

The Universal Frame Extractor is a flexible, production-ready tool designed to extract frames from any anime series or video collection. It supports multiple extraction strategies and is fully configurable.

## Key Features

### 🎯 Multiple Extraction Modes

1. **Scene-Based** (`--mode scene`)
   - Detects scene changes automatically
   - Extracts N frames from each scene
   - Best for character-focused datasets

2. **Interval-Based** (`--mode interval`)
   - Extracts frames at fixed intervals
   - Can use time-based (seconds) or frame-based intervals
   - Best for complete temporal coverage

3. **Hybrid** (`--mode hybrid`)
   - Combines both strategies
   - Maximum coverage
   - Best for initial exploration

### 🔧 Fully Configurable

- Custom episode number patterns (regex)
- Adjustable scene detection sensitivity
- Flexible interval settings
- Quality control options
- Parallel processing support

### 🌐 Universal Design

- Works with any video format (MP4, MKV, FLV, AVI, etc.)
- Supports any naming convention
- Not tied to specific anime series
- Handles various episode numbering schemes

---

## Quick Start

### Basic Usage

Extract frames using scene detection (default):
```bash
conda run -n blip2-env python3 scripts/tools/universal_frame_extractor.py \
  /path/to/videos
```

This will:
- Detect scenes in all videos
- Extract 3 frames per scene
- Save to `{input_dir}/extracted_frames/`

### Interval-Based Extraction

Extract one frame every 5 seconds:
```bash
conda run -n blip2-env python3 scripts/tools/universal_frame_extractor.py \
  /path/to/videos \
  --mode interval \
  --interval-seconds 5
```

Extract every 100th frame:
```bash
conda run -n blip2-env python3 scripts/tools/universal_frame_extractor.py \
  /path/to/videos \
  --mode interval \
  --interval-frames 100
```

### Hybrid Mode

Combine both strategies:
```bash
conda run -n blip2-env python3 scripts/tools/universal_frame_extractor.py \
  /path/to/videos \
  --mode hybrid \
  --interval-seconds 10 \
  --frames-per-scene 3
```

---

## Detailed Examples

### Example 1: Inazuma Eleven

```bash
conda run -n blip2-env python3 scripts/tools/universal_frame_extractor.py \
  /mnt/c/AI_LLM_projects/ai_warehouse/training_data/inazuma-eleven/raw_videos \
  --mode scene \
  --scene-threshold 27 \
  --frames-per-scene 3 \
  --workers 8 \
  --output-dir /mnt/c/AI_LLM_projects/ai_warehouse/cache/inazuma-eleven/extracted_frames
```

**Why these settings?**
- `scene-threshold 27`: Optimized for anime content
- `frames-per-scene 3`: Balance between coverage and storage
- `workers 8`: Utilize CPU efficiently

### Example 2: Naruto (Different Naming Convention)

Videos named: `Naruto_S01E01.mkv`, `Naruto_S01E02.mkv`

```bash
conda run -n blip2-env python3 scripts/tools/universal_frame_extractor.py \
  /path/to/naruto \
  --mode scene \
  --episode-pattern "E(\d+)" \
  --start 1 \
  --end 50
```

**Episode Pattern**: `E(\d+)` matches "E01", "E02", etc.

### Example 3: One Piece (Time-lapse Dataset)

Extract every 10 seconds for environment/background dataset:

```bash
conda run -n blip2-env python3 scripts/tools/universal_frame_extractor.py \
  /path/to/onepiece \
  --mode interval \
  --interval-seconds 10 \
  --jpeg-quality 90 \
  --workers 16
```

### Example 4: Attack on Titan (High-Quality Character Dataset)

More frames per scene, higher sensitivity:

```bash
conda run -n blip2-env python3 scripts/tools/universal_frame_extractor.py \
  /path/to/aot \
  --mode scene \
  --scene-threshold 20 \
  --frames-per-scene 5 \
  --jpeg-quality 98 \
  --workers 8
```

**Settings Explanation**:
- Lower threshold (20) = more scenes detected
- 5 frames per scene = better expression coverage
- 98% quality = minimal compression artifacts

### Example 5: Mixed Strategy for Complete Dataset

```bash
conda run -n blip2-env python3 scripts/tools/universal_frame_extractor.py \
  /path/to/videos \
  --mode hybrid \
  --scene-threshold 25 \
  --frames-per-scene 4 \
  --interval-seconds 8 \
  --workers 12
```

---

## Configuration Guide

### Extraction Modes

| Mode | When to Use | Pros | Cons |
|------|-------------|------|------|
| **Scene** | Character-focused training | Semantic coverage, efficient | May miss static scenes |
| **Interval** | Environmental datasets, testing | Complete coverage, predictable | May include redundant frames |
| **Hybrid** | Initial exploration, comprehensive datasets | Maximum coverage | Larger dataset size |

### Scene Detection Parameters

#### `--scene-threshold` (default: 27.0)

Controls sensitivity of scene detection:

| Value | Sensitivity | Use Case |
|-------|-------------|----------|
| 15-20 | Very High | Action anime, rapid cuts |
| 25-30 | Medium | Standard anime |
| 35-40 | Low | Slow-paced shows, dialogues |

**How to choose?**
- Start with default (27)
- If missing important scenes: lower the value
- If too many similar frames: raise the value

#### `--frames-per-scene` (default: 3)

How many frames to extract from each scene:

| Count | Storage | Coverage | Best For |
|-------|---------|----------|----------|
| 1-2 | Small | Basic | Quick overview |
| 3-4 | Medium | Good | Character training |
| 5-7 | Large | Excellent | Detailed expressions |

### Interval Parameters

#### Time-Based: `--interval-seconds`

Extract one frame every N seconds:

```bash
--interval-seconds 1    # Very dense, 1 frame/sec
--interval-seconds 5    # Balanced
--interval-seconds 10   # Sparse, key moments
--interval-seconds 30   # Very sparse, scene overview
```

**Calculation**:
- 20-minute episode = 1200 seconds
- Interval 5s = 240 frames per episode
- Interval 10s = 120 frames per episode

#### Frame-Based: `--interval-frames`

Extract every Nth frame:

```bash
--interval-frames 30   # Every 30 frames (~1.25s at 24fps)
--interval-frames 100  # Every 100 frames (~4s at 24fps)
--interval-frames 240  # Every 240 frames (~10s at 24fps)
```

**Use Cases**:
- Frame-based: Consistent across different frame rates
- Time-based: Easier to reason about

### Quality Settings

#### `--jpeg-quality` (default: 95)

JPEG compression quality (1-100):

| Quality | File Size | Visual Quality | Use Case |
|---------|-----------|----------------|----------|
| 100 | Largest | Lossless | Archive, reference |
| 95 | Large | Visually lossless | Training (recommended) |
| 85 | Medium | Good | Web, preview |
| 70 | Small | Acceptable | Quick testing |

**Recommendation**: Use 95 for training, 85 for testing.

### Performance Settings

#### `--workers`

Number of parallel processes:

```bash
--workers 4    # Conservative, low memory
--workers 8    # Balanced (recommended)
--workers 16   # Maximum speed (high memory)
```

**Guidelines**:
- Default: auto-detect (min of CPU cores and episode count)
- More workers = faster, but more RAM usage
- Optimal: 8-12 workers for most systems

---

## Episode Numbering Patterns

The tool uses regex to extract episode numbers from filenames.

### Common Patterns

| Format Example | Pattern | Regex |
|----------------|---------|-------|
| `episode_01.mp4` | Default digits | `(\d+)` |
| `S01E05.mkv` | Episode after E | `E(\d+)` |
| `闪电十一人 127集.flv` | Chinese format | `(\d+)` |
| `Anime_s01_ep12.mp4` | After "ep" | `ep(\d+)` |
| `[SubGroup] Title - 03.mkv` | Before extension | `-\s*(\d+)` |

### Custom Pattern Example

For `[HorribleSubs] Anime Name - 05 [720p].mkv`:

```bash
--episode-pattern "-\s*(\d+)\s*\["
```

This matches: `- 05 [`

**Testing Your Pattern**:
```python
import re
filename = "[HorribleSubs] Anime - 05 [720p].mkv"
pattern = r"-\s*(\d+)\s*\["
match = re.search(pattern, filename)
if match:
    print(f"Episode: {match.group(1)}")  # Output: Episode: 05
```

---

## Output Structure

### Directory Layout

```
extracted_frames/
├── episode_001/
│   ├── scene0000_pos0_frame000000_t1.00s.jpg
│   ├── scene0000_pos1_frame000001_t5.00s.jpg
│   ├── scene0000_pos2_frame000002_t9.04s.jpg
│   ├── scene0001_pos0_frame000003_t10.92s.jpg
│   └── ...
├── episode_002/
│   └── ...
└── extraction_results.json
```

### Filename Formats

#### Scene-Based Mode

Format: `scene{num}_pos{position}_frame{count}_t{timestamp}s.jpg`

Example: `scene0042_pos1_frame002156_t847.25s.jpg`
- Scene 42 in the video
- Position 1 within that scene (0, 1, 2 for 3 frames)
- Frame 2156 overall
- At timestamp 847.25 seconds (14:07)

#### Interval-Based Mode

Format: `interval{index}_frame{num}_t{timestamp}s.jpg`

Example: `interval00042_frame012500_t520.83s.jpg`
- 42nd extracted frame
- Frame 12500 in video
- At 520.83 seconds (8:40)

#### Hybrid Mode

Contains both `scene*.jpg` and `interval*.jpg` files in same directory.

### Results JSON

`extraction_results.json` contains:

```json
{
  "total_episodes": 126,
  "successful_episodes": 126,
  "total_frames": 173826,
  "output_dir": "/path/to/output",
  "config": {
    "mode": "scene",
    "scene_threshold": 27.0,
    "frames_per_scene": 3,
    "interval_seconds": null,
    "interval_frames": null,
    "jpeg_quality": 95
  }
}
```

---

## Advanced Usage

### Processing Specific Episode Range

Process only episodes 50-75:
```bash
python universal_frame_extractor.py /path/to/videos \
  --start 50 \
  --end 75
```

### Custom Output Locations

```bash
python universal_frame_extractor.py /path/to/videos \
  --output-dir /external/ssd/frames \
  --temp-dir /tmp/video_processing
```

**Why separate temp directory?**
- Temp files are large (~19GB for 126 episodes)
- Using fast SSD for temp improves performance
- Can clean up temp files after processing

### Skip Scene Boundary Protection

By default, the first/last 10% of each scene is skipped to avoid transition artifacts.

To disable:
```bash
python universal_frame_extractor.py /path/to/videos \
  --no-skip-boundaries
```

**When to use?**
- Short scenes where skipping removes too much
- Transition effects are part of the artistic style
- You want maximum frame count

---

## Performance Tuning

### Optimizing for Speed

```bash
python universal_frame_extractor.py /path/to/videos \
  --mode interval \
  --interval-seconds 15 \
  --workers 16 \
  --jpeg-quality 85
```

- Interval mode is faster than scene detection
- Larger intervals = fewer frames to write
- More workers = more parallel processing
- Lower quality = faster encoding

**Expected**: ~15 minutes for 126 episodes

### Optimizing for Quality

```bash
python universal_frame_extractor.py /path/to/videos \
  --mode hybrid \
  --scene-threshold 20 \
  --frames-per-scene 5 \
  --interval-seconds 5 \
  --jpeg-quality 98 \
  --workers 8
```

- Hybrid mode = maximum coverage
- Lower threshold = more scenes
- More frames per scene
- Higher quality
- Moderate workers to avoid I/O bottleneck

**Expected**: ~90 minutes for 126 episodes, ~50GB output

### Optimizing for Storage

```bash
python universal_frame_extractor.py /path/to/videos \
  --mode scene \
  --scene-threshold 35 \
  --frames-per-scene 2 \
  --jpeg-quality 85
```

- Higher threshold = fewer scenes
- Fewer frames per scene
- Lower quality

**Expected**: ~8GB output for 126 episodes

---

## Troubleshooting

### Issue: No Frames Extracted

**Symptoms**:
```
[Episode 001] Processing: video.mp4
[SceneDetect] Found 0 scenes
[Extract] ✓ Extracted 0 frames
```

**Causes & Solutions**:

1. **Threshold too high**
   ```bash
   # Try lower threshold
   --scene-threshold 15
   ```

2. **Video too short**
   - Check video file is valid: `ffmpeg -i video.mp4`
   - Ensure video has actual content

3. **Wrong mode for content**
   - Switch to interval mode if scene detection fails
   ```bash
   --mode interval --interval-seconds 5
   ```

### Issue: Too Many Frames

**Symptoms**: Hundreds of frames per scene, storage exploding

**Solution**: Raise threshold, reduce frames per scene
```bash
--scene-threshold 35 --frames-per-scene 2
```

### Issue: Episode Numbers Not Detected

**Symptoms**: All episodes numbered 0

**Solution**: Check filename and adjust pattern

```bash
# Debug: print what pattern matches
import re
filename = "your_video_name.mp4"
pattern = r'(\d+)'  # your pattern
match = re.search(pattern, filename)
print(match.group(1) if match else "No match")

# Then use custom pattern
--episode-pattern "your_custom_pattern"
```

### Issue: Out of Memory

**Symptoms**: Process killed with no error

**Solutions**:
1. Reduce workers: `--workers 4`
2. Process in batches: `--start 1 --end 50`, then `--start 51 --end 100`
3. Close other applications
4. Use interval mode (less memory intensive)

### Issue: Conversion Failures

**Symptoms**: "Conversion failed" errors

**Solutions**:
1. Check ffmpeg installed: `ffmpeg -version`
2. Check disk space: `df -h`
3. Manually test problem file:
   ```bash
   ffmpeg -i problem_file.flv test.mp4
   ```
4. Skip problem files and process manually later

---

## Integration with Training Pipeline

### Step 1: Extract Frames

```bash
python universal_frame_extractor.py \
  /path/to/anime/videos \
  --mode scene \
  --scene-threshold 27 \
  --frames-per-scene 3
```

### Step 2: Review and Select Character Frames

Manually or with tools:
```bash
# View extracted frames for specific episode
ls -lh extracted_frames/episode_068/

# Copy relevant frames to character folder
cp extracted_frames/episode_068/scene*.jpg \
   /path/to/characters/toramaru/gold_standard/
```

### Step 3: Generate Captions

```bash
python scripts/tools/caption_gold_standard.py toramaru
```

### Step 4: Prepare Training Data

```bash
python scripts/tools/prepare_training_data.py toramaru --repeat 10
```

### Step 5: Train LoRA

Use prepared data with kohya_ss or sd-scripts

---

## Comparison with Old Script

| Feature | Old (batch_extract_all_episodes.py) | New (universal_frame_extractor.py) |
|---------|-------------------------------------|-------------------------------------|
| **Scope** | Inazuma Eleven only | Any anime/video series |
| **Episode Pattern** | Hardcoded Chinese format | Fully customizable regex |
| **Extraction Modes** | Scene-based only | Scene, Interval, Hybrid |
| **Interval Support** | ❌ No | ✅ Yes (time & frame-based) |
| **Code Structure** | Procedural | Object-oriented (Strategy pattern) |
| **Comments** | Minimal | Extensive documentation |
| **Configurability** | Basic | Highly flexible |
| **Extendability** | Difficult | Easy (add new strategies) |

### Migration Guide

Old command:
```bash
python batch_extract_all_episodes.py \
  --workers 8 \
  --threshold 27 \
  --frames-per-scene 3
```

New equivalent:
```bash
python universal_frame_extractor.py \
  /mnt/c/AI_LLM_projects/ai_warehouse/training_data/inazuma-eleven/raw_videos \
  --mode scene \
  --workers 8 \
  --scene-threshold 27 \
  --frames-per-scene 3
```

**Key Differences**:
1. Input directory is now positional argument (required)
2. Threshold renamed to `--scene-threshold`
3. Can now use `--mode` to switch strategies

---

## Best Practices

### 1. Start with Scene Mode

For character-focused datasets:
```bash
--mode scene --scene-threshold 27 --frames-per-scene 3
```

This gives semantic coverage (different scenes = different contexts).

### 2. Test on Small Sample First

Process a few episodes to tune parameters:
```bash
--start 1 --end 5
```

Check results, adjust threshold/frames, then process full series.

### 3. Use Hybrid for Exploration

When unfamiliar with content:
```bash
--mode hybrid --interval-seconds 15
```

Gives both semantic and temporal coverage.

### 4. Monitor Disk Space

Check space before processing:
```bash
df -h
```

Estimate output size:
- Scene mode: ~500-800 MB per episode
- Interval (5s): ~300-500 MB per episode
- Interval (10s): ~150-250 MB per episode

### 5. Keep Temp Files for Reprocessing

If you might reprocess with different settings:
- Keep temp directory (converted MP4s)
- Conversion is slowest step
- Rerunning with different threshold/frames is fast

To reprocess:
```bash
# First run creates MP4s
python universal_frame_extractor.py /videos --mode scene

# Rerun with different settings (reuses MP4s)
python universal_frame_extractor.py /videos --mode interval --interval-seconds 10
```

### 6. Use Quality 95 for Training

Balance between quality and storage:
```bash
--jpeg-quality 95
```

For testing/preview, use 85.

---

## Future Enhancements

Potential additions to the tool:

1. **Quality Filtering**
   - Blur detection to skip low-quality frames
   - Brightness/contrast analysis
   - Face detection for character-focused extraction

2. **Smart Deduplication**
   - Perceptual hashing to remove similar frames
   - Saves storage while maintaining coverage

3. **GPU Acceleration**
   - CUDA-based scene detection
   - 3-5× speedup potential

4. **Adaptive Thresholds**
   - Auto-tune scene detection per episode
   - Different content needs different sensitivity

5. **Resume Capability**
   - Checkpoint system
   - Resume from last processed episode

6. **Web Dashboard**
   - Real-time progress visualization
   - Preview extracted frames
   - Interactive parameter tuning

---

## Conclusion

The Universal Frame Extractor provides a flexible, production-ready solution for extracting frames from any anime series. Its multiple extraction strategies, extensive configuration options, and clear documentation make it suitable for various use cases from quick testing to building comprehensive training datasets.

**Key Takeaways**:
- ✅ Works with any anime series
- ✅ Multiple extraction strategies
- ✅ Highly configurable
- ✅ Well-documented code
- ✅ Production-ready performance

For questions or issues, refer to the inline code comments or raise an issue in the project repository.
