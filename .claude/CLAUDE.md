
**Assistant scope:** This document instructs Claude Code to work on a **Pixar‑style 3D character pipeline** (video → frames → layered segmentation → embeddings → clustering → captioning → dataset → LoRA → evaluation → video synthesis). All code **and comments** must be **English‑only**. When explaining choices or summarizing results for the user, prefer **Traditional Chinese**.

---

## Project Overview

**Goal:** Build a high‑quality, scalable pipeline to curate training data and train LoRA adapters for **3D animated, human‑centric characters** (Pixar‑style), using:
- **Layered segmentation** to isolate characters from backgrounds.
- **Feature embeddings + clustering** to auto‑organize identities and poses.
- **VLM‑assisted captioning** to produce consistent, style‑aware descriptions.
- **Training + automated testing** to select the best LoRA checkpoints.
- **Video generation tools** for overnight batch workflows.

The project already includes guides and scripts for:
- **Frame extraction** with scene/interval/hybrid modes.
- **Layered segmentation** (character/background/mask).
- **Character clustering** (CLIP + HDBSCAN).
- **3D‑specific parameters & practices** (alpha/blur thresholds, dataset sizes).
- **LoRA testing & batch video utilities**.

Use these as the *source of truth* when implementing or tuning components.

---

## Working Principles (Important)

1. **Code & comments in English.** Summaries to user in Chinese when asked.
2. **Small, verifiable steps:** produce **minimal runnable units** first (CLI scripts, smoke tests, unit tests), then iterate.
3. **Determinism:** seed everything; write artifacts (metrics, previews) to `runs/<timestamp>/` folders.
4. **Re‑use guides:** before inventing new flags or algorithms, check existing docs and align with them.
5. **3D vs 2D:** always prefer the **3D‑optimized defaults** (alpha, blur, cluster size) unless the user explicitly targets 2D.
6. **File Management Policy — CRITICAL:**
   - **Always modify and extend existing files.** Do NOT create new files while keeping old versions as "legacy".
   - When improving code or documentation, **directly overwrite** the original file.
   - Can rename files if needed, but **delete old versions immediately** after migration.
   - This prevents file proliferation and maintains clean repository structure.
   - Exception: version-controlled snapshots in `docs/archive/` if explicitly requested.
7. **Interactive Review Tools — REQUIRED:**
   - For all automated processes with human-in-the-loop review (clustering, segmentation QA, caption validation), **provide web-based interactive UI** (HTML/CSS/JavaScript).
   - Interactive tools must support:
     * Visual inspection of results (image thumbnails, grid views)
     * Manual corrections (move frames between clusters, rename, merge, split)
     * Metadata editing (add/remove tags, update captions)
     * Export corrected results back to pipeline
   - Use simple, standalone HTML files that can run in browser without server (or minimal Flask/FastAPI backend).
   - UI must be user-friendly, keyboard-navigable, and responsive.

8. **Multi-Character Instance Handling — CRITICAL:**
   - **NEVER assume one frame = one character.** 3D animation scenes typically have multiple characters.
   - **Use instance-level segmentation (SAM2)** to extract EACH character separately from frames.
   - **Use face-centric identity clustering** to group by WHO characters are, not visual similarity.
   - **Complete Pipeline:**
     1. Frame → **Instance Segmentation** (SAM2) → Multiple character instances per frame
     2. Instances → **Face Detection + Recognition** (ArcFace) → Identity embeddings
     3. Identity Embeddings → **Identity Clustering** (HDBSCAN) → Character-specific folders
     4. Interactive review → Name clusters by character identity
     5. (Optional) **Pose/View Subclustering** (RTM-Pose) → Subdivide by pose+view for balanced training
   - **Avoid traditional CLIP-only clustering** which groups by visual similarity (may mix characters from same scene).
   - **Advanced: Pose/View Subclustering** improves LoRA training by:
     * Separating same identity into pose/view buckets (front/three-quarter/profile/back)
     * Enabling balanced sampling across different angles and poses
     * Improving caption consistency within each bucket
   - See `docs/guides/MULTI_CHARACTER_CLUSTERING.md` for complete methodology and technical details.

---

## Repository Map (expected)

```

scripts/  
core/ # logging, io, config helpers  
generic/  
video/ # frame extraction, interpolation, synthesis  
segmentation/ # layered segmentation, inpainting  
clustering/ # embeddings, HDBSCAN, interactive review  
training/ # caption gen, dataset builders, augmentation  
evaluation/ # LoRA testing & metrics  
setup/ # env checks  
3d\_anime/ # 3D‑specific pipelines & batch jobs  
docs/  
guides/tools/ # CLI usage details  
3d\_anime\_specific/ # 3D tips, parameters, FAQs  
reference/ # API notes, data formats  
configs/ # TOML/YAML configs for training & tools  
prompts/ # prompt libraries for testing & synthesis  
requirements/ # env lock files  
logs/ # processing logs

```

---

## End‑to‑End Pipeline

```

Video  
→ Universal Frame Extraction (scene/interval/hybrid)  
→ Layered Segmentation (character/background/mask)  
→ Embeddings (CLIP/SigLIP/InternVL vision tower)  
→ Dimensionality Reduction (PCA/UMAP)  
→ Clustering (HDBSCAN; optional KMeans/Agglomerative)  
→ Cluster Review (merge/split/rename/noise filtering)  
→ Caption Generation (VLM, schema‑guided)  
→ Dataset Assembly (images + captions + metadata)  
→ LoRA Training (character/style)  
→ Automated Testing & Metrics (checkpoint selection)  
→ (Optional) Frame interpolation + video synthesis

````

**Authoritative references in repo:**

- Universal frame extraction usage and parameters. (See *Universal Frame Extraction Guide*)
- Layered segmentation models, inpainting choices, output structure. (See *Layered Segmentation Guide*)
- Character clustering with CLIP + HDBSCAN, reports and visualization. (See *Character Clustering Guide*)
- 3D‑specific processing settings & step‑by‑step pipeline. (See *3D Animation Processing Guide*)
- 2D vs 3D parameter table and rationale. (See *2D Anime vs 3D Animation Parameters*)
- LoRA checkpoint testing & comparison workflow. (See *LoRA Testing Guide*)
- Nightly/overnight batch processing, interpolation & synthesis. (See *Video Generation & Processing Guide*)

---

## Model Choices (by task)

### 1) Segmentation (3D human‑centric) — **Use this order of preference**
- **ISNet (RMBG/ISNet family via `rembg`) — _Recommended default for 3D human characters_**: clean matte edges, robust under cinematic lighting and soft DoF.
- **SAM2** — best when **multiple people / heavy occlusions** are present; consistent instance masks; slower but precise.
- **U²‑Net** — fast fallback when GPU memory is tight or for bulk preview runs.
- **(Optional) Depth‑aided matte**: **ZoeDepth/MiDaS** to refine alpha in overlapping regions and hair/ear contours.
- **Inpainting:** OpenCV **telea** for speed; **LaMa** for higher‑quality background fills.

**3D defaults** (tuneable via flags/config):
- `alpha-threshold = 0.15` (soft anti‑aliased edges in 3D).
- `blur-threshold = 80` (allow DoF blur common in cinematic shots).
- `min-size = 128` (discard tiny characters).

### 2) Face/Person Aids (optional but recommended)
- **RetinaFace/YOLOv8‑face** for face region proposals that improve downstream clustering.
- **Pose (OpenPose/RTM‑Pose)** if pose‑aware labeling is needed (optional).

### 3) Embeddings (for clustering & retrieval)
- **CLIP** (ViT‑B/32 → fast; ViT‑L/14 → better, slower) baseline embeddings.
- **SigLIP** for stronger text‑free visual embeddings in some cases.
- **InternVL / Qwen2‑VL (vision tower only)** as alternative encoders specialized for animated content.
  - *Do not* run the full VLM for embedding; use the **vision encoder outputs** only.

### 4) Dimensionality Reduction
- **PCA** for whitening/initial reduction → **UMAP** for manifold structure used by HDBSCAN.

### 5) Clustering
- **HDBSCAN** as default (auto‑k, noise labeling, variable cluster sizes).
- **KMeans** (fixed‑k) for scenarios requiring stable bucket counts.
- **Agglomerative** for hierarchical inspection.

**3D defaults:**
- `min_cluster_size = 10–15`
- `min_samples = 2`
- Similarity thresholds a bit looser than 2D due to lighting/pose variance.

### 6) Captioning (VLM‑assisted)
- Use **Qwen2‑VL / InternVL2** *after* clustering to:
  - Name/label clusters (character name/scene tag).
  - Generate **frame‑level captions** that emphasize **3D terminology** (materials, lighting, camera).
  - Generate **sequence‑level summaries** for motion description.
- Prefer **schema‑guided outputs** (JSON Schema → deterministic fields like `character`, `camera`, `materials`, `lighting`, `final_caption`).

### 7) Training (LoRA)
- Character LoRA: 200–500 curated images often sufficient for 3D.
- Avoid color jitter & horizontal flips (breaks PBR/material and asymmetry).
- Keep token length 40–77; track CLIP skip (often 2).

### 8) Evaluation & Synthesis
- Automated **checkpoint battles** (fixed prompts, seeds) to pick the best LoRA.
- Compute **CLIP score**, character consistency, and qualitative grids.
- Optional **frame interpolation (RIFE)** + **ffmpeg** synthesis for showcase videos.

---

## Canonical CLI Recipes

> **Frame Extraction (scene or interval or hybrid)**
```bash
python scripts/generic/video/universal_frame_extractor.py \
  --input /path/to/pixar_movie.mp4 \
  --output /mnt/data/ai_data/datasets/3d-anime/movie_name/frames \
  --mode scene \
  --scene-threshold 0.3 \
  --min-scene-length 15 \
  --quality high
````

> **Layered Segmentation (3D defaults)**

```bash
python scripts/generic/segmentation/layered_segmentation.py \
  --input-dir /mnt/data/ai_data/datasets/3d-anime/movie_name/frames \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/movie_name/segmented \
  --model isnet \
  --alpha-threshold 0.15 \
  --blur-threshold 80 \
  --min-size 128 \
  --extract-characters \
  --batch-size 8 \
  --gpu-id 0
```

> **Character Clustering (CLIP + HDBSCAN, 3D defaults)**

```bash
python scripts/generic/clustering/character_clustering.py \
  --input-dir /mnt/data/ai_data/datasets/3d-anime/movie_name/segmented/characters \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/movie_name/clustered \
  --min-cluster-size 12 \
  --min-samples 2 \
  --quality-filter \
  --use-face-detection \
  --similarity-threshold 0.70
```

> **Interactive Review (merge/split/rename/noise)**

```bash
python scripts/generic/clustering/interactive_character_selector.py \
  --cluster-dir /mnt/data/ai_data/datasets/3d-anime/movie_name/clustered \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/movie_name/clustered_refined
```

> **Prepare Training Data (3D caption prefix)**

```bash
python scripts/generic/training/prepare_training_data.py \
  --character-dirs /mnt/data/ai_data/datasets/3d-anime/movie_name/clustered_refined/character_0 \
  --output-dir /mnt/data/ai_data/training_data/3d_characters/character_name \
  --character-name "character_name" \
  --generate-captions \
  --caption-model qwen2_vl \
  --caption-prefix "a 3d animated human character, pixar style, smooth shading, studio lighting" \
  --augment \
  --target-size 400
```

> **LoRA Checkpoint Testing (battle grid)**

```bash
python scripts/evaluation/test_lora_checkpoints.py \
  /path/to/lora_dir \
  --base-model /path/to/base_model.safetensors \
  --output-dir outputs/lora_testing/character_name \
  --device cuda
```

> **Overnight Batch (optional)**

```bash
bash scripts/batch/overnight_processing.sh
```

* * *

File/Folder Contracts
---------------------

*   **Segmentation outputs** must include: `character/`, `background/`, `masks/`, `segmentation_results.json`.
*   **Clustering outputs** must include: `character_*/` folders, `noise/`, `cluster_report.json`, visualization PNG.
*   **Training dataset** must include: `images/`, `captions/`, `metadata.json`.
*   **Evaluation outputs** must include: per‑checkpoint subfolders, `quality_evaluation.json`, grid previews and comparison JSON/PNG.

All tools should accept `--output-dir` and never overwrite without `--force`.

* * *

3D‑Specific Defaults & Rationale
--------------------------------

*   **Alpha threshold 0.15**: softer anti‑aliased edges → better mattes.
*   **Blur threshold 80**: tolerate cinematic DoF; still reject truly blurry frames.
*   **Min cluster size 10–15, min samples 2**: 3D identity is consistent → smaller, tighter clusters suffice.
*   **Dataset size 200–500** for character LoRA: 3D requires fewer examples due to model consistency.
*   **Disable color jitter & flips**: preserve PBR materials and asymmetric accessories.
*   **Keep captions 40–77 tokens**: stable with SD‑style trainers; avoid verbose prose.

* * *

VLM Usage Policy (Qwen2‑VL / InternVL2)
---------------------------------------

1.  **Not for clustering.** Use vision encoders only for embeddings; keep the LLM out of this stage.
2.  **Yes for semantics.** After clusters form, sample 3–5 images per cluster and ask VLM to:
    *   Produce a **cluster label** (“Caleb in blue hoodie under warm indoor light”).
    *   Generate **frame‑level captions** that emphasize **materials, lighting, camera** for 3D.
    *   Optionally, produce **sequence‑level** motion summaries (multi‑frame input).
3.  **Schema‑guided outputs.** Enforce fields like `character`, `outfit`, `materials (skin/cloth/hair)`, `lighting (key/rim/fill)`, `camera (close‑up/three‑quarter/low‑angle)`, `final_caption`.
4.  **QC loop.** Reject captions with hallucinated details; prefer neutral terms if uncertain.

* * *

Advanced 3D Enhancements
------------------------

### A) Pose‑aware Subclustering (face → pose → view buckets)

**Purpose:** Improve cluster purity for 3D human characters by separating the same identity into **pose/view buckets** (e.g., front/three‑quarter/profile; standing vs. running), which stabilizes captioning and LoRA training.

**Algorithmic steps:**

1.  **Identity‑first**: run SAM2 instance segmentation + ArcFace identity clustering (HDBSCAN).
2.  **Pose estimation** (RTM‑Pose) to obtain keypoints → normalize scale & rotation.
3.  **View classification** (front, 3/4, profile, back) based on keypoint geometry (automatic).
4.  **Pose feature extraction**: normalize keypoint coordinates to remove position/scale effects.
5.  **Subcluster** within each identity cluster by **pose+view features** (UMAP + HDBSCAN or KMeans with k=3–5).
6.  **Caption policy**: include pose/view terms (e.g., _three‑quarter view, shoulder‑up, neutral stance_).

**Integrated in main pipeline:**

```bash
# Automatically prompts for pose subclustering during execution
bash scripts/pipelines/run_multi_character_clustering.sh luca
```

**Or run standalone:**

```bash
conda run -n ai_env python scripts/generic/clustering/pose_subclustering.py \
  /path/to/identity_clusters \
  --output-dir /path/to/pose_subclusters \
  --pose-model rtmpose-m \
  --device cuda \
  --method umap_hdbscan \
  --min-cluster-size 5 \
  --visualize
```

**Training benefit:** Balanced angle/pose buckets → more uniform datasets → better generalization and less overfitting to a single view.

* * *

### B) Temporal Consistency (shot‑level gating + dedup)

**Purpose:** Reduce near‑duplicates and leakage by **organizing frames per shot** and sampling consistently across time.

**Algorithmic steps:**

1.  **Scene/shot detection** (PySceneDetect or histogram/cosine embedding diff) to segment video into shots.
2.  **Shot‑aware sampling**: from each shot, keep **N representative frames** (diversity by time index + blur filter).
3.  **Per‑shot dedup**: perceptual hash (pHash/SSIM) to remove near‑duplicates **within** and **across** shots.
4.  **Shot‑level splits**: build train/val **by shots**, not random frames, to avoid leakage.
5.  **Temporal captioning** (optional): for motion LoRA or sequence descriptions, sample 8–12 frames from a single shot and ask the VLM for a sequence summary.

**CLI (reference):**

```bash
python scripts/generic/video/temporal_consistency_sampler.py \
  --frames /.../frames \
  --shots-json /.../shots.json \
  --rep-per-shot 5 \
  --phash-threshold 12 \
  --ssim-threshold 0.92 \
  --output /.../frames_temporal_consistent
```

**Training benefit:** Less redundancy, cleaner splits, stronger validation metrics that reflect real performance.

* * *

Deliverables Claude Should Produce (when asked)
-----------------------------------------------

*   Runnable CLI scripts for: extraction, segmentation, embeddings, clustering, review UI, captioning, dataset build, evaluation, video synth.
*   Config presets for **3D defaults** (TOML/YAML), plus **per‑title overrides**.
*   Unit tests for core transforms (I/O, mask ops, UMAP/HDBSCAN wrappers).
*   Markdown quickstarts placed under `docs/3d_anime_specific/` and `docs/guides/tools/`.
*   A **metrics report** template (CSV + Markdown) covering: counts per stage, rejection reasons, cluster sizes, caption stats, LoRA comparisons.

* * *

Environment Notes
-----------------

*   Python 3.10; CUDA GPU available.
*   PyTorch pinned per user’s environment (e.g., `torch==2.7.1` + CUDA 12.8 wheels).
*   All paths configurable; prefer **absolute paths** under the user’s **AI Warehouse** layout.
*   Long jobs must stream logs to `logs/` and rotate per run ID.

* * *

Safety & Quality Gates
----------------------

*   **Deduplication** (pHash/SSIM) before captioning/training.
*   **NSFW filtering** (only keep “safe” frames for character LoRA).
*   **Validation splits** by **scene** not random frames (avoid leakage).
*   **Checkpoint selection** via automated tests, then **human review** of top‑k.

* * *

References (internal guides to consult while coding)
----------------------------------------------------

*   _3D Animation LoRA Pipeline_ — overall pipeline + scripts/structure.
*   _3D Animation Features_ — why 3D needs different thresholds and policies.
*   _3D Animation Processing Guide_ — step‑by‑step recipes and CLI examples.
*   _2D Anime vs 3D Animation Parameters_ — side‑by‑side defaults and reasons.
*   _Character Clustering Guide_ — CLIP+HDBSCAN usage and outputs.
*   _Layered Segmentation Guide_ — models, inpainting, output contracts.
*   _LoRA Testing Guide_ — checkpoint battle workflow & metrics.
*   _Video Generation & Processing Guide_ — overnight batch, interpolation, synthesis.  
    """

