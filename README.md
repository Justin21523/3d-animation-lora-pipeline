# 3D Animation LoRA Pipeline

A portfolio-ready ML pipeline for preparing animated character LoRA datasets. It turns animation inputs into staged artifacts: frames, detections, foreground/background cutouts, pose records, embeddings, LoRA-ready datasets, inference samples, and animation frames.

The repository is intentionally file-driven. There is no production database or web API; YAML/TOML configs and parquet metadata are the contracts between stages.

## Why This Project Matters

- Demonstrates end-to-end ML pipeline engineering, not just a single image-generation script.
- Separates real GPU/model workflows from CPU-safe stub workflows for repeatable demos.
- Uses config-driven orchestration for 2D/3D animation differences, training presets, and model paths.
- Includes batch automation, monitoring scripts, evaluation utilities, and a static portfolio demo site.

## Demo First

The fastest way to understand the project is the CPU/stub demo. It does not require private media, GPU weights, OpenAI keys, or ComfyUI.

```bash
pip install -r requirements/core.txt -r requirements/dev.txt
bash bash/run_full_pipeline_stub.sh
python scripts/demo/run_demo_pipeline.py --skip-pipeline
python -m http.server 8080 -d portfolio-web
```

Open `http://localhost:8080`.

The static site in `portfolio-web/` is suitable for GitHub Pages, Netlify, Vercel, or the included Nginx Dockerfile.

The demo generator also creates product-style synthetic assets under `portfolio-web/assets/demo/`:

- character dataset sheet
- frame-to-training-sample before/after
- training metrics snapshot
- checkpoint evaluation matrix
- generated motion strip
- browser screenshots and a short MP4 walkthrough

## Pipeline Stages

The demo-safe flow runs these stages:

1. Frame extraction
2. Perceptual dedupe
3. Detection and tracking
4. Foreground/background segmentation
5. Pose extraction
6. Embedding generation
7. Character LoRA dataset assembly
8. ControlNet pose dataset assembly
9. LoRA + ControlNet inference samples
10. Animation export, upscaling, and interpolation

## Repository Structure

- `anime_pipeline/`: packaged Python modules for frames, detection, segmentation, pose, embeddings, datasets, training, inference, restoration, and captioning.
- `scripts/`: CLI wrappers, batch jobs, training orchestration, evaluation, monitoring, and demo utilities.
- `configs/`: global, project, stage, training, batch, and evaluation configuration.
- `requirements/`: modular dependency sets.
- `tests/`: pytest suites, including demo-safe smoke tests.
- `portfolio-web/`: static portfolio/demo website.
- `docker/`: Nginx static hosting config for the portfolio website.

Large generated artifacts, model weights, raw media, logs, and checkpoints are intentionally excluded from git.

## Testing

Recommended demo-safe verification:

```bash
./tests/run_tests.sh
```

Equivalent direct command:

```bash
python -m pytest \
  tests/demo \
  tests/test_config.py \
  tests/test_frames.py \
  tests/test_detection.py \
  tests/test_segmentation.py \
  tests/test_pose.py \
  tests/test_embeddings_and_datasets.py \
  tests/test_training.py \
  tests/test_controlnet_training.py \
  tests/test_inference_animation.py \
  tests/test_upscale_and_interpolate.py \
  -q
```

Full `python -m pytest tests/` may require GPU/model dependency alignment because some tests exercise heavy diffusers/inpainting/training paths.

## Real Model Workflow

For real runs, install the full environment and point configs at local models/data:

```bash
pip install -r requirements/all.txt
bash scripts/setup/install_pipeline_dependencies.sh
python -m scripts.core.pipeline validate --project luca
python -m scripts.core.pipeline run --project luca --device cuda
```

Common external requirements:

- CUDA-capable GPU for real model inference/training.
- Local model warehouse, usually `/mnt/c/ai_models`.
- Dataset warehouse, usually `/mnt/data/ai_data`.
- Optional `OPENAI_API_KEY` or `LLM_VENDOR_API_KEY` for API captioning/refinement workflows.
- Optional ComfyUI for visual workflow testing.

## Current Demo Status

- CPU/stub pipeline: works.
- Static demo website: available in `portfolio-web/`.
- Demo manifest: `portfolio-web/demo-data/manifest.json`.
- Product-style demo assets: available in `portfolio-web/assets/demo/`.
- Screenshots and video: available in `portfolio-web/assets/screenshots/` and `portfolio-web/assets/video/`.
- 3D pipeline status command: safe to run for config/status inspection.
- Heavy tests and real training flows: require environment-specific dependency/model setup.

## Interview Highlights

- End-to-end data contracts through parquet metadata.
- Config-driven 2D/3D parameterization.
- CPU-safe stubs for reproducible demos and CI.
- Realistic ML production concerns: batch jobs, model registries, monitoring, checkpoints, evaluation, and deployment packaging.
- Clear separation between public demo artifacts and private/generated training assets.
