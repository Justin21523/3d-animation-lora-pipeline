# Repository Guidelines

## Project Structure & Module Organization

- `scripts/`: primary CLI tools and batch workflows.
  - `scripts/core/pipeline/`: stage orchestration (`python -m scripts.core.pipeline ...`).
  - `scripts/generic/`: reusable stages (video, segmentation, clustering, training prep).
  - `scripts/batch/`, `scripts/training/`, `scripts/monitoring/`: long-running automation and dashboards.
- `anime_pipeline/`: packaged pipeline library (2D-focused modules: config, frames, detection/tracking, segmentation, clustering, datasets, training, restoration).
- `configs/`: shared configuration (global + per-stage + per-project). Prefer config-driven paths over hard-coded machine paths.
- `requirements/`: modular dependency lists (install `requirements/all.txt` for a full environment).
- `tests/`: `pytest` suites and a small runner script (`tests/run_tests.sh`).
- Large/local artifacts are intentionally not versioned: `data*/`, `datasets/`, `models/`, `outputs/`, `logs/`, `checkpoints/`.

## Build, Test, and Development Commands

- Install dependencies (pip): `pip install -r requirements/all.txt`
- Install GPU/CV stack into conda env (`ai_env`): `bash scripts/setup/install_pipeline_dependencies.sh`
- (Optional) Set up a dedicated Kohya training env: `bash setup_kohya_env.sh`
- 3D orchestrator CLI: `python -m scripts.core.pipeline run --project luca --device cuda`
- 2D orchestrator CLI: `python scripts/run_pipeline.py --project simpsons --mode 2d`
- Run tests: `./tests/run_tests.sh` (or `python -m pytest tests/ -q`)

## Coding Style & Naming Conventions

- Python 3.10+, 4-space indentation, PEP8, type hints where practical.
- Code and comments: English; prefer `logging` over `print` (except for CLI summaries).
- Naming: `snake_case` for files/functions, `PascalCase` for classes; add new tools under `scripts/<area>/...` or `anime_pipeline/<area>/...`.
- Avoid hardcoded machine paths; route path changes through `configs/` and use `Path` utilities.

## Testing Guidelines

- Framework: `pytest`; name tests `test_*.py`.
- Mock GPU/IO-heavy dependencies so tests can run on CPU-only machines (CPU smoke tests > GPU-only tests).

## Commit & Pull Request Guidelines

- Commits: short, imperative, capitalized (e.g., `Add ...`, `Update ...`, `Fix ...`).
- PRs: include stage/tool impact, exact command(s) to reproduce, and note any `configs/` schema/path changes; attach representative logs/screenshots when behavior changes.

## Data & Security Notes

- Do not commit raw media, checkpoints, or secrets; keep tokens and private paths in environment variables or private overrides.
