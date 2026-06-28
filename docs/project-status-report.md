# Project Status Report

Date: 2026-06-28

## 1. Project Goal and Positioning

3D Animation LoRA Pipeline is a file-driven ML pipeline for turning animation footage into trainable LoRA/ControlNet datasets and reviewable generation artifacts. The public-facing goal is now a portfolio demo platform: reviewers can understand the data flow, outputs, screenshots, video, deployment, and engineering tradeoffs without private footage, model weights, GPU access, or external API keys.

The project is best positioned as:

| Position | Description |
| --- | --- |
| Primary | ML data engineering and training pipeline showcase |
| Secondary | Computer vision orchestration and dataset preparation toolkit |
| Public demo | Static GitHub Pages site backed by synthetic/anonymized artifacts |
| Not positioned as | Hosted ML inference SaaS or persistent backend API |

## 2. Completed Capabilities

| Area | Completed |
| --- | --- |
| Pipeline orchestration | Stage-based Python CLI with resource/status reporting |
| Demo-safe mode | CPU-safe stub/demo artifacts and manifest generation |
| Static frontend | `portfolio-web/` landing/demo page with results, pipeline cards, scenarios, media, architecture, and runbook |
| Sample data | Synthetic product-style demo outputs and stage manifest |
| Screenshots | Desktop home, desktop results, and mobile demo screenshots |
| Demo video | Short MP4 walkthrough committed with the static site |
| Tests | Demo manifest/media tests and demo-safe smoke suite |
| Docker | Nginx static image for `portfolio-web/` |
| Deployment | `gh-pages` public demo and GitHub Actions workflow for repeatable publishing |
| Portfolio integration | Main portfolio detail page has cover, media gallery, demo video, live demo, GitHub, and README links |

## 3. Gaps and Risks

| Gap / Risk | Severity | Notes | Mitigation |
| --- | --- | --- | --- |
| Real model execution is environment-specific | Medium | GPU, CUDA, local model paths, and raw media are not available in generic CI | Keep CPU-safe demo mode as the stable public path |
| Full test suite may touch heavy dependencies | Medium | Some tests can require diffusion/inpainting/training dependency alignment | Use `./tests/run_tests.sh` for interview-safe verification |
| Repo contains many batch/research scripts | Medium | Reviewers may not know which path is stable | README now defines the reviewer path and stage matrix |
| Public demo is static | Low | It cannot run real inference in browser | Position it as a showcase, not a hosted ML service |
| Raw datasets/checkpoints are excluded | Low | Reviewers cannot inspect private source media | Synthetic/anonymized assets demonstrate the workflow safely |

## 4. Technical Architecture

| Layer | Current implementation |
| --- | --- |
| Frontend | Static HTML/CSS/JS in `portfolio-web/` |
| Backend | Python CLI and stage modules under `scripts/` and `anime_pipeline/` |
| Database | None; parquet/CSV/JSON files act as stage contracts |
| API | None; browser fetches static `demo-data/manifest.json` |
| Model runtime | Local GPU workstation for real workflows |
| Demo runtime | CPU-safe stub outputs and synthetic public assets |
| Deployment | GitHub Pages via `gh-pages`; Docker/Nginx for local/container preview |

## 5. Demo Strategy

The demo is structured around what an interviewer can evaluate quickly:

1. Open the portfolio case study.
2. Confirm the project has a real cover, screenshots, video, demo URL, GitHub URL, and README.
3. Open the live demo.
4. Inspect product-style results first.
5. Scroll through stage cards to understand the pipeline.
6. Review architecture and runbook.
7. Optionally run the local smoke tests.

## 6. Verification Results

Latest local verification:

| Check | Result |
| --- | --- |
| `python scripts/demo/run_demo_pipeline.py --skip-pipeline` | Passed |
| `python -m pytest tests/demo -q` | 3 passed |
| `./tests/run_tests.sh` | 26 passed |
| `docker build -f docker/portfolio.Dockerfile -t 3d-animation-lora-pipeline-demo .` | Passed |
| `python -m scripts.core.pipeline status --project <local_project_id>` | Passed locally with configured project |

Latest public verification:

| URL | Expected |
| --- | --- |
| `https://justin21523.github.io/3d-animation-lora-pipeline/` | Live demo page |
| `https://justin21523.github.io/3d-animation-lora-pipeline/assets/video/demo-walkthrough.mp4` | Demo MP4 |
| `https://justin21523.github.io/3d-animation-lora-pipeline/assets/screenshots/demo-home-desktop.png` | Demo screenshot |
| `https://justin21523.github.io/zh-TW/projects/3d-animation-lora-pipeline/` | Main portfolio project page |

## 7. Distance to Interview-Ready Demo

Current status: interview-ready as a static ML pipeline showcase.

Remaining improvements that are useful but not blockers:

| Improvement | Priority |
| --- | --- |
| Add automated Playwright screenshot regeneration | Medium |
| Add more scenario tabs for data quality and checkpoint comparison | Medium |
| Add CI smoke tests for the static demo workflow | Medium |
| Consolidate older batch scripts into fewer documented entry points | Low |
| Add more anonymized video examples | Low |

## 8. Reviewer Highlights

- The pipeline is not a single prompt script; it models an end-to-end data workflow.
- Stage metadata makes the system inspectable and reproducible.
- The public demo avoids private data and heavy compute while preserving the real workflow shape.
- The project includes static deployment, Docker packaging, tests, screenshots, video, and main portfolio integration.
- Documentation now includes architecture diagrams, data-flow diagrams, stage tables, deployment diagrams, risk tables, and demo instructions.
