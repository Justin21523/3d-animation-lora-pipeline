# Deployment

This project should be deployed as a static portfolio/demo site. The ML pipeline itself remains a local or workstation workflow because real runs require GPU weights, private media, and large generated artifacts.

## Recommended: GitHub Pages

Publish the `portfolio-web/` directory.

Typical setup:

1. Commit `portfolio-web/`.
2. In GitHub repository settings, enable Pages.
3. Use a GitHub Actions workflow or a Pages source that serves the static directory.

Local preview:

```bash
python -m http.server 8080 -d portfolio-web
```

## Netlify or Vercel

Use these settings:

- Build command: none
- Publish directory: `portfolio-web`
- Framework preset: static site

## Docker/Nginx

The included Dockerfile serves `portfolio-web/` through Nginx.

```bash
docker build -f docker/portfolio.Dockerfile -t 3d-animation-lora-pipeline-demo .
docker run --rm -p 8080:80 3d-animation-lora-pipeline-demo
```

Then open `http://localhost:8080`.

## Demo Data Refresh

Regenerate CPU-safe pipeline outputs and update the static manifest:

```bash
bash bash/run_full_pipeline_stub.sh
python scripts/demo/run_demo_pipeline.py --skip-pipeline
```

The manifest is written to:

```text
portfolio-web/demo-data/manifest.json
```

## Not Recommended as Primary Demo

Render/Railway-style app hosting is unnecessary for the portfolio site because there is no persistent backend service. Real ML execution should stay on a GPU workstation or a dedicated batch environment.
