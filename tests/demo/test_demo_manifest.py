import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_demo_manifest_shape():
    manifest_path = ROOT / "portfolio-web" / "demo-data" / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert data["project"] == "3D Animation LoRA Pipeline"
    assert data["mode"] == "cpu_stub_demo"
    assert data["summary"]["stages_total"] == 10
    assert len(data["stages"]) == 10
    assert all("label" in stage and "metadata" in stage for stage in data["stages"])
    assert data["product_results"]["metrics"]
    assert data["product_results"]["deliverables"]


def test_product_demo_assets_exist():
    manifest_path = ROOT / "portfolio-web" / "demo-data" / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))

    for relative_path in data["product_results"]["assets"].values():
        asset_path = ROOT / "portfolio-web" / relative_path
        assert asset_path.exists(), relative_path
        assert asset_path.stat().st_size > 1000, relative_path


def test_portfolio_site_entrypoint_exists():
    index_path = ROOT / "portfolio-web" / "index.html"
    html = index_path.read_text(encoding="utf-8")

    assert "Animation LoRA Pipeline" in html
    assert "demo-data/manifest.json" in html
    assert 'id="results"' in html
    assert "pipeline" in html.lower()
