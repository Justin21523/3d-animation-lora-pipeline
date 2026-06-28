async function loadManifest() {
  try {
    const response = await fetch("demo-data/manifest.json");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } catch (error) {
    console.warn("Using embedded fallback manifest", error);
    return null;
  }
}

function renderSummary(data) {
  if (!data) return;
  document.getElementById("stages-complete").textContent = `${data.summary.stages_complete}/${data.summary.stages_total}`;
  document.getElementById("metadata-rows").textContent = data.summary.metadata_rows;
  document.getElementById("image-artifacts").textContent = data.summary.image_artifacts;
  document.getElementById("demo-status").textContent = data.summary.demo_ready ? "Demo ready" : "Needs attention";
}

function renderStages(data) {
  const grid = document.getElementById("stage-grid");
  if (!data || !grid) return;
  grid.innerHTML = data.stages
    .map(
      (stage) => `
        <article class="stage-card">
          <span class="number">${stage.order}</span>
          <strong>${stage.label}</strong>
          <span>${stage.status}</span>
          <dl>
            <div><dt>Rows</dt><dd>${stage.rows}</dd></div>
            <div><dt>Images</dt><dd>${stage.images}</dd></div>
          </dl>
        </article>
      `
    )
    .join("");
}

function renderScenarios(data) {
  const list = document.getElementById("scenario-list");
  if (!data || !list) return;
  list.innerHTML = data.scenarios
    .map(
      (scenario) => `
        <article>
          <h3>${scenario.title}</h3>
          <p>${scenario.description}</p>
        </article>
      `
    )
    .join("");
}

function renderProductResults(data) {
  if (!data || !data.product_results) return;
  const results = data.product_results;
  const headline = document.getElementById("results-headline");
  if (headline) headline.textContent = results.headline;

  const beforeAfter = document.getElementById("before-after-image");
  if (beforeAfter && results.assets.before_after) {
    beforeAfter.src = results.assets.before_after;
  }

  const metrics = document.getElementById("result-metrics");
  if (metrics) {
    metrics.innerHTML = results.metrics
      .map(
        (metric) => `
          <article>
            <span>${metric.value}</span>
            <strong>${metric.label}</strong>
            <small>${metric.trend}</small>
          </article>
        `
      )
      .join("");
  }

  const gallery = document.getElementById("result-gallery");
  if (gallery) {
    const cards = [
      ["Character dataset sheet", results.assets.character_sheet],
      ["Training metrics", results.assets.training_metrics],
      ["Checkpoint matrix", results.assets.evaluation_matrix],
      ["Animation strip", results.assets.animation_strip],
    ];
    gallery.innerHTML = cards
      .map(
        ([label, src]) => `
          <figure>
            <img src="${src}" alt="${label}" loading="lazy" />
            <figcaption>${label}</figcaption>
          </figure>
        `
      )
      .join("");
  }

  const deliverables = document.getElementById("deliverable-grid");
  if (deliverables) {
    deliverables.innerHTML = results.deliverables
      .map((item) => `<article><span>✓</span><strong>${item}</strong></article>`)
      .join("");
  }

  const screenshotGrid = document.getElementById("screenshot-grid");
  if (screenshotGrid && results.media && results.media.screenshots) {
    screenshotGrid.innerHTML = results.media.screenshots
      .map(
        (src) => `
          <figure>
            <img src="${src}" alt="Demo website screenshot" loading="lazy" />
            <figcaption>${src.split("/").pop()}</figcaption>
          </figure>
        `
      )
      .join("");
  }
}

loadManifest().then((data) => {
  renderSummary(data);
  renderStages(data);
  renderScenarios(data);
  renderProductResults(data);
});
