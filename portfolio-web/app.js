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

loadManifest().then((data) => {
  renderSummary(data);
  renderStages(data);
  renderScenarios(data);
});
