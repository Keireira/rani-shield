"use strict";

const $ = (id) => document.getElementById(id);
const viewModes = ["original", "split", "side", "result", "diff"];
const stageSpecs = {
  phantomseal: { label: "PhantomSeal", fields: {
    iterations: { label: "Optimization steps", min: 10, max: 2000, step: 1, value: 1000, integer: true },
  } },
  faceshield: { label: "FaceShield", fields: {
    iterations: { label: "Optimization steps", min: 1, max: 200, step: 1, value: 30, integer: true },
    noise_clamp: { label: "Noise limit / 255", min: 1, max: 32, step: 1, value: 12, integer: true },
    step_size: { label: "Step size", min: 0.25, max: 4, step: 0.25, value: 1 },
    analysis_size: { label: "Analysis size (px)", min: 64, max: 1280, step: 64, value: 512, integer: true, multiple: 64 },
  } },
  nullswap: { label: "NullSwap", fields: {
    strength: { label: "Perturbation strength", min: 0, max: 4, step: 0.25, value: 1 },
    seed: { label: "Seed", min: 0, max: 4294967295, step: 1, value: 0, integer: true },
  } },
};
function newStage(type) { return { type, params: Object.fromEntries(Object.entries(stageSpecs[type].fields).map(([key, field]) => [key, field.value])) }; }
function copyRecipe(value) { return { mode: "custom", max_size: value.max_size, stages: value.stages.map(stage => ({ type: stage.type, params: { ...stage.params } })) }; }
const defaults = { mode: "custom", max_size: 1280, stages: [newStage("phantomseal"), newStage("faceshield")] };
const methodLabels = { custom: "Custom pipeline", phantomseal_faceshield: "PhantomSeal → FaceShield", phantomseal: "PhantomSeal", faceshield: "FaceShield", nullswap: "NullSwap (reconstruction)" };
const backendLabels = { simswap: "SimSwap", diffswap: "DiffSwap", infoswap: "InfoSwap", uniface: "UniFace" };
const state = {
  uploadedSource: null, source: null, result: null, stage1: null, sourceImage: null, resultImage: null,
  mode: "original", split: 0.5, zoom: 1, panX: 0, panY: 0, rect: null,
  busy: false, uploading: false, initializing: true, ready: false, uploadVersion: 0, generation: 0,
  assetIndex: new Map(), engines: {},
  testing: { ready: false, message: "Test availability is not available from the server." },
  testableResultIds: new Set(), latestTestJobIds: new Map(), activeTestId: null, busyKind: null,
  testVersion: 0, testResultId: null, testJobId: null, testData: null, testError: "",
  testStage: "", testProgress: 0, testElapsed: 0,
  testBaselineCutoff: 0.80, testVisualConfirmations: new Set(),
  pair: null, pairLoading: null, pairVersion: 0, diffCanvas: null, diffKey: "",
  metricsVersion: 0, resultVersion: 0, space: false, drag: null,
  settingsDraft: null, presets: [], pipeline: copyRecipe(defaults), intermediateAssets: [],
};
let toastTimer, drawFrame, transparencyTile;

function toast(message, error = false) {
  $("toast").textContent = message;
  $("toast").classList.toggle("error", error);
  $("toast").classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => $("toast").classList.remove("show"), error ? 6500 : 3500);
}

async function api(path, data) {
  const response = await fetch(path, data === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data),
  });
  let value;
  try { value = await response.json(); }
  catch { throw new Error("Invalid server response. Check the Rani Shield terminal."); }
  if (!response.ok) {
    const error = new Error(value.error || value.detail || `Server error ${response.status}`);
    error.status = response.status;
    throw error;
  }
  return value;
}

function loadImage(asset) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(`Could not open ${asset.name || "the photo"}.`));
    image.src = asset.url;
  });
}

function setText(id, text) { if ($(id)) $(id).textContent = text; }
function dimensions(asset) { return asset ? `${asset.width.toLocaleString("en-US")} × ${asset.height.toLocaleString("en-US")}` : "No photo"; }

function invalidateDiff() {
  state.pairVersion++;
  state.pair = state.pairLoading = state.diffCanvas = null;
  state.diffKey = "";
}

function resetView() {
  state.zoom = 1;
  state.panX = state.panY = 0;
  state.split = 0.5;
  requestDraw();
}

function persistSelection() {
  try {
    localStorage.setItem("faceshield.selection.v1", JSON.stringify({ uploadedSource: state.uploadedSource, source: state.source, result: state.result, stage1: state.stage1, intermediateAssets: state.intermediateAssets }));
  } catch { /* Saving a browser selection is optional. */ }
}

function recipeFor(prefix = "") { return prefix ? state.settingsDraft || state.pipeline : state.pipeline; }
function methodNeeds(recipe) {
  const types = new Set((recipe?.stages || []).map(stage => stage.type));
  return Object.fromEntries(Object.keys(stageSpecs).map(type => [type, types.has(type)]));
}
function methodProblem(recipe) {
  for (const [engine, used] of Object.entries(methodNeeds(recipe))) {
    if (used && state.engines[engine]?.ready === false) return state.engines[engine].message || `${stageSpecs[engine].label} is not ready.`;
  }
  return "";
}
function recipeProblem(prefix = "") {
  try { normalizedSettings({ ...recipeFor(prefix), max_size: numericFieldValue(`${prefix}max-size`) }); return ""; }
  catch (error) { return error.message; }
}
function syncMethodControls(prefix = "") {
  const recipe = recipeFor(prefix), locked = state.busy || state.initializing;
  $(`${prefix}max-size`).disabled = locked;
  $(`${prefix}add-stage-type`).disabled = locked || recipe.stages.length >= 8;
  $(`${prefix}add-stage`).disabled = locked || recipe.stages.length >= 8;
  const inputs = $(`${prefix}pipeline-stages`).querySelectorAll("input, select, button");
  inputs.forEach(input => { input.disabled = locked || input.dataset.limitDisabled === "true"; });
  setText(`${prefix}pipeline-count`, `${recipe.stages.length} / 8 stages`);
  const problem = methodProblem(recipe), status = $(`${prefix}method-readiness`);
  status.classList.toggle("unavailable", Boolean(problem));
  status.textContent = state.initializing ? "Checking methods…" : problem || Object.entries(methodNeeds(recipe))
    .filter(([, used]) => used)
    .map(([engine]) => state.engines[engine]?.message || `${stageSpecs[engine].label}: ${state.engines[engine]?.ready === true ? "ready" : "status unavailable"}`)
    .join(" · ");
  const error = recipeProblem(prefix);
  setText(`${prefix}pipeline-error`, error); $(`${prefix}pipeline-error`).hidden = !error;
  if (!prefix) $("chain-note").hidden = recipe.stages.length < 2;
}
function stageNote(stage) {
  if (stage.type === "nullswap" && !Number.isFinite(stage.params.strength)) return "Choose a perturbation strength between 0 and 4. Seed must be a whole number.";
  if (stage.type === "nullswap") return `PhantomSeal's reconstruction of NullSwap. Maximum change: ${Number((8 * stage.params.strength).toFixed(2))}/255 before PNG rounding. Strength 0 adds no perturbation. The same seed repeats the initial noise.`;
  if (stage.type === "faceshield") return "The noise limit applies to this stage's input. Analysis size controls internal processing; output dimensions stay unchanged.";
  return "Optimization steps affect processing time. Each stage receives the previous stage's output.";
}
function renderPipeline(prefix = "", focusId = null) {
  const recipe = recipeFor(prefix), container = $(`${prefix}pipeline-stages`);
  container.replaceChildren();
  recipe.stages.forEach((stage, index) => {
    const card = testNode("section", "pipeline-stage");
    card.setAttribute("aria-label", `Stage ${index + 1}: ${stageSpecs[stage.type].label}`);
    const heading = testNode("div", "pipeline-stage-heading");
    heading.appendChild(testNode("strong", "stage-number", String(index + 1).padStart(2, "0")));
    const type = testNode("select", "stage-type"); type.id = `${prefix}stage-${index}-type`;
    type.setAttribute("aria-label", `Stage ${index + 1} filter`);
    type.replaceChildren(...Object.entries(stageSpecs).map(([key, spec]) => new Option(spec.label, key)));
    type.value = stage.type;
    type.addEventListener("change", () => changeStageType(index, type.value, prefix));
    heading.appendChild(type); card.appendChild(heading);
    const actions = testNode("div", "stage-actions");
    for (const [name, label, limited, callback] of [
      ["up", "Move up", index === 0, () => moveStage(index, -1, prefix)],
      ["down", "Move down", index === recipe.stages.length - 1, () => moveStage(index, 1, prefix)],
      ["remove", "Remove", recipe.stages.length === 1, () => removeStage(index, prefix)],
    ]) {
      const button = testNode("button", "stage-action", label); button.type = "button"; button.id = `${prefix}stage-${index}-${name}`;
      button.dataset.limitDisabled = String(limited); button.setAttribute("aria-label", `${label} stage ${index + 1}`);
      button.addEventListener("click", callback); actions.appendChild(button);
    }
    card.appendChild(actions);
    const parameters = testNode("div", "stage-parameters");
    for (const [key, spec] of Object.entries(stageSpecs[stage.type].fields)) {
      const input = testNode("input"), label = testNode("label", "runtime-control");
      input.id = `${prefix}stage-${index}-${key}`; input.type = "number";
      input.min = spec.min; input.max = spec.max; input.step = spec.step; input.value = Number.isNaN(stage.params[key]) ? "" : stage.params[key];
      input.inputMode = spec.integer ? "numeric" : "decimal";
      label.htmlFor = input.id; label.appendChild(testNode("span", "", spec.label)); label.appendChild(input);
      input.addEventListener("input", () => {
        setStageParameter(index, key, String(input.value).trim() === "" ? NaN : Number(input.value), prefix);
        note.textContent = stageNote(recipeFor(prefix).stages[index]);
      });
      parameters.appendChild(label);
    }
    card.appendChild(parameters);
    const note = testNode("p", "stage-note", stageNote(stage)); card.appendChild(note);
    container.appendChild(card);
  });
  syncMethodControls(prefix);
  if (focusId) $(focusId)?.focus();
}
function pipelineChanged(prefix = "", render = true, focusId = null) {
  if (render) renderPipeline(prefix, focusId); else syncMethodControls(prefix);
  if (prefix) { if (!recipeProblem(prefix)) dialogMessage(); }
  else { if (!recipeProblem()) persistLiveSettings(); updateCards(); }
}
function addStage(type, prefix = "") {
  const recipe = recipeFor(prefix);
  if (state.busy || state.initializing || (prefix && !state.settingsDraft) || recipe.stages.length >= 8 || !Object.hasOwn(stageSpecs, type)) return;
  recipe.stages.push(newStage(type)); pipelineChanged(prefix, true, `${prefix}stage-${recipe.stages.length - 1}-type`);
}
function removeStage(index, prefix = "") {
  const recipe = recipeFor(prefix);
  if (state.busy || state.initializing || (prefix && !state.settingsDraft) || recipe.stages.length <= 1 || !recipe.stages[index]) return;
  recipe.stages.splice(index, 1); pipelineChanged(prefix, true, `${prefix}stage-${Math.min(index, recipe.stages.length - 1)}-type`);
}
function moveStage(index, direction, prefix = "") {
  const recipe = recipeFor(prefix), next = index + direction;
  if (state.busy || state.initializing || (prefix && !state.settingsDraft) || ![-1, 1].includes(direction) || !recipe.stages[index] || !recipe.stages[next]) return;
  [recipe.stages[index], recipe.stages[next]] = [recipe.stages[next], recipe.stages[index]];
  pipelineChanged(prefix, true, `${prefix}stage-${next}-type`);
}
function changeStageType(index, type, prefix = "") {
  const recipe = recipeFor(prefix);
  if (state.busy || state.initializing || (prefix && !state.settingsDraft) || !recipe.stages[index] || !Object.hasOwn(stageSpecs, type)) return;
  if (recipe.stages[index].type === type) return;
  recipe.stages[index] = newStage(type); pipelineChanged(prefix, true, `${prefix}stage-${index}-type`);
}
function setStageParameter(index, key, value, prefix = "") {
  const recipe = recipeFor(prefix), stage = recipe.stages[index];
  if (state.busy || state.initializing || (prefix && !state.settingsDraft) || !stage || !Object.hasOwn(stageSpecs[stage.type].fields, key)) return;
  stage.params[key] = value; pipelineChanged(prefix, false);
}
function updateCards() {
  syncMethodControls();
  syncMethodControls("dialog-");
  $("empty-state").hidden = Boolean(state.sourceImage);
  $("processing-feedback").hidden = !state.busy;
  $("process-button").disabled = state.busy || state.uploading || state.initializing || !state.ready || !state.uploadedSource || Boolean(methodProblem(state.pipeline) || recipeProblem());
  $("process-button").title = methodProblem(state.pipeline) || recipeProblem() || "Protect the uploaded photo";
  $("upload-button").disabled = state.busy || state.initializing;
  $("file-input").disabled = state.busy || state.initializing;
  $("settings-button").disabled = state.busy || state.initializing;
  $("export-button").disabled = state.busy || state.uploading || !(state.mode === "original" ? state.source : state.result);
  $("export-stage1-button").hidden = !state.stage1;
  $("export-stage1-button").disabled = state.busy || state.uploading || !state.stage1;
  $("reset-button").disabled = state.busy || state.initializing;
  updateIntermediateExports();
  document.querySelectorAll(".mode-tabs [data-mode]").forEach((button) => {
    button.disabled = button.dataset.mode !== "original" && !state.result;
    button.classList.toggle("active", button.dataset.mode === state.mode);
    button.setAttribute("aria-pressed", String(button.dataset.mode === state.mode));
  });
  $("canvas-viewport").classList.toggle("has-image", Boolean(state.source));
  updateTestControls();
  updatePresetButtons();
}

function testEligible() { return Boolean(state.result && state.testableResultIds.has(state.result.id)); }
function currentTest(resultId, version) { return version === state.testVersion && state.result?.id === resultId && state.testResultId === resultId; }
function invalidateTest() {
  state.testVersion++;
  state.testVisualConfirmations.clear();
  state.testResultId = state.testJobId = state.testData = null;
  state.testError = state.testStage = ""; state.testProgress = state.testElapsed = 0;
  $("test-results").replaceChildren(); $("test-results").hidden = true;
  $("test-error").hidden = true; $("test-feedback").hidden = true;
  if ($("test-dialog").open) $("test-dialog").close();
}
function selectedBackend() { return $("test-backend").value || "simswap"; }
function selectedTestReadiness() { return state.testing?.backends?.[selectedBackend()] || (selectedBackend() === "simswap" ? state.testing : {ready: false, message: "This model is unavailable."}); }
function changeTestBackend() {
  if (state.busy) return;
  const resultId = state.result?.id;
  state.testVersion++; state.testVisualConfirmations.clear();
  state.testResultId = resultId; state.testJobId = state.testData = null;
  state.testError = state.testStage = "";
  state.latestTestJobIds.delete(resultId);
  renderTestResults(); updateTestControls();
}
function updateTestControls() {
  const ownTest = state.busy && state.busyKind === "test" && state.testResultId === state.result?.id;
  const blocked = state.busy || state.uploading || state.initializing;
  // Reopening an active test is a read-only action; its Run control stays locked.
  $("test-protection-button").disabled = !testEligible() || state.uploading || state.initializing || (state.busy && !ownTest);
  $("test-protection-button").textContent = ownTest ? "View test progress" : "Test protection";
  $("test-protection-button").title = testEligible() ? "Compare real face-swap outputs for this result" : "Protect a photo in this session to enable its test";
  $("test-backend").disabled = blocked;
  $("test-target").disabled = blocked || selectedTestReadiness()?.ready !== true;
  $("test-jpeg95").disabled = blocked || selectedTestReadiness()?.ready !== true;
  $("test-baseline-cutoff").disabled = blocked;
  $("run-test-button").disabled = blocked || !testEligible() || selectedTestReadiness()?.ready !== true || !["cage", "wlh", "zxy"].includes($("test-target").value);
  $("run-test-button").textContent = ownTest ? "Testing…" : `Run ${backendLabels[selectedBackend()]} test`;
  $("test-readiness").textContent = !state.result ? "Protect a photo first." : !testEligible() ? "This result is not available for testing. Protect the photo again." : selectedTestReadiness()?.ready === true ? `This ${backendLabels[selectedBackend()]} test runs locally. ${selectedBackend() === "diffswap" ? "DiffSwap uses 200 diffusion steps and takes longer. " : ""}Its swap images do not replace your protected photo.` : selectedTestReadiness()?.message || "The selected test is unavailable.";
  $("test-error").textContent = state.testError;
  $("test-error").hidden = !state.testError;
  $("test-feedback").hidden = !ownTest;
  $("test-progress").value = state.testProgress;
  $("test-status").textContent = state.testStage || "Preparing test…";
  const seconds = Math.max(0, Math.floor(state.testElapsed || 0));
  $("test-progress-label").textContent = `${Math.round(state.testProgress)}% · ${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
}
function testNode(tag, className = "", text = "") {
  const node = document.createElement(tag); node.className = className;
  if (text) node.textContent = text;
  return node;
}
function testAssetUrl(asset) { return typeof asset?.url === "string" && /^\/(?:api\/image|assets)\/[a-zA-Z0-9_-]+$/.test(asset.url) ? asset.url : null; }
function measuredTestRow(row) {
  return Boolean(row && row.source_detected === true && row.swap_detected === true && typeof row.score === "number" && Number.isFinite(row.score));
}
function baselineConfirmationKey(condition) { return JSON.stringify([state.testResultId, state.testJobId, condition.name || "condition"]); }
function baselineGate(condition) {
  const clean = (Array.isArray(condition.rows) ? condition.rows : []).find((row) => row.key === "clean");
  const cutoff = state.testBaselineCutoff;
  const validCutoff = typeof cutoff === "number" && Number.isFinite(cutoff) && cutoff >= 0 && cutoff <= 1;
  const numericPassed = validCutoff && measuredTestRow(clean) && clean.score >= cutoff;
  const visuallyConfirmed = numericPassed && state.testVisualConfirmations.has(baselineConfirmationKey(condition));
  return { clean, cutoff, validCutoff, numericPassed, visuallyConfirmed, accepted: numericPassed && visuallyConfirmed };
}
function setVisualBaseline(condition, checked) {
  if (!state.testData?.conditions?.includes(condition)) return false;
  const key = baselineConfirmationKey(condition);
  if (checked && baselineGate(condition).numericPassed) state.testVisualConfirmations.add(key);
  else state.testVisualConfirmations.delete(key);
  return state.testVisualConfirmations.has(key);
}
function changeBaselineCutoff() {
  const value = String($("test-baseline-cutoff").value).trim(), number = value ? Number(value) : NaN;
  const valid = Number.isFinite(number) && number >= 0 && number <= 1 && Math.abs(number * 100 - Math.round(number * 100)) < 1e-6;
  state.testBaselineCutoff = valid ? number : null;
  state.testVisualConfirmations.clear();
  $("test-baseline-cutoff").setAttribute("aria-invalid", String(!valid));
  renderTestResults();
}
function baselineDecisionControls(condition) {
  const panel = testNode("div", "test-baseline-gate"), title = testNode("strong"), detail = testNode("p");
  panel.appendChild(title); panel.appendChild(detail);
  const label = testNode("label", "test-visual-confirmation"), checkbox = testNode("input");
  checkbox.type = "checkbox";
  label.appendChild(checkbox); label.appendChild(testNode("span", "", "The unprotected swap looks like me"));
  const effect = testNode("div", "test-comparison-effect");
  const version = state.testVersion, data = state.testData, cutoffAtRender = state.testBaselineCutoff;
  function refresh() {
    const gate = baselineGate(condition), rawScore = gate.clean?.score;
    const scoreText = typeof rawScore === "number" && Number.isFinite(rawScore) ? rawScore.toFixed(4) : "unavailable";
    const cutoffText = gate.validCutoff ? gate.cutoff.toFixed(2) : "invalid";
    checkbox.checked = gate.visuallyConfirmed;
    checkbox.disabled = !gate.numericPassed || state.busy || state.initializing;
    effect.replaceChildren();
    if (!gate.numericPassed) {
      title.textContent = "Inconclusive — weak baseline";
      panel.className = "test-baseline-gate baseline-weak";
      const reason = !gate.validCutoff ? "Choose a cutoff from 0.00 to 1.00 in steps of 0.01." : !measuredTestRow(gate.clean) ? "The unprotected source or swap has no successful, measurable face match." : "The unprotected swap is below the selected quality cutoff.";
      detail.textContent = `Unprotected swap cosine: ${scoreText}. Quality cutoff: ${cutoffText}. ${reason} No protection-effect conclusion can be drawn from this result.`;
    } else if (!gate.visuallyConfirmed) {
      title.textContent = "Inconclusive — confirm the baseline appearance";
      panel.className = "test-baseline-gate baseline-unconfirmed";
      detail.textContent = `Unprotected swap cosine: ${scoreText}. Quality cutoff: ${cutoffText}. The numeric gate is met, but you must confirm the unprotected swap looks like you before a protection-effect comparison is shown.`;
    } else {
      title.textContent = "Baseline accepted for comparison";
      panel.className = "test-baseline-gate baseline-accepted";
      detail.textContent = `Unprotected swap cosine: ${scoreText}. Quality cutoff: ${cutoffText}. You confirmed its appearance for this condition. This accepts the baseline, not the defense.`;
      const rows = Array.isArray(condition.rows) ? condition.rows : [], stage1 = rows.find((row) => row.key === "stage1"), final = rows.find((row) => row.key === "final");
      const reference = stage1 || gate.clean;
      if (measuredTestRow(reference) && measuredTestRow(final)) {
        // Recalculate from accepted, detected rows; a server delta cannot bypass the gate.
        const delta = final.score - reference.score;
        const heading = stage1 ? "Chain − PhantomSeal cosine" : "Protected − unprotected cosine";
        effect.appendChild(testNode("p", "test-delta", `${heading}: ${delta > 0 ? "+" : ""}${delta.toFixed(4)}. Lower cosine means less identity similarity in this accepted-baseline comparison; it is not a protection guarantee.`));
      } else effect.appendChild(testNode("p", "test-score-note", "Protected swap scores are unavailable; no comparative delta can be measured."));
    }
  }
  checkbox.addEventListener("change", () => {
    if (version !== state.testVersion || data !== state.testData || cutoffAtRender !== state.testBaselineCutoff) return;
    setVisualBaseline(condition, checkbox.checked); refresh();
  });
  refresh();
  return { panel, label, effect };
}
function renderTestResults() {
  const container = $("test-results"); container.replaceChildren();
  const data = state.testData;
  container.hidden = !data;
  if (!data) return;
  const context = testNode("div", "test-result-context");
  const references = testNode("div", "test-reference-images");
  for (const [asset, caption] of [[state.source, "Your clean original"], [data.target, "Demo target"]]) {
    const url = testAssetUrl(asset);
    if (!url) continue;
    const figure = testNode("figure", "test-reference"), image = testNode("img", "test-reference-image");
    image.src = url; image.alt = caption;
    figure.appendChild(image); figure.appendChild(testNode("figcaption", "", caption));
    references.appendChild(figure);
  }
  context.appendChild(testNode("strong", "", `${data.backend_label || backendLabels[data.backend || "simswap"]} comparison`));
  context.appendChild(references);
  context.appendChild(testNode("p", "", "Compare the unprotected swap with your clean original first. Protection-effect comparisons stay hidden until the numeric quality gate and your visual confirmation both accept the baseline."));
  container.appendChild(context);
  for (const condition of Array.isArray(data.conditions) ? data.conditions : []) {
    const section = testNode("section", "test-condition");
    section.appendChild(testNode("h3", "", condition.name === "jpeg95" ? "JPEG95" : condition.name === "png" ? "PNG" : String(condition.name || "Test")));
    const decision = baselineDecisionControls(condition);
    section.appendChild(decision.panel);
    const grid = testNode("div", "test-swap-grid");
    for (const row of Array.isArray(condition.rows) ? condition.rows : []) {
      const card = testNode("article", "test-swap-card");
      const rowLabel = row.key === "clean" ? "Unprotected swap" : row.label || ({ stage1: "PhantomSeal", final: "Protected result" }[row.key]) || "Swap";
      card.appendChild(testNode("h4", "", rowLabel));
      const swapUrl = testAssetUrl(row.swap);
      if (swapUrl) {
        const image = testNode("img", "test-swap-image"); image.src = swapUrl; image.alt = `${rowLabel} ${data.backend_label || backendLabels[data.backend || "simswap"]} output`;
        card.appendChild(image);
      } else card.appendChild(testNode("div", "test-swap-placeholder", "Swap image unavailable"));
      const finite = typeof row.score === "number" && Number.isFinite(row.score) && row.source_detected === true && row.swap_detected === true;
      card.appendChild(testNode("p", "test-score", finite ? `FaceNet cosine · ${row.score.toFixed(4)}` : "FaceNet cosine · unavailable"));
      const detection = row.source_detected === false ? "Source face not detected" : row.swap_detected === false ? "Swap face not detected" : !finite ? "Similarity could not be measured" : "Similarity to the clean original identity";
      card.appendChild(testNode("p", "test-score-note", detection));
      grid.appendChild(card);
    }
    section.appendChild(grid);
    section.appendChild(decision.label); section.appendChild(decision.effect);
    container.appendChild(section);
  }
  container.appendChild(testNode("p", "test-limit-note", data.note || "One model and target do not establish general protection. JPEG and other edits can change the outcome."));
  container.appendChild(testNode("p", "test-limit-note", "The photo is tested as an identity source, not as the destination of a face replacement. SimSwap and FaceNet also participate in PhantomSeal; these measurements are diagnostic."));
}
async function openTestDialog() {
  if (!testEligible() || state.initializing || state.uploading || (state.busy && state.busyKind !== "test")) return;
  const resultId = state.result.id;
  if (state.testResultId !== resultId) {
    invalidateTest(); state.testResultId = resultId;
  }
  if (!$("test-dialog").open) $("test-dialog").showModal();
  renderTestResults();
  updateTestControls();
  const previous = state.latestTestJobIds.get(resultId);
  if (previous && !state.testJobId && !state.testData && !state.activeTestId) await watchTestJob(previous, resultId);
}
function closeTestDialog() { $("test-dialog").close(); }
async function startProtectionTest() {
  if (state.busy || state.uploading || state.initializing || !testEligible() || selectedTestReadiness()?.ready !== true) return;
  const resultId = state.result.id, targetKey = $("test-target").value;
  if (!["cage", "wlh", "zxy"].includes(targetKey)) { state.testError = "Choose a supported demo face."; updateTestControls(); return; }
  const version = ++state.testVersion, owner = `starting-${version}`;
  state.testVisualConfirmations.clear();
  state.testResultId = resultId; state.testJobId = state.testData = null; state.testError = "";
  state.testProgress = state.testElapsed = 0; state.testStage = `Preparing ${backendLabels[selectedBackend()]} test…`;
  state.activeTestId = owner; renderTestResults(); busy(true, "test");
  try {
    const job = await api("/api/test", { result_id: resultId, target_key: targetKey, backend: selectedBackend(), jpeg95: $("test-jpeg95").checked });
    if (!job.id || (job.kind && job.kind !== "test")) throw new Error("The server did not create a test job.");
    state.latestTestJobIds.set(resultId, job.id);
    await watchTestJob(job.id, resultId, null, version);
  } catch (error) {
    if (currentTest(resultId, version)) { state.testError = error.message; state.testStage = "Test failed"; updateTestControls(); }
  } finally {
    if (state.activeTestId === owner) { state.activeTestId = null; busy(false); }
  }
}
async function watchTestJob(id, expectedResult = state.result?.id, firstJob = null, suppliedVersion = null) {
  const version = suppliedVersion ?? ++state.testVersion;
  if (version === state.testVersion && state.result?.id === expectedResult) {
    if (state.testJobId !== id || state.testResultId !== expectedResult) state.testVisualConfirmations.clear();
    state.testResultId = expectedResult; state.testJobId = id; state.testError = "";
  }
  state.activeTestId = id; busy(true, "test");
  let failures = 0, pending = firstJob;
  try {
    for (;;) {
      let job;
      try { job = pending || await api(`/api/job/${encodeURIComponent(id)}`); pending = null; failures = 0; }
      catch (error) {
        if (error.status || ++failures > 8) throw error;
        if (currentTest(expectedResult, version)) { state.testStage = "Connection interrupted. Reconnecting…"; updateTestControls(); }
        setText("live-status", "Test connection interrupted. Reconnecting…");
        await new Promise((resolve) => setTimeout(resolve, 1000)); continue;
      }
      if (job.kind !== "test") throw new Error("The returned job is not a protection test.");
      if (job.result_id && expectedResult && job.result_id !== expectedResult) throw new Error("The test belongs to a different protected result.");
      const progress = Math.max(0, Math.min(100, Number(job.progress) || 0));
      const elapsed = Math.max(0, Number(job.elapsed) || 0);
      $("progress-bar").value = progress; setText("progress-label", `${Math.round(progress)}%`);
      setText("live-status", job.stage || "Testing protection…");
      if (currentTest(expectedResult, version)) {
        $("test-backend").value = job.backend || "simswap";
        state.testProgress = progress; state.testElapsed = elapsed; state.testStage = job.stage || "Testing protection…";
        updateTestControls();
      }
      if (job.status === "error") throw new Error(job.error || "The face-swap test failed.");
      if (job.status === "done") {
        if (!job.test || !Array.isArray(job.test.conditions)) throw new Error("The completed test has no results.");
        state.latestTestJobIds.set(expectedResult, id);
        if (currentTest(expectedResult, version)) {
          state.testData = job.test; state.testProgress = 100; state.testStage = "Test complete";
          renderTestResults(); updateTestControls(); toast("Face-swap test complete");
        }
        return job.test;
      }
      await new Promise((resolve) => setTimeout(resolve, 700));
    }
  } catch (error) {
    if (currentTest(expectedResult, version)) { state.testError = error.message; state.testStage = "Test failed"; updateTestControls(); }
    return null;
  } finally {
    if (state.activeTestId === id) { state.activeTestId = null; busy(false); }
  }
}

function setMetrics(metrics) {
  setText("metric-ssim", metrics?.ssim == null ? "—" : Number(metrics.ssim).toFixed(4));
  setText("metric-psnr", !metrics ? "—" : metrics.identical ? "∞ dB" : metrics.psnr_db == null ? "—" : `${Number(metrics.psnr_db).toFixed(2)} dB`);
  setText("metric-changed", metrics?.changed_pct == null ? "—" : `${Number(metrics.changed_pct).toFixed(1)}%`);
}

async function updateMetrics() {
  const version = ++state.metricsVersion;
  if (!state.source || !state.result) { setMetrics(null); return; }
  try {
    const metrics = await api("/api/compare", { source_id: state.source.id, result_id: state.result.id });
    if (version === state.metricsVersion) setMetrics(metrics);
  } catch (error) { if (version === state.metricsVersion) { setMetrics(null); toast(error.message, true); } }
}

async function useSource(asset, originalUpload = true, uploadToken = null) {
  const image = await loadImage(asset);
  // Decode can complete out of order even after the HTTP response is current.
  if (uploadToken !== null && uploadToken !== state.uploadVersion) return false;
  invalidateTest();
  state.generation++;
  state.metricsVersion++;
  state.resultVersion++;
  if (originalUpload) state.uploadedSource = asset;
  state.source = asset;
  state.sourceImage = image;
  state.result = state.resultImage = state.stage1 = null;
  state.intermediateAssets = [];
  invalidateDiff();
  setMetrics(null);
  setMode("original");
  resetView();
  $("result-summary").hidden = true;
  persistSelection();
  updateCards();
  return true;
}

async function useResult(asset, metrics) {
  if (!state.source) throw new Error("Open an original photo first.");
  if (asset.width !== state.source.width || asset.height !== state.source.height) {
    throw new Error("The comparison source and protected photo have different dimensions.");
  }
  const generation = state.generation;
  const version = ++state.resultVersion;
  const image = await loadImage(asset);
  if (generation !== state.generation || version !== state.resultVersion) return false;
  if (state.result?.id !== asset.id) invalidateTest();
  state.result = asset;
  state.resultImage = image;
  invalidateDiff();
  updateCards();
  requestDraw();
  persistSelection();
  if (metrics) { state.metricsVersion++; setMetrics(metrics); }
  else updateMetrics();
  return true;
}

async function upload(file) {
  if (!file) return;
  if (state.busy) { toast("Wait for the current job to finish."); return; }
  if (state.initializing) { toast("Connecting to Rani Shield…"); return; }
  if (file.size > 32 * 1024 * 1024) { toast("Maximum file size: 32 MB.", true); return; }
  const version = ++state.uploadVersion;
  state.uploading = true;
  updateCards();
  try {
    const data = await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = () => reject(new Error("Could not read the file."));
      reader.readAsDataURL(file);
    });
    if (version !== state.uploadVersion) return;
    const asset = await api("/api/upload", { name: file.name, data, role: "source" });
    if (version !== state.uploadVersion) return;
    if (await useSource(asset, true, version)) toast("Photo loaded");
  } catch (error) { if (version === state.uploadVersion) toast(error.message, true); }
  finally { if (version === state.uploadVersion) { state.uploading = false; updateCards(); } }
}

function setMode(mode) {
  if (!viewModes.includes(mode)) return;
  if (mode !== "original" && !state.result) mode = "original";
  state.mode = mode;
  $("diff-controls").hidden = mode !== "diff";
  $("canvas-viewport").dataset.mode = mode;
  $("export-button").title = mode === "diff" ? "Export the full-resolution difference map" : mode === "original" ? "Export original as PNG" : "Export protected photo as PNG";
  updateCards();
  requestDraw();
}

// Canvas interaction follows the DeSynth editor: fit, pan, split, side-by-side.
function imageRect(image, area) {
  const fit = Math.min(area.width / image.naturalWidth, area.height / image.naturalHeight);
  const scale = fit * state.zoom;
  const width = image.naturalWidth * scale;
  const height = image.naturalHeight * scale;
  return { x: area.x + (area.width - width) / 2 + state.panX, y: area.y + (area.height - height) / 2 + state.panY, width, height, scale };
}

async function exactPixels(asset) {
  const response = await fetch(`/api/pixels/${encodeURIComponent(asset.id)}`);
  if (!response.ok) throw new Error("Could not read the photo pixel data.");
  const pixels = new Uint8Array(await response.arrayBuffer());
  if (pixels.length !== asset.width * asset.height * 4) throw new Error("Incomplete RGBA pixel data.");
  return pixels;
}

function prepareDiffPixels() {
  if (state.pair) return Promise.resolve();
  if (state.pairLoading) return state.pairLoading;
  if (!state.source || !state.result) return Promise.resolve();
  const version = state.pairVersion;
  state.pairLoading = Promise.all([exactPixels(state.source), exactPixels(state.result)])
    .then(([a, b]) => { if (version === state.pairVersion) { state.pair = { a, b }; requestDraw(); } })
    .catch((error) => { if (version === state.pairVersion) toast(error.message, true); })
    .finally(() => { if (version === state.pairVersion) state.pairLoading = null; });
  return state.pairLoading;
}

function diffImage() {
  if (!state.sourceImage || !state.resultImage) return null;
  const gain = Math.min(32, Number($("diff-gain").value));
  const channel = $("diff-channel").value;
  const key = `${state.source.id}:${state.result.id}:${gain}:${channel}`;
  if (state.diffKey === key) return state.diffCanvas;
  if (!state.pair) { prepareDiffPixels(); return null; }
  const { a, b } = state.pair;
  const canvas = document.createElement("canvas");
  canvas.width = state.sourceImage.naturalWidth;
  canvas.height = state.sourceImage.naturalHeight;
  const context = canvas.getContext("2d");
  const output = context.createImageData(canvas.width, canvas.height);
  const selected = { red: 0, green: 1, blue: 2 }[channel];
  for (let index = 0; index < a.length; index += 4) {
    for (let color = 0; color < 3; color++) {
      const offset = index + (selected ?? color);
      output.data[index + color] = Math.floor(Math.min(255, Math.abs(a[offset] - b[offset]) * gain));
    }
    output.data[index + 3] = 255;
  }
  context.putImageData(output, 0, 0);
  state.diffCanvas = canvas;
  state.diffKey = key;
  return canvas;
}

function requestDraw() {
  cancelAnimationFrame(drawFrame);
  drawFrame = requestAnimationFrame(draw);
}

function draw() {
  const viewport = $("canvas-viewport");
  const canvas = $("comparison-canvas");
  const width = viewport.clientWidth, height = viewport.clientHeight;
  if (!width || !height) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
    canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr);
  }
  canvas.style.width = `${width}px`; canvas.style.height = `${height}px`;
  const context = canvas.getContext("2d");
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  context.clearRect(0, 0, width, height);
  $("divider").hidden = $("before-label").hidden = $("after-label").hidden = true;
  if (!state.sourceImage) return;
  if (!transparencyTile) {
    transparencyTile = document.createElement("canvas");
    transparencyTile.width = transparencyTile.height = 16;
    const tile = transparencyTile.getContext("2d");
    tile.fillStyle = "#282828"; tile.fillRect(0, 0, 16, 16);
    tile.fillStyle = "#353535"; tile.fillRect(0, 0, 8, 8); tile.fillRect(8, 8, 8, 8);
  }
  const drawAsset = (image, rect) => {
    context.fillStyle = context.createPattern(transparencyTile, "repeat");
    context.fillRect(rect.x, rect.y, rect.width, rect.height);
    context.drawImage(image, rect.x, rect.y, rect.width, rect.height);
  };
  const rect = imageRect(state.sourceImage, { x: 0, y: 0, width, height });
  state.rect = rect;
  setText("zoom-label", `${Math.round(rect.scale * 100)}%`);
  if (state.mode === "side" && state.resultImage) {
    const half = (width - 12) / 2;
    const left = imageRect(state.sourceImage, { x: 0, y: 0, width: half, height });
    const right = imageRect(state.resultImage, { x: half + 12, y: 0, width: half, height });
    context.save(); context.beginPath(); context.rect(0, 0, width / 2, height); context.clip(); drawAsset(state.sourceImage, left); context.restore();
    context.save(); context.beginPath(); context.rect(width / 2, 0, width / 2, height); context.clip(); drawAsset(state.resultImage, right); context.restore();
    $("before-label").hidden = $("after-label").hidden = false;
    setText("zoom-label", `${Math.round(left.scale * 100)}%`);
  } else if (state.mode === "split" && state.resultImage) {
    context.save(); context.beginPath(); context.rect(rect.x + rect.width * state.split, rect.y, rect.width * (1 - state.split), rect.height); context.clip(); drawAsset(state.resultImage, rect); context.restore();
    context.save(); context.beginPath(); context.rect(rect.x, rect.y, rect.width * state.split, rect.height); context.clip(); drawAsset(state.sourceImage, rect); context.restore();
    const x = rect.x + rect.width * state.split, top = Math.max(0, rect.y), bottom = Math.min(height, rect.y + rect.height);
    const divider = $("divider");
    divider.hidden = x < 0 || x > width || bottom <= top;
    divider.style.left = `${x}px`; divider.style.top = `${top}px`; divider.style.height = `${bottom - top}px`;
    $("divider-handle").setAttribute("aria-valuenow", String(Math.round(state.split * 100)));
    $("divider-handle").setAttribute("aria-valuetext", `${Math.round(state.split * 100)} percent original`);
    $("before-label").hidden = $("after-label").hidden = false;
  } else if (state.mode === "diff" && state.resultImage) {
    const image = diffImage(); if (image) drawAsset(image, rect);
  } else drawAsset(state.mode === "result" && state.resultImage ? state.resultImage : state.sourceImage, rect);
}

function zoomBy(factor) { state.zoom = Math.max(0.25, Math.min(8, state.zoom * factor)); requestDraw(); }
function updateSplit(clientX) {
  const bounds = $("canvas-viewport").getBoundingClientRect();
  if (state.rect) state.split = Math.max(0, Math.min(1, (clientX - bounds.left - state.rect.x) / state.rect.width));
  requestDraw();
}

function setupCanvas() {
  const viewport = $("canvas-viewport"), handle = $("divider-handle"), canvas = $("comparison-canvas");
  handle.addEventListener("pointerdown", (event) => {
    event.preventDefault(); event.stopPropagation(); state.drag = { type: "split" };
    handle.setPointerCapture(event.pointerId); updateSplit(event.clientX);
  });
  handle.addEventListener("pointermove", (event) => { if (state.drag?.type === "split") updateSplit(event.clientX); });
  for (const name of ["pointerup", "pointercancel"]) handle.addEventListener(name, () => { state.drag = null; });
  handle.addEventListener("keydown", (event) => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    state.split = event.key === "Home" ? 0 : event.key === "End" ? 1 : Math.max(0, Math.min(1, state.split + (event.key === "ArrowLeft" ? -1 : 1) * (event.shiftKey ? 0.1 : 0.02)));
    requestDraw();
  });
  canvas.addEventListener("pointerdown", (event) => {
    if (!state.source) return;
    event.preventDefault();
    if (state.mode === "split" && state.resultImage && state.zoom <= 1 && !state.space && !event.shiftKey) { state.drag = { type: "split" }; updateSplit(event.clientX); }
    else state.drag = { type: "pan", x: event.clientX, y: event.clientY, panX: state.panX, panY: state.panY };
    canvas.setPointerCapture(event.pointerId); viewport.classList.add("dragging");
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!state.drag) return;
    if (state.drag.type === "split") updateSplit(event.clientX);
    else { state.panX = state.drag.panX + event.clientX - state.drag.x; state.panY = state.drag.panY + event.clientY - state.drag.y; requestDraw(); }
  });
  for (const name of ["pointerup", "pointercancel"]) canvas.addEventListener(name, () => { state.drag = null; viewport.classList.remove("dragging"); });
  viewport.addEventListener("wheel", (event) => { if (!state.source) return; event.preventDefault(); zoomBy(Math.exp(-event.deltaY * 0.0015)); }, { passive: false });
  canvas.addEventListener("dblclick", resetView);
  new ResizeObserver(requestDraw).observe(viewport);
}

function validateNumber(number, name, spec) {
  if (typeof number !== "number" || !Number.isFinite(number) || number < spec.min || number > spec.max || (spec.integer && !Number.isInteger(number))) {
    throw new Error(`${name}: choose ${spec.integer ? "a whole number" : "a number"} between ${spec.min} and ${spec.max}.`);
  }
  if (spec.multiple && number % spec.multiple !== 0) throw new Error(`${name}: use a multiple of ${spec.multiple}.`);
  return number;
}
function normalizedSettings(value = {}, legacy = false) {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Invalid settings.");
  const mode = value.mode ?? (legacy ? "faceshield" : "custom");
  if (typeof mode !== "string" || !Object.hasOwn(methodLabels, mode)) throw new Error("Choose a supported protection method.");
  let stages = value.stages;
  if (mode !== "custom") {
    const types = mode === "phantomseal_faceshield" ? ["phantomseal", "faceshield"] : [mode];
    stages = types.map(type => {
      const names = type === "phantomseal" ? { iterations: "phantom_iterations" } : type === "nullswap" ? { strength: "nullswap_strength", seed: "nullswap_seed" } : { iterations: "iterations", noise_clamp: "noise_clamp", step_size: "step_size", analysis_size: "analysis_size" };
      return { type, params: Object.fromEntries(Object.entries(names).filter(([, old]) => Object.hasOwn(value, old)).map(([key, old]) => [key, value[old]])) };
    });
  } else if (stages === undefined && value.mode === undefined) stages = defaults.stages;
  if (!Array.isArray(stages) || stages.length < 1 || stages.length > 8) throw new Error("Choose between 1 and 8 pipeline stages.");
  const result = { mode: "custom", max_size: validateNumber(value.max_size ?? 1280, "Maximum long edge", {min:128,max:1280,integer:true}), stages: [] };
  for (const [index, stage] of stages.entries()) {
    if (!stage || typeof stage !== "object" || Array.isArray(stage) || typeof stage.type !== "string" || !Object.hasOwn(stageSpecs, stage.type)) throw new Error(`Stage ${index + 1}: choose a supported filter.`);
    if (Object.keys(stage).some(key => !["type", "params"].includes(key))) throw new Error(`Stage ${index + 1}: unknown stage fields.`);
    const params = stage.params === undefined ? {} : stage.params;
    if (!params || typeof params !== "object" || Array.isArray(params)) throw new Error(`Stage ${index + 1}: invalid parameters.`);
    const spec = stageSpecs[stage.type];
    for (const key of Object.keys(params)) if (!Object.hasOwn(spec.fields, key)) throw new Error(`Stage ${index + 1}: unsupported parameter ${key}.`);
    const normalized = {};
    for (const [key, field] of Object.entries(spec.fields)) normalized[key] = validateNumber(Object.hasOwn(params, key) ? params[key] : field.value, `Stage ${index + 1} ${field.label}`, field);
    result.stages.push({ type: stage.type, params: normalized });
  }
  return result;
}
function numericFieldValue(id) { return String($(id).value).trim() === "" ? NaN : Number($(id).value); }
function settings() { return normalizedSettings({ ...state.pipeline, max_size: numericFieldValue("max-size") }); }
function applySettings(value, legacy = false) {
  state.pipeline = normalizedSettings(value, legacy);
  $("max-size").value = state.pipeline.max_size;
  renderPipeline(); refreshControlValues(); updateCards();
}
function persistLiveSettings() {
  try { localStorage.setItem("faceshield.settings.v1", JSON.stringify(settings())); }
  catch { toast("Could not save settings in this browser.", true); }
}
function refreshControlValues() {
  syncMethodControls();
  setText("max-size-value", `${$("max-size").value} px`);
  setText("diff-gain-value", `×${$("diff-gain").value}`);
  document.querySelectorAll('input[type="range"]').forEach((range) => {
    range.style.setProperty("--range-progress", `${100 * (Number(range.value) - Number(range.min)) / (Number(range.max) - Number(range.min))}%`);
  });
}
function readDialogSettings() { return normalizedSettings({ ...state.settingsDraft, max_size: numericFieldValue("dialog-max-size") }); }
function dialogMessage(message = "") { $("settings-message").textContent = message; $("settings-message").hidden = !message; }
function renderSettingsDialog() {
  if (!state.settingsDraft) return;
  $("dialog-max-size").value = state.settingsDraft.max_size;
  renderPipeline("dialog-"); dialogMessage(); updatePresetButtons();
}
function openSettingsDialog() {
  if (state.busy || state.initializing) return;
  state.settingsDraft = copyRecipe(state.pipeline);
  state.settingsDraft.max_size = numericFieldValue("max-size");
  $("preset-name").value = "";
  renderPresets(""); renderSettingsDialog(); $("settings-dialog").showModal();
}
function closeSettingsDialog() { $("settings-dialog").close(); state.settingsDraft = null; dialogMessage(); updatePresetButtons(); }
function saveSettingsDialog() {
  if (state.busy || state.initializing || !state.settingsDraft) return;
  try { applySettings(readDialogSettings()); persistLiveSettings(); closeSettingsDialog(); toast("Settings saved"); }
  catch (error) { dialogMessage(error.message); }
}
function presetName(value) {
  if (typeof value !== "string") throw new Error("Enter a preset name.");
  const name = value.trim().replace(/\s+/gu, " ");
  if (!name || name.length > 80) throw new Error("Preset name must contain 1–80 characters.");
  return name;
}
function presetNameKey(value) { return presetName(value).normalize("NFKC").toLowerCase(); }
function readPresets() {
  try {
    const saved = JSON.parse(localStorage.getItem("faceshield.presets.v1") || '{"version":1,"presets":[]}');
    if (saved.version !== 1 || !Array.isArray(saved.presets)) throw new Error("Invalid preset storage.");
    const ids = new Set(), names = new Set(); state.presets = [];
    for (const item of saved.presets) {
      if (!item || typeof item.id !== "string" || !item.id || ids.has(item.id)) continue;
      try {
        const name = presetName(item.name), key = presetNameKey(name);
        if (names.has(key)) continue;
        const params = normalizedSettings(item.params, true);
        ids.add(item.id); names.add(key); state.presets.push({ id: item.id, name, params });
      } catch { /* Ignore invalid browser entries. */ }
    }
  } catch { state.presets = []; toast("Could not read saved presets.", true); }
}
function persistPresets(presets) {
  try { localStorage.setItem("faceshield.presets.v1", JSON.stringify({ version: 1, presets })); return true; }
  catch { toast("Could not save presets in this browser.", true); return false; }
}
function renderPresets(selected = "") {
  $("preset-select").replaceChildren(new Option("Custom", ""), ...state.presets.map((item) => new Option(item.name, item.id)));
  $("preset-select").value = state.presets.some((item) => item.id === selected) ? selected : "";
  updatePresetButtons();
}
function updatePresetButtons() {
  const disabled = !state.settingsDraft || state.busy;
  for (const id of ["preset-select", "preset-name", "save-preset"]) $(id).disabled = disabled;
  $("delete-preset").disabled = disabled || !state.presets.some((item) => item.id === $("preset-select").value);
}
function selectPreset() {
  const selected = state.presets.find((item) => item.id === $("preset-select").value);
  if (!selected || !state.settingsDraft) { updatePresetButtons(); return; }
  state.settingsDraft = normalizedSettings(selected.params); $("preset-name").value = selected.name; renderSettingsDialog();
}
function savePreset() {
  if (!state.settingsDraft || state.busy) return;
  try {
    const name = presetName($("preset-name").value), params = readDialogSettings();
    const existing = state.presets.find((item) => presetNameKey(item.name) === presetNameKey(name));
    const entry = { id: existing?.id || crypto.randomUUID(), name, params };
    const presets = existing ? state.presets.map((item) => item.id === entry.id ? entry : item) : [...state.presets, entry];
    if (!persistPresets(presets)) return;
    state.presets = presets; state.settingsDraft = params; $("preset-name").value = name; renderPresets(entry.id); dialogMessage(); toast(existing ? "Preset updated" : "Preset saved");
  } catch (error) { dialogMessage(error.message); }
}
function deletePreset() {
  if (!state.settingsDraft || state.busy) return;
  const selected = $("preset-select").value;
  if (!state.presets.some((item) => item.id === selected)) return;
  const presets = state.presets.filter((item) => item.id !== selected);
  if (!persistPresets(presets)) return;
  state.presets = presets; $("preset-name").value = ""; renderPresets(""); toast("Preset deleted");
}

function busy(value, kind = "protect") {
  state.busy = value; state.busyKind = value ? kind : null; document.body.dataset.busy = String(value);
  setText("process-label", value ? kind === "test" ? "Testing…" : "Protecting…" : "Protect photo");
  $("settings-form").querySelectorAll("input, select, button").forEach((input) => { input.disabled = value; });
  updateCards();
  if (state.testData) renderTestResults();
}

async function processImage() {
  if (state.busy || state.uploading || state.initializing || state.settingsDraft || !state.uploadedSource || !state.ready) return;
  const sourceId = state.uploadedSource.id;
  try {
    const params = settings();
    const unavailable = methodProblem(params);
    if (unavailable) throw new Error(unavailable);
    invalidateTest(); state.stage1 = null; state.intermediateAssets = []; busy(true); $("result-summary").hidden = true;
    $("progress-bar").value = 0; setText("progress-label", "Preparing");
    setText("live-status", "Loading models. The first run takes longer…");
    const job = await api("/api/process", { source_id: sourceId, params });
    await watchJob(job.id, sourceId);
  } catch (error) { toast(error.message, true); }
  finally { busy(false); }
}

async function watchJob(id, expectedSource = null, firstJob = null) {
  let failures = 0, pending = firstJob;
  for (;;) {
    let job;
    try { job = pending || await api(`/api/job/${encodeURIComponent(id)}`); pending = null; failures = 0; }
    catch (error) {
      if (error.status || ++failures > 8) throw error;
      setText("live-status", "Connection interrupted. Reconnecting…");
      await new Promise((resolve) => setTimeout(resolve, 1000)); continue;
    }
    if (job.kind === "test") return watchTestJob(id, job.result_id || state.result?.id, job);
    if (expectedSource && job.source_id && job.source_id !== expectedSource) throw new Error("The job does not match this photo.");
    if (job.source_id && state.uploadedSource?.id !== job.source_id && state.assetIndex.has(job.source_id)) {
      state.uploadedSource = state.assetIndex.get(job.source_id);
    }
    $("progress-bar").value = Math.max(0, Math.min(100, Number(job.progress) || 0));
    const seconds = Math.max(0, Math.floor(job.elapsed || 0));
    setText("progress-label", `${Math.round(Number(job.progress) || 0)}% · ${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`);
    setText("live-status", job.stage || "Protecting…");
    if (job.stage1) { state.stage1 = job.stage1; updateCards(); persistSelection(); }
    if (job.status === "error") throw new Error(job.error || "Protection failed.");
    if (job.status === "done") {
      if (!job.result) throw new Error("The completed job has no protected photo.");
      // Use a same-sized baseline for comparison without losing the full input.
      if (job.source) await useSource(job.source, false);
      if (!state.uploadedSource && job.source) state.uploadedSource = job.source;
      await useResult(job.result, job.metrics);
      state.testableResultIds.add(job.result.id);
      state.stage1 = job.stage1 || null;
      state.intermediateAssets = validIntermediateAssets(job.intermediate_assets);
      setMode("split"); persistSelection();
      setText("result-summary", `${methodLabels[job.params?.mode || job.method] || "Protection complete"} · ${dimensions(job.result)} px · ${Number(job.elapsed || 0).toFixed(1)} s`);
      $("result-summary").hidden = false; toast("Photo protected"); return;
    }
    await new Promise((resolve) => setTimeout(resolve, 700));
  }
}

async function download() {
  if (state.busy || state.uploading) return;
  if (state.mode === "diff") {
    await prepareDiffPixels(); const canvas = diffImage();
    if (!canvas) { toast("Difference map is not ready.", true); return; }
    canvas.toBlob((blob) => {
      if (!blob) { toast("Could not export the difference map.", true); return; }
      const url = URL.createObjectURL(blob);
      downloadUrl(url, `${state.source.name.replace(/\.[^.]+$/, "")}_diff_${$("diff-channel").value}_x${$("diff-gain").value}.png`);
      setTimeout(() => URL.revokeObjectURL(url), 10000);
    }, "image/png");
  } else {
    const asset = state.mode === "original" ? state.source : state.result;
    if (asset) downloadUrl(asset.url, asset.name.replace(/\.[^.]+$/, "") + ".png");
  }
}
function validIntermediateAssets(value) {
  return (Array.isArray(value) ? value : []).filter(item => item && Number.isInteger(item.index) && item.index >= 1 && item.index <= 8 && Object.hasOwn(stageSpecs, item.type) && typeof item.asset?.id === "string" && typeof item.asset?.name === "string" && testAssetUrl(item.asset));
}
function updateIntermediateExports() {
  const select = $("stage-output-select"), selected = select.value;
  $("stage-output-controls").hidden = state.intermediateAssets.length === 0;
  select.replaceChildren(...state.intermediateAssets.map(item => new Option(`${item.index}. ${stageSpecs[item.type].label}${item.asset.id === state.result?.id ? " · Final" : ""}`, String(item.index))));
  select.value = state.intermediateAssets.some(item => String(item.index) === selected) ? selected : String(state.intermediateAssets[0]?.index || "");
  select.disabled = $("export-stage-button").disabled = state.busy || state.uploading || !state.intermediateAssets.length;
}
function downloadIntermediate() {
  if (state.busy || state.uploading) return;
  const item = state.intermediateAssets.find(stage => String(stage.index) === $("stage-output-select").value);
  if (item) downloadUrl(item.asset.url, item.asset.name.replace(/\.[^.]+$/, "") + ".png");
}
function downloadStage1() {
  if (state.busy || state.uploading || !state.stage1) return;
  downloadUrl(state.stage1.url, state.stage1.name.replace(/\.[^.]+$/, "") + ".png");
}
function downloadUrl(url, name) {
  const link = document.createElement("a"); link.href = url; link.download = name;
  document.body.appendChild(link); link.click(); link.remove();
}

function bindEvents() {
  $("upload-button").addEventListener("click", () => { if (!state.busy && !state.initializing) $("file-input").click(); });
  $("file-input").addEventListener("change", (event) => { upload(event.target.files[0]); event.target.value = ""; });
  let dragDepth = 0;
  $("drop-zone").addEventListener("dragover", (event) => event.preventDefault());
  $("drop-zone").addEventListener("dragenter", (event) => { event.preventDefault(); if (state.busy || state.initializing) return; dragDepth++; $("drop-zone").classList.add("drag-over"); });
  $("drop-zone").addEventListener("dragleave", () => { if (--dragDepth <= 0) { dragDepth = 0; $("drop-zone").classList.remove("drag-over"); } });
  $("drop-zone").addEventListener("drop", (event) => { event.preventDefault(); dragDepth = 0; $("drop-zone").classList.remove("drag-over"); upload(event.dataTransfer.files[0]); });
  document.querySelectorAll(".mode-tabs [data-mode]").forEach((button) => button.addEventListener("click", () => setMode(button.dataset.mode)));
  $("zoom-in").addEventListener("click", () => zoomBy(1.25)); $("zoom-out").addEventListener("click", () => zoomBy(0.8));
  $("fit-button").addEventListener("click", resetView); $("export-button").addEventListener("click", download);
  $("export-stage1-button").addEventListener("click", downloadStage1);
  $("export-stage-button").addEventListener("click", downloadIntermediate);
  $("test-protection-button").addEventListener("click", openTestDialog);
  $("run-test-button").addEventListener("click", startProtectionTest);
  $("test-backend").addEventListener("change", changeTestBackend);
  $("test-baseline-cutoff").addEventListener("input", changeBaselineCutoff);
  for (const id of ["close-test", "dismiss-test"]) $(id).addEventListener("click", closeTestDialog);
  $("test-dialog").addEventListener("cancel", (event) => { event.preventDefault(); closeTestDialog(); });
  $("test-dialog").addEventListener("click", (event) => {
    if (event.target !== event.currentTarget) return;
    const box = event.currentTarget.getBoundingClientRect();
    if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) closeTestDialog();
  });
  $("process-button").addEventListener("click", processImage);
  $("settings-form").addEventListener("submit", (event) => { event.preventDefault(); processImage(); });
  $("max-size").addEventListener("input", () => { state.pipeline.max_size = numericFieldValue("max-size"); refreshControlValues(); if (!recipeProblem()) persistLiveSettings(); updateCards(); });
  for (const prefix of ["", "dialog-"]) $(`${prefix}add-stage`).addEventListener("click", () => addStage($(`${prefix}add-stage-type`).value, prefix));
  $("reset-button").addEventListener("click", () => { applySettings(defaults); persistLiveSettings(); toast("Parameters reset"); });
  $("diff-gain").addEventListener("input", () => { refreshControlValues(); requestDraw(); }); $("diff-channel").addEventListener("change", requestDraw);
  $("settings-button").addEventListener("click", openSettingsDialog);
  for (const id of ["close-settings", "cancel-settings"]) $(id).addEventListener("click", closeSettingsDialog);
  $("save-settings").addEventListener("click", saveSettingsDialog);
  $("preset-select").addEventListener("change", selectPreset); $("save-preset").addEventListener("click", savePreset); $("delete-preset").addEventListener("click", deletePreset);
  $("dialog-max-size").addEventListener("input", () => { if (state.settingsDraft) state.settingsDraft.max_size = numericFieldValue("dialog-max-size"); syncMethodControls("dialog-"); });
  $("settings-dialog").addEventListener("cancel", (event) => { event.preventDefault(); closeSettingsDialog(); });
  $("settings-dialog").addEventListener("close", () => { state.settingsDraft = null; updatePresetButtons(); });
  $("settings-dialog").addEventListener("click", (event) => {
    if (event.target !== event.currentTarget) return;
    const box = event.currentTarget.getBoundingClientRect();
    if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) closeSettingsDialog();
  });
  $("help-button").addEventListener("click", () => $("help-dialog").showModal()); $("close-help").addEventListener("click", () => $("help-dialog").close());
  document.addEventListener("keydown", (event) => {
    if (document.querySelector("dialog[open]")) return;
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "o") { event.preventDefault(); if (!state.busy && !state.initializing) $("file-input").click(); return; }
    if (event.target.closest("input, select, textarea, button, summary")) return;
    if (event.code === "Space") { state.space = true; event.preventDefault(); }
    if (event.key === "+" || event.key === "=") zoomBy(1.25); if (event.key === "-") zoomBy(0.8); if (event.key.toLowerCase() === "f") resetView();
    if (/^[1-5]$/.test(event.key)) setMode(viewModes[Number(event.key) - 1]);
  });
  document.addEventListener("keyup", (event) => { if (event.code === "Space") state.space = false; });
  window.addEventListener("blur", () => { state.space = false; state.drag = null; }); setupCanvas();
}

async function restoreSelection(uploadToken = state.uploadVersion) {
  let saved;
  try { saved = JSON.parse(localStorage.getItem("faceshield.selection.v1") || "null"); } catch { return; }
  if (!saved?.source || !saved?.uploadedSource) return;
  const valid = (asset) => asset && typeof asset.id === "string" && typeof asset.name === "string" && /^\/api\/image\/[a-zA-Z0-9_-]+$/.test(asset.url) && Number.isInteger(asset.width) && Number.isInteger(asset.height) && asset.width > 0 && asset.height > 0;
  if (!valid(saved.source) || !valid(saved.uploadedSource) || (saved.result && !valid(saved.result))) return;
  try {
    // Verify the full original still exists before offering another run.
    await loadImage(saved.uploadedSource);
    if (uploadToken !== state.uploadVersion) return;
    state.uploadedSource = saved.uploadedSource;
    if (!await useSource(saved.source, false, uploadToken)) return;
    if (saved.result) { await useResult(saved.result); setMode("split"); }
    state.stage1 = saved.stage1 && valid(saved.stage1) && state.assetIndex.has(saved.stage1.id) ? saved.stage1 : null;
    state.intermediateAssets = validIntermediateAssets(saved.intermediateAssets).filter(item => state.assetIndex.has(item.asset.id));
    updateCards(); persistSelection();
  } catch {
    if (uploadToken !== state.uploadVersion) return;
    state.uploadedSource = state.source = state.result = state.stage1 = state.sourceImage = state.resultImage = null;
    state.intermediateAssets = [];
    try { localStorage.removeItem("faceshield.selection.v1"); } catch { /* Optional. */ }
    toast("Previous photo is no longer available. Open it again."); updateCards();
  }
}

async function initialize() {
  bindEvents(); readPresets(); renderPresets();
  try { const saved = localStorage.getItem("faceshield.settings.v1"); applySettings(saved ? JSON.parse(saved) : defaults, Boolean(saved)); }
  catch { applySettings(defaults); toast("Could not read saved settings. Using defaults.", true); }
  updateCards();
  try {
    const session = await api("/api/state"); state.ready = session.runtime?.ready !== false;
    state.engines = session.engines || {};
    state.testing = session.testing || { ready: false, message: "This server does not provide face-swap tests." };
    state.testableResultIds = new Set(session.testable_result_ids || []);
    state.latestTestJobIds = new Map(Object.entries(session.latest_test_job_ids || {}));
    const targetKeys = ["cage", "wlh", "zxy"];
    if (Array.isArray(state.testing.targets)) {
      const targets = state.testing.targets.filter((target) => targetKeys.includes(target.key));
      const selected = $("test-target").value;
      $("test-target").replaceChildren(...targets.map((target) => new Option(`Demo face ${targetKeys.indexOf(target.key) + 1}`, target.key)));
      $("test-target").value = targets.some((target) => target.key === selected) ? selected : targets[0]?.key || "";
    }
    state.assetIndex = new Map((session.assets || []).map((asset) => [asset.id, asset]));
    await restoreSelection();
    const active = session.active_job;
    if (active) {
      const id = typeof active === "string" ? active : active.id;
      if (id) {
        const watching = resumeActiveJob(id, session.active_job_kind || active.kind);
        state.initializing = false; updateCards();
        await watching;
      }
    }
  } catch (error) { toast(error.message, true); }
  finally { state.initializing = false; updateCards(); requestDraw(); }
}

async function resumeActiveJob(id, kindHint = null) {
  // Look up kind before choosing a watcher; a test never adopts protection assets.
  busy(true, kindHint === "test" ? "test" : "protect");
  try {
    const job = await api(`/api/job/${encodeURIComponent(id)}`);
    if (job.kind === "test") {
      const resultId = job.result_id || state.result?.id;
      if (resultId) state.latestTestJobIds.set(resultId, id);
      return await watchTestJob(id, resultId, job);
    }
    return await watchJob(id, null, job);
  }
  finally { busy(false); }
}

initialize();
