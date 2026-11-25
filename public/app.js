const COLORS = {
  original_gwo: "#dc2626",
  nonlinear_a_gwo: "#2563eb",
  dual_criteria_gwo: "#d97706",
  hybrid_gwo: "#059669",
};

const state = {
  meta: null,
  activeView: "compare",
  lastRun: null,
  auxMode: "a_curve",
  suiteData: null,
  arenaData: null,
  arenaIter: 0,
  arenaTimer: null,
  codeCache: {},
  currentModalSource: "",
};

const $ = (id) => document.getElementById(id);

function getParams() {
  return {
    n_agents: Math.max(6, Math.min(50, Number($("nAgents").value) || 20)),
    max_iter: Math.max(10, Math.min(120, Number($("maxIter").value) || 40)),
    pop_weight: Math.max(0.05, Math.min(0.6, Number($("popWeight").value) || 0.25)),
    seed: Number($("seedInput").value) || 42,
  };
}

function themeColor(varName) {
  return getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
}

async function fetchJson(url, opts = {}) {
  const res = await fetch(url, opts);
  if (!res.ok) throw new Error(`Request failed (${res.status})`);
  return res.json();
}

// =========================================================================
// Theme & Navigation
// =========================================================================
function initNavigation() {
  document.querySelectorAll(".nav-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const target = btn.dataset.view;
      state.activeView = target;
      document.querySelectorAll(".nav-btn").forEach((b) => b.classList.toggle("active", b === btn));
      document.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${target}`));

      $("problemControlWrap").classList.toggle("hidden", target === "arena" || target === "suite" || target === "algorithms");
      $("benchmark2dWrap").classList.toggle("hidden", target !== "arena");
      $("runBtn").textContent =
        target === "suite" ? "Run 12 Problems" :
        target === "arena" ? "Run 2D Swarm" : "Run Comparison";

      if (target === "suite" && !state.suiteData) runSuite();
      if (target === "arena" && !state.arenaData) runArena2D();
    });
  });

  $("themeToggle").addEventListener("click", () => {
    const html = document.documentElement;
    const next = html.getAttribute("data-theme") === "dark" ? "light" : "dark";
    html.setAttribute("data-theme", next);
    $("themeLabel").textContent = next === "dark" ? "Light" : "Dark";
    if (state.lastRun) {
      drawLineChart($("convCanvas"), state.lastRun.results, "convergence_curve");
      drawAuxChart();
      renderDomainSolution();
    }
    if (state.arenaData) renderArenaFrame(state.arenaIter);
  });

  $("runBtn").addEventListener("click", () => {
    if (state.activeView === "suite") runSuite();
    else if (state.activeView === "arena") runArena2D();
    else runComparison();
  });

  $("runSuiteBtn").addEventListener("click", runSuite);
  $("exportCsvBtn").addEventListener("click", exportSuiteCsv);

  document.querySelectorAll("#auxChartToggle .seg-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.auxMode = btn.dataset.mode;
      document.querySelectorAll("#auxChartToggle .seg-btn").forEach((b) => b.classList.toggle("active", b === btn));
      drawAuxChart();
    });
  });

  $("solutionAlgoSelect").addEventListener("change", renderDomainSolution);
  $("problemSelect").addEventListener("change", runComparison);
  $("benchmark2dSelect").addEventListener("change", runArena2D);

  // Code Modal
  $("closeModalBtn").addEventListener("click", closeCodeModal);
  $("codeModal").addEventListener("click", (e) => {
    if (e.target === $("codeModal")) closeCodeModal();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeCodeModal();
  });
  $("copyCodeBtn").addEventListener("click", async () => {
    await navigator.clipboard.writeText(state.currentModalSource);
    $("copyCodeBtn").textContent = "Copied";
    setTimeout(() => ($("copyCodeBtn").textContent = "Copy"), 1200);
  });

  // 2D Arena Playback
  $("playPauseBtn").addEventListener("click", toggleArenaPlay);
  $("resetAnimBtn").addEventListener("click", () => {
    stopArenaPlay();
    state.arenaIter = 0;
    $("iterSlider").value = 0;
    renderArenaFrame(0);
  });
  $("iterSlider").addEventListener("input", (e) => {
    stopArenaPlay();
    state.arenaIter = Number(e.target.value);
    renderArenaFrame(state.arenaIter);
  });
}

// =========================================================================
// Show Code Modal (Shows ONLY the clicked algorithm's clean Python code)
// =========================================================================
async function openCodeModal(algoId) {
  if (!state.codeCache[algoId]) {
    state.codeCache[algoId] = await fetchJson(`/api/code/${algoId}`);
  }
  const data = state.codeCache[algoId];
  state.currentModalSource = data.source;
  $("modalTitle").textContent = `${data.code} — ${data.name}`;
  $("modalFile").textContent = data.filename;
  $("codeContent").textContent = data.source;
  $("codeModal").classList.remove("hidden");
}

function closeCodeModal() {
  $("codeModal").classList.add("hidden");
}

// =========================================================================
// Lightweight High-DPI Line Chart (Zero external chart bloat)
// =========================================================================
function drawLineChart(canvas, seriesList, fieldKey) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || 600;
  const h = 280;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);

  const pad = { top: 18, right: 18, bottom: 32, left: 58 };
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;

  ctx.clearRect(0, 0, w, h);

  const allVals = seriesList.flatMap((s) => s[fieldKey]);
  let minV = Math.min(...allVals);
  let maxV = Math.max(...allVals);
  if (Math.abs(maxV - minV) < 1e-9) {
    minV -= 1;
    maxV += 1;
  } else {
    const margin = (maxV - minV) * 0.08;
    minV -= margin;
    maxV += margin;
  }

  const gridColor = themeColor("--border");
  const textColor = themeColor("--text-muted");

  ctx.strokeStyle = gridColor;
  ctx.fillStyle = textColor;
  ctx.font = '11px "IBM Plex Mono", monospace';
  ctx.lineWidth = 1;

  for (let i = 0; i <= 4; i++) {
    const frac = i / 4;
    const y = pad.top + frac * plotH;
    const val = maxV - frac * (maxV - minV);
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(w - pad.right, y);
    ctx.stroke();

    const label = Math.abs(val) >= 100 ? val.toFixed(0) : Math.abs(val) >= 1 ? val.toFixed(2) : val.toFixed(4);
    ctx.textAlign = "right";
    ctx.fillText(label, pad.left - 8, y + 4);
  }

  const maxLen = Math.max(...seriesList.map((s) => s[fieldKey].length));
  ctx.textAlign = "center";
  [0, Math.floor((maxLen - 1) / 2), maxLen - 1].forEach((idx) => {
    const x = pad.left + (idx / Math.max(1, maxLen - 1)) * plotW;
    ctx.fillText(`t=${idx + 1}`, x, h - 10);
  });

  seriesList.forEach((s) => {
    const arr = s[fieldKey];
    ctx.strokeStyle = COLORS[s.id];
    ctx.lineWidth = s.id === "hybrid_gwo" ? 2.6 : 1.9;
    ctx.beginPath();
    arr.forEach((v, i) => {
      const x = pad.left + (i / Math.max(1, arr.length - 1)) * plotW;
      const y = pad.top + ((maxV - v) / (maxV - minV)) * plotH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });
}

function renderLegend(container, results) {
  container.innerHTML = results
    .map(
      (r) =>
        `<span class="legend-item"><span class="swatch" style="background:${COLORS[r.id]}"></span>${r.name}</span>`
    )
    .join("");
}

function drawAuxChart() {
  if (!state.lastRun) return;
  const field = state.auxMode === "a_curve" ? "a_curve" : "diversity_curve";
  drawLineChart($("auxCanvas"), state.lastRun.results, field);
  renderLegend($("auxLegend"), state.lastRun.results);
}

// =========================================================================
// View 1: Single Problem Comparison
// =========================================================================
async function runComparison() {
  const btn = $("runBtn");
  btn.disabled = true;
  try {
    const params = getParams();
    const data = await fetchJson("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ problem_id: $("problemSelect").value, ...params }),
    });
    state.lastRun = data;
    renderComparison(data);
  } finally {
    btn.disabled = false;
  }
}

function renderComparison(data) {
  const p = data.problem;
  $("probTitle").textContent = p.name;
  $("probDesc").textContent = p.description;
  $("probModeTag").textContent = p.mode === "max" ? "MAXIMIZE" : "MINIMIZE";
  $("probCatTag").textContent = p.category;
  $("probDimTag").textContent = `${p.dim}D`;
  $("convDirectionHint").textContent = p.mode === "max" ? "Higher is better" : "Lower is better";

  const winner = data.results.find((r) => r.is_winner);
  $("winnerSummary").textContent = `Best: ${winner.name} (${winner.best_fitness})`;

  $("algoCards").innerHTML = data.results
    .map((r) => {
      const deltaClass = r.delta_vs_original_pct > 0 ? "pos-delta" : r.delta_vs_original_pct < 0 ? "neg-delta" : "";
      const deltaText =
        r.id === "original_gwo"
          ? "Baseline"
          : `${r.delta_vs_original_pct >= 0 ? "+" : ""}${r.delta_vs_original_pct}% vs Orig`;
      return `
        <div class="algo-card ${r.is_winner ? "winner" : ""}">
          <div>
            <div class="algo-card-top">
              <span class="algo-badge">
                <span class="swatch" style="background:${COLORS[r.id]}"></span>${r.code}
              </span>
              ${r.is_winner ? '<span class="win-label">Best</span>' : ""}
            </div>
            <h3>${r.name}</h3>
            <div class="algo-meta">a: ${r.a_type} &middot; ${r.leader_type}</div>
          </div>

          <div>
            <div class="metric-val">${r.best_fitness}</div>
            <div class="metric-sub">
              <span class="${deltaClass}">${deltaText}</span>
              <span class="mono">t=${r.convergence_iter} &middot; ${r.runtime_ms}ms</span>
            </div>
          </div>

          <div class="algo-card-foot">
            <span class="muted-text small">${p.unit}</span>
            <button class="btn-secondary small-btn show-code-btn" data-algo="${r.id}">Show Code</button>
          </div>
        </div>
      `;
    })
    .join("");

  document.querySelectorAll(".show-code-btn").forEach((b) => {
    b.addEventListener("click", () => openCodeModal(b.dataset.algo));
  });

  drawLineChart($("convCanvas"), data.results, "convergence_curve");
  renderLegend($("convLegend"), data.results);
  drawAuxChart();

  // Solution algorithm dropdown
  const sel = $("solutionAlgoSelect");
  const prev = sel.value;
  sel.innerHTML = data.results.map((r) => `<option value="${r.id}">${r.name}</option>`).join("");
  sel.value = prev && data.results.some((r) => r.id === prev) ? prev : data.winner_id;
  renderDomainSolution();

  // Initial Population Audit (from Combined GWO or Dual-Criteria GWO)
  const auditSource = data.results.find((r) => r.id === "hybrid_gwo") || data.results.find((r) => r.id === "dual_criteria_gwo");
  if (auditSource && auditSource.initial_pop_audit.length) {
    $("initAnchorLabel").textContent = `Anchor P*: ${auditSource.init_best_pop_value}`;
    $("auditTable").querySelector("tbody").innerHTML = auditSource.initial_pop_audit
      .map((row) => {
        const cls =
          row.role === "Alpha" ? "role-alpha" :
          row.role === "Beta" ? "role-beta" :
          row.role === "Delta" ? "role-delta" : "role-omega";
        return `
          <tr>
            <td class="mono">${row.rank}</td>
            <td class="mono">${row.wolf}</td>
            <td><span class="role-pill ${cls}">${row.role}</span></td>
            <td class="mono">${row.pop_value}</td>
            <td class="mono">${row.fitness}</td>
            <td class="mono">${row.combined}</td>
          </tr>
        `;
      })
      .join("");
  }
}

function renderDomainSolution() {
  if (!state.lastRun) return;
  const aid = $("solutionAlgoSelect").value;
  const res = state.lastRun.results.find((r) => r.id === aid);
  if (!res) return;
  const dec = res.decoded;
  const box = $("domainVisualizer");

  if (dec.type === "knapsack") {
    box.innerHTML = `
      <div class="kv-grid">
        <div class="kv-item"><div class="kv-label">Packed Value</div><div class="kv-val">${dec.total_value}</div></div>
        <div class="kv-item"><div class="kv-label">Weight / Capacity</div><div class="kv-val">${dec.total_weight} / ${dec.capacity} kg</div></div>
        <div class="kv-item"><div class="kv-label">Items Picked</div><div class="kv-val">${dec.selected_count} / ${dec.items.length}</div></div>
        <div class="kv-item"><div class="kv-label">Constraint</div><div class="kv-val">${dec.feasible ? "Feasible" : "Exceeded"}</div></div>
      </div>
      <div class="knap-grid">
        ${dec.items
          .map(
            (it) =>
              `<div class="knap-cell ${it.picked ? "picked" : ""}">#${it.id}<br/>${it.value} / ${it.weight}kg</div>`
          )
          .join("")}
      </div>
    `;
  } else if (dec.type === "tsp") {
    box.innerHTML = `
      <div class="kv-grid" style="margin-bottom:0.75rem;">
        <div class="kv-item"><div class="kv-label">Tour Length</div><div class="kv-val">${dec.tour_length} km</div></div>
        <div class="kv-item"><div class="kv-label">Cities Visited</div><div class="kv-val">${dec.tour.length}</div></div>
      </div>
      <canvas id="tspCanvas" width="520" height="230" style="width:100%;border:1px solid var(--border);border-radius:7px;"></canvas>
    `;
    const c = $("tspCanvas");
    const ctx = c.getContext("2d");
    ctx.clearRect(0, 0, c.width, c.height);
    const pad = 24;
    const mapX = (x) => pad + (x / 100) * (c.width - 2 * pad);
    const mapY = (y) => c.height - pad - (y / 100) * (c.height - 2 * pad);

    ctx.strokeStyle = COLORS[aid];
    ctx.lineWidth = 2;
    ctx.beginPath();
    dec.tour.forEach((cityIdx, idx) => {
      const [cx, cy] = dec.coords[cityIdx];
      if (idx === 0) ctx.moveTo(mapX(cx), mapY(cy));
      else ctx.lineTo(mapX(cx), mapY(cy));
    });
    const [sx, sy] = dec.coords[dec.tour[0]];
    ctx.lineTo(mapX(sx), mapY(sy));
    ctx.stroke();

    ctx.fillStyle = themeColor("--text");
    dec.coords.forEach(([cx, cy]) => {
      ctx.beginPath();
      ctx.arc(mapX(cx), mapY(cy), 4.2, 0, Math.PI * 2);
      ctx.fill();
    });
  } else if (dec.type === "facility") {
    box.innerHTML = `
      <div class="kv-grid" style="margin-bottom:0.75rem;">
        <div class="kv-item"><div class="kv-label">Demand Covered</div><div class="kv-val">${dec.covered_pct}%</div></div>
        <div class="kv-item"><div class="kv-label">Service Radius</div><div class="kv-val">${dec.radius} km</div></div>
      </div>
      <canvas id="facCanvas" width="520" height="230" style="width:100%;border:1px solid var(--border);border-radius:7px;"></canvas>
    `;
    const c = $("facCanvas");
    const ctx = c.getContext("2d");
    const pad = 20;
    const mapX = (x) => pad + (x / 100) * (c.width - 2 * pad);
    const mapY = (y) => c.height - pad - (y / 100) * (c.height - 2 * pad);

    ctx.fillStyle = themeColor("--text-muted");
    dec.zones.forEach(([zx, zy], idx) => {
      ctx.beginPath();
      ctx.arc(mapX(zx), mapY(zy), 2.5 + dec.demands[idx] * 0.04, 0, Math.PI * 2);
      ctx.fill();
    });
    dec.hubs.forEach(([hx, hy]) => {
      ctx.strokeStyle = COLORS[aid];
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(mapX(hx), mapY(hy), (dec.radius / 100) * (c.height - 2 * pad), 0, Math.PI * 2);
      ctx.stroke();
      ctx.fillStyle = COLORS[aid];
      ctx.fillRect(mapX(hx) - 5, mapY(hy) - 5, 10, 10);
    });
  } else {
    const entries = Object.entries(dec).filter(([k]) => k !== "type");
    box.innerHTML = `<div class="kv-grid">${entries
      .map(([k, v]) => `<div class="kv-item"><div class="kv-label">${k}</div><div class="kv-val">${v}</div></div>`)
      .join("")}</div>`;
  }
}

// =========================================================================
// View 2: 12-Problem Benchmark Suite
// =========================================================================
async function runSuite() {
  const btn = $("runSuiteBtn");
  btn.disabled = true;
  $("runBtn").disabled = true;
  try {
    const params = getParams();
    const data = await fetchJson("/api/suite", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    });
    state.suiteData = data;
    renderSuite(data);
  } finally {
    btn.disabled = false;
    $("runBtn").disabled = false;
  }
}

function renderSuite(data) {
  $("suiteScoreboard").innerHTML = state.meta.algorithms
    .map((a) => {
      const wins = data.wins[a.id] || 0;
      const avgD = data.avg_delta_vs_original[a.id];
      const deltaTxt = a.id === "original_gwo" ? "Baseline" : `Avg ${avgD >= 0 ? "+" : ""}${avgD}% vs Orig`;
      return `
        <div class="score-tile">
          <div class="tile-title"><span class="swatch" style="background:${COLORS[a.id]}"></span>${a.name}</div>
          <div class="tile-wins">${wins} / ${data.total_problems} Wins</div>
          <div class="tile-delta mono">${deltaTxt}</div>
        </div>
      `;
    })
    .join("");

  $("suiteTable").querySelector("tbody").innerHTML = data.rows
    .map(
      (r) => `
      <tr>
        <td><strong>${r.problem}</strong><br/><span class="muted-text small">${r.category}</span></td>
        <td><span class="tag ${r.mode === "min" ? "muted" : ""}">${r.mode === "max" ? "MAX" : "MIN"}</span></td>
        <td class="mono">${r.dim}D</td>
        <td class="mono">${r.scores.original_gwo}</td>
        <td class="mono">${r.scores.nonlinear_a_gwo}</td>
        <td class="mono">${r.scores.dual_criteria_gwo}</td>
        <td class="mono">${r.scores.hybrid_gwo}</td>
        <td><span class="win-label">${r.winner_name}</span></td>
      </tr>
    `
    )
    .join("");
}

function exportSuiteCsv() {
  if (!state.suiteData) return;
  const header = ["Problem", "Category", "Mode", "Dim", "Original GWO", "Non-Linear a GWO", "Dual-Criteria GWO", "Combined GWO", "Winner"];
  const lines = [header.join(",")];
  state.suiteData.rows.forEach((r) => {
    lines.push([
      `"${r.problem}"`, r.category, r.mode, r.dim,
      r.scores.original_gwo, r.scores.nonlinear_a_gwo, r.scores.dual_criteria_gwo, r.scores.hybrid_gwo,
      `"${r.winner_name}"`,
    ].join(","));
  });
  const blob = new Blob([lines.join("\n")], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "gwo_12_problem_benchmark.csv";
  a.click();
  URL.revokeObjectURL(url);
}

// =========================================================================
// View 3: 2D Live Swarm Visualizer (HTML5 Canvas)
// =========================================================================
async function runArena2D() {
  stopArenaPlay();
  $("runBtn").disabled = true;
  try {
    const params = getParams();
    const data = await fetchJson("/api/arena2d", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        benchmark_id: $("benchmark2dSelect").value,
        n_agents: params.n_agents,
        max_iter: Math.min(params.max_iter, 70),
        pop_weight: params.pop_weight,
        seed: params.seed,
      }),
    });
    state.arenaData = data;
    state.arenaIter = 0;
    $("iterSlider").max = data.max_iter - 1;
    $("iterSlider").value = 0;

    $("arenaGrid").innerHTML = data.runs
      .map(
        (r) => `
        <div class="arena-card">
          <div class="arena-card-head">
            <span class="algo-badge"><span class="swatch" style="background:${COLORS[r.id]}"></span><strong>${r.name}</strong></span>
            <span class="mono small" id="arena-fit-${r.id}">f = ${r.convergence_curve[0]}</span>
          </div>
          <canvas id="arena-canvas-${r.id}" width="440" height="280"></canvas>
        </div>
      `
      )
      .join("");

    renderArenaFrame(0);
  } finally {
    $("runBtn").disabled = false;
  }
}

function renderArenaFrame(iter) {
  if (!state.arenaData) return;
  const bm = state.arenaData.benchmark;
  const grid = bm.grid;
  const rows = grid.length;
  const cols = grid[0].length;
  $("iterReadout").textContent = `Iter ${iter + 1} / ${state.arenaData.max_iter}`;

  const isDark = document.documentElement.getAttribute("data-theme") === "dark";

  state.arenaData.runs.forEach((run) => {
    const c = $(`arena-canvas-${run.id}`);
    if (!c) return;
    const ctx = c.getContext("2d");
    const w = c.width;
    const h = c.height;
    const cellW = w / cols;
    const cellH = h / rows;

    // Subtle grayscale/slate topographic contour shading
    const span = Math.max(1e-6, bm.z_max - bm.z_min);
    for (let r = 0; r < rows; r++) {
      for (let col = 0; col < cols; col++) {
        const norm = Math.sqrt((grid[r][col] - bm.z_min) / span);
        const shade = isDark ? Math.round(18 + norm * 52) : Math.round(245 - norm * 65);
        ctx.fillStyle = `rgb(${shade}, ${shade + 4}, ${shade + 10})`;
        ctx.fillRect(col * cellW, (rows - 1 - r) * cellH, cellW + 0.6, cellH + 0.6);
      }
    }

    const toX = (x) => ((x - bm.lb) / (bm.ub - bm.lb)) * w;
    const toY = (y) => h - ((y - bm.lb) / (bm.ub - bm.lb)) * h;

    // Global Optimum Crosshair
    const ox = toX(bm.optimum[0]);
    const oy = toY(bm.optimum[1]);
    ctx.strokeStyle = isDark ? "#facc15" : "#111827";
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.moveTo(ox - 7, oy); ctx.lineTo(ox + 7, oy);
    ctx.moveTo(ox, oy - 7); ctx.lineTo(ox, oy + 7);
    ctx.stroke();

    // Wolf Pack Agents
    const pts = run.trajectories[iter] || [];
    ctx.fillStyle = COLORS[run.id];
    pts.forEach(([px, py]) => {
      ctx.beginPath();
      ctx.arc(toX(px), toY(py), 4, 0, Math.PI * 2);
      ctx.fill();
    });

    // Alpha Leader Diamond
    const alpha = (run.leaders[iter] || [])[0];
    if (alpha) {
      const ax = toX(alpha[0]);
      const ay = toY(alpha[1]);
      ctx.fillStyle = "#ffffff";
      ctx.strokeStyle = COLORS[run.id];
      ctx.lineWidth = 2.4;
      ctx.beginPath();
      ctx.moveTo(ax, ay - 7);
      ctx.lineTo(ax + 7, ay);
      ctx.lineTo(ax, ay + 7);
      ctx.lineTo(ax - 7, ay);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
    }

    const fitEl = $(`arena-fit-${run.id}`);
    if (fitEl) fitEl.textContent = `f = ${run.convergence_curve[iter]} · a = ${run.a_curve[iter]}`;
  });
}

function toggleArenaPlay() {
  if (!state.arenaData) return;
  if (state.arenaTimer) {
    stopArenaPlay();
    return;
  }
  $("playPauseBtn").textContent = "Pause";
  state.arenaTimer = setInterval(() => {
    state.arenaIter = (state.arenaIter + 1) % state.arenaData.max_iter;
    $("iterSlider").value = state.arenaIter;
    renderArenaFrame(state.arenaIter);
    if (state.arenaIter === state.arenaData.max_iter - 1) stopArenaPlay();
  }, 160);
}

function stopArenaPlay() {
  if (state.arenaTimer) {
    clearInterval(state.arenaTimer);
    state.arenaTimer = null;
  }
  $("playPauseBtn").textContent = "Play";
}

// =========================================================================
// View 4: Algorithms & Code Specs
// =========================================================================
function renderAlgorithmSpecs() {
  $("algoSpecGrid").innerHTML = state.meta.algorithms
    .map(
      (a) => `
      <div class="spec-card">
        <div>
          <span class="algo-badge"><span class="swatch" style="background:${COLORS[a.id]}"></span>${a.code}</span>
          <h3>${a.name}</h3>
          <p>${a.summary}</p>
        </div>
        <div class="spec-pills">
          <span class="tag">Convergence a(t): ${a.a_type}</span>
          <span class="tag muted">Leaders: ${a.leader_type}</span>
        </div>
        <div class="spec-footer">
          <span class="mono small muted-text">Python implementation</span>
          <button class="btn-primary small-btn spec-code-btn" data-algo="${a.id}">Show Code</button>
        </div>
      </div>
    `
    )
    .join("");

  document.querySelectorAll(".spec-code-btn").forEach((b) => {
    b.addEventListener("click", () => openCodeModal(b.dataset.algo));
  });
}

// =========================================================================
// Bootstrap
// =========================================================================
async function bootstrap() {
  initNavigation();
  state.meta = await fetchJson("/api/meta");

  $("problemSelect").innerHTML = state.meta.problems
    .map((p) => `<option value="${p.id}">[${p.mode.toUpperCase()}] ${p.name}</option>`)
    .join("");

  $("benchmark2dSelect").innerHTML = state.meta.benchmarks_2d
    .map((b) => `<option value="${b.id}">${b.name} (2D)</option>`)
    .join("");

  renderAlgorithmSpecs();
  await runComparison();
}

bootstrap();
