// ── Socket.IO connection ──────────────────────────────────────────────────────
const socket = io();

// ── Canvas setup ─────────────────────────────────────────────────────────────
const heatCanvas  = document.getElementById("heatmap");
const heatCtx     = heatCanvas.getContext("2d");
const cbCanvas    = document.getElementById("colorbar");
const cbCtx       = cbCanvas.getContext("2d");
const W = heatCanvas.width, H = heatCanvas.height;

// ── State ─────────────────────────────────────────────────────────────────────
let held       = false;        // H — freeze frame
let focusMode  = false;        // F — auto-range to current min/max
let paletteIdx = 0;            // P — cycle palettes
let useCelsius = true;         // unit toggle
let focusMin   = null, focusMax = null;  // focus range
const PALETTES = ["iron", "gray", "inferno"];

// ── Palette: Iron ─────────────────────────────────────────────────────────────
function mapRange(x, inMin, inMax, outMin, outMax) {
  const r = inMax - inMin;
  let m = r !== 0 ? (x - inMin) / r : (x - inMin !== 0 ? x - inMin : 0.5);
  m = m * (outMax - outMin) + outMin;
  return outMin <= outMax ? Math.max(Math.min(m, outMax), outMin)
                           : Math.min(Math.max(m, outMax), outMin);
}

function ironColor(index) {
  index = Math.max(0, Math.min(1, index));
  const g = 0.5, band = index * 600;
  let r, gr, b;
  if (band < 70)       { r = 0.1; gr = 0.1; b = Math.pow(0.2 + 0.8 * mapRange(band, 0, 70, 0, 1), g); }
  else if (band < 200) { r = Math.pow(mapRange(band, 70, 200, 0, 0.6), g); gr = 0; b = 1; }
  else if (band < 300) { r = Math.pow(mapRange(band, 200, 300, 0.6, 1.0), g); gr = 0; b = Math.pow(mapRange(band, 200, 300, 1, 0), g); }
  else if (band < 400) { r = 1; gr = Math.pow(mapRange(band, 300, 400, 0, 0.5), g); b = 0; }
  else if (band < 500) { r = 1; gr = Math.pow(mapRange(band, 400, 500, 0.5, 1), g); b = 0; }
  else                 { r = 1; gr = 1; b = Math.pow(mapRange(band, 500, 580, 0, 1), g); }
  return [r * 255, gr * 255, b * 255].map(Math.round);
}

function grayColor(t) { const v = Math.round(t * 255); return [v, v, v]; }

function infernoColor(t) {
  t = Math.max(0, Math.min(1, t));
  // Simple inferno approximation
  const r = Math.round(255 * Math.min(1, t < 0.5 ? 2*t*t : -1 + (4 - 2*t)*t));
  const g = Math.round(255 * Math.min(1, t < 0.4 ? 0 : (t - 0.4) * 2.5));
  const b = Math.round(255 * Math.min(1, t < 0.3 ? t * 2 : Math.max(0, 0.6 - (t - 0.3) * 2)));
  return [r, g, b];
}

// Pre-build LUT for all palettes
function buildLUT(palette) {
  return Array.from({length: 256}, (_, i) => {
    const t = i / 255;
    if (palette === "iron")    return ironColor(t);
    if (palette === "gray")    return grayColor(t);
    if (palette === "inferno") return infernoColor(t);
    return ironColor(t);
  });
}
let LUT = buildLUT("iron");

// ── Bilinear interpolation (8×8 → W×H) ───────────────────────────────────────
function renderHeatmap(grid8, minT, maxT) {
  const imgData = heatCtx.createImageData(W, H);
  const data    = imgData.data;
  const range   = (maxT - minT) || 1;
  for (let py = 0; py < H; py++) {
    for (let px = 0; px < W; px++) {
      const gx = px * 7 / (W - 1), gy = py * 7 / (H - 1);
      const x0 = Math.floor(gx), x1 = Math.min(x0 + 1, 7);
      const y0 = Math.floor(gy), y1 = Math.min(y0 + 1, 7);
      const fx = gx - x0, fy = gy - y0;
      const val = grid8[y0][x0]*(1-fx)*(1-fy) + grid8[y0][x1]*fx*(1-fy)
                + grid8[y1][x0]*(1-fx)*fy     + grid8[y1][x1]*fx*fy;
      const norm  = Math.max(0, Math.min(1, (val - minT) / range));
      const [r, g, b] = LUT[Math.round(norm * 255)];
      const off = (py * W + px) * 4;
      data[off] = r; data[off+1] = g; data[off+2] = b; data[off+3] = 255;
    }
  }
  heatCtx.putImageData(imgData, 0, 0);

  // Draw crosshair at centre
  heatCtx.strokeStyle = "rgba(255,255,255,0.6)";
  heatCtx.lineWidth   = 1;
  const cx = W/2, cy = H/2, cs = 12;
  heatCtx.beginPath(); heatCtx.moveTo(cx - cs, cy); heatCtx.lineTo(cx + cs, cy); heatCtx.stroke();
  heatCtx.beginPath(); heatCtx.moveTo(cx, cy - cs); heatCtx.lineTo(cx, cy + cs); heatCtx.stroke();
  heatCtx.beginPath(); heatCtx.arc(cx, cy, cs * 0.6, 0, Math.PI * 2); heatCtx.stroke();
}

// ── Colourbar ─────────────────────────────────────────────────────────────────
function drawColorbar() {
  const h = cbCanvas.height;
  for (let y = 0; y < h; y++) {
    const t = 1 - y / (h - 1);
    const [r, g, b] = LUT[Math.round(t * 255)];
    cbCtx.fillStyle = `rgb(${r},${g},${b})`;
    cbCtx.fillRect(0, y, cbCanvas.width, 1);
  }
}
drawColorbar();

// ── FPS ───────────────────────────────────────────────────────────────────────
let frameTimes = [];
function updateFPS() {
  const now = Date.now();
  frameTimes = frameTimes.filter(t => now - t < 2000);
  frameTimes.push(now);
  const fps = frameTimes.length > 1
    ? (frameTimes.length / ((frameTimes[frameTimes.length-1] - frameTimes[0]) / 1000)).toFixed(1)
    : "--";
  document.getElementById("fps-label").textContent = fps + " fps";
}

// ── Temperature formatting ────────────────────────────────────────────────────
function fmtTemp(c) {
  if (useCelsius) return c.toFixed(1) + "°C";
  return (c * 9/5 + 32).toFixed(1) + "°F";
}

// ── Thermal frame handler ─────────────────────────────────────────────────────
let lastFrame = null;

socket.on("thermal_frame", (data) => {
  lastFrame = data;
  if (!held) applyFrame(data);
});

function applyFrame(data) {
  updateFPS();
  const grid = data.grid;
  let minT = data.min_c, maxT = data.max_c;
  if (focusMode && focusMin !== null) { minT = focusMin; maxT = focusMax; }

  renderHeatmap(grid, minT, maxT);

  document.getElementById("temp-max").textContent = fmtTemp(data.max_c);
  document.getElementById("temp-avg").textContent = fmtTemp(data.avg_c);
  document.getElementById("temp-min").textContent = fmtTemp(data.min_c);
  document.getElementById("cb-max").textContent   = fmtTemp(maxT);
  document.getElementById("cb-mid").textContent   = fmtTemp((maxT + minT) / 2);
  document.getElementById("cb-min").textContent   = fmtTemp(minT);

  const pred = data.prediction;
  if (pred) {
    const badge = document.getElementById("ml-badge");
    const conf  = document.getElementById("ml-conf");
    badge.textContent = pred.label.toUpperCase();
    conf.textContent  = pred.label !== "N/A" ? Math.round(pred.confidence * 100) + "%" : "--";
    badge.className   = "ml-badge " + (pred.label === "healthy" ? "" : pred.label === "N/A" ? "na" : "unhealthy");
  }
}

// ── Camera controls ───────────────────────────────────────────────────────────
document.addEventListener("keydown", (e) => {
  const k = e.key.toUpperCase();
  if (k === "H") toggleHold();
  if (k === "F") toggleFocus();
  if (k === "P") cyclePalette();
});

document.getElementById("btn-hold").addEventListener("click", toggleHold);
document.getElementById("btn-focus").addEventListener("click", toggleFocus);
document.getElementById("btn-palette").addEventListener("click", cyclePalette);

function toggleHold() {
  held = !held;
  const btn = document.getElementById("btn-hold");
  btn.classList.toggle("active", held);
  btn.textContent = held ? "▶ Live" : "⏸ Hold";
  if (!held && lastFrame) applyFrame(lastFrame);
}

function toggleFocus() {
  focusMode = !focusMode;
  if (focusMode && lastFrame) {
    focusMin = lastFrame.min_c;
    focusMax = lastFrame.max_c;
  }
  const btn = document.getElementById("btn-focus");
  btn.classList.toggle("active", focusMode);
}

function cyclePalette() {
  paletteIdx = (paletteIdx + 1) % PALETTES.length;
  LUT = buildLUT(PALETTES[paletteIdx]);
  drawColorbar();
  const names = ["Iron", "Gray", "Inferno"];
  document.getElementById("btn-palette").textContent = "🎨 " + names[paletteIdx];
  if (lastFrame && !held) applyFrame(lastFrame);
}

// ── Unit toggle ───────────────────────────────────────────────────────────────
document.getElementById("btn-unit").addEventListener("click", () => {
  useCelsius = !useCelsius;
  const btn = document.getElementById("btn-unit");
  btn.textContent = useCelsius ? "°C" : "°F";
  btn.classList.toggle("active", useCelsius);
  if (lastFrame) applyFrame(lastFrame);
});

// ── Alarm slider ──────────────────────────────────────────────────────────────
document.getElementById("alarm-slider").addEventListener("input", function() {
  document.getElementById("alarm-val").textContent = this.value;
});

// ── Vibration Chart ───────────────────────────────────────────────────────────
const MAX_POINTS = 80;
const vibChart = new Chart(document.getElementById("vib-chart"), {
  type: "line",
  data: {
    labels: Array(MAX_POINTS).fill(""),
    datasets: [
      { label:"X",   data:[], borderColor:"#f87171", borderWidth:1.5, pointRadius:0, tension:0.3, fill:false },
      { label:"Y",   data:[], borderColor:"#4ade80", borderWidth:1.5, pointRadius:0, tension:0.3, fill:false },
      { label:"Z",   data:[], borderColor:"#60a5fa", borderWidth:1.5, pointRadius:0, tension:0.3, fill:false },
      { label:"RMS", data:[], borderColor:"#facc15", borderWidth:2,   pointRadius:0, tension:0.3, fill:false },
    ]
  },
  options: {
    animation: false, responsive: true,
    plugins: { legend: { display: false } },
    scales: {
      x: { display: false },
      y: { min: -3, max: 3,
           ticks: { color:"#6b6b8d", font:{ size:10 } },
           grid:  { color:"#1a1a2e" } }
    }
  }
});

socket.on("vibration", (d) => {
  document.getElementById("vib-rms").textContent  = d.rms_g.toFixed(3);
  document.getElementById("vib-peak").textContent = d.peak_g.toFixed(3);
  const badge = document.getElementById("vib-status");
  badge.textContent = d.status || "--";
  badge.className   = "vib-badge " + (d.status === "WARNING" ? "warning" : d.status === "HIGH" ? "high" : "");
  if (d.available) {
    [d.ax, d.ay, d.az, d.rms_g].forEach((v, i) => {
      const ds = vibChart.data.datasets[i];
      ds.data.push(v);
      if (ds.data.length > MAX_POINTS) ds.data.shift();
    });
    vibChart.update("none");
  }
});

// ── Status dots ───────────────────────────────────────────────────────────────
socket.on("status", (s) => {
  setDot("mpu-dot",   s.mpu_available,  "mpu-label",   "MPU6050");
  setDot("model-dot", s.model_loaded,   "model-label", "Model");
});
socket.on("connect",    () => setDot("serial-dot", true,  "serial-label", "Serial"));
socket.on("disconnect", () => setDot("serial-dot", false, "serial-label", "Serial"));

function setDot(dotId, online) {
  document.getElementById(dotId).className = "dot " + (online ? "online" : "offline");
}

// ── Data collection ───────────────────────────────────────────────────────────
document.getElementById("btn-start").addEventListener("click", async () => {
  const label = document.getElementById("collect-label").value.trim();
  if (!label) { alert("Enter a label first"); return; }
  const res  = await fetch("/collect/start", { method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({label}) });
  const data = await res.json();
  document.getElementById("collect-status").textContent = `● Collecting: ${data.label}`;
  document.getElementById("btn-start").disabled = true;
  document.getElementById("btn-stop").disabled  = false;
});

document.getElementById("btn-stop").addEventListener("click", async () => {
  await fetch("/collect/stop", {method:"POST"});
  document.getElementById("collect-status").textContent = "✓ Stopped.";
  document.getElementById("btn-start").disabled = false;
  document.getElementById("btn-stop").disabled  = true;
  refreshModels();
});

// ── Train ─────────────────────────────────────────────────────────────────────
document.getElementById("btn-train").addEventListener("click", async () => {
  const btn = document.getElementById("btn-train");
  btn.textContent = "⏳ Training..."; btn.disabled = true;
  const out = document.getElementById("train-output");
  out.style.display = "block"; out.textContent = "Training started...\n";
  await fetch("/train", {method:"POST"});
});

socket.on("train_complete", (d) => {
  const btn = document.getElementById("btn-train");
  btn.textContent = "🧠 Train Model"; btn.disabled = false;
  const out = document.getElementById("train-output");
  out.textContent = (d.success ? "✅ Done!\n" : "❌ Error:\n") + d.output;
  if (d.success) refreshModels();
});

// ── Load model ────────────────────────────────────────────────────────────────
async function refreshModels() {
  const res  = await fetch("/models");
  const data = await res.json();
  const sel  = document.getElementById("model-select");
  sel.innerHTML = '<option value="">-- Select model --</option>';
  data.models.forEach(m => {
    const opt = document.createElement("option"); opt.value = m; opt.textContent = m; sel.appendChild(opt);
  });
}

document.getElementById("btn-load-model").addEventListener("click", async () => {
  const name = document.getElementById("model-select").value;
  if (!name) return;
  await fetch("/models/load", { method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({name}) });
  setDot("model-dot", true);
});

// ── Notes panel ───────────────────────────────────────────────────────────────
const notesLog = document.getElementById("notes-log");

document.getElementById("btn-notes-save").addEventListener("click", () => {
  const txt = document.getElementById("notes-input").value.trim();
  if (!txt) return;
  const ts = new Date().toLocaleTimeString();

  // Build a snapshot of current readings
  const rms  = document.getElementById("vib-rms").textContent;
  const peak = document.getElementById("vib-peak").textContent;
  const stat = document.getElementById("vib-status").textContent;
  const max  = document.getElementById("temp-max").textContent;
  const pred = document.getElementById("ml-badge").textContent;
  const conf = document.getElementById("ml-conf").textContent;

  const entry = document.createElement("div");
  entry.className = "note-entry";
  entry.innerHTML = `
    <div class="note-ts">${ts}</div>
    <div class="note-snap">🌡 ${max} &nbsp; 📳 RMS:${rms}g Peak:${peak}g [${stat}] &nbsp; 🤖 ${pred} ${conf}</div>
    <div class="note-text">${txt}</div>`;
  notesLog.prepend(entry);

  document.getElementById("notes-input").value = "";
});

document.getElementById("btn-notes-clear").addEventListener("click", () => {
  notesLog.innerHTML = "";
});

// Init
refreshModels();
