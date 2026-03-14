// ── Socket.IO connection ──────────────────────────────────────────────────────
const socket = io();

// ── Canvas setup ─────────────────────────────────────────────────────────────
const heatCanvas  = document.getElementById("heatmap");
const heatCtx     = heatCanvas.getContext("2d");
const cbCanvas    = document.getElementById("colorbar");
const cbCtx       = cbCanvas.getContext("2d");
const W = heatCanvas.width, H = heatCanvas.height;

// ── Iron palette (matches PyGamer) ───────────────────────────────────────────
function mapRange(x, inMin, inMax, outMin, outMax) {
  const inRange = inMax - inMin;
  const inDelta = x - inMin;
  let mapped = inRange !== 0 ? inDelta / inRange : (inDelta !== 0 ? inDelta : 0.5);
  mapped = mapped * (outMax - outMin) + outMin;
  if (outMin <= outMax) return Math.max(Math.min(mapped, outMax), outMin);
  return Math.min(Math.max(mapped, outMax), outMin);
}

function ironColor(index) {
  index = Math.max(0, Math.min(1, index));
  const gamma = 0.5;
  const band = index * 600;
  let red, grn, blu;
  if (band < 70) {
    red = 0.1; grn = 0.1;
    blu = Math.pow(0.2 + 0.8 * mapRange(band, 0, 70, 0, 1), gamma);
  } else if (band < 200) {
    red = Math.pow(mapRange(band, 70, 200, 0, 0.6), gamma); grn = 0;
    blu = Math.pow(1.0, gamma);
  } else if (band < 300) {
    red = Math.pow(mapRange(band, 200, 300, 0.6, 1.0), gamma); grn = 0;
    blu = Math.pow(mapRange(band, 200, 300, 1.0, 0.0), gamma);
  } else if (band < 400) {
    red = Math.pow(1, gamma);
    grn = Math.pow(mapRange(band, 300, 400, 0.0, 0.5), gamma); blu = 0;
  } else if (band < 500) {
    red = Math.pow(1, gamma);
    grn = Math.pow(mapRange(band, 400, 500, 0.5, 1.0), gamma); blu = 0;
  } else {
    red = Math.pow(1, gamma); grn = Math.pow(1, gamma);
    blu = Math.pow(mapRange(band, 500, 580, 0.0, 1.0), gamma);
  }
  return [Math.round(red*255), Math.round(grn*255), Math.round(blu*255)];
}

// Pre-build LUT
const IRON_LUT = Array.from({length:256}, (_, i) => ironColor(i/255));

// ── Bilinear interpolation (8×8 → W×H) ───────────────────────────────────────
function renderHeatmap(grid8, minT, maxT) {
  const imgData = heatCtx.createImageData(W, H);
  const data    = imgData.data;
  const range   = (maxT - minT) || 1;

  for (let py = 0; py < H; py++) {
    for (let px = 0; px < W; px++) {
      // Grid coordinate in 0–7 space
      const gx = px * 7 / (W - 1);
      const gy = py * 7 / (H - 1);
      const x0 = Math.floor(gx), x1 = Math.min(x0 + 1, 7);
      const y0 = Math.floor(gy), y1 = Math.min(y0 + 1, 7);
      const fx = gx - x0, fy = gy - y0;

      const val = grid8[y0][x0] * (1-fx)*(1-fy) +
                  grid8[y0][x1] *    fx *(1-fy) +
                  grid8[y1][x0] * (1-fx)*   fy  +
                  grid8[y1][x1] *    fx *   fy;

      const norm  = Math.max(0, Math.min(1, (val - minT) / range));
      const idx   = Math.round(norm * 255);
      const [r,g,b] = IRON_LUT[idx];
      const off   = (py * W + px) * 4;
      data[off]   = r; data[off+1] = g; data[off+2] = b; data[off+3] = 255;
    }
  }
  heatCtx.putImageData(imgData, 0, 0);
}

// ── Draw colorbar ─────────────────────────────────────────────────────────────
function drawColorbar() {
  const h = cbCanvas.height;
  for (let y = 0; y < h; y++) {
    const t = 1 - y / (h - 1);
    const [r,g,b] = ironColor(t);
    cbCtx.fillStyle = `rgb(${r},${g},${b})`;
    cbCtx.fillRect(0, y, cbCanvas.width, 1);
  }
}
drawColorbar();

// ── FPS counter ───────────────────────────────────────────────────────────────
let frameTimes = [];
function updateFPS() {
  const now = Date.now();
  frameTimes = frameTimes.filter(t => now - t < 2000);
  frameTimes.push(now);
  const fps = frameTimes.length > 1 ? (frameTimes.length / ((frameTimes[frameTimes.length-1] - frameTimes[0]) / 1000)).toFixed(1) : "--";
  document.getElementById("fps-label").textContent = fps + " fps";
}

// ── Thermal frame handler ─────────────────────────────────────────────────────
socket.on("thermal_frame", (data) => {
  updateFPS();
  const grid = data.grid;
  const maxC = data.max_c, minC = data.min_c, avgC = data.avg_c;

  renderHeatmap(grid, minC, maxC);

  const toF = c => (c * 9/5 + 32).toFixed(1);
  document.getElementById("temp-max").textContent = toF(maxC) + "°F";
  document.getElementById("temp-avg").textContent = toF(avgC) + "°F";
  document.getElementById("temp-min").textContent = toF(minC) + "°F";
  document.getElementById("cb-max").textContent   = toF(maxC) + "°F";
  document.getElementById("cb-mid").textContent   = toF((maxC+minC)/2) + "°F";
  document.getElementById("cb-min").textContent   = toF(minC) + "°F";

  // ML prediction
  const pred = data.prediction;
  if (pred) {
    const badge = document.getElementById("ml-badge");
    const conf  = document.getElementById("ml-conf");
    badge.textContent = pred.label.toUpperCase();
    conf.textContent  = pred.label !== "N/A" ? Math.round(pred.confidence * 100) + "%" : "--";
    badge.className   = "ml-badge " + (pred.label === "healthy" ? "" : pred.label === "N/A" ? "na" : "unhealthy");
  }
});

// ── Vibration Chart (Chart.js) ────────────────────────────────────────────────
const MAX_POINTS = 80;
const labels = Array(MAX_POINTS).fill("");

const vibChart = new Chart(document.getElementById("vib-chart"), {
  type: "line",
  data: {
    labels: labels,
    datasets: [
      { label:"X",   data: [], borderColor:"#f87171", borderWidth:1.5, pointRadius:0, tension:0.3, fill:false },
      { label:"Y",   data: [], borderColor:"#4ade80", borderWidth:1.5, pointRadius:0, tension:0.3, fill:false },
      { label:"Z",   data: [], borderColor:"#60a5fa", borderWidth:1.5, pointRadius:0, tension:0.3, fill:false },
      { label:"RMS", data: [], borderColor:"#facc15", borderWidth:2,   pointRadius:0, tension:0.3, fill:false },
    ]
  },
  options: {
    animation: false,
    responsive: true,
    plugins: { legend: { display: false } },
    scales: {
      x: { display: false },
      y: {
        min: -3, max: 3,
        ticks: { color:"#6b6b8d", font:{ size:10 } },
        grid:  { color:"#1a1a2e" }
      }
    }
  }
});

function pushVibData(ax, ay, az, rms) {
  [ax, ay, az, rms].forEach((v, di) => {
    const ds = vibChart.data.datasets[di];
    ds.data.push(v);
    if (ds.data.length > MAX_POINTS) ds.data.shift();
  });
  vibChart.update("none");
}

socket.on("vibration", (d) => {
  document.getElementById("vib-rms").textContent  = d.rms_g.toFixed(3);
  document.getElementById("vib-peak").textContent = d.peak_g.toFixed(3);

  const badge = document.getElementById("vib-status");
  badge.textContent = d.status || "--";
  badge.className   = "vib-badge " + (d.status === "WARNING" ? "warning" : d.status === "HIGH" ? "high" : "");

  if (d.available) {
    pushVibData(d.ax, d.ay, d.az, d.rms_g);
  }
});

// ── Status dots ───────────────────────────────────────────────────────────────
socket.on("status", (s) => {
  setDot("mpu-dot",   s.mpu_available,  "mpu-label",   "MPU6050");
  setDot("model-dot", s.model_loaded,   "model-label", "Model");
});

socket.on("connect",    () => setDot("serial-dot", true,  "serial-label", "Serial"));
socket.on("disconnect", () => setDot("serial-dot", false, "serial-label", "Serial"));

function setDot(dotId, online, labelId, name) {
  const dot = document.getElementById(dotId);
  dot.className = "dot " + (online ? "online" : "offline");
}

// ── Alarm slider ──────────────────────────────────────────────────────────────
document.getElementById("alarm-slider").addEventListener("input", function() {
  document.getElementById("alarm-val").textContent = this.value;
});

// ── Data collection ───────────────────────────────────────────────────────────
document.getElementById("btn-start").addEventListener("click", async () => {
  const label = document.getElementById("collect-label").value.trim();
  if (!label) { alert("Enter a label first"); return; }
  const res = await fetch("/collect/start", {
    method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify({label})
  });
  const data = await res.json();
  document.getElementById("collect-status").textContent = `Collecting: ${data.label} → ${data.file}`;
  document.getElementById("btn-start").disabled = true;
  document.getElementById("btn-stop").disabled  = false;
});

document.getElementById("btn-stop").addEventListener("click", async () => {
  await fetch("/collect/stop", {method:"POST"});
  document.getElementById("collect-status").textContent = "Stopped.";
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
  await fetch("/models/load", {
    method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify({name})
  });
  setDot("model-dot", true, "model-label", "Model");
});

// Init
refreshModels();
