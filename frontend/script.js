/**
 * script.js — House Price Predictor Pro
 * ======================================
 * Handles:
 *  - Flask API communication (/predict, /metrics, /features)
 *  - Form validation + sample data
 *  - Result card rendering with animations
 *  - Model comparison table + bar chart (Chart.js)
 *  - Feature importance chart
 *  - Toast notifications
 *  - Property showcase card clicks
 */

"use strict";

// ── Config ────────────────────────────────────────────────────────
const API_BASE = "http://localhost:5000";
let featureChartInstance = null;
let modelBarChartInstance = null;

// ── Utility: Toast ────────────────────────────────────────────────
function showToast(message, type = "info", duration = 3500) {
  const toast = document.getElementById("toast");
  toast.className = `toast show fixed bottom-6 right-6 z-50 px-6 py-4 rounded-2xl text-sm font-medium shadow-2xl toast-${type}`;
  toast.textContent = message;
  setTimeout(() => {
    toast.className = toast.className.replace("show", "").trim() + " hidden";
  }, duration);
}

// ── Utility: Number formatting ────────────────────────────────────
function fmtUSD(val) {
  return new Intl.NumberFormat("en-US", {
    style: "currency", currency: "USD", maximumFractionDigits: 0,
  }).format(val);
}
function fmtINR(lakhs) {
  return `₹${lakhs.toFixed(2)} Lakhs`;
}
function fmtNum(val) {
  return Number(val).toLocaleString("en-IN");
}

// ── API: Check health ─────────────────────────────────────────────
async function checkApiHealth() {
  const statusEl = document.getElementById("api-status-text");
  try {
    const res = await fetch(`${API_BASE}/`, { signal: AbortSignal.timeout(3000) });
    if (res.ok) {
      statusEl.textContent = "API Connected";
      statusEl.className = "text-xs text-emerald-400 font-medium";
    } else {
      throw new Error("bad status");
    }
  } catch {
    statusEl.textContent = "API Offline — train model first";
    statusEl.className = "text-xs text-rose-400 font-medium";
  }
}

// ── API: Load metrics ─────────────────────────────────────────────
async function loadMetrics() {
  const loading = document.getElementById("metrics-loading");
  const tableWrap = document.getElementById("metrics-table-wrap");
  const chartCard = document.getElementById("model-chart-card");
  const tbody = document.getElementById("metrics-tbody");
  const docBody = document.getElementById("doc-table-body");
  const heroR2 = document.getElementById("hero-r2");

  try {
    const res = await fetch(`${API_BASE}/metrics`);
    if (!res.ok) throw new Error("metrics not available");
    const data = await res.json();
    if (!data.success) throw new Error(data.error);

    const models = data.models;
    const bestName = data.best_model;

    // Update hero R²
    const bestMetric = models.find(m => m.name === bestName);
    if (bestMetric) {
      heroR2.textContent = `${(bestMetric.r2 * 100).toFixed(1)}%`;
    }

    // Build table
    tbody.innerHTML = "";
    docBody.innerHTML = "";

    models.forEach(m => {
      const isBest = m.name === bestName;
      const row = document.createElement("tr");
      row.className = "text-sm border-b border-white/5";
      row.innerHTML = `
        <td class="py-4 pr-6 text-gray-200 font-medium">
          ${m.name}${isBest ? '<span class="badge-best">★ Best</span>' : ""}
        </td>
        <td class="py-4 pr-6 text-right font-mono ${isBest ? "text-violet-400 font-bold" : "text-gray-300"}">${m.r2}</td>
        <td class="py-4 pr-6 text-right font-mono text-gray-400">${fmtNum(m.rmse)}</td>
        <td class="py-4 pr-6 text-right font-mono text-gray-400">${fmtNum(m.mae)}</td>
        <td class="py-4 pr-6 text-right font-mono text-gray-500">${m.cv_r2_mean} ± ${m.cv_r2_std}</td>
        <td class="py-4 text-right font-mono text-gray-500">${m.train_time_s}s</td>
      `;
      tbody.appendChild(row);

      // Doc table
      const docRow = document.createElement("tr");
      docRow.innerHTML = `
        <td class="py-2 pr-4">${m.name}${isBest ? " ★" : ""}</td>
        <td class="py-2 pr-4 text-right">${m.r2}</td>
        <td class="py-2 pr-4 text-right">${fmtNum(m.rmse)}</td>
        <td class="py-2 text-right">${fmtNum(m.mae)}</td>
      `;
      docBody.appendChild(docRow);
    });

    loading.classList.add("hidden");
    tableWrap.classList.remove("hidden");
    chartCard.classList.remove("hidden");

    // Chart.js bar chart
    buildModelBarChart(models, bestName);

  } catch (err) {
    loading.innerHTML = `<div class="text-gray-600 text-sm italic py-8">
      Model metrics not available. Start the Flask server and train the model:
      <br/><code class="text-violet-400">python backend/train_model.py</code>
      then <code class="text-violet-400">python backend/app.py</code>
    </div>`;
  }
}

// ── Chart: Model bar chart ────────────────────────────────────────
function buildModelBarChart(models, bestName) {
  const ctx = document.getElementById("model-bar-chart").getContext("2d");
  if (modelBarChartInstance) modelBarChartInstance.destroy();

  const labels = models.map(m => m.name);
  const values = models.map(m => m.r2);
  const colors = models.map(m =>
    m.name === bestName
      ? "rgba(139,92,246,0.85)"
      : "rgba(139,92,246,0.3)"
  );
  const borders = models.map(m =>
    m.name === bestName ? "#8b5cf6" : "rgba(139,92,246,0.5)"
  );

  modelBarChartInstance = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: "R² Score",
        data: values,
        backgroundColor: colors,
        borderColor: borders,
        borderWidth: 1.5,
        borderRadius: 8,
      }]
    },
    options: {
      indexAxis: "y",
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => ` R² = ${ctx.raw}`,
          }
        }
      },
      scales: {
        x: {
          min: 0, max: 1,
          grid: { color: "rgba(255,255,255,0.05)" },
          ticks: { color: "#9ca3af", font: { size: 11 } },
        },
        y: {
          grid: { display: false },
          ticks: { color: "#e5e7eb", font: { size: 12 } },
        }
      }
    }
  });
}

// ── API: Load features ────────────────────────────────────────────
async function loadFeatureChart() {
  try {
    const res = await fetch(`${API_BASE}/features`);
    if (!res.ok) throw new Error("features unavailable");
    const data = await res.json();
    if (!data.success) throw new Error(data.error);
    return data.features;
  } catch {
    return null;
  }
}

// ── Chart: Feature importance ─────────────────────────────────────
function buildFeatureChart(features) {
  const ctx = document.getElementById("feature-chart").getContext("2d");
  if (featureChartInstance) featureChartInstance.destroy();

  const labels = features.map(f => f.feature);
  const values = features.map(f => f.importance);

  const gradient = ctx.createLinearGradient(0, 0, 300, 0);
  gradient.addColorStop(0, "rgba(139,92,246,0.9)");
  gradient.addColorStop(1, "rgba(99,102,241,0.5)");

  featureChartInstance = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: "Importance",
        data: values,
        backgroundColor: gradient,
        borderRadius: 6,
        borderSkipped: false,
      }]
    },
    options: {
      indexAxis: "y",
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => ` ${(ctx.raw * 100).toFixed(2)}%`,
          }
        }
      },
      scales: {
        x: {
          grid: { color: "rgba(255,255,255,0.04)" },
          ticks: { color: "#9ca3af", font: { size: 10 } },
        },
        y: {
          grid: { display: false },
          ticks: { color: "#d1d5db", font: { size: 11 } },
        }
      }
    }
  });
}

// ── Form: Validate ────────────────────────────────────────────────
function validateForm(data) {
  const errors = [];
  const rules = {
    LotArea:     { min: 500,  max: 215000, label: "Lot Area" },
    OverallQual: { min: 1,    max: 10,     label: "Overall Quality" },
    YearBuilt:   { min: 1872, max: 2024,   label: "Year Built" },
    GrLivArea:   { min: 334,  max: 6000,   label: "Living Area" },
    TotalBsmtSF: { min: 0,    max: 6000,   label: "Basement SF" },
    GarageCars:  { min: 0,    max: 5,      label: "Garage Cars" },
    FullBath:    { min: 0,    max: 8,      label: "Full Bathrooms" },
    BedroomAbvGr:{ min: 0,    max: 12,     label: "Bedrooms" },
  };
  for (const [key, rule] of Object.entries(rules)) {
    const val = Number(data[key]);
    if (isNaN(val)) {
      errors.push(`${rule.label} must be a number.`);
    } else if (val < rule.min || val > rule.max) {
      errors.push(`${rule.label}: must be between ${rule.min} and ${rule.max}.`);
    }
  }
  return errors;
}

// ── Form: Set loading state ───────────────────────────────────────
function setLoading(isLoading) {
  const btn = document.getElementById("predict-btn");
  const text = document.getElementById("btn-text");
  const spinner = document.getElementById("btn-spinner");
  btn.disabled = isLoading;
  text.classList.toggle("hidden", isLoading);
  spinner.classList.toggle("hidden", !isLoading);
}

// ── Form: Fill sample data ────────────────────────────────────────
function fillSample(sampleData) {
  const defaults = {
    LotArea: 8450, OverallQual: 7, YearBuilt: 2003,
    GrLivArea: 1710, TotalBsmtSF: 856, GarageCars: 2,
    FullBath: 2, BedroomAbvGr: 3,
  };
  const data = sampleData || defaults;
  for (const [key, val] of Object.entries(data)) {
    const el = document.getElementById(key);
    if (el) el.value = val;
  }
  showToast("✨ Sample data loaded!", "info");
}

// ── Result: Render result card ────────────────────────────────────
function renderResult(data, features) {
  const placeholder = document.getElementById("result-placeholder");
  const resultSec = document.getElementById("result-section");

  placeholder.classList.add("hidden");
  resultSec.classList.remove("hidden");

  // Price
  document.getElementById("result-price").textContent = data.price_formatted;
  document.getElementById("result-price").classList.add("count-up");
  document.getElementById("result-usd").textContent =
    `≈ ${fmtUSD(data.price_usd)} USD`;

  // Model
  document.getElementById("result-model").textContent =
    `${data.model_used}`;

  // Category badge
  const catEl = document.getElementById("result-category");
  catEl.textContent = `${data.category_icon} ${data.category}`;
  catEl.className = `category-badge px-5 py-2 rounded-full text-base font-semibold cat-${data.category.toLowerCase()}`;

  // Confidence
  const conf = data.confidence;
  document.getElementById("confidence-val").textContent = `${conf}%`;
  setTimeout(() => {
    document.getElementById("confidence-bar").style.width = `${conf}%`;
  }, 100);

  // Model metrics
  const mm = data.model_metrics;
  document.getElementById("metric-r2").textContent = mm.r2 ?? "—";
  document.getElementById("metric-rmse").textContent = mm.rmse ? fmtNum(mm.rmse) : "—";
  document.getElementById("metric-mae").textContent = mm.mae ? fmtNum(mm.mae) : "—";

  // Feature chart
  if (features) {
    buildFeatureChart(features);
  }

  // Scroll
  resultSec.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// ── Form: Submit ──────────────────────────────────────────────────
async function handlePredict(e) {
  e.preventDefault();

  const form = document.getElementById("predict-form");
  const errorEl = document.getElementById("form-error");
  errorEl.classList.add("hidden");

  // Collect data
  const formData = Object.fromEntries(new FormData(form));
  const numericData = {};
  for (const [k, v] of Object.entries(formData)) {
    numericData[k] = parseFloat(v);
  }

  // Validate
  const errors = validateForm(numericData);
  if (errors.length > 0) {
    errorEl.textContent = errors.join(" ");
    errorEl.classList.remove("hidden");
    return;
  }

  setLoading(true);

  try {
    // Predict + features in parallel
    const [predRes, featuresData] = await Promise.all([
      fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(numericData),
      }),
      loadFeatureChart(),
    ]);

    if (!predRes.ok) {
      const errData = await predRes.json().catch(() => ({}));
      throw new Error(errData.error || `Server error: ${predRes.status}`);
    }

    const result = await predRes.json();
    if (!result.success) throw new Error(result.error);

    renderResult(result, featuresData);
    showToast(`🏠 Predicted: ${result.price_formatted}`, "success");

  } catch (err) {
    let msg = err.message;
    if (msg.includes("fetch") || msg.includes("Failed")) {
      msg = "Cannot reach API. Start the Flask server: python backend/app.py";
    }
    errorEl.textContent = `⚠ ${msg}`;
    errorEl.classList.remove("hidden");
    showToast("Prediction failed — check API server", "error");
  } finally {
    setLoading(false);
  }
}

// ── Showcase: Property card clicks ────────────────────────────────
function initPropertyCards() {
  document.querySelectorAll(".property-card").forEach(card => {
    card.addEventListener("click", () => {
      const sample = JSON.parse(card.dataset.sample || "{}");
      fillSample(sample);
      document.getElementById("predictor").scrollIntoView({ behavior: "smooth" });
      showToast("Property data loaded! Click Predict Price.", "info");
    });
  });
}

// ── Init ──────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", async () => {
  // Health check
  await checkApiHealth();

  // Sample data button
  document.getElementById("sample-btn").addEventListener("click", () => fillSample());

  // Form submit
  document.getElementById("predict-form").addEventListener("submit", handlePredict);

  // Property cards
  initPropertyCards();

  // Load metrics table
  loadMetrics();

  // Input: clear error on type
  document.querySelectorAll(".form-input").forEach(input => {
    input.addEventListener("input", () => {
      input.classList.remove("input-error");
      document.getElementById("form-error").classList.add("hidden");
    });
  });

  // Navbar active link scroll highlight
  const sections = document.querySelectorAll("section[id]");
  const navLinks = document.querySelectorAll("nav a");
  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        navLinks.forEach(link => {
          const href = link.getAttribute("href")?.replace("#", "");
          link.classList.toggle("text-violet-400", href === entry.target.id);
          link.classList.toggle("text-gray-400", href !== entry.target.id);
        });
      }
    });
  }, { rootMargin: "-40% 0px -55% 0px" });
  sections.forEach(s => observer.observe(s));
});
