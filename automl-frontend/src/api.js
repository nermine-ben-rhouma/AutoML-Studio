// ============================================================
// AutoML Studio — Service API (connexion FastAPI)
// ============================================================

const BASE_URL = "http://localhost:8000";

// ── HELPERS ─────────────────────────────────────────────────
const handleResponse = async (res) => {
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    const detail = err.detail;
    let message = res.statusText;
    if (typeof detail === "string") {
      message = detail;
    } else if (Array.isArray(detail)) {
      message = detail.map((d) => d.msg || JSON.stringify(d)).join(" — ");
    } else if (detail && typeof detail === "object") {
      message = detail.message || JSON.stringify(detail);
    }
    throw new Error(message || `Erreur serveur (${res.status})`);
  }
  return res.json();
};

// ── UPLOAD DATASET ───────────────────────────────────────────
export const uploadDataset = async (file) => {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE_URL}/upload`, { method: "POST", body: form });
  return handleResponse(res);
};

// ── TRAIN ────────────────────────────────────────────────────
export const trainModels = async (payload) => {
  const res = await fetch(`${BASE_URL}/train`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return handleResponse(res);
};

// ── EXPERIMENTS ──────────────────────────────────────────────
export const getExperiments = async () => {
  const res = await fetch(`${BASE_URL}/experiments`);
  return handleResponse(res);
};

export const getExperimentRuns = async (experimentName) => {
  const res = await fetch(`${BASE_URL}/experiments/${encodeURIComponent(experimentName)}/runs`);
  return handleResponse(res);
};

// ── MODELS REGISTRY ──────────────────────────────────────────
export const getRegisteredModels = async () => {
  const res = await fetch(`${BASE_URL}/models`);
  return handleResponse(res);
};

// ── RAPPORT ──────────────────────────────────────────────────
export const generateReport = async (payload) => {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 120000);
  try {
    const res = await fetch(`${BASE_URL}/report`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    return handleResponse(res);
  } catch (e) {
    if (e.name === "AbortError") {
      throw new Error("Délai dépassé (120 s) — réduisez le nombre de features ou réessayez.");
    }
    throw e;
  } finally {
    clearTimeout(timeout);
  }
};

// ── STATS ────────────────────────────────────────────────────
export const getStats = async () => {
  const res = await fetch(`${BASE_URL}/stats`);
  return handleResponse(res);
};

// ── HEALTH ───────────────────────────────────────────────────
export const checkHealth = async () => {
  try {
    const res = await fetch(`${BASE_URL}/health`);
    return handleResponse(res);
  } catch {
    return null;
  }
};
