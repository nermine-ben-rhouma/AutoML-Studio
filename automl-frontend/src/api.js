// ============================================================
// AutoML Studio — Service API (connexion FastAPI)
// ============================================================

const BASE_URL = "http://localhost:8000";

// ── HELPERS ─────────────────────────────────────────────────
const handleResponse = async (res) => {
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Erreur serveur");
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
