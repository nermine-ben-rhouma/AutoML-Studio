// ============================================================
// AutoML Studio — Service API (connexion FastAPI + JWT)
// ============================================================

import { getToken, triggerUnauthorized } from "./authStorage";

const BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

const authHeaders = (extra = {}) => {
  const headers = { ...extra };
  const token = getToken();
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }
  return headers;
};

const handleResponse = async (res, { clearOn401 = true } = {}) => {
  if (res.status === 401 && clearOn401) {
    triggerUnauthorized();
    throw new Error("Session expirée — reconnectez-vous.");
  }
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    const detail = err.detail;
    let message = res.statusText;
    if (typeof detail === "string") {
      message = detail;
    } else if (Array.isArray(detail)) {
      message = detail
        .map((d) => d.msg || d.message || (typeof d === "string" ? d : JSON.stringify(d)))
        .join(" — ");
    } else if (detail && typeof detail === "object") {
      message = detail.message || JSON.stringify(detail);
    }
    throw new Error(message || `Erreur serveur (${res.status})`);
  }
  return res.json();
};

const apiFetch = (path, options = {}) => {
  const headers = authHeaders(options.headers || {});
  return fetch(`${BASE_URL}${path}`, { ...options, headers });
};

// ── AUTH ─────────────────────────────────────────────────────
export const getAuthConfig = async () => {
  const res = await fetch(`${BASE_URL}/auth/config`);
  return handleResponse(res);
};

export const login = async (email, password) => {
  const res = await fetch(`${BASE_URL}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  return handleResponse(res, { clearOn401: false });
};

export const register = async (email, password, fullName) => {
  const res = await fetch(`${BASE_URL}/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password, full_name: fullName || null }),
  });
  return handleResponse(res, { clearOn401: false });
};

export const fetchMe = async () => {
  const res = await apiFetch("/auth/me");
  return handleResponse(res);
};

// ── UPLOAD DATASET ───────────────────────────────────────────
export const uploadDataset = async (file) => {
  const form = new FormData();
  form.append("file", file);
  const res = await apiFetch("/upload", { method: "POST", body: form });
  return handleResponse(res);
};

// ── PREPROCESS ───────────────────────────────────────────────
export const preprocessDataset = async (payload) => {
  const res = await apiFetch("/preprocess", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return handleResponse(res);
};

// ── TRAIN ────────────────────────────────────────────────────
export const trainModels = async (payload) => {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 900000); // 15 min (gros CSV)
  try {
    const res = await apiFetch("/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    return handleResponse(res);
  } catch (e) {
    if (e.name === "AbortError") {
      throw new Error(
        "Entraînement trop long (15 min). Réduisez les algorithmes ou le nombre de lignes."
      );
    }
    if (e.message === "Failed to fetch") {
      throw new Error(
        "Connexion au serveur interrompue. Vérifiez que le backend tourne avec " +
          "python run_dev.py (pas uvicorn --reload seul, qui coupe l'entraînement quand MLflow écrit)."
      );
    }
    throw e;
  } finally {
    clearTimeout(timeout);
  }
};

// ── EXPERIMENTS ──────────────────────────────────────────────
export const getExperiments = async () => {
  const res = await apiFetch("/experiments");
  return handleResponse(res);
};

export const getExperimentRuns = async (experimentName) => {
  const res = await apiFetch(`/experiments/${encodeURIComponent(experimentName)}/runs`);
  return handleResponse(res);
};

// ── MODELS REGISTRY ──────────────────────────────────────────
export const getRegisteredModels = async () => {
  const res = await apiFetch("/models");
  return handleResponse(res);
};

// ── RAPPORT ──────────────────────────────────────────────────
export const generateReport = async (payload) => {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 120000);
  try {
    const res = await apiFetch("/report", {
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
  const res = await apiFetch("/stats");
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

export { BASE_URL };
