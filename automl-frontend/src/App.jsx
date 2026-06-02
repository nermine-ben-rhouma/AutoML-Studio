import { useState, useEffect, useCallback } from "react";
import UploadStep from "./components/UploadStep";
import PreprocessStep from "./components/PreprocessStep";
import ConfigStep from "./components/ConfigStep";
import Dashboard from "./components/Dashboard";
import MLflowPanel from "./components/MLflowPanel";
import HistoryPanel from "./components/Historypanel";
import LoginPage from "./components/LoginPage";
import { checkHealth, getStats, fetchMe, getAuthConfig } from "./api";
import { getToken, getStoredUser, clearAuth, setOnUnauthorized } from "./authStorage";
import "./App.css";

export default function App() {
  const [step, setStep] = useState("upload");
  const [dataset, setDataset] = useState(null);
  const [config, setConfig] = useState(null);
  const [notif, setNotif] = useState(null);
  const [backendOk, setBackendOk] = useState(null);
  const [stats, setStats] = useState(null);
  const [showMLflow, setShowMLflow] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [user, setUser] = useState(null);
  const [authReady, setAuthReady] = useState(false);
  const [authEnabled, setAuthEnabled] = useState(true);

  const handleLogout = useCallback(() => {
    clearAuth();
    setUser(null);
    setStep("upload");
    setDataset(null);
    setConfig(null);
    setStats(null);
  }, []);

  useEffect(() => {
    setOnUnauthorized(handleLogout);
    return () => setOnUnauthorized(null);
  }, [handleLogout]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const cfg = await getAuthConfig();
        if (cancelled) return;
        setAuthEnabled(cfg.auth_enabled !== false);
        if (!cfg.auth_enabled) {
          setUser({ email: "dev@local", full_name: "Mode sans auth" });
          setAuthReady(true);
          return;
        }
        const token = getToken();
        if (token) {
          try {
            const me = await fetchMe();
            if (!cancelled) setUser(me);
          } catch {
            clearAuth();
          }
        } else {
          const stored = getStoredUser();
          if (stored && !token) clearAuth();
        }
      } catch {
        if (!cancelled) setAuthEnabled(true);
      } finally {
        if (!cancelled) setAuthReady(true);
      }
    })();
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    if (!user) return;
    checkHealth().then((h) => {
      setBackendOk(!!h);
      if (h) getStats().then(setStats).catch(() => { });
    });
  }, [user]);

  const showNotif = (msg, type = "success") => {
    setNotif({ msg, type });
    setTimeout(() => setNotif(null), 4000);
  };

  const handleUpload = (ds) => { setDataset(ds); setStep("preprocess"); showNotif(`✅ "${ds.filename || ds.name}" chargé — ${ds.rows} lignes`); };
  const handlePreprocess = (ds) => { setDataset(ds); setStep("config"); showNotif("✅ Dataset nettoyé — prêt pour la configuration"); };
  const handleConfig = (cfg) => { setConfig(cfg); setStep("dashboard"); };
  const handleReset = () => { setStep("upload"); setDataset(null); setConfig(null); };
  const refreshStats = () => getStats().then(setStats).catch(() => { });

  const STEPS = [
    { id: "upload", icon: "📁", label: "Upload" },
    { id: "preprocess", icon: "🧹", label: "Nettoyage" },
    { id: "config", icon: "⚙️", label: "Config" },
    { id: "dashboard", icon: "📊", label: "Résultats" },
  ];
  const stepIndex = STEPS.findIndex(s => s.id === step);

  if (!authReady) {
    return (
      <div className="auth-page">
        <p className="auth-loading">Chargement…</p>
      </div>
    );
  }

  if (authEnabled && !user) {
    return (
      <LoginPage
        onSuccess={setUser}
        defaultEmail="admin@automl.local"
      />
    );
  }

  return (
    <div className="app-root">
      <header className="app-header">
        <div className="logo" onClick={handleReset} style={{ cursor: "pointer" }}>
          <div className="logo-icon">⚗️</div>
          <div>
            <div className="logo-title">AutoML Studio</div>
            <div className="logo-sub">Machine Learning Avancée</div>
          </div>
        </div>

        <div className="stepper">
          {STEPS.map((s, i) => (
            <div key={s.id} className="stepper-item">
              <div className={`step-circle ${step === s.id ? "active" : ""} ${i < stepIndex ? "done" : ""}`}>{s.icon}</div>
              <div className={`step-label ${step === s.id ? "active" : ""}`}>{s.label}</div>
              {i < STEPS.length - 1 && <div className={`step-line ${i < stepIndex ? "done" : ""}`} />}
            </div>
          ))}
        </div>

        <div className="header-right">
          {user && (
            <div className="header-user" title={user.email}>
              👤 {user.full_name || user.email}
            </div>
          )}
          {stats && (
            <div className="header-stats" title={stats.mlflow_uri || ""}>
              <span>🔬 {stats.total_runs ?? 0} runs</span>
              <span>·</span>
              <span>🧪 {stats.total_experiments ?? 0} exp.</span>
              {stats.best_accuracy != null && (
                <>
                  <span>·</span>
                  <span>🏆 {(stats.best_accuracy * 100).toFixed(1)}%</span>
                </>
              )}
            </div>
          )}

          <button className="btn-mlflow" onClick={() => setShowMLflow(true)}>📊 MLflow</button>
          <button className="btn-history" onClick={() => setShowHistory(true)}>📋 Historique</button>

          {step !== "upload" && (
            <button className="btn-reset" onClick={handleReset}>↩ Nouveau</button>
          )}

          {authEnabled && (
            <button type="button" className="btn-logout" onClick={handleLogout}>
              Déconnexion
            </button>
          )}

          <div className={`status-pill ${backendOk === false ? "status-error" : ""}`}>
            <span className={`dot-live ${backendOk === false ? "dot-error" : ""}`} />
            {backendOk === null
              ? "Connexion..."
              : backendOk
                ? "FastAPI · MLflow ✅"
                : "Backend déconnecté ❌"}
          </div>
        </div>
      </header>

      {backendOk === false && (
        <div className="backend-banner">
          ⚠️ Backend non démarré — Ouvre un terminal et lance :
          <code> cd automl-backend && uvicorn main:app --reload</code>
        </div>
      )}

      <main className="app-main">
        {step === "upload" && <UploadStep onUpload={handleUpload} onNotif={showNotif} backendOk={backendOk} />}
        {step === "preprocess" && <PreprocessStep dataset={dataset} onDone={handlePreprocess} onBack={() => setStep("upload")} onNotif={showNotif} />}
        {step === "config" && <ConfigStep dataset={dataset} onConfig={handleConfig} onBack={() => setStep("preprocess")} onNotif={showNotif} />}
        {step === "dashboard" && <Dashboard dataset={dataset} config={config} onReset={handleReset} onNotif={showNotif} onRefreshStats={refreshStats} />}
      </main>

      {showMLflow && <MLflowPanel onClose={() => setShowMLflow(false)} onNotif={showNotif} />}
      {showHistory && <HistoryPanel onClose={() => setShowHistory(false)} onNotif={showNotif} />}

      {notif && (
        <div className={`notif notif-${notif.type}`}>
          <span>{notif.type === "success" ? "✅" : notif.type === "error" ? "❌" : "ℹ️"}</span>
          <span>{notif.msg}</span>
        </div>
      )}
    </div>
  );
}
