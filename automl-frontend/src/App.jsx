import { useState, useEffect } from "react";
import UploadStep     from "./components/UploadStep";
import PreprocessStep from "./components/PreprocessStep";
import ConfigStep     from "./components/ConfigStep";
import Dashboard      from "./components/Dashboard";
import MLflowPanel    from "./components/MLflowPanel";
import HistoryPanel   from "./components/Historypanel";   
import { checkHealth, getStats } from "./api";
import "./App.css";

export default function App() {
  const [step, setStep]               = useState("upload");
  const [dataset, setDataset]         = useState(null);
  const [config, setConfig]           = useState(null);
  const [notif, setNotif]             = useState(null);
  const [backendOk, setBackendOk]     = useState(null);
  const [stats, setStats]             = useState(null);
  const [showMLflow, setShowMLflow]   = useState(false);
  const [showHistory, setShowHistory] = useState(false);  

  useEffect(() => {
    checkHealth().then(h => {
      setBackendOk(!!h);
      if (h) getStats().then(setStats).catch(() => {});
    });
  }, []);

  const showNotif = (msg, type = "success") => {
    setNotif({ msg, type });
    setTimeout(() => setNotif(null), 4000);
  };

  const handleUpload     = (ds)  => { setDataset(ds); setStep("preprocess"); showNotif(`✅ "${ds.filename || ds.name}" chargé — ${ds.rows} lignes`); };
  const handlePreprocess = (ds)  => { setDataset(ds); setStep("config"); showNotif("✅ Dataset nettoyé — prêt pour la configuration"); };
  const handleConfig     = (cfg) => { setConfig(cfg); setStep("dashboard"); };
  const handleReset      = ()    => { setStep("upload"); setDataset(null); setConfig(null); };
  const refreshStats     = ()    => getStats().then(setStats).catch(() => {});

  const STEPS = [
    { id: "upload",     icon: "📁", label: "Upload"    },
    { id: "preprocess", icon: "🧹", label: "Nettoyage" },
    { id: "config",     icon: "⚙️",  label: "Config"    },
    { id: "dashboard",  icon: "📊", label: "Résultats" },
  ];
  const stepIndex = STEPS.findIndex(s => s.id === step);

  return (
    <div className="app-root">
      <header className="app-header">
        <div className="logo" onClick={handleReset} style={{cursor:"pointer"}}>
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
          {stats && (
            <div className="header-stats">
              <span>🔬 {stats.total_runs} runs</span>
              <span>·</span>
              <span>🧪 {stats.total_experiments} exp.</span>
            </div>
          )}

          {/* ── boutons panneaux ── */}
          <button className="btn-mlflow"   onClick={() => setShowMLflow(true)}>📊 MLflow</button>
          <button className="btn-history"  onClick={() => setShowHistory(true)}>📋 Historique</button> {/* ← AJOUT */}

          {step !== "upload" && (
            <button className="btn-reset" onClick={handleReset}>↩ Nouveau</button>
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
          <span> puis </span>
          <code>mlflow ui --backend-store-uri sqlite:///mlflow.db</code>
        </div>
      )}

      <main className="app-main">
        {step === "upload"     && <UploadStep     onUpload={handleUpload}                           onNotif={showNotif} backendOk={backendOk} />}
        {step === "preprocess" && <PreprocessStep dataset={dataset}                                 onDone={handlePreprocess} onBack={() => setStep("upload")} onNotif={showNotif} />}
        {step === "config"     && <ConfigStep     dataset={dataset}                                 onConfig={handleConfig} onBack={() => setStep("preprocess")} onNotif={showNotif} />}
        {step === "dashboard"  && <Dashboard      dataset={dataset} config={config}                 onReset={handleReset} onNotif={showNotif} onRefreshStats={refreshStats} />}
      </main>

      {/* Panneaux latéraux */}
      {showMLflow  && <MLflowPanel  onClose={() => setShowMLflow(false)}  onNotif={showNotif} />}
      {showHistory && <HistoryPanel onClose={() => setShowHistory(false)} onNotif={showNotif} />} {/* ← AJOUT */}

      {notif && (
        <div className={`notif notif-${notif.type}`}>
          <span>{notif.type === "success" ? "✅" : notif.type === "error" ? "❌" : "ℹ️"}</span>
          <span>{notif.msg}</span>
        </div>
      )}
    </div>
  );
}