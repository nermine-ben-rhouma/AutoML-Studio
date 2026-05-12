import { useState, useEffect } from "react";
import { getExperiments, getExperimentRuns, getRegisteredModels } from "../api";

export default function MLflowPanel({ onClose, onNotif }) {
  const [experiments, setExperiments]   = useState([]);
  const [selectedExp, setSelectedExp]   = useState(null);
  const [runs, setRuns]                 = useState([]);
  const [models, setModels]             = useState([]);
  const [activeTab, setActiveTab]       = useState("experiments");
  const [loading, setLoading]           = useState(true);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    setLoading(true);
    try {
      const [exps, mods] = await Promise.all([getExperiments(), getRegisteredModels()]);
      setExperiments(exps);
      setModels(mods);
      if (exps.length > 0) loadRuns(exps[0].name);
    } catch(e) {
      onNotif("❌ Impossible de charger MLflow", "error");
    }
    setLoading(false);
  };

  const loadRuns = async (expName) => {
    setSelectedExp(expName);
    try {
      const r = await getExperimentRuns(expName);
      setRuns(r);
    } catch(e) {
      setRuns([]);
    }
  };

  return (
    <div className="mlflow-overlay" onClick={e=>e.target===e.currentTarget&&onClose()}>
      <div className="mlflow-panel">
        {/* HEADER */}
        <div className="mlflow-panel-header">
          <div style={{display:"flex",alignItems:"center",gap:12}}>
            <div className="mlflow-panel-icon">🔬</div>
            <div>
              <div className="mlflow-panel-title">MLflow Dashboard</div>
              <div className="mlflow-panel-sub">Tracking · Model Registry · Experiments</div>
            </div>
          </div>
          <div style={{display:"flex",gap:8}}>
            <a href="http://localhost:5000" target="_blank" rel="noreferrer" className="btn-mlflow-open">
              Ouvrir MLflow UI ↗
            </a>
            <button className="mlflow-close" onClick={onClose}>✕</button>
          </div>
        </div>

        {/* TABS */}
        <div className="mlflow-panel-tabs">
          {[{id:"experiments",label:"🧪 Expériences"},{id:"runs",label:"▶ Runs"},{id:"models",label:"📦 Model Registry"}].map(t=>(
            <button key={t.id} className={`mlflow-panel-tab ${activeTab===t.id?"mlflow-tab-active":""}`}
              onClick={()=>setActiveTab(t.id)}>{t.label}</button>
          ))}
        </div>

        <div className="mlflow-panel-body">
          {loading ? (
            <div style={{textAlign:"center",padding:"3rem",color:"var(--muted)"}}>
              <div className="spinner" style={{margin:"0 auto 12px"}}/>
              <div>Chargement MLflow...</div>
            </div>
          ) : (
            <>
              {/* EXPERIMENTS */}
              {activeTab==="experiments" && (
                <div className="mlflow-experiments">
                  <div className="mlflow-section-title">
                    {experiments.length} expérience{experiments.length!==1?"s":""} trouvée{experiments.length!==1?"s":""}
                  </div>
                  {experiments.length===0 ? (
                    <div className="mlflow-empty">Aucune expérience — Lance un entraînement d'abord</div>
                  ) : (
                    experiments.map(exp=>(
                      <div key={exp.id} className={`mlflow-exp-card ${selectedExp===exp.name?"mlflow-exp-active":""}`}
                        onClick={()=>{loadRuns(exp.name);setActiveTab("runs");}}>
                        <div className="mlflow-exp-name">🧪 {exp.name}</div>
                        <div className="mlflow-exp-meta">ID : {exp.id} · {exp.lifecycle}</div>
                        <div className="mlflow-exp-arrow">Voir les runs →</div>
                      </div>
                    ))
                  )}
                </div>
              )}

              {/* RUNS */}
              {activeTab==="runs" && (
                <div>
                  <div style={{display:"flex",gap:8,marginBottom:12,flexWrap:"wrap"}}>
                    {experiments.map(exp=>(
                      <button key={exp.id}
                        className={`feat-btn ${selectedExp===exp.name?"feat-btn-active":""}`}
                        onClick={()=>loadRuns(exp.name)}>
                        🧪 {exp.name}
                      </button>
                    ))}
                  </div>
                  <div className="mlflow-section-title">{runs.length} run{runs.length!==1?"s":""} — {selectedExp}</div>
                  {runs.length===0 ? (
                    <div className="mlflow-empty">Aucun run dans cette expérience</div>
                  ) : (
                    <div className="mlflow-runs-table-wrap">
                      <table className="compare-table">
                        <thead><tr>
                          <th>Algorithme</th><th>Task</th><th>Target</th>
                          <th>Accuracy/R²</th><th>F1/RMSE</th><th>CV</th>
                          <th>Statut</th><th>Date</th>
                        </tr></thead>
                        <tbody>
                          {runs.map(r=>(
                            <tr key={r.run_id}>
                              <td><strong>{r.algo||r.run_name}</strong></td>
                              <td><span className={`example-badge ${r.task_type==="classification"?"badge-class":"badge-reg"}`}>
                                {r.task_type==="classification"?"🏷 Classif.":"📈 Régress."}
                              </span></td>
                              <td style={{fontFamily:"monospace",fontSize:"0.7rem"}}>{r.target}</td>
                              <td style={{color:"var(--green1)",fontWeight:800}}>
                                {r.metrics?.accuracy?(r.metrics.accuracy*100).toFixed(1)+"%":r.metrics?.r2?.toFixed(3)||"—"}
                              </td>
                              <td>{r.metrics?.f1_score?.toFixed(3)||r.metrics?.rmse?.toFixed(3)||"—"}</td>
                              <td>{r.metrics?.cv_score?.toFixed(3)||r.metrics?.cv_r2?.toFixed(3)||"—"}</td>
                              <td><span className={`badge ${r.status==="FINISHED"?"badge-best":"badge-good"}`}>{r.status}</span></td>
                              <td style={{fontSize:"0.65rem",color:"var(--muted)"}}>{new Date(r.start_time).toLocaleDateString("fr-FR")}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              )}

              {/* MODELS */}
              {activeTab==="models" && (
                <div>
                  <div className="mlflow-section-title">
                    {models.length} modèle{models.length!==1?"s":""} enregistré{models.length!==1?"s":""}
                  </div>
                  {models.length===0 ? (
                    <div className="mlflow-empty">Aucun modèle dans le registry — Entraîne un modèle d'abord</div>
                  ) : (
                    models.map(m=>(
                      <div key={m.name} className="mlflow-model-card">
                        <div className="mlflow-model-name">📦 {m.name}</div>
                        <div className="mlflow-model-versions">
                          {m.versions.map(v=>(
                            <div key={v.version} className="mlflow-version">
                              <span className="badge badge-good">v{v.version}</span>
                              <span style={{fontSize:"0.65rem",color:"var(--muted)"}}>{v.stage}</span>
                              <code style={{fontSize:"0.6rem",color:"var(--muted)"}}>{v.run_id?.slice(0,8)}...</code>
                            </div>
                          ))}
                        </div>
                      </div>
                    ))
                  )}
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
