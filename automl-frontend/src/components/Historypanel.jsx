import { useState, useEffect } from "react";
import { getExperiments, getExperimentRuns, getRegisteredModels } from "../api";

const ALGO_ICONS = {
  rf:"🌲", svm:"⚡", lr:"📈", knn:"🔵", dt:"🌳",
  nb:"🎲", ridge:"🔺", lasso:"🎯", svr:"⚡"
};

export default function HistoryPanel({ onClose, onNotif }) {
  const [experiments, setExperiments] = useState([]);
  const [selectedExp, setSelectedExp] = useState(null);
  const [runs, setRuns]               = useState([]);
  const [models, setModels]           = useState([]);
  const [activeTab, setActiveTab]     = useState("runs");
  const [loading, setLoading]         = useState(true);
  const [search, setSearch]           = useState("");
  const [sortBy, setSortBy]           = useState("date");
  const [filterTask, setFilterTask]   = useState("all");

  useEffect(() => { loadAll(); }, []);

  const loadAll = async () => {
    setLoading(true);
    try {
      const [exps, mods] = await Promise.all([getExperiments(), getRegisteredModels()]);
      setExperiments(exps);
      setModels(mods);
      if (exps.length > 0) {
        setSelectedExp(exps[0].name);
        await loadRuns(exps[0].name);
      }
    } catch {
      onNotif("❌ Impossible de charger l'historique MLflow", "error");
    }
    setLoading(false);
  };

  const loadRuns = async (expName) => {
    setSelectedExp(expName);
    try {
      const r = await getExperimentRuns(expName);
      setRuns(r);
    } catch { setRuns([]); }
  };

  // Filtrage + tri des runs
  const filteredRuns = runs
    .filter(r => {
      const matchSearch = !search ||
        r.algo?.toLowerCase().includes(search.toLowerCase()) ||
        r.target?.toLowerCase().includes(search.toLowerCase());
      const matchTask = filterTask === "all" || r.task_type === filterTask;
      return matchSearch && matchTask;
    })
    .sort((a, b) => {
      if (sortBy === "date")     return new Date(b.start_time) - new Date(a.start_time);
      if (sortBy === "accuracy") return (b.metrics?.accuracy || 0) - (a.metrics?.accuracy || 0);
      if (sortBy === "r2")       return (b.metrics?.r2 || 0) - (a.metrics?.r2 || 0);
      return 0;
    });

  const bestRun = filteredRuns.reduce((best, r) => {
    const score = r.metrics?.accuracy || r.metrics?.r2 || 0;
    const bScore = best ? (best.metrics?.accuracy || best.metrics?.r2 || 0) : -1;
    return score > bScore ? r : best;
  }, null);

  const exportHistory = () => {
    const header = "Run ID,Algorithme,Task,Target,Accuracy,F1,R2,RMSE,CV,Date";
    const rows = filteredRuns.map(r =>
      `${r.run_id},${r.algo},${r.task_type},${r.target},` +
      `${r.metrics?.accuracy||""},${r.metrics?.f1_score||""},` +
      `${r.metrics?.r2||""},${r.metrics?.rmse||""},` +
      `${r.metrics?.cv_score||r.metrics?.cv_r2||""},${r.start_time}`
    );
    const blob = new Blob([[header,...rows].join("\n")], {type:"text/csv"});
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `historique_${selectedExp}.csv`; a.click();
    onNotif("📥 Historique exporté !", "success");
  };

  return (
    <div className="panel-overlay" onClick={e => e.target === e.currentTarget && onClose()}>
      <div className="history-panel">

        {/* HEADER */}
        <div className="panel-header" style={{background:"linear-gradient(135deg,#ecfdf5,#f0fdf8)"}}>
          <div style={{display:"flex",alignItems:"center",gap:12}}>
            <div className="panel-header-icon" style={{background:"linear-gradient(135deg,#00a86b,#00c980)"}}>📋</div>
            <div>
              <div className="panel-header-title">Historique des Expérimentations</div>
              <div className="panel-header-sub">{runs.length} runs · {experiments.length} expériences · MLflow</div>
            </div>
          </div>
          <div style={{display:"flex",gap:8}}>
            <button className="btn-export-csv" onClick={exportHistory}>📥 Export CSV</button>
            <a href="http://localhost:5000" target="_blank" rel="noreferrer" className="btn-mlflow-open">MLflow UI ↗</a>
            <button className="panel-close" onClick={onClose}>✕</button>
          </div>
        </div>

        {/* EXPERIMENTS TABS */}
        <div className="history-exp-tabs">
          {experiments.map(exp => (
            <button key={exp.id}
              className={`history-exp-tab ${selectedExp===exp.name?"active":""}`}
              onClick={() => loadRuns(exp.name)}>
              🧪 {exp.name}
            </button>
          ))}
        </div>

        {/* TABS */}
        <div className="panel-tabs">
          {[
            {id:"runs",   label:`▶ Runs (${runs.length})`},
            {id:"models", label:`📦 Models (${models.length})`},
            {id:"stats",  label:"📊 Statistiques"},
          ].map(t => (
            <button key={t.id}
              className={`panel-tab ${activeTab===t.id?"panel-tab-active":""}`}
              onClick={() => setActiveTab(t.id)}>{t.label}</button>
          ))}
        </div>

        <div className="panel-body">
          {loading ? (
            <div className="panel-loading"><div className="spinner"/>Chargement...</div>
          ) : (
            <>
              {/* ── TAB RUNS ── */}
              {activeTab==="runs" && (
                <div className="fade-in" style={{display:"flex",flexDirection:"column",gap:12}}>

                  {/* FILTRES */}
                  <div className="history-filters">
                    <input className="history-search" placeholder="🔍 Rechercher algorithme ou target..."
                      value={search} onChange={e => setSearch(e.target.value)}/>
                    <select className="config-select" style={{width:160}}
                      value={filterTask} onChange={e => setFilterTask(e.target.value)}>
                      <option value="all">Tous les types</option>
                      <option value="classification">Classification</option>
                      <option value="regression">Régression</option>
                    </select>
                    <select className="config-select" style={{width:160}}
                      value={sortBy} onChange={e => setSortBy(e.target.value)}>
                      <option value="date">Trier par date</option>
                      <option value="accuracy">Trier par Accuracy</option>
                      <option value="r2">Trier par R²</option>
                    </select>
                    <div className="history-count">{filteredRuns.length} résultat{filteredRuns.length!==1?"s":""}</div>
                  </div>

                  {/* BEST RUN BANNER */}
                  {bestRun && (
                    <div className="history-best-banner">
                      <span>🏆</span>
                      <div>
                        <div style={{fontWeight:900,fontSize:"0.85rem"}}>
                          Meilleur run : {ALGO_ICONS[bestRun.algo]} {bestRun.algo?.toUpperCase()}
                        </div>
                        <div style={{fontSize:"0.68rem",color:"var(--muted)"}}>
                          Target : {bestRun.target} ·
                          {bestRun.metrics?.accuracy
                            ? ` Accuracy : ${(bestRun.metrics.accuracy*100).toFixed(1)}%`
                            : ` R² : ${bestRun.metrics?.r2?.toFixed(3)}`}
                        </div>
                      </div>
                    </div>
                  )}

                  {/* RUNS LIST */}
                  {filteredRuns.length === 0 ? (
                    <div className="panel-empty">Aucun run trouvé</div>
                  ) : (
                    <div className="runs-list">
                      {filteredRuns.map((r, i) => (
                        <div key={r.run_id}
                          className={`run-item ${r.run_id===bestRun?.run_id?"run-item-best":""}`}>
                          <div className="run-item-header">
                            <div className="run-item-left">
                              <span className="run-item-icon">{ALGO_ICONS[r.algo] || "🤖"}</span>
                              <div>
                                <div className="run-item-algo">{r.algo || "Unknown"}</div>
                                <div className="run-item-meta">
                                  <span className={`run-task-badge ${r.task_type==="classification"?"badge-class":"badge-reg"}`}>
                                    {r.task_type==="classification"?"🏷 Classif.":"📈 Régress."}
                                  </span>
                                  <span>Target : <strong>{r.target}</strong></span>
                                  <span>·</span>
                                  <span style={{fontFamily:"monospace",fontSize:"0.62rem",color:"var(--muted)"}}>
                                    {r.run_id?.slice(0,8)}...
                                  </span>
                                </div>
                              </div>
                            </div>
                            <div className="run-item-right">
                              {r.run_id===bestRun?.run_id && (
                                <span className="badge badge-best">🏆 BEST</span>
                              )}
                              <span className={`run-status ${r.status==="FINISHED"?"status-ok":"status-fail"}`}>
                                {r.status==="FINISHED"?"✅":"❌"}
                              </span>
                              <span className="run-date">
                                {new Date(r.start_time).toLocaleDateString("fr-FR")}
                              </span>
                            </div>
                          </div>

                          {/* MÉTRIQUES */}
                          <div className="run-item-metrics">
                            {r.task_type==="classification" ? (
                              <>
                                {r.metrics?.accuracy   !== undefined && <MetricPill label="Accuracy"  value={`${(r.metrics.accuracy*100).toFixed(1)}%`} color="green"/>}
                                {r.metrics?.f1_score   !== undefined && <MetricPill label="F1"        value={r.metrics.f1_score.toFixed(3)}            color="blue"/>}
                                {r.metrics?.auc_roc    !== undefined && <MetricPill label="AUC"       value={r.metrics.auc_roc.toFixed(3)}             color="purple"/>}
                                {r.metrics?.cv_score   !== undefined && <MetricPill label="CV"        value={r.metrics.cv_score.toFixed(3)}            color="orange"/>}
                                {r.metrics?.overfitting_gap !== undefined && (
                                  <MetricPill label="Overfit Δ"
                                    value={r.metrics.overfitting_gap.toFixed(3)}
                                    color={r.metrics.overfitting_gap > 0.1 ? "red" : "green"}/>
                                )}
                              </>
                            ) : (
                              <>
                                {r.metrics?.r2   !== undefined && <MetricPill label="R²"   value={r.metrics.r2.toFixed(3)}   color="green"/>}
                                {r.metrics?.rmse !== undefined && <MetricPill label="RMSE" value={r.metrics.rmse.toFixed(3)} color="blue"/>}
                                {r.metrics?.mae  !== undefined && <MetricPill label="MAE"  value={r.metrics.mae.toFixed(3)}  color="orange"/>}
                                {r.metrics?.cv_r2 !== undefined && <MetricPill label="CV R²" value={r.metrics.cv_r2.toFixed(3)} color="purple"/>}
                              </>
                            )}
                            {r.metrics?.training_time !== undefined && (
                              <MetricPill label="⏱" value={`${r.metrics.training_time}s`} color="muted"/>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* ── TAB MODELS ── */}
              {activeTab==="models" && (
                <div className="fade-in" style={{display:"flex",flexDirection:"column",gap:10}}>
                  <div className="panel-section-title">{models.length} modèle{models.length!==1?"s":""} dans le registry</div>
                  {models.length === 0 ? (
                    <div className="panel-empty">Aucun modèle enregistré — lancez un entraînement</div>
                  ) : (
                    models.map(m => (
                      <div key={m.name} className="model-registry-card">
                        <div className="model-registry-name">📦 {m.name}</div>
                        <div className="model-registry-versions">
                          {m.versions.map(v => (
                            <div key={v.version} className="model-version-item">
                              <span className="badge badge-good">v{v.version}</span>
                              <span className="version-stage">{v.stage || "None"}</span>
                              <code className="version-runid">{v.run_id?.slice(0,10)}...</code>
                            </div>
                          ))}
                        </div>
                      </div>
                    ))
                  )}
                </div>
              )}

              {/* ── TAB STATS ── */}
              {activeTab==="stats" && (
                <div className="fade-in" style={{display:"flex",flexDirection:"column",gap:14}}>
                  <div className="stats-overview">
                    {[
                      {icon:"🧪",label:"Expériences",val:experiments.length,color:"green"},
                      {icon:"▶",label:"Runs totaux",val:runs.length,color:"blue"},
                      {icon:"📦",label:"Modèles registry",val:models.length,color:"purple"},
                      {icon:"✅",label:"Runs réussis",val:runs.filter(r=>r.status==="FINISHED").length,color:"green"},
                    ].map(s => (
                      <div key={s.label} className={`stat-card stat-${s.color}`}>
                        <div style={{fontSize:"1.5rem",marginBottom:4}}>{s.icon}</div>
                        <div className="stat-num">{s.val}</div>
                        <div className="stat-lbl">{s.label}</div>
                      </div>
                    ))}
                  </div>

                  {/* Répartition par algo */}
                  <div className="chart-card">
                    <div className="chart-title">🤖 Runs par algorithme</div>
                    <div className="bar-chart">
                      {Object.entries(
                        runs.reduce((acc, r) => {
                          acc[r.algo || "unknown"] = (acc[r.algo || "unknown"] || 0) + 1;
                          return acc;
                        }, {})
                      ).sort((a,b) => b[1]-a[1]).map(([algo, count], i) => (
                        <div className="bar-row" key={algo}>
                          <div className="bar-label">{ALGO_ICONS[algo]||"🤖"} {algo}</div>
                          <div className="bar-track">
                            <div className="bar-fill" style={{
                              width:`${(count/runs.length)*100}%`,
                              background:`hsl(${i*40},70%,50%)`
                            }}>{count}</div>
                          </div>
                          <div className="bar-pct">{count}</div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Répartition Classification vs Régression */}
                  <div className="chart-card">
                    <div className="chart-title">📊 Classification vs Régression</div>
                    <div className="bar-chart">
                      {[
                        {label:"Classification",count:runs.filter(r=>r.task_type==="classification").length,color:"#2563eb"},
                        {label:"Régression",    count:runs.filter(r=>r.task_type==="regression").length,   color:"#f59e0b"},
                      ].map(item => (
                        <div className="bar-row" key={item.label}>
                          <div className="bar-label">{item.label}</div>
                          <div className="bar-track">
                            <div className="bar-fill" style={{
                              width:`${runs.length>0?(item.count/runs.length)*100:0}%`,
                              background:item.color
                            }}>{item.count}</div>
                          </div>
                          <div className="bar-pct">{item.count}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function MetricPill({ label, value, color }) {
  const colors = {
    green:"#ecfdf5,#065f46,#6ee7b7",
    blue:"#eff6ff,#1e40af,#93c5fd",
    purple:"#faf5ff,#6b21a8,#d8b4fe",
    orange:"#fffbeb,#92400e,#fde68a",
    red:"#fef2f2,#991b1b,#fca5a5",
    muted:"#f1f5f9,#64748b,#cbd5e1",
  };
  const [bg,text,border] = (colors[color]||colors.muted).split(",");
  return (
    <div style={{background:bg,color:text,border:`1px solid ${border}`,
      borderRadius:6,padding:"3px 8px",fontSize:"0.65rem",fontWeight:700,
      display:"flex",gap:4,alignItems:"center"}}>
      <span style={{opacity:0.7}}>{label}</span>
      <strong style={{fontFamily:"monospace"}}>{value}</strong>
    </div>
  );
}