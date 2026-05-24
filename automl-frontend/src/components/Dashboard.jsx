import { useState, useEffect, useRef, useCallback } from "react";
import { trainModels, generateReport } from "../api";
import ReportPanel from "./ReportPanel";
import { buildReportPayload } from "../reportPayload";

const ALGO_NAMES = {rf:"Random Forest",svm:"SVM",lr:"Logistic/Linear Reg.",knn:"KNN",dt:"Decision Tree",nb:"Naive Bayes",ridge:"Ridge",lasso:"Lasso",svr:"SVR"};
const ALGO_ICONS = {rf:"🌲",svm:"⚡",lr:"📈",knn:"🔵",dt:"🌳",nb:"🎲",ridge:"🔺",lasso:"🎯",svr:"⚡"};
const COLORS = ["#00a86b","#2563eb","#f59e0b","#7c3aed","#ef4444","#ec4899"];

export default function Dashboard({ dataset, config, onReset, onNotif, onRefreshStats }) {
  const [activeTab, setActiveTab]   = useState("results");
  const [results, setResults]       = useState(null);
  const [loading, setLoading]       = useState(true);
  const [progress, setProgress]     = useState(0);
  const [trainInfo, setTrainInfo]   = useState(null);
  const [error, setError]           = useState(null);
  const [showReport, setShowReport] = useState(false);
  const [reportData, setReportData] = useState(null);
  const [reportLoading, setReportLoading] = useState(false);
  const [reportError, setReportError] = useState(null);
  const trainingStarted = useRef(false);

  const isClassif = config.taskType === "classification";

  useEffect(() => {
    if (trainingStarted.current) return;
    trainingStarted.current = true;
    runTraining();
  }, []);

  const runTraining = async () => {
    setLoading(true); setProgress(0); setError(null);
    const iv = setInterval(() => setProgress(p => p < 85 ? p + 3 : p), 200);
    try {
      const payload = {
        dataset_id:      dataset.dataset_id,
        task_type:       config.taskType,
        target:          config.target,
        features:        config.features,
        algorithms:      config.selectedAlgos,
        test_size:       config.testSize,
        experiment_name: config.experimentName || "AutoML_Studio",
      };
      const data = await trainModels(payload);
      clearInterval(iv);
      setProgress(100);
      setTimeout(() => {
        setResults(data.results);
        setTrainInfo(data);
        setLoading(false);
        onNotif(`✅ Entraînement terminé ! Meilleur : ${ALGO_NAMES[data.best_model] || data.best_model}`, "success");
        onRefreshStats?.();
      }, 400);
    } catch(e) {
      clearInterval(iv);
      setError(e.message);
      setLoading(false);
      onNotif(`❌ Erreur : ${e.message}`, "error");
    }
  };

  // ✅ FIX: Vérifier que results est un tableau non vide avant réduire
  const best = results && Array.isArray(results) && results.length > 0 ? (isClassif
    ? results.reduce((a,b) => (a?.accuracy || 0) > (b?.accuracy || 0) ? a : b)
    : results.reduce((a,b) => (a?.r2 || 0) > (b?.r2 || 0) ? a : b)
  ) : null;

  const loadReport = useCallback(async () => {
    setReportLoading(true);
    setReportError(null);
    setReportData(null);
    setShowReport(true);
    try {
      const payload = buildReportPayload(dataset, config, trainInfo, results, best);
      const data = await generateReport(payload);
      if (!data?.sections?.length) {
        throw new Error("Le serveur a renvoyé un rapport vide.");
      }
      setReportData(data);
      onNotif("📋 Rapport généré (4 questions)", "success");
    } catch (e) {
      const msg = e?.message || "Impossible de générer le rapport.";
      setReportError(msg);
      onNotif(`❌ Rapport : ${msg}`, "error");
    } finally {
      setReportLoading(false);
    }
  }, [dataset, config, trainInfo, results, best, onNotif]);

  const exportCSV = () => {
    if (!results) return;
    const header = isClassif
      ? "Modele,Accuracy,F1,Precision,Recall,AUC,Train Acc,CV,Temps,Run ID"
      : "Modele,R2,RMSE,MAE,Train RMSE,CV R2,Temps,Run ID";
    const rows = results.map(r => isClassif
      ? `${r.algo_name},${r.accuracy},${r.f1},${r.precision},${r.recall},${r.auc},${r.train_accuracy},${r.cv_score},${r.time}s,${r.run_id}`
      : `${r.algo_name},${r.r2},${r.rmse},${r.mae},${r.train_rmse},${r.cv_r2},${r.time}s,${r.run_id}`
    );
    const csv = [header,...rows].join("\n");
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([csv],{type:"text/csv"}));
    a.download = `automl_${config.taskType}_${dataset.name}`; a.click();
    onNotif("📥 Résultats exportés !", "success");
  };

  const TABS = [
    {id:"results", label:"📊 Résultats"},
    {id:"compare", label:"⚖️ Comparaison"},
    {id:"mlflow",  label:"🔬 MLflow Runs"},
    {id:"data",    label:"🗂 Données"},
  ];

  // FIX 1: Infobar items définis avec des id stables pour les keys
  const infoItems = [
    {id:"name",    icon:"📂", text: dataset.name || dataset.filename},
    {id:"task",    icon: isClassif ? "🏷" : "📈", text: isClassif ? "Classification" : "Régression"},
    {id:"target",  icon:"🎯", text: <><span>Target : </span><strong>{config.target}</strong></>},
    {id:"shape",   icon:"🔢", text: `${config.features.length} features · ${dataset.rows} lignes`},
    {id:"algos",   icon:"🤖", text: `${config.selectedAlgos.length} algorithmes`},
    {id:"mlflow",  icon:"🔬", text: <><span>MLflow : </span><strong>{config.experimentName}</strong></>},
  ];

  return (
    <div className="dashboard-page">

      {/* INFO BAR — FIX 1: Fragment avec key stable via import { Fragment } */}
      <div className="dash-infobar">
        {infoItems.map((item, i) => (
          // FIX: utiliser un Fragment importé avec key, pas <> qui ne supporte pas key
          <div key={item.id} style={{display:"contents"}}>
            <div className="infobar-item">
              <span className="infobar-icon">{item.icon}</span>
              <span>{item.text}</span>
            </div>
            {i < infoItems.length - 1 && <div className="infobar-sep" />}
          </div>
        ))}
      </div>

      {/* LOADING */}
      {loading && (
        <div className="loading-overlay">
          <div className="loading-card">
            <div className="loading-icon">⚗️</div>
            <div className="loading-title">Entraînement en cours via FastAPI...</div>
            <div className="loading-bar-wrap">
              <div className="loading-bar-fill" style={{width:`${progress}%`}}/>
            </div>
            <div className="loading-steps">
              {/* FIX 2: key sur le label du step (stable et unique) */}
              {["Prétraitement","Entraînement","MLflow tracking","Évaluation"].map((s, i) => (
                <div key={s} className={`loading-step ${progress > i * 25 ? "step-done" : ""}`}>
                  {progress > i * 25 ? "✅" : "⏳"} {s}
                </div>
              ))}
            </div>
            <div className="loading-pct">{progress}%</div>
          </div>
        </div>
      )}

      {/* ERROR */}
      {error && !loading && (
        <div className="error-card">
          <div className="error-icon">❌</div>
          <div className="error-title">Erreur d'entraînement</div>
          <div className="error-msg">{error}</div>
          <div style={{display:"flex",gap:10,marginTop:16}}>
            <button className="btn-launch btn-launch-active" style={{width:"auto",padding:"10px 20px"}} onClick={runTraining}>🔄 Réessayer</button>
            <button className="btn-back" onClick={onReset}>↩ Retour</button>
          </div>
        </div>
      )}

      {!loading && !error && results && best && (
        <>
          {/* BEST BANNER */}
          <div className="best-banner">
            <div className="best-left">
              <div className="best-crown">🏆</div>
              <div>
                <div className="best-label">Meilleur modèle</div>
                <div className="best-name">{ALGO_ICONS[best?.algo_id || "rf"]} {best?.algo_name || "N/A"}</div>
              </div>
            </div>
            <div className="best-scores">
              {isClassif ? (
                <>
                  <div className="best-score"><div className="bs-val">{((best?.accuracy || 0)*100).toFixed(1)}%</div><div className="bs-lbl">Accuracy</div></div>
                  <div className="best-score"><div className="bs-val">{(best?.f1 || 0).toFixed(3)}</div><div className="bs-lbl">F1-Score</div></div>
                  <div className="best-score"><div className="bs-val">{(best?.auc || 0).toFixed(3)}</div><div className="bs-lbl">AUC-ROC</div></div>
                  <div className="best-score"><div className="bs-val">{(best?.cv_score || 0).toFixed(3)}</div><div className="bs-lbl">CV Score</div></div>
                </>
              ) : (
                <>
                  <div className="best-score"><div className="bs-val">{(best?.r2 || 0).toFixed(3)}</div><div className="bs-lbl">R²</div></div>
                  <div className="best-score"><div className="bs-val">{(best?.rmse || 0).toFixed(3)}</div><div className="bs-lbl">RMSE</div></div>
                  <div className="best-score"><div className="bs-val">{(best?.mae || 0).toFixed(3)}</div><div className="bs-lbl">MAE</div></div>
                  <div className="best-score"><div className="bs-val">{(best?.cv_r2 || 0).toFixed(3)}</div><div className="bs-lbl">CV R²</div></div>
                </>
              )}
            </div>
            <div style={{display:"flex",gap:8,flexWrap:"wrap"}}>
              <button
                type="button"
                className="btn-report"
                onClick={loadReport}
                disabled={reportLoading}
                title="Rapport : importance, stabilité, erreurs, biais/variance"
              >
                {reportLoading ? "⏳ Rapport…" : `📋 Rapport — ${best?.algo_name || "meilleur modèle"}`}
              </button>
              <button className="btn-export-csv" onClick={exportCSV}>📥 Export CSV</button>
              <a className="btn-export-csv" href="http://localhost:5000" target="_blank" rel="noreferrer">🔬 MLflow UI</a>
            </div>
          </div>

          {showReport && (
            <ReportPanel
              report={reportData}
              loading={reportLoading}
              error={reportError}
              bestModel={best}
              onClose={() => {
                setShowReport(false);
                setReportData(null);
                setReportError(null);
              }}
              onRetry={loadReport}
            />
          )}

          {/* TABS */}
          <div className="dash-tabs">
            {TABS.map(t => (
              <button key={t.id} className={`dash-tab ${activeTab === t.id ? "dash-tab-active" : ""}`}
                onClick={() => setActiveTab(t.id)}>{t.label}</button>
            ))}
          </div>

          {/* TAB RESULTS */}
          {activeTab === "results" && (
            <div className="dash-content fade-in">
              <div className={`metrics-grid metrics-${isClassif ? 4 : 3}`}>
                {isClassif ? (
                  <>
                    <MetricCard emoji="🎯" label="Accuracy"  value={`${((best?.accuracy || 0)*100).toFixed(1)}%`} color="green"/>
                    <MetricCard emoji="⚖️" label="F1-Score"  value={(best?.f1 || 0).toFixed(3)} color="blue"/>
                    <MetricCard emoji="🔍" label="Precision" value={(best?.precision || 0).toFixed(3)} color="orange"/>
                    <MetricCard emoji="📡" label="Recall"    value={(best?.recall || 0).toFixed(3)} color="pink"/>
                  </>
                ) : (
                  <>
                    <MetricCard emoji="📐" label="R²"   value={(best?.r2 || 0).toFixed(3)}   color="green"  delta="↑ meilleur proche de 1"/>
                    <MetricCard emoji="📉" label="RMSE" value={(best?.rmse || 0).toFixed(3)} color="blue"   delta="↓ meilleur proche de 0"/>
                    <MetricCard emoji="📏" label="MAE"  value={(best?.mae || 0).toFixed(3)}  color="orange" delta="↓ meilleur proche de 0"/>
                  </>
                )}
              </div>

              <div className="charts-row">
                <div className="chart-card">
                  <div className="chart-title">{isClassif ? "🏆 Accuracy" : "📊 R²"} par algorithme</div>
                  <div className="bar-chart">
                    {/* FIX 3: key sur run_id (unique par run) plutôt que algo_id (peut se répéter) */}
                    {[...results].sort((a,b) => isClassif ? (b?.accuracy || 0) - (a?.accuracy || 0) : (b?.r2 || 0) - (a?.r2 || 0)).map((r, i) => {
                      const val = isClassif ? (r?.accuracy || 0) : (r?.r2 || 0);
                      return (
                        <div className="bar-row" key={r.run_id}>
                          <div className="bar-label">{ALGO_ICONS[r.algo_id]} {(r.algo_name || "").split(" ")[0]}</div>
                          <div className="bar-track">
                            <div className="bar-fill" style={{
                              width: `${Math.max(0, Math.min(100, val * 100))}%`,
                              background: r.algo_id === best?.algo_id
                                ? "linear-gradient(90deg,#00a86b,#00c980)"
                                : COLORS[i % COLORS.length]
                            }}>
                              {(val * 100).toFixed(1)}%
                            </div>
                          </div>
                          <div className="bar-pct">{val.toFixed(3)}</div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                <div className="chart-card">
                  <div className="chart-title">🔍 Overfitting (Train vs Test)</div>
                  <div className="bar-chart">
                    {/* FIX 4: key sur run_id */}
                    {results.map((r, i) => {
                      const tr  = isClassif ? (r?.train_accuracy || 0) : Math.max(0, 1 - (r?.train_rmse || 10) / 10);
                      const te  = isClassif ? (r?.accuracy || 0) : (r?.r2 || 0);
                      const gap = Math.abs(tr - te);
                      return (
                        <div key={r.run_id} style={{marginBottom:14}}>
                          <div style={{fontSize:"0.7rem",fontWeight:700,marginBottom:4,color:"var(--text2)",display:"flex",justifyContent:"space-between"}}>
                            <span>{ALGO_ICONS[r.algo_id]} {r.algo_name}</span>
                            <span style={{fontFamily:"monospace",color:gap>0.1?"var(--red)":"var(--green1)"}}>Δ={gap.toFixed(3)} {gap>0.1?"⚠️":"✅"}</span>
                          </div>
                          <div style={{display:"flex",gap:4}}>
                            {/* FIX 5: key stable = run_id + label */}
                            {[{label:"Train",val:tr},{label:"Test",val:te}].map(({label, val}) => (
                              <div key={`${r.run_id}-${label}`} style={{flex:1}}>
                                <div style={{fontSize:"0.58rem",color:"var(--muted)",marginBottom:2}}>{label} {(val*100).toFixed(1)}%</div>
                                <div className="bar-track" style={{height:16}}>
                                  <div className="bar-fill" style={{
                                    width:`${Math.max(0,Math.min(100,val*100))}%`,
                                    background: COLORS[i % COLORS.length],
                                    opacity: label === "Train" ? 0.5 : 1
                                  }}/>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>

              {/* FEATURE IMPORTANCE */}
              {best?.feature_importance && Object.keys(best.feature_importance).length > 0 && (
                <div className="chart-card">
                  <div className="chart-title">🏆 Feature Importance — {best?.algo_name}</div>
                  <div className="bar-chart">
                    {/* FIX 6: key sur feat (nom de feature, stable et unique dans ce contexte) */}
                    {Object.entries(best.feature_importance).sort((a,b) => b[1]-a[1]).slice(0,10).map(([feat, val], i) => (
                      <div className="bar-row" key={feat}>
                        <div className="bar-label">{feat}</div>
                        <div className="bar-track">
                          <div className="bar-fill" style={{width:`${val*100*3}%`,background:COLORS[i%COLORS.length]}}>
                            {(val*100).toFixed(1)}%
                          </div>
                        </div>
                        <div className="bar-pct">{val.toFixed(3)}</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* CONFUSION MATRIX */}
              {isClassif && best?.confusion_matrix && (
                <div className="chart-card">
                  <div className="chart-title">🔲 Matrice de Confusion — {best?.algo_name}</div>
                  <div className="cm-wrap">
                    <div className="confusion-matrix">
                      <div className="cm-header"/>
                      <div className="cm-header">Prédit : 0</div>
                      <div className="cm-header">Prédit : 1</div>
                      <div className="cm-header">Réel : 0</div>
                      <div className="cm-cell cm-tn"><div className="cm-val">{best.confusion_matrix[0]?.[0] || 0}</div><div className="cm-pct">TN</div></div>
                      <div className="cm-cell cm-fp"><div className="cm-val">{best.confusion_matrix[0]?.[1] || 0}</div><div className="cm-pct">FP</div></div>
                      <div className="cm-header">Réel : 1</div>
                      <div className="cm-cell cm-fn"><div className="cm-val">{best.confusion_matrix[1]?.[0] || 0}</div><div className="cm-pct">FN</div></div>
                      <div className="cm-cell cm-tp"><div className="cm-val">{best.confusion_matrix[1]?.[1] || 0}</div><div className="cm-pct">TP</div></div>
                    </div>
                    <div className="cm-explain">
                      {/* FIX 7: key sur le label de chaque explication */}
                      {[
                        {key:"tn", cls:"cm-exp-green", label:"TN", desc:"Vrais Négatifs"},
                        {key:"fp", cls:"cm-exp-red",   label:"FP", desc:"Faux Positifs (Erreur type I)"},
                        {key:"fn", cls:"cm-exp-red",   label:"FN", desc:"Faux Négatifs (Erreur type II)"},
                        {key:"tp", cls:"cm-exp-green", label:"TP", desc:"Vrais Positifs"},
                      ].map(({key, cls, label, desc}) => (
                        <div key={key} className={`cm-exp-item ${cls}`}>
                          <strong>{label}</strong> {desc}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* TAB COMPARE */}
          {activeTab === "compare" && (
            <div className="dash-content fade-in">
              <div className="chart-card">
                <div className="chart-title">⚖️ Tableau Comparatif — Données réelles
                  <button className="btn-export-csv" style={{marginLeft:"auto"}} onClick={exportCSV}>📥 Export CSV</button>
                </div>
                <table className="compare-table">
                  <thead>
                    <tr>
                      <th>Modèle</th>
                      {isClassif ? (
                        /* FIX 8: key sur chaque <th> */
                        <>
                          <th key="acc">Accuracy</th><th key="f1">F1</th>
                          <th key="prec">Precision</th><th key="rec">Recall</th>
                          <th key="auc">AUC</th><th key="cv">CV</th>
                          <th key="train">Train Acc</th><th key="overfit">Δ Overfit</th>
                        </>
                      ) : (
                        <>
                          <th key="r2">R²</th><th key="rmse">RMSE</th>
                          <th key="mae">MAE</th><th key="trmse">Train RMSE</th>
                          <th key="cvr2">CV R²</th>
                        </>
                      )}
                      <th>Temps</th><th>Run ID</th>
                    </tr>
                  </thead>
                  <tbody>
                    {/* FIX 9: key sur run_id dans le tbody */}
                    {[...results].sort((a,b) => isClassif ? (b?.accuracy || 0) - (a?.accuracy || 0) : (b?.r2 || 0) - (a?.r2 || 0)).map(r => (
                      <tr key={r.run_id} className={r.algo_id === best?.algo_id ? "row-best" : ""}>
                        <td><strong>{ALGO_ICONS[r.algo_id]} {r.algo_name}</strong></td>
                        {isClassif ? (
                          <>
                            <td style={r.algo_id===best?.algo_id?{color:"var(--green1)",fontWeight:900}:{}}>{((r?.accuracy || 0)*100).toFixed(1)}%</td>
                            <td>{(r?.f1 || 0).toFixed(3)}</td>
                            <td>{(r?.precision || 0).toFixed(3)}</td>
                            <td>{(r?.recall || 0).toFixed(3)}</td>
                            <td>{(r?.auc || 0).toFixed(3)}</td>
                            <td>{(r?.cv_score || 0).toFixed(3)}</td>
                            <td style={{color:"var(--muted)"}}>{((r?.train_accuracy || 0)*100).toFixed(1)}%</td>
                            <td style={{color:(r?.overfitting_gap || 0)>0.1?"var(--red)":"var(--green1)",fontFamily:"monospace"}}>
                              {(r?.overfitting_gap || 0).toFixed(3)}{(r?.overfitting_gap || 0)>0.1?" ⚠️":" ✅"}
                            </td>
                          </>
                        ) : (
                          <>
                            <td style={r.algo_id===best?.algo_id?{color:"var(--green1)",fontWeight:900}:{}}>{(r?.r2 || 0).toFixed(3)}</td>
                            <td>{(r?.rmse || 0).toFixed(3)}</td>
                            <td>{(r?.mae || 0).toFixed(3)}</td>
                            <td style={{color:"var(--muted)"}}>{(r?.train_rmse || 0).toFixed(3)}</td>
                            <td>{(r?.cv_r2 || 0).toFixed(3)}</td>
                          </>
                        )}
                        <td style={{color:"var(--muted)"}}>{r.time}s</td>
                        <td><code style={{fontSize:"0.6rem",color:"var(--muted)"}}>{r.run_id?.slice(0,8)}...</code></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* TAB MLFLOW */}
          {activeTab === "mlflow" && (
            <div className="dash-content fade-in">
              <div className="mlflow-info-card">
                <div className="mlflow-header">
                  <div className="mlflow-logo">🔬</div>
                  <div>
                    <div className="mlflow-title">Expérience MLflow : <strong>{config.experimentName}</strong></div>
                    <div className="mlflow-sub">{results.length} runs enregistrés · SQLite backend</div>
                  </div>
                  <a href="http://localhost:5000" target="_blank" rel="noreferrer" className="btn-mlflow-open">
                    Ouvrir MLflow UI →
                  </a>
                </div>
                <div className="mlflow-runs">
                  {/* FIX 10: key sur run_id (identifiant MLflow unique par run) */}
                  {results.map((r, i) => (
                    <div key={r.run_id} className={`mlflow-run-card ${r.algo_id === best?.algo_id ? "mlflow-run-best" : ""}`}>
                      <div className="run-header">
                        <span className="run-algo">{ALGO_ICONS[r.algo_id]} {r.algo_name}</span>
                        {r.algo_id === best?.algo_id && <span className="badge-best">🏆 MEILLEUR</span>}
                        <span className="run-time">{r.time}s</span>
                      </div>
                      <div className="run-metrics">
                        {isClassif ? (
                          /* FIX 11: key sur label dans les métriques */
                          [
                            {label:"Accuracy", val:`${((r?.accuracy || 0)*100).toFixed(1)}%`},
                            {label:"F1",       val:(r?.f1 || 0).toFixed(3)},
                            {label:"AUC",      val:(r?.auc || 0).toFixed(3)},
                            {label:"CV",       val:(r?.cv_score || 0).toFixed(3)},
                            {label:"Overfit Δ",val:(r?.overfitting_gap || 0).toFixed(3), color:(r?.overfitting_gap || 0)>0.1?"var(--red)":"var(--green1)"},
                          ].map(({label, val, color}) => (
                            <div key={label} className="run-metric">
                              <span>{label}</span>
                              <strong style={color?{color}:{}}>{val}</strong>
                            </div>
                          ))
                        ) : (
                          [
                            {label:"R²",   val:(r?.r2 || 0).toFixed(3)},
                            {label:"RMSE", val:(r?.rmse || 0).toFixed(3)},
                            {label:"MAE",  val:(r?.mae || 0).toFixed(3)},
                            {label:"CV R²",val:(r?.cv_r2 || 0).toFixed(3)},
                          ].map(({label, val}) => (
                            <div key={label} className="run-metric">
                              <span>{label}</span>
                              <strong>{val}</strong>
                            </div>
                          ))
                        )}
                      </div>
                      <div className="run-id"><code>Run ID : {r.run_id}</code></div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* TAB DATA */}
          {activeTab === "data" && (
            <div className="dash-content fade-in">
              <div className="chart-card">
                <div className="chart-title">
                  🗂 {dataset.name || dataset.filename}
                  <span className="chart-sub">{dataset.rows} lignes · {dataset.columns.length} colonnes</span>
                </div>
                <div className="data-table-wrap">
                  <table className="data-table">
                    <thead>
                      <tr>
                        {/* FIX 12: key sur c dans le thead */}
                        {dataset.columns.map(c => (
                          <th key={c} style={c === config.target ? {color:"var(--orange)"} : {}}>
                            {c === config.target ? "🎯 " : ""}{c}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {/* FIX 13: key composite rowIndex-stable sur les lignes du preview */}
                      {(dataset.preview || []).slice(0, 10).map((row, i) => (
                        <tr key={`row-${i}`}>
                          {dataset.columns.map(c => (
                            <td key={c} style={c === config.target ? {fontWeight:800,color:"var(--orange)"} : {}}>
                              {row[c] ?? ""}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

function MetricCard({emoji, label, value, color, delta}) {
  return (
    <div className={`metric-card m-${color}`}>
      <div className="metric-emoji">{emoji}</div>
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
      {delta && <div className="metric-delta">{delta}</div>}
    </div>
  );
}