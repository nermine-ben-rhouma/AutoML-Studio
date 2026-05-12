import { useState, useEffect, useMemo } from "react";

const ALGOS_CLASSIFICATION = [
  {id:"rf", name:"Random Forest",       icon:"🌲", desc:"Ensemble robuste, peu sensible au surapprentissage"},
  {id:"svm",name:"SVM",                 icon:"⚡", desc:"Séparateur à vaste marge, haute dimension"},
  {id:"lr", name:"Logistic Regression", icon:"📈", desc:"Modèle linéaire simple et interprétable"},
  {id:"knn",name:"KNN",                 icon:"🔵", desc:"Classification par k plus proches voisins"},
  {id:"dt", name:"Decision Tree",       icon:"🌳", desc:"Arbre de décision, très interprétable"},
  {id:"nb", name:"Naive Bayes",         icon:"🎲", desc:"Probabiliste, rapide sur grands datasets"},
];
const ALGOS_REGRESSION = [
  {id:"rf",   name:"Random Forest",    icon:"🌲", desc:"Ensemble d'arbres pour régression"},
  {id:"lr",   name:"Linear Regression",icon:"📈", desc:"Régression linéaire classique"},
  {id:"ridge",name:"Ridge Regression", icon:"🔺", desc:"Régression L2, réduit l'overfitting"},
  {id:"lasso",name:"Lasso Regression", icon:"🎯", desc:"Régression L1, sélection de features"},
  {id:"svr",  name:"SVR",              icon:"⚡", desc:"Support Vector Regression"},
  {id:"dt",   name:"Decision Tree",    icon:"🌳", desc:"Arbre de décision pour régression"},
];

function normalizeColName(col) {
  return String(col).trim().replace(/\s+/g, "_");
}

// ── ANALYSE DE COLONNE ────────────────────────────────────
/**
 * Analyse une colonne et retourne ses caractéristiques.
 * Utilisé pour filtrer les colonnes inadaptées comme target.
 */
function analyzeColumn(colName, dataset) {
  const vals = (dataset.preview || [])
    .map(r => r[colName])
    .filter(v => v != null && v !== "" && v !== "null" && v !== "NaN");

  if (vals.length === 0) return { type: "empty", unique: 0, ratio: 0 };

  const uniq       = new Set(vals.map(v => String(v).trim()));
  const uniqueCount = dataset.unique_counts?.[colName] ?? uniq.size;
  const totalRows   = dataset.rows ?? vals.length;
  const ratio       = uniqueCount / totalRows;
  const allNumeric  = vals.every(v => !isNaN(parseFloat(v)) && isFinite(v));

  // Détecter si c'est probablement un ID / texte libre
  const looksLikeId = ratio > 0.8 || uniqueCount > 100;
  // Détecter si c'est numérique continu (régression)
  const looksLikeContinuous = allNumeric && uniqueCount > 20 && ratio > 0.05;
  // Détecter si c'est une bonne cible de classification
  const looksLikeLabel = !looksLikeId && uniqueCount >= 2 && uniqueCount <= 50;

  let suggestedTask = null;
  if (looksLikeId)         suggestedTask = null;            // inadaptée
  else if (looksLikeLabel) suggestedTask = "classification";
  else if (looksLikeContinuous) suggestedTask = "regression";

  return {
    type:         looksLikeId ? "id" : allNumeric ? "numeric" : "categorical",
    unique:       uniqueCount,
    ratio:        ratio,
    allNumeric,
    looksLikeId,
    looksLikeLabel,
    looksLikeContinuous,
    suggestedTask,
  };
}

function getNumericCols(dataset) {
  if (dataset?.numeric_cols?.length) return dataset.numeric_cols;
  if (dataset?.dtypes) {
    return Object.entries(dataset.dtypes)
      .filter(([, t]) => t.startsWith("int") || t.startsWith("float"))
      .map(([col]) => col);
  }
  return [];
}

function getUsableFeatures(dataset, excludeCol) {
  if (!dataset?.columns) return [];
  const normalized = normalizeColName(excludeCol);
  return dataset.columns.filter(c => {
    if (c === excludeCol || normalizeColName(c) === normalized) return false;
    const vals = (dataset.preview || []).map(r => r[c]).filter(v => v != null && v !== "");
    return vals.length > 0;
  });
}

export default function ConfigStep({ dataset, onConfig, onBack, onNotif }) {
  const [taskType,       setTaskType]      = useState(null);
  const [target,         setTarget]        = useState("");
  const [features,       setFeatures]      = useState([]);
  const [selectedAlgos,  setSelectedAlgos] = useState([]);
  const [testSize,       setTestSize]      = useState(20);
  const [expName,        setExpName]       = useState("AutoML_Studio");
  const [showAllTargets, setShowAllTargets]= useState(false); // FIX: toggle pour voir toutes les colonnes

  const numericCols = useMemo(() => getNumericCols(dataset), [dataset]);

  // ── ANALYSE DE TOUTES LES COLONNES ──────────────────────
  const colAnalysis = useMemo(() => {
    const result = {};
    (dataset?.columns || []).forEach(col => {
      result[col] = analyzeColumn(col, dataset);
    });
    return result;
  }, [dataset]);

  // FIX: Séparer les colonnes en "recommandées" et "autres" pour le select target
  const { recommendedTargets, otherTargets } = useMemo(() => {
    const recommended = [];
    const other       = [];
    (dataset?.columns || []).forEach(col => {
      const info = colAnalysis[col];
      if (info.looksLikeId || info.type === "empty") {
        other.push(col); // IDs et colonnes vides → groupe "autres"
      } else {
        recommended.push(col);
      }
    });
    return { recommendedTargets: recommended, otherTargets: other };
  }, [colAnalysis]);

  // ── AUTO-DÉTECTION ───────────────────────────────────────
  const targetInfo = target ? colAnalysis[target] : null;
  const autoDetected = targetInfo?.suggestedTask ?? null;

  // FIX: Alerte si l'utilisateur choisit une colonne inadaptée
  const targetWarning = useMemo(() => {
    if (!target || !targetInfo) return null;
    if (targetInfo.looksLikeId) {
      return `⚠️ '${target}' semble être un identifiant (${Math.round(targetInfo.ratio * 100)}% de valeurs uniques). ` +
             `Choisissez une colonne avec peu de catégories distinctes.`;
    }
    if (targetInfo.unique < 2) {
      return `⚠️ '${target}' ne contient qu'une seule valeur unique — impossible à prédire.`;
    }
    return null;
  }, [target, targetInfo]);

  useEffect(() => {
    if (!target) return;
    // Suggestion de tâche uniquement si pas encore choisi ou si on change de target
    if (autoDetected && !taskType) setTaskType(autoDetected);
  }, [target]);

  useEffect(() => {
    if (!target) return;
    setFeatures(getUsableFeatures(dataset, target));
  }, [target]);

  useEffect(() => {
    if (taskType === "classification") setSelectedAlgos(["rf", "svm", "lr"]);
    else if (taskType === "regression") setSelectedAlgos(["rf", "lr", "ridge"]);
  }, [taskType]);

  const algos            = taskType === "classification" ? ALGOS_CLASSIFICATION : ALGOS_REGRESSION;
  const availableFeatures = useMemo(() => getUsableFeatures(dataset, target), [dataset, target]);
  const nullPerCol       = dataset?.null_per_col   || {};
  const uniqueCounts     = dataset?.unique_counts  || {};

  const toggleAlgo = id =>
    setSelectedAlgos(prev =>
      prev.includes(id) ? (prev.length > 1 ? prev.filter(a => a !== id) : prev) : [...prev, id]
    );
  const toggleFeat = col =>
    setFeatures(prev =>
      prev.includes(col) ? (prev.length > 1 ? prev.filter(f => f !== col) : prev) : [...prev, col]
    );

  // ── VALIDATION ───────────────────────────────────────────
  const validationError = useMemo(() => {
    if (!target)                  return "⚠️ Choisissez une variable cible";
    if (targetWarning)            return targetWarning; // FIX: bloquer si colonne inadaptée
    if (!taskType)                return "⚠️ Choisissez le type de tâche";
    if (features.length === 0)    return "⚠️ Sélectionnez au moins une feature";
    if (selectedAlgos.length === 0) return "⚠️ Sélectionnez au moins un algorithme";
    const trainCount = Math.round(dataset.rows * (100 - testSize) / 100);
    if (trainCount < 10) return `⚠️ Trop peu d'exemples d'entraînement (${trainCount}). Réduisez le test size.`;
    return null;
  }, [target, targetWarning, taskType, features, selectedAlgos, testSize, dataset.rows]);

  const canGo = !validationError;

  const handleLaunch = () => {
    if (!canGo) return;
    onConfig({
      taskType,
      target:         normalizeColName(target),
      features:       features.map(normalizeColName),
      selectedAlgos,
      testSize:       testSize / 100,
      experimentName: expName.trim() || "AutoML_Studio",
      dataset_id:     dataset.dataset_id,
    });
  };

  // ── RENDER OPTION TARGET ─────────────────────────────────
  const renderTargetOption = (col) => {
    const info = colAnalysis[col];
    const u    = uniqueCounts[col] ?? info?.unique ?? "?";
    const nans = nullPerCol[col];
    let label  = col;
    if (info?.looksLikeId)          label += ` ⛔ (ID — ${u} valeurs uniques)`;
    else if (info?.suggestedTask === "classification") label += ` ✅ Classification (${u} classes)`;
    else if (info?.suggestedTask === "regression")     label += ` 📈 Régression (${u} valeurs)`;
    else                                               label += ` (${u} uniques)`;
    if (nans > 0)                   label += ` ⚠️ ${nans} NaN`;
    return <option key={col} value={col}>{label}</option>;
  };

  return (
    <div className="config-page">
      <div className="config-container">
        <div className="config-header">
          <div className="config-dataset-badge">
            <span>📂</span>
            <span>{dataset.name}</span>
            <span className="sep">·</span>
            <span>{dataset.rows} lignes</span>
            <span className="sep">·</span>
            <span>{dataset.columns.length} colonnes</span>
            {dataset.null_count > 0 && (
              <><span className="sep">·</span>
              <span style={{color:"#f59e0b"}}>⚠️ {dataset.null_count} valeurs manquantes</span></>
            )}
          </div>
          <h2 className="config-title">Configuration du Projet ML</h2>
        </div>

        <div className="config-grid">
          <div className="config-left">

            {/* CIBLE */}
            <div className="config-card">
              <div className="config-card-num">1</div>
              <div className="config-card-body">
                <div className="config-card-title">Variable Cible (Target)</div>
                <div className="config-card-desc">
                  La colonne à prédire — choisissez une colonne avec peu de catégories (classification) ou une valeur numérique continue (régression)
                </div>

                <select
                  className="config-select"
                  value={target}
                  onChange={e => setTarget(e.target.value)}
                >
                  <option value="">— Choisir une colonne —</option>

                  {/* FIX: Colonnes recommandées en premier */}
                  {recommendedTargets.length > 0 && (
                    <optgroup label="✅ Colonnes recommandées">
                      {recommendedTargets.map(renderTargetOption)}
                    </optgroup>
                  )}

                  {/* FIX: Colonnes déconseillées (IDs, etc.) dans un groupe séparé */}
                  {otherTargets.length > 0 && (showAllTargets || target && otherTargets.includes(target)) && (
                    <optgroup label="⛔ Colonnes déconseillées (IDs, vides)">
                      {otherTargets.map(renderTargetOption)}
                    </optgroup>
                  )}
                </select>

                {/* FIX: Bouton pour voir aussi les colonnes déconseillées */}
                {otherTargets.length > 0 && (
                  <button
                    style={{fontSize:"0.75rem", color:"#6b7280", background:"none", border:"none", cursor:"pointer", marginTop:"0.25rem", padding:0}}
                    onClick={() => setShowAllTargets(v => !v)}
                  >
                    {showAllTargets ? "▲ Masquer les colonnes déconseillées" : `▼ Voir aussi ${otherTargets.length} colonne(s) déconseillée(s)`}
                  </button>
                )}

                {/* FIX: Avertissement si colonne inadaptée choisie */}
                {targetWarning && (
                  <div style={{
                    marginTop:"0.5rem", padding:"0.5rem 0.75rem",
                    background:"#fef3c7", borderLeft:"3px solid #f59e0b",
                    borderRadius:"0.25rem", fontSize:"0.82rem", color:"#92400e"
                  }}>
                    {targetWarning}
                  </div>
                )}

                {/* Auto-détection */}
                {autoDetected && target && !targetWarning && (
                  <div className={`auto-detect ${autoDetected === "classification" ? "detect-class" : "detect-reg"}`}>
                    {autoDetected === "classification" ? "🏷" : "📈"} Auto-détecté :{" "}
                    <strong>{autoDetected === "classification" ? "Classification" : "Régression"}</strong>
                    <span style={{fontSize:"0.75rem", opacity:0.7}}>
                      {" "}— {uniqueCounts[target] ?? targetInfo?.unique ?? "?"} valeurs uniques
                    </span>
                  </div>
                )}
              </div>
            </div>

            {/* TYPE DE TÂCHE */}
            <div className="config-card">
              <div className="config-card-num">2</div>
              <div className="config-card-body">
                <div className="config-card-title">Type de Tâche ML</div>
                <div className="config-card-desc">Classification = catégories · Régression = valeur continue</div>
                <div className="task-choice">
                  <div className={`task-btn ${taskType === "classification" ? "task-active-class" : ""}`}
                    onClick={() => setTaskType("classification")}>
                    <div className="task-icon">🏷</div>
                    <div className="task-name">Classification</div>
                    <div className="task-desc">Prédire une catégorie<br />(Oui/Non, 0/1, walk/run...)</div>
                    <div className="task-algos">RF · SVM · KNN · LR · DT · NB</div>
                  </div>
                  <div className={`task-btn ${taskType === "regression" ? "task-active-reg" : ""}`}
                    onClick={() => setTaskType("regression")}>
                    <div className="task-icon">📈</div>
                    <div className="task-name">Régression</div>
                    <div className="task-desc">Prédire une valeur<br />(prix, score, mesure...)</div>
                    <div className="task-algos">RF · LR · Ridge · Lasso · SVR · DT</div>
                  </div>
                </div>
              </div>
            </div>

            {/* SPLIT */}
            <div className="config-card">
              <div className="config-card-num">3</div>
              <div className="config-card-body">
                <div className="config-card-title">Split Train / Test</div>
                <div className="config-card-desc">Proportion des données réservées à l'évaluation</div>
                <div className="split-row">
                  <span className="split-label">Train</span>
                  <input type="range" min={10} max={40} step={5} value={testSize}
                    onChange={e => setTestSize(parseInt(e.target.value))} className="split-range"/>
                  <span className="split-label">Test</span>
                </div>
                <div className="split-bar">
                  <div className="split-train" style={{width:`${100-testSize}%`}}>Train {100-testSize}%</div>
                  <div className="split-test"  style={{width:`${testSize}%`}}>Test {testSize}%</div>
                </div>
                <div className="split-nums">
                  <span>{Math.round(dataset.rows * (100 - testSize) / 100)} exemples</span>
                  <span>{Math.round(dataset.rows * testSize / 100)} exemples</span>
                </div>
                {Math.round(dataset.rows * (100 - testSize) / 100) < 20 && (
                  <div style={{color:"#ef4444", fontSize:"0.8rem", marginTop:"0.5rem"}}>
                    ⚠️ Moins de 20 exemples d'entraînement — résultats peu fiables
                  </div>
                )}
              </div>
            </div>

            {/* EXPÉRIENCE */}
            <div className="config-card">
              <div className="config-card-num">6</div>
              <div className="config-card-body">
                <div className="config-card-title">🔬 Nom de l'Expérience MLflow</div>
                <div className="config-card-desc">Les runs seront groupés sous ce nom dans MLflow</div>
                <input className="config-select" value={expName}
                  onChange={e => setExpName(e.target.value)} placeholder="ex: RunWalk_v1"/>
              </div>
            </div>
          </div>

          <div className="config-right">
            {/* FEATURES */}
            <div className="config-card">
              <div className="config-card-num">4</div>
              <div className="config-card-body">
                <div className="config-card-title">
                  Features d'entrée
                  <span className="feat-count">{features.length}/{availableFeatures.length}</span>
                </div>
                <div className="config-card-desc">Variables utilisées pour la prédiction</div>
                {availableFeatures.length === 0 && target && (
                  <div style={{color:"#f59e0b", padding:"0.5rem"}}>
                    ⚠️ Aucune feature disponible — choisissez d'abord une variable cible
                  </div>
                )}
                <div className="features-grid">
                  {availableFeatures.map(col => {
                    const isNum  = numericCols.includes(col);
                    const hasNull = nullPerCol[col] > 0;
                    // FIX: Indiquer si la feature est aussi potentiellement un ID
                    const isId   = colAnalysis[col]?.looksLikeId;
                    return (
                      <div key={col}
                        className={`feature-chip ${features.includes(col) ? "feat-active" : "feat-inactive"}`}
                        onClick={() => toggleFeat(col)}
                        title={isId ? "Colonne ID — peu utile comme feature" : hasNull ? `${nullPerCol[col]} valeurs manquantes` : ""}
                      >
                        <span className="feat-type">{isNum ? "123" : "Abc"}</span>
                        <span className="feat-name">{col}</span>
                        {hasNull && <span style={{fontSize:"0.7rem"}}>⚠️</span>}
                        {isId    && <span style={{fontSize:"0.7rem"}}>🆔</span>}
                        {features.includes(col) ? "✓" : "+"}
                      </div>
                    );
                  })}
                </div>
                <div className="feat-actions">
                  <button className="feat-btn" onClick={() => setFeatures([...availableFeatures])}>
                    Tout sélectionner
                  </button>
                  <button className="feat-btn" onClick={() => {
                    const numOnly = availableFeatures.filter(c => numericCols.includes(c));
                    if (numOnly.length > 0) setFeatures(numOnly);
                    else onNotif?.("Aucune colonne numérique disponible", "warning");
                  }}>
                    Numériques seulement
                  </button>
                  {/* FIX: Bouton pour exclure les IDs */}
                  <button className="feat-btn" onClick={() => {
                    const noIds = availableFeatures.filter(c => !colAnalysis[c]?.looksLikeId);
                    if (noIds.length > 0) setFeatures(noIds);
                    else onNotif?.("Toutes les colonnes semblent être des IDs", "warning");
                  }}>
                    Sans IDs
                  </button>
                </div>
              </div>
            </div>

            {/* ALGOS */}
            {taskType && (
              <div className="config-card">
                <div className="config-card-num">5</div>
                <div className="config-card-body">
                  <div className="config-card-title">Algorithmes à tester</div>
                  <div className="config-card-desc">Sélectionnez un ou plusieurs algorithmes à comparer</div>
                  <div className="algo-select-grid">
                    {algos.map(a => (
                      <div key={a.id}
                        className={`algo-select-card ${selectedAlgos.includes(a.id) ? "algo-sel-active" : ""}`}
                        onClick={() => toggleAlgo(a.id)}>
                        <div className="algo-sel-icon">{a.icon}</div>
                        <div>
                          <div className="algo-sel-name">{a.name}</div>
                          <div className="algo-sel-desc">{a.desc}</div>
                        </div>
                        <div className="algo-sel-check">{selectedAlgos.includes(a.id) ? "✓" : ""}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="config-actions">
          <button className="btn-back" onClick={onBack}>← Retour</button>
          <button
            className={`btn-launch ${canGo ? "btn-launch-active" : "btn-launch-disabled"}`}
            disabled={!canGo}
            onClick={handleLaunch}
          >
            🚀 Lancer l'analyse ML
          </button>
        </div>
        {validationError && (
          <div className="config-hint">{validationError}</div>
        )}
      </div>
    </div>
  );
}