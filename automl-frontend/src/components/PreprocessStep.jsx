import { useState } from "react";

const BASE_URL = "http://localhost:8000";

export default function PreprocessStep({ dataset, onDone, onBack, onNotif }) {
  const [options, setOptions] = useState({
    drop_duplicates: true,
    handle_nulls: "mean",       // mean | median | mode | drop
    remove_outliers: false,
    outlier_method: "iqr",      // iqr | zscore
    normalize: false,
    drop_cols: [],
  });
  const [loading, setLoading] = useState(false);
  const [result, setResult]   = useState(null);

  const toggle = (key) => setOptions(o => ({ ...o, [key]: !o[key] }));
  const set    = (key, val) => setOptions(o => ({ ...o, [key]: val }));

  const toggleCol = (col) => {
    setOptions(o => ({
      ...o,
      drop_cols: o.drop_cols.includes(col)
        ? o.drop_cols.filter(c => c !== col)
        : [...o.drop_cols, col],
    }));
  };

  const handleClean = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${BASE_URL}/preprocess`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataset_id: dataset.dataset_id, ...options }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || "Erreur serveur");
      }
      const data = await res.json();
      setResult(data);
      onNotif(`✅ Dataset nettoyé — ${data.rows_after} lignes restantes`, "success");
    } catch (e) {
      onNotif(`❌ ${e.message}`, "error");
    }
    setLoading(false);
  };

  const nullStrategy = [
    { value: "mean",   label: "Moyenne",  icon: "➗" },
    { value: "median", label: "Médiane",  icon: "📊" },
    { value: "mode",   label: "Mode",     icon: "🔁" },
    { value: "drop",   label: "Supprimer les lignes", icon: "🗑" },
  ];

  return (
    <div className="preprocess-page">
      <div className="preprocess-container">

        {/* Header */}
        <div className="pp-header">
          <div className="pp-header-icon">🧹</div>
          <div>
            <h2 className="pp-title">Prétraitement du Dataset</h2>
            <p className="pp-sub">
              <strong>{dataset.filename || dataset.name}</strong> &nbsp;·&nbsp;
              {dataset.rows} lignes &nbsp;·&nbsp; {dataset.columns.length} colonnes &nbsp;·&nbsp;
              <span className={dataset.null_count > 0 ? "warn-text" : "ok-text"}>
                {dataset.null_count} valeurs nulles
              </span>
            </p>
          </div>
        </div>

        <div className="pp-grid">

          {/* Colonne gauche — options */}
          <div className="pp-options">

            {/* Doublons */}
            <div className="pp-card">
              <div className="pp-card-head">
                <span className="pp-card-icon">🔁</span>
                <span className="pp-card-title">Doublons</span>
                <label className="toggle-switch">
                  <input type="checkbox" checked={options.drop_duplicates}
                    onChange={() => toggle("drop_duplicates")} />
                  <span className="toggle-slider" />
                </label>
              </div>
              <p className="pp-card-desc">Supprime les lignes identiques du dataset.</p>
            </div>

            {/* Valeurs nulles */}
            <div className="pp-card">
              <div className="pp-card-head">
                <span className="pp-card-icon">❓</span>
                <span className="pp-card-title">Valeurs nulles</span>
                <span className="pp-badge pp-badge-warn">{dataset.null_count} nulls</span>
              </div>
              <p className="pp-card-desc">Stratégie de remplacement des valeurs manquantes :</p>
              <div className="null-strategy-grid">
                {nullStrategy.map(s => (
                  <button
                    key={s.value}
                    className={`strategy-btn ${options.handle_nulls === s.value ? "active" : ""}`}
                    onClick={() => set("handle_nulls", s.value)}
                  >
                    <span>{s.icon}</span>
                    <span>{s.label}</span>
                  </button>
                ))}
              </div>
            </div>

            {/* Outliers */}
            <div className="pp-card">
              <div className="pp-card-head">
                <span className="pp-card-icon">📡</span>
                <span className="pp-card-title">Valeurs aberrantes</span>
                <label className="toggle-switch">
                  <input type="checkbox" checked={options.remove_outliers}
                    onChange={() => toggle("remove_outliers")} />
                  <span className="toggle-slider" />
                </label>
              </div>
              <p className="pp-card-desc">Détecte et supprime les outliers des colonnes numériques.</p>
              {options.remove_outliers && (
                <div className="outlier-methods">
                  {[
                    { value: "iqr",    label: "IQR (Q1-Q3)",     desc: "Robuste, recommandé" },
                    { value: "zscore", label: "Z-Score (±3σ)",   desc: "Distribution normale" },
                  ].map(m => (
                    <button
                      key={m.value}
                      className={`method-btn ${options.outlier_method === m.value ? "active" : ""}`}
                      onClick={() => set("outlier_method", m.value)}
                    >
                      <strong>{m.label}</strong>
                      <span>{m.desc}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* Normalisation */}
            <div className="pp-card">
              <div className="pp-card-head">
                <span className="pp-card-icon">📐</span>
                <span className="pp-card-title">Normalisation</span>
                <label className="toggle-switch">
                  <input type="checkbox" checked={options.normalize}
                    onChange={() => toggle("normalize")} />
                  <span className="toggle-slider" />
                </label>
              </div>
              <p className="pp-card-desc">Standardise les colonnes numériques (z-score, moyenne=0, σ=1).</p>
            </div>
          </div>

          {/* Colonne droite — colonnes à supprimer */}
          <div className="pp-cols-panel">
            <div className="pp-card pp-card-full">
              <div className="pp-card-head">
                <span className="pp-card-icon">🗂</span>
                <span className="pp-card-title">Colonnes à exclure</span>
                {options.drop_cols.length > 0 && (
                  <span className="pp-badge pp-badge-red">{options.drop_cols.length} sélectionnées</span>
                )}
              </div>
              <p className="pp-card-desc">Cliquez sur une colonne pour la marquer à supprimer.</p>
              <div className="cols-grid">
                {dataset.columns.map(col => {
                  const isNum = (dataset.numeric_cols || []).includes(col);
                  const isCat = (dataset.categorical_cols || []).includes(col);
                  const isDropped = options.drop_cols.includes(col);
                  return (
                    <button
                      key={col}
                      className={`col-tag ${isDropped ? "col-tag-dropped" : ""} ${isNum ? "col-tag-num" : "col-tag-cat"}`}
                      onClick={() => toggleCol(col)}
                      title={isNum ? "Numérique" : "Catégorielle"}
                    >
                      <span className="col-type-dot" />
                      <span>{col}</span>
                      {isDropped && <span className="col-x">✕</span>}
                    </button>
                  );
                })}
              </div>
              <div className="cols-legend">
                <span><span className="legend-dot legend-num" />Numérique</span>
                <span><span className="legend-dot legend-cat" />Catégorielle</span>
                <span><span className="legend-dot legend-drop" />Exclue</span>
              </div>
            </div>

            {/* Résumé des actions */}
            <div className="pp-card pp-summary">
              <div className="pp-card-head">
                <span className="pp-card-icon">📋</span>
                <span className="pp-card-title">Résumé des actions</span>
              </div>
              <ul className="summary-list">
                {options.drop_duplicates   && <li>✓ Suppression des doublons</li>}
                {options.handle_nulls === "drop"
                  ? <li>✓ Suppression des lignes avec nulls</li>
                  : <li>✓ Remplacement des nulls par la {options.handle_nulls === "mean" ? "moyenne" : options.handle_nulls === "median" ? "médiane" : "mode"}</li>
                }
                {options.remove_outliers   && <li>✓ Suppression outliers ({options.outlier_method.toUpperCase()})</li>}
                {options.normalize         && <li>✓ Normalisation (StandardScaler)</li>}
                {options.drop_cols.length > 0 && <li>✓ Suppression de {options.drop_cols.length} colonne(s)</li>}
              </ul>
            </div>
          </div>
        </div>

        {/* Résultat après nettoyage */}
        {result && (
          <div className="pp-result">
            <div className="pp-result-title">✅ Résultat du nettoyage</div>
            <div className="pp-result-stats">
              {[
                { label: "Avant",          val: result.rows_before,              cls: "" },
                { label: "Après",          val: result.rows_after,               cls: "stat-green" },
                { label: "Lignes supprimées", val: result.rows_before - result.rows_after, cls: "stat-red" },
                { label: "Doublons",       val: result.duplicates_removed,       cls: "stat-orange" },
                { label: "Nulls traités",  val: result.nulls_handled,            cls: "stat-blue" },
                { label: "Outliers",       val: result.outliers_removed ?? "—",  cls: "stat-purple" },
              ].map(s => (
                <div key={s.label} className={`result-stat ${s.cls}`}>
                  <div className="result-stat-val">{s.val}</div>
                  <div className="result-stat-lbl">{s.label}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Actions */}
        <div className="pp-actions">
          <button className="btn-back" onClick={onBack}>← Retour</button>
          <button
            className="btn-clean"
            onClick={handleClean}
            disabled={loading}
          >
            {loading ? <><span className="spinner-sm" /> Nettoyage...</> : "🧹 Nettoyer le dataset"}
          </button>
          <button
            className={`btn-continue ${!result ? "btn-disabled" : ""}`}
            onClick={() => result && onDone({ ...dataset, ...result })}
            disabled={!result}
          >
            Continuer vers la Configuration →
          </button>
        </div>

      </div>
    </div>
  );
}