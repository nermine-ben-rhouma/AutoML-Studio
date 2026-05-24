/** Rapport style PDF (diabetes-ml-dashboard) — AutoML Studio */

function ChartImg({ src, alt }) {
  if (!src) return null;
  return <img className="report-chart-img" src={`data:image/png;base64,${src}`} alt={alt} />;
}

function ChartRow({ charts, alt = "graphique" }) {
  const list = (charts || []).filter(Boolean);
  if (!list.length) return null;
  return (
    <div className={`report-charts-row charts-${Math.min(list.length, 3)}`}>
      {list.map((c, i) => (
        <ChartImg key={i} src={c} alt={`${alt} ${i + 1}`} />
      ))}
    </div>
  );
}

export default function ReportPanel({ report, loading, error, bestModel, onClose, onRetry }) {
  const isClassif = report?.task_type === "classification";
  const p = report?.project;

  const downloadMarkdown = () => {
    if (!report) return;
    const lines = [`# ${report.title}`, report.subtitle || "", ""];
    report.sections.forEach((s) => {
      lines.push(`## ${s.title}`, "", s.answer || "", "");
    });
    const blob = new Blob([lines.join("\n")], { type: "text/markdown;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `rapport_automl_${report.best_algo_id}.md`;
    a.click();
  };

  const renderSection = (section) => {
    switch (section.id) {
      case "1":
        return (
          <>
            <p className="report-q-subtitle">{section.subtitle}</p>
            <ChartRow charts={section.charts} alt="Feature importance" />
            <h4 className="report-h4">Résultats</h4>
            <p className="report-table-caption">Tableau d&apos;importance des features :</p>
            {section.importance_table?.length > 0 && (
              <table className="report-table report-table-pdf">
                <thead>
                  <tr>
                    <th>Feature</th>
                    <th>Importance</th>
                    <th>Rang</th>
                  </tr>
                </thead>
                <tbody>
                  {section.importance_table.map((row) => (
                    <tr key={row.feature}>
                      <td>{row.feature}</td>
                      <td>{row.importance}</td>
                      <td style={{ color: row.rank_color, fontWeight: 800 }}>{row.rank}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
            <h4 className="report-h4">Analyse et Interprétation</h4>
            <ul className="report-interp-list">
              {section.interpretation?.map((item) => (
                <li key={item.feature}>
                  <span className="interp-feat">{item.feature}</span> ({item.percent}%) —{" "}
                  <strong>{item.rank_label} prédicteur</strong> :{" "}
                  <span dangerouslySetInnerHTML={{ __html: item.text.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>") }} />
                </li>
              ))}
            </ul>
          </>
        );

      case "2":
        return (
          <>
            <p className="report-q-subtitle">{section.subtitle}</p>
            <ChartRow charts={section.charts} alt="Stabilité" />
            <h4 className="report-h4">Résultats</h4>
            {section.stability_table?.rows?.length > 0 && (
              <table className="report-table report-table-pdf">
                <thead>
                  <tr>
                    <th>random_state</th>
                    <th>{isClassif ? "Accuracy" : "R²"}</th>
                    <th>{isClassif ? "F1-Score" : "R² test"}</th>
                    <th>Variation</th>
                  </tr>
                </thead>
                <tbody>
                  {section.stability_table.rows.map((row, i) => (
                    <tr key={i} className={row.random_state === "Moyenne" ? "row-mean" : ""}>
                      <td>{row.random_state}</td>
                      <td>{row.accuracy}</td>
                      <td>{row.f1}</td>
                      <td>{row.variation}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
            <h4 className="report-h4">Statistiques</h4>
            <p className={`report-stats report-stats-${section.stats_class || "low"}`}>
              {section.statistics}
            </p>
            {section.answer && <p className="report-response">{section.answer}</p>}
          </>
        );

      case "3":
        return (
          <>
            <p className="report-q-subtitle">{section.subtitle}</p>
            <ChartRow charts={section.charts} alt="Erreurs" />
            {section.error_summary_table?.length > 0 && (
              <>
                <h4 className="report-h4">Résumé des Erreurs</h4>
                <table className="report-table report-table-pdf">
                  <thead>
                    <tr>
                      <th>Type d&apos;Erreur</th>
                      <th>Nombre</th>
                      <th>Impact</th>
                    </tr>
                  </thead>
                  <tbody>
                    {section.error_summary_table.map((row) => (
                      <tr key={row.type} className={row.critical ? "row-critical" : ""}>
                        <td>{row.type}</td>
                        <td>
                          {row.count} ({row.pct})
                        </td>
                        <td>{row.impact}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </>
            )}
            <h4 className="report-h4">Exemples Détaillés</h4>
            {section.detailed_examples?.map((ex) => (
              <div key={ex.index ?? ex.title} className={`error-card error-${ex.title_class || "fp"}`}>
                <div className={`error-card-head head-${ex.title_class || "fp"}`}>{ex.title}</div>
                <p className="error-features">{ex.features_line}</p>
                {ex.probability_line && <p className="error-proba">{ex.probability_line}</p>}
                <p className="error-reality">{ex.reality}</p>
                <p className="error-analysis" dangerouslySetInnerHTML={{
                  __html: (ex.analysis || "").replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>"),
                }} />
              </div>
            ))}
            {section.answer && <p className="report-response">{section.answer}</p>}
          </>
        );

      case "4": {
        const cols = section.table_columns;
        return (
          <>
            <p className="report-q-subtitle">{section.subtitle}</p>
            <ChartRow charts={section.charts} alt="Biais variance" />
            <h4 className="report-h4">{section.table_title || "Tableau d'Analyse"}</h4>
            {section.table?.length > 0 && (
              <table className="report-table report-table-pdf report-table-bv">
                <thead>
                  <tr>
                    {cols ? (
                      cols.map((h) => <th key={h}>{h}</th>)
                    ) : (
                      <>
                        {(section.table_param_keys || []).map((k) => (
                          <th key={k}>{k}</th>
                        ))}
                        <th>{isClassif ? "Train Acc" : "Train R²"}</th>
                        <th>{isClassif ? "Test Acc" : "Test R²"}</th>
                        <th>Biais</th>
                        <th>Variance</th>
                        <th>Statut</th>
                      </>
                    )}
                  </tr>
                </thead>
                <tbody>
                  {section.table.map((row, i) => (
                    <tr key={i}>
                      {cols ? (
                        cols.map((h) => {
                          const map = {
                            "n_est": row.n_est ?? row.n_estimators,
                            "max_d": row.max_d ?? row.max_depth,
                            "Train Acc": row.train_accuracy,
                            "Test Acc": row.test_accuracy,
                            "Train R²": row.train_r2,
                            "Test R²": row.test_r2,
                            Biais: row.bias,
                            Variance: row.variance,
                            Statut: (
                              <span className={`statut-badge statut-${row.statut_class}`}>
                                {row.statut === "Overfitting" && "⚠ "}
                                {row.statut === "Underfitting" && "✕ "}
                                {row.statut === "Équilibré" && "✓ "}
                                {row.statut}
                              </span>
                            ),
                          };
                          return <td key={h}>{map[h] ?? row[h]}</td>;
                        })
                      ) : (
                        <>
                          {(section.table_param_keys || []).map((k) => (
                            <td key={k}>{row[k]}</td>
                          ))}
                          <td>{isClassif ? row.train_accuracy : row.train_r2}</td>
                          <td>{isClassif ? row.test_accuracy : row.test_r2}</td>
                          <td>{row.bias}</td>
                          <td>{row.variance}</td>
                          <td>
                            <span className={`statut-badge statut-${row.statut_class}`}>{row.statut}</span>
                          </td>
                        </>
                      )}
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
            {section.conclusions && (
              <div className="report-q4-answers">
                <p>{section.conclusions.overfitting}</p>
                <p>{section.conclusions.underfitting}</p>
                <p>{section.conclusions.balanced}</p>
              </div>
            )}
          </>
        );
      }

      default:
        return null;
    }
  };

  return (
    <div className="report-overlay" onClick={onClose}>
      <div className="report-modal report-modal-pdf" onClick={(e) => e.stopPropagation()}>
        <div className="report-header">
          <h2>📋 Rapport d&apos;analyse</h2>
          <button type="button" className="report-close" onClick={onClose}>
            ✕
          </button>
        </div>

        {loading && (
          <div className="report-loading">
            <div className="loading-icon">📊</div>
            <p>
              Génération du rapport PDF pour <strong>{bestModel?.algo_name}</strong>…
            </p>
            <p className="report-loading-hint">Graphiques + tableaux (30–60 s)</p>
            <div className="loading-bar-wrap" style={{ marginTop: 12 }}>
              <div className="loading-bar-fill report-indeterminate" />
            </div>
          </div>
        )}

        {error && !loading && (
          <div className="report-error">
            <p>❌ {error}</p>
            <button type="button" className="btn-report-generate" onClick={onRetry}>
              Réessayer
            </button>
          </div>
        )}

        {report && !loading && !error && (
          <div className="report-body report-pdf-body">
            <header className="report-pdf-cover">
              <div className="report-pdf-app">AutoML Studio</div>
              <h1>{report.title}</h1>
              <p className="report-pdf-sub">{report.subtitle}</p>
              <div className="report-pdf-meta">
                <span>📂 {p?.dataset_name}</span>
                <span>🎯 {report.target}</span>
                <span>{isClassif ? "🏷 Classification" : "📈 Régression"}</span>
                <span>🤖 {report.best_algo_name}</span>
              </div>
            </header>

            {report.sections.map((section) => (
              <section key={section.id} className="report-pdf-page">
                <h2 className="report-question-title">{section.title}</h2>
                {renderSection(section)}
              </section>
            ))}

            <div className="report-footer">
              <button type="button" className="btn-export-csv" onClick={downloadMarkdown}>
                📥 Markdown
              </button>
              <button type="button" className="btn-export-csv" onClick={onRetry}>
                🔄 Régénérer
              </button>
              <button type="button" className="btn-back" onClick={onClose}>
                Fermer
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
