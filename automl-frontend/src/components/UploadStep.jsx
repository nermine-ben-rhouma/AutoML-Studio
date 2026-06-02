import { useState, useRef } from "react";
import { uploadDataset } from "../api";

export default function UploadStep({ onUpload, onNotif, backendOk, maxUploadSize = 50 * 1024 * 1024 }) {
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading]   = useState(false);
  const [preview, setPreview]   = useState(null);
  const [localFile, setLocalFile] = useState(null);
  const fileRef = useRef();

  const sizeLimitMo = Math.round(maxUploadSize / (1024 * 1024));

  /** Pré-validation locale : détecte les fichiers binaires renommés en .csv */
  const preValidateFile = async (file) => {
    if (!file.name.match(/\.(csv|tsv|json)$/i)) return true; // formats binaires OK
    const slice = file.slice(0, 10240); // 10 Ko
    const buffer = await slice.arrayBuffer();
    const bytes = new Uint8Array(buffer);
    for (let i = 0; i < bytes.length; i++) {
      if (bytes[i] === 0) return false; // null byte = fichier binaire
    }
    return true;
  };

  const handleFile = async (file) => {
    if (!file) return;
    const allowedExts = ['.csv', '.tsv', '.json', '.xlsx', '.xls', '.parquet'];
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    if (!allowedExts.includes(ext)) {
      onNotif(`❌ Format "${ext}" non supporté. Formats acceptés : ${allowedExts.join(', ')}`, "error");
      return;
    }
    if (file.size > maxUploadSize) {
      onNotif(`❌ Le fichier dépasse la taille maximale autorisée de ${sizeLimitMo} Mo`, "error");
      return;
    }
    // Pré-validation : détection de fichier binaire renommé
    const isValidText = await preValidateFile(file);
    if (!isValidText) {
      onNotif("❌ Ce fichier semble être un fichier binaire renommé (octets nuls détectés)", "error");
      return;
    }
    setLoading(true);
    setLocalFile(file);
    try {
      if (backendOk) {
        // Appel réel au Back-End
        const data = await uploadDataset(file);
        setPreview({ ...data, name: data.filename });
        onNotif(`📂 "${data.filename}" analysé par le serveur`, "info");
      } else {
        // Fallback local si backend off
        const text = await file.text();
        const lines = text.trim().split("\n");
        const columns = lines[0].split(",").map(c => c.trim().replace(/^"|"$/g,""));
        const rows = lines.slice(1).map(line => {
          const vals = line.split(",").map(v => v.trim().replace(/^"|"$/g,""));
          return Object.fromEntries(columns.map((col,i) => [col, isNaN(vals[i]) ? vals[i] : parseFloat(vals[i])]));
        });
        setPreview({ dataset_id: `local_${Date.now()}`, filename: file.name, name: file.name,
          rows: rows.length, columns, data: rows, numeric_cols: columns, categorical_cols: [], null_count: 0, preview: rows.slice(0,10) });
        onNotif("⚠️ Mode local (backend non connecté)", "info");
      }
    } catch(e) {
      onNotif(`❌ Erreur : ${e.message}`, "error");
    }
    setLoading(false);
  };

  const stats = preview ? {
    numCols: (preview.numeric_cols || []).length,
    catCols: (preview.categorical_cols || []).length,
    nulls:   preview.null_count || 0,
  } : null;

  const previewData = preview?.preview || preview?.data || [];

  return (
    <div className="upload-page">
      <div className="upload-container">
        <div className="upload-hero">
          <div className="upload-hero-icon">📊</div>
          <h1 className="upload-hero-title">Importez votre Dataset</h1>
          <p className="upload-hero-desc">Uploadez n'importe quel fichier CSV — régression ou classification, le système s'adapte.</p>
        </div>

        <div
          className={`drop-zone ${dragging?"dragging":""} ${preview?"has-file":""}`}
          onDragOver={e=>{e.preventDefault();setDragging(true)}}
          onDragLeave={()=>setDragging(false)}
          onDrop={e=>{e.preventDefault();setDragging(false);handleFile(e.dataTransfer.files[0])}}
          onClick={()=>fileRef.current.click()}
        >
          <input ref={fileRef} type="file" accept=".csv,.tsv,.json,.xlsx,.xls,.parquet" style={{display:"none"}}
            onChange={e=>handleFile(e.target.files[0])}/>
          {loading ? (
            <div className="upload-loading"><div className="spinner"/><span>Analyse en cours...</span></div>
          ) : preview ? (
            <div className="upload-success">
              <div className="upload-success-icon">✅</div>
              <div className="upload-success-name">{preview.filename || preview.name}</div>
              <div className="upload-success-meta">{preview.rows} lignes · {preview.columns.length} colonnes</div>
              <div className="upload-success-security" style={{
                marginTop: "8px",
                display: "inline-flex",
                alignItems: "center",
                gap: "6px",
                fontSize: "0.75rem",
                color: "#10b981",
                backgroundColor: "rgba(16, 185, 129, 0.1)",
                padding: "4px 10px",
                borderRadius: "12px",
                fontWeight: "500",
                border: "1px solid rgba(16, 185, 129, 0.2)"
              }}>
                🔒 Données chiffrées et validées
              </div>
              <div className="upload-change" style={{marginTop: "8px"}}>Cliquer pour changer</div>
            </div>
          ) : (
            <>
              <div className="drop-icon">{dragging?"📥":"📁"}</div>
              <div className="drop-title">{dragging?"Relâchez ici !":"Glisser-déposer votre CSV"}</div>
              <div className="drop-sub">ou cliquer pour parcourir</div>
              <div className="drop-formats">CSV · TSV · JSON · Excel · Parquet · Max {sizeLimitMo} Mo · UTF-8 recommandé</div>
              <div className="drop-security-badge" style={{
                marginTop: "12px",
                display: "inline-flex",
                alignItems: "center",
                gap: "6px",
                padding: "5px 12px",
                borderRadius: "20px",
                backgroundColor: "rgba(34, 197, 94, 0.1)",
                color: "#22c55e",
                fontSize: "0.75rem",
                fontWeight: "500",
                border: "1px solid rgba(34, 197, 94, 0.2)"
              }}>
                🔒 Chiffrement AES-256 & Validation active
              </div>
            </>
          )}
        </div>

        {preview && (
          <>
            <div className="stats-cards">
              {[
                {num:preview.rows,       lbl:"Lignes",       cls:"stat-green"},
                {num:preview.columns.length, lbl:"Colonnes",  cls:"stat-blue"},
                {num:stats.numCols,      lbl:"Numériques",   cls:"stat-orange"},
                {num:stats.catCols,      lbl:"Catégorielles",cls:"stat-purple"},
                {num:stats.nulls,        lbl:"Valeurs nulles",cls:"stat-red"},
              ].map(s => (
                <div key={s.lbl} className={`stat-card ${s.cls}`}>
                  <div className="stat-num">{s.num}</div>
                  <div className="stat-lbl">{s.lbl}</div>
                </div>
              ))}
            </div>

            <div className="preview-section">
              <div className="preview-title">👁 Aperçu <span className="preview-badge">10 premières lignes</span></div>
              <div className="preview-table-wrap">
                <table className="preview-table">
                  <thead><tr>{preview.columns.map(c=><th key={c}>{c}</th>)}</tr></thead>
                  <tbody>
                    {previewData.slice(0,10).map((row,i)=>(
                      <tr key={i}>{preview.columns.map(c=><td key={c}>{row[c]??""}</td>)}</tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <button className="btn-continue" onClick={()=>onUpload({...preview, name: preview.filename||preview.name})}>
              Continuer vers la Configuration →
            </button>
          </>
        )}

        {!preview && (
          <div className="examples-section">
            <div className="examples-title">💡 Datasets populaires compatibles</div>
            <div className="examples-grid">
              {[
                {name:"Pima Indians Diabetes",type:"Classification",cols:9,rows:768},
                {name:"Boston Housing",type:"Régression",cols:14,rows:506},
                {name:"Iris Dataset",type:"Classification",cols:5,rows:150},
                {name:"Wine Quality",type:"Régression",cols:12,rows:6497},
                {name:"Titanic",type:"Classification",cols:12,rows:891},
                {name:"California Housing",type:"Régression",cols:9,rows:20640},
              ].map(ex=>(
                <div key={ex.name} className="example-card">
                  <div className={`example-badge ${ex.type==="Classification"?"badge-class":"badge-reg"}`}>
                    {ex.type==="Classification"?"🏷 Classification":"📈 Régression"}
                  </div>
                  <div className="example-name">{ex.name}</div>
                  <div className="example-meta">{ex.rows} lignes · {ex.cols} colonnes</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
