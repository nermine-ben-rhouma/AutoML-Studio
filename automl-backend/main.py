# ============================================================
# AutoML Studio — Back-End FastAPI + MLflow
# ============================================================
from fastapi import Depends, FastAPI, UploadFile, File, HTTPException, Request
from cryptography.fernet import Fernet
import base64
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Annotated, List, Optional, Dict, Any
import pandas as pd
import numpy as np
import io, time, json, os, re

# ML
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import (LogisticRegression, LinearRegression,
                                   Ridge, Lasso)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, roc_auc_score, confusion_matrix,
                              mean_squared_error, mean_absolute_error, r2_score)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import warnings
import logging
from pathlib import Path
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
logging.getLogger("mlflow").setLevel(logging.WARNING)

# MLflow
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

from mlflow_utils import (
    bootstrap_mlflow_client,
    cleanup_mlflow_store,
    disk_free_bytes,
    ensure_disk_space,
    prune_experiment_runs,
    prune_registered_model_versions,
    reset_mlflow_artifacts,
)
from report_generator import generate_model_report
from auth import (
    auth_router,
    get_current_user,
    init_default_admin,
    AUTH_DISABLED,
    JWT_SECRET,
    JWT_DEFAULT_SECRET,
)

CurrentUser = Annotated[dict, Depends(get_current_user)]

# ── CONFIG MLFLOW ────────────────────────────────────────────
_BACKEND_DIR = Path(__file__).resolve().parent
_DATASETS_DIR = _BACKEND_DIR / "datasets"
_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_MAX_RUNS = int(os.getenv("MLFLOW_MAX_RUNS", "20"))
MLFLOW_MAX_MODEL_VERSIONS = int(os.getenv("MLFLOW_MAX_MODEL_VERSIONS", "3"))
MLFLOW_REGISTER_MODELS = os.getenv("MLFLOW_REGISTER_MODELS", "false").lower() in ("1", "true", "yes")
# SVM/SVR sur grands jeux : sous-échantillonnage pour éviter blocages (15k+ lignes)
MAX_ROWS_FOR_SVM = int(os.getenv("MAX_ROWS_FOR_SVM", "5000"))

# Client MLflow (répare automatiquement une mlflow.db corrompue)
client, MLFLOW_TRACKING_URI = bootstrap_mlflow_client(MLFLOW_TRACKING_URI)

# ── CORS (front React : 3000 par défaut, 3001 si port occupé, etc.) ──
_DEFAULT_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
]
_cors_env = os.getenv("CORS_ORIGINS", "").strip()
CORS_ORIGINS = (
    [o.strip() for o in _cors_env.split(",") if o.strip()]
    if _cors_env
    else _DEFAULT_CORS_ORIGINS
)
# Tout port local si besoin (ex. npm start sur 3002)
CORS_ALLOW_ORIGIN_REGEX = os.getenv(
    "CORS_ALLOW_ORIGIN_REGEX",
    r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
)


def _log_sklearn_model(model, algo_id: str, algo_name: str, register: bool = False):
    """Log model (MLflow 2.13 : artifact_path, pas name)."""
    artifact_path = f"model_{algo_id}"
    kwargs: Dict[str, Any] = {"await_registration_for": 0}
    if register:
        safe = re.sub(r"[^a-zA-Z0-9_\-\. ]", "_", algo_name).replace(" ", "_")
        kwargs["registered_model_name"] = f"AutoML_{safe}"
    mlflow.sklearn.log_model(model, artifact_path=artifact_path, **kwargs)

# ── FASTAPI APP ──────────────────────────────────────────────
app = FastAPI(
    title="AutoML Studio API",
    description="API professionnelle ML avec MLflow tracking",
    version="1.0.0"
)

# Import metrics middleware
from metrics import MetricsMiddleware

app.add_middleware(MetricsMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_origin_regex=CORS_ALLOW_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)


@app.on_event("startup")
def _startup_auth():
    init_default_admin()
    if AUTH_DISABLED:
        logging.warning("AUTH_DISABLED=true — routes ML sans JWT")
    elif JWT_SECRET == JWT_DEFAULT_SECRET:
        logging.warning("JWT_SECRET par défaut — définissez une clé secrète en production")
    try:
        mlflow.search_experiments(max_results=1)
        logging.info("MLflow connecté — %s", MLFLOW_TRACKING_URI)
    except Exception as exc:
        logging.error("MLflow non disponible après démarrage: %s", exc)

# ── MODÈLES DISPONIBLES ───────────────────────────────────────
CLASSIFIERS = {
    "rf":  lambda: RandomForestClassifier(n_estimators=100, random_state=42),
    "svm": lambda: SVC(probability=True, random_state=42),
    "lr":  lambda: LogisticRegression(max_iter=1000, random_state=42),
    "knn": lambda: KNeighborsClassifier(n_neighbors=5),
    "dt":  lambda: DecisionTreeClassifier(random_state=42),
    "nb":  lambda: GaussianNB(),
}

REGRESSORS = {
    "rf":    lambda: RandomForestRegressor(n_estimators=100, random_state=42),
    "lr":    lambda: LinearRegression(),
    "ridge": lambda: Ridge(alpha=1.0),
    "lasso": lambda: Lasso(alpha=1.0),
    "svr":   lambda: SVR(kernel="rbf"),
    "dt":    lambda: DecisionTreeRegressor(random_state=42),
}

ALGO_NAMES = {
    "rf":    "Random Forest",
    "svm":   "SVM",
    "lr":    "Logistic/Linear Reg.",
    "knn":   "KNN",
    "dt":    "Decision Tree",
    "nb":    "Naive Bayes",
    "ridge": "Ridge Regression",
    "lasso": "Lasso Regression",
    "svr":   "SVR",
}

_ALGO_NAME_TO_ID = {v.lower(): k for k, v in ALGO_NAMES.items()}


def _row_val(row, key: str, default: str = "") -> str:
    if key not in row.index:
        return default
    v = row[key]
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return default
    return str(v)


def _resolve_algo_id(row) -> str:
    """ID court (rf, svm…) pour l'UI — depuis params.algorithm ou tags."""
    aid = _row_val(row, "params.algorithm")
    if aid:
        return aid
    name = _row_val(row, "tags.algorithm_name")
    if name:
        found = _ALGO_NAME_TO_ID.get(name.lower())
        if found:
            return found
    run_name = _row_val(row, "tags.mlflow.runName")
    if run_name:
        for algo_id, label in ALGO_NAMES.items():
            if run_name.startswith(label):
                return algo_id
    return "unknown"


def _format_mlflow_run_row(row) -> Dict[str, Any]:
    algo_id = _resolve_algo_id(row)
    algo_name = _row_val(row, "tags.algorithm_name") or ALGO_NAMES.get(algo_id, algo_id)
    task_type = _row_val(row, "tags.task_type") or _row_val(row, "params.task_type")
    target = _row_val(row, "tags.target_column") or _row_val(row, "params.target")
    metrics = {
        k.replace("metrics.", ""): round(v, 4)
        for k, v in row.items()
        if k.startswith("metrics.") and not pd.isna(v)
    }
    return {
        "run_id":     row["run_id"],
        "run_name":   _row_val(row, "tags.mlflow.runName"),
        "algo":       algo_id,
        "algo_name":  algo_name,
        "task_type":  task_type,
        "target":     target,
        "status":     row["status"],
        "start_time": str(row["start_time"]),
        "metrics":    metrics,
    }

# ── STOCKAGE EN MÉMOIRE ──────────────────────────────────────
datasets_store: Dict[str, pd.DataFrame] = {}


_DATA_DIR = _BACKEND_DIR / "data"
_DATA_DIR.mkdir(parents=True, exist_ok=True)
_KEY_FILE = _DATA_DIR / "encryption.key"
MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE", str(50 * 1024 * 1024)))  # 50 Mo par défaut
_MAX_DATASET_ROWS = int(os.getenv("MAX_DATASET_ROWS", "500000"))  # Protection DoS : 500k lignes
_MAX_DATASET_COLS = int(os.getenv("MAX_DATASET_COLS", "500"))      # Protection DoS : 500 colonnes
_TEXT_FORMATS = {".csv", ".tsv", ".json"}                          # Formats texte (seuls vérifiés pour null bytes)


def _get_encryption_cipher() -> Fernet:
    key = os.getenv("DATA_ENCRYPTION_KEY")
    if not key:
        if _KEY_FILE.exists():
            key = _KEY_FILE.read_text(encoding="utf-8").strip()
        else:
            key = Fernet.generate_key().decode("utf-8")
            try:
                _KEY_FILE.write_text(key, encoding="utf-8")
                # Restreindre les permissions : lecture/écriture owner uniquement (Unix/Linux)
                if os.name != "nt":
                    os.chmod(_KEY_FILE, 0o600)
            except OSError as e:
                logging.warning("Could not save encryption key: %s", e)
    
    try:
        decoded = base64.urlsafe_b64decode(key.encode("utf-8"))
        if len(decoded) != 32:
            raise ValueError("La clé doit faire 32 octets après décodage base64.")
        return Fernet(key.encode("utf-8"))
    except Exception as e:
        logging.error("Clé DATA_ENCRYPTION_KEY invalide : %s. Utilisation d'une clé temporaire.", e)
        fallback_key = Fernet.generate_key()
        return Fernet(fallback_key)


def _persist_dataset(dataset_id: str, df: pd.DataFrame) -> None:
    try:
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode("utf-8")
        
        cipher = _get_encryption_cipher()
        encrypted_bytes = cipher.encrypt(csv_bytes)
        
        path = _DATASETS_DIR / f"{dataset_id}.csv"
        path.write_bytes(encrypted_bytes)
    except OSError as e:
        logging.warning("Could not persist dataset %s: %s", dataset_id, e)


def _load_dataset(dataset_id: str) -> pd.DataFrame:
    if dataset_id in datasets_store:
        return datasets_store[dataset_id].copy()
    path = _DATASETS_DIR / f"{dataset_id}.csv"
    if path.exists():
        file_bytes = path.read_bytes()
        
        # Tentative de déchiffrement
        cipher = _get_encryption_cipher()
        try:
            decrypted_bytes = cipher.decrypt(file_bytes)
            csv_text = decrypted_bytes.decode("utf-8")
        except Exception:
            # Échec du déchiffrement -> rétrocompatibilité avec les datasets non chiffrés
            logging.info("Le déchiffrement a échoué pour %s. Essai de lecture en texte clair.", dataset_id)
            try:
                csv_text = file_bytes.decode("utf-8")
            except UnicodeDecodeError:
                raise HTTPException(500, "Le fichier de données est corrompu ou illisible.")
        
        buf = io.StringIO(csv_text)
        df = pd.read_csv(buf)
        datasets_store[dataset_id] = df
        return df.copy()
    raise HTTPException(
        404,
        "Dataset non trouvé — refaites l'upload ou l'entraînement (session serveur expirée).",
    )


def _slim_train_results(results: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    if not results:
        return None
    slim = []
    for r in results:
        if not r.get("algo_id"):
            continue
        entry = {
            "algo_id": str(r.get("algo_id")),
            "algo_name": r.get("algo_name"),
            "accuracy": r.get("accuracy"),
            "r2": r.get("r2"),
            "f1": r.get("f1"),
            "rmse": r.get("rmse"),
        }
        slim.append({k: v for k, v in entry.items() if v is not None})
    return slim or None

# ── SCHEMAS ──────────────────────────────────────────────────
class TrainRequest(BaseModel):
    dataset_id: str
    task_type: str                    # classification | regression
    target: str
    features: List[str]
    algorithms: List[str]
    test_size: float = 0.2
    experiment_name: Optional[str] = "AutoML_Studio"

class PreprocessRequest(BaseModel):
    dataset_id: str
    drop_duplicates: bool = True
    handle_nulls: str = "mean"        # mean | median | mode | drop
    remove_outliers: bool = False
    outlier_method: str = "iqr"       # iqr | zscore
    normalize: bool = False
    drop_cols: List[str] = []

class PredictRequest(BaseModel):
    run_id: str
    data: List[Dict[str, Any]]

class ReportRequest(BaseModel):
    dataset_id: str
    task_type: str
    target: str
    features: List[str] = Field(default_factory=list)
    test_size: float = 0.2
    best_algo_id: str
    best_algo_name: Optional[str] = None
    train_results: Optional[List[Dict[str, Any]]] = None
    dataset_name: Optional[str] = None
    experiment_name: Optional[str] = None

# ── ROUTES ───────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "AutoML Studio API v1.0", "mlflow_uri": MLFLOW_TRACKING_URI}

@app.get("/health")
def health():
    # Public — pas de JWT
    return {
        "status": "ok",
        "mlflow": "connected",
        "tracking_uri": MLFLOW_TRACKING_URI,
        "disk_free_mb": round(disk_free_bytes() / (1024 * 1024), 2),
    }


@app.get("/metrics")
def metrics():
    """Endpoint public pour Prometheus. Actualise la télémétrie avant de l'exporter."""
    from metrics import update_active_users_count, update_infra_metrics
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    from fastapi import Response
    
    update_active_users_count()
    update_infra_metrics()
    
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
        headers={"Cache-Control": "no-cache"}
    )


@app.post("/maintenance/cleanup")
def maintenance_cleanup(
    user: CurrentUser,
    keep_runs: int = MLFLOW_MAX_RUNS,
    keep_versions: int = MLFLOW_MAX_MODEL_VERSIONS,
):
    """Supprime les anciens runs MLflow et versions du registry pour libérer de l'espace."""
    try:
        return cleanup_mlflow_store(client, keep_runs, keep_versions)
    except OSError as e:
        raise HTTPException(507, str(e)) from e


@app.post("/maintenance/reset")
def maintenance_reset(user: CurrentUser):
    """Réinitialise mlruns/ et mlflow.db + supprime tous les datasets chiffrés sur disque."""
    try:
        result = reset_mlflow_artifacts()
        # Vider le cache mémoire des datasets
        datasets_store.clear()
        # Supprimer physiquement tous les datasets chiffrés
        deleted_datasets = []
        for f in _DATASETS_DIR.glob("*.csv"):
            try:
                f.unlink()
                deleted_datasets.append(f.name)
            except OSError as exc:
                logging.warning("Impossible de supprimer le dataset %s : %s", f.name, exc)
        result["datasets_deleted"] = len(deleted_datasets)
        result["datasets_store_cleared"] = True
        return result
    except OSError as e:
        raise HTTPException(507, str(e)) from e

# ── FORMATS SUPPORTÉS ────────────────────────────────────────
SUPPORTED_EXTENSIONS = {
    ".csv":     "csv",
    ".tsv":     "tsv",
    ".json":    "json",
    ".xlsx":    "excel",
    ".xls":     "excel",
    ".parquet": "parquet",
}

def read_dataframe(content: bytes, filename: str) -> pd.DataFrame:
    ext = os.path.splitext(filename.lower())[1]
    fmt = SUPPORTED_EXTENSIONS.get(ext)
    if fmt is None:
        raise ValueError(
            f"Format '{ext}' non supporté. Formats acceptés : "
            + ", ".join(SUPPORTED_EXTENSIONS.keys())
        )
    buf = io.BytesIO(content)
    if fmt == "csv":     return pd.read_csv(buf)
    if fmt == "tsv":     return pd.read_csv(buf, sep="\t")
    if fmt == "json":    return pd.read_json(buf)
    if fmt == "excel":   return pd.read_excel(buf)
    if fmt == "parquet": return pd.read_parquet(buf)
    raise ValueError(f"Format non géré : {fmt}")

def validate_and_sanitize_csv(content: bytes, filename: str) -> pd.DataFrame:
    ext = os.path.splitext(filename.lower())[1]

    # 1. Octets nuls — vérification uniquement pour les formats TEXTE (CSV/TSV/JSON)
    #    Les formats binaires (Excel, Parquet) contiennent légitimement des octets nuls.
    if ext in _TEXT_FORMATS and b'\x00' in content:
        raise ValueError(
            "Le fichier contient des octets nuls (null bytes). "
            "Il s'agit probablement d'un fichier binaire renommé en .csv."
        )

    # 2. Lecture du DataFrame pour valider la structure
    try:
        df = read_dataframe(content, filename)
    except Exception as e:
        raise ValueError(f"Structure de fichier invalide ou corrompue : {e}")

    # 3. Vérification de la présence de données minimales
    if df.empty or len(df) < 1:
        raise ValueError("Le fichier importé est vide.")
    if len(df.columns) < 1:
        raise ValueError("Le fichier doit contenir au moins une colonne de données.")

    # 4. Limites de dimensions — protection contre les dénis de service (DoS)
    if len(df) > _MAX_DATASET_ROWS:
        raise ValueError(
            f"Le fichier dépasse la limite de {_MAX_DATASET_ROWS:,} lignes "
            f"({len(df):,} lignes détectées). Réduisez votre dataset avant l'import."
        )
    if len(df.columns) > _MAX_DATASET_COLS:
        raise ValueError(
            f"Le fichier dépasse la limite de {_MAX_DATASET_COLS} colonnes "
            f"({len(df.columns)} colonnes détectées)."
        )

    # 5. Assainissement des noms de colonnes (caractères dangereux)
    sanitized_cols = [
        re.sub(r'[\x00\r\n<>]', '', str(c)).strip()[:255] or f"col_{i}"
        for i, c in enumerate(df.columns)
    ]
    # Résolution des doublons après assainissement
    seen: Dict[str, int] = {}
    deduped_cols = []
    for col in sanitized_cols:
        if col in seen:
            seen[col] += 1
            deduped_cols.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            deduped_cols.append(col)
    df.columns = deduped_cols

    # 6. Protection contre les injections de formules CSV (OWASP CSV Injection)
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].apply(
            lambda x: f"'{x}" if isinstance(x, str) and len(x) > 0 and x[0] in ('=', '+', '-', '@') else x
        )

    return df


# ── UPLOAD DATASET ───────────────────────────────────────────
@app.post("/upload")
async def upload_dataset(request: Request, user: CurrentUser, file: UploadFile = File(...)):
    # 1. Vérification de l'en-tête Content-Length (si disponible)
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            413,
            f"Fichier trop volumineux. La taille maximale autorisée est de {MAX_UPLOAD_SIZE // (1024*1024)} Mo."
        )

    # 2. Lecture sécurisée par chunks pour éviter la saturation de la mémoire vive
    content = bytearray()
    chunk_size = 1024 * 1024  # 1 Mo
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        content.extend(chunk)
        if len(content) > MAX_UPLOAD_SIZE:
            raise HTTPException(
                413,
                f"Fichier trop volumineux. La taille maximale autorisée est de {MAX_UPLOAD_SIZE // (1024*1024)} Mo."
            )

    # 3. Validation de sécurité et assainissement
    try:
        df = validate_and_sanitize_csv(bytes(content), file.filename)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(400, f"Erreur de validation du fichier : {str(e)}")

    dataset_id = f"ds_{int(time.time())}"
    datasets_store[dataset_id] = df
    _persist_dataset(dataset_id, df)

    num_cols   = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols   = df.select_dtypes(exclude=np.number).columns.tolist()
    null_count = int(df.isnull().sum().sum())
    preview    = df.head(10).fillna("").to_dict(orient="records")

    return {
        "dataset_id":       dataset_id,
        "filename":         file.filename,
        "rows":             len(df),
        "columns":          df.columns.tolist(),
        "numeric_cols":     num_cols,
        "categorical_cols": cat_cols,
        "null_count":       null_count,
        "preview":          preview,
        "dtypes":           {col: str(dtype) for col, dtype in df.dtypes.items()},
    }

# ── PREPROCESS DATASET ───────────────────────────────────────
@app.post("/preprocess")
def preprocess(req: PreprocessRequest, user: CurrentUser):
    try:
        df = _load_dataset(req.dataset_id)
    except HTTPException:
        raise HTTPException(404, "Dataset non trouvé — veuillez le re-uploader")
    rows_before        = len(df)
    duplicates_removed = 0
    nulls_handled      = 0
    outliers_removed   = 0

    cols_to_drop = [c for c in req.drop_cols if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    if req.drop_duplicates:
        before = len(df)
        df = df.drop_duplicates()
        duplicates_removed = before - len(df)

    nulls_handled = int(df.isnull().sum().sum())
    if req.handle_nulls == "drop":
        df = df.dropna()
    else:
        num_cols = df.select_dtypes(include=np.number).columns
        cat_cols = df.select_dtypes(exclude=np.number).columns
        if req.handle_nulls == "mean":
            df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
        elif req.handle_nulls == "median":
            df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        elif req.handle_nulls == "mode":
            df[num_cols] = df[num_cols].fillna(df[num_cols].mode().iloc[0])
        for col in cat_cols:
            df[col] = df[col].fillna(
                df[col].mode().iloc[0] if not df[col].mode().empty else "unknown"
            )

    if req.remove_outliers:
        num_cols = df.select_dtypes(include=np.number).columns
        before   = len(df)
        if req.outlier_method == "iqr":
            mask = pd.Series([True] * len(df), index=df.index)
            for col in num_cols:
                Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                IQR = Q3 - Q1
                mask &= (df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)
            df = df[mask]
        elif req.outlier_method == "zscore":
            from scipy import stats as scipy_stats
            mask = pd.Series([True] * len(df), index=df.index)
            for col in num_cols:
                z = np.abs(scipy_stats.zscore(df[col].dropna()))
                outlier_idx = df[col].dropna().index[z > 3]
                mask[outlier_idx] = False
            df = df[mask]
        outliers_removed = before - len(df)

    if req.normalize:
        num_cols = df.select_dtypes(include=np.number).columns
        scaler   = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])

    datasets_store[req.dataset_id] = df
    _persist_dataset(req.dataset_id, df)

    return {
        "dataset_id":         req.dataset_id,
        "rows_before":        rows_before,
        "rows_after":         len(df),
        "columns":            df.columns.tolist(),
        "numeric_cols":       df.select_dtypes(include=np.number).columns.tolist(),
        "categorical_cols":   df.select_dtypes(exclude=np.number).columns.tolist(),
        "null_count":         int(df.isnull().sum().sum()),
        "duplicates_removed": duplicates_removed,
        "nulls_handled":      nulls_handled,
        "outliers_removed":   outliers_removed,
        "preview":            df.head(10).fillna("").to_dict(orient="records"),
    }

# ── HELPERS TRAIN ────────────────────────────────────────────
def prepare_features(X: pd.DataFrame) -> pd.DataFrame:
    """Encode toutes les colonnes catégorielles + remplace les NaN."""
    X = X.copy()
    for col in X.select_dtypes(exclude=np.number).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    for col in X.columns:
        if X[col].isnull().any():
            med = X[col].median()
            X[col] = X[col].fillna(med if pd.notna(med) else 0)
    return X

def prepare_target(y: pd.Series, task_type: str):
    """Valide et prépare la colonne cible selon le type de tâche."""
    target_encoder = None
    if task_type == "classification":
        n_unique = y.nunique()
        if pd.api.types.is_float_dtype(y) and n_unique > 20:
            raise HTTPException(
                400,
                f"La colonne cible contient {n_unique} valeurs numériques continues. "
                "Utilisez 'régression' comme type de tâche, ou choisissez une colonne "
                "avec des classes discrètes pour la classification."
            )
        if y.dtype == object:
            target_encoder = LabelEncoder()
            y = pd.Series(target_encoder.fit_transform(y), index=y.index)
        else:
            y = y.astype(int)
    else:
        y = pd.to_numeric(y, errors="coerce")
        if y.isnull().all():
            raise HTTPException(400, "La colonne cible ne contient aucune valeur numérique valide.")
        y = y.fillna(y.median())
    return y, target_encoder


def _subsample_xy(X: np.ndarray, y, max_rows: int):
    if len(X) <= max_rows:
        return X, y
    idx = np.random.RandomState(42).choice(len(X), max_rows, replace=False)
    y_sub = y.iloc[idx] if hasattr(y, "iloc") else y[idx]
    return X[idx], y_sub


def _safe_cv_score(model, X, y, cv_folds: int, scoring: str) -> float:
    """Évite un CV très lent sur SVM/SVR avec beaucoup de lignes."""
    name = type(model).__name__
    if len(X) > 3000 and name in ("SVC", "SVR"):
        return 0.0
    try:
        return float(cross_val_score(model, X, y, cv=cv_folds, scoring=scoring).mean())
    except Exception:
        return 0.0


# ── TRAIN ────────────────────────────────────────────────────
@app.post("/train")
def train(req: TrainRequest, user: CurrentUser):
    try:
        return _train_impl(req)
    except HTTPException:
        raise
    except Exception as exc:
        logging.exception("Erreur /train")
        raise HTTPException(500, f"Erreur d'entraînement : {exc}") from exc


def _train_impl(req: TrainRequest):
    try:
        df = _load_dataset(req.dataset_id)
    except HTTPException:
        raise HTTPException(404, "Dataset non trouvé — veuillez le re-uploader")

    if req.target not in df.columns:
        raise HTTPException(400, f"Colonne cible '{req.target}' introuvable")

    # Utiliser toutes les colonnes si aucune feature précisée
    features = req.features if req.features else [c for c in df.columns if c != req.target]
    missing  = [f for f in features if f not in df.columns]
    if missing:
        raise HTTPException(400, f"Features manquantes : {missing}")

    df = df.dropna(subset=[req.target])
    if len(df) < 10:
        raise HTTPException(400, "Pas assez de données après nettoyage (minimum 10 lignes).")

    datasets_store[req.dataset_id] = df
    _persist_dataset(req.dataset_id, df)

    X = prepare_features(df[features])
    y, target_encoder = prepare_target(df[req.target], req.task_type)

    # Adapter le nombre de folds cross-validation à la taille du dataset
    cv_folds = max(2, min(5, len(X) // 10))

    # Split stratifié si possible
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=req.test_size, random_state=42,
            stratify=y if req.task_type == "classification" else None
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=req.test_size, random_state=42
        )

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    try:
        exp = mlflow.get_experiment_by_name(req.experiment_name)
        experiment_id = exp.experiment_id if exp else mlflow.create_experiment(req.experiment_name)
    except Exception:
        experiment_id = mlflow.create_experiment(req.experiment_name)

    try:
        ensure_disk_space(_BACKEND_DIR)
    except OSError as e:
        raise HTTPException(507, str(e)) from e

    results = []
    algos   = CLASSIFIERS if req.task_type == "classification" else REGRESSORS
    register_models = MLFLOW_REGISTER_MODELS

    for algo_id in req.algorithms:
        if algo_id not in algos:
            continue

        model     = algos[algo_id]()
        algo_name = ALGO_NAMES.get(algo_id, algo_id)

        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=f"{algo_name}_{int(time.time())}"
        ) as run:
            run_id = run.info.run_id
            t0     = time.time()

            mlflow.log_param("algorithm",  algo_id)
            mlflow.log_param("task_type",  req.task_type)
            mlflow.log_param("target",     req.target)
            mlflow.log_param("n_features", len(features))
            mlflow.log_param("n_train",    len(X_train))
            mlflow.log_param("n_test",     len(X_test))
            mlflow.log_param("test_size",  req.test_size)
            mlflow.log_param("features",   json.dumps(features))

            X_fit, y_fit = X_train_sc, y_train
            if algo_id in ("svm", "svr") and len(X_train_sc) > MAX_ROWS_FOR_SVM:
                X_fit, y_fit = _subsample_xy(X_train_sc, y_train, MAX_ROWS_FOR_SVM)
                mlflow.log_param("train_subsample", len(X_fit))

            model.fit(X_fit, y_fit)
            elapsed = round(time.time() - t0, 3)

            y_pred       = model.predict(X_test_sc)
            y_pred_train = model.predict(X_train_sc)

            result = {"algo_id": algo_id, "algo_name": algo_name,
                      "run_id": run_id, "time": elapsed}

            if req.task_type == "classification":
                acc       = accuracy_score(y_test, y_pred)
                train_acc = accuracy_score(y_train, y_pred_train)
                f1        = f1_score(y_test, y_pred, average="weighted", zero_division=0)
                prec      = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                rec       = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                auc = 0.0
                try:
                    if len(np.unique(y_test)) == 2:
                        auc = roc_auc_score(y_test, model.predict_proba(X_test_sc)[:, 1])
                except Exception:
                    pass
                cv = _safe_cv_score(model, X_fit, y_fit, cv_folds, "accuracy")
                cm = confusion_matrix(y_test, y_pred).tolist()

                mlflow.log_metric("accuracy",        acc)
                mlflow.log_metric("train_accuracy",  train_acc)
                mlflow.log_metric("f1_score",        f1)
                mlflow.log_metric("precision",       prec)
                mlflow.log_metric("recall",          rec)
                mlflow.log_metric("auc_roc",         auc)
                mlflow.log_metric("cv_score",        cv)
                mlflow.log_metric("overfitting_gap", train_acc - acc)
                mlflow.log_metric("training_time",   elapsed)

                result.update({
                    "accuracy": round(acc, 4), "train_accuracy": round(train_acc, 4),
                    "f1": round(f1, 4), "precision": round(prec, 4),
                    "recall": round(rec, 4), "auc": round(auc, 4),
                    "cv_score": round(cv, 4),
                    "overfitting_gap": round(train_acc - acc, 4),
                    "confusion_matrix": cm,
                })

            else:
                mse       = mean_squared_error(y_test, y_pred)
                rmse      = float(np.sqrt(mse))
                mae       = mean_absolute_error(y_test, y_pred)
                r2        = r2_score(y_test, y_pred)
                train_mse = mean_squared_error(y_train, y_pred_train)
                train_r2  = r2_score(y_train, y_pred_train)
                cv = _safe_cv_score(model, X_fit, y_fit, cv_folds, "r2")

                mlflow.log_metric("rmse",          rmse)
                mlflow.log_metric("mae",           mae)
                mlflow.log_metric("r2",            r2)
                mlflow.log_metric("train_rmse",    float(np.sqrt(train_mse)))
                mlflow.log_metric("train_r2",      train_r2)
                mlflow.log_metric("cv_r2",         cv)
                mlflow.log_metric("training_time", elapsed)

                result.update({
                    "rmse": round(rmse, 4), "mae": round(mae, 4),
                    "r2": round(r2, 4), "train_rmse": round(float(np.sqrt(train_mse)), 4),
                    "train_r2": round(train_r2, 4), "cv_r2": round(cv, 4),
                })

            feat_importance = {}
            if hasattr(model, "feature_importances_"):
                feat_importance = dict(zip(features, [round(v, 4) for v in model.feature_importances_]))
                mlflow.log_dict(feat_importance, "feature_importance.json")
            result["feature_importance"] = feat_importance

            try:
                _log_sklearn_model(model, algo_id, algo_name, register=register_models)
            except (OSError, MlflowException) as e:
                err = str(e).lower()
                if "disk is full" in err or "no space" in err or "errno 28" in err:
                    raise HTTPException(
                        507,
                        "Disque plein : appelez POST /maintenance/cleanup ou /maintenance/reset, "
                        "puis libérez de l'espace sur le disque C:.",
                    ) from e
                raise HTTPException(500, f"Échec enregistrement MLflow : {e}") from e

            mlflow.set_tag("algorithm_name", algo_name)
            mlflow.set_tag("task_type",      req.task_type)
            mlflow.set_tag("target_column",  req.target)

            results.append(result)

    if not results:
        raise HTTPException(400, "Aucun algorithme valide sélectionné.")

    try:
        prune_experiment_runs(client, experiment_id, keep_last=MLFLOW_MAX_RUNS)
        if register_models:
            prune_registered_model_versions(client, keep_versions=MLFLOW_MAX_MODEL_VERSIONS)
    except Exception:
        pass

    best = max(results, key=lambda r: r.get("accuracy", 0) if req.task_type == "classification" else r.get("r2", -999))

    return {
        "status":          "success",
        "experiment_name": req.experiment_name,
        "task_type":       req.task_type,
        "target":          req.target,
        "features_used":   features,
        "n_train":         len(X_train),
        "n_test":          len(X_test),
        "results":         results,
        "best_model":      best["algo_id"],
        "mlflow_ui":       "http://localhost:5000",
    }


@app.post("/report")
def generate_report(req: ReportRequest, user: CurrentUser):
    """Rapport d'analyse adapté au meilleur modèle (questions 1 à 5)."""
    if not (req.best_algo_id or "").strip():
        raise HTTPException(400, "best_algo_id requis (meilleur modèle après entraînement).")
    logging.info(
        "Report request: dataset=%s target=%s algo=%s n_features=%s",
        req.dataset_id, req.target, req.best_algo_id, len(req.features or []),
    )
    df = _load_dataset(req.dataset_id)
    if req.target not in df.columns:
        raise HTTPException(
            400,
            f"Colonne cible '{req.target}' introuvable. Colonnes disponibles : {list(df.columns)[:12]}",
        )
    features = [f for f in (req.features or []) if f in df.columns and f != req.target]
    if not features:
        features = [c for c in df.columns if c != req.target]
    try:
        report = generate_model_report(
            df=df,
            target=req.target,
            features=features,
            task_type=req.task_type,
            test_size=req.test_size,
            best_algo_id=req.best_algo_id,
            best_algo_name=req.best_algo_name,
            train_results=_slim_train_results(req.train_results),
            dataset_name=req.dataset_name,
            experiment_name=req.experiment_name,
        )
        if report.get("project"):
            report["project"]["dataset_id"] = req.dataset_id
        if not report.get("sections"):
            raise HTTPException(500, "Rapport vide généré par le serveur.")
        return report
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        logging.exception("Report generation failed")
        raise HTTPException(500, f"Échec génération du rapport : {e}") from e

# ── EXPERIMENTS ──────────────────────────────────────────────
@app.get("/experiments")
def get_experiments(user: CurrentUser):
    try:
        exps = mlflow.search_experiments()
        return [{"id": e.experiment_id, "name": e.name,
                 "lifecycle": e.lifecycle_stage} for e in exps]
    except Exception:
        return []

@app.get("/experiments/{experiment_name}/runs")
def get_runs(experiment_name: str, user: CurrentUser):
    try:
        exp = mlflow.get_experiment_by_name(experiment_name)
        if not exp:
            return []
        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], order_by=["start_time DESC"])
        return [_format_mlflow_run_row(row) for _, row in runs.iterrows()]
    except Exception as e:
        raise HTTPException(500, str(e))

# ── MODELS REGISTRY ──────────────────────────────────────────
@app.get("/models")
def get_registered_models(user: CurrentUser):
    try:
        models = client.search_registered_models()
        result = []
        for m in models:
            versions = client.get_latest_versions(m.name)
            result.append({
                "name":     m.name,
                "versions": [{"version": v.version, "stage": v.current_stage,
                               "run_id": v.run_id} for v in versions],
            })
        return result
    except Exception:
        return []

# ── PREDICT ──────────────────────────────────────────────────
@app.post("/predict")
def predict(req: PredictRequest, user: CurrentUser):
    try:
        run = client.get_run(req.run_id)
        algo_tag = run.data.tags.get("algorithm_name", "")
        algo_id = next((k for k, v in ALGO_NAMES.items() if v == algo_tag), None)
        artifact = f"model_{algo_id}" if algo_id else None
        if not artifact:
            for name in ("model_rf", "model_svm", "model_lr", "model_knn", "model_dt", "model_nb",
                         "model_ridge", "model_lasso", "model_svr"):
                try:
                    mlflow.sklearn.load_model(f"runs:/{req.run_id}/{name}")
                    artifact = name
                    break
                except Exception:
                    continue
        if not artifact:
            raise ValueError("Aucun artefact modèle trouvé pour ce run.")
        model = mlflow.sklearn.load_model(f"runs:/{req.run_id}/{artifact}")
        df        = pd.DataFrame(req.data)
        preds     = model.predict(df).tolist()
        return {"predictions": preds, "run_id": req.run_id}
    except Exception as e:
        raise HTTPException(500, f"Erreur de prédiction : {str(e)}")

# ── STATS GLOBALES ───────────────────────────────────────────
@app.get("/stats")
def get_stats(user: CurrentUser):
    try:
        exps = mlflow.search_experiments()
        all_runs = []
        for e in exps:
            df = mlflow.search_runs(experiment_ids=[e.experiment_id], max_results=10_000)
            for _, row in df.iterrows():
                all_runs.append(_format_mlflow_run_row(row))

        by_algo: Dict[str, int] = {}
        finished = 0
        best_accuracy = 0.0
        best_r2 = -999.0
        for r in all_runs:
            aid = r.get("algo") or "unknown"
            by_algo[aid] = by_algo.get(aid, 0) + 1
            if r.get("status") == "FINISHED":
                finished += 1
            m = r.get("metrics") or {}
            if m.get("accuracy") is not None:
                best_accuracy = max(best_accuracy, float(m["accuracy"]))
            if m.get("r2") is not None:
                best_r2 = max(best_r2, float(m["r2"]))

        return {
            "total_experiments": len(exps),
            "total_runs":        len(all_runs),
            "runs_finished":     finished,
            "total_datasets":    len(datasets_store),
            "runs_by_algo":      by_algo,
            "best_accuracy":     round(best_accuracy, 4) if best_accuracy else None,
            "best_r2":           round(best_r2, 4) if best_r2 > -999 else None,
            "mlflow_uri":        MLFLOW_TRACKING_URI,
            "mlflow_ui":         "http://localhost:5000",
        }
    except Exception:
        return {
            "total_experiments": 0,
            "total_runs": 0,
            "runs_finished": 0,
            "total_datasets": len(datasets_store),
            "runs_by_algo": {},
            "best_accuracy": None,
            "best_r2": None,
            "mlflow_uri": MLFLOW_TRACKING_URI,
            "mlflow_ui": "http://localhost:5000",
        }