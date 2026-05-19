# ============================================================
# AutoML Studio — Back-End FastAPI + MLflow
# ============================================================
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
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
from mlflow.sklearn import SERIALIZATION_FORMAT_SKOPS
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

from mlflow_utils import (
    cleanup_mlflow_store,
    disk_free_bytes,
    ensure_disk_space,
    prune_experiment_runs,
    prune_registered_model_versions,
    reset_mlflow_artifacts,
)

# ── CONFIG MLFLOW ────────────────────────────────────────────
_BACKEND_DIR = Path(__file__).resolve().parent
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_MAX_RUNS = int(os.getenv("MLFLOW_MAX_RUNS", "20"))
MLFLOW_MAX_MODEL_VERSIONS = int(os.getenv("MLFLOW_MAX_MODEL_VERSIONS", "3"))
MLFLOW_REGISTER_MODELS = os.getenv("MLFLOW_REGISTER_MODELS", "false").lower() in ("1", "true", "yes")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


def _log_sklearn_model(model, algo_id: str, algo_name: str, register: bool = False):
    """Log model with current MLflow API (name + skops, optional registry)."""
    kwargs = {
        "name": f"model_{algo_id}",
        "serialization_format": SERIALIZATION_FORMAT_SKOPS,
        "await_registration_for": 0,
    }
    if register:
        safe = re.sub(r"[^a-zA-Z0-9_\-\. ]", "_", algo_name).replace(" ", "_")
        kwargs["registered_model_name"] = f"AutoML_{safe}"
    mlflow.sklearn.log_model(model, **kwargs)

# ── FASTAPI APP ──────────────────────────────────────────────
app = FastAPI(
    title="AutoML Studio API",
    description="API professionnelle ML avec MLflow tracking",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    "lr":    "Logistic_Linear_Regression",
    "knn":   "KNN",
    "dt":    "Decision Tree",
    "nb":    "Naive Bayes",
    "ridge": "Ridge Regression",
    "lasso": "Lasso Regression",
    "svr":   "SVR",
}

# ── STOCKAGE EN MÉMOIRE ──────────────────────────────────────
datasets_store: Dict[str, pd.DataFrame] = {}

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

# ── ROUTES ───────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "AutoML Studio API v1.0", "mlflow_uri": MLFLOW_TRACKING_URI}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "mlflow": "connected",
        "tracking_uri": MLFLOW_TRACKING_URI,
        "disk_free_mb": round(disk_free_bytes() / (1024 * 1024), 2),
    }


@app.post("/maintenance/cleanup")
def maintenance_cleanup(keep_runs: int = MLFLOW_MAX_RUNS, keep_versions: int = MLFLOW_MAX_MODEL_VERSIONS):
    """Supprime les anciens runs MLflow et versions du registry pour libérer de l'espace."""
    try:
        return cleanup_mlflow_store(client, keep_runs, keep_versions)
    except OSError as e:
        raise HTTPException(507, str(e)) from e


@app.post("/maintenance/reset")
def maintenance_reset():
    """Réinitialise mlruns/ et mlflow.db (toutes les expériences seront perdues)."""
    try:
        return reset_mlflow_artifacts()
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

# ── UPLOAD DATASET ───────────────────────────────────────────
@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    content = await file.read()
    try:
        df = read_dataframe(content, file.filename)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(400, f"Erreur lecture du fichier : {str(e)}")

    dataset_id = f"ds_{int(time.time())}"
    datasets_store[dataset_id] = df

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
def preprocess(req: PreprocessRequest):
    if req.dataset_id not in datasets_store:
        raise HTTPException(404, "Dataset non trouvé — veuillez le re-uploader")

    df = datasets_store[req.dataset_id].copy()
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

# ── TRAIN ────────────────────────────────────────────────────
@app.post("/train")
def train(req: TrainRequest):
    if req.dataset_id not in datasets_store:
        raise HTTPException(404, "Dataset non trouvé — veuillez le re-uploader")

    df = datasets_store[req.dataset_id].copy()

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

            model.fit(X_train_sc, y_train)
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
                try:
                    cv = cross_val_score(model, X_train_sc, y_train, cv=cv_folds, scoring="accuracy").mean()
                except Exception:
                    cv = 0.0
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
                try:
                    cv = cross_val_score(model, X_train_sc, y_train, cv=cv_folds, scoring="r2").mean()
                except Exception:
                    cv = 0.0

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

# ── EXPERIMENTS ──────────────────────────────────────────────
@app.get("/experiments")
def get_experiments():
    try:
        exps = mlflow.search_experiments()
        return [{"id": e.experiment_id, "name": e.name,
                 "lifecycle": e.lifecycle_stage} for e in exps]
    except Exception:
        return []

@app.get("/experiments/{experiment_name}/runs")
def get_runs(experiment_name: str):
    try:
        exp = mlflow.get_experiment_by_name(experiment_name)
        if not exp:
            return []
        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], order_by=["start_time DESC"])
        result = []
        for _, row in runs.iterrows():
            result.append({
                "run_id":     row["run_id"],
                "run_name":   row.get("tags.mlflow.runName", ""),
                "algo":       row.get("tags.algorithm_name", ""),
                "task_type":  row.get("tags.task_type", ""),
                "target":     row.get("tags.target_column", ""),
                "status":     row["status"],
                "start_time": str(row["start_time"]),
                "metrics":    {k.replace("metrics.", ""): round(v, 4)
                               for k, v in row.items()
                               if k.startswith("metrics.") and not pd.isna(v)},
            })
        return result
    except Exception as e:
        raise HTTPException(500, str(e))

# ── MODELS REGISTRY ──────────────────────────────────────────
@app.get("/models")
def get_registered_models():
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
def predict(req: PredictRequest):
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
def get_stats():
    try:
        exps       = mlflow.search_experiments()
        total_runs = sum(len(mlflow.search_runs(experiment_ids=[e.experiment_id])) for e in exps)
        return {
            "total_experiments": len(exps),
            "total_runs":        total_runs,
            "total_datasets":    len(datasets_store),
            "mlflow_ui":         "http://localhost:5000",
        }
    except Exception:
        return {"total_experiments": 0, "total_runs": 0, "total_datasets": len(datasets_store)}