# ============================================================
# AutoML Studio — Back-End FastAPI + MLflow
# Compatible Windows + Linux + macOS
# ============================================================
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import io, time, json, os, tempfile  # FIX: tempfile pour cross-platform

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

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# ── CONFIG MLFLOW ──────────────────────────────────────────
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

# ── FASTAPI APP ────────────────────────────────────────────
app = FastAPI(
    title="AutoML Studio API",
    description="API ML professionnelle avec MLflow tracking",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── MODÈLES ────────────────────────────────────────────────
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
    "lr":    "Logistic/Linear Regression",
    "knn":   "KNN",
    "dt":    "Decision Tree",
    "nb":    "Naive Bayes",
    "ridge": "Ridge Regression",
    "lasso": "Lasso Regression",
    "svr":   "SVR",
}

# Stockage en mémoire
datasets_store: Dict[str, pd.DataFrame] = {}
preprocessors_store: Dict[str, Dict] = {}

# ── SCHEMAS ────────────────────────────────────────────────
class TrainRequest(BaseModel):
    dataset_id: str
    task_type: str
    target: str
    features: List[str]
    algorithms: List[str]
    test_size: float = 0.2
    experiment_name: Optional[str] = "AutoML_Studio"

class PredictRequest(BaseModel):
    run_id: str
    data: List[Dict[str, Any]]

# ── HELPERS DATA ───────────────────────────────────────────
def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie un DataFrame :
    - Normalise les noms de colonnes
    - Convertit les colonnes object contenant des nombres
    - Remplace inf/-inf par NaN
    """
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]

    for col in df.columns:
        if df[col].dtype == object:
            # FIX pandas 2.0 : remplacer errors='ignore' par try/except
            try:
                converted = pd.to_numeric(df[col])
                df[col] = converted
            except (ValueError, TypeError):
                pass  # Colonne vraiment catégorielle, on la laisse

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


def encode_features(X: pd.DataFrame, encoders: Dict = None, fit: bool = True) -> tuple:
    """
    Encode les colonnes catégorielles avec LabelEncoder.
    fit=True  → crée et entraîne les encoders (phase train)
    fit=False → utilise les encoders existants (phase predict)
    """
    if encoders is None:
        encoders = {}

    X = X.copy()

    for col in X.columns:
        if X[col].dtype == object or str(X[col].dtype) == "category":
            X[col] = X[col].astype(str).fillna("__missing__")
            if fit:
                le = LabelEncoder()
                unique_vals = list(X[col].unique()) + ["__unknown__"]
                le.fit(unique_vals)
                encoders[col] = le
            else:
                le = encoders.get(col)
                if le is None:
                    le = LabelEncoder()
                    le.fit(list(X[col].unique()) + ["__unknown__"])
                    encoders[col] = le
                else:
                    known = set(le.classes_)
                    X[col] = X[col].apply(lambda v: v if v in known else "__unknown__")
            X[col] = le.transform(X[col])
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    return X, encoders


def fill_missing(X: pd.DataFrame) -> pd.DataFrame:
    """Remplit les NaN avec la médiane (ou 0 si médiane indisponible)."""
    for col in X.columns:
        if X[col].isnull().any():
            median_val = X[col].median()
            X[col] = X[col].fillna(0 if pd.isna(median_val) else median_val)
    return X


def safe_cv_folds(n_samples: int, n_classes: int = 1, task: str = "classification") -> int:
    """Calcule un nombre de folds CV sécurisé selon la taille du dataset."""
    if task == "classification":
        max_folds = n_samples // max(n_classes, 1)
    else:
        max_folds = n_samples // 2
    return max(2, min(5, max_folds))


def get_or_create_experiment(name: str) -> str:
    """Récupère ou crée une expérience MLflow de manière idempotente."""
    exp = mlflow.get_experiment_by_name(name)
    if exp is not None:
        if exp.lifecycle_stage == "deleted":
            client.restore_experiment(exp.experiment_id)
        return exp.experiment_id
    try:
        return mlflow.create_experiment(name)
    except Exception:
        exp = mlflow.get_experiment_by_name(name)
        if exp:
            return exp.experiment_id
        raise


# ── ROUTES ─────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "AutoML Studio API v2.1", "mlflow_uri": MLFLOW_TRACKING_URI}


@app.get("/health")
def health():
    return {"status": "ok", "mlflow": "connected", "tracking_uri": MLFLOW_TRACKING_URI}


# ── UPLOAD ─────────────────────────────────────────────────
@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith((".csv", ".tsv", ".txt")):
        raise HTTPException(400, "Seuls les fichiers CSV/TSV sont acceptés")

    content = await file.read()
    try:
        sample = content[:2048].decode("utf-8", errors="replace")
        sep = "," if sample.count(",") >= sample.count(";") else ";"
        df = pd.read_csv(io.BytesIO(content), sep=sep)
        if len(df.columns) == 1:
            df = pd.read_csv(io.BytesIO(content), sep="\t")
    except Exception as e:
        raise HTTPException(400, f"Erreur lecture CSV : {str(e)}")

    df = prepare_dataframe(df)

    dataset_id = f"ds_{int(time.time())}"
    datasets_store[dataset_id] = df

    num_cols  = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols  = df.select_dtypes(exclude=np.number).columns.tolist()
    null_count = int(df.isnull().sum().sum())

    preview_df = df.head(10).copy()
    preview_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    preview = preview_df.where(pd.notnull(preview_df), None).to_dict(orient="records")

    unique_counts = {col: int(df[col].nunique()) for col in df.columns}
    null_per_col  = {col: int(df[col].isnull().sum()) for col in df.columns}

    return {
        "dataset_id":       dataset_id,
        "filename":         file.filename,
        "rows":             len(df),
        "columns":          df.columns.tolist(),
        "numeric_cols":     num_cols,
        "categorical_cols": cat_cols,
        "null_count":       null_count,
        "null_per_col":     null_per_col,
        "unique_counts":    unique_counts,
        "preview":          preview,
        "dtypes":           {col: str(dtype) for col, dtype in df.dtypes.items()},
    }


# ── TRAIN ──────────────────────────────────────────────────
@app.post("/train")
def train(req: TrainRequest):

    # 1. Récupérer le dataset
    if req.dataset_id not in datasets_store:
        raise HTTPException(404, "Dataset non trouvé — veuillez re-uploader le fichier")

    df = datasets_store[req.dataset_id].copy()

    # Normaliser les noms de colonnes
    req_features = [f.strip().replace(" ", "_") for f in req.features]
    req_target   = req.target.strip().replace(" ", "_")

    # 2. Vérifications colonnes
    if req_target not in df.columns:
        raise HTTPException(
            400,
            f"Colonne cible '{req_target}' introuvable. "
            f"Colonnes disponibles : {df.columns.tolist()}"
        )
    missing_feats = [f for f in req_features if f not in df.columns]
    if missing_feats:
        raise HTTPException(
            400,
            f"Features manquantes : {missing_feats}. "
            f"Colonnes disponibles : {df.columns.tolist()}"
        )

    # 3. Préparation
    df = df.dropna(subset=[req_target])
    if len(df) < 10:
        raise HTTPException(
            400,
            f"Dataset trop petit après suppression des NaN cible : "
            f"{len(df)} lignes. Minimum requis : 10."
        )

    X = df[req_features].copy()
    y = df[req_target].copy()

    # Encodage target
    target_encoder = None
    if req.task_type == "classification":
        if y.dtype == object or y.dtype.name == "category":
            target_encoder = LabelEncoder()
            y = pd.Series(target_encoder.fit_transform(y.astype(str)), index=y.index)
        else:
            y = y.astype(int)
    else:
        y = pd.to_numeric(y, errors="coerce")
        if y.isnull().any():
            mask = y.notna()
            X = X.loc[mask]
            y = y.loc[mask]

    # Encodage features
    X, feature_encoders = encode_features(X, fit=True)
    X = fill_missing(X)

    if X.isnull().any().any() or np.isinf(X.values).any():
        raise HTTPException(
            400,
            "Des valeurs invalides persistent après nettoyage. Vérifiez votre dataset."
        )

    # 4. Validation de la target (classification)
    if req.task_type == "classification":
        unique_vals = np.unique(y)
        n_classes   = len(unique_vals)
        n_samples   = len(y)

        if n_classes < 2:
            raise HTTPException(
                400,
                f"La colonne '{req_target}' ne contient qu'une seule classe ({unique_vals[0]}). "
                "Choisissez une autre colonne cible."
            )

        ratio_unique = n_classes / n_samples
        if n_classes > 20 and ratio_unique > 0.5:
            raise HTTPException(
                400,
                f"La colonne '{req_target}' contient {n_classes} classes différentes "
                f"({ratio_unique*100:.0f}% de valeurs uniques) — elle ressemble à un identifiant. "
                "Pour la classification, choisissez une colonne avec peu de catégories (ex: 'activity', 'label')."
            )

        min_class_count = int(pd.Series(y).value_counts().min())
        if min_class_count < 2:
            raise HTTPException(
                400,
                f"La colonne '{req_target}' contient une classe avec seulement {min_class_count} exemple(s). "
                "Filtrez vos données ou choisissez une autre colonne cible."
            )
    else:
        n_classes = 1

    # 5. Split train/test
    min_test_samples = max(2, n_classes)
    actual_test_size = max(req.test_size, min_test_samples / len(y))
    actual_test_size = min(actual_test_size, 0.4)

    try:
        stratify = y if req.task_type == "classification" else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=actual_test_size, random_state=42, stratify=stratify
        )
    except ValueError:
        # Fallback sans stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=actual_test_size, random_state=42
        )

    # 6. Normalisation
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # 7. Expérience MLflow
    experiment_id = get_or_create_experiment(req.experiment_name)

    algos   = CLASSIFIERS if req.task_type == "classification" else REGRESSORS
    results = []
    n_cv    = safe_cv_folds(len(X_train), n_classes, req.task_type)

    for algo_id in req.algorithms:
        if algo_id not in algos:
            continue

        # KNN adaptatif
        if algo_id == "knn" and req.task_type == "classification":
            min_cls = int(pd.Series(y_train).value_counts().min())
            n_neighbors = max(1, min(5, min_cls - 1))
            model = KNeighborsClassifier(n_neighbors=n_neighbors)
        else:
            model = algos[algo_id]()

        algo_name = ALGO_NAMES.get(algo_id, algo_id)

        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=f"{algo_name}_{int(time.time())}"
        ) as run:
            run_id = run.info.run_id
            t0     = time.time()

            mlflow.log_param("algorithm",  algo_id)
            mlflow.log_param("task_type",  req.task_type)
            mlflow.log_param("target",     req_target)
            mlflow.log_param("n_features", len(req_features))
            mlflow.log_param("n_train",    len(X_train))
            mlflow.log_param("n_test",     len(X_test))
            mlflow.log_param("test_size",  actual_test_size)
            mlflow.log_param("features",   json.dumps(req_features))
            mlflow.log_param("n_cv_folds", n_cv)

            # 8. Entraînement
            try:
                model.fit(X_train_sc, y_train)
            except Exception as e:
                mlflow.set_tag("status", "failed")
                mlflow.set_tag("error",  str(e))
                continue

            elapsed      = round(time.time() - t0, 3)
            y_pred       = model.predict(X_test_sc)
            y_pred_train = model.predict(X_train_sc)

            result = {
                "algo_id":   algo_id,
                "algo_name": algo_name,
                "run_id":    run_id,
                "time":      elapsed,
            }

            # 9. Métriques
            if req.task_type == "classification":
                acc       = accuracy_score(y_test, y_pred)
                train_acc = accuracy_score(y_train, y_pred_train)
                f1        = f1_score(y_test, y_pred, average="weighted", zero_division=0)
                prec      = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                rec       = recall_score(y_test, y_pred, average="weighted", zero_division=0)

                try:
                    cv = cross_val_score(model, X_train_sc, y_train, cv=n_cv, scoring="accuracy").mean()
                except Exception:
                    cv = acc

                try:
                    if n_classes == 2:
                        y_proba = model.predict_proba(X_test_sc)[:, 1]
                        auc = roc_auc_score(y_test, y_proba)
                    else:
                        auc = 0.0
                except Exception:
                    auc = 0.0

                labels = sorted(np.unique(np.concatenate([y_test, y_pred])))
                cm     = confusion_matrix(y_test, y_pred, labels=labels).tolist()

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
                    "accuracy":         round(acc, 4),
                    "train_accuracy":   round(train_acc, 4),
                    "f1":               round(f1, 4),
                    "precision":        round(prec, 4),
                    "recall":           round(rec, 4),
                    "auc":              round(auc, 4),
                    "cv_score":         round(float(cv), 4),
                    "overfitting_gap":  round(train_acc - acc, 4),
                    "confusion_matrix": cm,
                })

            else:  # régression
                mse       = mean_squared_error(y_test, y_pred)
                rmse      = float(np.sqrt(mse))
                mae       = float(mean_absolute_error(y_test, y_pred))
                r2        = float(r2_score(y_test, y_pred))
                train_mse = mean_squared_error(y_train, y_pred_train)
                train_r2  = float(r2_score(y_train, y_pred_train))

                try:
                    cv = float(cross_val_score(model, X_train_sc, y_train, cv=n_cv, scoring="r2").mean())
                except Exception:
                    cv = r2

                mlflow.log_metric("rmse",          rmse)
                mlflow.log_metric("mae",           mae)
                mlflow.log_metric("r2",            r2)
                mlflow.log_metric("train_rmse",    float(np.sqrt(train_mse)))
                mlflow.log_metric("train_r2",      train_r2)
                mlflow.log_metric("cv_r2",         cv)
                mlflow.log_metric("training_time", elapsed)

                result.update({
                    "rmse":       round(rmse, 4),
                    "mae":        round(mae, 4),
                    "r2":         round(r2, 4),
                    "train_rmse": round(float(np.sqrt(train_mse)), 4),
                    "train_r2":   round(train_r2, 4),
                    "cv_r2":      round(cv, 4),
                })

            # Feature importance
            feat_importance = {}
            if hasattr(model, "feature_importances_"):
                feat_importance = {
                    k: round(float(v), 4)
                    for k, v in zip(req_features, model.feature_importances_)
                }
                mlflow.log_dict(feat_importance, "feature_importance.json")

            result["feature_importance"] = feat_importance

            # Stocker le preprocessor en mémoire pour /predict
            preprocessors_store[run_id] = {
                "scaler":           scaler,
                "feature_encoders": feature_encoders,
                "target_encoder":   target_encoder,
                "features":         req_features,
                "task_type":        req.task_type,
            }

            # Sauvegarder le modèle dans MLflow
            try:
                mlflow.sklearn.log_model(
                    model,
                    f"model_{algo_id}",
                    registered_model_name=f"AutoML_{algo_name.replace(' ', '_')}",
                )
            except Exception:
                mlflow.sklearn.log_model(model, f"model_{algo_id}")

            # FIX WINDOWS: utiliser tempfile au lieu de /tmp/ (inexistant sur Windows)
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pkl", prefix=f"scaler_{run_id}_")
            try:
                os.close(tmp_fd)  # Fermer le fd avant d'écrire avec joblib
                joblib.dump(
                    {
                        "scaler":           scaler,
                        "feature_encoders": feature_encoders,
                        "target_encoder":   target_encoder,
                        "features":         req_features,
                    },
                    tmp_path,
                )
                mlflow.log_artifact(tmp_path, "preprocessor")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

            mlflow.set_tag("algorithm_name", algo_name)
            mlflow.set_tag("task_type",      req.task_type)
            mlflow.set_tag("target_column",  req_target)
            mlflow.set_tag("status",         "success")

            results.append(result)

    if not results:
        raise HTTPException(500, "Aucun modèle n'a pu être entraîné. Vérifiez vos données.")

    if req.task_type == "classification":
        best = max(results, key=lambda r: r["accuracy"])
    else:
        best = max(results, key=lambda r: r["r2"])

    return {
        "status":          "success",
        "experiment_name": req.experiment_name,
        "experiment_id":   experiment_id,
        "task_type":       req.task_type,
        "target":          req_target,
        "n_train":         len(X_train),
        "n_test":          len(X_test),
        "n_cv_folds":      n_cv,
        "results":         results,
        "best_model":      best["algo_id"],
        "mlflow_ui":       "http://localhost:5000",
    }


# ── PREDICT ────────────────────────────────────────────────
@app.post("/predict")
def predict(req: PredictRequest):
    if req.run_id not in preprocessors_store:
        raise HTTPException(
            404,
            f"Run '{req.run_id}' non trouvé ou serveur redémarré. Ré-entraînez le modèle."
        )

    prep      = preprocessors_store[req.run_id]
    scaler    = prep["scaler"]
    encoders  = prep["feature_encoders"]
    features  = prep["features"]
    task_type = prep["task_type"]
    t_enc     = prep["target_encoder"]

    try:
        X_pred = pd.DataFrame(req.data)
    except Exception as e:
        raise HTTPException(400, f"Format de données invalide : {e}")

    X_pred.columns = [str(c).strip().replace(" ", "_") for c in X_pred.columns]

    missing = [f for f in features if f not in X_pred.columns]
    if missing:
        raise HTTPException(400, f"Colonnes manquantes dans les données : {missing}")

    X_pred = X_pred[features].copy()
    X_pred.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_pred, _ = encode_features(X_pred, encoders=encoders, fit=False)
    X_pred    = fill_missing(X_pred)
    X_pred_sc = scaler.transform(X_pred)

    try:
        artifacts = client.list_artifacts(req.run_id)
        model_art = next((a for a in artifacts if a.path.startswith("model_")), None)
        if model_art is None:
            raise HTTPException(404, "Modèle non trouvé dans MLflow")
        model = mlflow.sklearn.load_model(f"runs:/{req.run_id}/{model_art.path}")
    except Exception as e:
        raise HTTPException(500, f"Erreur chargement modèle : {e}")

    preds = model.predict(X_pred_sc)

    if task_type == "classification" and t_enc is not None:
        preds_decoded = t_enc.inverse_transform(preds.astype(int))
    else:
        preds_decoded = preds

    probas = None
    if hasattr(model, "predict_proba"):
        try:
            probas = model.predict_proba(X_pred_sc).tolist()
        except Exception:
            pass

    return {
        "predictions":   preds_decoded.tolist(),
        "probabilities": probas,
        "n_predictions": len(preds),
    }


# ── EXPERIMENTS ────────────────────────────────────────────
@app.get("/experiments")
def get_experiments():
    try:
        exps = mlflow.search_experiments()
        return [
            {"id": e.experiment_id, "name": e.name, "lifecycle": e.lifecycle_stage}
            for e in exps
        ]
    except Exception:
        return []


@app.get("/experiments/{experiment_name}/runs")
def get_runs(experiment_name: str):
    try:
        exp = mlflow.get_experiment_by_name(experiment_name)
        if not exp:
            return []
        runs = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["start_time DESC"]
        )
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
                "metrics": {
                    k.replace("metrics.", ""): round(float(v), 4)
                    for k, v in row.items()
                    if k.startswith("metrics.") and not pd.isna(v)
                },
            })
        return result
    except Exception as e:
        raise HTTPException(500, str(e))


# ── MODEL REGISTRY ─────────────────────────────────────────
@app.get("/models")
def get_registered_models():
    try:
        models = client.search_registered_models()
        return [
            {
                "name": m.name,
                "versions": [
                    {"version": v.version, "stage": v.current_stage, "run_id": v.run_id}
                    for v in client.get_latest_versions(m.name)
                ],
            }
            for m in models
        ]
    except Exception:
        return []


# ── STATS ──────────────────────────────────────────────────
@app.get("/stats")
def get_stats():
    try:
        exps       = mlflow.search_experiments()
        total_runs = sum(
            len(mlflow.search_runs(experiment_ids=[e.experiment_id]))
            for e in exps
        )
        return {
            "total_experiments": len(exps),
            "total_runs":        total_runs,
            "total_datasets":    len(datasets_store),
            "mlflow_ui":         "http://localhost:5000",
        }
    except Exception:
        return {
            "total_experiments": 0,
            "total_runs":        0,
            "total_datasets":    len(datasets_store),
        }