"""Rapport d'analyse adapté au meilleur modèle — robuste et autonome."""
from __future__ import annotations

import base64
import io
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, r2_score

from report_visuals import (
    OVERFIT_THRESHOLD,
    STABILITY_SEEDS,
    build_error_summary,
    build_importance_table,
    build_interpretation_items,
    build_stability_table,
    bv_row_status,
    chart_bv_gap,
    chart_bv_stacked,
    chart_bv_train_test,
    chart_confusion_matrix,
    chart_errors_scatter,
    chart_feature_importance,
    chart_stability_bars,
    chart_stability_line,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False
    plt = None

ALGO_NAMES = {
    "rf": "Random Forest",
    "svm": "SVM",
    "lr": "Logistic / Linear Regression",
    "knn": "KNN",
    "dt": "Decision Tree",
    "nb": "Naive Bayes",
    "ridge": "Ridge Regression",
    "lasso": "Lasso Regression",
    "svr": "SVR",
}

CLASSIFICATION_ALGOS = {"rf", "svm", "lr", "knn", "dt", "nb"}
REGRESSION_ALGOS = {"rf", "lr", "ridge", "lasso", "svr", "dt"}

SEEDS = [42, 0, 99, 7, 13, 21, 77]
_REPORT_MAX_TRAIN = 800
_PERM_SAMPLE = 60
_PERM_REPEATS = 2

def _assignment_sections(task_type: str, model_name: str, algo_id: str) -> List[Dict[str, Any]]:
    """Titres style rapport PDF (diabetes-ml-dashboard)."""
    return [
        {
            "id": "1",
            "title": "Question 1 : Importance des Features",
            "subtitle": f"Q1 — Feature Importance ({model_name})",
        },
        {
            "id": "2",
            "title": "Question 2 : Stabilité des Prédictions",
            "subtitle": "Q2 — Stabilité des prédictions",
        },
        {
            "id": "3",
            "title": "Question 3 : Analyse des Erreurs",
            "subtitle": "Q3 — Analyse des Erreurs",
        },
        {
            "id": "4",
            "title": "Question 4 : Biais et Variance (Hyperparamètres)",
            "subtitle": "Q4 — Analyse Biais / Variance",
        },
    ]


def _json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, bool, int, float)):
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if np.isnan(v) or np.isinf(v) else v
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return str(obj)


def prepare_features_report(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for col in X.select_dtypes(exclude=np.number).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    for col in X.columns:
        if X[col].isnull().any():
            med = X[col].median()
            X[col] = X[col].fillna(med if pd.notna(med) else 0)
    return X


def prepare_target_report(y: pd.Series, task_type: str) -> Tuple[pd.Series, None]:
    y = y.copy()
    if task_type == "classification":
        if pd.api.types.is_float_dtype(y) and y.nunique() > 20:
            raise ValueError(
                f"Cible avec {y.nunique()} valeurs continues — utilisez la régression "
                "ou une colonne catégorielle."
            )
        if y.dtype == object or str(y.dtype) == "category":
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y.astype(str)), index=y.index)
        else:
            y = y.astype(int)
    else:
        y = pd.to_numeric(y, errors="coerce")
        if y.isnull().all():
            raise ValueError("Colonne cible sans valeurs numériques valides.")
        y = y.fillna(y.median())
    return y, None


def _fig_to_base64(fig) -> Optional[str]:
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii")
    except Exception as e:
        logger.warning("Chart failed: %s", e)
        try:
            plt.close(fig)
        except Exception:
            pass
        return None


def _feature_importance_chart(features: List[str], importances: np.ndarray, model_name: str) -> Optional[str]:
    if not _HAS_MPL or not features:
        return None
    order = np.argsort(importances)[::-1][: min(15, len(features))]
    labels = [features[int(i)] for i in order][::-1]
    values = importances[order][::-1]
    fig, ax = plt.subplots(figsize=(8, max(3, len(labels) * 0.35)))
    ax.barh(labels, values, color="#00a86b")
    ax.set_xlabel("Importance")
    ax.set_title(f"Importance des variables — {model_name}")
    return _fig_to_base64(fig)


def _cap_train_set(X_train, y_train):
    if len(X_train) <= _REPORT_MAX_TRAIN:
        return X_train, y_train
    idx = np.random.RandomState(42).choice(len(X_train), _REPORT_MAX_TRAIN, replace=False)
    if hasattr(X_train, "iloc"):
        return X_train.iloc[idx], y_train.iloc[idx]
    return X_train[idx], y_train[idx]


def _score_model(model, X_train, X_test, y_train, y_test, task_type: str) -> Tuple[float, float]:
    X_fit, y_fit = _cap_train_set(X_train, y_train)
    model.fit(X_fit, y_fit)
    if task_type == "classification":
        return (
            float(accuracy_score(y_train, model.predict(X_train))),
            float(accuracy_score(y_test, model.predict(X_test))),
        )
    return (
        float(r2_score(y_train, model.predict(X_train))),
        float(r2_score(y_test, model.predict(X_test))),
    )


def _create_model(algo_id: str, task_type: str, random_state: Optional[int] = 42, **params):
    rs = random_state if random_state is not None else 42
    if task_type == "classification":
        if algo_id not in CLASSIFICATION_ALGOS:
            raise ValueError(f"Algorithme {algo_id} invalide pour la classification.")
        factories = {
            "rf": lambda: RandomForestClassifier(
                n_estimators=int(params.get("n_estimators", 100)),
                max_depth=params.get("max_depth"),
                random_state=rs,
            ),
            "svm": lambda: SVC(
                probability=True, C=float(params.get("C", 1.0)),
                gamma=params.get("gamma", "scale"), random_state=rs, cache_size=200,
            ),
            "lr": lambda: LogisticRegression(
                max_iter=500, solver="lbfgs", C=float(params.get("C", 1.0)), random_state=rs,
            ),
            "knn": lambda: KNeighborsClassifier(n_neighbors=int(params.get("n_neighbors", 5))),
            "dt": lambda: DecisionTreeClassifier(
                max_depth=params.get("max_depth"),
                min_samples_split=int(params.get("min_samples_split", 2)),
                random_state=rs,
            ),
            "nb": lambda: GaussianNB(var_smoothing=float(params.get("var_smoothing", 1e-9))),
        }
    else:
        if algo_id not in REGRESSION_ALGOS:
            raise ValueError(f"Algorithme {algo_id} invalide pour la régression.")
        factories = {
            "rf": lambda: RandomForestRegressor(
                n_estimators=int(params.get("n_estimators", 100)),
                max_depth=params.get("max_depth"),
                random_state=rs,
            ),
            "lr": lambda: LinearRegression(fit_intercept=bool(params.get("fit_intercept", True))),
            "ridge": lambda: Ridge(alpha=float(params.get("alpha", 1.0))),
            "lasso": lambda: Lasso(alpha=float(params.get("alpha", 1.0)), max_iter=3000),
            "svr": lambda: SVR(kernel="rbf", C=float(params.get("C", 1.0))),
            "dt": lambda: DecisionTreeRegressor(
                max_depth=params.get("max_depth"),
                min_samples_split=int(params.get("min_samples_split", 2)),
                random_state=rs,
            ),
        }
    return factories[algo_id]()


def _get_importances(model, algo_id: str, X_train, y_train, feat_cols: List[str]) -> np.ndarray:
    n_feat = len(feat_cols)
    uniform = np.ones(n_feat, dtype=float) / max(n_feat, 1)
    try:
        if hasattr(model, "feature_importances_"):
            imp = np.asarray(model.feature_importances_, dtype=float)
            if len(imp) == n_feat:
                s = imp.sum()
                return imp / s if s > 0 else uniform
        if hasattr(model, "coef_"):
            c = np.asarray(model.coef_)
            coef = np.mean(np.abs(c), axis=0) if c.ndim > 1 else np.abs(c.ravel())
            if len(coef) == n_feat:
                s = coef.sum()
                return coef / s if s > 0 else uniform
        if algo_id in ("svm", "knn") and n_feat > 12:
            return uniform
        n = min(_PERM_SAMPLE, len(X_train))
        if n < 5:
            return uniform
        idx = np.random.RandomState(42).choice(len(X_train), n, replace=False)
        Xs = X_train.iloc[idx] if hasattr(X_train, "iloc") else X_train[idx]
        ys = y_train.iloc[idx] if hasattr(y_train, "iloc") else y_train[idx]
        perm = permutation_importance(model, Xs, ys, n_repeats=_PERM_REPEATS, random_state=42, n_jobs=1)
        imp = np.asarray(perm.importances_mean, dtype=float)
        if len(imp) == n_feat:
            s = imp.sum()
            return imp / s if s > 0 else uniform
    except Exception as e:
        logger.warning("Importances fallback: %s", e)
    return uniform


def _stability_section(algo_id, X_train, X_test, y_train, y_test, task_type):
    scores = []
    for seed in SEEDS:
        try:
            if algo_id in ("knn", "nb"):
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_train, y_train, test_size=0.25, random_state=seed,
                )
                m = _create_model(algo_id, task_type, random_state=None)
                m.fit(X_tr, y_tr)
                if task_type == "classification":
                    scores.append(float(accuracy_score(y_te, m.predict(X_te))))
                else:
                    scores.append(float(r2_score(y_te, m.predict(X_te))))
            else:
                m = _create_model(algo_id, task_type, random_state=seed)
                X_fit, y_fit = _cap_train_set(X_train, y_train)
                m.fit(X_fit, y_fit)
                if task_type == "classification":
                    scores.append(float(accuracy_score(y_test, m.predict(X_test))))
                else:
                    scores.append(float(r2_score(y_test, m.predict(X_test))))
        except Exception as e:
            logger.warning("Stability seed %s failed: %s", seed, e)
            scores.append(0.0)

    arr = np.asarray(scores, dtype=float)
    std, mean = float(arr.std()), float(arr.mean())
    name = ALGO_NAMES.get(algo_id, algo_id)
    if std < 0.02:
        level = "faible"
        text = f"**{name}** : variabilité faible — modèle robuste aux changements de seed."
    elif std < 0.05:
        level = "moderee"
        text = f"**{name}** : variabilité modérée — performances globalement stables."
    else:
        level = "elevee"
        text = f"**{name}** : variabilité élevée — sensibilité aux conditions d'entraînement."
    return {
        "metric": "accuracy" if task_type == "classification" else "r2",
        "model_name": name,
        "seeds": SEEDS,
        "scores": [round(float(s), 4) for s in scores],
        "mean": round(mean, 4),
        "std": round(std, 4),
        "robustness_level": level,
    }, text


def _error_examples(model, X_test, y_test, feat_cols, task_type):
    examples = []
    patterns: List[str] = []
    try:
        if task_type == "classification":
            pred = model.predict(X_test)
            yv = np.asarray(y_test)
            wrong = np.where(pred != yv)[0]
            if len(wrong) == 0:
                return [], [], "Aucune erreur sur l'ensemble de test — le modèle classifie parfaitement ce split."
            feat_vals = {f: [] for f in feat_cols}
            for idx in wrong[:3]:
                i = int(idx)
                row_feats = {f: float(X_test.iloc[i][f]) for f in feat_cols if f in X_test.columns}
                for f, v in row_feats.items():
                    feat_vals[f].append(v)
                examples.append({
                    "index": i,
                    "true_label": int(yv[i]),
                    "predicted_label": int(pred[i]),
                    "features": row_feats,
                    "analysis": (
                        f"Ligne #{i} : le modèle prédit la classe {int(pred[i])} "
                        f"au lieu de {int(yv[i])}. Valeurs atypiques ou frontière de décision probable."
                    ),
                })
            for f, vals in feat_vals.items():
                if vals and len(vals) >= 2:
                    patterns.append(f"{f} souvent extrême dans les erreurs (moy. {np.mean(vals):.2f})")
            if not patterns:
                patterns.append("Erreurs dispersées sans variable dominante — cas limites isolés.")
            answer = (
                f"{len(wrong)} erreur(s) sur {len(yv)} tests. "
                f"{len(examples)} cas détaillés ci-dessous. "
                + (" ".join(patterns[:3]))
            )
            return examples, patterns, answer
        pred = model.predict(X_test)
        yv = np.asarray(y_test, dtype=float)
        residuals = np.abs(yv - pred)
        for idx in np.argsort(residuals)[::-1][:3]:
            i = int(idx)
            examples.append({
                "index": i,
                "true_label": round(float(yv[i]), 4),
                "predicted_label": round(float(pred[i]), 4),
                "features": {f: float(X_test.iloc[i][f]) for f in feat_cols if f in X_test.columns},
                "analysis": (
                    f"Ligne #{i} : résidu {residuals[i]:.3f} "
                    f"(vrai {yv[i]:.3f}, prédit {pred[i]:.3f}) — sous-estimation ou sur-estimation locale."
                ),
            })
        patterns.append("Les plus gros résidus indiquent des points mal calibrés ou des valeurs aberrantes.")
        return examples, patterns, f"{len(examples)} pires prédictions (résidus les plus élevés)."
    except Exception as e:
        logger.warning("Error examples failed: %s", e)
        return [], [], f"Analyse des erreurs non disponible : {e}"


def _bias_variance_grid(algo_id, X_train, X_test, y_train, y_test, task_type):
    grids = {
        "rf": [
            {"n_estimators": 10, "max_depth": 2},
            {"n_estimators": 10, "max_depth": 5},
            {"n_estimators": 50, "max_depth": 3},
            {"n_estimators": 50, "max_depth": 10},
            {"n_estimators": 100, "max_depth": 5},
            {"n_estimators": 200, "max_depth": None},
        ],
        "dt": [
            {"max_depth": 3, "min_samples_split": 2},
            {"max_depth": 10, "min_samples_split": 2},
            {"max_depth": None, "min_samples_split": 10},
        ],
        "svm": [{"C": 0.1}, {"C": 1}, {"C": 10}],
        "lr": (
            [{"C": 0.1}, {"C": 1}, {"C": 10}]
            if task_type == "classification"
            else [{"fit_intercept": True}, {"fit_intercept": False}]
        ),
        "knn": [{"n_neighbors": 3}, {"n_neighbors": 5}, {"n_neighbors": 11}],
        "nb": [{"var_smoothing": 1e-9}, {"var_smoothing": 1e-5}],
        "ridge": [{"alpha": 0.1}, {"alpha": 1}, {"alpha": 100}],
        "lasso": [{"alpha": 0.1}, {"alpha": 1}, {"alpha": 100}],
        "svr": [{"C": 0.1}, {"C": 1}, {"C": 10}],
    }
    grid = grids.get(algo_id, grids["dt"])
    param_keys = list(grid[0].keys()) if grid else []
    train_key = "train_accuracy" if task_type == "classification" else "train_r2"
    test_key = "test_accuracy" if task_type == "classification" else "test_r2"
    rows = []

    for params in grid:
        try:
            m = _create_model(algo_id, task_type, random_state=42, **params)
            train_m, test_m = _score_model(m, X_train, X_test, y_train, y_test, task_type)
            row = {
                **{k: ("None" if v is None else v) for k, v in params.items()},
                train_key: round(train_m, 4),
                test_key: round(test_m, 4),
                "bias": round(max(0.0, 1.0 - train_m), 4),
                "variance": round(max(0.0, train_m - test_m), 4),
                "gap": round(train_m - test_m, 4),
            }
            if algo_id == "rf":
                row["n_est"] = row.get("n_estimators")
                row["max_d"] = row.get("max_depth")
            rows.append(row)
        except Exception as e:
            logger.warning("BV grid point failed: %s", e)

    if not rows:
        rows = [{train_key: 0.0, test_key: 0.0, "bias": 0.0, "variance": 0.0, "gap": 0.0}]
        if param_keys:
            rows[0][param_keys[0]] = "N/A"

    skip = {train_key, test_key, "bias", "variance", "gap"}
    if not param_keys:
        param_keys = [k for k in rows[0] if k not in skip] or ["config"]

    over = max(rows, key=lambda r: float(r.get("gap", 0)))
    under = min(rows, key=lambda r: float(r.get(test_key, 0)))
    balanced = min(rows, key=lambda r: abs(float(r.get("gap", 0))))

    def _fmt(row):
        parts = [f"{k}={row.get(k, 'N/A')}" for k in param_keys]
        return ", ".join(parts) if parts else f"test={row.get(test_key)}"

    if algo_id == "rf":
        param_keys = ["n_estimators", "max_depth"]

    labels, train_s, test_s, gaps = [], [], [], []
    for row in rows:
        if algo_id == "rf":
            labels.append(f"n={row.get('n_estimators','?')} d={row.get('max_depth','?')}")
        else:
            labels.append(_fmt(row)[:28])
        train_s.append(float(row[train_key]))
        test_s.append(float(row[test_key]))
        gap = float(row.get("gap", 0))
        gaps.append(gap)
        statut, statut_class = bv_row_status(gap, float(row[test_key]))
        row["statut"] = statut
        row["statut_class"] = statut_class

    conclusions = {
        "overfitting": (
            f"**Quel paramétrage montre overfitting ?** — {_fmt(over)} "
            f"(écart {over['gap']:.4f} > seuil {OVERFIT_THRESHOLD})."
        ),
        "underfitting": (
            f"**Quel paramétrage montre underfitting ?** — {_fmt(under)} "
            f"({test_key} = {under[test_key]})."
        ),
        "balanced": (
            f"**Quel paramétrage semble équilibré ?** — {_fmt(balanced)} "
            f"(écart {balanced['gap']:.4f})."
        ),
    }
    chart_meta = {"labels": labels, "train": train_s, "test": test_s, "gaps": gaps, "bias": [r["bias"] for r in rows], "variance": [r["variance"] for r in rows]}
    return rows, conclusions, param_keys, chart_meta


def _stability_pdf(algo_id, X_train, X_test, y_train, y_test, task_type, model_name):
    accs, f1s = [], []
    metric_name = "Accuracy" if task_type == "classification" else "R²"
    for seed in STABILITY_SEEDS:
        try:
            if algo_id in ("knn", "nb"):
                X_tr, X_te, y_tr, y_te = train_test_split(X_train, y_train, test_size=0.25, random_state=seed)
                m = _create_model(algo_id, task_type, random_state=None)
                m.fit(X_tr, y_tr)
                pred = m.predict(X_te)
                if task_type == "classification":
                    accs.append(float(accuracy_score(y_te, pred)))
                    f1s.append(float(f1_score(y_te, pred, average="weighted", zero_division=0)))
                else:
                    accs.append(float(r2_score(y_te, pred)))
                    f1s.append(accs[-1])
            else:
                m = _create_model(algo_id, task_type, random_state=seed)
                X_fit, y_fit = _cap_train_set(X_train, y_train)
                m.fit(X_fit, y_fit)
                pred = m.predict(X_test)
                if task_type == "classification":
                    accs.append(float(accuracy_score(y_test, pred)))
                    f1s.append(float(f1_score(y_test, pred, average="weighted", zero_division=0)))
                else:
                    accs.append(float(r2_score(y_test, pred)))
                    f1s.append(accs[-1])
        except Exception as e:
            logger.warning("stability seed %s: %s", seed, e)
            accs.append(0.0)
            f1s.append(0.0)

    f1_label = "F1-Score" if task_type == "classification" else "R² (test)"
    charts = [
        chart_stability_line(STABILITY_SEEDS, accs, metric_name),
        chart_stability_bars(STABILITY_SEEDS, f1s, f1_label),
    ]
    stab_table = build_stability_table(STABILITY_SEEDS, accs, f1s) if task_type == "classification" else {
        "rows": [{"random_state": s, "accuracy": round(a, 4), "f1": round(f, 4), "variation": ""} for s, a, f in zip(STABILITY_SEEDS, accs, f1s)],
        "stats": f"Écart R² : {max(accs)-min(accs):.4f}",
        "qual_class": "low",
    }
    std = float(np.std(accs))
    text = (
        f"Variabilité **{stab_table.get('qualitative', 'modérée')}** "
        f"(σ={std:.4f}) — le modèle **{model_name}** "
        + ("est robuste aux changements de random_state." if std < 0.04 else "montre une sensibilité aux conditions d'entraînement.")
    )
    return {"charts": charts, "table": stab_table, "answer": text, "seeds": STABILITY_SEEDS, "accs": accs, "f1s": f1s}


def _errors_pdf(model, X_test, y_test, feat_cols, task_type, importances):
    charts = []
    examples = []
    patterns = []
    if task_type != "classification":
        return {"charts": charts, "examples": examples, "patterns": patterns, "error_table": [], "answer": ""}

    pred = model.predict(X_test)
    yv = np.asarray(y_test)
    wrong = np.where(pred != yv)[0]
    order = np.argsort(importances)[::-1]
    fx = feat_cols[int(order[0])] if len(order) else feat_cols[0]
    fy = feat_cols[int(order[1])] if len(order) > 1 else feat_cols[0]

    try:
        cm = confusion_matrix(yv, pred)
        if cm.shape[0] <= 4:
            charts.append(chart_confusion_matrix(cm))
        charts.append(chart_errors_scatter(X_test, y_test, pred, fx, fy))
        error_table = build_error_summary(cm, len(yv))
    except Exception as e:
        logger.warning("error charts: %s", e)
        error_table = []

    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_test)
        except Exception:
            proba = None

    for n, idx in enumerate(wrong[:3]):
        i = int(idx)
        true_v, pred_v = int(yv[i]), int(pred[i])
        if pred_v == 1 and true_v == 0:
            err_type, critical = "Faux Positif", False
        elif pred_v == 0 and true_v == 1:
            err_type, critical = "Faux Négatif", True
        else:
            err_type, critical = "Erreur de classification", False
        feats = ", ".join(f"{f}={X_test.iloc[i][f]:.2f}" for f in feat_cols[:6] if f in X_test.columns)
        p_txt = ""
        if proba is not None and proba.shape[1] >= 2:
            p = float(proba[i, 1])
            p_txt = f"Probabilité classe positive = {p:.3f} (prédit: {pred_v}). "
        examples.append({
            "index": i,
            "title": f"Exemple {n + 1}: {err_type}" + (" (CRITIQUE)" if critical else ""),
            "title_class": "critical" if critical else "fp",
            "features_line": feats,
            "probability_line": p_txt,
            "reality": f"Réalité = {true_v}",
            "predicted": f"Prédit = {pred_v}",
            "analysis": (
                f"**Analyse :** {p_txt}Écart entre vérité ({true_v}) et prédiction ({pred_v}). "
                f"Variables clés au moment de l'erreur : {feats}. "
                + ("Cas limite / frontière de décision." if not critical else "Échec critique — cas positif non détecté.")
            ),
        })

    if len(wrong) == 0:
        answer = "Aucune erreur sur l'ensemble de test."
    else:
        answer = f"{len(wrong)} erreurs sur {len(yv)} ({100*len(wrong)/len(yv):.1f}%)."
    return {"charts": charts, "examples": examples, "patterns": patterns, "error_table": error_table, "answer": answer}


def _resolve_best_algo(best_algo_id, train_results, task_type):
    allowed = CLASSIFICATION_ALGOS if task_type == "classification" else REGRESSION_ALGOS
    if best_algo_id and best_algo_id in allowed:
        return best_algo_id
    if train_results:
        metric = "accuracy" if task_type == "classification" else "r2"
        valid = [r for r in train_results if r.get("algo_id") in allowed]
        if valid:
            return max(valid, key=lambda r: float(r.get(metric) or -1e9)).get("algo_id", "rf")
    return "rf" if task_type == "classification" else "lr"


def generate_model_report(
    df: pd.DataFrame,
    target: str,
    features: List[str],
    task_type: str,
    test_size: float = 0.2,
    best_algo_id: Optional[str] = None,
    best_algo_name: Optional[str] = None,
    train_results: Optional[List[Dict[str, Any]]] = None,
    dataset_name: Optional[str] = None,
    experiment_name: Optional[str] = None,
) -> Dict[str, Any]:
    t_start = time.perf_counter()
    task_type = (task_type or "classification").lower().strip()
    if task_type not in ("classification", "regression"):
        raise ValueError("task_type doit être 'classification' ou 'regression'.")

    algo_id = _resolve_best_algo(best_algo_id, train_results, task_type)
    model_name = best_algo_name or ALGO_NAMES.get(algo_id, algo_id)

    if test_size > 1:
        test_size = test_size / 100.0
    test_size = max(0.05, min(0.5, float(test_size)))

    if target not in df.columns:
        raise ValueError(f"Colonne cible '{target}' introuvable.")

    feat_cols = [f for f in (features or []) if f in df.columns and f != target]
    if not feat_cols:
        feat_cols = [c for c in df.columns if c != target]
    if not feat_cols:
        raise ValueError("Aucune colonne feature disponible.")

    work = df.dropna(subset=[target]).copy()
    if len(work) < 10:
        raise ValueError("Minimum 10 lignes requises pour le rapport.")

    X = prepare_features_report(work[feat_cols])
    y, _ = prepare_target_report(work[target], task_type)

    stratify = y if task_type == "classification" and y.nunique() > 1 else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stratify,
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42,
        )

    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=feat_cols, index=X_train.index)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=feat_cols, index=X_test.index)

    model = _create_model(algo_id, task_type)
    X_fit, y_fit = _cap_train_set(X_train, y_train)
    model.fit(X_fit, y_fit)

    if algo_id == "rf" and hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
        s = importances.sum()
        if s > 0:
            importances = importances / s
        import_method = "feature_importances_ (Random Forest entraîné)"
    else:
        importances = _get_importances(model, algo_id, X_train, y_train, feat_cols)
        if algo_id == "dt":
            import_method = "feature_importances_ (Decision Tree)"
        elif algo_id in ("lr", "ridge", "lasso"):
            import_method = "coefficients absolus du modèle linéaire"
        else:
            import_method = "importance par permutation (proxy pour ce modèle)"

    order = np.argsort(importances)[::-1]
    top3 = []
    for i in order[:3]:
        idx = int(i)
        if idx < len(feat_cols):
            top3.append((feat_cols[idx], float(importances[idx])))
    if not top3:
        top3 = [(feat_cols[0], 1.0)]

    chart_b64 = chart_feature_importance(feat_cols, importances, model_name)
    imp_table = build_importance_table(feat_cols, importances)
    interpretation = build_interpretation_items(feat_cols, importances, target)

    try:
        stab_pdf = _stability_pdf(algo_id, X_train, X_test, y_train, y_test, task_type, model_name)
    except Exception as e:
        logger.exception("stability")
        stab_pdf = {"charts": [], "table": {"rows": [], "stats": str(e)}, "answer": str(e)}

    try:
        err_pdf = _errors_pdf(model, X_test, y_test, feat_cols, task_type, importances)
        if task_type == "regression":
            ex, pat, ans = _error_examples(model, X_test, y_test, feat_cols, task_type)
            err_pdf["detailed_examples"] = [
                {
                    "title": f"Exemple {i+1}: Fort résidu",
                    "title_class": "fp",
                    "features_line": ", ".join(f"{k}={v:.2f}" for k, v in (ex.get("features") or {}).items()),
                    "reality": f"Réalité = {ex.get('true_label')}",
                    "predicted": f"Prédit = {ex.get('predicted_label')}",
                    "analysis": f"**Analyse :** {ex.get('analysis', '')}",
                }
                for i, ex in enumerate(ex)
            ]
            err_pdf["answer"] = ans
    except Exception as e:
        logger.exception("errors")
        err_pdf = {"charts": [], "examples": [], "error_table": [], "answer": str(e), "detailed_examples": []}

    metric_label = "Accuracy" if task_type == "classification" else "R²"
    try:
        bv_table, bv_conclusions, param_keys, bv_charts_meta = _bias_variance_grid(
            algo_id, X_train, X_test, y_train, y_test, task_type,
        )
    except Exception as e:
        logger.exception("bias_variance")
        bv_table, bv_conclusions, param_keys, bv_charts_meta = [], {}, [], {}
    bv_chart_list = []
    if bv_charts_meta.get("labels"):
        bv_chart_list = [
            chart_bv_train_test(bv_charts_meta["labels"], bv_charts_meta["train"], bv_charts_meta["test"], metric_label),
            chart_bv_stacked(bv_charts_meta["labels"], bv_charts_meta["bias"], bv_charts_meta["variance"]),
            chart_bv_gap(bv_charts_meta["labels"], bv_charts_meta["gaps"]),
        ]

    logger.info("Report done in %.1fs algo=%s", time.perf_counter() - t_start, algo_id)

    assignment = _assignment_sections(task_type, model_name, algo_id)

    def _merge(template: Dict, content: Dict) -> Dict:
        return {**template, **content}

    bv_cols = (
        ["n_est", "max_d", "Train Acc", "Test Acc", "Biais", "Variance", "Statut"]
        if algo_id == "rf" and task_type == "classification"
        else ["n_est", "max_d", "Train R²", "Test R²", "Biais", "Variance", "Statut"]
        if algo_id == "rf"
        else None
    )

    sections = [
        _merge(assignment[0], {
            "chart_base64": chart_b64,
            "charts": [c for c in [chart_b64] if c],
            "importance_table": imp_table,
            "interpretation": interpretation,
            "import_method": import_method,
        }),
        _merge(assignment[1], {
            "answer": stab_pdf.get("answer", ""),
            "charts": [c for c in stab_pdf.get("charts", []) if c],
            "stability_table": stab_pdf.get("table", {}),
            "statistics": stab_pdf.get("table", {}).get("stats", ""),
            "stats_class": stab_pdf.get("table", {}).get("qual_class", "low"),
        }),
        _merge(assignment[2], {
            "answer": err_pdf.get("answer", ""),
            "charts": [c for c in err_pdf.get("charts", []) if c],
            "error_summary_table": err_pdf.get("error_table", []),
            "detailed_examples": err_pdf.get("examples", []),
        }),
        _merge(assignment[3], {
            "charts": [c for c in bv_chart_list if c],
            "table": bv_table,
            "table_title": "Tableau d'Analyse",
            "table_columns": bv_cols,
            "table_param_keys": ["n_estimators", "max_depth"] if algo_id == "rf" else param_keys,
            "conclusions": bv_conclusions,
            "metric_label": metric_label,
        }),
    ]

    result = {
        "title": "Rapport d'analyse — AutoML Studio",
        "subtitle": f"Meilleur modèle : {model_name}",
        "best_algo_id": algo_id,
        "best_algo_name": model_name,
        "task_type": task_type,
        "target": target,
        "n_samples": int(len(work)),
        "n_features": int(len(feat_cols)),
        "import_method": import_method,
        "project": {
            "app": "AutoML Studio",
            "dataset_name": dataset_name or "Dataset",
            "dataset_id": None,
            "experiment_name": experiment_name or "AutoML_Studio",
            "target": target,
            "task_type": task_type,
            "best_model": model_name,
            "best_algo_id": algo_id,
            "features_count": len(feat_cols),
            "test_size_pct": round(test_size * 100, 1),
        },
        "sections": sections,
        "generated_at": pd.Timestamp.now().isoformat(),
        "duration_sec": round(time.perf_counter() - t_start, 1),
    }
    return _json_safe(result)


generate_rf_report = generate_model_report
