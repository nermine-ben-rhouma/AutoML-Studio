"""Graphiques et tableaux style rapport PDF (diabetes-ml-dashboard)."""
from __future__ import annotations

import base64
import io
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False
    plt = None

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, r2_score

BAR_COLORS = [
    "#e67e22", "#1abc9c", "#e84393", "#3498db", "#9b59b6",
    "#2ecc71", "#f39c12", "#16a085", "#d35400", "#8e44ad",
]
RANK_COLORS = ["#c0392b", "#27ae60", "#e67e22", "#3498db", "#7f8c8d"]
RANK_LABELS = ["1er", "2e", "3e", "4e", "5e", "6e", "7e", "8e"]
STABILITY_SEEDS = [0, 1, 21, 42, 80, 123, 200, 777]
OVERFIT_THRESHOLD = 0.05


def fig_to_b64(fig, dpi: int = 110) -> Optional[str]:
    if fig is None:
        return None
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii")
    except Exception:
        try:
            plt.close(fig)
        except Exception:
            pass
        return None


def chart_feature_importance(
    features: List[str], importances: np.ndarray, model_name: str, q_label: str = "Q1",
) -> Optional[str]:
    if not _HAS_MPL or not len(features):
        return None
    order = np.argsort(importances)[::-1]
    n = min(len(order), 12)
    order = order[:n]
    labels = [features[int(i)] for i in order][::-1]
    vals = importances[order][::-1]
    mean_v = float(np.mean(importances[order]))
    colors = [BAR_COLORS[i % len(BAR_COLORS)] for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(9, max(3.5, len(labels) * 0.42)))
    bars = ax.barh(labels, vals, color=colors, edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(v + 0.005, bar.get_y() + bar.get_height() / 2, f"{v:.4f}",
                va="center", fontsize=8, color="#333")
    ax.axvline(mean_v, color="#e74c3c", linestyle="--", linewidth=1.5, label=f"Moyenne ({mean_v:.3f})")
    ax.set_xlabel("Importance", fontsize=10)
    ax.set_title(f"{q_label} — Feature Importance ({model_name})", fontsize=11, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0, max(vals) * 1.15 if len(vals) else 1)
    fig.tight_layout()
    return fig_to_b64(fig)


def build_importance_table(features: List[str], importances: np.ndarray, top_n: int = 8) -> List[Dict]:
    order = np.argsort(importances)[::-1][:top_n]
    rows = []
    for rank, idx in enumerate(order):
        i = int(idx)
        rows.append({
            "feature": features[i],
            "importance": round(float(importances[i]), 4),
            "rank": RANK_LABELS[rank] if rank < len(RANK_LABELS) else f"{rank + 1}e",
            "rank_color": RANK_COLORS[min(rank, len(RANK_COLORS) - 1)],
        })
    return rows


def build_interpretation_items(
    features: List[str], importances: np.ndarray, target: str, top_n: int = 3,
) -> List[Dict]:
    order = np.argsort(importances)[::-1][:top_n]
    verdicts = ["COHÉRENT", "CONFORME", "JUSTIFIÉ", "PLAUSIBLE"]
    items = []
    for rank, idx in enumerate(order):
        i = int(idx)
        pct = float(importances[i]) * 100
        feat = features[i]
        ord_label = RANK_LABELS[rank] if rank < len(RANK_LABELS) else f"{rank + 1}e"
        items.append({
            "feature": feat,
            "percent": round(pct, 1),
            "rank_label": ord_label,
            "verdict": verdicts[min(rank, len(verdicts) - 1)],
            "text": (
                f"**{feat}** ({pct:.1f}%) — **{ord_label} prédicteur** : "
                f"variable fortement associée à la cible « {target} ». "
                f"Le modèle s'appuie sur cette feature pour discriminer les classes — "
                f"résultat **{verdicts[min(rank, len(verdicts) - 1)]}** avec la structure du jeu de données."
            ),
        })
    return items


def chart_stability_line(seeds: List[int], scores: List[float], metric: str = "Accuracy") -> Optional[str]:
    if not _HAS_MPL or not seeds:
        return None
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.plot(seeds, scores, "o-", color="#27ae60", linewidth=2, markersize=6, label=metric)
    mean_s = float(np.mean(scores))
    std_s = float(np.std(scores))
    ax.axhline(mean_s, color="#e74c3c", linestyle="--", linewidth=1.5, label=f"Moyenne ({mean_s:.4f})")
    ax.fill_between(seeds, mean_s - std_s, mean_s + std_s, alpha=0.15, color="#27ae60")
    ax.set_xlabel("random_state", fontsize=9)
    ax.set_ylabel(metric, fontsize=9)
    ax.set_title(f"{metric} selon random_state", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig_to_b64(fig)


def chart_stability_bars(seeds: List[int], scores: List[float], metric: str = "F1-Score") -> Optional[str]:
    if not _HAS_MPL or not seeds:
        return None
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.bar([str(s) for s in seeds], scores, color="#3498db", edgecolor="white")
    mean_s = float(np.mean(scores))
    ax.axhline(mean_s, color="#e74c3c", linestyle="--", linewidth=1.5, label=f"Moyenne ({mean_s:.4f})")
    ax.set_xlabel("random_state", fontsize=9)
    ax.set_ylabel(metric, fontsize=9)
    ax.set_title(f"{metric} selon random_state", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig_to_b64(fig)


def build_stability_table(seeds: List[int], accs: List[float], f1s: List[float]) -> Dict:
    accs = list(accs)
    f1s = list(f1s)
    i_min = int(np.argmin(accs))
    i_max = int(np.argmax(accs))
    rows = []
    for s, a, f in zip(seeds, accs, f1s):
        var = ""
        if s == seeds[i_min]:
            var = "Min"
        elif s == seeds[i_max]:
            var = "Max"
        rows.append({
            "random_state": s,
            "accuracy": round(a, 4),
            "f1": round(f, 4),
            "variation": var,
        })
    rows.append({
        "random_state": "Moyenne",
        "accuracy": round(float(np.mean(accs)), 4),
        "f1": round(float(np.mean(f1s)), 4),
        "variation": "—",
    })
    gap = float(max(accs) - min(accs))
    pct = gap * 100
    if gap < 0.04:
        qual, qual_class = "Très faible", "low"
    elif gap < 0.08:
        qual, qual_class = "Modérée", "mid"
    else:
        qual, qual_class = "Élevée", "high"
    stats = (
        f"Écart Min-Max : {accs[i_max]:.4f} − {accs[i_min]:.4f} = {gap:.4f} ({pct:.2f}%) → **{qual}**"
    )
    return {"rows": rows, "gap": round(gap, 4), "gap_pct": round(pct, 2), "qualitative": qual, "qual_class": qual_class, "stats": stats}


def chart_confusion_matrix(cm: np.ndarray) -> Optional[str]:
    if not _HAS_MPL:
        return None
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    im = ax.imshow(cm, cmap="Greens", aspect="auto")
    n = cm.shape[0]
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="black", fontsize=11, fontweight="bold")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"Prédit {j}" for j in range(n)])
    ax.set_yticklabels([f"Réel {i}" for i in range(n)])
    ax.set_title("Matrice de Confusion", fontsize=10, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig_to_b64(fig)


def chart_errors_scatter(
    X_test, y_test, y_pred, feat_x: str, feat_y: str,
) -> Optional[str]:
    if not _HAS_MPL or feat_x not in X_test.columns or feat_y not in X_test.columns:
        return None
    yv = np.asarray(y_test)
    yp = np.asarray(y_pred)
    correct = yv == yp
    fp = (~correct) & (yp == 1) if yv.max() <= 1 else ~correct
    fn = (~correct) & (yv == 1) if yv.max() <= 1 else np.zeros_like(correct)

    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.scatter(X_test.loc[correct, feat_x], X_test.loc[correct, feat_y], c="#2ecc71", alpha=0.5, s=18, label="Correct")
    if fp.any():
        ax.scatter(X_test.loc[fp, feat_x], X_test.loc[fp, feat_y], c="#9b59b6", marker="^", s=50,
                   label=f"Faux Positifs ({fp.sum()})", edgecolors="k", linewidths=0.3)
    if fn.any():
        ax.scatter(X_test.loc[fn, feat_x], X_test.loc[fn, feat_y], c="#e74c3c", marker="x", s=50,
                   label=f"Faux Négatifs ({fn.sum()})", linewidths=1.5)
    ax.set_xlabel(feat_x)
    ax.set_ylabel(feat_y)
    ax.set_title(f"Erreurs : FP et FN ({feat_x} vs {feat_y})", fontsize=9, fontweight="bold")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig_to_b64(fig)


def build_error_summary(cm: np.ndarray, n_test: int) -> List[Dict]:
    if cm.shape != (2, 2):
        tn = int(np.trace(cm)) if cm.size else 0
        total_err = n_test - tn
        return [{"type": "Erreurs totales", "count": total_err, "pct": f"{100*total_err/max(n_test,1):.1f}%", "impact": "Voir matrice de confusion", "critical": False}]
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    total = max(n_test, 1)
    return [
        {"type": "Faux Positifs (FP)", "count": fp, "pct": f"{100*fp/total:.1f}%",
         "impact": "Classe positive prédite à tort", "critical": False},
        {"type": "Faux Négatifs (FN)", "count": fn, "pct": f"{100*fn/total:.1f}%",
         "impact": "Cas positifs non détectés — **CRITIQUE**", "critical": True},
        {"type": "Total", "count": fp + fn, "pct": f"{100*(fp+fn)/total:.1f}%", "impact": "—", "critical": False},
    ]


def chart_bv_train_test(labels: List[str], train_s: List[float], test_s: List[float], metric: str) -> Optional[str]:
    if not _HAS_MPL:
        return None
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(x, train_s, "o-", color="#27ae60", label="Train", linewidth=2)
    ax.plot(x, test_s, "o-", color="#3498db", label="Test", linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel(metric)
    ax.set_title(f"Train vs Test {metric}", fontsize=10, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig_to_b64(fig)


def chart_bv_stacked(labels: List[str], bias: List[float], variance: List[float]) -> Optional[str]:
    if not _HAS_MPL:
        return None
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.bar(x, bias, label="Biais", color="#fd79a8")
    ax.bar(x, variance, bottom=bias, label="Variance", color="#e17055")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax.set_title("Biais vs Variance", fontsize=10, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    return fig_to_b64(fig)


def chart_bv_gap(labels: List[str], gaps: List[float]) -> Optional[str]:
    if not _HAS_MPL:
        return None
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    colors = ["#e74c3c" if g > OVERFIT_THRESHOLD else "#3498db" for g in gaps]
    ax.bar(x, gaps, color=colors)
    ax.axhline(OVERFIT_THRESHOLD, color="#e74c3c", linestyle="--", linewidth=1.5, label="Seuil overfitting (5%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("Train − Test")
    ax.set_title("Écart Train-Test (Overfitting)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig_to_b64(fig)


def bv_row_status(gap: float, test_acc: float) -> Tuple[str, str]:
    if gap > OVERFIT_THRESHOLD:
        return "Overfitting", "warn"
    if test_acc < 0.65:
        return "Underfitting", "bad"
    return "Équilibré", "ok"
