/** Icônes et libellés des algorithmes (alignés sur le backend). */

export const ALGO_ICONS = {
  rf: "🌲",
  svm: "⚡",
  lr: "📈",
  knn: "🔵",
  dt: "🌳",
  nb: "🎲",
  ridge: "🔺",
  lasso: "🎯",
  svr: "⚡",
  unknown: "🤖",
};

export const ALGO_LABELS = {
  rf: "Random Forest",
  svm: "SVM",
  lr: "Logistic/Linear Reg.",
  knn: "KNN",
  dt: "Decision Tree",
  nb: "Naive Bayes",
  ridge: "Ridge Regression",
  lasso: "Lasso Regression",
  svr: "SVR",
  unknown: "Inconnu",
};

export function algoIcon(algoId) {
  return ALGO_ICONS[algoId] || ALGO_ICONS.unknown;
}

export function algoLabel(run) {
  if (!run) return ALGO_LABELS.unknown;
  return run.algo_name || ALGO_LABELS[run.algo] || run.algo || ALGO_LABELS.unknown;
}

/** Compte les runs par algo (clé = id court). */
export function countByAlgo(runs) {
  return runs.reduce((acc, r) => {
    const id = r.algo || "unknown";
    acc[id] = (acc[id] || 0) + 1;
    return acc;
  }, {});
}
