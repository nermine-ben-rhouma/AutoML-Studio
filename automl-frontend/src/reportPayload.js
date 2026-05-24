/** Payload POST /report — aligné sur le meilleur modèle entraîné */
export function buildReportPayload(dataset, config, trainInfo, trainResults, bestModel) {
  if (!dataset?.dataset_id) {
    throw new Error("Dataset manquant — refaites l'upload.");
  }
  const algoId = bestModel?.algo_id || trainInfo?.best_model;
  if (!algoId) {
    throw new Error("Meilleur modèle inconnu — relancez l'entraînement.");
  }

  const target = config?.target || trainInfo?.target;
  if (!target) {
    throw new Error("Colonne cible manquante.");
  }

  const features =
    (Array.isArray(config?.features) && config.features.length > 0
      ? config.features
      : trainInfo?.features_used) || [];

  let testSize = config?.testSize ?? trainInfo?.test_size ?? 0.2;
  if (testSize > 1) testSize = testSize / 100;

  return {
    dataset_id: dataset.dataset_id,
    dataset_name: dataset.name || dataset.filename || dataset.dataset_id,
    experiment_name: config?.experimentName || trainInfo?.experiment_name || "AutoML_Studio",
    task_type: config?.taskType || trainInfo?.task_type || "classification",
    target,
    features,
    test_size: testSize,
    best_algo_id: algoId,
    best_algo_name: bestModel?.algo_name || algoId,
    train_results: (trainResults || []).map((r) => ({
      algo_id: r.algo_id,
      algo_name: r.algo_name,
      accuracy: r.accuracy ?? null,
      r2: r.r2 ?? null,
    })),
  };
}
