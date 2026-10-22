"""Seen/unseen metrics for generalized open-vocabulary classification."""

from typing import Dict

import numpy as np


def compute_ovdg_metrics(
    labels: np.ndarray, predictions: np.ndarray, seen_mask: np.ndarray
) -> Dict[str, object]:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    seen_mask = np.asarray(seen_mask, dtype=bool)
    if labels.shape != predictions.shape:
        raise ValueError("labels and predictions must have the same shape")

    sample_seen = seen_mask[labels]
    sample_unseen = ~sample_seen

    def accuracy(mask):
        if not np.any(mask):
            return float("nan")
        return float(np.mean(labels[mask] == predictions[mask]))

    seen_accuracy = accuracy(sample_seen)
    unseen_accuracy = accuracy(sample_unseen)
    if np.isnan(seen_accuracy) or np.isnan(unseen_accuracy):
        harmonic_mean = float("nan")
    else:
        harmonic_mean = float(
            2.0 * seen_accuracy * unseen_accuracy
            / max(seen_accuracy + unseen_accuracy, 1e-12)
        )

    per_class = {}
    for class_index in np.unique(labels):
        mask = labels == class_index
        per_class[int(class_index)] = accuracy(mask)
    return {
        "overall_accuracy": float(np.mean(labels == predictions)),
        "seen_accuracy": seen_accuracy,
        "unseen_accuracy": unseen_accuracy,
        "harmonic_mean": harmonic_mean,
        "per_class_accuracy": per_class,
        "num_seen_samples": int(sample_seen.sum()),
        "num_unseen_samples": int(sample_unseen.sum()),
    }

