"""Metrics for fixed-taxonomy open-set recognition."""

from typing import Dict

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, roc_curve


def compute_open_set_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    unknown_scores: np.ndarray,
    num_known_classes: int,
    closed_set_predictions: np.ndarray = None,
) -> Dict[str, object]:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    unknown_scores = np.asarray(unknown_scores, dtype=np.float64)
    if not (labels.shape == predictions.shape == unknown_scores.shape):
        raise ValueError("labels, predictions and unknown_scores must have equal shape")
    if closed_set_predictions is None:
        closed_set_predictions = predictions
    closed_set_predictions = np.asarray(closed_set_predictions, dtype=np.int64)
    if closed_set_predictions.shape != labels.shape:
        raise ValueError("closed_set_predictions must have the same shape as labels")
    known = labels < num_known_classes
    unknown = ~known

    def accuracy(mask):
        return float(np.mean(labels[mask] == predictions[mask])) if np.any(mask) else float("nan")

    known_accuracy = accuracy(known)
    unknown_recall = accuracy(unknown)
    harmonic = (
        float(2 * known_accuracy * unknown_recall / max(known_accuracy + unknown_recall, 1e-12))
        if np.isfinite(known_accuracy) and np.isfinite(unknown_recall)
        else float("nan")
    )
    binary_unknown = unknown.astype(np.int64)
    auroc = float(roc_auc_score(binary_unknown, unknown_scores)) if known.any() and unknown.any() else float("nan")
    aupr = float(average_precision_score(binary_unknown, unknown_scores)) if unknown.any() else float("nan")
    fpr95 = float("nan")
    if known.any() and unknown.any():
        fpr, tpr, _ = roc_curve(binary_unknown, unknown_scores)
        candidates = np.flatnonzero(tpr >= 0.95)
        if len(candidates):
            fpr95 = float(fpr[candidates[0]])

    oscr = float("nan")
    if known.any() and unknown.any():
        thresholds = np.r_[-np.inf, np.sort(np.unique(unknown_scores)), np.inf]
        known_correct = closed_set_predictions[known] == labels[known]
        ccr = np.asarray([
            np.mean(known_correct & (unknown_scores[known] <= threshold))
            for threshold in thresholds
        ])
        false_acceptance = np.asarray([
            np.mean(unknown_scores[unknown] <= threshold) for threshold in thresholds
        ])
        order = np.argsort(false_acceptance)
        oscr = float(np.trapz(ccr[order], false_acceptance[order]))

    per_class = {
        int(index): accuracy(labels == index)
        for index in np.unique(labels)
    }
    return {
        "overall_accuracy": float(np.mean(labels == predictions)),
        "known_accuracy": known_accuracy,
        "unknown_recall": unknown_recall,
        "harmonic_mean": harmonic,
        "macro_f1": float(f1_score(labels, predictions, labels=np.arange(num_known_classes + 1), average="macro", zero_division=0)),
        "auroc_unknown": auroc,
        "aupr_unknown": aupr,
        "fpr95_unknown": fpr95,
        "oscr": oscr,
        "per_class_accuracy": per_class,
        "num_known_samples": int(known.sum()),
        "num_unknown_samples": int(unknown.sum()),
    }

