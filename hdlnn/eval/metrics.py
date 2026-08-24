"""Shared, score-direction-aware evaluation metrics."""
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import psutil
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

SCORE_DIRECTION = "higher_is_anomalous"


def _validated_arrays(labels: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores):
        raise ValueError("labels and scores must be equally sized one-dimensional arrays")
    if not np.isin(labels, (0, 1)).all() or not np.isfinite(scores).all():
        raise ValueError("labels must be 0/1 and scores must be finite")
    return labels, scores


def roc_operating_points(labels: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ROC points using the project-wide higher-score-is-anomalous convention."""
    labels, scores = _validated_arrays(labels, scores)
    if not (np.any(labels == 0) and np.any(labels == 1)):
        return np.array([0.0]), np.array([0.0]), np.array([np.inf])
    return roc_curve(labels, scores, pos_label=1, drop_intermediate=False)


def calculate_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """AUROC with correct score tie handling."""
    labels, scores = _validated_arrays(labels, scores)
    if not (np.any(labels == 0) and np.any(labels == 1)):
        return 0.5
    return float(roc_auc_score(labels, scores))


def calculate_fpr_at_95_tpr(labels: np.ndarray, scores: np.ndarray, target_tpr: float = 0.95) -> float:
    """Minimum empirical FPR among thresholds whose TPR is at least target."""
    if not 0.0 < target_tpr <= 1.0:
        raise ValueError("target_tpr must be in (0, 1]")
    fpr, tpr, _ = roc_operating_points(labels, scores)
    eligible = fpr[tpr >= target_tpr]
    return float(np.min(eligible)) if len(eligible) else 1.0


def compute_metrics_suite(labels: List[int], scores: List[float], predictions: List[int],
                          latency_seconds: float, num_samples: int) -> Dict[str, Any]:
    """Compute classification, ranking, and process-RSS metrics."""
    labels_arr, scores_arr = _validated_arrays(np.asarray(labels), np.asarray(scores))
    preds_arr = np.asarray(predictions, dtype=int)
    if preds_arr.shape != labels_arr.shape or not np.isin(preds_arr, (0, 1)).all():
        raise ValueError("predictions must be 0/1 and align with labels")
    tp = int(np.sum((labels_arr == 1) & (preds_arr == 1)))
    fp = int(np.sum((labels_arr == 0) & (preds_arr == 1)))
    fn = int(np.sum((labels_arr == 1) & (preds_arr == 0)))
    tn = int(np.sum((labels_arr == 0) & (preds_arr == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    both_classes = np.any(labels_arr == 0) and np.any(labels_arr == 1)
    return {
        "score_direction": SCORE_DIRECTION,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "precision": precision, "recall": recall, "f1_score": f1,
        "auroc": calculate_auroc(labels_arr, scores_arr),
        "pr_auc": float(average_precision_score(labels_arr, scores_arr)) if both_classes else 0.0,
        "fpr_at_95_tpr": calculate_fpr_at_95_tpr(labels_arr, scores_arr),
        "latency_ms_per_flow": (latency_seconds / num_samples) * 1000.0 if num_samples else 0.0,
        "throughput_flows_sec": num_samples / latency_seconds if latency_seconds > 0 else 0.0,
        "peak_rss_mb": psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024),
        "sample_size": num_samples,
    }
