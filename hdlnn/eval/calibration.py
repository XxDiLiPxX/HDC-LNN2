"""Validation-only fixed-threshold calibration for higher-is-anomalous scores."""
from typing import Dict, Tuple

import numpy as np

from hdlnn.eval.metrics import calculate_fpr_at_95_tpr


def _metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = scores >= threshold
    tp = float(np.sum((labels == 1) & pred)); fp = float(np.sum((labels == 0) & pred))
    fn = float(np.sum((labels == 1) & ~pred)); tn = float(np.sum((labels == 0) & ~pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"threshold": float(threshold), "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "fpr": fp / (fp + tn) if fp + tn else 0.0}


def calibrate_threshold(labels: np.ndarray, scores: np.ndarray, method: str = "f1_max",
                        min_precision: float = 0.95, target_tpr: float = 0.95) -> Tuple[float, Dict[str, object]]:
    """Select one exact score threshold from validation data, then freeze it."""
    labels = np.asarray(labels, dtype=int); scores = np.asarray(scores, dtype=float)
    if len(labels) != len(scores) or not (np.any(labels == 0) and np.any(labels == 1)):
        raise ValueError("threshold calibration requires validation normal and anomaly labels")
    points = [_metrics(labels, scores, t) for t in np.unique(scores)[::-1]]
    method = method.lower()
    if method == "f1_max":
        selected = max(points, key=lambda p: (p["f1"], p["precision"], p["recall"], -p["fpr"]))
    elif method == "precision_priority":
        selected = max([p for p in points if p["precision"] >= min_precision] or points,
                       key=lambda p: (p["recall"], p["precision"], -p["fpr"]))
    elif method == "recall_priority":
        selected = max(points, key=lambda p: (p["recall"], p["precision"], -p["fpr"]))
    elif method in {"security_constrained", "tpr_95"}:
        selected = min([p for p in points if p["recall"] >= target_tpr] or points,
                       key=lambda p: (p["fpr"], -p["precision"], -p["recall"]))
    elif method == "normal_percentile":
        target_fpr = config.divergence.get("target_fpr", 0.05) if 'config' in globals() else 0.05
        normal_scores = scores[labels == 0]
        threshold = float(np.percentile(normal_scores, (1.0 - target_fpr) * 100))
        selected = _metrics(labels, scores, threshold)
    else:
        raise ValueError(f"Unsupported calibration method: {method}")
    return float(selected["threshold"]), {"method": method, "target_tpr": target_tpr,
        "min_precision": min_precision, "selected": selected,
        "validation_fpr_at_95_tpr": calculate_fpr_at_95_tpr(labels, scores, target_tpr)}
