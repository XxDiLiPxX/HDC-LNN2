import numpy as np
import time
import psutil
import os
from typing import Dict, Any, List

def calculate_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Calculates Area Under the ROC Curve (AUROC) using the rank-sum Mann-Whitney U test formula."""
    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)
    
    if n_pos == 0 or n_neg == 0:
        return 0.5
        
    # Rank scores (higher rank to higher score)
    ranks = np.argsort(np.argsort(scores)) + 1
    pos_ranks_sum = np.sum(ranks[labels == 1])
    
    u_stat = pos_ranks_sum - (n_pos * (n_pos + 1)) / 2
    auc = u_stat / (n_pos * n_neg)
    return float(auc)

def calculate_fpr_at_95_tpr(labels: np.ndarray, scores: np.ndarray) -> float:
    """Calculates False Positive Rate (FPR) at 95% True Positive Rate (TPR) threshold."""
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]
    
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return 0.0
        
    # Threshold at 95% TPR is the 5th percentile of positive scores
    sorted_pos = np.sort(pos_scores)
    idx = int(np.floor(0.05 * len(sorted_pos)))
    threshold = sorted_pos[idx]
    
    # False positives are negative flows scoring above this threshold
    fp = np.sum(neg_scores >= threshold)
    fpr = fp / len(neg_scores)
    return float(fpr)

def compute_metrics_suite(
    labels: List[int],
    scores: List[float],
    predictions: List[int],
    latency_seconds: float,
    num_samples: int
) -> Dict[str, Any]:
    """Computes full metrics suite including classification, statistical, and engineering targets."""
    labels_arr = np.array(labels)
    scores_arr = np.array(scores)
    preds_arr = np.array(predictions)
    
    # Classification metrics
    tp = int(np.sum((labels_arr == 1) & (preds_arr == 1)))
    fp = int(np.sum((labels_arr == 0) & (preds_arr == 1)))
    fn = int(np.sum((labels_arr == 1) & (preds_arr == 0)))
    tn = int(np.sum((labels_arr == 0) & (preds_arr == 0)))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    auroc = calculate_auroc(labels_arr, scores_arr)
    fpr_at_95_tpr = calculate_fpr_at_95_tpr(labels_arr, scores_arr)
    
    # Import scikit-learn for PR-AUC
    from sklearn.metrics import average_precision_score
    pr_auc = float(average_precision_score(labels_arr, scores_arr))
    
    # Performance metrics
    latency_ms = (latency_seconds / num_samples) * 1000.0 if num_samples > 0 else 0.0
    throughput = num_samples / latency_seconds if latency_seconds > 0 else 0.0
    
    # Peak Memory usage (RSS)
    process = psutil.Process(os.getpid())
    peak_rss_mb = process.memory_info().rss / (1024 * 1024)
    
    return {
        "confusion_matrix": {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn
        },
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auroc": auroc,
        "pr_auc": pr_auc,
        "fpr_at_95_tpr": fpr_at_95_tpr,
        "latency_ms_per_flow": latency_ms,
        "throughput_flows_sec": throughput,
        "peak_rss_mb": peak_rss_mb,
        "sample_size": num_samples
    }
