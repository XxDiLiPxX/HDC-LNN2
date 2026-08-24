import pytest
import numpy as np
from hdlnn.eval.metrics import calculate_auroc, calculate_fpr_at_95_tpr, compute_metrics_suite

def test_metrics_calculations():
    # Setup simple dummy arrays (perfect classification)
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    
    auroc = calculate_auroc(labels, scores)
    assert auroc == pytest.approx(1.0)
    
    # Test random classification ranges
    random_scores = np.array([0.5, 0.1, 0.8, 0.2, 0.3, 0.7, 0.4, 0.6])
    auroc_rand = calculate_auroc(labels, random_scores)
    assert 0.0 <= auroc_rand <= 1.0
    
    # Test FPR at 95% TPR
    # Positive scores: [0.5, 0.6, 0.7, 0.8] (4 scores)
    # sorted_pos = [0.5, 0.6, 0.7, 0.8]
    # idx = int(0.05 * 4) = 0 -> threshold is sorted_pos[0] = 0.5
    # Negative scores: [0.1, 0.2, 0.3, 0.4] -> None are >= 0.5 -> FP = 0 -> FPR = 0.0
    fpr = calculate_fpr_at_95_tpr(labels, scores)
    assert fpr == 0.0
    
    # Add a false positive to negatives
    scores_with_fp = np.array([0.1, 0.2, 0.3, 0.6, 0.5, 0.6, 0.7, 0.8])
    # Negatives: [0.1, 0.2, 0.3, 0.6] -> One value (0.6) is >= 0.5 -> FP = 1 -> FPR = 1/4 = 0.25
    fpr_with_fp = calculate_fpr_at_95_tpr(labels, scores_with_fp)
    assert fpr_with_fp == 0.25

def test_compute_metrics_suite():
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]
    predictions = [0, 1, 1, 0]  # TP=1, FP=1, TN=1, FN=1
    
    suite = compute_metrics_suite(
        labels=labels,
        scores=scores,
        predictions=predictions,
        latency_seconds=1.0,
        num_samples=4
    )
    
    assert suite["precision"] == 0.5
    assert suite["recall"] == 0.5
    assert suite["f1_score"] == 0.5
    assert suite["auroc"] == pytest.approx(1.0)
    assert suite["pr_auc"] == pytest.approx(1.0)
    assert suite["sample_size"] == 4
    assert suite["throughput_flows_sec"] == 4.0
    assert suite["latency_ms_per_flow"] == 250.0
    assert "peak_rss_mb" in suite
    assert suite["score_direction"] == "higher_is_anomalous"

def test_roc_metrics_handle_tied_scores_consistently():
    # A tied normal/anomaly score must receive the same ROC treatment in AUROC
    # and the operating point calculation; the 95% TPR point includes the tie.
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.2, 0.8, 0.8, 0.9])
    assert calculate_auroc(labels, scores) == pytest.approx(0.875)
    assert calculate_fpr_at_95_tpr(labels, scores) == pytest.approx(0.5)
