import os
import sys
import json
import time
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Tuple
from sklearn.metrics import (
    roc_curve, precision_recall_curve, f1_score, precision_score,
    recall_score, roc_auc_score, average_precision_score, confusion_matrix
)

import torch
import torch.nn as nn
from hdlnn.common.config import load_config
from hdlnn.common.seeding import set_seed
from hdlnn.data.loaders import load_dataset_flows
from hdlnn.data.splitter import split_dataset
from hdlnn.data.quality import verify_split_quality, audit_pipeline_leakage
from hdlnn.eval.injector import inject_threat_scenarios
from hdlnn.hdc.encoder import RecordEncoder
from hdlnn.lnn.model import LNNSequenceModel
from hdlnn.divergence.scorer import DivergenceScorer
from hdlnn.eval.harness import prepare_sequences
from hdlnn.contracts.schemas import TrajectoryState

def run_phase3_and_4(dataset_name: str, config_dataset_path: str, source_csv: str, limit: int = None):
    print("=" * 70)
    print(f"STARTING SEPARABILITY & THRESHOLD CALIBRATION ANALYSIS: {dataset_name}")
    print("=" * 70)
    
    # 1. Load config & seed
    config = load_config(Path("configs/base.yaml"), Path(config_dataset_path))
    set_seed(config.seed)
    
    # 2. Load and prepare flows
    flows = load_dataset_flows(config, Path("."), source_file=source_csv, limit=limit)
    flows = inject_threat_scenarios(flows, seed=config.seed, categorical_cols=config.categorical_columns, numerical_cols=config.numerical_columns)
    train_flows, val_flows, test_flows = split_dataset(flows, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=config.seed)
    
    verify_split_quality(train_flows, val_flows, test_flows)
    audit_pipeline_leakage(train_flows, val_flows, test_flows, config.categorical_columns, config.numerical_columns)
    
    D = config.hdc.get("dimension", 10000)
    hidden_dim = config.model.get("hidden_dim", 64)
    
    # 3. Fit HDC Encoder on Train
    encoder = RecordEncoder(D=D, categorical_columns=config.categorical_columns, numerical_columns=config.numerical_columns)
    encoder.fit(train_flows)
    
    train_inputs = torch.stack([ev.vector for ev in encoder.encode_batch(train_flows)])
    val_inputs = torch.stack([ev.vector for ev in encoder.encode_batch(val_flows)])
    test_inputs = torch.stack([ev.vector for ev in encoder.encode_batch(test_flows)])
    
    # 4. Train LNN Model
    model = LNNSequenceModel(input_dim=D, hidden_dim=hidden_dim, proj_dim=D)
    xs, ys, dts = prepare_sequences(train_flows, train_inputs, seq_len=16)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.model.get("lr", 0.001))
    criterion = nn.MSELoss()
    epochs = config.model.get("epochs", 2)
    batch_size = config.model.get("batch_size", 256)
    
    model.train()
    for ep in range(epochs):
        for b in range(0, len(xs), batch_size):
            xb = xs[b : b + batch_size]
            yb = ys[b : b + batch_size]
            dtb = dts[b : b + batch_size]
            optimizer.zero_grad()
            out_seq, _ = model.forward(xb, dt=dtb)
            pred_seq = model.predict_next_vector(out_seq)
            loss = criterion(pred_seq, yb)
            loss.backward()
            optimizer.step()
            
    # 5. Extract Validation Hidden States step-by-step
    model.eval()
    val_states = []
    entity_states_val = {}
    for idx, flow in enumerate(val_flows):
        eid = flow.entity_id
        h_prev = entity_states_val.get(eid, torch.zeros(hidden_dim))
        x_val = val_inputs[idx]
        dt_val = torch.tensor([flow.dt])
        with torch.no_grad():
            h_next = model.step(x_val, h_prev, dt_val).squeeze(0)
        entity_states_val[eid] = h_next
        val_states.append(h_next)
    val_states = torch.stack(val_states)
    
    val_labels = np.array([f.label for f in val_flows])
    val_normal_indices = [i for i, f in enumerate(val_flows) if f.label == 0]
    val_normal_states = val_states[val_normal_indices]
    
    # 6. Fit Reference Manifold strictly on validation normal states
    scoring_mode = config.divergence.get("mode", "mahalanobis")
    scorer = DivergenceScorer(mode=scoring_mode, threshold_k=config.divergence.get("threshold_k", 3.0), hidden_dim=hidden_dim, model=model)
    scorer.fit_mahalanobis_threshold(val_normal_states)
    
    # Compute Mahalanobis distance scores for ALL validation flows
    val_scores = scorer.manifold.compute_mahalanobis_distance(val_states).numpy()
    
    # ----------------------------------------------------
    # PHASE 3: SEPARABILITY DIAGNOSTICS (Validation Split)
    # ----------------------------------------------------
    val_norm_scores = val_scores[val_labels == 0]
    val_att_scores = val_scores[val_labels == 1]
    
    norm_mean, norm_std = float(np.mean(val_norm_scores)), float(np.std(val_norm_scores))
    norm_median, norm_min, norm_max = float(np.median(val_norm_scores)), float(np.min(val_norm_scores)), float(np.max(val_norm_scores))
    
    att_mean, att_std = float(np.mean(val_att_scores)), float(np.std(val_att_scores))
    att_median, att_min, att_max = float(np.median(val_att_scores)), float(np.min(val_att_scores)), float(np.max(val_att_scores))
    
    # Percentiles of normal validation scores
    norm_p95 = float(np.percentile(val_norm_scores, 95))
    norm_p99 = float(np.percentile(val_norm_scores, 99))
    
    # Overlap metrics: fraction of attack scores falling BELOW normal percentiles
    frac_att_below_norm_p95 = float(np.mean(val_att_scores < norm_p95))
    frac_att_below_norm_median = float(np.mean(val_att_scores < norm_median))
    
    print("\n" + "-" * 60)
    print(f"VALIDATION SCORE DISTRIBUTIONS ({dataset_name}):")
    print("-" * 60)
    print(f"Normal Count: {len(val_norm_scores)} | Attack Count: {len(val_att_scores)}")
    print(f"Normal  Scores: Mean={norm_mean:.4f}, Std={norm_std:.4f}, Median={norm_median:.4f}, Min={norm_min:.4f}, Max={norm_max:.4f}, P95={norm_p95:.4f}")
    print(f"Attack  Scores: Mean={att_mean:.4f}, Std={att_std:.4f}, Median={att_median:.4f}, Min={att_min:.4f}, Max={att_max:.4f}")
    print(f"Overlap (% Attack Scores < Normal P95): {frac_att_below_norm_p95 * 100:.2f}%")
    print(f"Overlap (% Attack Scores < Normal Median): {frac_att_below_norm_median * 100:.2f}%")
    
    # High-TPR ROC exploration on validation
    val_fpr_arr, val_tpr_arr, val_thresh_arr = roc_curve(val_labels, val_scores)
    
    # Interpolate/find FPR at 90%, 95%, 99% TPR on validation
    def get_fpr_and_thresh_at_tpr(target_tpr, tprs, fprs, threshs):
        idx = np.where(tprs >= target_tpr)[0]
        if len(idx) > 0:
            i = idx[0]
            return float(fprs[i]), float(threshs[i]), float(tprs[i])
        return 1.0, float(threshs[-1]), 1.0
        
    val_fpr_90, val_th_90, actual_tpr_90 = get_fpr_and_thresh_at_tpr(0.90, val_tpr_arr, val_fpr_arr, val_thresh_arr)
    val_fpr_95, val_th_95, actual_tpr_95 = get_fpr_and_thresh_at_tpr(0.95, val_tpr_arr, val_fpr_arr, val_thresh_arr)
    val_fpr_99, val_th_99, actual_tpr_99 = get_fpr_and_thresh_at_tpr(0.99, val_tpr_arr, val_fpr_arr, val_thresh_arr)
    
    print("\n" + "-" * 60)
    print(f"HIGH-TPR REGION FPR ON VALIDATION ({dataset_name}):")
    print("-" * 60)
    print(f"TPR = 90% -> FPR = {val_fpr_90:.4f} (Threshold = {val_th_90:.4f}, Actual TPR = {actual_tpr_90:.4f})")
    print(f"TPR = 95% -> FPR = {val_fpr_95:.4f} (Threshold = {val_th_95:.4f}, Actual TPR = {actual_tpr_95:.4f})")
    print(f"TPR = 99% -> FPR = {val_fpr_99:.4f} (Threshold = {val_th_99:.4f}, Actual TPR = {actual_tpr_99:.4f})")
    
    # Generate Diagnostic Plots
    out_dir = Path("runs") / f"diagnostics_{dataset_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Overlaid Histogram
    plt.figure(figsize=(8, 5))
    bins = np.linspace(min(val_scores), max(val_scores), 60)
    plt.hist(val_norm_scores, bins=bins, alpha=0.6, color="#2b5c8f", label=f"Normal (N={len(val_norm_scores)})", density=True)
    plt.hist(val_att_scores, bins=bins, alpha=0.6, color="#d9534f", label=f"Attack (N={len(val_att_scores)})", density=True)
    plt.axvline(norm_p95, color="#2b5c8f", linestyle="--", label=f"Normal 95th Percentile ({norm_p95:.2f})")
    plt.axvline(val_th_95, color="#d9534f", linestyle=":", label=f"95% TPR Threshold ({val_th_95:.2f})")
    plt.title(f"Validation Mahalanobis Score Distributions - {dataset_name}", fontsize=12, fontweight="bold")
    plt.xlabel("Mahalanobis Anomaly Distance")
    plt.ylabel("Density")
    plt.legend(frameon=True)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    hist_path = out_dir / "val_score_distribution.png"
    plt.savefig(hist_path, dpi=150)
    plt.close()
    
    # 2. ROC Curve with 95% TPR point marked & High-TPR Zoom Inset
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    val_auroc = float(roc_auc_score(val_labels, val_scores))
    
    # Full ROC
    ax1.plot(val_fpr_arr, val_tpr_arr, color="#2b5c8f", lw=2, label=f"ROC (AUC = {val_auroc:.4f})")
    ax1.plot([0, 1], [0, 1], color="#888888", linestyle="--")
    ax1.scatter([val_fpr_95], [actual_tpr_95], color="#d9534f", s=80, zorder=5, label=f"95% TPR (FPR={val_fpr_95:.4f})")
    ax1.set_title(f"Validation ROC Curve - {dataset_name}", fontweight="bold")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle=":", alpha=0.5)
    
    # Zoom In: 85% - 100% TPR
    ax2.plot(val_fpr_arr, val_tpr_arr, color="#2b5c8f", lw=2, label="ROC Curve")
    ax2.scatter([val_fpr_90], [actual_tpr_90], color="#f0ad4e", s=70, zorder=5, label=f"90% TPR (FPR={val_fpr_90:.4f})")
    ax2.scatter([val_fpr_95], [actual_tpr_95], color="#d9534f", s=80, zorder=5, label=f"95% TPR (FPR={val_fpr_95:.4f})")
    ax2.scatter([val_fpr_99], [actual_tpr_99], color="#8b0000", s=70, zorder=5, label=f"99% TPR (FPR={val_fpr_99:.4f})")
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.85, 1.01])
    ax2.set_title(f"High-TPR Region Zoom (85-100% TPR) - {dataset_name}", fontweight="bold")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.legend(loc="lower right")
    ax2.grid(True, linestyle=":", alpha=0.5)
    
    plt.tight_layout()
    roc_path = out_dir / "val_roc_curve.png"
    plt.savefig(roc_path, dpi=150)
    plt.close()
    
    # 3. Precision-Recall Curve
    val_pr_arr, val_rec_arr, _ = precision_recall_curve(val_labels, val_scores)
    val_prauc = float(average_precision_score(val_labels, val_scores))
    plt.figure(figsize=(7, 5.5))
    plt.plot(val_rec_arr, val_pr_arr, color="#2e7d32", lw=2, label=f"PR Curve (PR-AUC = {val_prauc:.4f})")
    plt.title(f"Validation Precision-Recall Curve - {dataset_name}", fontsize=12, fontweight="bold")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    pr_path = out_dir / "val_precision_recall_curve.png"
    plt.savefig(pr_path, dpi=150)
    plt.close()
    
    print(f"Saved validation plots to {out_dir}")
    
    # ----------------------------------------------------
    # PHASE 4: THRESHOLD CALIBRATION EXPLORATION
    # ----------------------------------------------------
    print("\n" + "=" * 70)
    print(f"PHASE 4: COMPUTING 5 CANDIDATE THRESHOLDS ON VALIDATION ({dataset_name})")
    print("=" * 70)
    
    cand_thresholds = np.linspace(min(val_scores), max(val_scores), 1000)
    
    best_f1 = -1.0
    th_max_f1 = float(scorer.mahalanobis_threshold)
    
    best_rec_for_high_prec = -1.0
    th_prec_priority = float(scorer.mahalanobis_threshold)
    
    best_prec_for_high_rec = -1.0
    th_rec_priority = float(scorer.mahalanobis_threshold)
    
    min_fpr_for_sec = 999.0
    th_sec_constrained = float(scorer.mahalanobis_threshold)
    
    for th in cand_thresholds:
        preds = (val_scores > th).astype(int)
        cm = confusion_matrix(val_labels, preds)
        tn, fp, fn, tp = cm.ravel()
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        # 1. F1-maximizing threshold
        if f1 > best_f1:
            best_f1 = float(f1)
            th_max_f1 = float(th)
            
        # 2. Precision-priority threshold (target ~95%+ precision while maximizing recall)
        if prec >= 0.95:
            if rec > best_rec_for_high_prec:
                best_rec_for_high_prec = float(rec)
                th_prec_priority = float(th)
                
        # 3. Recall-priority threshold (target ~95%+ recall while maximizing precision)
        if rec >= 0.95:
            if prec > best_prec_for_high_rec:
                best_prec_for_high_rec = float(prec)
                th_rec_priority = float(th)
                
        # 4. Security-constrained threshold (minimize FPR subject to recall >= 70%)
        if rec >= 0.70:
            if fpr < min_fpr_for_sec:
                min_fpr_for_sec = float(fpr)
                th_sec_constrained = float(th)
                
    # Fallback if precision priority did not hit 0.95
    if best_rec_for_high_prec < 0:
        precisions = [precision_score(val_labels, (val_scores > t).astype(int), zero_division=0) for t in cand_thresholds]
        th_prec_priority = float(cand_thresholds[np.argmax(precisions)])
        
    # 5. Current Baseline Threshold (mean + k*std) and 95% TPR Threshold
    th_baseline = float(scorer.mahalanobis_threshold)
    th_95_tpr = float(val_th_95)
    
    candidates = {
        "Current Baseline (Formula k)": th_baseline,
        "Candidate 1 (F1-Maximizing)": th_max_f1,
        "Candidate 2 (Precision-Priority)": th_prec_priority,
        "Candidate 3 (Recall-Priority)": th_rec_priority,
        "Candidate 4 (Security-Constrained rec>=70%)": th_sec_constrained,
        "Candidate 5 (Exact 95% TPR Operating Point)": th_95_tpr
    }
    
    print("Candidate Thresholds Calibrated from Validation:")
    for name, val in candidates.items():
        print(f"  {name:<45}: Threshold = {val:.4f}")
        
    # ----------------------------------------------------
    # EVALUATE ALL 5 CANDIDATES ON FROZEN TEST SET ONCE
    # ----------------------------------------------------
    print("\n" + "=" * 70)
    print(f"EVALUATING CANDIDATES ON FROZEN TEST SET ({dataset_name})")
    print("=" * 70)
    
    # Extract test hidden states step-by-step
    test_states = []
    entity_states_test = {}
    test_labels = np.array([f.label for f in test_flows])
    
    start_eval = time.perf_counter()
    for idx, flow in enumerate(test_flows):
        eid = flow.entity_id
        h_prev = entity_states_test.get(eid, torch.zeros(hidden_dim))
        x_test = test_inputs[idx]
        dt_test = torch.tensor([flow.dt])
        with torch.no_grad():
            h_next = model.step(x_test, h_prev, dt_test).squeeze(0)
        entity_states_test[eid] = h_next
        test_states.append(h_next)
    test_duration = time.perf_counter() - start_eval
    test_states = torch.stack(test_states)
    
    # Mahalanobis test scores
    test_scores = scorer.manifold.compute_mahalanobis_distance(test_states).numpy()
    
    # Test Ranking Metrics (Threshold-independent)
    test_auroc = float(roc_auc_score(test_labels, test_scores))
    test_prauc = float(average_precision_score(test_labels, test_scores))
    
    # Calculate FPR@95%TPR on Test Set
    pos_test = test_scores[test_labels == 1]
    neg_test = test_scores[test_labels == 0]
    th_pos_95 = np.sort(pos_test)[int(np.floor(0.05 * len(pos_test)))]
    test_fpr_at_95_tpr = float(np.sum(neg_test >= th_pos_95) / len(neg_test))
    
    num_test = len(test_flows)
    lat_ms = float((test_duration / num_test) * 1000.0)
    throughput = float(num_test / test_duration if test_duration > 0 else 0.0)
    proc = psutil.Process(os.getpid())
    peak_rss = float(proc.memory_info().rss / (1024 * 1024))
    
    results_table = []
    
    for name, th in candidates.items():
        preds = (test_scores > th).astype(int)
        cm = confusion_matrix(test_labels, preds)
        tn, fp, fn, tp = cm.ravel()
        
        prec = float(precision_score(test_labels, preds, zero_division=0))
        rec = float(recall_score(test_labels, preds, zero_division=0))
        f1 = float(f1_score(test_labels, preds, zero_division=0))
        
        results_table.append({
            "Operating Point": name,
            "Threshold": float(th),
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
            "F1": f1,
            "Precision": prec,
            "Recall": rec,
            "AUROC": test_auroc,
            "PR-AUC": test_prauc,
            "FPR@95%TPR": test_fpr_at_95_tpr,
            "Latency (ms)": lat_ms,
            "Throughput (f/s)": throughput,
            "Peak RSS (MB)": peak_rss
        })
        
    df_results = pd.DataFrame(results_table)
    print(df_results.to_string(index=False))
    
    # Save full diagnostic results json
    full_output = {
        "dataset": dataset_name,
        "validation_stats": {
            "normal": {
                "count": int(len(val_norm_scores)), "mean": norm_mean, "std": norm_std,
                "median": norm_median, "min": norm_min, "max": norm_max, "p95": norm_p95
            },
            "attack": {
                "count": int(len(val_att_scores)), "mean": att_mean, "std": att_std,
                "median": att_median, "min": att_min, "max": att_max
            },
            "overlap_percent_below_norm_p95": frac_att_below_norm_p95 * 100.0,
            "overlap_percent_below_norm_median": frac_att_below_norm_median * 100.0,
            "high_tpr_fpr": {
                "fpr_at_90_tpr": val_fpr_90,
                "fpr_at_95_tpr": val_fpr_95,
                "fpr_at_99_tpr": val_fpr_99
            }
        },
        "test_results": results_table
    }
    with open(out_dir / "calibration_results.json", "w") as jf:
        json.dump(full_output, jf, indent=2)
        
    return full_output

if __name__ == "__main__":
    res_unsw = run_phase3_and_4(
        dataset_name="unsw_nb15_testing_set",
        config_dataset_path="configs/dataset_unsw_nb15.yaml",
        source_csv="datasets/UNSW_NB15_testing-set.csv",
        limit=30000
    )

    res_kdd = run_phase3_and_4(
        dataset_name="kdd_test",
        config_dataset_path="configs/dataset_kdd.yaml",
        source_csv="datasets/kdd_test.csv",
        limit=None
    )

    print("\nALL PHASE 3 & 4 DIAGNOSTICS COMPLETE.")
