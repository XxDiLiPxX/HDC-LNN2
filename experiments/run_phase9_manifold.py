import os
import sys
import json
import time
import psutil
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple
from sklearn.metrics import (
    roc_curve, precision_recall_curve, f1_score, precision_score,
    recall_score, roc_auc_score, average_precision_score, confusion_matrix
)
from sklearn.covariance import LedoitWolf, OAS, MinCovDet, EmpiricalCovariance
from sklearn.preprocessing import StandardScaler

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
from hdlnn.eval.harness import prepare_sequences

def compute_separability_metrics(val_labels: np.ndarray, val_scores: np.ndarray) -> Dict[str, Any]:
    """Computes distribution stats, % overlap below normal P95, and FPR at 90/95/99% TPR."""
    val_norm_scores = val_scores[val_labels == 0]
    val_att_scores = val_scores[val_labels == 1]
    
    norm_p95 = float(np.percentile(val_norm_scores, 95))
    norm_median = float(np.median(val_norm_scores))
    
    frac_att_below_norm_p95 = float(np.mean(val_att_scores < norm_p95))
    frac_att_below_norm_median = float(np.mean(val_att_scores < norm_median))
    
    val_fpr_arr, val_tpr_arr, val_thresh_arr = roc_curve(val_labels, val_scores)
    
    def get_fpr_at_tpr(target_tpr):
        idx = np.where(val_tpr_arr >= target_tpr)[0]
        if len(idx) > 0:
            return float(val_fpr_arr[idx[0]]), float(val_thresh_arr[idx[0]])
        return 1.0, float(val_thresh_arr[-1])
        
    fpr_90, th_90 = get_fpr_at_tpr(0.90)
    fpr_95, th_95 = get_fpr_at_tpr(0.95)
    fpr_99, th_99 = get_fpr_at_tpr(0.99)
    
    val_auroc = float(roc_auc_score(val_labels, val_scores))
    val_prauc = float(average_precision_score(val_labels, val_scores))
    
    # F1-max threshold calibration on validation
    cand_threshs = np.linspace(min(val_scores), max(val_scores), 1000)
    best_f1 = -1.0
    best_th = cand_threshs[0]
    for th in cand_threshs:
        preds = (val_scores > th).astype(int)
        f1 = f1_score(val_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_th = float(th)
            
    return {
        "val_auroc": val_auroc,
        "val_prauc": val_prauc,
        "val_f1_max": best_f1,
        "best_threshold_val": best_th,
        "norm_mean": float(np.mean(val_norm_scores)),
        "norm_std": float(np.std(val_norm_scores)),
        "norm_median": norm_median,
        "norm_p95": norm_p95,
        "att_mean": float(np.mean(val_att_scores)),
        "att_std": float(np.std(val_att_scores)),
        "att_median": float(np.median(val_att_scores)),
        "overlap_pct_below_norm_p95": frac_att_below_norm_p95 * 100.0,
        "overlap_pct_below_norm_median": frac_att_below_norm_median * 100.0,
        "fpr_at_90_tpr": fpr_90,
        "fpr_at_95_tpr": fpr_95,
        "fpr_at_99_tpr": fpr_99,
        "th_at_95_tpr": th_95
    }

def evaluate_on_test_set(
    test_labels: np.ndarray,
    test_scores: np.ndarray,
    threshold: float,
    eval_duration_sec: float
) -> Dict[str, Any]:
    preds = (test_scores > threshold).astype(int)
    cm = confusion_matrix(test_labels, preds)
    tn, fp, fn, tp = cm.ravel()
    
    prec = float(precision_score(test_labels, preds, zero_division=0))
    rec = float(recall_score(test_labels, preds, zero_division=0))
    f1 = float(f1_score(test_labels, preds, zero_division=0))
    auroc = float(roc_auc_score(test_labels, test_scores))
    pr_auc = float(average_precision_score(test_labels, test_scores))
    
    # FPR@95%TPR on Test
    pos_test = test_scores[test_labels == 1]
    neg_test = test_scores[test_labels == 0]
    th_pos_95 = np.sort(pos_test)[int(np.floor(0.05 * len(pos_test)))]
    fpr_at_95_tpr = float(np.sum(neg_test >= th_pos_95) / len(neg_test))
    
    num_samples = len(test_labels)
    lat_ms = float((eval_duration_sec / num_samples) * 1000.0)
    throughput = float(num_samples / eval_duration_sec if eval_duration_sec > 0 else 0.0)
    proc = psutil.Process(os.getpid())
    peak_rss = float(proc.memory_info().rss / (1024 * 1024))
    
    return {
        "threshold": float(threshold),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "f1": f1, "precision": prec, "recall": rec,
        "auroc": auroc, "pr_auc": pr_auc, "fpr_at_95_tpr": fpr_at_95_tpr,
        "latency_ms": lat_ms, "throughput": throughput, "peak_rss_mb": peak_rss
    }

def run_phase9_experiments(dataset_name: str, config_dataset_path: str, source_csv: str, limit: int = None):
    print("="*75)
    print(f"STARTING PHASE 9 MANIFOLD IMPROVEMENT EXPERIMENTS: {dataset_name}")
    print("="*75)
    
    # 1. Setup
    config = load_config(Path("configs/base.yaml"), Path(config_dataset_path))
    set_seed(config.seed)
    
    flows = load_dataset_flows(config, Path("."), source_file=source_csv, limit=limit)
    flows = inject_threat_scenarios(flows, seed=config.seed, categorical_cols=config.categorical_columns, numerical_cols=config.numerical_columns)
    train_flows, val_flows, test_flows = split_dataset(flows, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=config.seed)
    
    D = config.hdc.get("dimension", 10000)
    hidden_dim = config.model.get("hidden_dim", 64)
    
    encoder = RecordEncoder(D=D, categorical_columns=config.categorical_columns, numerical_columns=config.numerical_columns)
    encoder.fit(train_flows)
    
    train_inputs = torch.stack([ev.vector for ev in encoder.encode_batch(train_flows)])
    val_inputs = torch.stack([ev.vector for ev in encoder.encode_batch(val_flows)])
    test_inputs = torch.stack([ev.vector for ev in encoder.encode_batch(test_flows)])
    
    # 2. Train LNN Model (Frozen representation)
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
            
    # 3. Extract Validation & Test Hidden States (FROZEN)
    model.eval()
    val_states_list = []
    entity_states_val = {}
    for idx, flow in enumerate(val_flows):
        eid = flow.entity_id
        h_prev = entity_states_val.get(eid, torch.zeros(hidden_dim))
        x_val = val_inputs[idx]
        dt_val = torch.tensor([flow.dt])
        with torch.no_grad():
            h_next = model.step(x_val, h_prev, dt_val).squeeze(0)
        entity_states_val[eid] = h_next
        val_states_list.append(h_next.numpy())
    val_states = np.array(val_states_list)
    val_labels = np.array([f.label for f in val_flows])
    
    test_states_list = []
    entity_states_test = {}
    start_test_eval = time.perf_counter()
    for idx, flow in enumerate(test_flows):
        eid = flow.entity_id
        h_prev = entity_states_test.get(eid, torch.zeros(hidden_dim))
        x_test = test_inputs[idx]
        dt_test = torch.tensor([flow.dt])
        with torch.no_grad():
            h_next = model.step(x_test, h_prev, dt_test).squeeze(0)
        entity_states_test[eid] = h_next
        test_states_list.append(h_next.numpy())
    test_eval_duration = time.perf_counter() - start_test_eval
    test_states = np.array(test_states_list)
    test_labels = np.array([f.label for f in test_flows])
    
    val_normal_indices = np.where(val_labels == 0)[0]
    val_normal_states = val_states[val_normal_indices]
    
    print(f"Frozen Hidden States: Val={val_states.shape}, Test={test_states.shape}, Val Normals={val_normal_states.shape}")
    
    # ----------------------------------------------------
    # DEFINE MANIFOLD FITTING VARIANTS
    # ----------------------------------------------------
    variants = {}
    
    # Baseline: Empirical Covariance with eps*I regularization
    def compute_baseline_mahalanobis(norm_X, val_X, test_X, eps=1e-5):
        mu = np.mean(norm_X, axis=0)
        cov = np.cov(norm_X, rowvar=False) + eps * np.eye(norm_X.shape[1])
        inv_cov = np.linalg.inv(cov)
        
        diff_val = val_X - mu
        sq_val = np.sum((diff_val @ inv_cov) * diff_val, axis=1)
        val_dists = np.sqrt(np.clip(sq_val, 0.0, None))
        
        diff_test = test_X - mu
        sq_test = np.sum((diff_test @ inv_cov) * diff_test, axis=1)
        test_dists = np.sqrt(np.clip(sq_test, 0.0, None))
        
        return val_dists, test_dists
    
    # Variant A1: Ledoit-Wolf Shrinkage Covariance
    def compute_ledoit_wolf_mahalanobis(norm_X, val_X, test_X):
        lw = LedoitWolf()
        lw.fit(norm_X)
        mu = lw.location_
        prec = lw.precision_ # precision matrix is the inverse covariance
        
        diff_val = val_X - mu
        sq_val = np.sum((diff_val @ prec) * diff_val, axis=1)
        val_dists = np.sqrt(np.clip(sq_val, 0.0, None))
        
        diff_test = test_X - mu
        sq_test = np.sum((diff_test @ prec) * diff_test, axis=1)
        test_dists = np.sqrt(np.clip(sq_test, 0.0, None))
        
        return val_dists, test_dists, float(lw.shrinkage_)
        
    # Variant A2: Oracle Approximating Shrinkage (OAS)
    def compute_oas_mahalanobis(norm_X, val_X, test_X):
        oas = OAS()
        oas.fit(norm_X)
        mu = oas.location_
        prec = oas.precision_
        
        diff_val = val_X - mu
        sq_val = np.sum((diff_val @ prec) * diff_val, axis=1)
        val_dists = np.sqrt(np.clip(sq_val, 0.0, None))
        
        diff_test = test_X - mu
        sq_test = np.sum((diff_test @ prec) * diff_test, axis=1)
        test_dists = np.sqrt(np.clip(sq_test, 0.0, None))
        
        return val_dists, test_dists, float(oas.shrinkage_)

    # Variant B: Hidden-State Whitening (Z-score + PCA / Mahalanobis on normalized features)
    def compute_whitened_mahalanobis(norm_X, val_X, test_X):
        scaler = StandardScaler()
        norm_normed = scaler.fit_transform(norm_X)
        val_normed = scaler.transform(val_X)
        test_normed = scaler.transform(test_X)
        
        lw = LedoitWolf()
        lw.fit(norm_normed)
        mu = lw.location_
        prec = lw.precision_
        
        diff_val = val_normed - mu
        sq_val = np.sum((diff_val @ prec) * diff_val, axis=1)
        val_dists = np.sqrt(np.clip(sq_val, 0.0, None))
        
        diff_test = test_normed - mu
        sq_test = np.sum((diff_test @ prec) * diff_test, axis=1)
        test_dists = np.sqrt(np.clip(sq_test, 0.0, None))
        
        return val_dists, test_dists

    # Variant C: Robust Covariance Estimation (Minimum Covariance Determinant - MCD / FastMCD)
    def compute_mcd_mahalanobis(norm_X, val_X, test_X):
        # Fast MCD with support_fraction=0.8 to ignore boundary outliers/noise
        mcd = MinCovDet(support_fraction=0.8, random_state=42)
        mcd.fit(norm_X)
        mu = mcd.location_
        prec = mcd.precision_
        
        diff_val = val_X - mu
        sq_val = np.sum((diff_val @ prec) * diff_val, axis=1)
        val_dists = np.sqrt(np.clip(sq_val, 0.0, None))
        
        diff_test = test_X - mu
        sq_test = np.sum((diff_test @ prec) * diff_test, axis=1)
        test_dists = np.sqrt(np.clip(sq_test, 0.0, None))
        
        return val_dists, test_dists

    # 4. Evaluate all variants on Validation Data
    print("\n--- COMPUTING SEPARABILITY FOR EACH MANIFOLD VARIANT ---")
    
    # 1. Baseline
    v_base_val, v_base_test = compute_baseline_mahalanobis(val_normal_states, val_states, test_states)
    sep_base = compute_separability_metrics(val_labels, v_base_val)
    variants["Baseline (Empirical cov + 1e-5*I)"] = (v_base_val, v_base_test, sep_base)
    
    # 2. Ledoit-Wolf
    v_lw_val, v_lw_test, lw_shrink = compute_ledoit_wolf_mahalanobis(val_normal_states, val_states, test_states)
    sep_lw = compute_separability_metrics(val_labels, v_lw_val)
    variants[f"Variant A1 (Ledoit-Wolf Shrinkage={lw_shrink:.3f})"] = (v_lw_val, v_lw_test, sep_lw)
    
    # 3. OAS
    v_oas_val, v_oas_test, oas_shrink = compute_oas_mahalanobis(val_normal_states, val_states, test_states)
    sep_oas = compute_separability_metrics(val_labels, v_oas_val)
    variants[f"Variant A2 (OAS Shrinkage={oas_shrink:.3f})"] = (v_oas_val, v_oas_test, sep_oas)
    
    # 4. Whitening + LedoitWolf
    v_wht_val, v_wht_test = compute_whitened_mahalanobis(val_normal_states, val_states, test_states)
    sep_wht = compute_separability_metrics(val_labels, v_wht_val)
    variants["Variant B (Standardization + LW Cov)"] = (v_wht_val, v_wht_test, sep_wht)
    
    # 5. Robust Covariance (MinCovDet)
    try:
        v_mcd_val, v_mcd_test = compute_mcd_mahalanobis(val_normal_states, val_states, test_states)
        sep_mcd = compute_separability_metrics(val_labels, v_mcd_val)
        variants["Variant C (Robust MinCovDet 80%)"] = (v_mcd_val, v_mcd_test, sep_mcd)
    except Exception as e:
        print(f"MCD fitting failed: {e}")
        
    # 5. Compile Validation Separability Diagnostic Table
    sep_summary = []
    for vname, (_, _, sep) in variants.items():
        sep_summary.append({
            "Manifold Variant": vname,
            "Val AUROC": sep["val_auroc"],
            "Val PR-AUC": sep["val_prauc"],
            "Val F1-Max": sep["val_f1_max"],
            "% Att < Norm P95": sep["overlap_pct_below_norm_p95"],
            "FPR @ 90% TPR": sep["fpr_at_90_tpr"],
            "FPR @ 95% TPR": sep["fpr_at_95_tpr"],
            "FPR @ 99% TPR": sep["fpr_at_99_tpr"],
            "Val Calib Thresh": sep["best_threshold_val"]
        })
        
    df_sep = pd.DataFrame(sep_summary)
    print("\n" + "="*80)
    print(f"VALIDATION SEPARABILITY COMPARISON TABLE ({dataset_name}):")
    print("="*80)
    print(df_sep.to_string(index=False))
    
    # 6. Evaluate all variants on FROZEN TEST SET using their respective validation-calibrated F1-max thresholds
    test_summary = []
    for vname, (_, test_scores, sep) in variants.items():
        calib_th = sep["best_threshold_val"]
        test_metrics = evaluate_on_test_set(test_labels, test_scores, calib_th, test_eval_duration)
        test_summary.append({
            "Manifold Variant": vname,
            "Threshold": test_metrics["threshold"],
            "TP": test_metrics["tp"], "FP": test_metrics["fp"], "FN": test_metrics["fn"], "TN": test_metrics["tn"],
            "F1": test_metrics["f1"],
            "Precision": test_metrics["precision"],
            "Recall": test_metrics["recall"],
            "AUROC": test_metrics["auroc"],
            "PR-AUC": test_metrics["pr_auc"],
            "FPR@95%TPR": test_metrics["fpr_at_95_tpr"],
            "Latency (ms)": test_metrics["latency_ms"],
            "Throughput (f/s)": test_metrics["throughput"],
            "Peak RSS (MB)": test_metrics["peak_rss_mb"]
        })
        
    df_test = pd.DataFrame(test_summary)
    print("\n" + "="*80)
    print(f"FROZEN TEST SET 9-METRIC EVALUATION ({dataset_name}) [F1-Max Operating Points]:")
    print("="*80)
    print(df_test.to_string(index=False))
    
    out_file = Path("runs") / f"phase9_manifold_results_{dataset_name}.json"
    with open(out_file, "w") as jf:
        json.dump({
            "dataset": dataset_name,
            "validation_separability": sep_summary,
            "test_evaluation": test_summary
        }, jf, indent=2)
        
    return sep_summary, test_summary

if __name__ == "__main__":
    res_unsw = run_phase9_experiments(
        dataset_name="unsw_nb15_testing_set",
        config_dataset_path="configs/dataset_unsw_nb15.yaml",
        source_csv="datasets/UNSW_NB15_testing-set.csv",
        limit=30000
    )
    
    res_kdd = run_phase9_experiments(
        dataset_name="kdd_test",
        config_dataset_path="configs/dataset_kdd.yaml",
        source_csv="datasets/kdd_test.csv",
        limit=None
    )
    
    print("\nALL PHASE 9 EXPERIMENTS COMPLETE.")
