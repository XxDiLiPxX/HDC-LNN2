import os
import sys
import json
import time
import psutil
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from hdlnn.common.config import load_config
from hdlnn.eval.harness import run_experiment

def evaluate_and_verify_final_hdclnn():
    print("="*80)
    print("RUNNING FINAL VERIFIED TEST RUN FOR OPTIMIZED HDC-LNN")
    print("="*80)
    
    # 1. UNSW-NB15 Final Run
    unsw_config = load_config(Path("configs/base.yaml"), Path("configs/dataset_unsw_nb15.yaml"))
    run_id_unsw = "final_optimized_hdclnn_unsw_nb15"
    res_unsw = run_experiment(
        config=unsw_config,
        run_id=run_id_unsw,
        baseline_name="hdc-lnn",
        inject_attacks=True,
        limit=30000,
        source_file="datasets/UNSW_NB15_testing-set.csv"
    )
    
    # Verify decisions.csv independently
    decisions_unsw_path = Path("runs") / run_id_unsw / "decisions.csv"
    df_unsw = pd.read_csv(decisions_unsw_path)
    
    y_true_unsw = df_unsw["actual_label"].to_numpy()
    y_pred_unsw = df_unsw["predicted_label"].to_numpy()
    y_score_unsw = df_unsw["drift_score"].to_numpy()
    
    cm_unsw = confusion_matrix(y_true_unsw, y_pred_unsw)
    tn_u, fp_u, fn_u, tp_u = cm_unsw.ravel()
    prec_u = float(precision_score(y_true_unsw, y_pred_unsw, zero_division=0))
    rec_u = float(recall_score(y_true_unsw, y_pred_unsw, zero_division=0))
    f1_u = float(f1_score(y_true_unsw, y_pred_unsw, zero_division=0))
    auroc_u = float(roc_auc_score(y_true_unsw, y_score_unsw))
    prauc_u = float(average_precision_score(y_true_unsw, y_score_unsw))
    
    pos_u = y_score_unsw[y_true_unsw == 1]
    neg_u = y_score_unsw[y_true_unsw == 0]
    th_95_u = np.sort(pos_u)[int(np.floor(0.05 * len(pos_u)))]
    fpr95_u = float(np.sum(neg_u >= th_95_u) / len(neg_u))
    
    print("\n--- INDEPENDENT SKLEARN VERIFICATION (UNSW-NB15) ---")
    print(f"TP={tp_u}, FP={fp_u}, FN={fn_u}, TN={tn_u}")
    print(f"F1={f1_u:.6f}, Precision={prec_u:.6f}, Recall={rec_u:.6f}")
    print(f"AUROC={auroc_u:.6f}, PR-AUC={prauc_u:.6f}, FPR@95%TPR={fpr95_u:.6f}")
    print(f"Latency={res_unsw['latency_ms_per_flow']:.4f} ms, Throughput={res_unsw['throughput_flows_sec']:.2f} f/s, Peak RSS={res_unsw['peak_rss_mb']:.2f} MB")
    
    # 2. KDD-Test Final Run
    kdd_config = load_config(Path("configs/base.yaml"), Path("configs/dataset_kdd.yaml"))
    run_id_kdd = "final_optimized_hdclnn_kdd_test"
    res_kdd = run_experiment(
        config=kdd_config,
        run_id=run_id_kdd,
        baseline_name="hdc-lnn",
        inject_attacks=True,
        limit=None,
        source_file="datasets/kdd_test.csv"
    )
    
    decisions_kdd_path = Path("runs") / run_id_kdd / "decisions.csv"
    df_kdd = pd.read_csv(decisions_kdd_path)
    
    y_true_kdd = df_kdd["actual_label"].to_numpy()
    y_pred_kdd = df_kdd["predicted_label"].to_numpy()
    y_score_kdd = df_kdd["drift_score"].to_numpy()
    
    cm_kdd = confusion_matrix(y_true_kdd, y_pred_kdd)
    tn_k, fp_k, fn_k, tp_k = cm_kdd.ravel()
    prec_k = float(precision_score(y_true_kdd, y_pred_kdd, zero_division=0))
    rec_k = float(recall_score(y_true_kdd, y_pred_kdd, zero_division=0))
    f1_k = float(f1_score(y_true_kdd, y_pred_kdd, zero_division=0))
    auroc_k = float(roc_auc_score(y_true_kdd, y_score_kdd))
    prauc_k = float(average_precision_score(y_true_kdd, y_score_kdd))
    
    pos_k = y_score_kdd[y_true_kdd == 1]
    neg_k = y_score_kdd[y_true_kdd == 0]
    th_95_k = np.sort(pos_k)[int(np.floor(0.05 * len(pos_k)))]
    fpr95_k = float(np.sum(neg_k >= th_95_k) / len(neg_k))
    
    print("\n--- INDEPENDENT SKLEARN VERIFICATION (KDD-TEST) ---")
    print(f"TP={tp_k}, FP={fp_k}, FN={fn_k}, TN={tn_k}")
    print(f"F1={f1_k:.6f}, Precision={prec_k:.6f}, Recall={rec_k:.6f}")
    print(f"AUROC={auroc_k:.6f}, PR-AUC={prauc_k:.6f}, FPR@95%TPR={fpr95_k:.6f}")
    print(f"Latency={res_kdd['latency_ms_per_flow']:.4f} ms, Throughput={res_kdd['throughput_flows_sec']:.2f} f/s, Peak RSS={res_kdd['peak_rss_mb']:.2f} MB")
    
    # Save combined summary
    final_output = {
        "unsw_nb15_optimized": {
            "confusion_matrix": {"tp": int(tp_u), "fp": int(fp_u), "fn": int(fn_u), "tn": int(tn_u)},
            "f1_score": f1_u, "precision": prec_u, "recall": rec_u,
            "auroc": auroc_u, "pr_auc": prauc_u, "fpr_at_95_tpr": fpr95_u,
            "latency_ms": res_unsw['latency_ms_per_flow'],
            "throughput_fps": res_unsw['throughput_flows_sec'],
            "peak_rss_mb": res_unsw['peak_rss_mb']
        },
        "kdd_test_optimized": {
            "confusion_matrix": {"tp": int(tp_k), "fp": int(fp_k), "fn": int(fn_k), "tn": int(tn_k)},
            "f1_score": f1_k, "precision": prec_k, "recall": rec_k,
            "auroc": auroc_k, "pr_auc": prauc_k, "fpr_at_95_tpr": fpr95_k,
            "latency_ms": res_kdd['latency_ms_per_flow'],
            "throughput_fps": res_kdd['throughput_flows_sec'],
            "peak_rss_mb": res_kdd['peak_rss_mb']
        }
    }
    
    with open(Path("runs") / "final_optimized_hdclnn_summary.json", "w") as jf:
        json.dump(final_output, jf, indent=2)
        
    print("\nFINAL EVALUATION & VERIFICATION COMPLETED SUCCESSFULLY.")

if __name__ == "__main__":
    evaluate_and_verify_final_hdclnn()
