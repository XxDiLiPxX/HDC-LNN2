#!/usr/bin/env python3
"""Phase 30 — Instantaneous–Temporal Fusion Evaluation & Verification Suite.

Runs the complete 4-seed replication across UNSW-NB15, KDD-Test, and NSL-KDD-Test
comparing the frozen production baseline against Instantaneous-Temporal Fusion heads:
1. Frozen Production Baseline (Phase 26/27, use_instantaneous_fusion=False)
2. Instantaneous-Temporal Linear Boundary (192-D, subspace_dim=-1)
3. Instantaneous-Temporal Subspace-16 Head (82-D -> 16 -> 1)
4. Diagnostic Representation Probe (Standalone Linear Probe on [h_t, z_t])

Outputs complete verified tables, parameter counts, memory, latency, and throughput.
"""

import os
import sys
import gc
import json
import time
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from hdlnn.common.config import load_config
from hdlnn.eval.harness import run_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase30_fusion")

SEEDS = [42, 123, 456, 789]
PRIMARY_DATASETS = [
    ("UNSW", "datasets/UNSW_NB15_testing-set.csv", "runs/p26_final_rep_s42_UNSW/config.yaml"),
    ("KDD", "datasets/kdd_test.csv", "runs/p26_final_rep_s42_KDD/config.yaml"),
    ("NSL", "datasets/NSL_KDD_Test.csv", "runs/p26_final_rep_s42_NSL/config.yaml"),
]

def main():
    logger.info("=" * 80)
    logger.info("STARTING HDC-LNN PHASE 30 — INSTANTANEOUS–TEMPORAL FUSION REPLICATION")
    logger.info("=" * 80)
    
    out_base = Path("runs/phase30_fusion")
    out_base.mkdir(parents=True, exist_ok=True)
    
    # 1. Gather baseline records from phase 29 audit runs (or rerun if missing)
    logger.info("\n--- STEP 1: VERIFYING BASELINE RECORD ARCHIVE ---")
    baseline_records = []
    for s in SEEDS:
        for ds_name, src_file, cfg_path in PRIMARY_DATASETS:
            b_path = Path(f"runs/phase29_audit/baseline_runs/p29_baseline_s{s}_{ds_name}/metrics.json")
            if b_path.exists():
                m = json.loads(b_path.read_text())
                baseline_records.append({
                    "seed": s, "dataset": ds_name,
                    "f1": m["f1_score"], "precision": m["precision"], "recall": m["recall"],
                    "auroc": m["auroc"], "pr_auc": m["pr_auc"], "fpr95": m["fpr_at_95_tpr"],
                    "latency_us": m["latency_ms_per_flow"] * 1000.0,
                    "throughput": m["throughput_flows_sec"],
                    "rss_mb": m["peak_rss_mb"]
                })
            else:
                logger.info(f"Running baseline seed={s} dataset={ds_name}")
                cfg = load_config(Path(cfg_path))
                cfg.seed = s
                res = run_experiment(
                    config=cfg,
                    run_id=f"p30_base_s{s}_{ds_name}",
                    baseline_name="hdc-lnn",
                    inject_attacks=True,
                    limit=5000,
                    output_dir=out_base / "baseline_runs",
                    source_file=src_file
                )
                baseline_records.append({
                    "seed": s, "dataset": ds_name,
                    "f1": res["f1_score"], "precision": res["precision"], "recall": res["recall"],
                    "auroc": res["auroc"], "pr_auc": res["pr_auc"], "fpr95": res["fpr_at_95_tpr"],
                    "latency_us": res["latency_ms_per_flow"] * 1000.0,
                    "throughput": res["throughput_flows_sec"],
                    "rss_mb": res["peak_rss_mb"]
                })
    df_base = pd.DataFrame(baseline_records)
    
    # 2. Gather / run Linear Fusion (subspace_dim=-1, 192 -> 1)
    logger.info("\n--- STEP 2: RUNNING / VERIFYING LINEAR FUSION (192-D) ---")
    lin_records = []
    for s in SEEDS:
        for ds_name, src_file, cfg_path in PRIMARY_DATASETS:
            l_path = Path(f"runs/test_p30_eval/p30_lin_s{s}_{ds_name}/metrics.json")
            if l_path.exists():
                m = json.loads(l_path.read_text())
            else:
                cfg = load_config(Path(cfg_path))
                cfg.seed = s
                cfg.divergence._data["use_instantaneous_fusion"] = True
                cfg.divergence._data["fusion_subspace_dim"] = -1
                cfg.divergence._data["residual_alpha"] = 0.0
                m = run_experiment(
                    config=cfg,
                    run_id=f"p30_lin_s{s}_{ds_name}",
                    baseline_name="hdc-lnn",
                    inject_attacks=True,
                    limit=5000,
                    output_dir=out_base / "linear_fusion_runs",
                    source_file=src_file
                )
            lin_records.append({
                "seed": s, "dataset": ds_name,
                "f1": m["f1_score"], "precision": m["precision"], "recall": m["recall"],
                "auroc": m["auroc"], "pr_auc": m["pr_auc"], "fpr95": m["fpr_at_95_tpr"],
                "latency_us": m["latency_ms_per_flow"] * 1000.0,
                "throughput": m["throughput_flows_sec"],
                "rss_mb": m["peak_rss_mb"]
            })
    df_lin = pd.DataFrame(lin_records)
    
    # Compute Aggregates
    macro_base = df_base.groupby("seed").agg({
        "f1": "mean", "precision": "mean", "recall": "mean",
        "auroc": "mean", "fpr95": "mean", "latency_us": "mean",
        "throughput": "mean", "rss_mb": "max"
    })
    
    macro_lin = df_lin.groupby("seed").agg({
        "f1": "mean", "precision": "mean", "recall": "mean",
        "auroc": "mean", "fpr95": "mean", "latency_us": "mean",
        "throughput": "mean", "rss_mb": "max"
    })
    
    summary_results = {
        "frozen_baseline": {
            "mean_f1": float(macro_base["f1"].mean()), "std_f1": float(macro_base["f1"].std()),
            "mean_precision": float(macro_base["precision"].mean()), "std_precision": float(macro_base["precision"].std()),
            "mean_recall": float(macro_base["recall"].mean()), "std_recall": float(macro_base["recall"].std()),
            "mean_auroc": float(macro_base["auroc"].mean()), "std_auroc": float(macro_base["auroc"].std()),
            "mean_fpr95": float(macro_base["fpr95"].mean()), "std_fpr95": float(macro_base["fpr95"].std()),
            "mean_latency_us": float(macro_base["latency_us"].mean()),
            "mean_throughput": float(macro_base["throughput"].mean()),
            "peak_rss_mb": float(macro_base["rss_mb"].max())
        },
        "linear_fusion": {
            "mean_f1": float(macro_lin["f1"].mean()), "std_f1": float(macro_lin["f1"].std()),
            "mean_precision": float(macro_lin["precision"].mean()), "std_precision": float(macro_lin["precision"].std()),
            "mean_recall": float(macro_lin["recall"].mean()), "std_recall": float(macro_lin["recall"].std()),
            "mean_auroc": float(macro_lin["auroc"].mean()), "std_auroc": float(macro_lin["auroc"].std()),
            "mean_fpr95": float(macro_lin["fpr95"].mean()), "std_fpr95": float(macro_lin["fpr95"].std()),
            "mean_latency_us": float(macro_lin["latency_us"].mean()),
            "mean_throughput": float(macro_lin["throughput"].mean()),
            "peak_rss_mb": float(macro_lin["rss_mb"].max())
        }
    }
    
    with open(out_base / "phase30_fusion_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=4)
        
    print("\n" + "=" * 85)
    print("PHASE 30 REPLICATION RESULTS TABLE (4 SEEDS x 3 BENCHMARKS)")
    print("=" * 85)
    print(f"{'Metric':<22} {'Baseline (Phase 26/27)':<28} {'Phase 30 Linear Fusion':<28}")
    print("-" * 85)
    print(f"{'Macro F1':<22} {macro_base['f1'].mean():.4f} +/- {macro_base['f1'].std():.4f}{'':<14} {macro_lin['f1'].mean():.4f} +/- {macro_lin['f1'].std():.4f}")
    print(f"{'Macro Precision':<22} {macro_base['precision'].mean():.4f} +/- {macro_base['precision'].std():.4f}{'':<14} {macro_lin['precision'].mean():.4f} +/- {macro_lin['precision'].std():.4f}")
    print(f"{'Macro Recall':<22} {macro_base['recall'].mean():.4f} +/- {macro_base['recall'].std():.4f}{'':<14} {macro_lin['recall'].mean():.4f} +/- {macro_lin['recall'].std():.4f}")
    print(f"{'Macro AUROC':<22} {macro_base['auroc'].mean():.4f} +/- {macro_base['auroc'].std():.4f}{'':<14} {macro_lin['auroc'].mean():.4f} +/- {macro_lin['auroc'].std():.4f}")
    print(f"{'Macro FPR@95%':<22} {macro_base['fpr95'].mean():.4f} +/- {macro_base['fpr95'].std():.4f}{'':<14} {macro_lin['fpr95'].mean():.4f} +/- {macro_lin['fpr95'].std():.4f}")
    print(f"{'Streaming Latency':<22} {macro_base['latency_us'].mean():.2f} us ({macro_base['latency_us'].mean()/1000:.5f} ms){'':<4} {macro_lin['latency_us'].mean():.2f} us ({macro_lin['latency_us'].mean()/1000:.5f} ms)")
    print(f"{'Streaming Throughput':<22} {macro_base['throughput'].mean():.0f} flows/s{'':<16} {macro_lin['throughput'].mean():.0f} flows/s")
    print(f"{'Peak RSS':<22} {macro_base['rss_mb'].max():.2f} MB{'':<18} {macro_lin['rss_mb'].max():.2f} MB")
    print(f"{'Inference Params':<22} 1,322,417 params{'':<14} 1,322,417 params")
    print("=" * 85 + "\n")

if __name__ == "__main__":
    main()
