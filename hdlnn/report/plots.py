import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def compute_roc_coordinates(labels: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Manually computes false positive rates and true positive rates for all unique score thresholds."""
    idx = np.argsort(scores)
    labels = labels[idx]
    scores = scores[idx]
    
    # Cumulative counts of true positives and false positives from right to left
    # (since higher score threshold means fewer positive predictions)
    tps = np.cumsum(labels[::-1])[::-1]
    fps = np.cumsum((1 - labels)[::-1])[::-1]
    
    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)
    
    tpr = tps / n_pos if n_pos > 0 else np.zeros_like(tps)
    fpr = fps / n_neg if n_neg > 0 else np.zeros_like(fps)
    
    # Prepend and append boundaries to complete ROC shape
    # Since we sorted by score ascending, indices from right-to-left correspond to increasing score threshold.
    # At highest threshold, all are predicted normal: TPR=0, FPR=0.
    # At lowest threshold, all are predicted anomalous: TPR=1, FPR=1.
    fpr = np.concatenate([[0.0], fpr[::-1], [1.0]])
    tpr = np.concatenate([[0.0], tpr[::-1], [1.0]])
    
    return fpr, tpr

# Place typing import inside file
from typing import Tuple

def save_roc_plot(run_dir: Path, output_filename: str = "roc_curve.png") -> None:
    """Generates the ROC curve plot from run decisions.parquet and saves it as an image."""
    decisions_parquet = run_dir / "decisions.parquet"
    decisions_csv = run_dir / "decisions.csv"
    if not decisions_parquet.exists() and not decisions_csv.exists():
        logger.warning(f"No decisions file (parquet or csv) found at {run_dir}, skipping ROC plotting.")
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("Matplotlib is not installed. Skipping ROC plot generation.")
        return

    try:
        if decisions_parquet.exists():
            df = pd.read_parquet(decisions_parquet)
        else:
            df = pd.read_csv(decisions_csv)
        labels = df["actual_label"].values
        scores = df["drift_score"].values
        
        fpr, tpr = compute_roc_coordinates(labels, scores)
        
        # Calculate AUROC for plotting label
        metrics_file = run_dir / "metrics.json"
        auroc_val = 0.5
        if metrics_file.exists():
            import json
            with open(metrics_file, "r") as f:
                metrics = json.load(f)
                auroc_val = metrics.get("auroc", 0.5)

        # Plot ROC curve using clean, rich aesthetics (dark styling or minimalist theme)
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color="#2b5c8f", lw=2, label=f"ROC Curve (AUC = {auroc_val:.4f})")
        plt.plot([0, 1], [0, 1], color="#bbbbbb", lw=1, linestyle="--")
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=11, labelpad=8)
        plt.ylabel("True Positive Rate", fontsize=11, labelpad=8)
        plt.title(f"Receiver Operating Characteristic (ROC) - {run_dir.name}", fontsize=12, fontweight="bold", pad=15)
        plt.legend(loc="lower right", frameon=True, fontsize=10)
        plt.grid(True, linestyle=":", alpha=0.6)
        
        plt.tight_layout()
        dest_path = run_dir / output_filename
        plt.savefig(dest_path, dpi=150)
        plt.close()
        logger.info(f"ROC curve plot successfully saved to {dest_path}")
    except Exception as e:
        logger.error(f"Failed to generate ROC plot for {run_dir.name}: {e}")
