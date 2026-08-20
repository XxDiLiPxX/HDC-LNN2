import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

def _format_metric(val: Any, decimals: int = 4) -> str:
    if val is None:
        return "nan"
    try:
        val_float = float(val)
        return f"{val_float:.{decimals}f}"
    except (ValueError, TypeError):
        return str(val)

def generate_comparison_table(runs_dir: Path = Path("runs")) -> str:
    """Scans the runs directory, parses all metrics.json files, and formats a markdown comparison table."""
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return "No runs folder found. Execute experiments first."

    run_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and (d / "metrics.json").exists()]
    if not run_dirs:
        return "No run metrics found in the runs directory."

    table_rows = []
    # Table Header
    headers = [
        "Run ID", "F1 Score", "Precision", "Recall", "AUROC", "PR-AUC",
        "FPR @ 95% TPR", "Throughput (flows/s)", "Peak RSS (MB)"
    ]
    
    for rdir in sorted(run_dirs):
        run_id = rdir.name
        metrics_file = rdir / "metrics.json"
        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            
            # Format row
            f1 = _format_metric(metrics.get('f1_score', 0.0), 4)
            prec = _format_metric(metrics.get('precision', 0.0), 4)
            rec = _format_metric(metrics.get('recall', 0.0), 4)
            auroc = _format_metric(metrics.get('auroc', 0.0), 4)
            prauc = _format_metric(metrics.get('pr_auc', 0.0), 4)
            fpr = _format_metric(metrics.get('fpr_at_95_tpr', 0.0), 4)
            thru = _format_metric(metrics.get('throughput_flows_sec', 0.0), 2)
            mem = _format_metric(metrics.get('peak_rss_mb', 0.0), 2)
            
            table_rows.append([
                run_id, f1, prec, rec, auroc, prauc, fpr, thru, mem
            ])
        except Exception as e:
            logger.error(f"Failed to read metrics from {metrics_file}: {e}")

    # Build markdown table string
    col_widths = [max(len(str(row[i])) for row in table_rows + [headers]) for i in range(len(headers))]
    
    # Header format
    header_str = " | ".join(headers[i].ljust(col_widths[i]) for i in range(len(headers)))
    sep_str = "-|-".join("-" * col_widths[i] for i in range(len(headers)))
    
    markdown_lines = [
        f"| {header_str} |",
        f"|:{sep_str}:|"
    ]
    
    for row in table_rows:
        row_str = " | ".join(str(row[i]).ljust(col_widths[i]) for i in range(len(headers)))
        markdown_lines.append(f"| {row_str} |")
        
    table_md = "\n".join(markdown_lines)
    return table_md

def generate_encoder_separability_table(runs_dir: Path = Path("runs"), dataset_path: Optional[str] = None) -> str:
    """Returns the empirical encoder geometric separability comparison markdown table."""
    headers = [
        "Encoder Architecture", "Centroid Cos Dist", "Norm Euclidean Dist", 
        "Fisher Ratio", "Intra Sim", "Inter Sim", "Separability Margin"
    ]
    
    # 1. Check if runs/encoder_separability.json exists
    json_path = runs_dir / "encoder_separability.json"
    results = None
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                results = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load {json_path}: {e}")

    # 2. If not cached, attempt live computation
    if not results:
        target_csv = dataset_path or "datasets/UNSW_NB15_testing-set.csv"
        if Path(target_csv).exists():
            try:
                import sys
                if str(Path.cwd()) not in sys.path:
                    sys.path.insert(0, str(Path.cwd()))
                from evaluate_encoder_separability import compute_encoder_separability
                results = compute_encoder_separability(dataset_path=target_csv, limit=5000)
            except Exception as e:
                logger.warning(f"Could not compute live encoder separability on {target_csv}: {e}")

    if not results:
        return "No empirical encoder separability metrics available."

    rows = []
    for enc_name, m in results.items():
        rows.append([
            enc_name,
            f"{m.get('cos_dist', 0.0):.4f}",
            f"{m.get('eucl_dist', 0.0):.4f}",
            f"{m.get('fisher_ratio', 0.0):.6f}",
            f"{m.get('intra_sim', 0.0):.4f}",
            f"{m.get('inter_sim', 0.0):.4f}",
            f"{m.get('margin', 0.0):.4f}"
        ])

    col_widths = [max(len(str(row[i])) for row in rows + [headers]) for i in range(len(headers))]
    header_str = " | ".join(headers[i].ljust(col_widths[i]) for i in range(len(headers)))
    sep_str = "-|-".join("-" * col_widths[i] for i in range(len(headers)))
    
    lines = [
        f"| {header_str} |",
        f"|:{sep_str}:|"
    ]
    for row in rows:
        lines.append(f"| " + " | ".join(str(row[i]).ljust(col_widths[i]) for i in range(len(headers))) + " |")
        
    return "\n".join(lines)

def print_and_save_comparison_table(runs_dir: Path = Path("runs"), output_file: Path = Path("comparison_table.md")) -> None:
    """Generates the comparison table, prints it to the console, and writes it to a file."""
    table_md = generate_comparison_table(runs_dir)
    enc_table_md = generate_encoder_separability_table()
    
    print("\n--- BASELINE MODEL COMPARISON TABLE ---")
    print(table_md)
    print("---------------------------------------\n")
    print("\n--- ENCODER SEPARABILITY COMPARISON TABLE ---")
    print(enc_table_md)
    print("---------------------------------------------\n")
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Baseline Model Comparison Table\n\n")
        f.write("## Sequence Models & Detectors\n\n")
        f.write(table_md)
        f.write("\n\n## Encoder-Only Geometric Separability (UNSW-NB15)\n\n")
        f.write(enc_table_md)
        f.write("\n")
    logger.info(f"Markdown comparison table written to {output_file}")
