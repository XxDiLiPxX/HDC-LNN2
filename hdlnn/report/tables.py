import json
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

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
            f1 = f"{metrics.get('f1_score', 0.0):.4f}"
            prec = f"{metrics.get('precision', 0.0):.4f}"
            rec = f"{metrics.get('recall', 0.0):.4f}"
            auroc = f"{metrics.get('auroc', 0.0):.4f}"
            prauc = f"{metrics.get('pr_auc', 0.0):.4f}"
            fpr = f"{metrics.get('fpr_at_95_tpr', 0.0):.4f}"
            thru = f"{metrics.get('throughput_flows_sec', 0.0):.2f}"
            mem = f"{metrics.get('peak_rss_mb', 0.0):.2f}"
            
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

def print_and_save_comparison_table(runs_dir: Path = Path("runs"), output_file: Path = Path("comparison_table.md")) -> None:
    """Generates the comparison table, prints it to the console, and writes it to a file."""
    table_md = generate_comparison_table(runs_dir)
    print("\n--- BASELINE MODEL COMPARISON TABLE ---")
    print(table_md)
    print("---------------------------------------\n")
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Baseline Model Comparison Table\n\n")
        f.write(table_md)
        f.write("\n")
    logger.info(f"Markdown comparison table written to {output_file}")
