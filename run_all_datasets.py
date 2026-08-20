import os
import sys
import subprocess
import json
import re
from pathlib import Path

def get_config_for_dataset(filepath: Path):
    try:
        with open(filepath, 'r') as f:
            first_line = f.readline().strip()
            
        first_row_split = [h.strip().lower() for h in re.split(r',|\s{2,}', first_line)]
        
        # Check IoT-23 / Zeek-style flow exports
        if "ts" in first_row_split and "uid" in first_row_split and "id.orig_h" in first_row_split:
            return "configs/dataset_iot23.yaml"

        # Check TON-IoT
        if "src_ip" in first_row_split and "dst_ip" in first_row_split and "label" in first_row_split:
            return "configs/dataset_ton_iot.yaml"
            
        # Check UNSW-NB15
        if "label" in first_row_split and "sttl" in first_row_split:
            return "configs/dataset_unsw_nb15.yaml"
            
        if len(first_row_split) == 49:
            return "configs/dataset_unsw_nb15.yaml"
            
        # Check KDD
        if "protocol_type" in first_row_split or "dst_host_count" in first_row_split or "labels" in first_row_split:
            return "configs/dataset_kdd.yaml"
            
        # Check Botnet
        is_botnet = False
        if len(first_row_split) == 35:
            try:
                float(first_row_split[0])
                is_botnet = True
            except ValueError:
                pass
                
        if is_botnet or ("attack" in first_row_split and "seq" in first_row_split):
            return "configs/dataset_unsw_botnet.yaml"
            
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        
    return None

def main():
    archive_dir = Path("datasets")
    venv_python = Path(".venv") / "Scripts" / "python.exe"
    python_executable = str(venv_python if venv_python.exists() else Path(sys.executable))
    
    # Discover and filter CSV files
    csv_files = sorted(list(archive_dir.glob("*.csv")), key=lambda x: x.name)
    datasets_to_run = []
    
    for f in csv_files:
        if "Feature_Names" in f.name:
            continue
        datasets_to_run.append(f)

    print(f"Found {len(datasets_to_run)} dataset(s) in {archive_dir} to process:")
    for f in datasets_to_run:
        print(f"  - {f.name} (Size: {f.stat().st_size / (1024*1024):.2f} MB)")
    print()

    summary_results = []
    
    for f in datasets_to_run:
        filename = f.name
        run_id = f"dataset_{filename.replace('.csv', '').replace(' ', '_')}"
        
        print("="*60)
        print(f"PREPARING EXPERIMENT FOR: {filename}")
        
        config_path = get_config_for_dataset(f)
        if not config_path:
            print(f"WARNING: Skipping {filename} - No matching configuration found for this schema.")
            summary_results.append({
                "dataset": filename,
                "error": "Skipped: Unknown Schema"
            })
            continue
            
        # Decide limit based on file size to ensure fast execution and avoid RAM issues
        size_mb = f.stat().st_size / (1024*1024)
        limit = None
        if size_mb > 50:
            limit = 20000
        elif size_mb > 10:
            limit = 30000
            
        print(f"File Size: {size_mb:.2f} MB | Limit Applied: {limit}")
        print(f"Detected Config: {config_path}")
        print("="*60)

        cmd = [
            python_executable, "-m", "hdlnn.pipeline",
            "--mode", "train",
            "--config-dataset", config_path,
            "--source-file", str(f),
            "--run-id", run_id
        ]
        if limit:
            cmd.extend(["--limit", str(limit)])
            
        try:
            subprocess.run(cmd, check=True)
            
            # Load results from metrics.json
            metrics_path = Path("runs") / run_id / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path, "r") as mf:
                    metrics = json.load(mf)
                
                # Append to summary
                summary_results.append({
                    "dataset": filename,
                    "f1_score": metrics.get("f1_score", 0.0),
                    "precision": metrics.get("precision", 0.0),
                    "recall": metrics.get("recall", 0.0),
                    "auroc": metrics.get("auroc", 0.0),
                    "pr_auc": metrics.get("pr_auc", 0.0),
                    "fpr_at_95_tpr": metrics.get("fpr_at_95_tpr", 0.0),
                    "throughput": metrics.get("throughput_flows_sec", 0.0),
                    "peak_rss": metrics.get("peak_rss_mb", 0.0)
                })
                auroc_val = metrics.get('auroc', 0.0)
                auroc_str = f"{auroc_val:.4f}" if isinstance(auroc_val, (int, float)) else str(auroc_val)
                print(f"SUCCESS: {filename} evaluated. AUROC: {auroc_str}")
            else:
                print(f"WARNING: metrics.json not found for {filename}")
                summary_results.append({
                    "dataset": filename,
                    "error": "Metrics not found"
                })
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Failed to run pipeline for {filename}: {e}")
            summary_results.append({
                "dataset": filename,
                "error": "Execution Failed"
            })
        except Exception as e:
            print(f"ERROR: Unexpected exception during {filename}: {e}")
            summary_results.append({
                "dataset": filename,
                "error": str(e)
            })

    # Generate summary report
    print("\n" + "#"*60)
    print("ALL RUNS COMPLETED. SUMMARY TABLE:")
    print("#"*60)
    
    # Save markdown report to dataset_runs_summary.md
    report_lines = [
        "# Consolidated Dataset Runs Report\n",
        "| Dataset | AUROC | PR-AUC | F1 Score | Precision | Recall | Throughput (f/s) | Peak RSS (MB) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :---: |"
    ]
    
    for r in summary_results:
        if "error" in r:
            report_lines.append(f"| {r['dataset']} | **ERROR** | {r['error']} | - | - | - | - | - |")
        else:
            auroc_str = f"{r['auroc']:.4f}" if isinstance(r['auroc'], (int, float)) else str(r['auroc'])
            pr_auc_str = f"{r['pr_auc']:.4f}" if isinstance(r['pr_auc'], (int, float)) else str(r['pr_auc'])
            f1_str = f"{r['f1_score']:.4f}" if isinstance(r['f1_score'], (int, float)) else str(r['f1_score'])
            prec_str = f"{r['precision']:.4f}" if isinstance(r['precision'], (int, float)) else str(r['precision'])
            rec_str = f"{r['recall']:.4f}" if isinstance(r['recall'], (int, float)) else str(r['recall'])
            report_lines.append(
                f"| {r['dataset']} | {auroc_str} | {pr_auc_str} | {f1_str} | {prec_str} | {rec_str} | {r['throughput']:.2f} | {r['peak_rss']:.2f} |"
            )
            
    report_content = "\n".join(report_lines)
    with open("dataset_runs_summary.md", "w") as rf:
        rf.write(report_content)
        
    print(report_content)
    print(f"\nReport written to dataset_runs_summary.md")

if __name__ == "__main__":
    main()
