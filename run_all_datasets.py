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
            return "configs/dataset_unsw_nb15_raw49.yaml"
            
        # Check KDD
        if "protocol_type" in first_row_split or "dst_host_count" in first_row_split or "labels" in first_row_split:
            return "configs/dataset_kdd.yaml"
        # NSL-KDD is distributed without a header. Its second column is the
        # protocol and it has the declared KDD field count.
        if len(first_row_split) in {42, 43} and len(first_row_split) > 1 and first_row_split[1] in {"tcp", "udp", "icmp"}:
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
    # Support the conventional virtual-environment layout on Windows and POSIX.
    venv_candidates = [
        Path(".venv") / "Scripts" / "python.exe",
        Path(".venv") / "bin" / "python",
    ]
    venv_python = next((candidate for candidate in venv_candidates if candidate.exists()), None)
    python_executable = str(venv_python if venv_python else Path(sys.executable))
    
    # Discover and filter CSV files
    csv_files = sorted(list(archive_dir.glob("*.csv")), key=lambda x: x.name)
    datasets_to_run = []
    
    for f in csv_files:
        if "Feature_Names" in f.name:
            continue
            
        # Filter duplicates and malformed names
        if "(" in f.name and ")" in f.name:
            print(f"Skipping duplicate/malformed dataset: {f.name}")
            continue
            
        # Filter statistically invalid/degenerate datasets (e.g. very small IoT-23 captures)
        size_mb = f.stat().st_size / (1024*1024)
        if size_mb < 1.0 and f.name.startswith("dataset"):
            print(f"Skipping degenerate/small dataset (< 1MB): {f.name} ({size_mb:.3f} MB)")
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
        metrics_path = Path("runs") / run_id / "metrics.json"
        if metrics_path.exists():
            print(f"SKIPPING: {filename} already has immutable run artifacts at {metrics_path.parent}")
            with open(metrics_path, "r") as mf:
                metrics = json.load(mf)
            summary_results.append({
                "dataset": filename,
                "f1_score": metrics.get("f1_score", 0.0), "precision": metrics.get("precision", 0.0),
                "recall": metrics.get("recall", 0.0), "auroc": metrics.get("auroc", 0.0),
                "pr_auc": metrics.get("pr_auc", 0.0), "fpr_at_95_tpr": metrics.get("fpr_at_95_tpr", 0.0),
                "throughput": metrics.get("throughput_flows_sec", 0.0), "peak_rss": metrics.get("peak_rss_mb", 0.0),
            })
            continue
        
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
    
    # Save markdown report to DATASET_EVALUATION_SUMMARY.md
    report_lines = [
        "# Multi-Dataset Evaluation Summary\n",
        "> Benchmark results across evaluated network flow datasets under entity-disjoint validation manifold calibration.\n",
        "## 1. Detection Performance Across Datasets\n",
        "| Model Architecture         | Dataset / Run ID                                 | F1 Score | Precision | Recall | AUROC  | PR-AUC | FPR @ 95% TPR | Latency (ms/flow) | Throughput (flows/s) | Peak RSS (MB) |",
        "|:---------------------------|--------------------------------------------------|----------|-----------|--------|--------|--------|---------------|-------------------|----------------------|--------------:|"
    ]
    
    for r in summary_results:
        if "error" in r:
            report_lines.append(f"| HDC-LNN (Our Method)       | {r['dataset']:<48} | **ERROR** | {r['error']} | - | - | - | - | - | - | - |")
        else:
            auroc_str = f"{r['auroc']:.4f}" if isinstance(r['auroc'], (int, float)) else str(r['auroc'])
            pr_auc_str = f"{r['pr_auc']:.4f}" if isinstance(r['pr_auc'], (int, float)) else str(r['pr_auc'])
            f1_str = f"{r['f1_score']:.4f}" if isinstance(r['f1_score'], (int, float)) else str(r['f1_score'])
            prec_str = f"{r['precision']:.4f}" if isinstance(r['precision'], (int, float)) else str(r['precision'])
            rec_str = f"{r['recall']:.4f}" if isinstance(r['recall'], (int, float)) else str(r['recall'])
            fpr_str = f"{r['fpr_at_95_tpr']:.4f}" if isinstance(r['fpr_at_95_tpr'], (int, float)) else str(r['fpr_at_95_tpr'])
            
            latency = 1000.0 / r['throughput'] if r['throughput'] > 0 else 0.0
            
            report_lines.append(
                f"| HDC-LNN (Our Method)       | {r['dataset']:<48} | {f1_str} | {prec_str} | {rec_str} | {auroc_str} | {pr_auc_str} | {fpr_str} | {latency:.2f} ms | {r['throughput']:.2f} | {r['peak_rss']:.2f} MB |"
            )
            
    report_content = "\n".join(report_lines)
    with open("DATASET_EVALUATION_SUMMARY.md", "w") as rf:
        rf.write(report_content)
        
    print(report_content)
    print(f"\nReport written to DATASET_EVALUATION_SUMMARY.md")

if __name__ == "__main__":
    main()
