import argparse
import sys
import logging
from pathlib import Path
from hdlnn.common.config import load_config
from hdlnn.common.seeding import set_seed
from hdlnn.common.logging import setup_logging
from hdlnn.eval.harness import run_experiment
from hdlnn.report.tables import print_and_save_comparison_table
from hdlnn.report.plots import save_roc_plot

logger = logging.getLogger("pipeline")

def main():
    parser = argparse.ArgumentParser(description="HYDRA-LNN v2 Command-Line Pipeline Orchestrator")
    parser.add_argument(
        "--mode", 
        choices=["train", "eval", "ablate", "report", "stream", "gui"], 
        required=True,
        help="Pipeline execution mode. 'ablate' runs all baselines; 'report' prints summary tables; 'stream' runs production streaming sidecar; 'gui' runs web dashboard."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port to run the GUI Dashboard server on (defaults to 8050)."
    )
    parser.add_argument(
        "--baseline", 
        default="hdc-lnn",
        choices=["hdc-lnn", "lstm", "cnn", "hdc-only", "lnn-only"],
        help="Baseline model configuration to train or evaluate."
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None,
        help="Optional limit on raw dataset records loaded (useful for fast testing)."
    )
    parser.add_argument(
        "--config-base", 
        default="configs/base.yaml",
        help="Path to global base config YAML."
    )
    parser.add_argument(
        "--config-dataset", 
        default="configs/dataset_unsw_nb15.yaml",
        help="Path to dataset-specific config YAML."
    )
    
    args = parser.parse_args()
    
    # 1. Setup logging
    setup_logging(level=logging.INFO)
    logger.info("Initializing pipeline orchestrator...")

    # 2. Load configurations
    base_path = Path(args.config_base)
    dataset_path = Path(args.config_dataset)
    if not base_path.exists():
        logger.error(f"Base configuration file not found at {base_path}")
        sys.exit(1)
        
    config = load_config(base_path, dataset_path)
    
    # 3. Set random seeds
    set_seed(config.seed)
    
    runs_dir = Path("runs")
    runs_dir.mkdir(parents=True, exist_ok=True)
    
    if args.mode == "ablate":
        # Ablate mode: run all baselines sequentially and report comparative stats
        logger.info("Running ablation study comparing all models...")
        baselines = ["hdc-lnn", "lstm", "cnn", "hdc-only", "lnn-only"]
        
        for baseline in baselines:
            run_id = f"{baseline}_run"
            try:
                run_experiment(
                    config=config,
                    run_id=run_id,
                    baseline_name=baseline,
                    inject_attacks=True,
                    limit=args.limit,
                    output_dir=runs_dir
                )
                # Plot ROC curve for this baseline
                save_roc_plot(runs_dir / run_id)
            except Exception as e:
                logger.error(f"Failed to execute experiment for baseline '{baseline}': {e}", exc_info=True)
                
        # Generate and save final summary table
        print_and_save_comparison_table(runs_dir=runs_dir)
        
    elif args.mode in ("train", "eval"):
        # Single baseline execution
        run_id = f"{args.baseline}_single_run"
        try:
            run_experiment(
                config=config,
                run_id=run_id,
                baseline_name=args.baseline,
                inject_attacks=True,
                limit=args.limit,
                output_dir=runs_dir
            )
            save_roc_plot(runs_dir / run_id)
            # Re-generate summary table to include this new single run
            print_and_save_comparison_table(runs_dir=runs_dir)
        except Exception as e:
            logger.error(f"Failed to execute single experiment for '{args.baseline}': {e}", exc_info=True)
            sys.exit(1)
            
    elif args.mode == "report":
        # Simply compile and print results for all completed runs
        logger.info("Compiling metrics summary reports...")
        print_and_save_comparison_table(runs_dir=runs_dir)
        
        # Save plots for all completed runs
        for run_path in runs_dir.iterdir():
            if run_path.is_dir() and (run_path / "decisions.parquet").exists():
                save_roc_plot(run_path)
                
    elif args.mode == "stream":
        from hdlnn.deploy.sidecar import run_streaming_sidecar_demo
        run_streaming_sidecar_demo(config, limit=args.limit)
        
    elif args.mode == "gui":
        from hdlnn.deploy.gui_server import run_server
        run_server(port=args.port)

if __name__ == "__main__":
    main()
