import os
import sys
import csv
import json
import time
import logging
import asyncio
import torch
import uvicorn
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, Request, Query, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from hdlnn.common.config import load_config
from hdlnn.eval.harness import run_experiment
from hdlnn.data.loaders import load_dataset_flows
from hdlnn.data.splitter import split_dataset
from run_all_baselines import match_config_for_dataset, find_dataset_files, BASELINES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("benchmark_dashboard")

app = FastAPI(title="HDC-LNN Benchmark & Telemetry Control Center")

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR = Path("runs")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# In-memory session tracking
uploaded_custom_path: Optional[Path] = None
uploaded_custom_name: str = ""
active_stream_task: Optional[asyncio.Task] = None
stop_stream_flag: bool = False

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            return f.read()
    return "<h3>Dashboard UI assets loading...</h3>"

@app.get("/style.css")
async def serve_css():
    return FileResponse(STATIC_DIR / "style.css")

@app.get("/app.js")
async def serve_js():
    return FileResponse(STATIC_DIR / "app.js")

@app.get("/api/datasets")
async def list_available_datasets():
    """Lists all standard datasets present in datasets/ directory along with their metadata."""
    files = find_dataset_files(Path("datasets"), all_datasets=True)
    datasets = []
    for f in files:
        cfg_p = match_config_for_dataset(f)
        datasets.append({
            "id": f.name,
            "name": f.stem,
            "filename": f.name,
            "config": cfg_p.name,
            "size_kb": round(f.stat().st_size / 1024, 1)
        })
    return {"status": "success", "datasets": datasets}

@app.get("/api/benchmarks_summary")
async def get_benchmarks_summary():
    """Loads all evaluated benchmark runs and returns them for comparison."""
    files = find_dataset_files(Path("datasets"), all_datasets=True)
    rows = []
    for f in files:
        stem = f.stem
        for baseline in BASELINES:
            b_clean = baseline.replace("-", "")
            run_id = f"eval_{stem}_{b_clean}"
            metrics_file = RUNS_DIR / run_id / "metrics.json"
            if metrics_file.exists():
                try:
                    with open(metrics_file, "r") as mf:
                        m = json.load(mf)
                    rows.append({
                        "dataset": stem,
                        "baseline": baseline,
                        "run_id": run_id,
                        "f1_score": m.get("f1_score", 0.0),
                        "precision": m.get("precision", 0.0),
                        "recall": m.get("recall", 0.0),
                        "auroc": m.get("auroc", 0.0),
                        "pr_auc": m.get("pr_auc", 0.0),
                        "fpr_at_95_tpr": m.get("fpr_at_95_tpr", 0.0),
                        "latency_ms_per_flow": m.get("latency_ms_per_flow", 0.0),
                        "throughput_flows_sec": m.get("throughput_flows_sec", 0.0),
                        "peak_rss_mb": m.get("peak_rss_mb", 0.0)
                    })
                except Exception:
                    pass
    return {"status": "success", "total_runs": len(rows), "runs": rows}

@app.post("/api/upload")
async def handle_file_upload(request: Request):
    """Raw stream body upload to bypass third-party multipart parsing dependencies."""
    global uploaded_custom_path, uploaded_custom_name
    try:
        content_type = request.headers.get("content-type", "")
        body = await request.body()
        dest_file = UPLOAD_DIR / "uploaded_dataset.csv"
        orig_filename = "uploaded_dataset.csv"
        
        if "boundary=" in content_type:
            boundary = content_type.split("boundary=")[-1].strip()
            boundary_bytes = ("--" + boundary).encode()
            parts = body.split(boundary_bytes)
            file_bytes = b""
            for part in parts:
                if b"filename=" in part and b"\r\n\r\n" in part:
                    header_bytes, content = part.split(b"\r\n\r\n", 1)
                    header_text = header_bytes.decode(errors="ignore")
                    for line in header_text.splitlines():
                        if "filename=" in line:
                            orig_filename = line.split("filename=")[-1].strip('"\r\n ')
                            break
                    file_bytes = content.rstrip(b"\r\n--").rstrip(b"\r\n")
                    break
            if not file_bytes:
                file_bytes = body
            with open(dest_file, "wb") as f:
                f.write(file_bytes)
        else:
            with open(dest_file, "wb") as f:
                f.write(body)
                
        uploaded_custom_path = dest_file
        uploaded_custom_name = orig_filename
        logger.info(f"Uploaded {orig_filename} successfully ({dest_file.stat().st_size} bytes).")
        return {"status": "success", "filename": orig_filename, "size": dest_file.stat().st_size}
    except Exception as e:
        logger.error(f"Failed to process upload: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=400)

@app.post("/api/evaluate_benchmark")
async def evaluate_benchmark_endpoint(
    dataset_name: str = Query("UNSW_NB15_testing-set.csv"),
    baseline_model: str = Query("hdc-lnn"),
    limit: int = Query(5000, ge=100, le=50000)
):
    """Executes the exact evaluation harness on demand and returns the empirical verification metrics."""
    global uploaded_custom_path, uploaded_custom_name
    try:
        base_cfg = Path("configs/base.yaml")
        if dataset_name == "custom" and uploaded_custom_path and uploaded_custom_path.exists():
            dataset_path = uploaded_custom_path
            cfg_path = match_config_for_dataset(Path(uploaded_custom_name))
        else:
            dataset_path = Path("datasets") / dataset_name
            if not dataset_path.exists():
                dataset_path = next(Path("datasets").glob("*.csv"))
            cfg_path = match_config_for_dataset(dataset_path)
            
        config = load_config(base_cfg, cfg_path)
        run_id = f"gui_{dataset_path.stem}_{baseline_model.replace('-', '')}"
        
        logger.info(f"Running on-demand benchmark evaluation: dataset={dataset_path.name}, baseline={baseline_model}, limit={limit}")
        metrics = run_experiment(
            config=config,
            baseline_name=baseline_model,
            run_id=run_id,
            output_dir=RUNS_DIR,
            limit=limit,
            source_file=str(dataset_path)
        )
        return {
            "status": "success",
            "dataset": dataset_path.stem,
            "baseline": baseline_model,
            "run_id": run_id,
            "metrics": metrics
        }
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.post("/api/stop")
async def stop_stream():
    global stop_stream_flag
    stop_stream_flag = True
    return {"status": "stopped"}

@app.get("/api/stream")
async def stream_live_telemetry(
    dataset_name: str = Query("UNSW_NB15_testing-set.csv"),
    baseline_model: str = Query("hdc-lnn"),
    speed: int = Query(200, ge=10, le=10000),
    limit: int = Query(2000, ge=50, le=20000),
    threshold_k: float = Query(3.0, ge=1.0, le=10.0)
):
    """Streams live step-by-step telemetry decisions via Server-Sent Events (SSE)."""
    global stop_stream_flag, uploaded_custom_path, uploaded_custom_name
    stop_stream_flag = False
    
    async def event_generator():
        global stop_stream_flag, uploaded_custom_path, uploaded_custom_name
        try:
            # 1. Resolve dataset file and config dynamically at stream time
            base_cfg = Path("configs/base.yaml")
            if dataset_name == "custom" and uploaded_custom_path and uploaded_custom_path.exists():
                dataset_path = uploaded_custom_path
                cfg_path = match_config_for_dataset(Path(uploaded_custom_name))
            else:
                dataset_path = Path("datasets") / dataset_name
                if not dataset_path.exists():
                    dataset_path = Path("datasets/UNSW_NB15_testing-set.csv")
                cfg_path = match_config_for_dataset(dataset_path)
                
            config = load_config(base_cfg, cfg_path)
            
            # Set deterministic seed
            from hdlnn.common.seeding import set_seed
            set_seed(42)
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42)
                
            # Load dataset flows up to limit * 4 so test split has full requested limit flows
            flows = load_dataset_flows(config, data_dir=Path("."), source_file=str(dataset_path), limit=max(limit * 4, 10000))
            if not flows:
                yield "data: " + json.dumps({"type": "error", "message": "No flows loaded from dataset."}) + "\n\n"
                return
                
            # Split dataset
            train_flows, val_flows, test_flows = split_dataset(flows, train_ratio=0.60, val_ratio=0.10, test_ratio=0.30, seed=42)
            if not train_flows:
                train_flows = flows[:int(len(flows)*0.6)]
                test_flows = flows[int(len(flows)*0.6):]
                
            if len(test_flows) < limit and len(flows) > len(test_flows):
                # If test split is smaller than limit, take up to limit from flows
                test_flows = flows[len(train_flows):len(train_flows) + limit]
                if len(test_flows) < limit:
                    test_flows = flows[:limit]
                    
            if len(test_flows) > limit:
                test_flows = test_flows[:limit]
            if len(train_flows) > limit:
                train_flows = train_flows[:limit]
                
            yield "data: " + json.dumps({
                "type": "system",
                "message": f"Splits prepared: Calibration Core={len(train_flows)}, Ingestion Stream={len(test_flows)}. Fitting HDC & LNN core..."
            }) + "\n\n"
            await asyncio.sleep(0.05)
            
            # 2. Fit HDC Encoder
            from hdlnn.hdc.encoder import RecordEncoder
            from hdlnn.lnn.model import LNNSequenceModel
            from hdlnn.contracts.schemas import TrajectoryState, AnomalyDecision
            from hdlnn.divergence.scorer import DivergenceScorer
            from hdlnn.deploy.simd_backend import SIMDEncoder
            from hdlnn.deploy.soar_connector import SIEMAlertHandler
            from hdlnn.eval.harness import prepare_sequences
            
            use_log = bool(config.get("use_log_transform", False))
            encoder = RecordEncoder(
                D=16384,
                categorical_columns=config.categorical_columns,
                numerical_columns=config.numerical_columns,
                use_log_transform=use_log
            )
            encoder.fit(train_flows)
            
            train_inputs = torch.stack([ev.vector for ev in encoder.encode_batch(train_flows)])
            val_inputs = torch.stack([ev.vector for ev in encoder.encode_batch(val_flows)])
            hidden_dim = config.model.get("hidden_dim", 64)
            model = LNNSequenceModel(input_dim=16384, hidden_dim=hidden_dim, proj_dim=16384)
            
            # Sequence Training
            xs, ys, dts = prepare_sequences(train_flows, train_inputs, seq_len=16)
            if len(xs) > 0:
                opt = torch.optim.Adam(model.parameters(), lr=0.001)
                crit = torch.nn.MSELoss()
                model.train()
                for epoch in range(2):
                    for b in range(0, min(len(xs), 512), 64):
                        opt.zero_grad()
                        p_seq, _ = model(xs[b:b+64], dt=dts[b:b+64])
                        p_proj = model.predict_next_vector(p_seq)
                        loss = crit(p_proj, ys[b:b+64])
                        loss.backward()
                        opt.step()
                        
            model.eval()
            
            # 3. Fit Reference Manifold on normal training flows (label == 0)
            train_states = []
            entity_states = {}
            for flow, vec in zip(train_flows, train_inputs):
                eid = flow.entity_id
                h_prev = entity_states.get(eid, torch.zeros(hidden_dim))
                dt_val = torch.tensor([flow.dt])
                with torch.no_grad():
                    h_next = model.step(vec, h_prev, dt_val).squeeze(0)
                entity_states[eid] = h_next
                if flow.label == 0:
                    train_states.append(h_next)
                    
            if not train_states:
                train_states = [h_next]
            train_normal_states = torch.stack(train_states)
            
            scorer = DivergenceScorer(mode="mahalanobis", hidden_dim=hidden_dim, model=model)
            scorer.manifold.fit(train_normal_states)
            
            # Compute distance stats on benign manifold
            norm_dists = scorer.manifold.compute_mahalanobis_distance(train_normal_states).detach().cpu().numpy()
            mean_d = float(np.mean(norm_dists))
            std_d = float(np.std(norm_dists))
            
            # Compute validation scores for threshold calibration (mirroring harness.py)
            val_scores = []
            val_labels = [f.label for f in val_flows]
            val_states_dict = {}
            for flow, vec in zip(val_flows, val_inputs):
                eid = flow.entity_id
                h_prev = val_states_dict.get(eid, torch.zeros(hidden_dim))
                dt_val = torch.tensor([flow.dt])
                with torch.no_grad():
                    h_next = model.step(vec, h_prev, dt_val).squeeze(0)
                val_states_dict[eid] = h_next
                d_val = float(scorer.manifold.compute_mahalanobis_distance(h_next.unsqueeze(0)).squeeze().item())
                val_scores.append(d_val)
                
            from hdlnn.eval.calibration import calibrate_threshold
            if len(set(val_labels)) > 1:
                base_threshold, _ = calibrate_threshold(np.array(val_labels), np.array(val_scores), method="f1_max")
                base_threshold = float(base_threshold)
            else:
                base_threshold = float(mean_d + 2.0 * std_d)
                
            # Allow user to tune sensitivity (k=3.0 applies optimal mean + 2.0*std / validation calibration)
            threshold = float(mean_d + (threshold_k / 3.0) * (2.0 * std_d))
            scorer.mahalanobis_threshold = threshold
            
            simd_encoder = SIMDEncoder(encoder)
            alert_handler = SIEMAlertHandler(config)
            
            yield "data: " + json.dumps({
                "type": "system",
                "message": f"Ledoit-Wolf Benign Manifold Calibrated: mean={mean_d:.2f}, std={std_d:.2f}, Threshold={threshold:.4f}. Streaming telemetry..."
            }) + "\n\n"
            await asyncio.sleep(0.05)
            
            # 4. Stream Evaluation Loop
            test_states = {}
            processed = 0
            anomalies = 0
            tp = 0
            fp = 0
            fn = 0
            tn = 0
            latencies = []
            start_stream_time = time.perf_counter()
            
            for idx, flow in enumerate(test_flows):
                if stop_stream_flag:
                    yield "data: " + json.dumps({"type": "system", "message": "Stream paused by user."}) + "\n\n"
                    break
                    
                t_start = time.perf_counter()
                encoded = simd_encoder.encode(flow)
                eid = flow.entity_id
                h_prev = test_states.get(eid, torch.zeros(hidden_dim))
                dt_step = torch.tensor([flow.dt])
                with torch.no_grad():
                    h_next = model.step(encoded.vector, h_prev, dt_step).squeeze(0)
                test_states[eid] = h_next
                
                # Anomaly check: compute exact Mahalanobis distance from benign manifold
                dist_tensor = scorer.manifold.compute_mahalanobis_distance(h_next.unsqueeze(0))
                score = float(dist_tensor.squeeze().item())
                is_pred_anomaly = bool(score >= threshold)
                
                decision = AnomalyDecision(
                    drift_score=score,
                    threshold=threshold,
                    is_anomaly=is_pred_anomaly,
                    scoring_mode="mahalanobis"
                )
                
                flow_latency_ms = (time.perf_counter() - t_start) * 1000.0
                latencies.append(flow_latency_ms)
                processed += 1
                
                # Confusion matrix
                actual_label = int(flow.label)
                if is_pred_anomaly:
                    anomalies += 1
                    if actual_label == 1:
                        tp += 1
                    else:
                        fp += 1
                else:
                    if actual_label == 1:
                        fn += 1
                    else:
                        tn += 1
                        
                precision = tp / max(1, tp + fp)
                recall = tp / max(1, tp + fn)
                f1 = (2 * precision * recall) / max(1e-6, precision + recall)
                
                # Threat category extraction
                threat_cat = "Normal"
                if actual_label == 1:
                    threat_cat = flow.categorical_fields.get("attack_cat") or flow.categorical_fields.get("service") or "Attack Threat"
                    if threat_cat in ("-", "normal", "0"):
                        threat_cat = "Exploit Threat"
                        
                # Alert dispatching
                if is_pred_anomaly:
                    cef_alert = alert_handler.dispatch_alert(eid, decision, flow.timestamp)
                    yield "data: " + json.dumps({
                        "type": "alert",
                        "text": cef_alert
                    }) + "\n\n"
                    
                elapsed = time.perf_counter() - start_stream_time
                curr_throughput = processed / max(0.0001, elapsed)
                
                # Yield live progress update
                yield "data: " + json.dumps({
                    "type": "progress",
                    "index": idx + 1,
                    "score": score,
                    "threshold": threshold,
                    "is_anomaly": is_pred_anomaly,
                    "actual_label": actual_label,
                    "threat_category": threat_cat if is_pred_anomaly or actual_label == 1 else "Normal",
                    "latency_ms": flow_latency_ms,
                    "throughput": curr_throughput,
                    "processed": processed,
                    "anomalies": anomalies,
                    "f1_score": f1,
                    "precision": precision,
                    "recall": recall
                }) + "\n\n"
                
                if speed > 0:
                    await asyncio.sleep(1.0 / speed)
                    
            # Complete summary
            avg_lat = sum(latencies) / max(1, len(latencies))
            yield "data: " + json.dumps({
                "type": "complete",
                "total_processed": processed,
                "total_anomalies": anomalies,
                "f1_score": f1,
                "precision": precision,
                "recall": recall,
                "avg_latency_ms": avg_lat,
                "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
            }) + "\n\n"
        except Exception as ex:
            logger.error(f"Stream error: {ex}", exc_info=True)
            yield "data: " + json.dumps({"type": "error", "message": str(ex)}) + "\n\n"
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")

def run_server(port: int = 8050):
    logger.info(f"Starting HDC-LNN Benchmark Dashboard on http://127.0.0.1:{port} ...")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")

if __name__ == "__main__":
    run_server()
