import os
import csv
import json
import time
import logging
import asyncio
import torch
import uvicorn
from pathlib import Path
from typing import List, Dict, Any, Generator
from fastapi import FastAPI, UploadFile, File, Query, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from hdlnn.contracts.schemas import CanonicalFlow, AnomalyDecision
from hdlnn.common.config import load_config
from hdlnn.hdc.encoder import RecordEncoder
from hdlnn.deploy.simd_backend import SIMDEncoder
from hdlnn.deploy.sidecar import StreamingSidecar
from hdlnn.deploy.soar_connector import SIEMAlertHandler
from hdlnn.lnn.model import LNNSequenceModel
from hdlnn.divergence.scorer import DivergenceScorer

logger = logging.getLogger("gui_server")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="HYDRA-LNN v2 Behavioral Drift Dashboard")

# Paths for static assets
CURRENT_DIR = Path(__file__).parent
STATIC_DIR = CURRENT_DIR / "static"
UPLOAD_DIR = CURRENT_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# State variables
stop_requested = False
uploaded_file_path = None

@app.get("/", response_class=HTMLResponse)
async def read_index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            return f.read()
    return "<h3>Error: index.html not found under static directory.</h3>"

@app.get("/style.css")
async def get_css():
    css_file = STATIC_DIR / "style.css"
    if css_file.exists():
        return FileResponse(css_file)
    return "CSS not found"

@app.get("/app.js")
async def get_js():
    js_file = STATIC_DIR / "app.js"
    if js_file.exists():
        return FileResponse(js_file)
    return "JS not found"

@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    global uploaded_file_path
    try:
        dest_file = UPLOAD_DIR / "uploaded.csv"
        with open(dest_file, "wb") as f:
            f.write(await file.read())
        uploaded_file_path = dest_file
        logger.info(f"File uploaded successfully to {uploaded_file_path}")
        return {"status": "success", "filename": file.filename}
    except Exception as e:
        logger.error(f"Failed to upload file: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/api/stop")
async def stop_stream():
    global stop_requested
    stop_requested = True
    logger.info("Simulation stop requested by user.")
    return {"status": "stopped"}

def load_csv_flows(file_path: Path, config: Any) -> List[CanonicalFlow]:
    """Dynamically parses any uploaded CSV dataset and maps columns onto CanonicalFlow."""
    logger.info(f"Dynamically parsing custom CSV: {file_path}")
    flows = []
    
    with open(file_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        try:
            headers = next(reader)
        except StopIteration:
            return []
            
        # Clean headers
        headers = [h.strip().lower() for h in headers]
        
        # Check key column indices
        label_idx = next((i for i, h in enumerate(headers) if h in ("label", "class")), None)
        attack_cat_idx = next((i for i, h in enumerate(headers) if h in ("attack_cat", "attack", "category", "threat_type")), None)
        dur_idx = next((i for i, h in enumerate(headers) if h in ("dur", "duration", "time_span", "time")), None)
        
        # Categorize other columns in CSV dynamically to expected configuration columns
        cat_cols_in_csv = []
        num_cols_in_csv = []
        
        for i, h in enumerate(headers):
            if i in (label_idx, attack_cat_idx, dur_idx):
                continue
            is_num = False
            # Check configured column exact match
            if any(col.lower() == h for col in config.numerical_columns):
                is_num = True
            elif any(col.lower() == h for col in config.categorical_columns):
                is_num = False
            else:
                # Basic guess: if header name looks numeric
                is_num = any(x in h for x in ("byte", "pkt", "count", "rate", "loss", "load", "depth", "len", "ttl", "rtt"))
                
            if is_num:
                num_cols_in_csv.append((h, i))
            else:
                cat_cols_in_csv.append((h, i))
                
        global_time = 0.0
        row_count = 0
        
        for row in reader:
            if not row:
                continue
                
            # Parse label
            label = 0
            if label_idx is not None and label_idx < len(row):
                try:
                    label = int(row[label_idx])
                except ValueError:
                    pass
            
            # Parse threat category
            attack_cat = "Normal"
            if attack_cat_idx is not None and attack_cat_idx < len(row):
                attack_cat = row[attack_cat_idx]
            elif label == 1:
                attack_cat = "Exploit/Anomaly"
                
            # Parse duration and delta-time
            dur = 0.01
            if dur_idx is not None and dur_idx < len(row):
                try:
                    dur = max(0.0001, float(row[dur_idx]))
                except ValueError:
                    pass
            
            global_time += dur
            entity_id = f"10.0.0.{(row_count % 100) + 1}"
            
            # Map fields ensuring all expected columns from config are populated to avoid KeyError
            categorical_fields = {}
            for k_idx, col in enumerate(config.categorical_columns):
                val = "-"
                # 1. Match by name
                name_match = next((row[idx] for name, idx in cat_cols_in_csv if name == col.lower() and idx < len(row)), None)
                if name_match is not None:
                    val = name_match
                # 2. Match positionally
                elif k_idx < len(cat_cols_in_csv):
                    idx = cat_cols_in_csv[k_idx][1]
                    if idx < len(row):
                        val = row[idx]
                categorical_fields[col] = val
                
            numerical_fields = {}
            for k_idx, col in enumerate(config.numerical_columns):
                val = 0.0
                # 1. Match by name
                name_match = next((row[idx] for name, idx in num_cols_in_csv if name == col.lower() and idx < len(row)), None)
                if name_match is not None:
                    try:
                        val = float(name_match)
                    except ValueError:
                        pass
                # 2. Match positionally
                elif k_idx < len(num_cols_in_csv):
                    idx = num_cols_in_csv[k_idx][1]
                    if idx < len(row):
                        try:
                            val = float(row[idx])
                        except ValueError:
                            pass
                numerical_fields[col] = val
                        
            flow = CanonicalFlow(
                entity_id=entity_id,
                timestamp=global_time,
                dt=dur,
                categorical_fields=categorical_fields,
                numerical_fields=numerical_fields,
                label=label,
                split="train" if row_count % 3 != 0 else "test" # 66% fit, 33% stream
            )
            flow.categorical_fields["attack_cat"] = attack_cat
            flows.append(flow)
            row_count += 1
            
    logger.info(f"Loaded {len(flows)} flows dynamically.")
    return flows

@app.get("/api/stream")
async def stream_sidecar_simulation(
    speed: int = Query(100, ge=1, le=5000),
    limit: int = Query(2000, ge=10, le=100000),
    k: float = Query(3.0, ge=1.0, le=10.0)
):
    global stop_requested, uploaded_file_path
    stop_requested = False
    
    # 1. Load Configurations
    config_base = Path("configs/base.yaml")
    config_dataset = Path("configs/dataset_unsw_nb15.yaml")
    config = load_config(config_base, config_dataset)
    
    # 2. Parse CSV source file
    if uploaded_file_path and uploaded_file_path.exists():
        flows = load_csv_flows(uploaded_file_path, config)
    else:
        # Fall back to UNSW-NB15 standard dataset if no uploaded file
        from hdlnn.data.loaders import load_raw_unsw_nb15
        flows = load_raw_unsw_nb15(config, Path("data"), limit=limit)
        
    if not flows:
        async def err_gen():
            yield "data: " + json.dumps({"type": "system", "message": "No flows loaded. Make sure the dataset is non-empty."}) + "\n\n"
        return StreamingResponse(err_gen(), media_type="text/event-stream")
        
    # Segment training vs testing
    if len(flows) > limit:
        flows = flows[:limit]
        
    train_flows = [f for f in flows if f.split == "train"]
    test_flows = [f for f in flows if f.split == "test"]
    
    if not train_flows:
        train_flows = flows[:int(len(flows)*0.7)]
        test_flows = flows[int(len(flows)*0.7):]
        
    test_flows = test_flows[:limit]
    
    logger.info(f"Simulation setup: train={len(train_flows)}, test={len(test_flows)}, speed={speed} flows/s, k={k}")

    async def sse_event_generator():
        global stop_requested
        
        yield "data: " + json.dumps({"type": "system", "message": f"Fitting RecordEncoder & calibration core on {len(train_flows)} training flows..."}) + "\n\n"
        await asyncio.sleep(0.1)
        
        # Fit Encoder
        input_dim = 16384  # packed binary layout dimension
        encoder = RecordEncoder(
            D=input_dim,
            categorical_columns=config.categorical_columns,
            numerical_columns=config.numerical_columns
        )
        encoder.fit(train_flows)
        
        train_inputs = torch.stack([ev.vector for ev in encoder.encode_batch(train_flows)])
        
        # LNN
        hidden_dim = config.model.get("hidden_dim", 64)
        model = LNNSequenceModel(input_dim=input_dim, hidden_dim=hidden_dim, proj_dim=input_dim)
        
        # Train 1 epoch fast
        from hdlnn.eval.harness import prepare_sequences
        xs, ys, dts = prepare_sequences(train_flows, train_inputs, seq_len=16)
        if len(xs) > 0:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
            criterion = torch.nn.MSELoss()
            model.train()
            for b in range(0, len(xs), 64):
                xb = xs[b : b + 64]
                yb = ys[b : b + 64]
                dtb = dts[b : b + 64]
                optimizer.zero_grad()
                pred_seq, _ = model(xb, dt=dtb)
                pred_seq_proj = model.predict_next_vector(pred_seq)
                loss = criterion(pred_seq_proj, yb)
                loss.backward()
                optimizer.step()
        
        # Reference Manifold calibration
        model.eval()
        val_states_list = []
        entity_states = {}
        for flow, vec in zip(train_flows, train_inputs):
            eid = flow.entity_id
            h_prev = entity_states.get(eid, torch.zeros(hidden_dim))
            dt_val = torch.tensor([flow.dt])
            with torch.no_grad():
                h_next = model.step(vec, h_prev, dt_val).squeeze(0)
            entity_states[eid] = h_next
            val_states_list.append(h_next)
            
        val_states = torch.stack(val_states_list)
        
        scorer = DivergenceScorer(mode=config.divergence.get("mode", "mahalanobis"), hidden_dim=hidden_dim, model=model)
        scorer.manifold.fit(val_states)
        
        # Initialize streaming sidecar dependencies
        simd_encoder = SIMDEncoder(encoder)
        alert_handler = SIEMAlertHandler(config)
        
        yield "data: " + json.dumps({"type": "system", "message": "Calibrated successfully. Beginning real-time ingestion loop."}) + "\n\n"
        await asyncio.sleep(0.1)
        
        # Stream ingestion loop
        total_anomalies = 0
        processed_count = 0
        latencies_ms = []
        start_sim_time = time.perf_counter()
        
        # Reset sidecar entity states
        sidecar_states = {}
        
        for idx, flow in enumerate(test_flows):
            if stop_requested:
                yield "data: " + json.dumps({"type": "system", "message": "Stream simulation terminated by user."}) + "\n\n"
                break
                
            start_step = time.perf_counter()
            
            # SIMD encode
            encoded = simd_encoder.encode(flow)
            
            # Sequential inference
            eid = flow.entity_id
            h_prev = sidecar_states.get(eid, torch.zeros(hidden_dim))
            dt_step = torch.tensor([flow.dt])
            with torch.no_grad():
                h_next = model.step(encoded.vector, h_prev, dt_step).squeeze(0)
            sidecar_states[eid] = h_next
            
            # Dynamic drift check
            score = scorer.score(h_next)
            threshold = k * scorer.manifold.mean
            is_anomaly = bool(score > threshold)
            
            step_latency = time.perf_counter() - start_step
            latencies_ms.append(step_latency * 1000.0)
            processed_count += 1
            
            # Throttle ingestion rate (speed sleep)
            await asyncio.sleep(1.0 / speed)
            
            # Alerting
            if is_anomaly:
                total_anomalies += 1
                decision = AnomalyDecision(drift_score=score, threshold=threshold, is_anomaly=True)
                cef_string = alert_handler.dispatch_alert(eid, decision, flow.timestamp)
                yield "data: " + json.dumps({
                    "type": "alert",
                    "alert_text": cef_string
                }) + "\n\n"
            
            # Progress update
            elapsed = time.perf_counter() - start_sim_time
            current_throughput = processed_count / max(0.0001, elapsed)
            
            yield "data: " + json.dumps({
                "type": "progress",
                "index": idx,
                "score": float(score),
                "threshold": float(threshold),
                "is_anomaly": is_anomaly,
                "latency": float(step_latency),
                "attack_cat": flow.categorical_fields.get("attack_cat") or ("Drift Anomaly" if is_anomaly else "Normal"),
                "throughput": current_throughput,
                "processed": processed_count,
                "anomalies": total_anomalies
            }) + "\n\n"

        # Simulation complete event
        avg_lat = sum(latencies_ms) / max(1, len(latencies_ms))
        yield "data: " + json.dumps({
            "type": "complete",
            "total_processed": processed_count,
            "total_anomalies": total_anomalies,
            "avg_latency_ms": avg_lat
        }) + "\n\n"

    return StreamingResponse(sse_event_generator(), media_type="text/event-stream")

def run_server(port: int = 8050):
    logger.info(f"Starting HYDRA-LNN v2 Dashboard server on http://localhost:{port} ...")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")

if __name__ == "__main__":
    run_server()
