import time
import torch
import logging
import threading
from typing import List, Dict, Any, Optional
from hdlnn.contracts.schemas import CanonicalFlow, TrajectoryState, AnomalyDecision
from hdlnn.deploy.simd_backend import SIMDEncoder
from hdlnn.deploy.kafka_ingest import LogIngestRingBuffer
from hdlnn.deploy.soar_connector import SIEMAlertHandler
from hdlnn.divergence.scorer import DivergenceScorer
from hdlnn.lnn.model import LNNSequenceModel

logger = logging.getLogger(__name__)

class StreamingSidecar:
    """Production daemon sidecar managing real-time flow ingestion, inference, and SIEM alerting."""
    def __init__(
        self,
        config: Any,
        model: LNNSequenceModel,
        scorer: DivergenceScorer,
        encoder: SIMDEncoder
    ):
        self.config = config
        self.model = model
        self.scorer = scorer
        self.encoder = encoder
        
        self.ring_buffer = LogIngestRingBuffer(max_capacity=100000)
        self.alert_handler = SIEMAlertHandler(config)
        
        self.entity_states: Dict[str, torch.Tensor] = {}
        self.processed_count = 0
        self.running = False
        
        # Benchmarking metrics
        self.latencies: List[float] = []

    def feed_flows_async(self, flows: List[CanonicalFlow]) -> None:
        """Helper to feed flows into the ring buffer from another thread."""
        for flow in flows:
            self.ring_buffer.put(flow)

    def start_processing(self, limit: Optional[int] = None) -> None:
        """Starts the consumer processing loop.
        
        Processes CanonicalFlows from the ring buffer, runs SIMD encoding, LNN inference,
        behavioral scoring, and dispatches alerts.
        """
        self.running = True
        self.processed_count = 0
        self.latencies.clear()
        
        logger.info("StreamingSidecar ingest processing loop started.")
        self.model.eval()
        
        start_time = time.perf_counter()
        
        while self.running:
            # Poll with timeout
            flow = self.ring_buffer.get(timeout=0.1)
            if flow is None:
                # Buffer empty, check if we've processed everything in batch mode
                if limit is not None and self.processed_count >= limit:
                    break
                continue
                
            step_start = time.perf_counter()
            eid = flow.entity_id
            
            # 1. SIMD encoding
            encoded_hv = self.encoder.encode(flow)
            
            # 2. Maintain continuous LNN sequence trajectory per Source IP
            if eid not in self.entity_states:
                self.entity_states[eid] = torch.zeros(self.model.cfc.state_size)
                
            h_prev = self.entity_states[eid]
            x_step = encoded_hv.vector
            dt_step = torch.tensor([flow.dt])
            
            # 3. LNN Step (continuous decay)
            with torch.no_grad():
                h_next = self.model.step(x_step, h_prev, dt_step).squeeze(0)
            self.entity_states[eid] = h_next
            
            # 4. Behavioral Score Anomaly
            traj = TrajectoryState(eid, flow.timestamp, h_next)
            decision = self.scorer.score(traj)
            
            # 5. SIEM Alerting on positive verdict
            if decision.is_anomaly:
                self.alert_handler.dispatch_alert(eid, decision, flow.timestamp)
                
            step_latency = time.perf_counter() - step_start
            self.latencies.append(step_latency)
            
            self.processed_count += 1
            if limit is not None and self.processed_count >= limit:
                break
                
        total_time = time.perf_counter() - start_time
        throughput = self.processed_count / total_time if total_time > 0 else 0.0
        avg_latency = (sum(self.latencies) / len(self.latencies)) * 1000.0 if self.latencies else 0.0
        
        logger.info(
            f"StreamingSidecar loop completed. Processed: {self.processed_count} events. "
            f"Throughput: {throughput:.2f} flows/sec. Average latency: {avg_latency:.4f} ms/flow."
        )

    def stop(self) -> None:
        """Stops the processing loop."""
        self.running = False

def run_streaming_sidecar_demo(config: Any, limit: Optional[int] = None) -> None:
    """End-to-end streaming live emulator demo for Phase 4 validation."""
    from pathlib import Path
    from hdlnn.data.loaders import load_raw_unsw_nb15
    from hdlnn.data.splitter import split_dataset
    from hdlnn.eval.injector import inject_threat_scenarios
    from hdlnn.data.quality import audit_pipeline_leakage
    from hdlnn.hdc.item_memory import ItemMemory
    from hdlnn.hdc.encoder import RecordEncoder
    from hdlnn.lnn.model import LNNSequenceModel
    from hdlnn.divergence.scorer import DivergenceScorer
    from hdlnn.contracts.schemas import TrajectoryState
    import numpy as np
    
    logger.info("Initializing Streaming Sidecar Demo...")
    data_dir = Path("data")
    
    # Load flows
    flows = load_raw_unsw_nb15(config, data_dir, limit=limit)
    flows = inject_threat_scenarios(flows, seed=config.seed)
    
    # Split
    train_flows, val_flows, test_flows = split_dataset(
        flows,
        train_ratio=config.get("train_ratio", 0.70),
        val_ratio=config.get("val_ratio", 0.15),
        test_ratio=config.get("test_ratio", 0.15),
        seed=config.seed
    )
    
    # Audit Leakage
    audit_pipeline_leakage(
        train_flows, val_flows, test_flows,
        categorical_cols=config.categorical_columns,
        numerical_cols=config.numerical_columns
    )
    
    # Fit RecordEncoder and Item Memory on Train ONLY
    input_dim = 16384  # Production StreamingSidecar uses 16384-bit SIMDEncoder
    hidden_dim = config.model.get("hidden_dim", 64)
    
    encoder = RecordEncoder(
        D=input_dim,
        categorical_columns=config.categorical_columns,
        numerical_columns=config.numerical_columns
    )
    encoder.fit(train_flows)
    
    train_inputs = torch.stack([ev.vector for ev in encoder.encode_batch(train_flows)])
    val_inputs = torch.stack([ev.vector for ev in encoder.encode_batch(val_flows)])
    
    # SIMD Encoder (16,384-bit packed binary hypervectors)
    simd_encoder = SIMDEncoder(encoder)
    
    # LNN Model
    model = LNNSequenceModel(input_dim=input_dim, hidden_dim=hidden_dim, proj_dim=input_dim)
    
    # Pre-train briefly on training data sequences (1 fast epoch)
    from hdlnn.eval.harness import prepare_sequences
    xs, ys, dts = prepare_sequences(train_flows, train_inputs, seq_len=16)
    if len(xs) > 0:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        criterion = torch.nn.MSELoss()
        model.train()
        for epoch in range(1):
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
                
    # Scorer setup
    scoring_mode = config.divergence.get("mode", "mahalanobis")
    scorer = DivergenceScorer(
        mode=scoring_mode,
        threshold_k=config.divergence.get("threshold_k", 3.0),
        hidden_dim=hidden_dim,
        model=model
    )
    
    # Score validation to calibrate threshold
    model.eval()
    val_states_list = []
    entity_states = {}
    for idx, flow in enumerate(val_flows):
        eid = flow.entity_id
        if eid not in entity_states:
            entity_states[eid] = torch.zeros(hidden_dim)
        h_prev = entity_states[eid]
        x_val = val_inputs[idx]
        dt_val = torch.tensor([flow.dt])
        with torch.no_grad():
            h_next = model.step(x_val, h_prev, dt_val).squeeze(0)
        entity_states[eid] = h_next
        val_states_list.append(h_next)
        
    val_normal_states = torch.stack([val_states_list[i] for i, f in enumerate(val_flows) if f.label == 0])
    scorer.manifold.fit(val_normal_states)
    
    # Static threshold calibration
    val_scores = []
    for state in val_states_list:
        dist = scorer.manifold.compute_mahalanobis_distance(state)
        val_scores.append(float(dist.item()))
    val_scores = np.array(val_scores)
    
    # Calibrate static threshold
    from sklearn.metrics import f1_score
    val_labels = np.array([f.label for f in val_flows])
    best_f1 = -1.0
    best_t = 3.0
    candidates = np.linspace(val_scores.min(), val_scores.max(), 100)
    for t in candidates:
        preds = (val_scores > t).astype(int)
        f1 = f1_score(val_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    scorer.mahalanobis_threshold = best_t
    
    # Initialize Sidecar
    sidecar = StreamingSidecar(config, model, scorer, simd_encoder)
    
    # Feed test flows asynchronously to simulate ingestion
    feed_thread = threading.Thread(target=sidecar.feed_flows_async, args=(test_flows,))
    feed_thread.start()
    
    # Start consumer loop
    sidecar.start_processing(limit=len(test_flows))
    feed_thread.join()
