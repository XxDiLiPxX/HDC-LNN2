import time
import json
import yaml
import logging
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from hdlnn.contracts.schemas import CanonicalFlow, EncodedHypervector, TrajectoryState
from hdlnn.data.loaders import load_raw_unsw_nb15
from hdlnn.data.splitter import split_dataset
from hdlnn.data.quality import verify_split_quality
from hdlnn.eval.injector import inject_threat_scenarios
from hdlnn.eval.metrics import compute_metrics_suite
from hdlnn.hdc.encoder import RecordEncoder
from hdlnn.lnn.model import LNNSequenceModel
from hdlnn.baselines.lstm import LSTMBaseline
from hdlnn.baselines.cnn import CNN1DBaseline
from hdlnn.divergence.scorer import DivergenceScorer

logger = logging.getLogger(__name__)

class RawFeatureMapper:
    """Helper to convert CanonicalFlow categorical/numerical fields into raw flat feature tensors."""
    def __init__(self, categorical_columns: List[str], numerical_columns: List[str]):
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.vocab: Dict[str, Dict[str, int]] = {}
        self.feature_dim = 0

    def fit(self, train_flows: List[CanonicalFlow]) -> None:
        # Build one-hot mappings for categoricals
        self.vocab = {}
        dim = len(self.numerical_columns)
        for col in self.categorical_columns:
            self.vocab[col] = {"<OOV>": 0}
            for flow in train_flows:
                val = flow.categorical_fields.get(col, "-")
                if val not in self.vocab[col]:
                    self.vocab[col][val] = len(self.vocab[col])
            dim += len(self.vocab[col])
        self.feature_dim = dim
        logger.info(f"Fitted RawFeatureMapper: Total dimension = {self.feature_dim}")

    def transform(self, flows: List[CanonicalFlow]) -> torch.Tensor:
        num_flows = len(flows)
        tensor = torch.zeros((num_flows, self.feature_dim), dtype=torch.float32)
        
        for i, flow in enumerate(flows):
            # 1. Fill numericals
            curr_idx = 0
            for col in self.numerical_columns:
                tensor[i, curr_idx] = flow.numerical_fields.get(col, 0.0)
                curr_idx += 1
            
            # 2. Fill one-hot categoricals
            for col in self.categorical_columns:
                val = flow.categorical_fields.get(col, "-")
                val_idx = self.vocab[col].get(val, 0)  # default to OOV index 0
                tensor[i, curr_idx + val_idx] = 1.0
                curr_idx += len(self.vocab[col])
                
        return tensor

def prepare_sequences(
    flows: List[CanonicalFlow], 
    input_tensors: torch.Tensor, 
    seq_len: int = 16
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Slices entity events into overlapping chunks of length seq_len for training."""
    entity_groups: Dict[str, List[Tuple[torch.Tensor, float]]] = {}
    for flow, vec in zip(flows, input_tensors):
        entity_groups.setdefault(flow.entity_id, []).append((vec, flow.dt))
        
    xs, ys, dts = [], [], []
    for entity_id, sequence in entity_groups.items():
        if len(sequence) < 2:
            continue
        
        # Sliding window sequence creation
        for i in range(0, len(sequence) - seq_len, seq_len // 2 or 1):
            chunk = sequence[i : i + seq_len + 1]
            if len(chunk) < 2:
                continue
                
            x_chunk = torch.stack([item[0] for item in chunk[:-1]])
            dt_chunk = torch.tensor([item[1] for item in chunk[:-1]])
            y_chunk = torch.stack([item[0] for item in chunk[1:]])
            
            xs.append(x_chunk)
            ys.append(y_chunk)
            dts.append(dt_chunk)
            
    if not xs:
        # Fallback if dataset is too small
        D = input_tensors.size(1)
        return torch.zeros((0, seq_len, D)), torch.zeros((0, seq_len, D)), torch.zeros((0, seq_len))
        
    return torch.stack(xs), torch.stack(ys), torch.stack(dts)

def run_experiment(
    config: Any,
    run_id: str,
    baseline_name: str = "hdc-lnn",
    inject_attacks: bool = True,
    limit: Optional[int] = None,
    output_dir: Path = Path("runs")
) -> Dict[str, Any]:
    """Runs a complete training, validation, and testing experiment for a given model baseline.
    
    Generates self-contained run folder under runs/<run_id>/ containing config, metrics, and decisions.
    """
    logger.info(f"Starting experiment {run_id} using model baseline: {baseline_name}")
    start_time_exp = time.perf_counter()

    # 1. Load data
    data_dir = Path(".")
    flows = load_raw_unsw_nb15(config, data_dir, limit=limit)
    
    # 2. Inject threat scenarios
    if inject_attacks:
        flows = inject_threat_scenarios(
            flows, 
            seed=config.seed,
            categorical_cols=config.categorical_columns,
            numerical_cols=config.numerical_columns
        )

    # 3. Split dataset
    train_flows, val_flows, test_flows = split_dataset(
        flows,
        train_ratio=config.get("train_ratio", 0.70),
        val_ratio=config.get("val_ratio", 0.15),
        test_ratio=config.get("test_ratio", 0.15),
        seed=config.seed
    )

    # 4. Verify quality & Audit Leakage
    quality_passed = verify_split_quality(train_flows, val_flows, test_flows)
    if not quality_passed:
        raise ValueError("Data pipeline split quality verification failed.")
        
    from hdlnn.data.quality import audit_pipeline_leakage
    audit_pipeline_leakage(
        train_flows, val_flows, test_flows,
        categorical_cols=config.categorical_columns,
        numerical_cols=config.numerical_columns
    )

    D = config.hdc.get("dimension", 10000)
    
    # 5. Extract inputs based on baseline
    # HDC-LNN, LSTM, CNN, HDC-only use HDC encoded inputs
    # LNN-only uses raw flat features
    is_raw_baseline = (baseline_name == "lnn-only")
    is_hdc_only = (baseline_name == "hdc-only")

    logger.info("Preparing model inputs...")
    if is_raw_baseline:
        mapper = RawFeatureMapper(config.categorical_columns, config.numerical_columns)
        mapper.fit(train_flows)
        input_dim = mapper.feature_dim
        
        train_inputs = mapper.transform(train_flows)
        val_inputs = mapper.transform(val_flows)
        test_inputs = mapper.transform(test_flows)
    else:
        # Fit HDC Encoder
        encoder = RecordEncoder(
            D=D,
            categorical_columns=config.categorical_columns,
            numerical_columns=config.numerical_columns
        )
        encoder.fit(train_flows)
        input_dim = D
        
        # Batch encode
        train_inputs = torch.stack([ev.vector for ev in encoder.encode_batch(train_flows)])
        val_inputs = torch.stack([ev.vector for ev in encoder.encode_batch(val_flows)])
        test_inputs = torch.stack([ev.vector for ev in encoder.encode_batch(test_flows)])

    # 6. Initialize baseline model
    hidden_dim = config.model.get("hidden_dim", 64)
    model: Optional[nn.Module] = None
    
    if baseline_name in ("hdc-lnn", "lnn-only"):
        model = LNNSequenceModel(input_dim=input_dim, hidden_dim=hidden_dim, proj_dim=input_dim)
    elif baseline_name == "lstm":
        model = LSTMBaseline(input_dim=input_dim, hidden_dim=hidden_dim, proj_dim=input_dim)
    elif baseline_name == "cnn":
        # CNN state size includes the input history
        model = CNN1DBaseline(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=3, proj_dim=input_dim)
    elif is_hdc_only:
        model = None
    else:
        raise ValueError(f"Unknown baseline name: {baseline_name}")

    # 7. Training loop (skipped for HDC-only baseline)
    if model is not None:
        logger.info("Preparing sequence batches for training...")
        # Prepare training sequences
        xs, ys, dts = prepare_sequences(train_flows, train_inputs, seq_len=16)
        
        if len(xs) > 0:
            logger.info(f"Training model ({len(xs)} sequences)...")
            optimizer = torch.optim.Adam(model.parameters(), lr=config.model.get("lr", 0.001))
            criterion = nn.MSELoss()
            
            epochs = config.model.get("epochs", 5)
            batch_size = config.model.get("batch_size", 64)
            
            model.train()
            for epoch in range(epochs):
                epoch_loss = 0.0
                num_batches = 0
                
                # Simple batching loop
                for b in range(0, len(xs), batch_size):
                    xb = xs[b : b + batch_size]
                    yb = ys[b : b + batch_size]
                    dtb = dts[b : b + batch_size]
                    
                    optimizer.zero_grad()
                    # Forward pass
                    if isinstance(model, LNNSequenceModel):
                        out_seq, _ = model.forward(xb, dt=dtb)
                    else:
                        out_seq, _ = model.forward(xb)
                        
                    # Project next-vector prediction
                    pred_seq = model.predict_next_vector(out_seq)
                    
                    loss = criterion(pred_seq, yb)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                
                avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
                logger.info(f"Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.6f}")
        else:
            logger.warning("No sequences prepared for training (dataset too small).")

    # 8. Fit Scorer parameters using validation normal states
    logger.info("Computing validation hidden states for normal traffic...")
    val_normal_indices = [i for i, f in enumerate(val_flows) if f.label == 0]
    
    scoring_mode = config.divergence.get("mode", "mahalanobis")
    
    if baseline_name == "hdc-only":
        scorer_dim = input_dim
    else:
        scorer_dim = hidden_dim
        
    scorer = DivergenceScorer(
        mode=scoring_mode,
        threshold_k=config.divergence.get("threshold_k", 3.0),
        hidden_dim=scorer_dim,
        model=model
    )

    # We evaluate states for ALL validation flows (both normal and attacks) to perform threshold calibration
    val_states_list = []
    entity_states: Dict[str, torch.Tensor] = {}
    
    if model is not None:
        model.eval()
        for idx, flow in enumerate(val_flows):
            eid = flow.entity_id
            if eid not in entity_states:
                if baseline_name == "cnn":
                    state_dim = 2 * input_dim + hidden_dim
                else:
                    state_dim = hidden_dim
                entity_states[eid] = torch.zeros(state_dim)
                
            h_prev = entity_states[eid]
            x_val = val_inputs[idx]
            dt_val = torch.tensor([flow.dt])
            
            with torch.no_grad():
                h_next = model.step(x_val, h_prev, dt_val).squeeze(0)
            entity_states[eid] = h_next
            
            # Extract score state slice for CNN
            if baseline_name == "cnn":
                state_to_score = h_next[-hidden_dim:]
            else:
                state_to_score = h_next
            val_states_list.append(state_to_score)
            
        # Fit Reference Manifold using validation NORMAL states only
        val_normal_states = torch.stack([val_states_list[i] for i, f in enumerate(val_flows) if f.label == 0])
        scorer.manifold.fit(val_normal_states)
    else:
        # HDC-only baseline: states are the raw hypervectors
        for idx, flow in enumerate(val_flows):
            val_states_list.append(val_inputs[idx])
            
        val_normal_states = torch.stack([val_states_list[i] for i, f in enumerate(val_flows) if f.label == 0])
        scorer.manifold.fit(val_normal_states)

    # Perform Validation-Calibrated Anomaly Thresholding Grid Search
    logger.info("Calibrating decision thresholds on validation split...")
    import numpy as np
    from sklearn.metrics import f1_score
    
    val_labels = np.array([f.label for f in val_flows])
    
    if scoring_mode == "mahalanobis":
        val_scores = []
        for state in val_states_list:
            dist_tensor = scorer.manifold.compute_mahalanobis_distance(state)
            val_scores.append(float(dist_tensor.item()))
            
        val_scores = np.array(val_scores)
        
        # Grid search threshold to maximize F1-score on validation split
        best_f1 = -1.0
        best_threshold = 0.0
        
        # Candidates from min to max validation scores
        min_score = float(val_scores.min())
        max_score = float(val_scores.max())
        candidates = np.linspace(min_score, max_score, 100)
        
        has_positives = np.sum(val_labels == 1) > 0
        if not has_positives:
            # Fallback to mean + 3.0 * std if no validation attacks are present
            best_threshold = float(val_scores.mean() + 3.0 * val_scores.std())
        else:
            for t in candidates:
                preds = (val_scores > t).astype(int)
                f1 = f1_score(val_labels, preds, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = t
                    
        scorer.mahalanobis_threshold = best_threshold
        logger.info(f"Validation Calibrated Mahalanobis Threshold: {scorer.mahalanobis_threshold:.4f} (Validation F1: {max(best_f1, 0.0):.4f})")
        
    elif scoring_mode == "cosine":
        # Grid search threshold_k multiplier for EWMA
        best_f1 = -1.0
        best_k = 3.0
        
        has_positives = np.sum(val_labels == 1) > 0
        if not has_positives:
            best_k = 3.0
        else:
            k_candidates = np.linspace(0.5, 5.0, 46)
            for k in k_candidates:
                # Reset EWMA stats for each candidate evaluation
                scorer.ewma_stats = {}
                preds = []
                for idx, (flow, state) in enumerate(zip(val_flows, val_states_list)):
                    scorer.register_observed(flow.entity_id, val_inputs[idx])
                    scorer.threshold_k = k
                    
                    traj = TrajectoryState(flow.entity_id, flow.timestamp, state)
                    dec = scorer.score(traj)
                    preds.append(1 if dec.is_anomaly else 0)
                    
                f1 = f1_score(val_labels, preds, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_k = k
                    
        scorer.threshold_k = best_k
        scorer.ewma_stats = {}  # Reset EWMA stats for clean test evaluation
        logger.info(f"Validation Calibrated Cosine EWMA multiplier k: {scorer.threshold_k:.2f} (Validation F1: {max(best_f1, 0.0):.4f})")

    # 9. Evaluate Test Split Step-by-Step
    logger.info("Evaluating test split step-by-step...")
    test_labels = []
    test_scores = []
    test_preds = []
    
    # Store decision records
    records_list = []
    
    test_entity_states: Dict[str, torch.Tensor] = {}
    start_eval = time.perf_counter()
    
    for idx, flow in enumerate(test_flows):
        x_test = test_inputs[idx]
        dt_test = torch.tensor([flow.dt])
        eid = flow.entity_id
        
        if model is not None:
            if eid not in test_entity_states:
                if baseline_name == "cnn":
                    state_dim = 2 * input_dim + hidden_dim
                else:
                    state_dim = hidden_dim
                test_entity_states[eid] = torch.zeros(state_dim)
                
            h_prev = test_entity_states[eid]
            with torch.no_grad():
                h_next = model.step(x_test, h_prev, dt_test).squeeze(0)
            test_entity_states[eid] = h_next
            if baseline_name == "cnn":
                state_to_score = h_next[-hidden_dim:]
            else:
                state_to_score = h_next
        else:
            # HDC-only: the "state" is the raw hypervector itself
            state_to_score = x_test
            
        # Register observation for Cosine scorer
        scorer.register_observed(eid, x_test)
        
        # Score anomaly
        traj = TrajectoryState(eid, flow.timestamp, state_to_score)
        decision = scorer.score(traj)
        
        test_labels.append(flow.label)
        test_scores.append(decision.drift_score)
        test_preds.append(1 if decision.is_anomaly else 0)
        
        records_list.append({
            "entity_id": eid,
            "timestamp": flow.timestamp,
            "actual_label": flow.label,
            "predicted_label": 1 if decision.is_anomaly else 0,
            "drift_score": decision.drift_score,
            "threshold": decision.threshold
        })
        
    eval_duration = time.perf_counter() - start_eval
    logger.info(f"Test evaluation completed in {eval_duration:.4f} seconds.")

    # 10. Compute full metrics suite
    metrics = compute_metrics_suite(
        labels=test_labels,
        scores=test_scores,
        predictions=test_preds,
        latency_seconds=eval_duration,
        num_samples=len(test_flows)
    )
    logger.info(f"Experiment {run_id} classification F1-Score = {metrics['f1_score']:.4f}")

    # 11. Write self-contained run folder
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config.to_dict(), f)
        
    # Save metrics
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
        
    # Save decisions as Parquet
    df_decisions = pd.DataFrame(records_list)
    df_decisions.to_parquet(run_dir / "decisions.parquet", index=False)
    
    logger.info(f"Run artifacts successfully committed to {run_dir}")
    
    return metrics
