import time
import json
import hashlib
import yaml
import logging
import gc
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from hdlnn.contracts.schemas import CanonicalFlow, EncodedHypervector, TrajectoryState
from hdlnn.data.loaders import load_dataset_flows
from hdlnn.data.splitter import split_dataset
from hdlnn.data.quality import verify_split_quality
from hdlnn.eval.injector import inject_threat_scenarios
from hdlnn.eval.metrics import compute_metrics_suite
from hdlnn.eval.calibration import calibrate_threshold
from hdlnn.hdc.encoder import RecordEncoder
from hdlnn.lnn.model import LNNSequenceModel
from hdlnn.baselines.lstm import LSTMBaseline
from hdlnn.baselines.cnn import CNN1DBaseline
from hdlnn.baselines.autoencoder import AETemporalModel
from hdlnn.divergence.scorer import DivergenceScorer
from hdlnn.divergence.ae_scorer import AEReconstructionScorer

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
        # Fallback if no full sequences of length seq_len can be formed:
        # Try progressively smaller sequence lengths (8, 4, 2)
        for fallback_len in (8, 4, 2):
            for entity_id, sequence in entity_groups.items():
                if len(sequence) > fallback_len:
                    for i in range(0, len(sequence) - fallback_len, fallback_len // 2 or 1):
                        chunk = sequence[i : i + fallback_len + 1]
                        if len(chunk) == fallback_len + 1:
                            x_chunk = torch.stack([item[0] for item in chunk[:-1]])
                            dt_chunk = torch.tensor([item[1] for item in chunk[:-1]])
                            y_chunk = torch.stack([item[0] for item in chunk[1:]])
                            xs.append(x_chunk)
                            ys.append(y_chunk)
                            dts.append(dt_chunk)
            if xs:
                break
        if not xs and len(input_tensors) > 0:
            # 1-step sequence fallback [N, 1, D]
            return input_tensors.unsqueeze(1), input_tensors.unsqueeze(1), torch.zeros((len(input_tensors), 1))
        elif not xs:
            D = input_tensors.size(1)
            return torch.zeros((0, seq_len, D)), torch.zeros((0, seq_len, D)), torch.zeros((0, seq_len))
        
    return torch.stack(xs), torch.stack(ys), torch.stack(dts)

def run_experiment(
    config: Any,
    run_id: str,
    baseline_name: str = "hdc-lnn",
    inject_attacks: bool = True,
    limit: Optional[int] = None,
    output_dir: Path = Path("runs"),
    source_file: Optional[str] = None,
    stride: int = 1
) -> Dict[str, Any]:
    """Runs a complete training, validation, and testing experiment for a given model baseline.
    
    Generates self-contained run folder under runs/<run_id>/ containing config, metrics, and decisions.
    """
    logger.info(f"Starting experiment {run_id} using model baseline: {baseline_name}")
    start_time_exp = time.perf_counter()
    from hdlnn.common.seeding import set_seed
    set_seed(config.seed)

    # 1. Load data
    data_dir = Path(".")
    if source_file:
        flows = load_dataset_flows(config, data_dir, source_file=source_file, limit=limit, stride=stride)
    else:
        raise ValueError("source_file is required for the current local dataset pipeline.")
    
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
        use_log = bool(config.get("use_log_transform", False))
        encoder = RecordEncoder(
            D=D,
            categorical_columns=config.categorical_columns,
            numerical_columns=config.numerical_columns,
            use_log_transform=use_log
        )
        encoder.fit(train_flows)
        input_dim = D
        
        # Batch direct tensor encode (memory efficient)
        train_inputs = encoder.encode_tensor(train_flows)
        
        logger.info("Encoding validation flows...")
        val_inputs = encoder.encode_tensor(val_flows)
        
        logger.info("Encoding test flows...")
        test_inputs = encoder.encode_tensor(test_flows)

    # 6. Initialize baseline model
    hidden_dim = config.model.get("hidden_dim", 64)
    model: Optional[nn.Module] = None
    
    if baseline_name in ("hdc-lnn", "lnn-only"):
        model = LNNSequenceModel(input_dim=input_dim, hidden_dim=hidden_dim, proj_dim=input_dim)
    elif baseline_name == "lstm":
        model = LSTMBaseline(input_dim=input_dim, hidden_dim=hidden_dim, proj_dim=input_dim)
    elif baseline_name == "cnn":
        model = CNN1DBaseline(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=3, proj_dim=input_dim)
    elif baseline_name == "autoencoder":
        model = AETemporalModel(input_dim=input_dim, hidden_dim=hidden_dim)
    elif baseline_name == "mamba2":
        from hdlnn.baselines.mamba import MambaSequenceModel
        model = MambaSequenceModel(input_dim=input_dim, hidden_dim=hidden_dim, proj_dim=input_dim)
    elif is_hdc_only:
        model = None
    else:
        raise ValueError(f"Unknown baseline name: {baseline_name}")

    # 7. Training loop (skipped for HDC-only baseline)
    if model is not None:
        logger.info("Preparing sequence batches for training...")
        seq_len = config.model.get("sequence_length", 16)
        
        # Build sequence indices directly into train_inputs to avoid 3 GB of redundant tensor duplication
        entity_groups = {}
        for idx, f in enumerate(train_flows):
            if baseline_name == "autoencoder" and f.label != 0:
                continue
            entity_groups.setdefault(f.entity_id, []).append((idx, f.dt))
            
        seq_indices = []
        for eid, seq in entity_groups.items():
            if len(seq) < 2:
                continue
            stride = seq_len // 2 or 1
            for i in range(0, len(seq) - seq_len, stride):
                chunk = seq[i : i + seq_len + 1]
                if len(chunk) == seq_len + 1:
                    x_idxs = [item[0] for item in chunk[:-1]]
                    y_idxs = [item[0] for item in chunk[1:]]
                    dts = [item[1] for item in chunk[:-1]]
                    seq_indices.append((x_idxs, y_idxs, dts))
        
        if not seq_indices:
            # Fallback for short entities
            for fallback_len in (8, 4, 2):
                for eid, seq in entity_groups.items():
                    if len(seq) > fallback_len:
                        for i in range(0, len(seq) - fallback_len, fallback_len // 2 or 1):
                            chunk = seq[i : i + fallback_len + 1]
                            if len(chunk) == fallback_len + 1:
                                x_idxs = [item[0] for item in chunk[:-1]]
                                y_idxs = [item[0] for item in chunk[1:]]
                                dts = [item[1] for item in chunk[:-1]]
                                seq_indices.append((x_idxs, y_idxs, dts))
                if seq_indices:
                    break

        if len(seq_indices) > 0:
            logger.info(f"Training model ({len(seq_indices)} sequences)...")
            lr = 0.005 if baseline_name == "autoencoder" else config.model.get("lr", 0.001)
            epochs = 15 if baseline_name == "autoencoder" else config.model.get("epochs", 5)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=lr, weight_decay=config.model.get("weight_decay", 0.0)
            )
            criterion = nn.MSELoss()
            batch_size = config.model.get("batch_size", 64)
            
            model.train()
            for epoch in range(epochs):
                epoch_loss = 0.0
                num_batches = 0
                
                for b in range(0, len(seq_indices), batch_size):
                    batch_items = seq_indices[b : b + batch_size]
                    xb = torch.stack([train_inputs[item[0]] for item in batch_items]).to(torch.float32)
                    yb = torch.stack([train_inputs[item[1]] for item in batch_items]).to(torch.float32)
                    dtb = torch.tensor([item[2] for item in batch_items], dtype=torch.float32)
                    
                    optimizer.zero_grad()
                    if baseline_name == "autoencoder":
                        recon_seq, _ = model.forward(xb)
                        loss = criterion(recon_seq, xb)
                    else:
                        if isinstance(model, LNNSequenceModel):
                            out_seq, _ = model.forward(xb, dt=dtb)
                        else:
                            out_seq, _ = model.forward(xb)
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
            
        del seq_indices
        del train_inputs
        gc.collect()

    # 8. Fit Scorer parameters using validation normal states
    logger.info("Computing validation hidden states for normal traffic...")
    val_normal_indices = [i for i, f in enumerate(val_flows) if f.label == 0]
    
    scoring_mode = config.divergence.get("mode", "mahalanobis")
    
    if baseline_name == "hdc-only":
        scorer_dim = input_dim
    else:
        scorer_dim = hidden_dim
        
    if baseline_name == "autoencoder":
        scorer = AEReconstructionScorer(
            model=model,
            threshold_k=config.divergence.get("ae_threshold_k", 1.0)
        )
    else:
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
                elif baseline_name in ("autoencoder", "lstm"):
                    state_dim = 2 * hidden_dim
                elif baseline_name == "mamba2":
                    state_dim = 2 * hidden_dim * 16
                else:
                    state_dim = hidden_dim
                entity_states[eid] = torch.zeros(1, state_dim)
                
            h_prev = entity_states[eid]
            x_val = val_inputs[idx].to(torch.float32).unsqueeze(0)
            dt_val = torch.tensor([[flow.dt]], dtype=torch.float32)
            
            with torch.no_grad():
                h_next = model.step(x_val, h_prev, dt_val)
            entity_states[eid] = h_next
            
            # Extract score state slice for CNN, Mamba2, and LSTM
            if baseline_name == "cnn":
                state_to_score = h_next[0, -hidden_dim:]
            elif baseline_name in ("mamba2", "lstm"):
                state_to_score = h_next[0, :hidden_dim]
            else:
                state_to_score = h_next[0]
            val_states_list.append(state_to_score)
            
        # Collect validation NORMAL states only for one-class threshold calibration
        val_normal_indices = [i for i, f in enumerate(val_flows) if f.label == 0]
        if len(val_normal_indices) > 0:
            val_normal_states = torch.stack([val_states_list[i] for i in val_normal_indices])
        else:
            logger.warning("No normal flows (label=0) present in validation split. Falling back to training normal states for threshold calibration.")
            train_normal_indices = [i for i, f in enumerate(train_flows) if f.label == 0]
            if len(train_normal_indices) > 0:
                train_normal_states_list = []
                entity_states_train = {}
                normal_train_flows = [train_flows[i] for i in train_normal_indices]
                if is_raw_baseline:
                    normal_train_inputs = mapper.transform(normal_train_flows)
                else:
                    normal_train_inputs = encoder.encode_tensor(normal_train_flows)
                for idx, flow in enumerate(normal_train_flows):
                    eid = flow.entity_id
                    h_prev = entity_states_train.get(eid, torch.zeros(1, state_dim))
                    x_tr = normal_train_inputs[idx].to(torch.float32).unsqueeze(0)
                    dt_tr = torch.tensor([[flow.dt]], dtype=torch.float32)
                    with torch.no_grad():
                        h_next = model.step(x_tr, h_prev, dt_tr)
                    entity_states_train[eid] = h_next
                    if baseline_name == "cnn":
                        st = h_next[0, -hidden_dim:]
                    elif baseline_name in ("mamba2", "lstm"):
                        st = h_next[0, :hidden_dim]
                    else:
                        st = h_next[0]
                    train_normal_states_list.append(st)
                val_normal_states = torch.stack(train_normal_states_list) if len(train_normal_states_list) > 0 else torch.stack(val_states_list)
            else:
                val_normal_states = torch.stack(val_states_list)
    else:
        # HDC-only baseline: states are the raw hypervectors
        for idx, flow in enumerate(val_flows):
            val_states_list.append(val_inputs[idx].to(torch.float32))
            
        val_normal_indices = [i for i, f in enumerate(val_flows) if f.label == 0]
        if len(val_normal_indices) > 0:
            val_normal_states = torch.stack([val_states_list[i] for i in val_normal_indices])
        else:
            logger.warning("No normal flows (label=0) present in validation split. Falling back to training normal states for threshold calibration.")
            train_normal_indices = [i for i, f in enumerate(train_flows) if f.label == 0]
            if len(train_normal_indices) > 0:
                val_normal_states = torch.stack([train_inputs[i] for i in train_normal_indices])
            else:
                val_normal_states = torch.stack(val_states_list)

    # Perform Validation-Calibrated Anomaly Thresholding
    logger.info("Calibrating decision thresholds on validation split...")
    validation_records: List[Dict[str, Any]] = []
    import numpy as np
    from sklearn.metrics import f1_score
    
    val_labels = np.array([f.label for f in val_flows])
    has_both_validation_classes = np.sum(val_labels == 1) > 0 and np.sum(val_labels == 0) > 0
    calibration_summary = {}
    
    if baseline_name == "autoencoder":
        if len(val_normal_indices) > 0:
            val_normal_inputs = torch.stack([val_inputs[i] for i in val_normal_indices])
        else:
            train_normal_indices = [i for i, f in enumerate(train_flows) if f.label == 0]
            if len(train_normal_indices) > 0:
                normal_train_flows = [train_flows[i] for i in train_normal_indices]
                if is_raw_baseline:
                    val_normal_inputs = mapper.transform(normal_train_flows)
                else:
                    val_normal_inputs = encoder.encode_tensor(normal_train_flows)
            else:
                val_normal_inputs = torch.stack(val_inputs)
        scorer.fit_threshold(val_normal_inputs, val_normal_states)
        calibration_summary = {"method": "ae_reconstruction_normal_fit", "threshold": float(scorer.reconstruction_threshold)}
        logger.info(f"Validation Calibrated AE Reconstruction Threshold on Normals: {scorer.reconstruction_threshold:.4f}")
    elif scoring_mode == "mahalanobis":
        if len(val_states_list) > 0:
            val_all_states = torch.stack(val_states_list)
            # Fit the reference only on validation normal states, then choose a
            # fixed threshold from validation labels. Test labels are never used.
            scorer.manifold.fit(val_normal_states)
            val_scores = scorer.manifold.compute_mahalanobis_distance(val_all_states).detach().cpu().numpy()
            calibration_method = config.divergence.get("calibration_method", "f1_max")
            
            if calibration_method == "normal_percentile" and not has_both_validation_classes:
                pass # it doesn't matter for now, it's just a fallback if needed
                
            if has_both_validation_classes:
                threshold, calibration_summary = calibrate_threshold(
                    val_labels, val_scores, method=calibration_method,
                    min_precision=config.divergence.get("min_precision", 0.95),
                    target_tpr=config.divergence.get("target_tpr", 0.95),
                )
            else:
                k_val = config.divergence.get("threshold_k", 3.0)
                threshold = float(np.mean(val_scores[val_labels == 0]) + k_val * np.std(val_scores[val_labels == 0]))
                calibration_summary = {"method": "mean_plus_k_std", "threshold": threshold}
            
            scorer.mahalanobis_threshold = threshold
            logger.info("Validation threshold frozen: method=%s threshold=%.4f", calibration_method, threshold)
            validation_records = [
                {"entity_id": flow.entity_id, "timestamp": flow.timestamp, "actual_label": int(label),
                 "predicted_label": int(score >= threshold), "drift_score": float(score), "threshold": float(threshold)}
                for flow, label, score in zip(val_flows, val_labels, val_scores)
            ]
        else:
            k_val = config.divergence.get("threshold_k", 3.0)
            scorer.threshold_k = k_val
            scorer.fit_mahalanobis_threshold(val_normal_states)
            logger.info(f"Validation Calibrated Mahalanobis Threshold on Normals: {scorer.mahalanobis_threshold:.4f} (k={scorer.threshold_k})")
        
    elif scoring_mode == "cosine":
        # 1. Establish initial_threshold from validation normal flows
        val_norm_dists = []
        if model is not None:
            model.eval()
            for idx in val_normal_indices:
                state = val_states_list[idx]
                with torch.no_grad():
                    pred_hv = model.predict_next_vector(state.unsqueeze(0)).squeeze(0)
                sim = torch.cosine_similarity(val_inputs[idx].to(torch.float32).unsqueeze(0), pred_hv.unsqueeze(0)).item()
                val_norm_dists.append(1.0 - sim)
        if len(val_norm_dists) > 0:
            scorer.initial_threshold = float(np.mean(val_norm_dists) + 1.0 * np.std(val_norm_dists))
            logger.info(f"Validation Calibrated Cosine Initial Threshold on Normals: {scorer.initial_threshold:.4f}")

        # 2. Grid search threshold_k multiplier for EWMA
        best_f1 = -1.0
        best_k = 1.0
        
        has_positives = np.sum(val_labels == 1) > 0
        if not has_positives:
            best_k = 1.0
        else:
            k_candidates = np.linspace(0.1, 4.0, 40)
            for k in k_candidates:
                # Reset EWMA stats for each candidate evaluation
                scorer.ewma_stats = {}
                preds = []
                for idx, (flow, state) in enumerate(zip(val_flows, val_states_list)):
                    scorer.register_observed(flow.entity_id, val_inputs[idx].to(torch.float32))
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

    # Release temporary training / validation structures to reduce Peak RSS
    del val_states_list
    gc.collect()

    # 9. Evaluate Test Split Step-by-Step
    logger.info("Evaluating test split step-by-step...")
    num_test = len(test_flows)
    test_labels = [flow.label for flow in test_flows]
    test_scores = np.empty(num_test, dtype=np.float32)
    test_preds = np.empty(num_test, dtype=np.int32)
    
    test_entity_states: Dict[str, torch.Tensor] = {}
    is_mahal = (scoring_mode == "mahalanobis" and baseline_name != "autoencoder")
    
    if is_mahal:
        inv_cov_fast = scorer.manifold.inv_cov
        mean_fast = scorer.manifold.mean
        th_fast = scorer.mahalanobis_threshold
        high_dim = (scorer.manifold.hidden_dim > 1000)

    start_eval = time.perf_counter()
    
    dt_test = torch.zeros((1, 1), dtype=torch.float32)
    
    with torch.inference_mode():
        for idx, flow in enumerate(test_flows):
            x_test = test_inputs[idx].to(torch.float32)
            dt_test[0, 0] = flow.dt
            eid = flow.entity_id
            
            if model is not None:
                if eid not in test_entity_states:
                    if baseline_name == "cnn":
                        state_dim = 2 * input_dim + hidden_dim
                    elif baseline_name in ("autoencoder", "lstm"):
                        state_dim = 2 * hidden_dim
                    elif baseline_name == "mamba2":
                        state_dim = 2 * hidden_dim * 16
                    else:
                        state_dim = hidden_dim
                    test_entity_states[eid] = torch.zeros(1, state_dim)
                    
                h_prev = test_entity_states[eid]
                h_next = model.step(x_test.unsqueeze(0), h_prev, dt_test)
                test_entity_states[eid] = h_next
                if baseline_name == "cnn":
                    state_to_score = h_next[0, -hidden_dim:]
                elif baseline_name in ("mamba2", "lstm"):
                    state_to_score = h_next[0, :hidden_dim]
                else:
                    state_to_score = h_next[0]
            else:
                # HDC-only: the "state" is the raw hypervector itself
                state_to_score = x_test
                
            if is_mahal:
                if high_dim:
                    sim = torch.cosine_similarity(state_to_score.unsqueeze(0), mean_fast.unsqueeze(0), dim=-1)
                    dist_val = float(torch.clamp(1.0 - sim, min=0.0).item())
                else:
                    diff = state_to_score - mean_fast
                    sq_dist = torch.clamp((diff @ inv_cov_fast) @ diff, min=0.0)
                    dist_val = float(torch.sqrt(sq_dist).item())
                test_scores[idx] = dist_val
                test_preds[idx] = 1 if dist_val >= th_fast else 0
            else:
                scorer.register_observed(eid, x_test)
                traj = TrajectoryState(eid, flow.timestamp, state_to_score)
                decision = scorer.score(traj)
                test_scores[idx] = decision.drift_score
                test_preds[idx] = 1 if decision.is_anomaly else 0
            
    eval_duration = time.perf_counter() - start_eval
    logger.info(f"Test evaluation completed in {eval_duration:.4f} seconds.")

    test_scores = test_scores.tolist()
    test_preds = test_preds.tolist()

    # Build concise records for logging
    records_list = [
        {
            "entity_id": test_flows[i].entity_id,
            "timestamp": test_flows[i].timestamp,
            "actual_label": test_labels[i],
            "predicted_label": test_preds[i],
            "drift_score": test_scores[i],
            "threshold": float(getattr(scorer, "mahalanobis_threshold", getattr(scorer, "threshold", 0.0)))
        }
        for i in range(min(num_test, 1000))
    ]

    # 10. Compute full metrics suite
    metrics = compute_metrics_suite(
        labels=test_labels,
        scores=test_scores,
        predictions=test_preds,
        latency_seconds=eval_duration,
        num_samples=len(test_flows)
    )
    if scoring_mode == "mahalanobis" and has_both_validation_classes:
        metrics["calibration"] = calibration_summary
    elif scoring_mode == "mahalanobis":
        metrics["calibration"] = {"method": "normal_mean_plus_k_std", "threshold": getattr(scorer, "mahalanobis_threshold", 0.0)}
    elif baseline_name == "autoencoder":
        metrics["calibration"] = calibration_summary
    else:
        metrics["calibration"] = {"method": "default", "threshold": getattr(scorer, "threshold", 0.0)}
    metrics["experiment"] = {
        "dataset": config.data.get("dataset_name", "unknown"), "model": baseline_name,
        "random_seed": config.seed, "hdc_dimension": D,
        "sequence_length": config.model.get("sequence_length", 16),
        "training_epochs": config.model.get("epochs", 5),
        "learning_rate": config.model.get("lr", 0.001),
        "config_hash": hashlib.sha256(json.dumps(config.to_dict(), sort_keys=True).encode()).hexdigest()[:16],
    }
    logger.info(f"Experiment {run_id} classification F1-Score = {metrics['f1_score']:.4f}")

    # 11. Write self-contained run folder
    run_dir = output_dir / run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing experiment artifacts: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config.to_dict(), f)
        
    # Save metrics
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
        
    # Save decisions as Parquet and CSV
    df_decisions = pd.DataFrame(records_list)
    decisions_parquet = run_dir / "decisions.parquet"
    decisions_csv = run_dir / "decisions.csv"
    try:
        df_decisions.to_parquet(decisions_parquet, index=False)
    except Exception:
        pass
    df_decisions.to_csv(decisions_csv, index=False)
    if validation_records:
        pd.DataFrame(validation_records).to_csv(run_dir / "validation_decisions.csv", index=False)
    
    logger.info(f"Run artifacts successfully committed to {run_dir}")
    
    return metrics
