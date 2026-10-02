import time
import json
import hashlib
import yaml
import logging
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
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
from hdlnn.baselines.ft_transformer import FTTransformerSequenceModel
from hdlnn.baselines.saint import SAINTSequenceModel
from hdlnn.common.model_utils import count_parameters
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
        backbone_units = config.model.get("backbone_units", 128)
        model = LNNSequenceModel(
            input_dim=input_dim, 
            hidden_dim=hidden_dim, 
            proj_dim=input_dim,
            backbone_units=backbone_units
        )
    elif baseline_name == "lstm":
        model = LSTMBaseline(input_dim=input_dim, hidden_dim=hidden_dim, proj_dim=input_dim)
    elif baseline_name == "cnn":
        model = CNN1DBaseline(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=3, proj_dim=input_dim)
    elif baseline_name == "autoencoder":
        model = AETemporalModel(input_dim=input_dim, hidden_dim=hidden_dim)
    elif baseline_name == "mamba2":
        from hdlnn.baselines.mamba import MambaSequenceModel
        model = MambaSequenceModel(input_dim=input_dim, hidden_dim=hidden_dim, proj_dim=input_dim)
    elif baseline_name in ("ft-transformer", "fttransformer"):
        model = FTTransformerSequenceModel(input_dim=input_dim, hidden_dim=hidden_dim, proj_dim=input_dim)
    elif baseline_name == "saint":
        model = SAINTSequenceModel(input_dim=input_dim, hidden_dim=hidden_dim, proj_dim=input_dim)
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
            aux_lambda = config.model.get("aux_separation_lambda", 0.0)
            sep_margin = config.model.get("aux_separation_margin", 4.0)
            train_labels_tensor = torch.tensor([f.label for f in train_flows], dtype=torch.long)
            
            use_curriculum = config.model.get("use_hard_curriculum", False)
            if use_curriculum:
                # Precompute sequence hardness based on minority attack presence
                # Sequences containing hard minority attacks or boundary transitions get scheduled later in training
                seq_hardness = []
                for item in seq_indices:
                    y_lbls = [train_labels_tensor[idx].item() for idx in item[1]]
                    # Sequences with mixed normal and attack transitions represent critical boundary transitions
                    n_att = sum(y_lbls)
                    hardness = float(n_att) / len(y_lbls) if len(y_lbls) > 0 else 0.0
                    seq_hardness.append(hardness)
                seq_hardness = np.array(seq_hardness)
                easy_indices = [seq_indices[i] for i in np.where(seq_hardness == 0.0)[0]]
                hard_indices = [seq_indices[i] for i in np.where(seq_hardness > 0.0)[0]]
            
            for epoch in range(epochs):
                epoch_loss = 0.0
                num_batches = 0
                
                # Active curriculum sequence list for this epoch
                if use_curriculum and len(hard_indices) > 0 and len(easy_indices) > 0:
                    # Epoch 0: standard mixed baseline; Epoch 1+: progressively oversample boundary sequences
                    hard_ratio = min(0.3 + 0.3 * epoch, 0.8)
                    n_hard_sample = int(len(seq_indices) * hard_ratio)
                    n_easy_sample = len(seq_indices) - n_hard_sample
                    active_seqs = (
                        [easy_indices[i % len(easy_indices)] for i in range(n_easy_sample)] +
                        [hard_indices[i % len(hard_indices)] for i in range(n_hard_sample)]
                    )
                else:
                    active_seqs = seq_indices
                
                for b in range(0, len(active_seqs), batch_size):
                    batch_items = active_seqs[b : b + batch_size]
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
                        
                        # Base predictive MSE loss with optional training-only hard-example weighting
                        hp_weight = config.model.get("hard_positive_weight", 1.0)
                        hn_weight = config.model.get("hard_negative_weight", 1.0)
                        
                        if hp_weight != 1.0 or hn_weight != 1.0:
                            y_idxs_flat = [idx for item in batch_items for idx in item[1]]
                            lbls = train_labels_tensor[y_idxs_flat].to(pred_seq.device)
                            weights = torch.ones_like(lbls, dtype=torch.float32)
                            weights[lbls == 1] = hp_weight
                            weights[lbls == 0] = hn_weight
                            
                            diff_sq = (pred_seq - yb).pow(2).mean(dim=-1) # [B, L]
                            loss = (diff_sq.view(-1) * weights).mean()
                        else:
                            loss = criterion(pred_seq, yb)
                        
                        # Training-only auxiliary manifold-aligned separation & boundary loss
                        if aux_lambda > 0.0:
                            y_idxs_flat = [idx for item in batch_items for idx in item[1]]
                            lbls_flat = train_labels_tensor[y_idxs_flat]
                            out_flat = out_seq.reshape(-1, out_seq.size(-1))
                            
                            norm_mask = (lbls_flat == 0)
                            att_mask = (lbls_flat == 1)
                            norm_h = out_flat[norm_mask]
                            att_h = out_flat[att_mask]
                            
                            if len(norm_h) > 1 and len(att_h) > 0:
                                c_norm = norm_h.mean(dim=0, keepdim=True)
                                # Intra-normal compactness (variance minimization)
                                var_loss = F.mse_loss(norm_h, c_norm.expand_as(norm_h))
                                # Attack separation from normal center
                                d_att = torch.norm(att_h - c_norm, dim=-1)
                                sep_loss = torch.relu(sep_margin - d_att).mean()
                                
                                # Optional boundary pair separation loss
                                lambda_b = config.model.get("aux_boundary_lambda", 0.0)
                                if lambda_b > 0.0 and len(norm_h) > 0 and len(att_h) > 0:
                                    p_dist = torch.cdist(att_h, norm_h) # [N_att, N_norm]
                                    min_dist, _ = torch.min(p_dist, dim=1)
                                    b_margin = config.model.get("aux_boundary_margin", 3.0)
                                    b_loss = torch.relu(b_margin - min_dist).mean()
                                    loss = loss + aux_lambda * (var_loss + sep_loss) + lambda_b * b_loss
                                else:
                                    loss = loss + aux_lambda * (var_loss + sep_loss)
                    
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
        n_clusters = config.divergence.get("n_clusters", 1)
        cov_type = config.divergence.get("covariance_type", "full")
        trim_ratio = config.divergence.get("trim_ratio", 0.02)
        use_lr = config.divergence.get("use_likelihood_ratio", False)
        lr_eps = config.divergence.get("lr_epsilon", 0.5)
        scorer = DivergenceScorer(
            mode=scoring_mode,
            threshold_k=config.divergence.get("threshold_k", 3.0),
            hidden_dim=scorer_dim,
            model=model,
            n_clusters=n_clusters,
            covariance_type=cov_type,
            trim_ratio=trim_ratio,
            use_likelihood_ratio=use_lr,
            lr_epsilon=lr_eps
        )

    # We evaluate states for ALL validation flows (both normal and attacks) to perform threshold calibration
    val_states_list = []
    val_z_list = []
    use_inst_fusion = config.divergence.get("use_instantaneous_fusion", False)
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
                if use_inst_fusion and isinstance(model, LNNSequenceModel):
                    h_next, z_next = model.step(x_val, h_prev, dt_val, return_instantaneous=True)
                    val_z_list.append(z_next[0])
                else:
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
            # Fit normal reference manifold on validation normal states
            scorer.manifold.fit(val_normal_states)
            
            # If likelihood ratio is enabled, fit attack manifold on validation anomalies
            val_att_indices = [i for i, f in enumerate(val_flows) if f.label == 1]
            if getattr(scorer, "use_likelihood_ratio", False) and len(val_att_indices) >= 10:
                val_att_states = torch.stack([val_states_list[i] for i in val_att_indices])
                scorer.fit_attack_manifold(val_att_states)
                if hasattr(scorer, "fit_discriminant_subspace"):
                    scorer.fit_discriminant_subspace(val_normal_states, val_att_states)
                if hasattr(scorer, "fit_linear_boundary") and config.divergence.get("use_linear_boundary", False):
                    val_labels_arr = np.array([f.label for f in val_flows])
                    scorer.fit_linear_boundary(val_all_states, val_labels_arr)
                if hasattr(scorer, "fit_quadratic_boundary") and config.divergence.get("use_quadratic_boundary", False):
                    val_labels_arr = np.array([f.label for f in val_flows])
                    scorer.fit_quadratic_boundary(val_all_states, val_labels_arr)
                if hasattr(scorer, "fit_distilled_student") and config.divergence.get("use_distilled_student", False):
                    val_labels_arr = np.array([f.label for f in val_flows])
                    scorer.fit_distilled_student(val_all_states, val_labels_arr)
                if hasattr(scorer, "fit_fusion_head") and use_inst_fusion and len(val_z_list) == len(val_all_states):
                    val_labels_arr = np.array([f.label for f in val_flows])
                    val_all_z = torch.stack(val_z_list)
                    sub_dim = config.divergence.get("fusion_subspace_dim", None)
                    scorer.fit_fusion_head(val_all_states, val_all_z, val_labels_arr, subspace_dim=sub_dim)
                    scorer.residual_alpha = float(config.divergence.get("residual_alpha", 0.0))
                    if scorer.residual_alpha > 0.0:
                        d_norm_v = scorer.manifold.compute_mahalanobis_distance(val_all_states)
                        d_att_v = scorer.attack_manifold.compute_mahalanobis_distance(val_all_states)
                        val_lr = (d_norm_v / (d_att_v + scorer.lr_epsilon)).cpu().numpy()
                        scorer.base_lr_mean = float(val_lr.mean())
                        scorer.base_lr_std = float(val_lr.std())
                elif hasattr(scorer, "fit_discriminative_head") and config.divergence.get("use_discriminative_head", False):
                    val_labels_arr = np.array([f.label for f in val_flows])
                    scorer.fit_discriminative_head(val_all_states, val_labels_arr)
                elif hasattr(scorer, "fit_teacher_mlp") and config.divergence.get("use_teacher_mlp", False):
                    val_labels_arr = np.array([f.label for f in val_flows])
                    scorer.fit_teacher_mlp(val_all_states, val_labels_arr)
                    scorer.residual_alpha = float(config.divergence.get("residual_alpha", 0.0))
                    if scorer.residual_alpha > 0.0:
                        d_norm_v = scorer.manifold.compute_mahalanobis_distance(val_all_states)
                        d_att_v = scorer.attack_manifold.compute_mahalanobis_distance(val_all_states)
                        val_lr = (d_norm_v / (d_att_v + scorer.lr_epsilon)).cpu().numpy()
                        scorer.base_lr_mean = float(val_lr.mean())
                        scorer.base_lr_std = float(val_lr.std())
                    scorer.gate_margin = float(config.divergence.get("gate_margin", 0.0))
                    if scorer.gate_margin > 0.0:
                        # Precompute validation base LR scores to set gate_th
                        d_norm_v = scorer.manifold.compute_mahalanobis_distance(val_all_states)
                        d_att_v = scorer.attack_manifold.compute_mahalanobis_distance(val_all_states)
                        val_lr = (d_norm_v / (d_att_v + scorer.lr_epsilon)).cpu().numpy()
                        # Calibrate base LR threshold
                        best_th = 1.0
                        best_f1 = -1.0
                        for cand in np.percentile(val_lr, np.linspace(10, 90, 80)):
                            pred = (val_lr >= cand).astype(int)
                            p_c = (pred & val_labels_arr).sum() / max(pred.sum(), 1)
                            r_c = (pred & val_labels_arr).sum() / max(val_labels_arr.sum(), 1)
                            f1_c = 2 * p_c * r_c / max(p_c + r_c, 1e-9)
                            if f1_c > best_f1:
                                best_f1 = f1_c
                                best_th = float(cand)
                        scorer.gate_th = best_th
                        logger.info("Fitted teacher gate threshold: %.4f (margin: %.2f)", scorer.gate_th, scorer.gate_margin)
                
            # Vectorized scoring across all validation flows
            if getattr(scorer, "use_instantaneous_fusion", False) and len(val_z_list) == len(val_all_states):
                val_scores_list = [scorer.compute_fusion_score(st, zt) for st, zt in zip(val_all_states, val_z_list)]
            else:
                val_scores_list = [scorer.compute_score(st) for st in val_all_states]
            val_scores = np.array(val_scores_list, dtype=np.float32)
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

    if model is not None and hasattr(model, "fuse_for_inference"):
        model.fuse_for_inference()

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
        
        # Fast inlined manifold scoring caches
        m_norm = scorer.manifold
        means_n_fast = m_norm.means_stacked
        invs_n_fast = m_norm.inv_covs_stacked
        use_lr_fast = scorer.use_likelihood_ratio and scorer.attack_manifold is not None and scorer.attack_manifold.is_fit
        lr_eps_fast = scorer.lr_epsilon
        if use_lr_fast:
            means_a_fast = scorer.attack_manifold.means_stacked
            invs_a_fast = scorer.attack_manifold.inv_covs_stacked
            w_disc_fast = scorer.w_discriminant
            z_shift_fast = scorer.z_shift
            z_scale_fast = scorer.z_scale
            w_bound_fast = scorer.w_boundary
            b_bound_fast = scorer.b_boundary
            w_quad_fast = scorer.w_quad
            b_quad_fast = scorer.b_quad
            w_sub_fast = scorer.w_sub_proj
            w1_stud_fast = scorer.w1_student
            b1_stud_fast = scorer.b1_student
            w2_stud_fast = scorer.w2_student
            b2_stud_fast = scorer.b2_student
            w1_teach_fast = scorer.w1_teacher
            b1_teach_fast = scorer.b1_teacher
            w2_teach_fast = scorer.w2_teacher
            b2_teach_fast = scorer.b2_teacher
            h_buf_teach = scorer.h_buf_teacher
            w1_head_fast = scorer.w1_head
            b1_head_fast = scorer.b1_head
            w2_head_fast = scorer.w2_head
            b2_head_fast = scorer.b2_head
            w3_head_fast = scorer.w3_head
            b3_head_fast = scorer.b3_head
            h1_buf_head = scorer.h1_buf_head
            h2_buf_head = scorer.h2_buf_head
            gate_th_fast = scorer.gate_th
            gate_margin_fast = scorer.gate_margin
            res_alpha_fast = scorer.residual_alpha
            base_mean_fast = scorer.base_lr_mean
            base_std_fast = scorer.base_lr_std
            use_fusion_fast = getattr(scorer, "use_instantaneous_fusion", False) and scorer.w1_fusion is not None
            w_disc_z_fast = scorer.w_discriminant_z
            w1_fusion_fast = scorer.w1_fusion
            b1_fusion_fast = scorer.b1_fusion
            w2_fusion_fast = scorer.w2_fusion
            b2_fusion_fast = scorer.b2_fusion
            h_buf_fusion = scorer.h_buf_fusion
            w_sub_z_fast = scorer.w_sub_z
            is_subspace_fusion_fast = scorer.is_subspace_fusion
            
            # Precompute joint 8-mode quadratic manifold kernel constants
            K_n = means_n_fast.shape[0]
            K_a = means_a_fast.shape[0]
            H_dim = means_n_fast.shape[1]
            q_n_c = torch.stack([torch.mv(invs_n_fast[k], means_n_fast[k]) for k in range(K_n)])
            c_n_c = torch.stack([torch.dot(means_n_fast[k], q_n_c[k]) for k in range(K_n)])
            q_a_c = torch.stack([torch.mv(invs_a_fast[k], means_a_fast[k]) for k in range(K_a)])
            c_a_c = torch.stack([torch.dot(means_a_fast[k], q_a_c[k]) for k in range(K_a)])
            
            invs_joint_fast = torch.cat([invs_n_fast.view(K_n * H_dim, H_dim), invs_a_fast.view(K_a * H_dim, H_dim)], dim=0)
            q_joint_fast = torch.cat([q_n_c, q_a_c], dim=0)
            c_joint_fast = torch.cat([c_n_c, c_a_c], dim=0)
            k_total = K_n + K_a
            k_n_split = K_n
        else:
            means_a_fast = None
            invs_a_fast = None
            w_disc_fast = None
            z_shift_fast = 0.0
            z_scale_fast = 1.0
            w_bound_fast = None
            b_bound_fast = 0.0
            w_quad_fast = None
            b_quad_fast = 0.0
            w_sub_fast = None
            w1_stud_fast = None
            b1_stud_fast = None
            w2_stud_fast = None
            b2_stud_fast = 0.0
            w1_teach_fast = None
            b1_teach_fast = None
            w2_teach_fast = None
            b2_teach_fast = 0.0
            h_buf_teach = None
            w1_head_fast = None
            b1_head_fast = None
            w2_head_fast = None
            b2_head_fast = None
            w3_head_fast = None
            b3_head_fast = 0.0
            h1_buf_head = None
            h2_buf_head = None
            gate_th_fast = 1.0
            gate_margin_fast = 0.0
            res_alpha_fast = 0.0
            base_mean_fast = 0.0
            base_std_fast = 1.0
            use_fusion_fast = False
            w_disc_z_fast = None
            w1_fusion_fast = None
            b1_fusion_fast = None
            w2_fusion_fast = None
            b2_fusion_fast = 0.0
            h_buf_fusion = None
            w_sub_z_fast = None
            is_subspace_fusion_fast = False
            invs_joint_fast = None
            q_joint_fast = None
            c_joint_fast = None
            k_total = 0
            k_n_split = 0

    start_eval = time.perf_counter()
    
    dt_test = torch.zeros((1, 1), dtype=torch.float32)
    x_buf = torch.empty((1, input_dim), dtype=torch.float32)
    feat66_buf = torch.empty(66, dtype=torch.float32)
    fusion_in_dim = w1_fusion_fast.size(1) if (is_mahal and use_fusion_fast and w1_fusion_fast is not None) else 195
    feat_fusion_buf = torch.empty(fusion_in_dim, dtype=torch.float32) if (is_mahal and use_fusion_fast) else None
    Ph_buf = torch.empty(k_total * H_dim, dtype=torch.float32) if k_total > 0 else None
    qh_buf = torch.empty(k_total, dtype=torch.float32) if k_total > 0 else None
    
    with torch.inference_mode():
        for idx, flow in enumerate(test_flows):
            x_raw = test_inputs[idx]
            x_buf[0].copy_(x_raw)
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
                if use_fusion_fast and isinstance(model, LNNSequenceModel):
                    h_next, z_next = model.step(x_buf, h_prev, dt_test, return_instantaneous=True)
                    z_to_score = z_next[0]
                else:
                    h_next = model.step(x_buf, h_prev, dt_test)
                    z_to_score = None
                test_entity_states[eid] = h_next
                if baseline_name == "cnn":
                    state_to_score = h_next[0, -hidden_dim:]
                elif baseline_name in ("mamba2", "lstm"):
                    state_to_score = h_next[0, :hidden_dim]
                else:
                    state_to_score = h_next[0]
            else:
                # HDC-only: the "state" is the raw hypervector itself
                state_to_score = x_raw
                z_to_score = None
                
            if is_mahal:
                if invs_joint_fast is not None:
                    # Specialized joint 8-mode quadratic manifold scoring (saves ~11 us/flow)
                    vec = state_to_score.view(-1)
                    torch.mv(invs_joint_fast, vec, out=Ph_buf)
                    Ph_view = Ph_buf.view(k_total, H_dim)
                    hPh = torch.sum(Ph_view * vec, dim=-1)
                    torch.mv(q_joint_fast, vec, out=qh_buf)
                    sq_joint = torch.clamp(hPh - 2.0 * qh_buf + c_joint_fast, min=0.0)
                    d_norm = torch.sqrt(sq_joint[:k_n_split].min())
                    d_att = torch.sqrt(sq_joint[k_n_split:].min())
                    raw_lr = d_norm / (d_att + lr_eps_fast)
                    
                    if use_fusion_fast and w1_fusion_fast is not None and z_to_score is not None:
                        # In-place Instantaneous-Temporal Fusion Head (195 -> 16 -> 1 or 82 -> 16 -> 1)
                        vec_z = z_to_score.view(-1)
                        fish_h = torch.dot(vec, w_disc_fast) if w_disc_fast is not None else 0.0
                        fish_z = torch.dot(vec_z, w_disc_z_fast) if w_disc_z_fast is not None else 0.0
                        
                        if w1_fusion_fast.size(1) == 192:
                            feat_fusion_buf[:64].copy_(vec)
                            feat_fusion_buf[64:192].copy_(vec_z)
                        elif w1_fusion_fast.size(1) == 67:
                            feat_fusion_buf[:64].copy_(vec)
                            feat_fusion_buf[64] = fish_h
                            feat_fusion_buf[65] = raw_lr
                            feat_fusion_buf[66] = fish_z
                        elif is_subspace_fusion_fast and w_sub_z_fast is not None:
                            z_proj = torch.mv(w_sub_z_fast.t(), vec_z)
                            feat_fusion_buf[:64].copy_(vec)
                            feat_fusion_buf[64:80].copy_(z_proj)
                            feat_fusion_buf[80] = fish_h
                            feat_fusion_buf[81] = fish_z
                            feat_fusion_buf[82] = raw_lr
                        else:
                            feat_fusion_buf[:64].copy_(vec)
                            feat_fusion_buf[64:192].copy_(vec_z)
                            feat_fusion_buf[192] = fish_h
                            feat_fusion_buf[193] = fish_z
                            feat_fusion_buf[194] = raw_lr
                            
                        if w2_fusion_fast is None:
                            logit = torch.dot(w1_fusion_fast[0], feat_fusion_buf) + b1_fusion_fast[0]
                        else:
                            if h_buf_fusion is None:
                                h_buf_fusion = torch.empty(16, dtype=torch.float32, device=state_to_score.device)
                            torch.mv(w1_fusion_fast, feat_fusion_buf, out=h_buf_fusion)
                            h_buf_fusion.add_(b1_fusion_fast)
                            torch.clamp_min_(h_buf_fusion, 0.0)
                            logit = torch.dot(h_buf_fusion, w2_fusion_fast[0]) + b2_fusion_fast
                        prob_f = float(torch.sigmoid(logit).item())
                        if res_alpha_fast > 0.0:
                            lr_item = float(raw_lr.item())
                            base_prob = 1.0 / (1.0 + np.exp(-(lr_item - base_mean_fast) / (base_std_fast + 1e-5)))
                            dist_val = float(base_prob + res_alpha_fast * (prob_f - base_prob))
                        else:
                            dist_val = prob_f
                    elif w1_head_fast is not None and w_disc_fast is not None:
                        # Direct in-place 66 -> 32 -> 16 -> 1 Discriminative Head
                        fish_p = torch.dot(vec, w_disc_fast)
                        feat66_buf[:64].copy_(vec)
                        feat66_buf[64] = fish_p
                        feat66_buf[65] = raw_lr
                        if h1_buf_head is None:
                            h1_buf_head = torch.empty(32, dtype=torch.float32, device=state_to_score.device)
                            h2_buf_head = torch.empty(16, dtype=torch.float32, device=state_to_score.device)
                        torch.mv(w1_head_fast, feat66_buf, out=h1_buf_head)
                        h1_buf_head.add_(b1_head_fast)
                        torch.clamp_min_(h1_buf_head, 0.0)
                        
                        torch.mv(w2_head_fast, h1_buf_head, out=h2_buf_head)
                        h2_buf_head.add_(b2_head_fast)
                        torch.clamp_min_(h2_buf_head, 0.0)
                        
                        logit = torch.dot(h2_buf_head, w3_head_fast[0]) + b3_head_fast
                        dist_val = float(torch.sigmoid(logit).item())
                    elif w1_teach_fast is not None and w_disc_fast is not None:
                        # Direct in-place 66 -> 16 -> 1 Teacher MLP with optional ambiguity gating
                        lr_item = float(raw_lr.item())
                        if gate_margin_fast > 0.0 and abs(lr_item - gate_th_fast) / max(gate_th_fast, 1e-4) >= gate_margin_fast:
                            dist_val = lr_item
                        else:
                            fish_p = torch.dot(vec, w_disc_fast)
                            feat66_buf[:64].copy_(vec)
                            feat66_buf[64] = fish_p
                            feat66_buf[65] = raw_lr
                            if h_buf_teach is None:
                                h_buf_teach = torch.empty(16, dtype=torch.float32, device=state_to_score.device)
                            torch.mv(w1_teach_fast, feat66_buf, out=h_buf_teach)
                            h_buf_teach.add_(b1_teach_fast)
                            torch.clamp_min_(h_buf_teach, 0.0)
                            logit = torch.dot(h_buf_teach, w2_teach_fast[0]) + b2_teach_fast
                            prob_t = float(torch.sigmoid(logit).item())
                            if res_alpha_fast > 0.0:
                                base_prob = 1.0 / (1.0 + np.exp(-(lr_item - base_mean_fast) / (base_std_fast + 1e-5)))
                                dist_val = float(base_prob + res_alpha_fast * (prob_t - base_prob))
                            else:
                                dist_val = prob_t
                elif means_n_fast is not None and invs_n_fast is not None:
                    # Fallback single-flow Mahalanobis
                    diff_n = state_to_score - means_n_fast
                    diff_inv_n = torch.bmm(diff_n.unsqueeze(1), invs_n_fast).squeeze(1)
                    sq_norm = torch.clamp(torch.sum(diff_inv_n * diff_n, dim=-1), min=0.0)
                    d_norm = torch.sqrt(sq_norm.min())
                    
                    if use_lr_fast and means_a_fast is not None and invs_a_fast is not None:
                        diff_a = state_to_score - means_a_fast
                        diff_inv_a = torch.bmm(diff_a.unsqueeze(1), invs_a_fast).squeeze(1)
                        sq_att = torch.clamp(torch.sum(diff_inv_a * diff_a, dim=-1), min=0.0)
                        d_att = torch.sqrt(sq_att.min())
                        raw_lr = d_norm / (d_att + lr_eps_fast)
                        
                        if use_fusion_fast and w1_fusion_fast is not None and z_to_score is not None:
                            vec_z = z_to_score.view(-1)
                            fish_h = torch.dot(vec, w_disc_fast) if w_disc_fast is not None else 0.0
                            fish_z = torch.dot(vec_z, w_disc_z_fast) if w_disc_z_fast is not None else 0.0
                            if w1_fusion_fast.size(1) == 67:
                                feat_fusion_buf[:64].copy_(vec)
                                feat_fusion_buf[64] = fish_h
                                feat_fusion_buf[65] = raw_lr
                                feat_fusion_buf[66] = fish_z
                            elif is_subspace_fusion_fast and w_sub_z_fast is not None:
                                z_proj = torch.mv(w_sub_z_fast.t(), vec_z)
                                feat_fusion_buf[:64].copy_(vec)
                                feat_fusion_buf[64:80].copy_(z_proj)
                                feat_fusion_buf[80] = fish_h
                                feat_fusion_buf[81] = fish_z
                                feat_fusion_buf[82] = raw_lr
                            else:
                                feat_fusion_buf[:64].copy_(vec)
                                feat_fusion_buf[64:192].copy_(vec_z)
                                feat_fusion_buf[192] = fish_h
                                feat_fusion_buf[193] = fish_z
                                feat_fusion_buf[194] = raw_lr
                                
                            if h_buf_fusion is None:
                                h_buf_fusion = torch.empty(16, dtype=torch.float32, device=state_to_score.device)
                            torch.mv(w1_fusion_fast, feat_fusion_buf, out=h_buf_fusion)
                            h_buf_fusion.add_(b1_fusion_fast)
                            torch.clamp_min_(h_buf_fusion, 0.0)
                            logit = torch.dot(h_buf_fusion, w2_fusion_fast[0]) + b2_fusion_fast
                            prob_f = float(torch.sigmoid(logit).item())
                            if res_alpha_fast > 0.0:
                                lr_item = float(raw_lr.item())
                                base_prob = 1.0 / (1.0 + np.exp(-(lr_item - base_mean_fast) / (base_std_fast + 1e-5)))
                                dist_val = float(base_prob + res_alpha_fast * (prob_f - base_prob))
                            else:
                                dist_val = prob_f
                        elif w1_head_fast is not None and w_disc_fast is not None:
                            # Direct in-place 66 -> 32 -> 16 -> 1 Discriminative Head
                            vec = state_to_score.view(-1)
                            fish_p = torch.dot(vec, w_disc_fast)
                            feat66 = torch.cat([vec, fish_p.view(1), raw_lr.view(1)], dim=0)
                            if h1_buf_head is None:
                                h1_buf_head = torch.empty(32, dtype=torch.float32, device=state_to_score.device)
                                h2_buf_head = torch.empty(16, dtype=torch.float32, device=state_to_score.device)
                            torch.mv(w1_head_fast, feat66, out=h1_buf_head)
                            h1_buf_head.add_(b1_head_fast)
                            torch.clamp_min_(h1_buf_head, 0.0)
                            
                            torch.mv(w2_head_fast, h1_buf_head, out=h2_buf_head)
                            h2_buf_head.add_(b2_head_fast)
                            torch.clamp_min_(h2_buf_head, 0.0)
                            
                            logit = torch.dot(h2_buf_head, w3_head_fast[0]) + b3_head_fast
                            dist_val = float(torch.sigmoid(logit).item())
                        elif w1_teach_fast is not None and w_disc_fast is not None:
                            # Direct in-place 66 -> 16 -> 1 Teacher MLP with optional ambiguity gating
                            lr_item = float(raw_lr.item())
                            if gate_margin_fast > 0.0 and abs(lr_item - gate_th_fast) / max(gate_th_fast, 1e-4) >= gate_margin_fast:
                                dist_val = lr_item
                            else:
                                vec = state_to_score.view(-1)
                                fish_p = torch.dot(vec, w_disc_fast)
                                feat66 = torch.cat([vec, fish_p.view(1), raw_lr.view(1)], dim=0)
                                if h_buf_teach is None:
                                    h_buf_teach = torch.empty(16, dtype=torch.float32, device=state_to_score.device)
                                torch.mv(w1_teach_fast, feat66, out=h_buf_teach)
                                h_buf_teach.add_(b1_teach_fast)
                                torch.clamp_min_(h_buf_teach, 0.0)
                                logit = torch.dot(h_buf_teach, w2_teach_fast[0]) + b2_teach_fast
                                prob_t = float(torch.sigmoid(logit).item())
                                if res_alpha_fast > 0.0:
                                    base_prob = 1.0 / (1.0 + np.exp(-(lr_item - base_mean_fast) / (base_std_fast + 1e-5)))
                                    dist_val = float(base_prob + res_alpha_fast * (prob_t - base_prob))
                                else:
                                    dist_val = prob_t
                        elif w_sub_fast is not None and w1_stud_fast is not None:
                            # 4D-Subspace + 5 -> 4 -> 1 Distilled Student MLP
                            vec = state_to_score.view(-1)
                            proj4 = torch.mv(w_sub_fast.t(), vec)
                            feat5 = torch.cat([proj4, raw_lr.view(1)])
                            h_s = torch.relu(torch.mv(w1_stud_fast, feat5) + b1_stud_fast)
                            logit = torch.dot(h_s, w2_stud_fast[0]) + b2_stud_fast
                            dist_val = float(torch.sigmoid(logit).item())
                        elif w_quad_fast is not None and w_disc_fast is not None:
                            # 5-D Fast Calibrated Quadratic Distilled Boundary
                            vec = state_to_score.view(-1)
                            fish = torch.dot(vec, w_disc_fast)
                            lr_v = raw_lr.squeeze()
                            feat5 = torch.stack([fish, lr_v, fish * fish, lr_v * lr_v, fish * lr_v])
                            logit = torch.dot(feat5, w_quad_fast) + b_quad_fast
                            dist_val = float(torch.sigmoid(logit).item())
                        elif w_bound_fast is not None and w_disc_fast is not None:
                            # 66-D Fast Calibrated Linear Decision Boundary
                            vec = state_to_score.view(-1)
                            fish_p = torch.dot(vec, w_disc_fast)
                            feat_66 = torch.cat([vec, fish_p.view(1), raw_lr.view(1)], dim=0)
                            logit = torch.dot(feat_66, w_bound_fast) + b_bound_fast
                            dist_val = float(torch.sigmoid(logit).item())
                        else:
                            raw_s = raw_lr
                            if w_disc_fast is not None:
                                proj = torch.dot(state_to_score.view(-1), w_disc_fast)
                                z = (proj - z_shift_fast) / z_scale_fast
                                sig = torch.sigmoid(z)
                                raw_s = raw_s * (1.0 + 0.5 * sig)
                            dist_val = float(raw_s.item())
                    else:
                        dist_val = float(d_norm.item())
                else:
                    dist_val = scorer.compute_score(state_to_score)
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
    if model is not None:
        param_counts = count_parameters(model)
        if hasattr(model, "export_inference_model"):
            inf_m = model.export_inference_model()
            inf_counts = count_parameters(inf_m)
            param_counts["inference_total_params"] = inf_counts["total_params"]
            param_counts["inference_params_m"] = inf_counts["params_m"]
            param_counts["inference_formatted"] = inf_counts["formatted"]
        metrics["parameters"] = param_counts
    else:
        metrics["parameters"] = {
            "total_params": 0, "trainable_params": 0, "params_m": 0.0, "trainable_m": 0.0, "formatted": "0.00M"
        }
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
