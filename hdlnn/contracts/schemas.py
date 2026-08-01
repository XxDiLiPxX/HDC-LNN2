from dataclasses import dataclass
from typing import Dict, Any, Optional 
import torch

@dataclass
class CanonicalFlow:
    """Standardized dataset record schema across UNSW-NB15 and CICIoT2023[cite: 2]."""
    entity_id: str          # e.g., Source IP for disjoint splitting[cite: 2]
    timestamp: float        # Event arrival timestamp in seconds[cite: 2]
    dt: float               # Delta time since last entity event ($\Delta t$)[cite: 1, 2]
    categorical_fields: Dict[str, str]   # IPs, ports, protocols, flags[cite: 2]
    numerical_fields: Dict[str, float]   # Bytes, duration, packet count[cite: 2]
    label: int              # 0 for Normal, 1 for Attack[cite: 2]
    split: str              # 'train', 'val', or 'test'[cite: 2]

@dataclass
class EncodedHypervector:
    """Fixed-size HDC representation of a CanonicalFlow[cite: 2]."""
    vector: torch.Tensor    # Shape: [D], Bipolar/Binary Tensor[cite: 1, 2]
    is_oov: bool            # True if unmapped tokens were mapped to OOV vector[cite: 2]

@dataclass
class TrajectoryState:
    """Continuous hidden state of an entity evaluated by LNN[cite: 2]."""
    entity_id: str
    timestamp: float
    state: torch.Tensor     # Shape: [hidden_dim][cite: 2]

@dataclass
class AnomalyDecision:
    """Final anomaly verdict emitted by the Divergence Engine[cite: 2]."""
    drift_score: float
    threshold: float
    is_anomaly: bool
    scoring_mode: str       # 'mahalanobis' or 'cosine'[cite: 3]