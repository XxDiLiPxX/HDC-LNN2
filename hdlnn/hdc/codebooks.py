import torch
import torchhd
import logging
from typing import Dict, List, Tuple, Any

logger = logging.getLogger(__name__)

class CodebookManager:
    """Manages HDC basis hypervectors for field keys and continuous/level features.
    
    Ensures that numeric normalization parameters are learned from training data only.
    """
    def __init__(self, D: int = 10000, num_levels: int = 100):
        self.D = D
        self.num_levels = num_levels
        self.key_vectors: Dict[str, torch.Tensor] = {}
        # Pre-generate level hypervectors using torchhd's linear level method
        self.level_vectors = torchhd.level(num_levels, D)
        self.min_max: Dict[str, Tuple[float, float]] = {}

    def get_key_vector(self, key: str) -> torch.Tensor:
        """Retrieves or generates a unique random hypervector for a given field key."""
        if key not in self.key_vectors:
            # Generate a random bipolar/bipolar-compatible hypervector for the key
            self.key_vectors[key] = torchhd.random(1, self.D).squeeze(0)
        return self.key_vectors[key]

    def fit_numerical_ranges(self, train_flows: List[Any], numerical_columns: List[str]):
        """Learns the min/max value ranges for continuous columns based on the training split only."""
        logger.info("Fitting numerical ranges on training split...")
        for col in numerical_columns:
            vals = [flow.numerical_fields[col] for flow in train_flows]
            if not vals:
                min_val, max_val = 0.0, 1.0
            else:
                min_val = float(min(vals))
                max_val = float(max(vals))
            
            # Avoid division by zero if all values are identical
            if min_val == max_val:
                max_val = min_val + 1.0
                
            self.min_max[col] = (min_val, max_val)
            logger.debug(f"Numerical column '{col}' range: [{min_val}, {max_val}]")

    def get_numerical_vector(self, key: str, val: float) -> torch.Tensor:
        """Maps a numeric value to a level hypervector using fitted ranges."""
        min_val, max_val = self.min_max.get(key, (0.0, 1.0))
        
        # Min-max normalization clamped to [0, 1]
        if max_val == min_val:
            norm_val = 0.0
        else:
            norm_val = (val - min_val) / (max_val - min_val)
        norm_val = max(0.0, min(1.0, norm_val))
        
        # Find index in level vectors
        level_idx = int(round(norm_val * (self.num_levels - 1)))
        return self.level_vectors[level_idx]
