import torch
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
        # Pre-generate level hypervectors with a deterministic bipolar ramp.
        self.level_vectors = self._build_level_vectors(num_levels, D)
        self.min_max: Dict[str, Tuple[float, float]] = {}

    def _build_level_vectors(self, num_levels: int, D: int) -> torch.Tensor:
        """Create a stable sequence of bipolar vectors for numeric binning."""
        if num_levels <= 1:
            return torch.ones((1, D), dtype=torch.float32)

        base = torch.randint(0, 2, (D,), dtype=torch.int8).float() * 2.0 - 1.0
        levels = [base]
        current = base.clone()
        flip_stride = max(1, D // max(1, num_levels))

        for idx in range(1, num_levels):
            current = current.clone()
            flip_start = ((idx - 1) * flip_stride) % D
            flip_end = min(D, flip_start + flip_stride)
            current[flip_start:flip_end] *= -1.0
            levels.append(current)

        return torch.stack(levels)

    def get_key_vector(self, key: str) -> torch.Tensor:
        """Retrieves or generates a unique random hypervector for a given field key."""
        if key not in self.key_vectors:
            # Generate a random bipolar/bipolar-compatible hypervector for the key
            self.key_vectors[key] = torch.randint(0, 2, (self.D,), dtype=torch.int8).float() * 2.0 - 1.0
        return self.key_vectors[key]

    def fit_numerical_ranges(self, train_flows: List[Any], numerical_columns: List[str], use_log_transform: bool = False):
        """Learns the min/max value ranges for continuous columns based on the training split only."""
        logger.info("Fitting numerical ranges on training split (use_log_transform=%s)...", use_log_transform)
        self.use_log_transform = use_log_transform
        for col in numerical_columns:
            raw_vals = [flow.numerical_fields[col] for flow in train_flows]
            if not raw_vals:
                min_val, max_val = 0.0, 1.0
            else:
                if self.use_log_transform:
                    transformed = [float(torch.log1p(torch.tensor(max(0.0, float(v)))).item()) for v in raw_vals]
                else:
                    transformed = [float(v) for v in raw_vals]
                min_val = float(min(transformed))
                max_val = float(max(transformed))
            
            # Avoid division by zero if all values are identical
            if min_val == max_val:
                max_val = min_val + 1.0
                
            self.min_max[col] = (min_val, max_val)
            logger.debug(f"Numerical column '{col}' range: [{min_val}, {max_val}]")

    def get_numerical_vector(self, key: str, val: float) -> torch.Tensor:
        """Maps a numeric value to a level hypervector using fitted ranges."""
        min_val, max_val = self.min_max.get(key, (0.0, 1.0))
        if getattr(self, "use_log_transform", True):
            val = float(torch.log1p(torch.tensor(max(0.0, float(val)))).item())
        
        # Min-max normalization clamped to [0, 1]
        if max_val == min_val:
            norm_val = 0.0
        else:
            norm_val = (val - min_val) / (max_val - min_val)
        norm_val = max(0.0, min(1.0, norm_val))
        
        # Find index in level vectors
        level_idx = int(round(norm_val * (self.num_levels - 1)))
        return self.level_vectors[level_idx]
