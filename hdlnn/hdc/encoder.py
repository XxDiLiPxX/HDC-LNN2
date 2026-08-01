import torch
import torchhd
import logging
from typing import List, Dict, Any
from hdlnn.contracts.interfaces import IEncoder
from hdlnn.contracts.schemas import CanonicalFlow, EncodedHypervector
from hdlnn.hdc.codebooks import CodebookManager
from hdlnn.hdc.item_memory import ItemMemory

logger = logging.getLogger(__name__)

class RecordEncoder(IEncoder):
    """Encodes a CanonicalFlow event record into a fixed-width bipolar hypervector."""
    
    def __init__(
        self, 
        D: int = 10000, 
        categorical_columns: List[str] = None, 
        numerical_columns: List[str] = None
    ):
        self.D = D
        self.categorical_columns = categorical_columns or []
        self.numerical_columns = numerical_columns or []
        
        self.codebooks = CodebookManager(D=D)
        self.item_memory = ItemMemory(D=D)

    def fit(self, train_flows: List[CanonicalFlow]) -> None:
        """Fits numerical ranges and builds item memory vocabulary from training data only."""
        # 1. Fit numerical min-max scales
        self.codebooks.fit_numerical_ranges(train_flows, self.numerical_columns)
        
        # 2. Build item memory from training categoricals
        self.item_memory.unlock()
        for flow in train_flows:
            for col in self.categorical_columns:
                val = flow.categorical_fields.get(col, "-")
                self.item_memory.get_vector(col, val)
        
        # Lock item memory to prevent future leakage
        self.item_memory.lock()

    def encode(self, flow: CanonicalFlow) -> EncodedHypervector:
        """Transforms a CanonicalFlow into a fixed D-dimensional bipolar hypervector."""
        return self.encode_batch([flow])[0]

    def encode_batch(self, flows: List[CanonicalFlow]) -> List[EncodedHypervector]:
        """Vectorized encoding of a batch of CanonicalFlow events into hypervectors."""
        num_flows = len(flows)
        if num_flows == 0:
            return []

        summed = torch.zeros((num_flows, self.D), dtype=torch.float32)
        any_oov = [False] * num_flows

        # 1. Process categorical fields (vectorized lookup via ItemMemory)
        for col in self.categorical_columns:
            key_vector = self.codebooks.get_key_vector(col)
            col_vals = [flow.categorical_fields.get(col, "-") for flow in flows]
            val_tensor, is_oov_list = self.item_memory.get_batch_vectors(col, col_vals)
            
            for i, is_oov in enumerate(is_oov_list):
                if is_oov:
                    any_oov[i] = True
            
            # Bind key_vector (broadcasts from [D] to [num_flows, D]) in-place to avoid intermediate allocation
            summed.addcmul_(val_tensor, key_vector.unsqueeze(0))

        # 2. Process numerical fields (fully vectorized lookup via level index mapping)
        for col in self.numerical_columns:
            key_vector = self.codebooks.get_key_vector(col)
            
            # Gather all values for the batch
            raw_vals = [flow.numerical_fields.get(col, 0.0) for flow in flows]
            vals_tensor = torch.tensor(raw_vals, dtype=torch.float32)
            
            # Normalize based on training range
            min_val, max_val = self.codebooks.min_max.get(col, (0.0, 1.0))
            if max_val == min_val:
                norm_vals = torch.zeros_like(vals_tensor)
            else:
                norm_vals = (vals_tensor - min_val) / (max_val - min_val)
            norm_vals = torch.clamp(norm_vals, 0.0, 1.0)
            
            # Find level indices
            level_indices = torch.round(norm_vals * (self.codebooks.num_levels - 1)).long()
            # Index directly into level_vectors [num_flows, D]
            val_tensor = self.codebooks.level_vectors[level_indices]
            
            # Bind with key vector in-place
            summed.addcmul_(val_tensor, key_vector.unsqueeze(0))

        # 3. Apply thresholding / majority rule sign operation
        bipolar_vectors = torch.sign(summed)
        bipolar_vectors[bipolar_vectors == 0] = 1.0

        return [
            EncodedHypervector(vector=bipolar_vectors[i], is_oov=any_oov[i])
            for i in range(num_flows)
        ]
