import torch
import numpy as np
import logging
from typing import List, Dict, Any
from hdlnn.contracts.interfaces import IEncoder
from hdlnn.contracts.schemas import CanonicalFlow, EncodedHypervector
from hdlnn.hdc.encoder import RecordEncoder

logger = logging.getLogger(__name__)

class SIMDEncoder(IEncoder):
    """Production-hardened SIMD hyperdimensional encoder.
    
    Uses packed 16,384-bit binary hypervectors (2048 bytes of uint8) and accelerates
    operations using PyTorch's optimized bitwise tensor operators.
    """
    def __init__(self, record_encoder: RecordEncoder):
        self.record_encoder = record_encoder
        self.categorical_columns = record_encoder.categorical_columns
        self.numerical_columns = record_encoder.numerical_columns
        self.dimension = 16384  # Cache-aligned 2KB footprint (16384 bits)
        self.num_bytes = self.dimension // 8
        self.device = torch.device("cpu")
        
        # Force generation of all column key vectors in the reference codebook manager
        for col in self.categorical_columns + self.numerical_columns:
            self.record_encoder.codebooks.get_key_vector(col)
        
        # 1. Compile Key-Value Codebooks in packed uint8 format
        self.key_codebook: Dict[str, torch.Tensor] = {}
        self.val_codebooks: Dict[str, Dict[str, torch.Tensor]] = {}
        self._compile_categorical_memories()
        
        # 2. Compile numerical range thermometer codebooks
        self.num_ranges: Dict[str, Dict[str, float]] = {}
        self.num_codebooks: Dict[str, List[torch.Tensor]] = {}
        self._compile_numerical_codebooks()
        
        # OOV packed vector (all zeros or random)
        self.oov_vector = torch.zeros(self.num_bytes, dtype=torch.uint8)

    def _compile_categorical_memories(self) -> None:
        """Translates the standard bipolar item memory into packed binary uint8 hypervectors."""
        # Convert keys
        for key, vec in self.record_encoder.codebooks.key_vectors.items():
            self.key_codebook[key] = self._pack_bipolar_to_uint8(vec)
            
        # Convert values
        for key in self.record_encoder.item_memory.vocab_ids:
            self.val_codebooks[key] = {}
            vocab_col = self.record_encoder.item_memory.vocab_ids[key]
            tensor_col = self.record_encoder.item_memory.vocab_tensors[key]
            for val, idx in vocab_col.items():
                bipolar_vec = tensor_col[idx]
                self.val_codebooks[key][val] = self._pack_bipolar_to_uint8(bipolar_vec)

    def _compile_numerical_codebooks(self) -> None:
        """Translates thermometer numeric codebooks to packed uint8 format."""
        for col in self.numerical_columns:
            if col not in self.record_encoder.codebooks.min_max:
                continue
            min_val, max_val = self.record_encoder.codebooks.min_max[col]
            self.num_ranges[col] = {
                "min": min_val,
                "max": max_val
            }
            
            self.num_codebooks[col] = []
            for bipolar_vec in self.record_encoder.codebooks.level_vectors:
                self.num_codebooks[col].append(self._pack_bipolar_to_uint8(bipolar_vec))

    def _pack_bipolar_to_uint8(self, bipolar_tensor: torch.Tensor) -> torch.Tensor:
        """Interpolates and packs a bipolar tensor (-1, 1) of any dimension into a uint8 packed tensor.
        
        Bipolar values of -1 map to binary 1, and +1 map to binary 0.
        """
        if len(bipolar_tensor) != self.dimension:
            indices = np.linspace(0, len(bipolar_tensor) - 1, self.dimension).astype(int)
            bipolar_tensor = bipolar_tensor[indices]
            
        binary = (bipolar_tensor < 0).to(torch.uint8)  # -1 -> 1, +1 -> 0
        binary_reshaped = binary.view(self.num_bytes, 8)
        shifts = torch.arange(7, -1, -1, dtype=torch.uint8, device=binary.device)
        packed = torch.sum(binary_reshaped * (1 << shifts), dim=-1).to(torch.uint8)
        return packed

    def _unpack_uint8_to_binary(self, packed_tensor: torch.Tensor) -> torch.Tensor:
        """Unpacks a uint8 packed tensor back to binary float32 tensor of shape [16384]."""
        shifts = torch.arange(7, -1, -1, dtype=torch.uint8, device=packed_tensor.device)
        unpacked = (packed_tensor.unsqueeze(-1) >> shifts) & 1
        return unpacked.view(-1).to(torch.float32)

    def _pack_binary_to_uint8(self, binary_tensor: torch.Tensor) -> torch.Tensor:
        """Packs a binary float32/int tensor of shape [16384] back to [2048] uint8."""
        binary_reshaped = (binary_tensor > 0.5).to(torch.uint8).view(self.num_bytes, 8)
        shifts = torch.arange(7, -1, -1, dtype=torch.uint8, device=binary_tensor.device)
        packed = torch.sum(binary_reshaped * (1 << shifts), dim=-1).to(torch.uint8)
        return packed

    def encode(self, flow: CanonicalFlow) -> EncodedHypervector:
        """Transforms a CanonicalFlow into a packed uint8 hypervector.
        
        Performs optimized bitwise binding (XOR) and majority bundling.
        """
        bound_vectors = []
        oov_keys = []
        oov_vals = []
        
        # 1. Encode Categorical Columns
        for col in self.categorical_columns:
            val = flow.categorical_fields.get(col, "-")
            
            # Key vector
            k_vec = self.key_codebook.get(col)
            if k_vec is None:
                oov_keys.append(col)
                k_vec = self.oov_vector
                
            # Value vector
            v_vec = self.val_codebooks.get(col, {}).get(val)
            if v_vec is None:
                oov_vals.append(f"{col}:{val}")
                v_vec = self.oov_vector
                
            bound = torch.bitwise_xor(k_vec, v_vec)
            bound_vectors.append(bound)

        # 2. Encode Numerical Columns
        for col in self.numerical_columns:
            val = flow.numerical_fields.get(col, 0.0)
            
            # Key vector
            k_vec = self.key_codebook.get(col)
            if k_vec is None:
                oov_keys.append(col)
                k_vec = self.oov_vector
                
            # Value level vector from thermometer range
            v_vec = self.oov_vector
            ranges = self.num_ranges.get(col)
            if ranges is not None:
                min_val = ranges["min"]
                max_val = ranges["max"]
                
                num_levels = len(self.num_codebooks[col])
                if max_val > min_val:
                    normalized = (val - min_val) / (max_val - min_val)
                    normalized = max(0.0, min(1.0, normalized))
                    idx = int(normalized * (num_levels - 1))
                else:
                    idx = 0
                v_vec = self.num_codebooks[col][idx]
                
            bound = torch.bitwise_xor(k_vec, v_vec)
            bound_vectors.append(bound)

        # 3. Bundle all bound vectors using bitwise majority-rule aggregation
        if not bound_vectors:
            packed_result = self.oov_vector
        else:
            stacked_packed = torch.stack(bound_vectors)  # Shape: [M, 2048]
            
            # Fast vectorized unpack
            shifts = torch.arange(7, -1, -1, dtype=torch.uint8, device=stacked_packed.device)
            unpacked = (stacked_packed.unsqueeze(-1) >> shifts) & 1
            unpacked_flat = unpacked.view(len(bound_vectors), self.dimension).to(torch.float32)
            
            summed = unpacked_flat.sum(dim=0)
            majority = (summed > (len(bound_vectors) / 2.0)).to(torch.float32)
            packed_result = self._pack_binary_to_uint8(majority)

        # Convert packed binary back to standard float bipolar vector
        float_bipolar = torch.ones(self.dimension, dtype=torch.float32)
        binary_flat = self._unpack_uint8_to_binary(packed_result)
        float_bipolar[binary_flat > 0.5] = -1.0
        
        return EncodedHypervector(
            vector=float_bipolar,
            is_oov=(len(oov_keys) > 0 or len(oov_vals) > 0)
        )
