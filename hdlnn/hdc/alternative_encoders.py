import math
import torch
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from hdlnn.contracts.interfaces import IEncoder
from hdlnn.contracts.schemas import CanonicalFlow, EncodedHypervector
from hdlnn.hdc.codebooks import CodebookManager
from hdlnn.hdc.item_memory import ItemMemory

logger = logging.getLogger(__name__)


class SAXEncoder(IEncoder):
    """Symbolic Aggregate approXimation (SAX) HDC Encoder conforming to IEncoder.
    
    Transforms numerical time-series attributes into symbolic alphabet representations via 
    Gaussian quantile breakpoints, then binds them with categorical item memory into 
    a D-dimensional hypervector.
    """
    
    def __init__(
        self,
        D: int = 10000,
        alphabet_size: int = 8,
        categorical_columns: Optional[List[str]] = None,
        numerical_columns: Optional[List[str]] = None
    ):
        self.D = D
        self.alphabet_size = alphabet_size
        self.categorical_columns = categorical_columns or []
        self.numerical_columns = numerical_columns or []
        
        self.codebooks = CodebookManager(D=D)
        self.item_memory = ItemMemory(D=D)
        
        # Gaussian distribution breakpoints for SAX alphabet
        # Precomputed standard normal quantiles for alphabet size
        from scipy.stats import norm
        quantiles = np.linspace(0, 1, alphabet_size + 1)[1:-1]
        self.breakpoints = norm.ppf(quantiles).tolist()
        
        # Fitted stats: col -> (mean, std)
        self.stats: Dict[str, tuple[float, float]] = {}
        
        # Alphabet symbol vectors: symbol_index -> torch.Tensor [D]
        self.alphabet_vectors: Dict[int, torch.Tensor] = {}

    def fit(self, train_flows: List[CanonicalFlow]) -> None:
        """Fits numerical means/stds and initializes categorical and symbol codebooks on training data."""
        # 1. Fit numerical mean and std
        self.stats = {}
        for col in self.numerical_columns:
            vals = [flow.numerical_fields.get(col, 0.0) for flow in train_flows]
            if vals:
                mean_v = float(np.mean(vals))
                std_v = float(np.std(vals))
                self.stats[col] = (mean_v, std_v if std_v > 1e-6 else 1.0)
            else:
                self.stats[col] = (0.0, 1.0)
                
        # 2. Build random hypervectors for SAX alphabet symbols
        torch.manual_seed(42)
        self.alphabet_vectors = {}
        for sym_idx in range(self.alphabet_size):
            vec = torch.randint(0, 2, (self.D,), dtype=torch.float32) * 2.0 - 1.0
            self.alphabet_vectors[sym_idx] = vec
            
        # 3. Fit categorical item memory
        self.item_memory.unlock()
        for flow in train_flows:
            for col in self.categorical_columns:
                val = flow.categorical_fields.get(col, "-")
                self.item_memory.get_vector(col, val)
        self.item_memory.lock()
        logger.info(f"Fitted SAXEncoder: alphabet_size={self.alphabet_size}, D={self.D}")

    def _symbolize(self, val: float, mean_v: float, std_v: float) -> int:
        """Assigns normalized continuous value to SAX alphabet symbol index."""
        z = (val - mean_v) / std_v
        for idx, bp in enumerate(self.breakpoints):
            if z < bp:
                return idx
        return len(self.breakpoints)

    def encode(self, flow: CanonicalFlow) -> EncodedHypervector:
        """Encodes a single flow into a D-dimensional SAX hypervector."""
        return self.encode_batch([flow])[0]

    def encode_batch(self, flows: List[CanonicalFlow]) -> List[EncodedHypervector]:
        """Encodes a batch of flows into SAX hypervectors."""
        num_flows = len(flows)
        if num_flows == 0:
            return []
            
        summed = torch.zeros((num_flows, self.D), dtype=torch.float32)
        any_oov = [False] * num_flows
        
        # Categoricals binding
        for col in self.categorical_columns:
            key_vector = self.codebooks.get_key_vector(col)
            col_vals = [flow.categorical_fields.get(col, "-") for flow in flows]
            val_tensor, is_oov_list = self.item_memory.get_batch_vectors(col, col_vals)
            for i, is_oov in enumerate(is_oov_list):
                if is_oov:
                    any_oov[i] = True
            summed.addcmul_(val_tensor, key_vector.unsqueeze(0))
            
        # Numerical SAX symbol binding
        for col in self.numerical_columns:
            key_vector = self.codebooks.get_key_vector(col)
            mean_v, std_v = self.stats.get(col, (0.0, 1.0))
            sym_indices = [self._symbolize(flow.numerical_fields.get(col, 0.0), mean_v, std_v) for flow in flows]
            sym_tensor = torch.stack([self.alphabet_vectors[s] for s in sym_indices])
            summed.addcmul_(sym_tensor, key_vector.unsqueeze(0))
            
        bipolar_vectors = torch.sign(summed)
        bipolar_vectors[bipolar_vectors == 0] = 1.0
        
        return [
            EncodedHypervector(vector=bipolar_vectors[i], is_oov=any_oov[i])
            for i in range(num_flows)
        ]


class RFFEncoder(IEncoder):
    """Random Fourier Features (RFF) HDC Encoder conforming to IEncoder.
    
    Projects continuous feature vectors via random Fourier mappings (cos(Wx + b)) 
    to approximate shift-invariant RBF kernels in high-dimensional space.
    """
    
    def __init__(
        self,
        D: int = 10000,
        gamma: float = 1.0,
        categorical_columns: Optional[List[str]] = None,
        numerical_columns: Optional[List[str]] = None
    ):
        self.D = D
        self.gamma = gamma
        self.categorical_columns = categorical_columns or []
        self.numerical_columns = numerical_columns or []
        
        self.codebooks = CodebookManager(D=D)
        self.item_memory = ItemMemory(D=D)
        
        # RFF projection weights W [num_numerical, D] and bias b [D]
        self.W: Optional[torch.Tensor] = None
        self.b: Optional[torch.Tensor] = None
        self.min_max: Dict[str, tuple[float, float]] = {}

    def fit(self, train_flows: List[CanonicalFlow]) -> None:
        """Fits feature normalization and samples RFF projection weights."""
        num_num = len(self.numerical_columns)
        self.min_max = {}
        
        for col in self.numerical_columns:
            vals = [flow.numerical_fields.get(col, 0.0) for flow in train_flows]
            if vals:
                min_v, max_v = float(min(vals)), float(max(vals))
                self.min_max[col] = (min_v, max_v if max_v > min_v else min_v + 1.0)
            else:
                self.min_max[col] = (0.0, 1.0)
                
        # Sample W ~ N(0, 2*gamma) and b ~ Uniform(0, 2*pi)
        torch.manual_seed(42)
        if num_num > 0:
            self.W = torch.randn((num_num, self.D), dtype=torch.float32) * math.sqrt(2.0 * self.gamma)
            self.b = torch.rand(self.D, dtype=torch.float32) * 2.0 * math.pi
        else:
            self.W = torch.zeros((0, self.D))
            self.b = torch.zeros(self.D)
            
        # Fit categorical item memory
        self.item_memory.unlock()
        for flow in train_flows:
            for col in self.categorical_columns:
                val = flow.categorical_fields.get(col, "-")
                self.item_memory.get_vector(col, val)
        self.item_memory.lock()
        logger.info(f"Fitted RFFEncoder: num_features={num_num}, D={self.D}, gamma={self.gamma}")

    def encode(self, flow: CanonicalFlow) -> EncodedHypervector:
        """Encodes a single flow into an RFF hypervector."""
        return self.encode_batch([flow])[0]

    def encode_batch(self, flows: List[CanonicalFlow]) -> List[EncodedHypervector]:
        """Encodes a batch of flows using Random Fourier Features."""
        num_flows = len(flows)
        if num_flows == 0:
            return []
            
        summed = torch.zeros((num_flows, self.D), dtype=torch.float32)
        any_oov = [False] * num_flows
        
        # 1. Categorical binding
        for col in self.categorical_columns:
            key_vector = self.codebooks.get_key_vector(col)
            col_vals = [flow.categorical_fields.get(col, "-") for flow in flows]
            val_tensor, is_oov_list = self.item_memory.get_batch_vectors(col, col_vals)
            for i, is_oov in enumerate(is_oov_list):
                if is_oov:
                    any_oov[i] = True
            summed.addcmul_(val_tensor, key_vector.unsqueeze(0))
            
        # 2. Continuous RFF projection: sqrt(2/D) * cos(X W + b)
        if len(self.numerical_columns) > 0 and self.W is not None:
            # Build normalized matrix [num_flows, num_num]
            num_matrix = torch.zeros((num_flows, len(self.numerical_columns)), dtype=torch.float32)
            for j, col in enumerate(self.numerical_columns):
                min_v, max_v = self.min_max.get(col, (0.0, 1.0))
                for i, flow in enumerate(flows):
                    val = flow.numerical_fields.get(col, 0.0)
                    norm_v = (val - min_v) / (max_v - min_v)
                    num_matrix[i, j] = max(0.0, min(1.0, norm_v))
                    
            # Compute projection: [num_flows, D]
            proj = torch.matmul(num_matrix, self.W) + self.b.unsqueeze(0)
            rff_features = torch.cos(proj) * math.sqrt(2.0 / self.D)
            summed += rff_features
            
        bipolar_vectors = torch.sign(summed)
        bipolar_vectors[bipolar_vectors == 0] = 1.0
        
        return [
            EncodedHypervector(vector=bipolar_vectors[i], is_oov=any_oov[i])
            for i in range(num_flows)
        ]
