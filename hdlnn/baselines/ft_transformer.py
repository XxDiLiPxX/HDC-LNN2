import math
import logging
from typing import Tuple, Optional, Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from hdlnn.contracts.interfaces import ISequenceModel

logger = logging.getLogger(__name__)


class NumericalFeatureTokenizer(nn.Module):
    """Tokenizes continuous numerical features by applying per-feature linear transformations.
    
    Each continuous feature x_j is transformed into a token vector e_j = x_j * W_j + b_j.
    """
    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        self.n_features = n_features
        self.d_token = d_token
        # Weights: [n_features, d_token], Biases: [n_features, d_token]
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token))
        self.reset_parameters()

    def reset_parameters(self):
        d = self.d_token
        nn.init.uniform_(self.weight, -1.0 / math.sqrt(d), 1.0 / math.sqrt(d))
        nn.init.uniform_(self.bias, -1.0 / math.sqrt(d), 1.0 / math.sqrt(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., n_features] -> output: [..., n_features, d_token]
        x_expanded = x.unsqueeze(-1)  # [..., n_features, 1]
        tokens = x_expanded * self.weight + self.bias
        return tokens


class CategoricalFeatureTokenizer(nn.Module):
    """Tokenizes categorical features using per-feature embedding tables."""
    def __init__(self, cardinalities: List[int], d_token: int):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(card, d_token) for card in cardinalities
        ])
        self.d_token = d_token

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        # x_cat: [..., n_cat_features] of integer indices
        tokens = [emb(x_cat[..., i]) for i, emb in enumerate(self.embeddings)]
        return torch.stack(tokens, dim=-2)  # [..., n_cat_features, d_token]


class FTTransformerSequenceModel(nn.Module, ISequenceModel):
    """Compact FT-Transformer (Feature Tokenizer Transformer) tabular model.
    
    Adapted for cybersecurity stream/sequence anomaly detection in the HDC-LNN benchmark.
    Consists of:
      1. Compact Feature Tokenizer: maps input into a sequence of tabular tokens.
         Uses bottleneck in-projection when input_dim is large (e.g. D=10,000) to keep parameters
         fair and strictly within the benchmark constraint (~1.46M total parameters).
      2. CLS Token: learned prefix token aggregating feature-level contextual representations.
      3. Transformer Backbone: 3 Pre-Norm Transformer Encoder layers with multi-head self-attention.
      4. Projection Head: projects the contextual representation to output/hypervector space.
    
    Target config: 2-4 Transformer blocks, 4 heads, d_token=64, parameters << 1B.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 3,
        n_heads: int = 4,
        ffn_factor: float = 2.0,
        dropout: float = 0.1,
        proj_dim: Optional[int] = None,
        n_tokens: int = 16
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.proj_dim = proj_dim
        self.d_token = hidden_dim
        self.n_tokens = n_tokens
        d_ffn = int(self.d_token * ffn_factor)
        
        # Compact input tokenizer:
        # 1. Project input_dim -> hidden_dim (e.g. 10000 -> 64: 640k params)
        # 2. Expand hidden_dim -> n_tokens * d_token (e.g. 64 -> 16*64: 65k params)
        if input_dim != hidden_dim:
            self.in_proj = nn.Linear(input_dim, hidden_dim)
        else:
            self.in_proj = nn.Identity()
            
        self.token_expand = nn.Linear(hidden_dim, self.n_tokens * self.d_token)
            
        # CLS token prepended to token sequence: shape [1, 1, d_token]
        self.cls_token = nn.Parameter(torch.empty(1, 1, self.d_token))
        nn.init.normal_(self.cls_token, std=0.02)
        
        # Transformer Encoder layers (Pre-LN)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_token,
            nhead=self.n_heads,
            dim_feedforward=d_ffn,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)
        self.norm = nn.LayerNorm(self.d_token)
        
        # Output projection head to project representation back to hypervector dimension
        if proj_dim:
            self.projection_head = nn.Linear(self.d_token, proj_dim)
        else:
            self.projection_head = None

    def _tokenize_flow(self, x: torch.Tensor) -> torch.Tensor:
        """Converts input tensor into token sequence [batch_size, n_tokens + 1, d_token]."""
        batch_size = x.size(0)
        h = F.gelu(self.in_proj(x))
        flat_tokens = self.token_expand(h)  # [B, n_tokens * d_token]
        tokens = flat_tokens.view(batch_size, self.n_tokens, self.d_token)
            
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens_with_cls = torch.cat([cls, tokens], dim=1)  # [B, 1 + n_tokens, d_token]
        return tokens_with_cls

    def forward(
        self,
        x: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
        dt: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluates batch sequence updates over time.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            h0: Optional initial state (unused, for API parity)
            dt: Optional time deltas (unused)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
              - sequence representations [batch_size, seq_len, d_token]
              - final step representation [batch_size, d_token]
        """
        batch_size, seq_len, input_dim = x.shape
        x_flat = x.view(batch_size * seq_len, input_dim)
        
        tokens = self._tokenize_flow(x_flat)  # [B * L, 1 + n_tokens, d_token]
        encoded = self.transformer(tokens)  # [B * L, 1 + n_tokens, d_token]
        cls_out = self.norm(encoded[:, 0, :])  # Extract [CLS] token: [B * L, d_token]
        
        seq_out = cls_out.view(batch_size, seq_len, self.d_token)
        h_last = seq_out[:, -1, :]
        return seq_out, h_last

    def step(self, x: torch.Tensor, h_prev: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """Evaluates continuous state transition for a single time step.
        
        For streaming evaluation:
          Takes single flow x: [batch_size, input_dim] or [input_dim],
          computes its contextual CLS embedding,
          and returns it as the next state.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        tokens = self._tokenize_flow(x)
        encoded = self.transformer(tokens)
        cls_out = self.norm(encoded[:, 0, :])  # [batch_size, d_token]
        return cls_out

    def predict_next_vector(self, state: torch.Tensor) -> torch.Tensor:
        """Projects hidden state representation back to hypervector space (D)."""
        if self.projection_head is None:
            raise ValueError("Projection head not configured in FTTransformerSequenceModel.")
        return self.projection_head(state)
