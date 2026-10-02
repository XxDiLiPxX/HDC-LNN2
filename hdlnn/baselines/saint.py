import math
import logging
from typing import Tuple, Optional, Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from hdlnn.contracts.interfaces import ISequenceModel

logger = logging.getLogger(__name__)


class SAINTBlock(nn.Module):
    """SAINT Transformer Block: Alternating Feature (Column) Self-Attention and Intersample/Temporal Attention.
    
    Self-Attention and Intersample Attention (SAINT):
      1. Column Attention: Self-attention across feature tokens within each record.
      2. Row / Temporal Attention: Self-attention across intersample / temporal dimension within sequences.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        d_ffn: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        # 1. Column / Feature Self-Attention
        self.col_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.col_norm1 = nn.LayerNorm(d_model)
        self.col_ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
            nn.Dropout(dropout)
        )
        self.col_norm2 = nn.LayerNorm(d_model)
        
        # 2. Row / Temporal Intersample Attention
        self.row_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.row_norm1 = nn.LayerNorm(d_model)
        self.row_ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
            nn.Dropout(dropout)
        )
        self.row_norm2 = nn.LayerNorm(d_model)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Applies column/feature self-attention within each flow token representation.
        
        Args:
            x: Tensor of shape [N_samples, n_tokens, d_model]
        """
        norm_x = self.col_norm1(x)
        attn_out, _ = self.col_attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        x = x + self.col_ffn(self.col_norm2(x))
        return x

    def forward_row(self, x: torch.Tensor) -> torch.Tensor:
        """Applies row/intersample attention across sequence steps or samples.
        
        Args:
            x: Tensor of shape [batch_size, seq_len, n_tokens, d_model] or [batch_size, n_tokens, d_model]
        """
        if x.dim() == 4:
            # Shape: [B, L, N, d_model]
            B, L, N, D = x.shape
            if L > 1:
                # Attend across temporal sequence steps L for each feature token N
                norm_x = self.row_norm1(x)
                x_perm = norm_x.permute(0, 2, 1, 3).reshape(B * N, L, D)
                row_out, _ = self.row_attn(x_perm, x_perm, x_perm)
                row_back = row_out.view(B, N, L, D).permute(0, 2, 1, 3)
                x = x + row_back
            x = x + self.row_ffn(self.row_norm2(x))
            return x
        elif x.dim() == 3:
            # Shape: [B, N, d_model]
            B, N, D = x.shape
            if B > 1:
                norm_x = self.row_norm1(x)
                x_trans = norm_x.transpose(0, 1) # [N, B, D]
                row_out, _ = self.row_attn(x_trans, x_trans, x_trans)
                x = x + row_out.transpose(0, 1)
            x = x + self.row_ffn(self.row_norm2(x))
            return x
        else:
            return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Default forward when input is [batch_size, n_tokens, d_model]."""
        x = self.forward_features(x)
        x = self.forward_row(x)
        return x


class SAINTSequenceModel(nn.Module, ISequenceModel):
    """Compact SAINT (Self-Attention and Intersample Attention) tabular Transformer model.
    
    Adapted for cybersecurity stream/sequence anomaly detection in the HDC-LNN benchmark.
    Combines:
      1. Tabular Tokenizer: transforms tabular features into token vectors using a compact bottleneck.
      2. Learned CLS prefix token.
      3. SAINT Backbone: 2 SAINT blocks alternating column and row attention.
      4. Projection Head: projects contextual states back to hypervector space.
    
    Target config: 2 SAINT blocks, 4 heads, d_model=64, parameters << 1B (~1.49M params).
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
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
        self.d_model = hidden_dim
        self.n_tokens = n_tokens
        d_ffn = int(self.d_model * ffn_factor)
        
        # Compact input tokenizer
        if input_dim != hidden_dim:
            self.in_proj = nn.Linear(input_dim, hidden_dim)
        else:
            self.in_proj = nn.Identity()
            
        self.token_expand = nn.Linear(hidden_dim, self.n_tokens * self.d_model)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.empty(1, 1, self.d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        
        # SAINT blocks
        self.blocks = nn.ModuleList([
            SAINTBlock(d_model=self.d_model, n_heads=self.n_heads, d_ffn=d_ffn, dropout=dropout)
            for _ in range(self.n_layers)
        ])
        self.norm = nn.LayerNorm(self.d_model)
        
        # Projection head to map representation back to hypervector space
        if proj_dim:
            self.projection_head = nn.Linear(self.d_model, proj_dim)
        else:
            self.projection_head = None

    def _tokenize(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        h = F.gelu(self.in_proj(x))
        flat_tokens = self.token_expand(h)
        tokens = flat_tokens.view(batch_size, self.n_tokens, self.d_model)
        cls = self.cls_token.expand(batch_size, -1, -1)
        return torch.cat([cls, tokens], dim=1)  # [B, 1 + n_tokens, d_model]

    def forward(
        self,
        x: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
        dt: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluates batch sequence updates over time.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            h0: Optional initial state (unused)
            dt: Optional time deltas (unused)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Sequence outputs and the final hidden state.
        """
        batch_size, seq_len, input_dim = x.shape
        x_flat = x.view(batch_size * seq_len, input_dim)
        tokens = self._tokenize(x_flat)  # [B * L, N, d_model]
        N = tokens.shape[1]
        
        tokens_seq = tokens.view(batch_size, seq_len, N, self.d_model)
        
        for block in self.blocks:
            # 1. Feature attention within each flow
            tokens_flat = tokens_seq.view(batch_size * seq_len, N, self.d_model)
            tokens_flat = block.forward_features(tokens_flat)
            tokens_seq = tokens_flat.view(batch_size, seq_len, N, self.d_model)
            
            # 2. Row / Sequence attention across sequence steps
            tokens_seq = block.forward_row(tokens_seq)
            
        cls_out = self.norm(tokens_seq[:, :, 0, :])  # [B, L, d_model]
        h_last = cls_out[:, -1, :]
        return cls_out, h_last

    def step(self, x: torch.Tensor, h_prev: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """Evaluates continuous state transition for a single time step."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        tokens = self._tokenize(x)
        for block in self.blocks:
            tokens = block(tokens)
        cls_out = self.norm(tokens[:, 0, :])  # [batch_size, d_model]
        return cls_out

    def predict_next_vector(self, state: torch.Tensor) -> torch.Tensor:
        """Projects hidden state representation back to hypervector space (D)."""
        if self.projection_head is None:
            raise ValueError("Projection head not configured in SAINTSequenceModel.")
        return self.projection_head(state)
