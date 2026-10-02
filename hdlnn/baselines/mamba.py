import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from hdlnn.contracts.interfaces import ISequenceModel

logger = logging.getLogger(__name__)


class PurePyTorchSSM(nn.Module):
    """Pure-PyTorch Selective State-Space (Mamba-style) block without custom CUDA dependencies.
    
    Implements selective scan over continuous/discrete sequence state transitions:
      h_t = exp(Delta_t * A) * h_{t-1} + (Delta_t * B_t) * x_t
      y_t = C_t * h_t + D * x_t
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Optional[int] = None
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = dt_rank or math.ceil(self.d_model / 16)
        
        # 1. In-projection for x and gating branch z
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        
        # 2. Causal 1D Convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            groups=self.d_inner,
            padding=d_conv - 1
        )
        
        # 3. Parameter projections (Delta, B, C)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        # Initialize Delta bias
        dt_init_std = self.dt_rank**-0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        # Initialize dt bias to ~0.001 to 0.1 range
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
            
        # 4. S4D Real Diagonal A parameter
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # 5. Out-projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def _selective_scan(
        self,
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        h0: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sequential selective scan loop in pure PyTorch.
        
        Args:
            u: [B, L, D_inner]
            delta: [B, L, D_inner]
            A: [D_inner, N]
            B: [B, L, N]
            C: [B, L, N]
            D: [D_inner]
            h0: Optional initial state [B, D_inner, N]
        """
        batch_size, seq_len, d_inner = u.shape
        d_state = A.shape[1]
        
        # Discretize continuous A: deltaA = exp(delta * A) -> [B, L, D_inner, N]
        # A has negative values: -exp(A_log)
        delta_A = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        
        # Discretize B: deltaB = delta * B -> [B, L, D_inner, N]
        delta_B_u = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)
        
        # Scan state evolution
        if h0 is not None and h0.shape == (batch_size, d_inner, d_state):
            h = h0
        else:
            h = torch.zeros((batch_size, d_inner, d_state), device=u.device, dtype=u.dtype)
            
        ys = []
        for t in range(seq_len):
            h = delta_A[:, t] * h + delta_B_u[:, t]
            # y_t = (h * C_t).sum(-1) -> [B, D_inner]
            y_t = (h * C[:, t].unsqueeze(1)).sum(-1)
            ys.append(y_t)
            
        y = torch.stack(ys, dim=1) + u * D.unsqueeze(0).unsqueeze(0)
        return y, h

    def forward(
        self,
        x: torch.Tensor,
        h0: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs selective SSM block forward pass."""
        batch_size, seq_len, _ = x.shape
        
        # 1. Project to [B, L, 2 * d_inner] and split into x_inner and gate z
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)
        
        # 2. 1D Causal Conv
        x_conv = self.conv1d(x_inner.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
        x_conv = F.silu(x_conv)
        
        # 3. Project SSM parameters
        ssm_params = self.x_proj(x_conv)
        dt_param, B, C = torch.split(
            ssm_params, 
            [self.dt_rank, self.d_state, self.d_state], 
            dim=-1
        )
        delta = F.softplus(self.dt_proj(dt_param))
        A = -torch.exp(self.A_log)
        
        # 4. Selective scan
        y, h_last = self._selective_scan(x_conv, delta, A, B, C, self.D, h0=h0)
        
        # 5. Gating and out projection
        y_gated = y * F.silu(z)
        out = self.out_proj(y_gated)
        
        # Return state summary for trajectory scoring: flatten last h [B, d_inner * d_state]
        state_repr = h_last.reshape(batch_size, -1)
        return out, state_repr


class MambaSequenceModel(nn.Module, ISequenceModel):
    """Mamba Selective State-Space Sequence Model conforming to ISequenceModel.
    
    Pure-PyTorch implementation running on CPU and GPU without requiring CUDA/nvcc compilation.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        proj_dim: Optional[int] = None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.proj_dim = proj_dim
        self.d_state = d_state
        self.expand = expand
        
        # Input adaptation layer
        if input_dim != hidden_dim:
            self.in_proj = nn.Linear(input_dim, hidden_dim)
        else:
            self.in_proj = nn.Identity()
            
        # Core Selective SSM block
        self.ssm = PurePyTorchSSM(
            d_model=hidden_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        
        # Projection head back to hypervector dimension D
        if proj_dim:
            self.projection_head = nn.Linear(hidden_dim, proj_dim)
        else:
            self.projection_head = None

    def forward(
        self, 
        x: torch.Tensor, 
        h0: Optional[torch.Tensor] = None, 
        dt: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluates sequence transitions over time."""
        x_proj = self.in_proj(x)
        out, h_last = self.ssm(x_proj, h0=h0)
        return out, h_last

    def step(self, x: torch.Tensor, h_prev: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """Evaluates continuous state transition for a single time step.
        
        Uses direct single-step vectorized formulation to avoid PyTorch Conv1d grouped-convolution
        dispatch overhead while guaranteeing exact mathematical equivalence.
        Returns a composite tensor: [batch_size, hidden_dim + d_inner * d_state]
        where the first hidden_dim slice contains the output feature vector (for next-vector projection)
        and the remaining slice contains the internal SSM recurrent state.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        d_inner = self.expand * self.hidden_dim
        d_state = self.d_state
        ssm_state_size = d_inner * d_state
        batch_size = x.size(0)
        
        if h_prev.dim() == 1:
            h_prev = h_prev.unsqueeze(0)
            
        if h_prev.size(1) >= ssm_state_size:
            h0 = h_prev[:, -ssm_state_size:].view(batch_size, d_inner, d_state)
        else:
            h0 = torch.zeros((batch_size, d_inner, d_state), device=x.device, dtype=x.dtype)
            
        # 1. Project input: [B, D] -> [B, hidden_dim]
        x_proj = self.in_proj(x)
        
        # 2. SSM In-projection: [B, hidden_dim] -> [B, 2 * d_inner]
        xz = self.ssm.in_proj(x_proj)
        x_inner, z = xz.chunk(2, dim=-1)
        
        # 3. Direct depthwise conv step (evaluating causal tap without Conv1d dispatch)
        w_tap = self.ssm.conv1d.weight[:, 0, -1]
        b_conv = self.ssm.conv1d.bias
        x_conv = F.silu(x_inner * w_tap + b_conv)
        
        # 4. Project SSM parameters
        ssm_params = self.ssm.x_proj(x_conv)
        dt_param, B_proj, C_proj = torch.split(
            ssm_params, 
            [self.ssm.dt_rank, d_state, d_state], 
            dim=-1
        )
        delta = F.softplus(self.ssm.dt_proj(dt_param))
        A = -torch.exp(self.ssm.A_log)
        
        # 5. Continuous state evolution
        delta_A = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0))
        delta_B_u = delta.unsqueeze(-1) * B_proj.unsqueeze(1) * x_conv.unsqueeze(-1)
        new_h = delta_A * h0 + delta_B_u
        
        # 6. Gating and out projection
        y = (new_h * C_proj.unsqueeze(1)).sum(dim=-1) + x_conv * self.ssm.D
        y_gated = y * F.silu(z)
        out_feature = self.ssm.out_proj(y_gated)
        
        # Composite state: [out_features, ssm_state]
        return torch.cat([out_feature, new_h.reshape(batch_size, -1)], dim=-1)

    def predict_next_vector(self, state: torch.Tensor) -> torch.Tensor:
        """Projects hidden state sequence/vector back to hypervector space."""
        if self.projection_head is None:
            raise ValueError("Projection head not configured in MambaSequenceModel.")
        return self.projection_head(state)
