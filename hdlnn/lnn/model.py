import torch
import torch.nn as nn
from typing import Tuple, Optional
from ncps.torch import CfC
from hdlnn.contracts.interfaces import ISequenceModel

class LNNSequenceModel(nn.Module, ISequenceModel):
    """Continuous-time sequence model based on Closed-form Continuous-time (CfC) networks.
    
    Exposes unified batch forward and streaming step methods, guaranteeing numerical parity.
    """
    def __init__(self, input_dim: int, hidden_dim: int, proj_dim: Optional[int] = None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.proj_dim = proj_dim
        
        # CfC recurrent model
        self.cfc = CfC(
            input_size=input_dim,
            units=hidden_dim,
            batch_first=True,
            return_sequences=True
        )
        
        # Optional projection head to map continuous hidden state back to hypervector space
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
        """Evaluates batch sequence updates over time.
        
        Args:
            x: input tensor of shape [batch_size, seq_len, input_dim]
            h0: initial hidden state of shape [batch_size, hidden_dim]
            dt: delta timespans of shape [batch_size, seq_len]
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Sequence outputs and the final hidden state.
        """
        output_seq, h_last = self.cfc(x, hx=h0, timespans=dt)
        return output_seq, h_last

    def step(self, x: torch.Tensor, h_prev: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """Evaluates continuous state transition for a single time step.
        
        Runs the exact same sequence evaluation path to guarantee strict numerical parity.
        
        Args:
            x: current input vector of shape [batch_size, input_dim]
            h_prev: previous hidden state of shape [batch_size, hidden_dim]
            dt: delta time of shape [batch_size] or [batch_size, 1]
            
        Returns:
            torch.Tensor: The next hidden state vector of shape [batch_size, hidden_dim]
        """
        # Ensure dimensions match batched expectations
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if h_prev.dim() == 1:
            h_prev = h_prev.unsqueeze(0)
            
        # Form sequence of length 1
        x_seq = x.unsqueeze(1)  # [batch_size, 1, input_dim]
        
        if dt.dim() == 1:
            dt_seq = dt.unsqueeze(1)
        elif dt.dim() == 2:
            dt_seq = dt
        else:
            dt_seq = dt.view(x.size(0), 1)

        # Call underlying CfC module forward pass with seq_len = 1
        _, h_next = self.cfc(x_seq, hx=h_prev, timespans=dt_seq)
        return h_next

    def predict_next_vector(self, state: torch.Tensor) -> torch.Tensor:
        """Projects continuous hidden state vector back to the hypervector space (D)."""
        if self.projection_head is None:
            raise ValueError("Projection head not configured in LNNSequenceModel.")
        return self.projection_head(state)
