import torch
import torch.nn as nn
from typing import Tuple, Optional, Any
from hdlnn.contracts.interfaces import ISequenceModel

class LSTMBaseline(nn.Module, ISequenceModel):
    """Standard LSTM baseline sequence model.
    
    Processes the same sequence inputs but ignores time deltas (dt), mapping to standard RNN behavior.
    """
    def __init__(self, input_dim: int, hidden_dim: int, proj_dim: Optional[int] = None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.proj_dim = proj_dim
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        
        # Optional projection head to map state to hypervector space
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
        
        Note: LSTM ignores time spans (dt).
        """
        if h0 is not None:
            if isinstance(h0, torch.Tensor):
                # LSTM requires a tuple of (h0, c0). We initialize cell state to zeros.
                if h0.dim() == 1:
                    h0 = h0.unsqueeze(0)
                hx = (h0, torch.zeros_like(h0))
            else:
                hx = h0
        else:
            hx = None
            
        outputs, (h_n, c_n) = self.lstm(x, hx)
        # h_n shape is [num_layers, batch_size, hidden_dim]. We squeeze the layer dim (1).
        return outputs, h_n.squeeze(0)

    def step(self, x: torch.Tensor, h_prev: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """Evaluates continuous state transition for a single time step (ignores dt)."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x_seq = x.unsqueeze(1)  # [batch_size, 1, input_dim]
        
        if isinstance(h_prev, torch.Tensor):
            if h_prev.dim() == 1:
                h_prev = h_prev.unsqueeze(0)
            
            # Check if h_prev is composite [batch_size, 2 * hidden_dim] containing (h, c)
            if h_prev.size(-1) == 2 * self.hidden_dim:
                h0 = h_prev[:, :self.hidden_dim].unsqueeze(0)
                c0 = h_prev[:, self.hidden_dim:].unsqueeze(0)
                hx = (h0, c0)
            elif h_prev.size(-1) == self.hidden_dim:
                h0 = h_prev.unsqueeze(0)
                c0 = torch.zeros_like(h0)
                hx = (h0, c0)
            else:
                hx = (h_prev.unsqueeze(0), torch.zeros((1, h_prev.size(0), self.hidden_dim), device=h_prev.device, dtype=h_prev.dtype))
        else:
            hx = h_prev

        _, (h_n, c_n) = self.lstm(x_seq, hx)
        # Return composite state [h_n, c_n] of shape [batch_size, 2 * hidden_dim]
        return torch.cat([h_n.squeeze(0), c_n.squeeze(0)], dim=-1)

    def predict_next_vector(self, state: torch.Tensor) -> torch.Tensor:
        """Projects hidden state vector back to hypervector space (D)."""
        if self.projection_head is None:
            raise ValueError("Projection head not configured in LSTMBaseline.")
        return self.projection_head(state)
