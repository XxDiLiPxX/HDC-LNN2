import torch
import torch.nn as nn
from typing import Tuple, Optional
from hdlnn.contracts.interfaces import ISequenceModel

class AETemporalModel(nn.Module, ISequenceModel):
    """LSTM-based Autoencoder sequence model.
    
    Encodes the sequence into a bottleneck hidden state, and decodes it to reconstruct the inputs.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # LSTM encoder
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        
        # Linear decoder (projects hidden state back to input dimension)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(
        self, 
        x: torch.Tensor, 
        h0: Optional[torch.Tensor] = None, 
        dt: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluates batch sequence updates over time.
        
        Returns:
            dec_out: Reconstructed input sequence [batch_size, seq_len, input_dim]
            h_n: Final hidden state [batch_size, hidden_dim]
        """
        if h0 is not None:
            if isinstance(h0, torch.Tensor):
                if h0.dim() == 1:
                    h0 = h0.unsqueeze(0)
                hx = (h0, torch.zeros_like(h0))
            else:
                hx = h0
        else:
            hx = None
            
        enc_out, (h_n, _) = self.encoder(x, hx)
        dec_out = self.decoder(enc_out)
        
        return dec_out, h_n.squeeze(0)

    def step(self, x: torch.Tensor, h_prev: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """Evaluates continuous state transition for a single time step."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x_seq = x.unsqueeze(1)  # [batch_size, 1, input_dim]
        
        if isinstance(h_prev, torch.Tensor):
            if h_prev.dim() == 1:
                h_prev = h_prev.unsqueeze(0)
            if h_prev.size(1) == 2 * self.hidden_dim:
                h0 = h_prev[:, :self.hidden_dim].unsqueeze(0).contiguous()
                c0 = h_prev[:, self.hidden_dim:].unsqueeze(0).contiguous()
                hx = (h0, c0)
            elif h_prev.size(1) == self.hidden_dim:
                h0 = h_prev.unsqueeze(0).contiguous()
                c0 = torch.zeros_like(h0)
                hx = (h0, c0)
            else:
                hx = None
        else:
            hx = h_prev

        _, (h_n, c_n) = self.encoder(x_seq, hx)
        return torch.cat([h_n.squeeze(0), c_n.squeeze(0)], dim=-1)

    def reconstruct(self, state: torch.Tensor) -> torch.Tensor:
        """Projects hidden state vector back to reconstruct the input hypervector."""
        if state.dim() == 1:
            state = state.unsqueeze(0)
        h = state[:, :self.hidden_dim]
        return self.decoder(h)
        
    def predict_next_vector(self, state: torch.Tensor) -> torch.Tensor:
        """Alias for reconstruct, to maintain compatibility with Cosine EWMA scorer if needed."""
        return self.reconstruct(state)
