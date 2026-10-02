import torch
import torch.nn as nn
from typing import Tuple, Optional
from ncps.torch import CfC
from hdlnn.contracts.interfaces import ISequenceModel

class LNNSequenceModel(nn.Module, ISequenceModel):
    """Continuous-time sequence model based on Closed-form Continuous-time (CfC) networks.
    
    Exposes unified batch forward and streaming step methods, guaranteeing numerical parity.
    """
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        proj_dim: Optional[int] = None,
        backbone_units: int = 128
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.proj_dim = proj_dim
        self.backbone_units = backbone_units
        
        # CfC recurrent model with configurable backbone dimension
        self.cfc = CfC(
            input_size=input_dim,
            units=hidden_dim,
            backbone_units=backbone_units,
            batch_first=True,
            return_sequences=True
        )
        
        # Optional projection head to map continuous hidden state back to hypervector space
        # Optional projection head to map continuous hidden state back to hypervector space
        if proj_dim:
            self.projection_head = nn.Linear(hidden_dim, proj_dim)
        else:
            self.projection_head = None
            
        self._fused_W: Optional[torch.Tensor] = None
        self._fused_b: Optional[torch.Tensor] = None
        self._bb_linear = None
        self._bb_act = None

    def fuse_for_inference(self) -> None:
        """Precomputes fused linear weights for CfC cell heads (ff1, ff2, time_a, time_b)."""
        if hasattr(self.cfc, "rnn_cell"):
            cell = self.cfc.rnn_cell
            if hasattr(cell, "ff1") and hasattr(cell, "ff2") and hasattr(cell, "time_a") and hasattr(cell, "time_b"):
                with torch.no_grad():
                    self._fused_W = torch.cat([cell.ff1.weight, cell.ff2.weight, cell.time_a.weight, cell.time_b.weight], dim=0)
                    self._fused_b = torch.cat([cell.ff1.bias, cell.ff2.bias, cell.time_a.bias, cell.time_b.bias], dim=0)
                    self._bb_linear = cell.backbone[0]
                    self._bb_act = cell.backbone[1]

    def export_inference_model(self) -> "LNNSequenceModel":
        """Creates an inference-optimized copy by stripping training-only projection head."""
        inf_model = LNNSequenceModel(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            proj_dim=None,
            backbone_units=self.backbone_units
        )
        inf_model.cfc.load_state_dict(self.cfc.state_dict())
        inf_model.fuse_for_inference()
        inf_model.eval()
        return inf_model

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

    def step(self, x: torch.Tensor, h_prev: torch.Tensor, dt: torch.Tensor, return_instantaneous: bool = False):
        """Evaluates continuous state transition for a single time step.
        
        Uses the underlying CfC cell directly to eliminate sequence wrapping overhead
        while guaranteeing strict mathematical equivalence.
        Assumes inputs: x=[B, D] or [D], h_prev=[B, H] or [H], dt=[B] or [B, 1] or scalar.
        If return_instantaneous=True, returns tuple (h_next, bb_out) where bb_out is the 128-D instantaneous representation.
        """
        x_2d = x.unsqueeze(0) if x.dim() == 1 else (x.squeeze(1) if x.dim() == 3 else x)
        h_2d = h_prev.unsqueeze(0) if h_prev.dim() == 1 else h_prev
        dt_flat = dt.view(-1) if dt.dim() > 0 else dt.unsqueeze(0)
        
        # Fast fused single-flow path
        if self._fused_W is not None and x_2d.size(0) == 1:
            x_in = torch.cat([x_2d, h_2d], 1)
            bb_out = self._bb_act(self._bb_linear(x_in))
            fused = torch.nn.functional.linear(bb_out, self._fused_W, self._fused_b)
            ff1, ff2, t_a, t_b = torch.split(fused, self.hidden_dim, dim=1)
            ff1 = torch.tanh(ff1)
            ff2 = torch.tanh(ff2)
            t_interp = torch.sigmoid(t_a * dt_flat + t_b)
            h_next = ff1 * (1.0 - t_interp) + t_interp * ff2
            if return_instantaneous:
                return h_next, bb_out
            return h_next
            
        if hasattr(self.cfc, "rnn_cell"):
            cell = self.cfc.rnn_cell
            x_in = torch.cat([x_2d, h_2d], 1)
            if hasattr(cell, "backbone") and return_instantaneous:
                bb_out = cell.backbone(x_in)
            else:
                bb_out = None
            _, h_next = cell(x_2d, h_2d, dt_flat)
        else:
            x_seq = x_2d.unsqueeze(1)
            dt_seq = dt_flat.view(x_2d.size(0), 1)
            _, h_next = self.cfc(x_seq, hx=h_2d, timespans=dt_seq)
            bb_out = None
            
        if return_instantaneous:
            return h_next, bb_out
        return h_next

    def predict_next_vector(self, state: torch.Tensor) -> torch.Tensor:
        """Projects continuous hidden state vector back to the hypervector space (D)."""
        if self.projection_head is None:
            raise ValueError("Projection head not configured in LNNSequenceModel.")
        return self.projection_head(state)
