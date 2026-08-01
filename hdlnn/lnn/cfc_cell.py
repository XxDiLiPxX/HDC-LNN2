import torch
import ncps.torch

# Save the original forward method of CfCCell
_orig_forward = ncps.torch.CfCCell.forward

def _patched_forward(self, input, hx, ts):
    """Patches ncps CfCCell.forward to correctly handle 1D timespan (ts) tensor broadcasting
    when batch_size > 1.
    """
    if isinstance(ts, torch.Tensor) and ts.ndim == 1:
        ts = ts.unsqueeze(1)
    return _orig_forward(self, input, hx, ts)

# Dynamic monkey patch
ncps.torch.CfCCell.forward = _patched_forward

# Re-export CfCCell
CfCCell = ncps.torch.CfCCell

__all__ = ["CfCCell"]
