# Apply ncps CfCCell broadcasting patch on library load
import torch
import ncps.torch

_orig_forward = ncps.torch.CfCCell.forward

def _patched_forward(self, input, hx, ts):
    if isinstance(ts, torch.Tensor) and ts.ndim == 1:
        ts = ts.unsqueeze(1)
    return _orig_forward(self, input, hx, ts)

ncps.torch.CfCCell.forward = _patched_forward
