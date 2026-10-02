"""Model inspection and parameter counting utilities."""
from typing import Dict, Any
import torch.nn as nn

def count_parameters(model: nn.Module) -> Dict[str, Any]:
    """Calculate exact total and trainable parameter counts for any PyTorch model.
    
    Args:
        model: Instantiated torch.nn.Module
        
    Returns:
        Dict containing total_params, trainable_params, and params_m (in millions)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_m = round(total_params / 1e6, 4)
    trainable_m = round(trainable_params / 1e6, 4)
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "params_m": params_m,
        "trainable_m": trainable_m,
        "formatted": f"{params_m:.2f}M"
    }
