import random
import numpy as np
import torch

def set_seed(seed: int):
    """Sets the random seed for Python, NumPy, and PyTorch to ensure determinism."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic algorithms where possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
