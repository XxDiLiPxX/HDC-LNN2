import torch
import logging
import numpy as np
from sklearn.covariance import LedoitWolf

logger = logging.getLogger(__name__)

class ReferenceManifold:
    """Represents the statistical reference manifold of normal trajectory states.
    
    Fits the mean and inverse covariance parameters using normal-traffic validation states.
    Uses Ledoit-Wolf shrinkage covariance estimation by default.
    """
    def __init__(self, hidden_dim: int, eps: float = 1e-5):
        self.hidden_dim = hidden_dim
        self.eps = eps
        
        # Parameters (initialized to zeros/identity, updated on fit)
        self.mean = torch.zeros(hidden_dim)
        self.inv_cov = torch.eye(hidden_dim)
        
        self.is_fit = False

    def fit(self, states: torch.Tensor) -> None:
        """Fits the mean and regularized inverse covariance of normal trajectory states.
        
        Uses Ledoit-Wolf shrinkage covariance estimator.
        If the dimension is high (>1000), we skip covariance calculations for efficiency
        and use Cosine Distance.
        
        Args:
            states: Tensor of shape [N, hidden_dim] containing normal validation hidden states.
        """
        if states.dim() != 2 or states.size(1) != self.hidden_dim:
            raise ValueError(f"Expected input states of shape [N, {self.hidden_dim}], got {list(states.shape)}")
            
        logger.info(f"Fitting Reference Manifold on {len(states)} normal states...")
        
        # Calculate mean
        self.mean = states.mean(dim=0)
        
        if self.hidden_dim > 1000:
            # Skip covariance inversion for high-dimensional space (performance optimization)
            self.inv_cov = None
            self.is_fit = True
            logger.info("High-dimensional manifold: Skipping covariance calculation, defaulting to Cosine Distance.")
            return
            
        # Fit Ledoit-Wolf shrinkage covariance
        try:
            states_np = states.detach().cpu().numpy()
            lw = LedoitWolf()
            lw.fit(states_np)
            # Precision matrix is the inverse covariance matrix
            self.inv_cov = torch.from_numpy(lw.precision_).to(dtype=states.dtype, device=states.device)
            logger.info(f"Reference Manifold Ledoit-Wolf fitting completed (shrinkage={lw.shrinkage_:.4f}).")
        except Exception as e:
            logger.warning(f"Ledoit-Wolf fitting failed ({e}), falling back to empirical regularized covariance.")
            cov = torch.cov(states.T)
            regularizer = self.eps * torch.eye(self.hidden_dim, device=cov.device)
            cov = cov + regularizer
            self.inv_cov = torch.linalg.inv(cov)
            
        self.is_fit = True

    def compute_mahalanobis_distance(self, states: torch.Tensor) -> torch.Tensor:
        """Computes the distance (Mahalanobis or Cosine fallback) for a batch or single hidden state vector.
        
        Args:
            states: Tensor of shape [batch_size, hidden_dim] or [hidden_dim]
            
        Returns:
            Tensor: Distance(s).
        """
        if not self.is_fit:
            raise RuntimeError("Cannot compute distance: ReferenceManifold is not fit.")

        # Ensure parameters are on the same device/dtype as the input states
        self.mean = self.mean.to(device=states.device, dtype=states.dtype)
        if self.inv_cov is not None:
            self.inv_cov = self.inv_cov.to(device=states.device, dtype=states.dtype)

        is_batched = states.dim() == 2
        if not is_batched:
            states = states.unsqueeze(0)  # Shape: [1, hidden_dim]

        if self.hidden_dim > 1000:
            # High-dimensional Cosine Distance: 1.0 - CosineSimilarity
            sim = torch.cosine_similarity(states, self.mean.unsqueeze(0), dim=-1)
            dist = 1.0 - sim
            dist = torch.clamp(dist, min=0.0)
            if not is_batched:
                return dist.squeeze(0)
            return dist

        diff = states - self.mean  # Shape: [batch_size, hidden_dim]
        
        # Mahalanobis distance calculation: sqrt( (x-mu)^T * inv_cov * (x-mu) )
        temp = diff @ self.inv_cov
        squared_dist = torch.sum(temp * diff, dim=-1)
        
        # Clamp to 0.0 to prevent negative values from precision rounding
        squared_dist = torch.clamp(squared_dist, min=0.0)
        dist = torch.sqrt(squared_dist)

        if not is_batched:
            return dist.squeeze(0)  # return scalar tensor
        return dist
