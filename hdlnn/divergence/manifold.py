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
    def __init__(self, hidden_dim: int, eps: float = 1e-5, n_clusters: int = 1, covariance_type: str = "full", trim_ratio: float = 0.02):
        self.hidden_dim = hidden_dim
        self.eps = eps
        self.n_clusters = max(1, n_clusters)
        self.covariance_type = covariance_type.lower()
        self.trim_ratio = max(0.0, min(0.20, trim_ratio))
        
        # Single cluster parameters
        self.mean = torch.zeros(hidden_dim)
        self.inv_cov = torch.eye(hidden_dim)
        
        # Multi-cluster parameters
        self.means = []
        self.inv_covs = []
        
        # Precomputed stacked tensors for vectorized inference
        self.means_stacked: Optional[torch.Tensor] = None
        self.inv_covs_stacked: Optional[torch.Tensor] = None
        self.diag_inv_covs_stacked: Optional[torch.Tensor] = None
        self.shared_inv_cov: Optional[torch.Tensor] = None
        
        self.is_fit = False

    def fit(self, states: torch.Tensor) -> None:
        """Fits the mean and regularized inverse covariance of normal trajectory states.
        
        Supports single-cluster Gaussian and multi-cluster (K-Means + Ledoit-Wolf) reference manifolds.
        Optionally applies trimmed normal core estimation to filter out rare training outliers.
        
        Args:
            states: Tensor of shape [N, hidden_dim] containing normal validation hidden states.
        """
        if states.dim() != 2 or states.size(1) != self.hidden_dim:
            raise ValueError(f"Expected input states of shape [N, {self.hidden_dim}], got {list(states.shape)}")
            
        logger.info(f"Fitting Reference Manifold on {len(states)} normal states (n_clusters={self.n_clusters}, trim_ratio={self.trim_ratio})...")
        
        # Trim high-variance normal outliers if trim_ratio > 0
        if self.trim_ratio > 0.0 and len(states) >= 50:
            init_mean = states.mean(dim=0, keepdim=True)
            dists_sq = torch.sum((states - init_mean) ** 2, dim=-1).detach().cpu().numpy()
            cutoff = np.percentile(dists_sq, 100.0 * (1.0 - self.trim_ratio))
            keep_mask = (dists_sq <= cutoff)
            if np.sum(keep_mask) >= max(self.n_clusters * 10, self.hidden_dim):
                states = states[keep_mask]
                logger.info("Trimmed %d outlier normal states (retained %d states)", int(np.sum(~keep_mask)), len(states))

        # Calculate global mean
        self.mean = states.mean(dim=0)
        
        if self.hidden_dim > 1000:
            self.inv_cov = None
            self.is_fit = True
            logger.info("High-dimensional manifold: Skipping covariance calculation, defaulting to Cosine Distance.")
            return

        states_np = states.detach().cpu().numpy()
        
        # Fallback to single cluster if states count is too small
        effective_clusters = self.n_clusters
        if len(states) < effective_clusters * 20:
            effective_clusters = 1

        if effective_clusters > 1:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=effective_clusters, random_state=42, n_init=5).fit(states_np)
            self.means = []
            self.inv_covs = []
            for k in range(effective_clusters):
                c_states = states[kmeans.labels_ == k]
                c_mean = torch.from_numpy(kmeans.cluster_centers_[k]).to(dtype=states.dtype, device=states.device)
                self.means.append(c_mean)
                
                # Fit shrinkage covariance per cluster
                try:
                    if len(c_states) >= self.hidden_dim:
                        lw = LedoitWolf().fit(c_states.detach().cpu().numpy())
                        inv_cov = torch.from_numpy(lw.precision_).to(dtype=states.dtype, device=states.device)
                    else:
                        # Fallback to pooled shrinkage covariance
                        lw = LedoitWolf().fit(states_np)
                        inv_cov = torch.from_numpy(lw.precision_).to(dtype=states.dtype, device=states.device)
                except Exception:
                    cov = torch.cov(states.T) + self.eps * torch.eye(self.hidden_dim, device=states.device)
                    inv_cov = torch.linalg.inv(cov)
                self.inv_covs.append(inv_cov)
            
            # Precompute stacked tensors for vectorized inference
            self.means_stacked = torch.stack(self.means) # [K, hidden_dim]
            
            if self.covariance_type == "shared":
                # Compute single shared pooled shrinkage covariance across all normal states
                lw_shared = LedoitWolf().fit(states_np)
                self.shared_inv_cov = torch.from_numpy(lw_shared.precision_).to(dtype=states.dtype, device=states.device)
                logger.info(f"Precomputed shared covariance matrix for {effective_clusters} clusters.")
            elif self.covariance_type == "diagonal":
                # Extract diagonal elements of inverse covariance
                diag_list = [torch.diag(inv_cov) for inv_cov in self.inv_covs]
                self.diag_inv_covs_stacked = torch.stack(diag_list) # [K, hidden_dim]
                logger.info(f"Precomputed diagonal covariance vectors for {effective_clusters} clusters.")
            else:
                # Full covariance: stack into [K, hidden_dim, hidden_dim]
                self.inv_covs_stacked = torch.stack(self.inv_covs)
                logger.info(f"Precomputed vectorized full covariance tensors for {effective_clusters} clusters.")

            # Default global params to first cluster
            self.mean = self.means[0]
            self.inv_cov = self.inv_covs[0]
            logger.info(f"Fitted multi-cluster Reference Manifold with {effective_clusters} normal behavioral modes (type={self.covariance_type}).")
        else:
            self.means = [self.mean]
            self.inv_covs = []
            try:
                lw = LedoitWolf()
                lw.fit(states_np)
                self.inv_cov = torch.from_numpy(lw.precision_).to(dtype=states.dtype, device=states.device)
                logger.info(f"Reference Manifold Ledoit-Wolf fitting completed (shrinkage={lw.shrinkage_:.4f}).")
            except Exception as e:
                logger.warning(f"Ledoit-Wolf fitting failed ({e}), falling back to empirical regularized covariance.")
                cov = torch.cov(states.T)
                regularizer = self.eps * torch.eye(self.hidden_dim, device=cov.device)
                cov = cov + regularizer
                self.inv_cov = torch.linalg.inv(cov)
            self.inv_covs = [self.inv_cov]
            self.means_stacked = torch.stack(self.means)
            self.inv_covs_stacked = torch.stack(self.inv_covs)
            
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

        is_batched = states.dim() == 2
        if not is_batched:
            states = states.unsqueeze(0)  # Shape: [1, hidden_dim]

        if self.hidden_dim > 1000:
            sim = torch.cosine_similarity(states, self.mean.unsqueeze(0), dim=-1)
            dist = torch.clamp(1.0 - sim, min=0.0)
            if not is_batched:
                return dist.squeeze(0)
            return dist

        # Vectorized Multi-Cluster Fast Path
        if self.means_stacked is not None:
            means_st = self.means_stacked.to(device=states.device, dtype=states.dtype)
            
            if self.covariance_type == "shared" and self.shared_inv_cov is not None:
                # diff: [batch, K, hidden_dim]
                diff = states.unsqueeze(1) - means_st.unsqueeze(0)
                # Apply shared inverse covariance
                sh_inv = self.shared_inv_cov.to(device=states.device, dtype=states.dtype)
                diff_sh = diff @ sh_inv # [batch, K, hidden_dim]
                sq_dist = torch.clamp(torch.sum(diff_sh * diff, dim=-1), min=0.0) # [batch, K]
                dist = torch.sqrt(sq_dist).min(dim=-1).values # [batch]
                if not is_batched:
                    return dist.squeeze(0)
                return dist
            elif self.covariance_type == "diagonal" and self.diag_inv_covs_stacked is not None:
                # diff: [batch, K, hidden_dim]
                diff = states.unsqueeze(1) - means_st.unsqueeze(0)
                diag_st = self.diag_inv_covs_stacked.to(device=states.device, dtype=states.dtype)
                sq_dist = torch.clamp(torch.sum((diff ** 2) * diag_st.unsqueeze(0), dim=-1), min=0.0) # [batch, K]
                dist = torch.sqrt(sq_dist).min(dim=-1).values # [batch]
                if not is_batched:
                    return dist.squeeze(0)
                return dist
            elif self.inv_covs_stacked is not None:
                inv_st = self.inv_covs_stacked.to(device=states.device, dtype=states.dtype)
                if not is_batched or states.size(0) == 1:
                    # Single-flow fast path: avoid full batch broadcast
                    st_vec = states if states.dim() == 2 else states.unsqueeze(0)
                    diff = st_vec - means_st # [K, H] via broadcasting
                    diff_inv = torch.bmm(diff.unsqueeze(1), inv_st).squeeze(1) # [K, H]
                    sq_dist = torch.clamp(torch.sum(diff_inv * diff, dim=-1), min=0.0)
                    dist = torch.sqrt(sq_dist.min())
                    return dist
                else:
                    # Multi-flow batch path
                    diff = states.unsqueeze(1) - means_st.unsqueeze(0) # [B, K, H]
                    diff_inv = torch.einsum('bkh,khe->bke', diff, inv_st)
                    sq_dist = torch.clamp(torch.sum(diff_inv * diff, dim=-1), min=0.0) # [B, K]
                    return torch.sqrt(sq_dist).min(dim=-1).values

        # Single cluster fallback
        mean = self.mean.to(device=states.device, dtype=states.dtype)
        inv_cov = self.inv_cov.to(device=states.device, dtype=states.dtype)
        diff = states - mean  # Shape: [batch_size, hidden_dim]
        temp = diff @ inv_cov
        squared_dist = torch.clamp(torch.sum(temp * diff, dim=-1), min=0.0)
        dist = torch.sqrt(squared_dist)

        if not is_batched:
            return dist.squeeze(0)
        return dist
