import torch
import logging
from typing import Dict, Any, Optional
from hdlnn.contracts.interfaces import IDivergenceScorer
from hdlnn.contracts.schemas import TrajectoryState, AnomalyDecision

logger = logging.getLogger(__name__)

class AEReconstructionScorer(IDivergenceScorer):
    """Computes anomaly score based on autoencoder reconstruction error.
    
    Fits a reconstruction error threshold on normal validation data, and scores incoming states 
    by computing MSE between the actual observed hypervector and the reconstructed hypervector.
    """
    
    def __init__(
        self,
        model: Any,
        threshold_k: float = 1.0
    ):
        self.model = model
        self.threshold_k = threshold_k
        self.reconstruction_threshold = 0.0
        
        # Stores the current observed hypervector: entity_id -> torch.Tensor
        self.last_observed: Dict[str, torch.Tensor] = {}

    def register_observed(self, entity_id: str, vector: torch.Tensor) -> None:
        """Registers the actual observed hypervector for a given entity."""
        self.last_observed[entity_id] = vector

    def fit_threshold(self, normal_inputs: torch.Tensor, normal_states: Optional[torch.Tensor] = None) -> None:
        """Fits reconstruction threshold using normal val split inputs and trajectory states.
        
        Args:
            normal_inputs: Shape [batch_size, seq_len, input_dim] or [N, input_dim]
            normal_states: Optional shape [N, state_dim]
        """
        self.model.eval()
        with torch.no_grad():
            if normal_states is not None:
                reconstructions = self.model.reconstruct(normal_states)
                inputs_flat = normal_inputs.reshape(-1, normal_inputs.size(-1))
                recons_flat = reconstructions.reshape(-1, reconstructions.size(-1))
                errors = torch.mean((recons_flat - inputs_flat) ** 2, dim=-1)
            else:
                if normal_inputs.dim() == 2:
                    inputs = normal_inputs.unsqueeze(1)
                else:
                    inputs = normal_inputs
                    
                reconstructions, _ = self.model(inputs)
                inputs_flat = inputs.reshape(-1, inputs.size(-1))
                recons_flat = reconstructions.reshape(-1, reconstructions.size(-1))
                errors = torch.mean((recons_flat - inputs_flat) ** 2, dim=-1)
            
        mean_err = float(errors.mean())
        std_err = float(errors.std())
        self.reconstruction_threshold = mean_err + self.threshold_k * std_err
        
        logger.info(
            f"AE Threshold fit: mean={mean_err:.4f}, std={std_err:.4f}, "
            f"k={self.threshold_k}, final_threshold={self.reconstruction_threshold:.4f}"
        )

    def score(self, state: TrajectoryState) -> AnomalyDecision:
        """Computes reconstruction error anomaly score."""
        entity_id = state.entity_id
        if entity_id not in self.last_observed:
            # Fallback if no observed hypervector is registered yet
            return AnomalyDecision(
                drift_score=0.0,
                threshold=self.reconstruction_threshold,
                is_anomaly=False,
                scoring_mode="ae_reconstruction"
            )
            
        observed_hv = self.last_observed[entity_id]
        
        self.model.eval()
        with torch.no_grad():
            # state.state is the bottleneck vector from the encoder
            reconstruction = self.model.reconstruct(state.state.unsqueeze(0)).squeeze(0)
            
        mse = float(torch.mean((reconstruction - observed_hv) ** 2).item())
        is_anomaly = mse > self.reconstruction_threshold
        
        return AnomalyDecision(
            drift_score=mse,
            threshold=self.reconstruction_threshold,
            is_anomaly=is_anomaly,
            scoring_mode="ae_reconstruction"
        )
