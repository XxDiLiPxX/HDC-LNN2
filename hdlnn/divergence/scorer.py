import torch
import logging
from typing import Dict, Optional, Tuple, Any
from hdlnn.contracts.interfaces import IDivergenceScorer
from hdlnn.contracts.schemas import TrajectoryState, AnomalyDecision
from hdlnn.divergence.manifold import ReferenceManifold

logger = logging.getLogger(__name__)

class DivergenceScorer(IDivergenceScorer):
    """Computes behavioral drift anomalies using either Mahalanobis or Cosine EWMA modes."""
    
    def __init__(
        self,
        mode: str = "mahalanobis",
        threshold_k: float = 3.0,
        alpha: float = 0.05,
        hidden_dim: int = 64,
        warmup_steps: int = 0,
        model: Optional[Any] = None
    ):
        self.mode = mode.lower()
        self.threshold_k = threshold_k
        self.alpha = alpha
        self.hidden_dim = hidden_dim
        self.warmup_steps = warmup_steps
        self.model = model
        self.initial_threshold = 1.0
        
        # Mahalanobis components
        self.manifold = ReferenceManifold(hidden_dim=hidden_dim)
        self.mahalanobis_threshold = 0.0

        # Cosine EWMA tracking per entity: entity_id -> (mean, var, count)
        self.ewma_stats: Dict[str, Tuple[float, float, int]] = {}
        # Stores the current observed hypervector: entity_id -> torch.Tensor
        self.last_observed: Dict[str, torch.Tensor] = {}

    def register_observed(self, entity_id: str, vector: torch.Tensor) -> None:
        """Registers the actual observed hypervector for a given entity, required for Cosine mode."""
        self.last_observed[entity_id] = vector

    def fit_mahalanobis_threshold(self, normal_states: torch.Tensor) -> None:
        """Fits the ReferenceManifold and computes the static Mahalanobis threshold on normal validation data."""
        if self.mode != "mahalanobis":
            return
            
        # Fit the manifold
        self.manifold.fit(normal_states)
        
        # Calculate distances of the fitting normal states to set the threshold
        distances = self.manifold.compute_mahalanobis_distance(normal_states)
        mean_dist = float(distances.mean())
        std_dist = float(distances.std())
        
        # Static threshold: mean + k * std
        self.mahalanobis_threshold = mean_dist + self.threshold_k * std_dist
        logger.info(
            f"Mahalanobis threshold fit: mean={mean_dist:.4f}, std={std_dist:.4f}, "
            f"k={self.threshold_k}, final_threshold={self.mahalanobis_threshold:.4f}"
        )

    def score(self, state: TrajectoryState) -> AnomalyDecision:
        """Computes behavioral drift against reference manifold or LNN next-state predictions."""
        if self.mode == "mahalanobis":
            return self._score_mahalanobis(state)
        elif self.mode == "cosine":
            return self._score_cosine_ewma(state)
        else:
            raise ValueError(f"Unsupported scoring mode: {self.mode}")

    def _score_mahalanobis(self, state: TrajectoryState) -> AnomalyDecision:
        # Calculate Mahalanobis distance
        # state.state is of shape [hidden_dim]
        dist_tensor = self.manifold.compute_mahalanobis_distance(state.state)
        drift_score = float(dist_tensor.item())
        
        is_anomaly = drift_score > self.mahalanobis_threshold
        
        return AnomalyDecision(
            drift_score=drift_score,
            threshold=self.mahalanobis_threshold,
            is_anomaly=is_anomaly,
            scoring_mode="mahalanobis"
        )

    def _score_cosine_ewma(self, state: TrajectoryState) -> AnomalyDecision:
        if self.model is None:
            raise ValueError("LNN Sequence Model must be supplied to DivergenceScorer for Cosine mode.")
            
        entity_id = state.entity_id
        if entity_id not in self.last_observed:
            # Fallback if no observed hypervector is registered yet
            return AnomalyDecision(
                drift_score=0.0,
                threshold=1.0,
                is_anomaly=False,
                scoring_mode="cosine"
            )
            
        observed_hv = self.last_observed[entity_id]
        
        # 1. Project LNN state back to hypervector space
        self.model.eval()
        with torch.no_grad():
            predicted_hv = self.model.predict_next_vector(state.state.unsqueeze(0)).squeeze(0)
            
        # 2. Compute Cosine Distance: 1 - CosineSimilarity
        similarity = torch.cosine_similarity(observed_hv.unsqueeze(0), predicted_hv.unsqueeze(0))
        cosine_dist = float(1.0 - similarity.item())

        # 3. Update adaptive EWMA threshold
        stats = self.ewma_stats.get(entity_id, (0.0, 0.0, 0))
        mean_ewma, var_ewma, count = stats
        count += 1
        
        if count == 1:
            # Initialize statistics
            mean_ewma = cosine_dist
            var_ewma = 0.0
            threshold = self.initial_threshold
            is_anomaly = (cosine_dist > threshold) if (self.warmup_steps == 0 and threshold < 1.0) else False
        else:
            diff = cosine_dist - mean_ewma
            # Update mean EWMA
            mean_ewma = mean_ewma + self.alpha * diff
            # Update variance EWMA
            var_ewma = (1 - self.alpha) * (var_ewma + self.alpha * (diff ** 2))
            
            std_ewma = (var_ewma ** 0.5)
            threshold = mean_ewma + self.threshold_k * std_ewma
            
            # Anomaly verdict (suppressed during warmup period)
            if count <= self.warmup_steps:
                is_anomaly = False
            else:
                is_anomaly = cosine_dist > threshold

        # Save stats
        self.ewma_stats[entity_id] = (mean_ewma, var_ewma, count)
        
        return AnomalyDecision(
            drift_score=cosine_dist,
            threshold=threshold,
            is_anomaly=is_anomaly,
            scoring_mode="cosine"
        )
