from typing import List
from abc import ABC, abstractmethod
import torch
from .schemas import CanonicalFlow, EncodedHypervector, TrajectoryState, AnomalyDecision

class IEncoder(ABC):
    @abstractmethod
    def encode(self, flow: CanonicalFlow) -> EncodedHypervector:
        """Transforms a CanonicalFlow into a fixed $D$-dimensional hypervector[cite: 2]."""
        pass

    def encode_batch(self, flows: List[CanonicalFlow]) -> List[EncodedHypervector]:
        """Encodes a batch of CanonicalFlow events into hypervectors."""
        return [self.encode(f) for f in flows]

class ISequenceModel(ABC):
    @abstractmethod
    def step(self, x: torch.Tensor, h_prev: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """Evaluates continuous state transition over irregular dt[cite: 1, 2]."""
        pass

class IDivergenceScorer(ABC):
    @abstractmethod
    def score(self, state: TrajectoryState) -> AnomalyDecision:
        """Computes behavioral drift against the reference manifold[cite: 2, 3]."""
        pass