import pytest
import torch
from hdlnn.divergence.manifold import ReferenceManifold

def test_reference_manifold_mahalanobis_calculation():
    hidden_dim = 3
    eps = 1e-4
    manifold = ReferenceManifold(hidden_dim=hidden_dim, eps=eps)

    # Create dummy validation states (normal distribution)
    # Shape: [N, hidden_dim]
    torch.manual_seed(42)
    normal_states = torch.randn(100, hidden_dim) + torch.tensor([1.0, 2.0, 3.0])

    # Fit manifold
    manifold.fit(normal_states)

    assert manifold.is_fit is True
    assert manifold.mean.shape == (hidden_dim,)
    assert manifold.inv_cov.shape == (hidden_dim, hidden_dim)

    # Test distance calculation for a point exactly at the mean (should be ~0.0)
    dist_at_mean = manifold.compute_mahalanobis_distance(manifold.mean)
    assert float(dist_at_mean.item()) == pytest.approx(0.0, abs=1e-5)

    # Test distance calculation for a batch of points
    test_points = torch.tensor([
        [1.0, 2.0, 3.0],  # Close to mean
        [10.0, 20.0, 30.0],  # Far from mean
    ])
    distances = manifold.compute_mahalanobis_distance(test_points)
    assert distances.shape == (2,)
    assert distances[0] < distances[1]

def test_reference_manifold_covariance_regularization():
    hidden_dim = 2
    manifold = ReferenceManifold(hidden_dim=hidden_dim, eps=1e-3)

    # Create collinear states (perfectly correlated, singular covariance)
    # x2 = 2 * x1
    x1 = torch.linspace(-5, 5, 20).unsqueeze(1)
    states = torch.cat([x1, x1 * 2], dim=1)  # [20, 2]

    # Without regularization, linalg.inv would throw RuntimeError or yield NaNs.
    # We verify that with eps regularization, the fit is stable and invert succeeds.
    manifold.fit(states)
    assert manifold.is_fit is True
    assert not torch.isnan(manifold.inv_cov).any()
    assert not torch.isinf(manifold.inv_cov).any()
