import pytest
import torch
from hdlnn.baselines.ft_transformer import FTTransformerSequenceModel
from hdlnn.baselines.saint import SAINTSequenceModel
from hdlnn.common.model_utils import count_parameters


def test_ft_transformer_instantiation_and_parameter_count():
    model = FTTransformerSequenceModel(
        input_dim=10000,
        hidden_dim=64,
        n_layers=3,
        n_heads=4,
        proj_dim=10000
    )
    counts = count_parameters(model)
    assert counts["total_params"] < 1_000_000_000, "Parameter count must be strictly < 1B"
    # Compact configuration should be ~1.46M
    assert counts["total_params"] < 5_000_000, "Should be compact (<5M params)"
    assert counts["trainable_params"] == counts["total_params"]
    assert counts["params_m"] > 0.0


def test_saint_instantiation_and_parameter_count():
    model = SAINTSequenceModel(
        input_dim=10000,
        hidden_dim=64,
        n_layers=2,
        n_heads=4,
        proj_dim=10000
    )
    counts = count_parameters(model)
    assert counts["total_params"] < 1_000_000_000, "Parameter count must be strictly < 1B"
    # Compact configuration should be ~1.49M
    assert counts["total_params"] < 5_000_000, "Should be compact (<5M params)"
    assert counts["trainable_params"] == counts["total_params"]
    assert counts["params_m"] > 0.0


def test_ft_transformer_forward_and_step():
    model = FTTransformerSequenceModel(
        input_dim=10000,
        hidden_dim=64,
        n_layers=2,
        n_heads=4,
        proj_dim=10000
    )
    # Sequence forward
    x = torch.randn(2, 8, 10000)
    seq_out, h_last = model.forward(x)
    assert seq_out.shape == (2, 8, 64)
    assert h_last.shape == (2, 64)

    # Streaming single step
    x_single = torch.randn(2, 10000)
    dt = torch.zeros(2, 1)
    h_next = model.step(x_single, h_last, dt)
    assert h_next.shape == (2, 64)

    # Next vector prediction
    pred = model.predict_next_vector(h_next)
    assert pred.shape == (2, 10000)


def test_saint_forward_and_step():
    model = SAINTSequenceModel(
        input_dim=10000,
        hidden_dim=64,
        n_layers=2,
        n_heads=4,
        proj_dim=10000
    )
    # Sequence forward
    x = torch.randn(2, 8, 10000)
    seq_out, h_last = model.forward(x)
    assert seq_out.shape == (2, 8, 64)
    assert h_last.shape == (2, 64)

    # Streaming single step
    x_single = torch.randn(2, 10000)
    dt = torch.zeros(2, 1)
    h_next = model.step(x_single, h_last, dt)
    assert h_next.shape == (2, 64)

    # Next vector prediction
    pred = model.predict_next_vector(h_next)
    assert pred.shape == (2, 10000)
