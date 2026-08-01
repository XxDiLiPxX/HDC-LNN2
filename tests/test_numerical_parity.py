import pytest
import torch
from hdlnn.lnn.model import LNNSequenceModel

def test_batch_streaming_numerical_parity():
    # Setup hyperparameters
    input_dim = 128
    hidden_dim = 64
    batch_size = 4
    seq_len = 5
    
    # Set PyTorch seed to ensure reproducibility
    torch.manual_seed(42)
    
    # Initialize the model
    model = LNNSequenceModel(input_dim=input_dim, hidden_dim=hidden_dim)
    model.eval()  # Put in evaluation mode

    # Generate random input sequence x and time deltas dt
    x = torch.randn(batch_size, seq_len, input_dim)
    dt = torch.rand(batch_size, seq_len) + 0.1  # Ensure positive dt values

    # Initial state (zeros)
    h0 = torch.zeros(batch_size, hidden_dim)

    # 1. Evaluate batch forward pass
    with torch.no_grad():
        batch_outputs, batch_h_last = model.forward(x, h0=h0, dt=dt)

    # 2. Evaluate streaming step-by-step pass
    streaming_outputs = []
    h_prev = h0.clone()
    
    with torch.no_grad():
        for t in range(seq_len):
            x_t = x[:, t]  # Shape: [batch_size, input_dim]
            dt_t = dt[:, t]  # Shape: [batch_size]
            
            # Step evaluation
            h_next = model.step(x_t, h_prev, dt_t)
            streaming_outputs.append(h_next)
            h_prev = h_next

    # Stack streaming outputs into [batch_size, seq_len, hidden_dim]
    # In step() we get h_next of shape [batch_size, hidden_dim].
    # Stacking them along dim 1 gives [batch_size, seq_len, hidden_dim]
    streaming_outputs_tensor = torch.stack(streaming_outputs, dim=1)

    # 3. Assert absolute numerical parity
    # We compare the outputs at each step and the final hidden state
    assert torch.allclose(batch_outputs, streaming_outputs_tensor, atol=1e-6, rtol=1e-5), (
        "Batch outputs and streaming outputs do not match!"
    )
    
    assert torch.allclose(batch_h_last, h_prev, atol=1e-6, rtol=1e-5), (
        "Final batch hidden state and streaming hidden state do not match!"
    )
