import torch
from hdlnn.baselines.autoencoder import AETemporalModel
from hdlnn.divergence.ae_scorer import AEReconstructionScorer
from hdlnn.contracts.schemas import TrajectoryState

def test_ae():
    print("Testing AE Baseline and Scorer...")
    input_dim = 64
    hidden_dim = 32
    batch_size = 4
    seq_len = 5
    
    # Instantiate model and scorer
    model = AETemporalModel(input_dim=input_dim, hidden_dim=hidden_dim)
    scorer = AEReconstructionScorer(model=model, threshold_k=2.0)
    
    # 1. Test batch forward pass (simulate training)
    print("\n--- Testing Batch Forward ---")
    x_batch = torch.randn(batch_size, seq_len, input_dim)
    dec_out, h_n = model(x_batch)
    print(f"Input shape: {x_batch.shape}")
    print(f"Reconstruction shape: {dec_out.shape}")
    print(f"Hidden state shape: {h_n.shape}")
    assert dec_out.shape == x_batch.shape, "Decoder output shape mismatch"
    assert h_n.shape == (batch_size, hidden_dim), "Hidden state shape mismatch"
    
    # 2. Test threshold fitting (simulate normal val split)
    print("\n--- Testing Threshold Fitting ---")
    normal_inputs = torch.randn(100, seq_len, input_dim)
    scorer.fit_threshold(normal_inputs)
    print(f"Fitted Threshold: {scorer.reconstruction_threshold:.4f}")
    
    # 3. Test streaming step (simulate inference)
    print("\n--- Testing Streaming Step & Scoring ---")
    x_step = torch.randn(1, input_dim)
    dt = torch.tensor([1.0])
    h_prev = None
    
    entity_id = "test_entity"
    # Register the observed hypervector for the scorer
    scorer.register_observed(entity_id, x_step.squeeze(0))
    
    # Step the model
    h_next = model.step(x_step, h_prev, dt)
    print(f"Step hidden state shape: {h_next.shape}")
    assert h_next.shape == (1, 2 * hidden_dim), "Step hidden state shape mismatch"
    
    # Score the state
    state = TrajectoryState(entity_id=entity_id, timestamp=1.0, state=h_next.squeeze(0))
    decision = scorer.score(state)
    
    print(f"Anomaly Decision: {decision}")
    print("All tests passed!")

if __name__ == "__main__":
    test_ae()
