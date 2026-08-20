import torch
from hdlnn.contracts.schemas import CanonicalFlow
from hdlnn.contracts.interfaces import IEncoder, ISequenceModel
from hdlnn.hdc.alternative_encoders import SAXEncoder, RFFEncoder
from hdlnn.baselines.mamba import MambaSequenceModel


def test_encoders():
    print("=== Testing SAX and RFF Encoders ===")
    categorical_cols = ["proto", "service", "state"]
    numerical_cols = ["dur", "spkts", "dpkts", "sbytes", "dbytes"]
    
    # Create synthetic flows
    train_flows = [
        CanonicalFlow(
            entity_id="10.0.0.1",
            timestamp=float(i),
            dt=1.0,
            categorical_fields={"proto": "tcp", "service": "http", "state": "CON"},
            numerical_fields={"dur": 0.5 * i, "spkts": 10.0 + i, "dpkts": 8.0 + i, "sbytes": 500.0 * i, "dbytes": 1000.0 * i},
            label=0,
            split="train"
        )
        for i in range(1, 20)
    ]
    
    test_flow = CanonicalFlow(
        entity_id="10.0.0.1",
        timestamp=20.0,
        dt=1.0,
        categorical_fields={"proto": "tcp", "service": "http", "state": "CON"},
        numerical_fields={"dur": 10.0, "spkts": 30.0, "dpkts": 28.0, "sbytes": 10000.0, "dbytes": 20000.0},
        label=0,
        split="test"
    )
    
    # 1. Test SAX Encoder
    print("\n1. Testing SAXEncoder...")
    sax_enc = SAXEncoder(D=1024, alphabet_size=8, categorical_columns=categorical_cols, numerical_columns=numerical_cols)
    assert isinstance(sax_enc, IEncoder), "SAXEncoder must conform to IEncoder"
    sax_enc.fit(train_flows)
    
    hv_single = sax_enc.encode(test_flow)
    assert hv_single.vector.shape == (1024,), f"Expected shape (1024,), got {hv_single.vector.shape}"
    assert set(torch.unique(hv_single.vector).tolist()).issubset({-1.0, 1.0}), "Expected bipolar values"
    
    hv_batch = sax_enc.encode_batch(train_flows[:5])
    assert len(hv_batch) == 5, f"Expected batch size 5, got {len(hv_batch)}"
    assert hv_batch[0].vector.shape == (1024,)
    print("  -> SAXEncoder instantiation, fit, encode, and encode_batch: PASSED")
    
    # 2. Test RFF Encoder
    print("\n2. Testing RFFEncoder...")
    rff_enc = RFFEncoder(D=1024, gamma=0.5, categorical_columns=categorical_cols, numerical_columns=numerical_cols)
    assert isinstance(rff_enc, IEncoder), "RFFEncoder must conform to IEncoder"
    rff_enc.fit(train_flows)
    
    hv_rff_single = rff_enc.encode(test_flow)
    assert hv_rff_single.vector.shape == (1024,), f"Expected shape (1024,), got {hv_rff_single.vector.shape}"
    assert set(torch.unique(hv_rff_single.vector).tolist()).issubset({-1.0, 1.0}), "Expected bipolar values"
    
    hv_rff_batch = rff_enc.encode_batch(train_flows[:5])
    assert len(hv_rff_batch) == 5, f"Expected batch size 5, got {len(hv_rff_batch)}"
    assert hv_rff_batch[0].vector.shape == (1024,)
    print("  -> RFFEncoder instantiation, fit, encode, and encode_batch: PASSED")
    
    # 3. Test Mamba-2 SSM Model
    print("\n3. Testing MambaSequenceModel (Pure PyTorch SSM)...")
    mamba_model = MambaSequenceModel(input_dim=1024, hidden_dim=64, proj_dim=1024)
    assert isinstance(mamba_model, ISequenceModel)
    x_dummy = torch.randn(2, 8, 1024)
    out, h_last = mamba_model.forward(x_dummy)
    assert h_last.shape == (2, 2 * 64 * 16)
    print("  -> MambaSequenceModel forward pass: PASSED")
    
    print("\n=== ALL ENCODER AND MODEL TESTS COMPLETED ===")


if __name__ == "__main__":
    test_encoders()
