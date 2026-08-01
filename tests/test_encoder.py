import pytest
import torch
import torchhd
from hdlnn.contracts.schemas import CanonicalFlow
from hdlnn.hdc.encoder import RecordEncoder

def test_record_encoder_bipolar_output_and_oov_detection():
    # 1. Initialize RecordEncoder with a subset of features
    categorical_cols = ["proto", "state"]
    numerical_cols = ["dur", "sbytes"]
    D = 10000
    
    encoder = RecordEncoder(
        D=D, 
        categorical_columns=categorical_cols, 
        numerical_columns=numerical_cols
    )

    # 2. Create mock training flows
    train_flows = [
        CanonicalFlow("E1", 10.0, 0.0, {"proto": "tcp", "state": "CON"}, {"dur": 0.1, "sbytes": 100.0}, 0, "train"),
        CanonicalFlow("E2", 20.0, 0.0, {"proto": "udp", "state": "CON"}, {"dur": 0.5, "sbytes": 500.0}, 1, "train"),
    ]

    # Fit encoder
    encoder.fit(train_flows)

    # Verify min/max values are fit correctly
    assert encoder.codebooks.min_max["dur"] == (0.1, 0.5)
    assert encoder.codebooks.min_max["sbytes"] == (100.0, 500.0)

    # Verify item memory is locked after fit
    assert encoder.item_memory.locked is True

    # 3. Encode a training flow record
    flow_train = train_flows[0]
    encoded_train = encoder.encode(flow_train)
    
    assert encoded_train.is_oov is False
    assert encoded_train.vector.shape == (D,)
    
    # Check that vector contains only bipolar elements {-1.0, 1.0}
    unique_vals = set(encoded_train.vector.tolist())
    assert unique_vals.issubset({-1.0, 1.0})

    # 4. Encode a flow record with an unseen categorical token (should trigger OOV)
    flow_test_oov = CanonicalFlow(
        entity_id="E3",
        timestamp=30.0,
        dt=0.0,
        categorical_fields={"proto": "icmp", "state": "CON"},  # 'icmp' is OOV
        numerical_fields={"dur": 0.3, "sbytes": 300.0},
        label=0,
        split="test"
    )
    encoded_test_oov = encoder.encode(flow_test_oov)
    
    assert encoded_test_oov.is_oov is True
    assert encoded_test_oov.vector.shape == (D,)
    assert set(encoded_test_oov.vector.tolist()).issubset({-1.0, 1.0})

    # 5. Verify that numerical features scale appropriately
    # If we encode 0.1, 0.3, and 0.5 for dur, the hypervector similarity (cosine similarity) 
    # of 0.3 should be similar to both 0.1 and 0.5, and 0.1 and 0.5 should be less similar.
    # Note that numerical_vector uses Level vectors which scale smoothly.
    v_0_1 = encoder.codebooks.get_numerical_vector("dur", 0.1)
    v_0_3 = encoder.codebooks.get_numerical_vector("dur", 0.3)
    v_0_5 = encoder.codebooks.get_numerical_vector("dur", 0.5)

    sim_1_3 = torchhd.cosine_similarity(v_0_1, v_0_3)
    sim_3_5 = torchhd.cosine_similarity(v_0_3, v_0_5)
    sim_1_5 = torchhd.cosine_similarity(v_0_1, v_0_5)

    # 0.1 to 0.3 distance is closer than 0.1 to 0.5 distance, so similarity should be higher
    assert sim_1_3 > sim_1_5
    assert sim_3_5 > sim_1_5
