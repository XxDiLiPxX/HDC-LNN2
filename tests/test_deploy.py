import pytest
import torch
import time
from hdlnn.contracts.schemas import CanonicalFlow
from hdlnn.deploy.simd_backend import SIMDEncoder
from hdlnn.deploy.kafka_ingest import LogIngestRingBuffer
from hdlnn.deploy.sidecar import StreamingSidecar
from hdlnn.hdc.encoder import RecordEncoder
from hdlnn.divergence.scorer import DivergenceScorer
from hdlnn.lnn.model import LNNSequenceModel

class MockConfig:
    def __init__(self):
        self.categorical_columns = ["proto", "service"]
        self.numerical_columns = ["dur"]
        self.hdc = {"dimension": 16384, "numerical_levels": 10}
        self.lnn = {"hidden_dim": 16, "backbone": "cfc", "max_seq_len": 4}
        self.seed = 42

def test_simd_packing_unpacking():
    config = MockConfig()
    record_encoder = RecordEncoder(
        D=16384,
        categorical_columns=config.categorical_columns,
        numerical_columns=config.numerical_columns
    )
    flow = CanonicalFlow(
        entity_id="10.0.0.1", timestamp=100.0, dt=0.0,
        categorical_fields={"proto": "tcp", "service": "http"},
        numerical_fields={"dur": 0.05}, label=0, split="train"
    )
    record_encoder.fit([flow])
    
    encoder = SIMDEncoder(record_encoder)
    
    # Test random vector packing/unpacking parity
    rand_bipolar = torch.randint(0, 2, (16384,), dtype=torch.float32) * 2.0 - 1.0
    packed = encoder._pack_bipolar_to_uint8(rand_bipolar)
    assert packed.shape == (2048,)
    assert packed.dtype == torch.uint8
    
    unpacked = encoder._unpack_uint8_to_binary(packed)
    assert len(unpacked) == 16384
    
    # Parity check (binary 1 maps to bipolar -1.0, binary 0 maps to bipolar +1.0)
    reconstructed_bipolar = torch.ones(16384, dtype=torch.float32)
    reconstructed_bipolar[unpacked > 0.5] = -1.0
    
    assert torch.equal(rand_bipolar, reconstructed_bipolar)

def test_simd_encoder_encode():
    config = MockConfig()
    flow = CanonicalFlow(
        entity_id="10.0.0.1",
        timestamp=1000.0,
        dt=0.0,
        categorical_fields={"proto": "tcp", "service": "http"},
        numerical_fields={"dur": 0.05},
        label=0,
        split="train"
    )
    record_encoder = RecordEncoder(
        D=16384,
        categorical_columns=config.categorical_columns,
        numerical_columns=config.numerical_columns
    )
    record_encoder.fit([flow])
    
    encoder = SIMDEncoder(record_encoder)
    encoded = encoder.encode(flow)
    
    assert encoded.vector.shape == (16384,)
    assert encoded.is_oov is False

def test_kafka_ingest_ring_buffer():
    # Test capacity drop behavior
    buffer = LogIngestRingBuffer(max_capacity=5)
    
    for i in range(10):
        flow = CanonicalFlow(
            entity_id=f"10.0.0.{i}",
            timestamp=100.0 + i,
            dt=1.0,
            categorical_fields={},
            numerical_fields={},
            label=0,
            split="test"
        )
        buffer.put(flow)
        
    assert buffer.size() == 5
    assert buffer.get_dropped_count() == 5
    
    # Verify we get the last 5 elements (ring buffer dropped first 5)
    first_get = buffer.get()
    assert first_get.entity_id == "10.0.0.5"

def test_streaming_sidecar():
    config = MockConfig()
    
    flow1 = CanonicalFlow(
        entity_id="10.0.0.1", timestamp=100.0, dt=0.0,
        categorical_fields={"proto": "tcp", "service": "http"},
        numerical_fields={"dur": 0.05}, label=0, split="train"
    )
    flow2 = CanonicalFlow(
        entity_id="10.0.0.1", timestamp=101.0, dt=1.0,
        categorical_fields={"proto": "tcp", "service": "http"},
        numerical_fields={"dur": 0.05}, label=0, split="train"
    )
    
    record_encoder = RecordEncoder(
        D=16384,
        categorical_columns=config.categorical_columns,
        numerical_columns=config.numerical_columns
    )
    record_encoder.fit([flow1, flow2])
    
    simd_encoder = SIMDEncoder(record_encoder)
    model = LNNSequenceModel(input_dim=16384, hidden_dim=16, proj_dim=16384)
    
    scorer = DivergenceScorer(mode="mahalanobis", hidden_dim=16, model=model)
    # Fit scorer with simple dummy data
    normal_val_states = torch.randn(10, 16)
    scorer.manifold.fit(normal_val_states)
    scorer.mahalanobis_threshold = 5.0
    
    sidecar = StreamingSidecar(config, model, scorer, simd_encoder)
    
    # Feed flows
    sidecar.feed_flows_async([flow1, flow2])
    
    # Process
    sidecar.start_processing(limit=2)
    
    assert sidecar.processed_count == 2
    assert len(sidecar.latencies) == 2
    assert "10.0.0.1" in sidecar.entity_states
