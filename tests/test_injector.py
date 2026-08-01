import pytest
from hdlnn.contracts.schemas import CanonicalFlow
from hdlnn.eval.injector import inject_threat_scenarios

def test_threat_injector_correctness():
    # Setup background normal flows
    mock_flows = [
        CanonicalFlow("E1", 100.0, 0.0, {"proto": "tcp"}, {"dur": 0.1}, 0, "train"),
        CanonicalFlow("E2", 200.0, 0.0, {"proto": "udp"}, {"dur": 0.2}, 0, "train"),
        CanonicalFlow("E3", 300.0, 0.0, {"proto": "tcp"}, {"dur": 0.3}, 0, "train"),
    ]

    # Inject threat scenarios
    injected_flows = inject_threat_scenarios(
        mock_flows,
        inject_dns=True,
        inject_pass_hash=True,
        seed=42
    )

    # 1. Total count assertion
    # Original: 3 flows
    # DNS Scenario A: 3 flows
    # Pass-the-Hash Scenario B: 3 flows
    # Total should be 3 + 3 + 3 = 9 flows
    assert len(injected_flows) == 9

    # 2. Chronological ordering verification
    timestamps = [f.timestamp for f in injected_flows]
    assert sorted(timestamps) == timestamps

    # 3. Anomaly labelling checks
    attack_flows = [f for f in injected_flows if f.label == 1]
    assert len(attack_flows) == 6
    
    # Assert entity IDs match simulated designs
    dns_entity_flows = [f for f in attack_flows if f.entity_id == "10.0.0.901"]
    pass_hash_entity_flows = [f for f in attack_flows if f.entity_id == "10.0.0.902"]
    
    assert len(dns_entity_flows) == 3
    assert len(pass_hash_entity_flows) == 3

    # Check timestamps gaps for DNS Scenario A (t0, t0 + 450, t0 + 1800)
    # Background start is 100.0
    # Injected t0 is 100.0 + 10.0 = 110.0
    # t1 is 110.0 + 450.0 = 560.0
    # t2 is 560.0 + 1350.0 = 1910.0
    dns_times = [f.timestamp for f in dns_entity_flows]
    assert dns_times == [110.0, 560.0, 1910.0]

    # Check Pass-the-Hash Scenario B (tb0, tb0 + 10.0, tb1 + 5.0)
    # tb0 is 100.0 + 20.0 = 120.0
    # tb1 is 130.0
    # tb2 is 135.0
    ph_times = [f.timestamp for f in pass_hash_entity_flows]
    assert ph_times == [120.0, 130.0, 135.0]
