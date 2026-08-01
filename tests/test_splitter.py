import pytest
from hdlnn.contracts.schemas import CanonicalFlow
from hdlnn.data.splitter import split_dataset
from hdlnn.data.quality import verify_split_quality

def test_split_dataset_entity_disjoint_and_dt():
    # Construct mock flows
    # Entity 1: 3 flows, unordered timestamps
    # Entity 2: 2 flows
    # Entity 3: 1 flow
    mock_flows = [
        CanonicalFlow("E1", 10.0, 0.0, {}, {}, 0, ""),
        CanonicalFlow("E1", 5.0, 0.0, {}, {}, 0, ""),
        CanonicalFlow("E1", 15.0, 0.0, {}, {}, 0, ""),
        CanonicalFlow("E2", 20.0, 0.0, {}, {}, 1, ""),
        CanonicalFlow("E2", 22.0, 0.0, {}, {}, 1, ""),
        CanonicalFlow("E3", 30.0, 0.0, {}, {}, 0, ""),
    ]

    # Split: we have 3 unique entities.
    # Set ratios so train gets 1 entity, val gets 1 entity, test gets 1 entity (1/3 each)
    train, val, test = split_dataset(
        mock_flows,
        train_ratio=0.34,
        val_ratio=0.33,
        test_ratio=0.33,
        seed=42
    )

    # 1. Verify entity-disjoint property
    train_entities = {f.entity_id for f in train}
    val_entities = {f.entity_id for f in val}
    test_entities = {f.entity_id for f in test}

    assert len(train_entities) == 1
    assert len(val_entities) == 1
    assert len(test_entities) == 1

    # Check intersections are empty
    assert not train_entities.intersection(val_entities)
    assert not train_entities.intersection(test_entities)
    assert not val_entities.intersection(test_entities)

    # 2. Check sorting and dt calculations
    # Find E1 split to inspect E1 flows (should be chronological: 5.0, 10.0, 15.0)
    e1_flows = []
    for split_list in [train, val, test]:
        e1_flows.extend([f for f in split_list if f.entity_id == "E1"])
    
    e1_flows.sort(key=lambda f: f.timestamp)  # just to be sure we access them in order
    assert len(e1_flows) == 3
    assert e1_flows[0].timestamp == 5.0
    assert e1_flows[0].dt == 0.0
    assert e1_flows[1].timestamp == 10.0
    assert e1_flows[1].dt == 5.0
    assert e1_flows[2].timestamp == 15.0
    assert e1_flows[2].dt == 5.0

    # 3. Check E2 (should be 20.0, 22.0)
    e2_flows = []
    for split_list in [train, val, test]:
        e2_flows.extend([f for f in split_list if f.entity_id == "E2"])
    assert len(e2_flows) == 2
    assert e2_flows[0].timestamp == 20.0
    assert e2_flows[0].dt == 0.0
    assert e2_flows[1].timestamp == 22.0
    assert e2_flows[1].dt == 2.0

    # 4. Check quality check function passes
    assert verify_split_quality(train, val, test) is True
