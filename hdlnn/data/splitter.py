import random
import logging
from typing import List, Dict, Tuple
from hdlnn.contracts.schemas import CanonicalFlow

logger = logging.getLogger(__name__)

def split_dataset(
    flows: List[CanonicalFlow],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[CanonicalFlow], List[CanonicalFlow], List[CanonicalFlow]]:
    """Partitions flows into train, val, and test splits using an entity-disjoint policy.
    
    1. Groups records by entity_id.
    2. Sorts each entity's records chronologically and computes dt.
    3. Allocates entities to splits based on ratio and seed.
    """
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-9:
        raise ValueError("Train, val, and test ratios must sum to 1.0")

    # Group flows by entity_id
    entity_to_flows: Dict[str, List[CanonicalFlow]] = {}
    for flow in flows:
        entity_to_flows.setdefault(flow.entity_id, []).append(flow)

    # Sort flows chronologically for each entity and calculate dt
    for entity_id, entity_flows in entity_to_flows.items():
        # Sort by timestamp
        entity_flows.sort(key=lambda f: f.timestamp)
        
        # Calculate dt
        prev_time = None
        for flow in entity_flows:
            if prev_time is None:
                flow.dt = 0.0
            else:
                flow.dt = max(flow.timestamp - prev_time, 0.0)
            prev_time = flow.timestamp

    # Shuffle entity IDs deterministically to allocate splits
    unique_entities = sorted(list(entity_to_flows.keys()))
    
    # Threat validation entities are strictly mapped to test split (never leak into training)
    threat_entities = {"10.0.0.901", "10.0.0.902"}
    random_entities = [e for e in unique_entities if e not in threat_entities]
    
    rng = random.Random(seed)
    rng.shuffle(random_entities)

    num_entities = len(random_entities)
    
    if num_entities == 0:
        # Fallback if only threat entities are present
        train_entities = set()
        val_entities = set()
        test_entities = set(unique_entities)
    elif num_entities == 1:
        train_entities = set(random_entities)
        val_entities = set()
        test_entities = set(threat_entities).intersection(unique_entities)
    elif num_entities == 2:
        train_entities = {random_entities[0]}
        val_entities = {random_entities[1]}
        test_entities = set(threat_entities).intersection(unique_entities)
    else:
        train_cut = int(train_ratio * num_entities)
        val_cut = train_cut + int(val_ratio * num_entities)
        
        # Ensure val gets at least 1 entity
        if val_cut == train_cut:
            val_cut = train_cut + 1
            
        # Ensure test gets at least 1 entity
        if val_cut >= num_entities:
            val_cut = num_entities - 1
            # Adjust train_cut if necessary to keep val size >= 1
            if train_cut >= val_cut:
                train_cut = val_cut - 1
                
        train_entities = set(random_entities[:train_cut])
        val_entities = set(random_entities[train_cut:val_cut])
        test_entities = set(random_entities[val_cut:])
        
        # Merge threat entities into test set
        for te in threat_entities:
            if te in entity_to_flows:
                test_entities.add(te)


    train_flows: List[CanonicalFlow] = []
    val_flows: List[CanonicalFlow] = []
    test_flows: List[CanonicalFlow] = []

    for entity_id, entity_flows in entity_to_flows.items():
        if entity_id in train_entities:
            split_name = "train"
            split_list = train_flows
        elif entity_id in val_entities:
            split_name = "val"
            split_list = val_flows
        else:
            split_name = "test"
            split_list = test_flows

        for flow in entity_flows:
            flow.split = split_name
            split_list.append(flow)

    # Re-sort each final split list chronologically by timestamp (to maintain global time ordering for training/eval)
    train_flows.sort(key=lambda f: f.timestamp)
    val_flows.sort(key=lambda f: f.timestamp)
    test_flows.sort(key=lambda f: f.timestamp)

    logger.info(
        f"Split summary: "
        f"Train={len(train_flows)} flows ({len(train_entities)} entities), "
        f"Val={len(val_flows)} flows ({len(val_entities)} entities), "
        f"Test={len(test_flows)} flows ({len(test_entities)} entities)."
    )

    return train_flows, val_flows, test_flows
