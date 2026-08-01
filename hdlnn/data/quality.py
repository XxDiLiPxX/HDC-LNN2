import logging
from typing import List
from hdlnn.contracts.schemas import CanonicalFlow

logger = logging.getLogger(__name__)

def verify_split_quality(
    train_flows: List[CanonicalFlow],
    val_flows: List[CanonicalFlow],
    test_flows: List[CanonicalFlow]
) -> bool:
    """Performs rigorous quality checks on dataset splits to ensure data integrity and prevent leakage."""
    success = True

    # 1. Entity overlap check (Leakage prevention)
    train_entities = {f.entity_id for f in train_flows}
    val_entities = {f.entity_id for f in val_flows}
    test_entities = {f.entity_id for f in test_flows}

    overlap_train_val = train_entities.intersection(val_entities)
    overlap_train_test = train_entities.intersection(test_entities)
    overlap_val_test = val_entities.intersection(test_entities)

    if overlap_train_val:
        logger.error(f"Leakage detected: {len(overlap_train_val)} entities overlap between Train and Val splits.")
        success = False
    if overlap_train_test:
        logger.error(f"Leakage detected: {len(overlap_train_test)} entities overlap between Train and Test splits.")
        success = False
    if overlap_val_test:
        logger.error(f"Leakage detected: {len(overlap_val_test)} entities overlap between Val and Test splits.")
        success = False

    # 2. General constraints checks
    all_splits = [("Train", train_flows), ("Val", val_flows), ("Test", test_flows)]
    for split_name, flows in all_splits:
        for idx, flow in enumerate(flows):
            # Check split match
            if flow.split != split_name.lower():
                logger.error(f"[{split_name}] Flow at index {idx} has incorrect split tag: {flow.split}")
                success = False

            # Check binary label
            if flow.label not in (0, 1):
                logger.error(f"[{split_name}] Flow at index {idx} has invalid label: {flow.label}")
                success = False

            # Check delta time dt is valid
            if flow.dt < 0.0:
                logger.error(f"[{split_name}] Flow at index {idx} has negative dt: {flow.dt}")
                success = False

            # Check numeric values are valid (no NaN/Inf)
            for num_key, num_val in flow.numerical_fields.items():
                if num_val != num_val:  # NaN check
                    logger.error(f"[{split_name}] Flow at index {idx} numerical field '{num_key}' is NaN.")
                    success = False
                # check for inf
                if num_val == float('inf') or num_val == float('-inf'):
                    logger.error(f"[{split_name}] Flow at index {idx} numerical field '{num_key}' is Infinite.")
                    success = False

    if success:
        logger.info("All split quality and leakage checks passed successfully.")
    else:
        logger.error("Quality checks failed. Review error logs.")
    
    return success

def audit_pipeline_leakage(
    train_flows: List[CanonicalFlow],
    val_flows: List[CanonicalFlow],
    test_flows: List[CanonicalFlow],
    categorical_cols: List[str],
    numerical_cols: List[str]
) -> None:
    """Rigorous cybersecurity audit to verify zero feature leakage of labels or post-attack artifacts."""
    # 1. Verify target columns are not in input feature lists
    target_cols = {"label", "attack_cat", "id", "response_status", "explicit_alert"}
    for col in categorical_cols:
        if col in target_cols:
            raise ValueError(f"CRITICAL LEAKAGE: Target column '{col}' is present in categorical_columns!")
    for col in numerical_cols:
        if col in target_cols:
            raise ValueError(f"CRITICAL LEAKAGE: Target column '{col}' is present in numerical_columns!")
            
    # 2. Verify splits are entity-disjoint
    train_entities = {f.entity_id for f in train_flows}
    val_entities = {f.entity_id for f in val_flows}
    test_entities = {f.entity_id for f in test_flows}
    
    overlap_train_val = train_entities.intersection(val_entities)
    overlap_train_test = train_entities.intersection(test_entities)
    overlap_val_test = val_entities.intersection(test_entities)
    
    if overlap_train_val:
        raise ValueError(f"CRITICAL LEAKAGE: Entity overlap between Train and Val: {overlap_train_val}")
    if overlap_train_test:
        raise ValueError(f"CRITICAL LEAKAGE: Entity overlap between Train and Test: {overlap_train_test}")
    if overlap_val_test:
        raise ValueError(f"CRITICAL LEAKAGE: Entity overlap between Val and Test: {overlap_val_test}")
        
    logger.info("Cybersecurity leakage audit passed: strict entity disjointness and target label isolation verified.")
