import random
import logging
from typing import List, Dict, Any, Optional
from hdlnn.contracts.schemas import CanonicalFlow

logger = logging.getLogger(__name__)

def inject_threat_scenarios(
    flows: List[CanonicalFlow],
    inject_dns: bool = True,
    inject_pass_hash: bool = True,
    seed: int = 42,
    categorical_cols: Optional[List[str]] = None,
    numerical_cols: Optional[List[str]] = None
) -> List[CanonicalFlow]:
    """Injects synthetic threat validation scenarios into a list of normal/clean flows.
    
    Ensures all injected flows contain all keys from categorical_cols and numerical_cols
    to prevent key errors during encoder fitting.
    """
    logger.info("Injecting threat validation scenarios...")
    rng = random.Random(seed)
    
    # Copy flows to avoid mutating original list
    modified_flows = [
        CanonicalFlow(
            entity_id=f.entity_id,
            timestamp=f.timestamp,
            dt=f.dt,
            categorical_fields=dict(f.categorical_fields),
            numerical_fields=dict(f.numerical_fields),
            label=f.label,
            split=f.split
        )
        for f in flows
    ]

    # Find global start time to anchor injections
    if not modified_flows:
        logger.warning("No flows available to inject into.")
        return modified_flows
        
    start_time = min(f.timestamp for f in modified_flows)

    # Base helper to create full fields dict with defaults
    def create_fields(
        specific_cats: Dict[str, str],
        specific_nums: Dict[str, float]
    ) -> Tuple[Dict[str, str], Dict[str, float]]:
        cats = {col: "-" for col in (categorical_cols or [])}
        cats.update(specific_cats)
        
        nums = {col: 0.0 for col in (numerical_cols or [])}
        nums.update(specific_nums)
        
        return cats, nums

    # We will inject these attacks into a dedicated target entity to easily isolate and evaluate them
    if inject_dns:
        # Scenario A: Low-and-Slow DNS Tunneling
        # Space them across 1800 seconds
        entity_a = "10.0.0.901"
        t0 = start_time + 10.0
        t1 = t0 + 450.0
        t2 = t1 + 1350.0 # t0 + 1800.0
        
        logger.info(f"Injecting Scenario A (Low-and-Slow DNS) for entity {entity_a}")
        
        c0, n0 = create_fields(
            {"proto": "udp", "service": "dns", "state": "INT", "is_ftp_login": "0", "is_sm_ips_ports": "0"},
            {"dur": 0.001, "spkts": 2.0, "dpkts": 2.0, "sbytes": 120.0, "dbytes": 240.0, "rate": 2000.0}
        )
        c1, n1 = create_fields(
            {"proto": "udp", "service": "dns", "state": "INT", "is_ftp_login": "0", "is_sm_ips_ports": "0"},
            {"dur": 0.001, "spkts": 2.0, "dpkts": 2.0, "sbytes": 130.0, "dbytes": 260.0, "rate": 2000.0}
        )
        c2, n2 = create_fields(
            {"proto": "udp", "service": "dns", "state": "INT", "is_ftp_login": "0", "is_sm_ips_ports": "0"},
            {"dur": 0.001, "spkts": 2.0, "dpkts": 2.0, "sbytes": 150.0, "dbytes": 300.0, "rate": 2000.0}
        )
        
        dns_flows = [
            CanonicalFlow(entity_id=entity_a, timestamp=t0, dt=0.0, categorical_fields=c0, numerical_fields=n0, label=1, split=""),
            CanonicalFlow(entity_id=entity_a, timestamp=t1, dt=0.0, categorical_fields=c1, numerical_fields=n1, label=1, split=""),
            CanonicalFlow(entity_id=entity_a, timestamp=t2, dt=0.0, categorical_fields=c2, numerical_fields=n2, label=1, split="")
        ]
        modified_flows.extend(dns_flows)

    if inject_pass_hash:
        # Scenario B: Lateral Movement via Pass-the-Hash
        # auth -> SMB Exec -> LSASS Dump (short intervals)
        entity_b = "10.0.0.902"
        tb0 = start_time + 20.0
        tb1 = tb0 + 10.0
        tb2 = tb1 + 5.0
        
        logger.info(f"Injecting Scenario B (Lateral Movement Sequence) for entity {entity_b}")
        
        cb0, nb0 = create_fields(
            {"proto": "tcp", "service": "auth", "state": "CON", "is_ftp_login": "0", "is_sm_ips_ports": "0"},
            {"dur": 0.5, "spkts": 4.0, "dpkts": 4.0, "sbytes": 450.0, "dbytes": 450.0, "rate": 16.0}
        )
        cb1, nb1 = create_fields(
            {"proto": "tcp", "service": "smb", "state": "CON", "is_ftp_login": "0", "is_sm_ips_ports": "0"},
            {"dur": 1.2, "spkts": 10.0, "dpkts": 8.0, "sbytes": 1200.0, "dbytes": 800.0, "rate": 15.0}
        )
        cb2, nb2 = create_fields(
            {"proto": "tcp", "service": "lsass_dump", "state": "FIN", "is_ftp_login": "0", "is_sm_ips_ports": "0"},
            {"dur": 3.4, "spkts": 50.0, "dpkts": 45.0, "sbytes": 54000.0, "dbytes": 2000.0, "rate": 28.0}
        )
        
        lateral_flows = [
            CanonicalFlow(entity_id=entity_b, timestamp=tb0, dt=0.0, categorical_fields=cb0, numerical_fields=nb0, label=1, split=""),
            CanonicalFlow(entity_id=entity_b, timestamp=tb1, dt=0.0, categorical_fields=cb1, numerical_fields=nb1, label=1, split=""),
            CanonicalFlow(entity_id=entity_b, timestamp=tb2, dt=0.0, categorical_fields=cb2, numerical_fields=nb2, label=1, split="")
        ]
        modified_flows.extend(lateral_flows)

    # Re-sort all flows chronologically by timestamp
    modified_flows.sort(key=lambda f: f.timestamp)
    logger.info(f"Injection completed. Total flows: {len(modified_flows)}")
    return modified_flows

from typing import Tuple
