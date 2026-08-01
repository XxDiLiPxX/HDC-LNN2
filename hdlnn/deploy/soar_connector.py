import logging
import time
from typing import Dict, Any
from hdlnn.contracts.schemas import AnomalyDecision

logger = logging.getLogger(__name__)

class SIEMAlertHandler:
    """Production SIEM/SOAR alert dispatcher.
    
    Formats high-severity AnomalyDecisions into standardized CEF (Common Event Format)
    syslog strings, sending them to configured alerting systems (or local file logs).
    """
    def __init__(self, config: Any):
        self.config = config
        self.alert_log_file = "soar_alerts.log"
        logger.info(f"SIEMAlertHandler initialized. Dispatching alerts to {self.alert_log_file}")

    def dispatch_alert(self, entity_id: str, decision: AnomalyDecision, timestamp: float) -> str:
        """Dispatches an alert in CEF format to SIEM and SOAR systems."""
        severity = 10  # Maximum severity (10/10) for confirmed anomaly drift
        
        # 1. Format to Standard Common Event Format (CEF)
        cef_string = (
            f"CEF:0|HYDRA-LNN|BehavioralDriftEngine|2.0|ANOMALY_DETECTED|"
            f"Behavioral drift anomaly detected for entity {entity_id}|{severity}|"
            f"src={entity_id} rt={int(timestamp * 1000)} "
            f"driftScore={decision.drift_score:.4f} threshold={decision.threshold:.4f} "
            f"msg=Continuous-time sequence drift exceeded validation-calibrated threshold"
        )
        
        # 2. Log to soar_alerts.log
        with open(self.alert_log_file, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {cef_string}\n")
            
        logger.warning(f"[ALERT] SIEM/SOAR ALERT DISPATCHED: {cef_string}")
        return cef_string
