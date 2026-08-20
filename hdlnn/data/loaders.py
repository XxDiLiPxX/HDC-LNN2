import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from hdlnn.contracts.schemas import CanonicalFlow

logger = logging.getLogger(__name__)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


import math

def _normalize_label(value: Any) -> int:
    if value is None:
        return 0
    text = str(value).strip().lower()
    if text in {"0", "0.0", "normal", "benign", "legit", "legitimate", "false", "", "-", "nan", "null", "none"}:
        return 0
    if text in {"1", "1.0", "attack", "malicious", "true"}:
        return 1
    try:
        f_val = float(text)
        if math.isnan(f_val):
            return 0
        return 1 if f_val != 0.0 else 0
    except ValueError:
        # Non-numeric attack category names like 'exploits', 'dos', 'fuzzers', 'neptune', 'smurf', etc.
        return 1


def _load_csv_as_flows(
    csv_path: Path,
    categorical_columns: List[str],
    numerical_columns: List[str],
    label_column: str,
    entity_id_column: Optional[str] = None,
    sublabel_column: Optional[str] = None,
    limit: Optional[int] = None,
    entity_id_modulo: int = 254,
    stride: int = 1,
) -> List[CanonicalFlow]:
    flows: List[CanonicalFlow] = []
    global_time = 0.0

    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            if not row:
                continue
            if stride > 1 and (row_idx % stride != 0):
                continue

            entity_id = row.get(entity_id_column, "") if entity_id_column else ""
            if not entity_id or entity_id == "-" or entity_id.lower() == "nan":
                entity_id = f"10.0.0.{(row_idx % entity_id_modulo) + 1}"

            dur = _to_float(row.get("dur", row.get("duration", 0.0)))
            global_time += max(dur, 0.001)

            categorical_fields = {col: str(row.get(col, "-")) for col in categorical_columns}
            numerical_fields = {col: _to_float(row.get(col, 0.0)) for col in numerical_columns}

            label_val = row.get(label_column, None)
            if label_val is not None and label_val != "":
                label = _normalize_label(label_val)
            elif sublabel_column:
                label = _normalize_label(row.get(sublabel_column, 0))
            else:
                label = 0

            flows.append(
                CanonicalFlow(
                    entity_id=entity_id,
                    timestamp=global_time,
                    dt=0.0,
                    categorical_fields=categorical_fields,
                    numerical_fields=numerical_fields,
                    label=label,
                    split="",
                )
            )

            if limit and len(flows) >= limit:
                break

    logger.info("Loaded %s rows from %s (stride=%s)", len(flows), csv_path, stride)
    return flows


def load_dataset_flows(config: Any, data_dir: Path, source_file: str, limit: Optional[int] = None, stride: int = 1) -> List[CanonicalFlow]:
    csv_path = Path(source_file)
    if not csv_path.is_absolute():
        csv_path = Path(data_dir) / csv_path

    if not csv_path.exists():
        raise FileNotFoundError(f"Source CSV not found: {csv_path}")

    entity_id_column = config.data.get("entity_id_column")
    return _load_csv_as_flows(
        csv_path=csv_path,
        categorical_columns=list(config.categorical_columns),
        numerical_columns=list(config.numerical_columns),
        label_column=config.label_column,
        entity_id_column=entity_id_column,
        sublabel_column=config.get("sublabel_column", None),
        limit=limit,
        entity_id_modulo=config.data.get("entity_id_modulo", 254),
        stride=stride,
    )
