import urllib.request
import csv
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from hdlnn.contracts.schemas import CanonicalFlow

logger = logging.getLogger(__name__)

def download_file(url: str, dest_path: Path):
    """Downloads a file from a URL to dest_path if it doesn't already exist."""
    if dest_path.exists():
        logger.info(f"File already exists at {dest_path}")
        return
    
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {url} to {dest_path}...")
    req = urllib.request.Request(url, headers={'User-Agent': 'AntigravityAgent/1.0'})
    try:
        with urllib.request.urlopen(req) as response:
            with open(dest_path, 'wb') as f:
                f.write(response.read())
        logger.info("Download completed successfully.")
    except Exception as e:
        logger.error(f"Failed to download file from {url}: {e}")
        raise e

def load_raw_unsw_nb15(
    config: Any, 
    data_dir: Path, 
    limit: int = None
) -> List[CanonicalFlow]:
    """Downloads (if necessary) and loads the UNSW-NB15 training and testing CSVs.
    
    Synthesizes entity_id (based on row_id modulo) and timestamps (accumulated dur).
    """
    cache_dir = Path(data_dir) / config.data.get("cache_dir", "data/raw")
    cache_dir.mkdir(parents=True, exist_ok=True)

    urls = {
        "train": "https://raw.githubusercontent.com/Nir-J/ML-Projects/master/UNSW-Network_Packet_Classification/UNSW_NB15_training-set.csv",
        "test": "https://raw.githubusercontent.com/Nir-J/ML-Projects/master/UNSW-Network_Packet_Classification/UNSW_NB15_testing-set.csv"
    }
    
    files = {
        "train": cache_dir / "UNSW_NB15_training-set.csv",
        "test": cache_dir / "UNSW_NB15_testing-set.csv"
    }

    # Download if missing
    for split_key, url in urls.items():
        download_file(url, files[split_key])

    # Feature definitions
    categorical_cols = [
        "proto", "service", "state", "is_ftp_login", "is_sm_ips_ports"
    ]
    numerical_cols = [
        "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sttl", "dttl",
        "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit",
        "swin", "stcpb", "dtcpb", "dwin", "tcprtt", "synack", "ackdat", "smean",
        "dmean", "trans_depth", "response_body_len", "ct_srv_src", "ct_state_ttl",
        "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
        "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst"
    ]
    all_headers = [
        "id", "dur", "proto", "service", "state", "spkts", "dpkts", "sbytes",
        "dbytes", "rate", "sttl", "dttl", "sload", "dload", "sloss", "dloss",
        "sinpkt", "dinpkt", "sjit", "djit", "swin", "stcpb", "dtcpb", "dwin",
        "tcprtt", "synack", "ackdat", "smean", "dmean", "trans_depth",
        "response_body_len", "ct_srv_src", "ct_state_ttl", "ct_dst_ltm",
        "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "is_ftp_login",
        "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst",
        "is_sm_ips_ports", "attack_cat", "label"
    ]

    flows: List[CanonicalFlow] = []
    global_time = 0.0
    row_count = 0

    # Load both files to form the full base set
    for split_key in ["train", "test"]:
        file_path = files[split_key]
        logger.info(f"Parsing {file_path}...")
        with open(file_path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            # Detect header row and skip if present
            try:
                first_row = next(reader)
                if first_row and first_row[0] == "id":
                    # Header present, proceed
                    pass
                else:
                    # No header, rewind or parse first row
                    f.seek(0)
                    reader = csv.reader(f)
            except StopIteration:
                continue

            for row in reader:
                if not row:
                    continue
                
                # Map columns to their values
                row_dict = dict(zip(all_headers, row))
                
                # Parse label
                try:
                    label = int(row_dict.get("label", 0))
                except ValueError:
                    label = 0

                # Simulate entity_id using modulo mapping
                entity_modulo = config.data.get("entity_id_modulo", 254)
                entity_id = f"10.0.0.{(row_count % entity_modulo) + 1}"

                # Parse duration to advance global clock
                try:
                    dur = float(row_dict.get("dur", 0.0))
                except ValueError:
                    dur = 0.0
                # Advance global timestamp (ensure monotonic progression)
                global_time += max(dur, 0.001)
                timestamp = global_time

                # Map categorical fields
                categorical_fields: Dict[str, str] = {}
                for col in categorical_cols:
                    categorical_fields[col] = row_dict.get(col, "-")

                # Map numerical fields
                numerical_fields: Dict[str, float] = {}
                for col in numerical_cols:
                    try:
                        numerical_fields[col] = float(row_dict.get(col, 0.0))
                    except ValueError:
                        numerical_fields[col] = 0.0

                flow = CanonicalFlow(
                    entity_id=entity_id,
                    timestamp=timestamp,
                    dt=0.0,  # Computed later in splitter
                    categorical_fields=categorical_fields,
                    numerical_fields=numerical_fields,
                    label=label,
                    split=""  # Assigned later in splitter
                )
                flows.append(flow)
                row_count += 1

                if limit and row_count >= limit:
                    break
        if limit and row_count >= limit:
            break

    logger.info(f"Loaded {len(flows)} raw records successfully.")
    return flows
