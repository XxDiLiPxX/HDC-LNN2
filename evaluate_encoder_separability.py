import math
import logging
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

from hdlnn.contracts.schemas import CanonicalFlow
from hdlnn.hdc.encoder import RecordEncoder
from hdlnn.hdc.alternative_encoders import SAXEncoder, RFFEncoder
from hdlnn.data.loaders import _load_csv_as_flows

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_encoder_separability(
    dataset_path: str = "datasets/UNSW_NB15_testing-set.csv",
    limit: int = 5000,
    D: int = 10000
) -> Dict[str, Dict[str, float]]:
    """Evaluates geometric separability between normal and attack traffic representations across encoders."""
    categorical_cols = ["proto", "service", "state"]
    numerical_cols = [
        "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sttl", "dttl", 
        "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit", 
        "swin", "stcpb", "dtcpb", "dwin", "tcprtt", "synack", "ackdat", "smean", 
        "dmean", "trans_depth", "response_body_len"
    ]
    
    flows = _load_csv_as_flows(
        csv_path=Path(dataset_path),
        categorical_columns=categorical_cols,
        numerical_columns=numerical_cols,
        label_column="label",
        entity_id_column="srcip",
        limit=limit
    )
    logger.info(f"Loaded {len(flows)} flows for encoder separability evaluation.")
    
    # Split into train normals for fitting, and eval normals/attacks for evaluation
    normal_flows = [f for f in flows if f.label == 0]
    attack_flows = [f for f in flows if f.label == 1]
    
    split_idx = int(len(normal_flows) * 0.6)
    train_normals = normal_flows[:split_idx]
    eval_normals = normal_flows[split_idx:]
    eval_attacks = attack_flows[:len(eval_normals)]
    
    logger.info(f"Train Normals (fitting): {len(train_normals)}, Eval Normals: {len(eval_normals)}, Eval Attacks: {len(eval_attacks)}")
    
    encoders = {
        "HDC (RecordEncoder)": RecordEncoder(
            D=D, 
            categorical_columns=categorical_cols, 
            numerical_columns=numerical_cols
        ),
        "SAX (Symbolic Aggregate)": SAXEncoder(
            D=D, 
            alphabet_size=8, 
            categorical_columns=categorical_cols, 
            numerical_columns=numerical_cols
        ),
        "RFF (Random Fourier)": RFFEncoder(
            D=D, 
            gamma=0.5, 
            categorical_columns=categorical_cols, 
            numerical_columns=numerical_cols
        )
    }
    
    results = {}
    
    for name, encoder in encoders.items():
        logger.info(f"Fitting {name} on {len(train_normals)} normal flows...")
        encoder.fit(train_normals)
        
        logger.info(f"Encoding eval normal and attack flows with {name}...")
        norm_hvs = torch.stack([hv.vector for hv in encoder.encode_batch(eval_normals)])
        att_hvs = torch.stack([hv.vector for hv in encoder.encode_batch(eval_attacks)])
        
        # 1. Centroids
        mu_norm = norm_hvs.mean(dim=0)
        mu_att = att_hvs.mean(dim=0)
        
        # 2. Centroid Cosine Distance (1 - cos_sim)
        cos_sim = torch.cosine_similarity(mu_norm.unsqueeze(0), mu_att.unsqueeze(0)).item()
        centroid_cos_dist = float(1.0 - cos_sim)
        
        # 3. Normalized Euclidean Centroid Distance (||mu_0 - mu_1|| / sqrt(D))
        diff_norm = torch.norm(mu_norm - mu_att, p=2).item()
        eucl_dist = float(diff_norm / math.sqrt(D))
        
        # 4. Fisher-like Class Separability Ratio: ||mu_0 - mu_1||^2 / (var_0 + var_1)
        var_norm = norm_hvs.var(dim=0).sum().item()
        var_att = att_hvs.var(dim=0).sum().item()
        fisher_ratio = float((diff_norm ** 2) / (var_norm + var_att + 1e-8))
        
        # 5. Mean Intra-Normal Cosine Similarity vs Inter-Class Similarity
        norm_unit = norm_hvs / norm_hvs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        att_unit = att_hvs / att_hvs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        
        sub_n = min(len(norm_unit), 200)
        sub_a = min(len(att_unit), 200)
        intra_norm_sim = float(torch.mm(norm_unit[:sub_n], norm_unit[:sub_n].T).mean().item())
        inter_class_sim = float(torch.mm(norm_unit[:sub_n], att_unit[:sub_a].T).mean().item())
        margin = float(intra_norm_sim - inter_class_sim)
        
        results[name] = {
            "cos_dist": centroid_cos_dist,
            "eucl_dist": eucl_dist,
            "fisher_ratio": fisher_ratio,
            "intra_sim": intra_norm_sim,
            "inter_sim": inter_class_sim,
            "margin": margin
        }
        
    # Save results to JSON artifact
    output_dir = Path("runs")
    output_dir.mkdir(exist_ok=True, parents=True)
    json_path = output_dir / "encoder_separability.json"
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(results, jf, indent=2)
    logger.info(f"Saved empirical encoder separability metrics to {json_path}")
    
    return results


def print_separability_table(results: Dict[str, Dict[str, float]]) -> str:
    headers = [
        "Encoder Architecture", "Centroid Cos Dist", "Norm Euclidean Dist", 
        "Fisher Ratio", "Intra Sim", "Inter Sim", "Separability Margin"
    ]
    rows = []
    for enc_name, m in results.items():
        rows.append([
            enc_name,
            f"{m['cos_dist']:.4f}",
            f"{m['eucl_dist']:.4f}",
            f"{m['fisher_ratio']:.6f}",
            f"{m['intra_sim']:.4f}",
            f"{m['inter_sim']:.4f}",
            f"{m['margin']:.4f}"
        ])
        
    col_widths = [max(len(str(row[i])) for row in rows + [headers]) for i in range(len(headers))]
    header_str = " | ".join(headers[i].ljust(col_widths[i]) for i in range(len(headers)))
    sep_str = "-|-".join("-" * col_widths[i] for i in range(len(headers)))
    
    lines = [
        f"| {header_str} |",
        f"|:{sep_str}:|"
    ]
    for row in rows:
        lines.append(f"| " + " | ".join(str(row[i]).ljust(col_widths[i]) for i in range(len(headers))) + " |")
        
    table_str = "\n".join(lines)
    return table_str


if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Evaluate empirical geometric separability across HDC encoders.")
    parser.add_argument("--dataset", type=str, default="datasets/UNSW_NB15_testing-set.csv", help="Dataset CSV path.")
    parser.add_argument("--limit", type=int, default=5000, help="Number of flows to evaluate.")
    args = parser.parse_args()
    
    res = compute_encoder_separability(dataset_path=args.dataset, limit=args.limit)
    tbl = print_separability_table(res)
    print("\n--- ENCODER-ONLY SEPARABILITY COMPARISON TABLE ---")
    print(tbl)
    print("---------------------------------------------------\n")
