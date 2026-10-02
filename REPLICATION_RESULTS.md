# HDC-LNN Multi-Seed Replication and Benchmark Verification Report

Evaluation protocol: 4-seed replication (seeds 42, 123, 456, 789) across three disjoint primary benchmarks (UNSW-NB15 Test, KDD Test, NSL-KDD Test) and four IoT-23 datasets.  
Hardware environment: Apple Silicon CPU, PyTorch CPU backend, single-threaded streaming evaluation ($B=1$).  
Parameter boundary: $\le 1.45\text{M}$ parameters (strictly below the 1B budget).  

---

## 1. Master Replicated Results Summary (Primary Benchmarks)

| Model or Variant | Parameters | Macro F1 | Macro Precision | Macro Recall | Macro AUROC | Macro PR-AUC | Macro FPR@95% | Streaming Latency | Streaming Throughput | Peak RSS | Status |
| :--- | ---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| Temporal Manifold Baseline | 1,322,417 | **0.9355** $\pm$ 0.0088 | **0.9270** $\pm$ 0.0185 | 0.9447 $\pm$ 0.0054 | **0.9760** $\pm$ 0.0060 | 0.9800 $\pm$ 0.0042 | **0.1173** $\pm$ 0.0337 | 0.0601 ms ($60.1\,\mu\text{s}$) | 16,630 flows/s | 1.81 GB | Production Baseline |
| Linear Fusion ($192 \to 1$) | 1,322,417 | 0.9326 $\pm$ 0.0103 | 0.9172 $\pm$ 0.0185 | **0.9495** $\pm$ 0.0120 | 0.9713 $\pm$ 0.0083 | 0.9782 $\pm$ 0.0065 | 0.1443 $\pm$ 0.0545 | **0.0582 ms** ($58.2\,\mu\text{s}$) | **17,182 flows/s** | 1.80 GB | Production Candidate |
| Diagnostic Linear Probe ($[h_t, z_t]$) | 1,322,417 | *0.9613 $\pm$ 0.0041* | *0.9507 $\pm$ 0.0104* | *0.9725 $\pm$ 0.0068* | *0.9883 $\pm$ 0.0030* | *0.9888 $\pm$ 0.0034* | *0.0490 $\pm$ 0.0077* | N/A | N/A | N/A | Offline Diagnostic Ceiling |
| Mamba-2 (Pure PyTorch SSM) | 1,322,704 | 0.9265 | 0.8902 | 0.9672 | 0.9736 | 0.9778 | 0.1407 | 0.0600 ms ($60.0\,\mu\text{s}$) | 16,747 flows/s | 3.85 GB | Historical Baseline |
| FT-Transformer | 1,457,232 | 0.9403 | 0.9236 | 0.9580 | 0.9824 | 0.9855 | 0.1052 | 0.5400 ms ($540\,\mu\text{s}$) | 2,258 flows/s | 3.08 GB | Historical Baseline |
| SAINT | 1,490,704 | 0.9425 | 0.9328 | 0.9536 | 0.9821 | 0.9857 | 0.1147 | 0.4100 ms ($410\,\mu\text{s}$) | 2,867 flows/s | 1.12 GB | Historical Baseline |

---

## 2. Per-Seed Verification Grid (4 Seeds $\times$ 3 Variants)

| Seed | Model Variant | Macro F1 | Precision | Recall | AUROC | FPR@95% TPR | Streaming Latency | Streaming Throughput |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 42 | Temporal Manifold Baseline | 0.9421 | 0.9337 | 0.9512 | 0.9785 | 0.0936 | $59.94\,\mu\text{s}$ | 16,684 flows/s |
| 42 | Linear Fusion | 0.9331 | 0.9103 | 0.9578 | 0.9732 | 0.1169 | $58.11\,\mu\text{s}$ | 17,209 flows/s |
| 42 | Diagnostic Linear Probe | 0.9569 | 0.9391 | 0.9758 | 0.9868 | 0.0560 | N/A | N/A |
| 123 | Temporal Manifold Baseline | 0.9434 | 0.9490 | 0.9380 | 0.9831 | 0.0889 | $59.88\,\mu\text{s}$ | 16,702 flows/s |
| 123 | Linear Fusion | 0.9462 | 0.9444 | 0.9479 | 0.9796 | 0.1139 | $58.24\,\mu\text{s}$ | 17,171 flows/s |
| 123 | Diagnostic Linear Probe | 0.9661 | 0.9636 | 0.9686 | 0.9910 | 0.0407 | N/A | N/A |
| 456 | Temporal Manifold Baseline | 0.9247 | 0.9060 | 0.9448 | 0.9693 | 0.1618 | $60.22\,\mu\text{s}$ | 16,606 flows/s |
| 456 | Linear Fusion | 0.9213 | 0.9108 | 0.9332 | 0.9597 | 0.2260 | $58.56\,\mu\text{s}$ | 17,078 flows/s |
| 456 | Diagnostic Linear Probe | 0.9631 | 0.9468 | 0.9802 | 0.9908 | 0.0443 | N/A | N/A |
| 789 | Temporal Manifold Baseline | 0.9319 | 0.9195 | 0.9450 | 0.9733 | 0.1249 | $60.52\,\mu\text{s}$ | 16,530 flows/s |
| 789 | Linear Fusion | 0.9299 | 0.9034 | 0.9591 | 0.9725 | 0.1205 | $57.91\,\mu\text{s}$ | 17,269 flows/s |
| 789 | Diagnostic Linear Probe | 0.9591 | 0.9533 | 0.9653 | 0.9848 | 0.0550 | N/A | N/A |

---

## 3. Per-Dataset Breakdown (Averaged Over 4 Seeds)

| Dataset Benchmark | Evaluation Variant | F1 Score | Precision | Recall | AUROC | PR-AUC | FPR@95% TPR |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| UNSW-NB15 Test | Temporal Manifold Baseline | 0.9950 | 0.9986 | 0.9915 | 0.9988 | 0.9999 | 0.0000 |
| UNSW-NB15 Test | Linear Fusion | **0.9978** | **0.9986** | **0.9969** | **0.9991** | 0.9999 | 0.0000 |
| UNSW-NB15 Test | Diagnostic Linear Probe | 0.9980 | 0.9980 | 0.9980 | 0.9995 | 0.9999 | 0.0000 |
| KDD Test | Temporal Manifold Baseline | **0.8991** | **0.8798** | **0.9195** | **0.9658** | **0.9695** | **0.1887** |
| KDD Test | Linear Fusion | 0.8920 | 0.8793 | 0.9057 | 0.9601 | 0.9642 | 0.2479 |
| KDD Test | Diagnostic Linear Probe | 0.9324 | 0.9150 | 0.9510 | 0.9810 | 0.9830 | 0.0820 |
| NSL-KDD Test | Temporal Manifold Baseline | **0.9125** | **0.9026** | 0.9232 | **0.9635** | **0.9705** | **0.1632** |
| NSL-KDD Test | Linear Fusion | 0.9080 | 0.8737 | **0.9458** | 0.9546 | 0.9605 | 0.1851 |
| NSL-KDD Test | Diagnostic Linear Probe | 0.9535 | 0.9390 | 0.9685 | 0.9845 | 0.9840 | 0.0650 |

---

## 4. IoT-23 Streaming Benchmark Performance (Seed 42)

| Capture File | Model Variant | F1 Score | Precision | Recall | AUROC | FPR@95% TPR | Streaming Throughput |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `dataset17.csv` | Linear Fusion | 0.9742 | 0.9497 | 1.0000 | 0.9989 | 0.0250 | 20,729 flows/s |
| `dataset19.csv` | Linear Fusion | 0.9978 | 0.9971 | 0.9986 | 0.9999 | 0.0267 | 21,657 flows/s |
| `dataset5.csv` | Linear Fusion | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 21,435 flows/s |
| `dataset23.csv` | Linear Fusion | 0.9130 | 0.8417 | 1.0000 | 0.9793 | 0.2060 | 21,203 flows/s |

---

## 5. Reconciling the Diagnostic Probe and Deployed Fusion Head

The multi-seed evaluation highlights a performance difference between the diagnostic linear probe ($F_1 = 0.9613$, $\text{FPR95} = 0.0490$) and the deployed linear fusion head ($F_1 = 0.9326$, $\text{FPR95} = 0.1443$).

Two operational factors account for this:

1. The diagnostic probe trains across the full training partition (approximately 3,500 flows). By contrast, the streaming fusion head fits only on validation states (approximately 750 flows) to comply with the zero-shot training protocol.
2. In deployment, the decision threshold is calibrated exclusively on validation flows using the F1-maximizing operating point. On KDD and NSL-KDD, minor distribution shifts between validation flows and the disjoint test partition cause an over-prediction of anomalies. The baseline manifold likelihood kernel applies continuous covariance regularization that dampens extreme score drift.

The instantaneous projection $z_t$ contains discriminative features, but linear thresholding without spatial manifold density regularizers introduces test-time variance. Because of this stability, the temporal manifold baseline remains the primary deployed configuration ($F_1 = 0.9355$, Latency $= 0.0601\text{ ms}$).
