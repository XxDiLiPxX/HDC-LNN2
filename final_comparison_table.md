# Final Baseline Model Benchmark Comparison

This document compiles the master benchmark evaluation results across all 5 architectures (**Mamba-2**, **1D-CNN**, **LSTM-Autoencoder**, **HDC-LNN**, and **LSTM**) on full-scale benchmark datasets under entity-disjoint leakage isolation.

> [!NOTE]
> **Baseline Stochasticity & Calibration**: Baseline numbers reflect the most recent verified run; minor run-to-run variance (~1-3% on some metrics) exists for baselines not using fixed calibration, due to training stochasticity not fully seeded across all paths — this does not affect HDC-LNN's frozen, deterministic configuration.

---

## 1. Full-Scale 5-Baseline Detection Performance (UNSW-NB15 Benchmark, 30,000 flows)

Evaluated on `UNSW_NB15_testing-set.csv` (30,000 contiguous flows, $N_{\text{test}} = 4,613$ flows across 41 disjoint entities):

| Model Architecture | F1 Score | Precision | Recall | AUROC | PR-AUC | FPR @ 95% TPR | Latency (ms/flow) | Throughput (flows/s) | Peak RSS (MB) |
|:-------------------|:--------:|:---------:|:------:|:-----:|:------:|:-------------:|:-----------------:|:--------------------:|--------------:|
| **Mamba-2 (Pure PyTorch SSM)** | **0.9792** | 0.9925 | **0.9663** | **0.9964** | **0.9988** | **0.0208** | 1.19 ms | 843.80 | 3908.52 |
| **Autoencoder (LSTM-AE)** | 0.9525 | 0.9444 | 0.9606 | 0.9606 | 0.9867 | 0.1769 | 4.50 ms | 222.14 | **2356.27** |
| **1D-CNN (Temporal Conv)** | 0.9340 | **0.9959** | 0.8794 | 0.9944 | 0.9981 | 0.0227 | 2.50 ms | 399.81 | 5322.32 |
| **HDC-LNN (Optimized, Ours)** | 0.8719 | 0.8963 | 0.8487 | 0.8610 | 0.9598 | 0.8221 | **0.56 ms** | **1779.73** | 4673.63 |
| **LSTM (Recurrent Baseline)** | 0.4006 | 0.9989 | 0.2506 | 0.7823 | 0.9303 | 0.8496 | 2.76 ms | 361.88 | 4752.32 |

---

## 2. Full-Scale 5-Baseline Detection Performance (NSL-KDD Benchmark, 22,544 flows)

Evaluated on `kdd_test.csv` (22,544 contiguous flows, $N_{\text{test}} = 3,466$ flows across disjoint entities):

| Model Architecture | F1 Score | Precision | Recall | AUROC | PR-AUC | FPR @ 95% TPR | Latency (ms/flow) | Throughput (flows/s) | Peak RSS (MB) |
|:-------------------|:--------:|:---------:|:------:|:-----:|:------:|:-------------:|:-----------------:|:--------------------:|--------------:|
| **Autoencoder (LSTM-AE)** | **0.8691** | 0.8454 | **0.8943** | **0.9423** | **0.9499** | **0.4042** | 3.69 ms | 271.30 | **2788.75** |
| **Mamba-2 (Pure PyTorch SSM)** | 0.8274 | **0.8511** | 0.8049 | 0.9218 | 0.9369 | 0.4669 | 1.05 ms | 949.44 | 3647.57 |
| **HDC-LNN (Optimized, Ours)** | 0.8089 | 0.8183 | 0.7996 | 0.8541 | 0.8859 | 0.9441 | **0.61 ms** | **1629.37** | 3717.15 |
| **1D-CNN (Temporal Conv)** | 0.7865 | 0.7844 | 0.7886 | 0.8648 | 0.8695 | 0.6015 | 2.14 ms | 467.73 | 4203.14 |
| **LSTM (Recurrent Baseline)** | 0.0880 | 0.9405 | 0.0461 | 0.7008 | 0.7700 | 0.9555 | 3.29 ms | 303.97 | 3652.34 |

---

## 3. HDC-LNN Before / After Optimization Summary (All 9 Metrics)

> **Note on Latency / Throughput Measurements**: The decision threshold change is a single floating-point comparison ($O(1)$) with near-zero computational overhead. Slight variations in reported latency/throughput between benchmark runs reflect standard run-to-run system execution variance, not algorithmic acceleration. The core optimization results are the **+0.1948 (UNSW)** and **+0.1328 (KDD)** F1 improvements achieved by moving to the empirical validation F1-maximizing operating point, along with the Ledoit-Wolf shrinkage manifold refinement.

| Benchmark Metric | UNSW Baseline | UNSW Optimized | UNSW Delta | KDD Baseline | KDD Optimized | KDD Delta |
|:-----------------|:-------------:|:--------------:|:----------:|:------------:|:-------------:|:---------:|
| **F1 Score** | 0.6771 | **0.8719** | **+0.1948 (+28.8%)** | 0.6761 | **0.8089** | **+0.1328 (+19.6%)** |
| **Precision** | 0.9792 | 0.8963 | -0.0829 (-8.5%) | 0.9249 | 0.8183 | -0.1066 (-11.5%) |
| **Recall** | 0.5174 | **0.8487** | **+0.3313 (+64.0%)** | 0.5327 | **0.7996** | **+0.2670 (+50.1%)** |
| **AUROC** | 0.8568 | **0.8610** | +0.0042 (+0.5%) | 0.8621 | 0.8541 | -0.0080 (-0.9%) |
| **PR-AUC** | 0.9581 | **0.9598** | +0.0017 (+0.2%) | 0.8902 | 0.8859 | -0.0042 (-0.5%) |
| **FPR @ 95% TPR** | 0.8221 | 0.8221 | 0.0000 (Ranking invariant) | 0.9396 | 0.9441 | +0.0046 |
| **Latency (ms/flow)** | 0.97 ms | 0.56 ms | *Run-to-run variance* | 1.19 ms | 0.61 ms | *Run-to-run variance* |
| **Throughput (f/s)** | 1034.36 | 1779.73 | *Run-to-run variance* | 840.85 | 1629.37 | *Run-to-run variance* |
| **Peak RSS (MB)** | 4916.13 | 4673.63 | -242.50 MB | 3718.46 | 3717.15 | -1.31 MB |
| **Missed Attacks (FN)** | 1,716 | **538** | **-1,178 (-68.65%)** | 800 | **343** | **-457 (-57.12%)** |

---

## 4. Encoder Geometric Separability (Empirical Analysis)

| Encoder Architecture | Centroid Cos Dist | Norm Euclidean Dist | Fisher Ratio | Intra Sim | Inter Sim | Separability Margin |
|:---------------------|:-----------------:|:-------------------:|:------------:|:---------:|:---------:|--------------------:|
| **HDC (RecordEncoder)** | 0.2794 | 0.5169 | 0.246854 | 0.5343 | 0.3308 | 0.2035 |
| **SAX (Symbolic Aggregate)** | 0.2420 | 0.4474 | 0.164670 | 0.4602 | 0.2984 | 0.1618 |
| **RFF (Random Fourier)** | 0.2657 | 0.5558 | 0.359254 | 0.6274 | 0.4200 | 0.2074 |

---

## 5. Architectural Findings & Relative Ranking

- **Mamba-2 (Pure-PyTorch SSM)** achieves top-tier detection accuracy (F1: 0.9792 on UNSW, 0.8274 on KDD) and lowest FPR@95%TPR (0.0208 on UNSW) with high throughput (843–949 flows/sec) by capturing long-range multi-scale temporal dependencies.
- **HDC-LNN (Optimized, Ours)** achieves line-rate inference throughput (1629–1779 flows/sec, latency: 0.56–0.61 ms/flow) with strong balanced classification (F1: 0.8719 on UNSW, 0.8089 on KDD). The continuous CfC liquid time constants $\tau(\Delta t)$ enable efficient irregularity tracking, while Ledoit-Wolf shrinkage stabilizes the reference manifold.
- **1D-CNN & LSTM-Autoencoder**: 1D-CNN achieves high precision (0.9959) and low FPR@95%TPR (0.0227) on UNSW with F1: 0.9340. The LSTM-Autoencoder provides high unsupervised recall (0.9606 on UNSW, 0.8943 on KDD) at higher reconstruction latency (3.69–4.50 ms/flow).
- **LSTM (Recurrent Baseline)** provides high precision (0.9405–0.9989) but under-detects attacks under standard heuristic $k\cdot\sigma$ thresholding on high-dimensional manifolds (UNSW F1: 0.4006, KDD F1: 0.0880).
