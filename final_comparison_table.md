# Final Baseline Model Benchmark Comparison

This document compiles the master benchmark evaluation results across all 5 architectures (**Mamba-2**, **1D-CNN**, **LSTM-Autoencoder**, **HDC-LNN**, and **LSTM**) on full-scale benchmark datasets under entity-disjoint leakage isolation.

---

## 1. Full-Scale 5-Baseline Detection Performance (UNSW-NB15 Benchmark, 30,000 flows)

Evaluated on `UNSW_NB15_testing-set.csv` (30,000 contiguous flows, $N_{\text{test}} = 4,613$ flows across 41 disjoint entities):

| Model Architecture | F1 Score | Precision | Recall | AUROC | PR-AUC | FPR @ 95% TPR | Latency (ms/flow) | Throughput (flows/s) | Peak RSS (MB) |
|:-------------------|:--------:|:---------:|:------:|:-----:|:------:|:-------------:|:-----------------:|:--------------------:|--------------:|
| **Mamba-2 (Pure PyTorch SSM)** | **0.9923** | 0.9924 | **0.9921** | **0.9972** | **0.9991** | **0.0142** | 1.38 ms | 722.74 | 4830.91 |
| **Autoencoder (LSTM-AE)** | 0.9525 | 0.9444 | 0.9606 | 0.9606 | 0.9867 | 0.1769 | 5.48 ms | 182.48 | **2389.66** |
| **1D-CNN (Temporal Conv)** | 0.9297 | **0.9952** | 0.8723 | 0.9938 | 0.9979 | 0.0274 | 2.86 ms | 349.32 | 5384.56 |
| **HDC-LNN (Optimized, Ours)** | 0.8719 | 0.8963 | 0.8487 | 0.8610 | 0.9598 | 0.8221 | **0.83 ms** | **1205.46** | 4856.62 |
| **LSTM (Recurrent Baseline)** | 0.3265 | 0.9986 | 0.1952 | 0.7820 | 0.9292 | 0.8496 | 2.93 ms | 341.23 | 4615.88 |

---

## 2. Full-Scale 5-Baseline Detection Performance (NSL-KDD Benchmark, 22,544 flows)

Evaluated on `kdd_test.csv` (22,544 contiguous flows, $N_{\text{test}} = 3,466$ flows across disjoint entities):

| Model Architecture | F1 Score | Precision | Recall | AUROC | PR-AUC | FPR @ 95% TPR | Latency (ms/flow) | Throughput (flows/s) | Peak RSS (MB) |
|:-------------------|:--------:|:---------:|:------:|:-----:|:------:|:-------------:|:-----------------:|:--------------------:|--------------:|
| **Mamba-2 (Pure PyTorch SSM)** | **0.8523** | 0.7515 | **0.9845** | **0.9679** | **0.9700** | **0.1649** | **0.70 ms** | **1428.90** | 990.20 |
| **LSTM (Recurrent Baseline)** | 0.8373 | 0.8062 | 0.8708 | 0.9003 | 0.8816 | 0.3505 | 2.20 ms | 455.26 | 1050.36 |
| **HDC-LNN (Optimized, Ours)** | 0.8089 | **0.8183** | 0.7996 | 0.8541 | 0.8859 | 0.9441 | 0.72 ms | 1393.44 | 3739.97 |
| **Autoencoder (LSTM-AE)** | 0.7989 | 0.8242 | 0.7752 | 0.8928 | 0.8545 | 0.3840 | 2.77 ms | 361.39 | **926.90** |
| **1D-CNN (Temporal Conv)** | 0.7926 | 0.6862 | 0.9380 | 0.9121 | 0.9286 | 0.5206 | 1.73 ms | 579.22 | 1198.80 |

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
| **Latency (ms/flow)** | 0.97 ms | 0.83 ms | *Run-to-run variance* | 1.19 ms | 0.72 ms | *Run-to-run variance* |
| **Throughput (f/s)** | 1034.36 | 1205.46 | *Run-to-run variance* | 840.85 | 1393.44 | *Run-to-run variance* |
| **Peak RSS (MB)** | 4916.13 | 4856.62 | -59.51 MB | 3718.46 | 3739.97 | +21.51 MB |
| **Missed Attacks (FN)** | 1,716 | **538** | **-1,178 (-68.65%)** | 800 | **343** | **-457 (-57.12%)** |

---

## 4. Encoder Geometric Separability (Empirical Analysis)

| Encoder Architecture | Centroid Cos Dist | Norm Euclidean Dist | Fisher Ratio | Intra Sim | Inter Sim | Separability Margin |
|:---------------------|:-----------------:|:-------------------:|:------------:|:---------:|:---------:|--------------------:|
| **HDC (RecordEncoder)** | 0.2713 | 0.5110 | 0.242318 | 0.5374 | 0.3362 | 0.2012 |
| **SAX (Symbolic Aggregate)** | 0.2420 | 0.4474 | 0.164670 | 0.4602 | 0.2984 | 0.1618 |
| **RFF (Random Fourier)** | 0.2657 | 0.5558 | 0.359254 | 0.6274 | 0.4200 | 0.2074 |

---

## 5. Architectural Findings & Relative Ranking

- **Mamba-2 (Pure-PyTorch SSM)** achieves top-tier detection accuracy (F1: 0.9923 on UNSW, 0.8523 on KDD) and lowest FPR@95%TPR (0.0142) with high throughput (722–1428 flows/sec) by capturing long-range multi-scale temporal dependencies.
- **HDC-LNN (Optimized, Ours)** achieves line-rate inference throughput (1205–1393 flows/sec, latency: 0.72–0.83 ms/flow) with strong balanced classification (F1: 0.8719 on UNSW, 0.8089 on KDD). The continuous CfC liquid time constants $\tau(\Delta t)$ enable efficient irregularity tracking, while Ledoit-Wolf shrinkage stabilizes the reference manifold.
- **1D-CNN & LSTM-Autoencoder**: 1D-CNN achieves high precision (0.9952) and low FPR@95%TPR (0.0274) on UNSW but requires sliding window buffering (latency: 1.73–2.86 ms/flow). The LSTM-Autoencoder provides high unsupervised recall (0.9606) at higher reconstruction latency (2.77–5.48 ms/flow).
- **LSTM (Recurrent Baseline)** provides solid performance on simple tabular sequences (KDD F1: 0.8373) but degrades in high-dimensional continuous feature spaces (UNSW F1: 0.3265).
