# Model Card: HDC-LNN Instantaneous-Temporal Linear Fusion

## 1. Model Overview

```text
Model Name:
HDC-LNN Instantaneous-Temporal Linear Fusion

Status:
Production Baseline Architecture

Prior Prototype:
HDC-LNN Temporal-Only State Baseline

Diagnostic Reference:
Joint Representation Diagnostic Probe (Offline Ceiling)
```

---

## 2. Frozen Architectural Specifications

The production model uses the following parameters:
- HDC dimension ($D$): 10,000 (continuous codebooks and associative item memory).
- Instantaneous representation ($\mathbf{z}_t$): 128-dimensional projection with activation from the CfC backbone.
- Liquid temporal state ($\mathbf{h}_t$): 64-dimensional continuous-time closed-form ODE state.
- Sequence length: 16 time steps.
- Joint representation: $[\mathbf{h}_t, \mathbf{z}_t] \in \mathbb{R}^{192}$.
- Decision function: Linear fusion boundary $\sigma(\mathbf{w}^\top [\mathbf{h}_t, \mathbf{z}_t] + b)$ where $\mathbf{w} \in \mathbb{R}^{192}$ and $b \in \mathbb{R}$.
- Calibration: Decision threshold chosen by maximizing validation F1 and frozen for test evaluation.

---

## 3. Four-Seed Replicated Production Metrics

Evaluated across four independent seeds (42, 123, 456, 789) on three disjoint primary benchmarks (UNSW-NB15 Test, KDD Test, and NSL-KDD Test):

| Metric | Linear Fusion Production | Historical Baseline | Status |
| :--- | :---: | :---: | :--- |
| Macro F1 Score | 0.9326 $\pm$ 0.0103 | 0.9355 $\pm$ 0.0088 | Production Baseline |
| Macro Precision | 0.9172 $\pm$ 0.0185 | 0.9270 $\pm$ 0.0185 | Production Baseline |
| Macro Recall | 0.9495 $\pm$ 0.0120 | 0.9447 $\pm$ 0.0054 | Production Baseline |
| Macro AUROC | 0.9713 $\pm$ 0.0083 | 0.9760 $\pm$ 0.0060 | Production Baseline |
| Macro PR-AUC | 0.9766 $\pm$ 0.0047 | 0.9800 $\pm$ 0.0042 | Production Baseline |
| Macro FPR@95% TPR | 0.1443 $\pm$ 0.0545 | 0.1173 $\pm$ 0.0337 | Production Baseline |

---

## 4. Computational Profile

Measured under single-flow ($B=1$) streaming inference on an Apple Silicon CPU using PyTorch's CPU backend:

- Streaming latency: 0.059 ms per flow (59.2 microseconds, mean across 4 seeds)
- Streaming throughput: 16,889 flows per second (mean across 4 seeds)
- Peak RSS: 2,132 MB (2.08 GB)
- Total inference parameters: 1,321,344 (within the 1B parameter budget)

---

## 5. Diagnostic Representation Probe (Offline Reference)

Evaluated across the same four seeds using an offline linear classifier on $[\mathbf{h}_t, \mathbf{z}_t]$:

- Macro F1 Score: 0.9636 $\pm$ 0.0039
- Macro Precision: 0.9548 $\pm$ 0.0138
- Macro Recall: 0.9732 $\pm$ 0.0100
- Macro AUROC: 0.9888 $\pm$ 0.0022
- Macro PR-AUC: 0.9899 $\pm$ 0.0027
- Macro FPR@95% TPR: 0.0466 $\pm$ 0.0099
- Note: This measures representation capacity in offline analysis and is not deployed for online streaming.

---

## 6. Verification and Integrity Record

- Replicated across four independent random seeds (42, 123, 456, 789).
- Metrics recalculated and confirmed from raw confusion matrix counts ($F_1 = 2PR / (P+R)$).
- Thresholds calibrated strictly on validation data without access to test labels.
- Verified zero sample or entity overlap between training, validation, and test splits.
- Output from streaming step and sequence forward passes match within machine precision.
