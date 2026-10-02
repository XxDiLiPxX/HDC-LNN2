# HDC-LNN Results and Version Changelog

This document tracks the chronological record of model iterations, architectural changes, empirical results, and operational status.

---

## Chronological Development Log

| Release / Iteration | Date | Architectural Configuration | Macro F1 | Macro Precision | Macro Recall | Macro AUROC | Macro FPR@95% | Latency | Throughput | Operational Status |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Iteration 1** | Aug 2026 | Soft Teacher Residual Blending ($\alpha=0.50$) | 0.9389 (peak) | 0.9351 | 0.9535 | 0.9798 | 0.1165 | 0.0644 ms | 15,529 flows/s | Early Prototype |
| **Iteration 2** | Aug 2026 | Deeper MLP Head ($66 \to 32 \to 16 \to 1$) | 0.9378 | 0.9326 | 0.9439 | 0.9745 | 0.1385 | 0.0661 ms | 15,152 flows/s | Superseded (Overfitting) |
| **Iteration 3** | Aug 2026 | Training-time Representation Alignment ($\lambda=0.02, M=4.0$) | 0.9359 | 0.9257 | 0.9474 | 0.9774 | 0.1181 | 0.0647 ms | 15,480 flows/s | Superseded |
| **Iteration 4** | Aug 2026 | Boundary Loss Optimization ($\lambda_b=0.005, M=3.0$) | 0.9334 | 0.9334 | 0.9371 | 0.9752 | 0.1190 | 0.0648 ms | 15,450 flows/s | Superseded |
| **Iteration 5** | Aug 2026 | Joint Representation and Margin Tuning | 0.9355 | 0.9270 | 0.9447 | 0.9760 | 0.1170 | 0.0658 ms | 15,218 flows/s | Superseded |
| **Iteration 6** | Sep 2026 | Fused In-Place Manifold Scoring and CfC JIT Hot Path | 0.9355 | 0.9270 | 0.9447 | 0.9760 | 0.1173 | 0.0601 ms | 16,630 flows/s | Historical Baseline |
| **Iteration 7** | Sep 2026 | Information Bottleneck and Representation Audit | N/A | N/A | N/A | N/A | N/A | N/A | N/A | Diagnostic Audit |
| **Production** | Sep 2026 | Instantaneous-Temporal Linear Fusion ($[h_t, z_t]$) | **0.9326** | **0.9172** | **0.9495** | **0.9713** | **0.1443** | **0.0592 ms** | **16,889 flows/s** | Production Model |
| **Diagnostic** | Sep 2026 | Diagnostic Linear Probe ($[h_t, z_t]$ Offline) | *0.9636* | *0.9548* | *0.9732* | *0.9888* | *0.0466* | N/A | N/A | Offline Diagnostic Ceiling |

---

## Architecture Evolution and Model Status Summary

1. Authoritative Production Model (HDC-LNN Instantaneous-Temporal Linear Fusion):
   - Combines a 64-dimensional continuous-time liquid state $\mathbf{h}_t$ and a 128-dimensional instantaneous projection $\mathbf{z}_t$ into a joint representation $[\mathbf{h}_t, \mathbf{z}_t] \in \mathbb{R}^{192}$.
   - Verified 4-seed macro detection metrics: $F_1 = 0.9326 \pm 0.0103$, Precision $= 0.9172 \pm 0.0185$, Recall $= 0.9495 \pm 0.0120$, AUROC $= 0.9713 \pm 0.0083$, FPR@95% $= 0.1443 \pm 0.0545$.
   - Systems profile: Latency $= 0.05923\text{ ms/flow}$ ($59.23\,\mu\text{s}$), Throughput $= 16,889\text{ flows/s}$, Peak RSS $= 2,132.09\text{ MB}$, Inference Parameters $= 1,321,344$.

2. Temporal Manifold Baseline:
   - The earlier temporal-only manifold scoring configuration remains documented as a verified baseline ($F_1 = 0.9355$, Latency $= 0.0601\text{ ms}$).

3. Diagnostic Representation Probe (Offline Analysis):
   - Measures representation capacity under linear evaluation: $F_1 = 0.9636 \pm 0.0039$, $\text{AUROC} = 0.9888 \pm 0.0022$, $\text{FPR95} = 0.0466 \pm 0.0099$. This probe is evaluated offline only.
