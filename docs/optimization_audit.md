# HDC-LNN Optimization Audit

This document describes implementation behavior, not benchmark results. No new
accuracy or efficiency result is claimed until a dependency-complete experiment
writes a new `runs/<run_id>/metrics.json` artifact.

## Audited pipeline

| Stage | Implementation |
|---|---|
| Load and labels | `hdlnn/data/loaders.py`; labels are normalized to `0=normal`, `1=anomaly`. |
| Entity split and time deltas | `hdlnn/data/splitter.py`; entity-disjoint seeded allocation, per-entity chronological `dt`. |
| Numerical and categorical preprocessing | `RecordEncoder.fit` fits train-only min/max numerical ranges and train-only categorical item memory. Unseen values map to OOV after the memory is locked. |
| HDC encoding | `RecordEncoder.encode_batch`; bound categorical/numerical vectors are bundled by bipolar sign. Dimension is `hdc.dimension`. |
| Sequences | `prepare_sequences`; controlled by `model.sequence_length` (default 16). |
| Liquid model | `LNNSequenceModel`, an `ncps.torch.CfC` with `model.hidden_dim`; `step` uses the same CfC path as batch forward. |
| Objective and optimizer | Next-vector MSE with Adam; `model.lr`, `epochs`, `batch_size`, and `weight_decay` are configuration-controlled. |
| Normal manifold | `ReferenceManifold`; validation normal hidden states only, Ledoit-Wolf precision covariance (cosine fallback above 1000 dimensions). |
| Anomaly score | Mahalanobis/cosine distance. **Higher score means more anomalous.** |
| Threshold | `hdlnn/eval/calibration.py`; validation-only, exact observed score threshold, then frozen for test. |
| Ranking metrics | `hdlnn/eval/metrics.py`; sklearn ROC/PR routines with anomaly (`1`) as the positive class. |
| Efficiency | Test-loop wall time becomes latency/throughput; current process RSS is recorded. |

## Corrections and controls

- AUROC now handles tied scores correctly through `roc_auc_score`.
- FPR@95%TPR is now the minimum empirical FPR over ROC points with TPR at least
  0.95, using the same score direction and tie policy as ROC.
- Thresholding is `score >= threshold` throughout Mahalanobis calibration and
  inference.
- Calibration policies are configurable: `f1_max`, `precision_priority`,
  `recall_priority`, `security_constrained`, and `tpr_95`.
- Validation decisions and score-distribution/ROC/PR/threshold diagnostics are
  retained separately from untouched test decisions for Mahalanobis runs.
- Existing run folders cannot be overwritten; each result records its config
  hash, dimension, sequence length, learning rate, epoch count, and seed.

## Required experiment sequence

Run candidates with fixed entity split and seed in this order: calibration
policy, preprocessing candidate, LNN hyperparameters, sequence length
`8/16/32/64`, HDC dimension `1024/2048/4096/8192`, then manifold settings.
Choose using validation results only, freeze the selection, then create one
previously unused test run ID. Compare FPR@95%TPR first, then ranking/classification
metrics, while retaining the identical latency/RSS procedure.

## Known measurement limitation

`peak_rss_mb` is the process RSS sampled after test inference, not operating-
system peak RSS. It is consistent across in-process candidates but should be
renamed or replaced with an OS peak-RSS sampler before making strict peak-memory
claims.
