# Final Baseline Model Benchmark Comparison

> [!WARNING]
> **SMOKE TEST MODE ONLY (5,000 flows, 1 epoch)**: Small sample size results in under-conditioned validation manifolds. Do not cite for benchmarking.

## 1. Sequence & Generative Model Detection Performance

| Model Architecture         | Dataset / Run ID          | F1 Score | Precision | Recall | AUROC  | PR-AUC | FPR @ 95% TPR | Latency (ms/flow) | Throughput (flows/s) | Peak RSS (MB) |
|:---------------------------|---------------------------|----------|-----------|--------|--------|--------|---------------|-------------------|----------------------|--------------:|
| Mamba-2 (Pure PyTorch SSM) | eval_kdd_test_mamba2      | 0.8523   | 0.7515    | 0.9845 | 0.9679 | 0.9700 | 0.1649        | 0.70 ms           | 1428.90              | 990.20        |
| 1D-CNN (Temporal Conv)     | eval_kdd_test_cnn         | 0.7926   | 0.6862    | 0.9380 | 0.9121 | 0.9286 | 0.5206        | 1.73 ms           | 579.22               | 1198.80       |
| Autoencoder (LSTM-AE)      | eval_kdd_test_autoencoder | 0.7989   | 0.8242    | 0.7752 | 0.8928 | 0.8545 | 0.3840        | 2.77 ms           | 361.39               | 926.90        |
| HDC-LNN (Ours)             | eval_kdd_test_hdclnn      | 0.8267   | 0.7933    | 0.8630 | 0.9094 | 0.9123 | 0.4304        | 0.83 ms           | 1200.93              | 1049.96       |
| LSTM (Recurrent Baseline)  | eval_kdd_test_lstm        | 0.8373   | 0.8062    | 0.8708 | 0.9003 | 0.8816 | 0.3505        | 2.20 ms           | 455.26               | 1050.36       |

## 2. Encoder Geometric Separability (Empirical Analysis)

| Encoder Architecture     | Centroid Cos Dist | Norm Euclidean Dist | Fisher Ratio | Intra Sim | Inter Sim | Separability Margin |
|:-------------------------|-------------------|---------------------|--------------|-----------|-----------|--------------------:|
| HDC (RecordEncoder)      | 0.2713            | 0.5110              | 0.242318     | 0.5374    | 0.3362    | 0.2012              |
| SAX (Symbolic Aggregate) | 0.2420            | 0.4474              | 0.164670     | 0.4602    | 0.2984    | 0.1618              |
| RFF (Random Fourier)     | 0.2657            | 0.5558              | 0.359254     | 0.6274    | 0.4200    | 0.2074              |

## 3. Honest Architectural Analysis & Relative Ranking

- **Mamba-2 (Pure-PyTorch SSM)** achieves F1: 0.8523, AUROC: 0.9679, Precision: 0.7515, Recall: 0.9845 with throughput of 1428.90 flows/sec (latency: 0.70 ms/flow) using selective state-space recurrence.
- **1D-CNN & LSTM-Autoencoder**: 1D-CNN achieves F1: 0.7926 (AUROC: 0.9121, latency: 1.73 ms/flow), while LSTM-Autoencoder achieves F1: 0.7989 (AUROC: 0.8928) with reconstruction latency of 2.77 ms/flow.
- **HDC-LNN (Ours)** demonstrates throughput: 1200.93 flows/sec and latency: 0.83 ms/flow (Precision: 0.7933, Recall: 0.8630, F1: 0.8267, AUROC: 0.9094) under closed-form continuous-time liquid neural dynamics.
- **LSTM (Recurrent Baseline)** achieves F1: 0.8373, AUROC: 0.9003, Precision: 0.8062, Recall: 0.8708 across recurrent state transitions.
