# HDC-LNN: Continuous-Time Neural Hyperdimensional Computing for Telemetry Anomaly Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end framework and empirical benchmark comparing **Continuous-Time Closed-Form Liquid Neural Networks combined with Hyperdimensional Computing (HDC-LNN)** against modern state-space and sequence models (**Mamba-2**, **1D-CNN**, **LSTM-Autoencoder**, and **LSTM**) for line-rate cybersecurity anomaly detection.

---

## 📑 Table of Contents
1. [Architecture Overview](#-architecture-overview)
2. [Environment Setup & Activation](#-environment-setup--activation)
3. [Running the Baseline Benchmark Suite](#-running-the-baseline-benchmark-suite)
4. [Running Individual Baseline Models via CLI](#-running-individual-baseline-models-via-cli)
5. [Interactive Web Dashboard & Real-Time Telemetry Stream](#-interactive-web-dashboard--real-time-telemetry-stream)
6. [Encoder Geometric Separability Study](#-encoder-geometric-separability-study)
7. [Batch Multi-Dataset Runner](#-batch-multi-dataset-runner)
8. [Running Automated Tests](#-running-automated-tests)
9. [Artifacts, Reports & Metrics Structure](#-artifacts-reports--metrics-structure)

---

## 🧠 Architecture Overview

```
                          ┌────────────────────────┐
Raw Network Flow         │   HDC RecordEncoder    │  10,000-D Bipolar Vector
Telemetry (CSV / Zeek) ──>│ (Continuous Codebooks  │───────────────────────────┐
                          │  + Item Memory Binding)│                           │
                          └────────────────────────┘                           │
                                                                               ▼
┌────────────────────────┐    ┌───────────────────────────────────┐    ┌───────────────┐
│ One-Class Reference    │    │ Continuous-Time Closed-Form (CfC) │    │ Anomaly Score │
│ Normal Manifold Scorer │<───│      Liquid Neural Network        │<───│ $\mathbf{x}_t$│
│ (Mahalanobis Distance) │    │  (Adaptive Hidden States $h_t$)   │    │  $\Delta t_t$ │
└───────────┬────────────┘    └───────────────────────────────────┘    └───────────────┘
            │
            ▼
    Alert Decision: [Normal (0) / Attack (1)] + SIEM / SOAR Webhook
```

- **HDC Encoder ($D=10,000$)**: Maps continuous numeric metrics into orthogonal level hypervectors and binds categorical fields via associative item memory.
- **CfC Liquid Neural Network**: Computes continuous-time recurrent state transitions with non-linear time constants $\tau(\Delta t)$.
- **One-Class Reference Manifold**: Calibrates Mahalanobis/Cosine decision boundaries exclusively against normal traffic states during validation to eliminate target label leakage.

---

## ⚡ Environment Setup & Activation

### Step 1: Activate the Virtual Environment
Ensure your terminal session is inside the project root and activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS / Git Bash
source .venv/bin/activate
```

*(You will see `(.venv)` in your terminal prompt).*

### Step 2: Dataset & Runs Directory Setup
- `datasets/` and `runs/` folders exist in the repository structure but are intentionally kept empty of data in git tracking (`.gitignore`).
- Download benchmark datasets (e.g. UNSW-NB15, NSL-KDD, CICIoT2023) and place the CSV files directly inside `datasets/`. See [`datasets/README.md`](datasets/README.md) for details and expected file names.
- The `runs/` directory is automatically populated with `metrics.json`, `decisions.parquet`, and `roc_curve.png` whenever evaluation or training is run. It is normal and expected for `runs/` to be empty on a fresh clone. See [`runs/README.md`](runs/README.md).

---

## 🚀 Running the Baseline Benchmark Suite

The main benchmarking script is [`run_all_baselines.py`](run_all_baselines.py). It evaluates all 5 model architectures (**Mamba-2**, **1D-CNN**, **LSTM-Autoencoder**, **HDC-LNN**, and **LSTM**) under strict entity-disjoint leakage isolation.

### 1. Fast Smoke Test (Recommended First, finishes in ~15s)
Runs 5,000 flows with 1 epoch for rapid validation and pipeline sanity check:
```bash
python run_all_baselines.py --quick
```

### 2. Full-Scale Benchmark on Primary Dataset (UNSW-NB15)
Evaluates 30,000 contiguous flows with full validation manifold calibration:
```bash
python run_all_baselines.py --dataset UNSW_NB15_testing-set.csv
```

### 3. Generalization Benchmark on NSL-KDD
Evaluates the full model suite on the NSL-KDD intrusion detection dataset:
```bash
python run_all_baselines.py --dataset kdd_test.csv
```

### 4. Benchmark on ToN-IoT
```bash
python run_all_baselines.py --dataset ton-iot.csv
```

### 5. Benchmark Across ALL Datasets in `datasets/`
```bash
python run_all_baselines.py --all
```

> **CLI Options Summary:**
> - `--dataset <filename>`: Run all baselines against a specific CSV file in `datasets/`.
> - `--all`: Discover and evaluate all CSV dataset files in `datasets/`.
> - `--quick`: Capped smoke test (5,000 flows, 1 epoch).
> - `--limit <N>`: Explicitly set the number of telemetry flows evaluated.
> - `--output-table <path>`: Markdown table output path (defaults to `final_comparison_table.md`).

---

## 🔬 Running Individual Baseline Models via CLI

You can execute a single model baseline directly using [`hdlnn.pipeline`](hdlnn/pipeline.py):

```bash
# Evaluate HDC-LNN (Ours)
python -m hdlnn.pipeline --mode eval --baseline hdc-lnn --source-file datasets/UNSW_NB15_testing-set.csv

# Evaluate Mamba-2 (Pure PyTorch SSM)
python -m hdlnn.pipeline --mode eval --baseline mamba2 --source-file datasets/UNSW_NB15_testing-set.csv

# Evaluate 1D-CNN (Temporal Conv)
python -m hdlnn.pipeline --mode eval --baseline cnn --source-file datasets/UNSW_NB15_testing-set.csv

# Evaluate LSTM-Autoencoder
python -m hdlnn.pipeline --mode eval --baseline autoencoder --source-file datasets/UNSW_NB15_testing-set.csv

# Evaluate Recurrent LSTM
python -m hdlnn.pipeline --mode eval --baseline lstm --source-file datasets/UNSW_NB15_testing-set.csv
```

**Available Baselines:** `hdc-lnn`, `mamba2`, `cnn`, `autoencoder`, `lstm`, `hdc-only`, `lnn-only`.

---

## 🖥️ Interactive Web Dashboard & Real-Time Telemetry Stream

The project includes an interactive web interface powered by **FastAPI**, **Uvicorn**, and **WebSockets** with live telemetry simulation and anomaly charting.

### Launch the Dashboard:
```bash
python -m hdlnn.pipeline --mode gui --port 8050
```

### Accessing the Dashboard:
1. Open your browser and go to: **`http://localhost:8050`**
2. **Features available in the GUI**:
   - 📈 **Live Drift Monitoring**: Real-time continuous Mahalanobis divergence score charts with anomaly threshold lines.
   - ⚡ **Telemetry Stream Controller**: Start, pause, and adjust flow ingestion rate (flows/sec).
   - 🚨 **Synthetic Threat Injection**: Trigger real-time DNS Low-and-Slow exfiltration and Lateral Movement attacks on the fly.
   - 🎛️ **Threshold Tuning**: Adjust sensitivity multipliers ($k$) and observe live precision/recall impact.
   - 📋 **SOAR Alert Log**: Live feed of security incident response alerts and IP containment triggers.

---

## 📐 Encoder Geometric Separability Study

To evaluate how well different hyperdimensional encoding schemes separate normal traffic from attack patterns in hyperspace *prior* to neural network training:

```bash
python evaluate_encoder_separability.py
```

### What It Measures:
- **`RecordEncoder` (HDC)** vs. **`SAXEncoder` (Symbolic Aggregate)** vs. **`RFFEncoder` (Random Fourier Features)**.
- **Centroid Cosine Distance**: Angular separation of class means in 10,000-D space.
- **Normalized Euclidean Distance**: Scale-invariant Euclidean gap.
- **Fisher-like Ratio**: Ratio of between-class variance to within-class variance.
- **Separability Margin**: Net distinction ($\text{Intra-Normal Sim} - \text{Inter-Class Sim}$).
- Saves empirical metrics directly to [`runs/encoder_separability.json`](runs/encoder_separability.json).

---

## 📦 Batch Multi-Dataset Runner

To run HDC-LNN across all 31 dataset files in `datasets/` and generate a consolidated benchmark summary:

```bash
python run_all_datasets.py
```
> Outputs consolidated results to [`dataset_runs_summary.md`](dataset_runs_summary.md).

---

## 🧪 Running Automated Tests

Run the complete pytest test suite (covers encoder parity, manifold fitting, temporal splitters, SIMD fallback, and threat injectors):

```bash
pytest
```
*(All 15 tests pass in ~3 seconds).*

---

## 📊 Artifacts, Reports & Metrics Structure

Every benchmark execution saves structured, verifiable artifacts:

```
HDC-LNN2/
├── final_comparison_table.md       # Master benchmark comparison table + architectural ranking
├── datasets/                       # Local raw benchmark CSVs (gitignored except README)
│   └── README.md
└── runs/                           # Generated evaluation runs & checkpoints (gitignored except README)
    ├── README.md
    ├── <run_id>/
    │   ├── metrics.json            # Confusion matrix, F1, AUROC, latency, RSS, throughput
    │   ├── decisions.parquet       # Per-flow actual label, predicted label, drift score, timestamp
    │   └── config.yaml             # Exact snapshot of hyperparameters used for reproducibility
    └── encoder_separability.json   # Geometric clustering & separability metrics
```
