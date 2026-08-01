# HYDRA-LNN v2

### A Unified Hyperdimensional–Liquid Neural Architecture for Behavioral Drift & Temporal Anomaly Detection

**Consolidated Engineering Architecture Proposal** — reconciling the HYDRA-LNN product proposal with the HDC-LNN research specification into a single implementation path: research-grade rigor first, production hardening second.

| | |
|---|---|
| **Prepared by** | AI & Cybersecurity Systems Engineering — merged from HYDRA-LNN (product proposal) and HDC-LNN (research specification) |
| **Audience** | Executive Engineering Leadership, System Architecture Board, Research Engineering |
| **System class** | Neuro-symbolic behavioral drift / temporal anomaly detection engine |
| **Document status** | Unified design — supersedes both source documents |

> **⚠ What changed from the two source documents**
> This version keeps HYDRA-LNN's product framing, math appendix, and threat scenarios, and keeps HDC-LNN's layered architecture, leakage-prevention rules, and evaluation methodology. It removes unsupported performance claims (Section 5.2 of the original HYDRA-LNN proposal) and replaces them with benchmark *targets* that must be earned through the evaluation harness before appearing in a deck again.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [System Architecture — Layered View](#3-system-architecture--layered-view)
4. [Core Technical Approach](#4-core-technical-approach)
   - 4.1 [Spatial Layer — Hyperdimensional Encoding](#41-spatial-layer--hyperdimensional-encoding)
   - 4.2 [Temporal Layer — Liquid / Continuous-Time Core](#42-temporal-layer--liquid--continuous-time-core)
   - 4.3 [Divergence Engine — Anomaly Scoring](#43-divergence-engine--anomaly-scoring)
5. [Data Integrity & Leakage Prevention](#5-data-integrity--leakage-prevention)
6. [Software Engineering Discipline](#6-software-engineering-discipline)
7. [Performance Targets vs. Evaluation Methodology](#7-performance-targets-vs-evaluation-methodology)
8. [Threat Validation Scenarios](#8-threat-validation-scenarios)
9. [Mathematical Formulation](#9-mathematical-formulation)
10. [Implementation Roadmap](#10-implementation-roadmap)
11. [Out of Scope (Phase 1–2)](#11-out-of-scope-phase-12)
12. [Appendix — Data Model & Folder Layout](#12-appendix--data-model--folder-layout)

---

## 1. Executive Summary

Modern SIEM and EDR platforms struggle to detect slow, stealthy attacks hidden in massive, high-cardinality log streams. HYDRA-LNN addresses this by pairing two complementary techniques: **Hyperdimensional Computing (HDC)**, a fixed-memory symbolic encoder that represents any log event as a vector regardless of how many unique categorical values exist, and a **Liquid / continuous-time neural core (LTC / CfC)**, which reasons over irregular time gaps natively instead of forcing data into fixed windows.

This document unifies two prior proposals into one design. The original HYDRA-LNN proposal made a strong product case — sub-millisecond inference, cache-resident vectors, SIEM/SOAR integration — but stated its performance comparisons as fact without a benchmark methodology behind them. The companion HDC-LNN research specification defined that methodology in detail — entity-disjoint data splits, leakage-proof item memory, a purity-controlled anomaly reference manifold, named baselines — but stopped short of a production deployment path.

The unified architecture below treats the HDC-LNN research core as the **source of truth for the algorithm** and the HYDRA-LNN pipeline as the **target for production deployment**, connected by an explicit gate: no performance number is presented to stakeholders until it has been produced by the evaluation harness defined in Section 7, and no component ships to production until it satisfies the leakage-prevention rules in Section 5.

**Table 1. Provenance of each retained design decision**

| Kept from HYDRA-LNN (product proposal) | Kept from HDC-LNN (research spec) |
|---|---|
| Product framing, threat narratives, SOAR/SIEM integration target | Layered architecture (L1–L5) with strict downward dependencies |
| Closed-form ODE math for the liquid core (Section 9) | CfC via `ncps` as the reference implementation, not a from-scratch solver |
| Cache-line-aware vector sizing rationale | torchhd-based encoder, D=10,000 default, entity-disjoint splitting |
| Cosine-distance predictive scoring as a fast secondary signal | Mahalanobis divergence against a purity-controlled reference manifold as the primary scorer |
| Phase 4 production benchmarking & SIEM hook | Phase 3 evaluation harness, named baselines, ablation runner |

> **✗ Dropped from the original HYDRA-LNN proposal**
> The Section 5.2 comparison table (45–120ms Transformer vs. 8–15ms LSTM vs. <0.45ms HYDRA-LNN) is removed. It cited no dataset, hardware spec, or measurement protocol. Those numbers are replaced with the benchmark plan in Section 7 — the same claim can be made again once the harness actually produces it.

---

## 2. Problem Statement

### 2.1 High-Cardinality Explosion

Fields such as IPs, file hashes, dynamic ports, and URIs create sparse categorical spaces exceeding 10⁹ unique values. Standard one-hot or embedding-based approaches blow up in memory and fail on out-of-vocabulary (OOV) values encountered only at inference time.

### 2.2 Irregular Temporal Discretization

Real attacks — micro-exfiltration, slow beaconing, low-and-slow lateral movement — do not occur at fixed intervals. RNNs, LSTMs, and Transformers require fixed-window binning or zero-padding, which destroys exactly the temporal signal that would otherwise reveal the attack.

### 2.3 The Core Insight

The system needs to (a) handle unbounded categorical variety in constant memory, and (b) reason natively over irregular, continuous time rather than discrete steps — **and** it needs an evaluation discipline rigorous enough that its accuracy claims survive contact with a review board. The first two requirements motivate the HDC + LNN pairing; the third motivates everything in Sections 5–7.

---

## 3. System Architecture — Layered View

The system is organized into five strictly downward-dependent layers (L1 knows nothing above it), adopted directly from the HDC-LNN specification. This replaces the original HYDRA-LNN pipeline diagram, which mixed infrastructure concerns (Kafka ingest), algorithmic concerns (HDC/LNN), and product concerns (SOAR automation) into a single undifferentiated flow.

**Table 2. Layer responsibilities, dependency direction strictly downward**

| Layer | Name | Components |
|---|---|---|
| **L5** | Reporting | Tables, plots, dashboard — never recomputes a metric, only visualizes what L4 produced |
| **L4** | Evaluation | Benchmark harness, ablation runner, baselines (LSTM, CNN, HDC-only, LNN-only) |
| **L3** | Research Core | HDC Encoder → LNN Core → Divergence Engine — the core contribution |
| **L2** | Data Pipeline | Loaders, Splitter, Item Memory, quality checks |
| **L1** | Foundation | Contracts, Config, Logging, Seeds |

### Production Overlay

HYDRA-LNN's production concerns — Kafka ingestion, SIMD acceleration, SOAR automation — are not a sixth layer; they are a **deployment target that consumes L3's interfaces**. A streaming sidecar built in Phase 4 (Section 10) calls the same `IEncoder`, `ISequenceModel`, and `IDivergenceScorer` contracts that the research harness calls in Phase 3. This is what makes the batch/streaming numerical-parity rule in Section 6 load-bearing rather than aspirational: if production used a different code path than research, parity would be unenforceable.

### Per-Record Pipeline

1. Raw flow / log (`.csv`, `.parquet`, or streaming Sysmon / Netflow) enters the **Loader** and is mapped onto the common **CanonicalFlow** schema. The loader knows nothing about hypervectors, LNNs, or scoring.
2. The **Splitter / Item Memory** layer assigns the record to a fixed, versioned, entity-disjoint split and resolves categorical fields against the frozen item memory (Section 5).
3. The **HDC Encoder** binds and bundles the record's fields into one fixed-width **EncodedHypervector**.
4. The **LNN Core** (CfC / LTC) consumes the hypervector stream for that entity and updates a continuous-time **TrajectoryState**.
5. The **Divergence Engine** compares the trajectory state to the Reference Manifold (or, in predictive mode, to the model's own next-state prediction) and emits an **AnomalyDecision** — score, threshold, verdict.
6. In production, a high-severity **AnomalyDecision** is handed to the SOAR / SIEM connector (Section 10, Phase 4) exactly as HYDRA-LNN's original alerting engine described.

---

## 4. Core Technical Approach

### 4.1 Spatial Layer — Hyperdimensional Encoding

Each log event or flow record is mapped into a fixed-size, high-dimensional bipolar vector space using **torchhd** as the reference implementation (HDC-LNN), replacing the from-scratch AVX-512 bit-manipulation approach originally proposed in HYDRA-LNN. The from-scratch SIMD path is retained as an optional, later-phase hardware-acceleration backend once the algorithm is validated (see Phase 4, Section 10) — it is a performance optimization, not the reference definition of correctness.

- **Binding (⊗)** — pairs a field's key with its value (e.g., binds `source_ip` to `10.0.0.5`) while keeping the result mathematically distinct from both inputs.
- **Bundling (⊕)** — combines multiple bound key–value pairs into a single vector representing the whole event via addition and majority-rule thresholding.
- **Permutation (Πₖ)** — a cyclic shift encoding order and structure, such as a process-execution tree or network hop path.
- **Continuous fields** (byte counts, durations) use an ordered "thermometer" codebook so vector similarity scales smoothly with numeric distance.

Vector dimensionality: research default **D=10,000** (torchhd standard, HDC-LNN) rather than the 16,384-bit / 2,048-byte cache-line-tuned dimension proposed in HYDRA-LNN. **Dimensionality is a config-toggleable parameter**, not a hardcoded constant — the cache-alignment rationale is reintroduced explicitly as a production-tuning profile in Section 7, applied only after the research-default dimension has been validated for detection accuracy.

> **ℹ Stateless by design**
> The encoder is stateless per record — all "memory" lives in the frozen Item Memory (Section 5), not in the encoder itself. This is what makes the encoder safe to run in both batch (research) and streaming (production) modes without behavioral drift between the two.

### 4.2 Temporal Layer — Liquid / Continuous-Time Core

Rather than updating state at fixed time steps, the system models hidden state as a continuous function of time via a Liquid Time-Constant (LTC) / Closed-form Continuous-time (CfC) cell, implemented on top of the published `ncps` library (HDC-LNN) rather than a bespoke ODE solver. The closed-form update HYDRA-LNN specified in its appendix (Section 9) is mathematically consistent with CfC and is retained as the documented derivation — `ncps` is simply the audited, testable implementation of it.

The system's internal state naturally decays during quiet periods and updates rapidly during bursts of activity, without artificial time-windowing or padding. When a new event arrives after a gap of Δt — milliseconds or hours — the model computes the new state directly via closed-form integration, with no iterative replay of the gap required. This is what lets the system connect a DNS query from thirty minutes ago to one happening right now, without a sliding window ever holding both events simultaneously.

> **⚠ Execution consistency requirement**
> The LNN Core has a batch mode (training) and a streaming step mode (inference), and the two **must agree numerically**. This HDC-LNN requirement directly protects HYDRA-LNN's production latency claims: if streaming inference silently diverged from the model that was actually evaluated, the accuracy numbers in Section 7 would not describe the system running in production.

### 4.3 Divergence Engine — Anomaly Scoring

The two source documents proposed different scorers. Rather than pick one, the unified design exposes both behind the `IDivergenceScorer` interface, config-toggleable per deployment:

**Table 3. Two divergence-scoring modes, unified behind one interface**

| Mode | Method | Role |
|---|---|---|
| **Primary** | Mahalanobis distance from TrajectoryState to a Reference Manifold (mean, covariance) of normal-traffic hidden states | Statistically grounded drift score; the manifold is fit only on the normal-traffic slice of the validation split (Section 5) — auditable and reproducible |
| **Secondary / fast-path** | Cosine distance between the LNN's predicted next-state vector and the observed vector, adaptive threshold via EWMA (mean + k·σ) | Cheap, low-latency signal suitable for a streaming pre-filter ahead of the heavier Mahalanobis scorer in high-throughput production deployments |

**Recommendation:** validate detection quality (F1, AUROC, FPR@95%TPR — Section 7) using the Mahalanobis scorer as the primary metric during Phases 1–3. Promote the cosine/EWMA scorer to an always-on production pre-filter only in Phase 4, once its false-negative rate against the Mahalanobis scorer has itself been benchmarked — a fast filter that quietly drops true positives is worse than no filter.

---

## 5. Data Integrity & Leakage Prevention

This entire section is adopted from HDC-LNN; the original HYDRA-LNN proposal did not address data leakage anywhere. It is treated here as load-bearing engineering, not a research nicety — every performance claim carried forward from either source document is only meaningful if these rules hold.

**Table 4. Structural leakage controls carried over unchanged from HDC-LNN**

| Rule | Enforcement |
|---|---|
| **Item Memory leakage prevention** | HDC base hypervectors are fit on the TRAIN split only. Any categorical value seen for the first time at val/test time maps to a reserved OOV vector rather than being added on the fly. |
| **Manifold purity** | The Reference Manifold used for drift scoring is fit only on the normal-traffic slice of the VALIDATION split — never on test data or attack traffic. Fitting it on anything else would let the detector "see" what it is supposed to be catching. |
| **Strict, entity-disjoint splitting** | All flows for one source IP stay inside a single split. Splits are fixed, versioned, and never silently regenerated. |

> **✗ Why this section gates Section 7**
> HYDRA-LNN's original performance table implied a trained, evaluated system without ever describing a split, a held-out set, or an OOV policy. Any benchmark produced in Phase 3 (Section 10) that cannot show these three rules were followed does not get promoted into an executive-facing claim.

---

## 6. Software Engineering Discipline

- **Interface segregation** — every stage sits behind an interface (`IEncoder`, `ISequenceModel`, `IDivergenceScorer`) and is config-toggleable, which is what allows Section 4.3's dual scoring modes and Section 4.1's dual encoder backends to coexist without branching the codebase.
- **Execution consistency** — batch mode (training/eval) and streaming mode (latency benchmarking, production) must produce numerically identical outputs for the same inputs.
- **One shared metrics implementation** — the Evaluation Harness (L4) is the only place metrics are computed; the Reporting layer (L5) only visualizes what the harness already produced, so a dashboard number and a benchmark-run number can never silently disagree.
- **Self-contained run artifacts** — every harness run writes `runs/<run_id>/` containing a config snapshot, git commit hash, `metrics.json`, and `decisions.parquet`, so any number quoted externally is traceable back to an exact run.

---

## 7. Performance Targets vs. Evaluation Methodology

HYDRA-LNN's original memory/latency engineering targets are legitimate design goals and are kept as **targets**. They are explicitly separated from HDC-LNN's evaluation methodology, which is what will actually determine whether those targets are met and whether the system detects anything.

### 7.1 Production Engineering Targets (unvalidated until benchmarked)

**Table 5. Engineering targets — legitimate goals, explicitly not results**

| Target | Value | Basis |
|---|---|---|
| Per-event vector footprint | 2,048 bytes (2 KB) | 16,384-bit bipolar vector; fits inside a 32–48 KB L1 data cache — arithmetic fact, not a benchmark result |
| SIMD binding/bundling throughput | Design target, not yet measured | AVX-512 bitwise ops on an 8-core x86 target — to be measured against the harness's latency benchmark (7.2), not asserted |
| Vectorization throughput | >100,000 log vectorizations/sec (Phase 1 exit criterion) | Carried from HYDRA-LNN roadmap as a Phase 1 acceptance target |
| Streaming throughput | 50,000 events/sec (Phase 4 benchmarking load) | Carried from HYDRA-LNN roadmap as a Phase 4 acceptance target |

### 7.2 Evaluation Methodology (produces the actual numbers)

**Datasets**
- **UNSW-NB15** — primary, used for train / val / test.
- **CICIoT2023** — secondary, zero-shot cross-dataset transfer only.
- **NSL-KDD** — secondary, for comparability with prior published baselines.

**Metrics**
F1, Precision, Recall, AUROC, FPR@95%TPR, latency (ms, device-qualified), memory (peak RSS), sample efficiency, and cross-dataset transfer — this is the metric set that will finally produce a defensible version of HYDRA-LNN's original latency-comparison table.

**Baselines**
LSTM, CNN, HDC-only, and LNN-only, each trained with the identical split, preprocessing, and tuning budget as the hybrid model. This directly replaces the original, unmethodologied BERT/LSTM comparison row with baselines that are actually trained inside the same harness under the same rules.

> **ℹ Rule going forward**
> Any latency, throughput, or accuracy number presented outside this document must cite a `run_id` from Section 6's run artifacts. No comparison table gets rebuilt from memory or estimation the way Section 5.2 of the original HYDRA-LNN proposal was.

---

## 8. Threat Validation Scenarios

Retained from the original HYDRA-LNN proposal as target scenarios for the synthetic attack injection planned in Phase 3 (Section 10). They are illustrative narratives, not yet benchmark results — they become results once run through the Section 7 harness and logged as `runs/<run_id>/decisions.parquet` entries.

### Scenario A — Low-and-Slow DNS Tunneling

```
DNS Query 1 (t=0s) → DNS Query 2 (t=450s) → DNS Query 3 (t=1800s)
```

**Standard SIEM:** misses the pattern — falls outside a typical 5-minute sliding window.
**HYDRA-LNN v2:** the continuous-time core decays state smoothly across the full 1,800-second gap while preserving the trajectory's structure, so the connection remains detectable by the Divergence Engine.

### Scenario B — Lateral Movement via Pass-the-Hash

```
NTLM Auth (Host A) → SMB Exec (Host B) → LSASS Dump (Host B)
```

**Standard SIEM:** triggers isolated low-priority alerts with no aggregate context.
**HYDRA-LNN v2:** the HDC encoder binds host, method, and user identity together; the LNN Core tracks the sequence transition itself as a trajectory shift, and the Divergence Engine surfaces it as one coherent, high-severity `AnomalyDecision` instead of three disconnected alerts.

---

## 9. Mathematical Formulation

Retained from the HYDRA-LNN appendix, for engineering reviewers who want the underlying math. This is the derivation that `ncps`'s CfC implementation (Section 4.2) is expected to match under the batch/streaming parity rule.

**HDC event vector construction**

$$H_{event} = \text{sign}\left(\sum_{i=1}^{m} K_i \otimes V_i\right)$$

**Continuous-feature similarity (thermometer codebook)**

$$\langle L_a, L_b \rangle = 1 - \frac{2|a - b|}{N - 1}$$

**Liquid time-constant state dynamics**

$$\frac{dx(t)}{dt} = -\left[\frac{1}{\tau_x} + f(x(t), H(t); \theta)\right] \cdot x(t) + f(x(t), H(t); \theta) \cdot A$$

**Closed-form state update for arbitrary Δt**

$$x(t_k) = e^{-\Delta t(1/\tau_x + f_{k-1})} \cdot x(t_{k-1}) + \left(1 - e^{-\Delta t(1/\tau_x + f_{k-1})}\right) \cdot A$$

**Predictive anomaly score** (secondary / fast-path scorer, Section 4.3)

$$A(t_k) = 1 - \frac{H_{t_k} \cdot \hat{H}_{t_k}}{\|H_{t_k}\|_2 \cdot \|\hat{H}_{t_k}\|_2}$$

**Primary divergence score** (Mahalanobis, Section 4.3)

$$D_M(h_t) = \sqrt{(h_t - \mu)^T \Sigma^{-1} (h_t - \mu)}$$

where μ and Σ are the mean and covariance of the Reference Manifold, fit exclusively on the normal-traffic slice of the validation split (Section 5). This formula is new relative to both source documents' explicit appendices — it is the mathematical statement of what HDC-LNN's "Mahalanobis(h_t, Reference Manifold)" step actually computes.

---

## 10. Implementation Roadmap

The two source roadmaps are merged into one sequence. Research-correctness phases (adapted from HDC-LNN) come first and gate the production phases (adapted from HYDRA-LNN) — production hardening no longer starts in parallel with unvalidated algorithm work.

**Table 6. Unified roadmap — research phases (0–3, from HDC-LNN) gate production phases (4, from HYDRA-LNN)**

| Phase | Timeline | Focus | Exit criteria |
|---|---|---|---|
| **0** | Weeks 1–2 | Foundation & data pipeline (L1–L2): contracts, config, logging, seeds; loaders for UNSW-NB15 / CICIoT2023 / NSL-KDD; entity-disjoint splitter | Splits are versioned and committed; quality checks pass |
| **1** | Weeks 3–6 | HDC core: torchhd-based encoder, item memory fit on TRAIN only, OOV handling | >100,000 log vectorizations/sec; zero OOV leakage in val/test |
| **2** | Weeks 7–10 | LNN core & Divergence Engine: CfC via ncps; Reference Manifold fit on val-normal slice; both scorers implemented behind IDivergenceScorer | Batch/streaming numerical parity verified |
| **3** | Weeks 11–16 | Evaluation harness & validation: baselines (LSTM, CNN, HDC-only, LNN-only); synthetic attack injection (Section 8 scenarios); ablations | Full metrics suite produced with a citable run_id; this is what Section 7's numbers will come from |
| **4** | Weeks 17–22 | Production hardening: SIMD/AVX-512 backend, Kafka ring-buffer ingest, sidecar container, SOAR/SIEM connector | Streaming path benchmarked at 50,000 events/sec against Phase 3 accuracy numbers, not new ones |

---

## 11. Out of Scope (Phase 1–2)

Retained from HDC-LNN and extended slightly for consistency with the merged roadmap:

- Live packet capture
- Distributed training
- Adversarial defenses
- Production edge-deployment binary (this is explicitly the Phase 4 deliverable, not an earlier one)
- SOAR/SIEM connector work ahead of a Phase 3 exit — no production integration begins before the evaluation harness has produced a benchmarked model

---

## 12. Appendix — Data Model & Folder Layout

### Data Model Objects

| Object | Description |
|---|---|
| `CanonicalFlow` | One normalized flow record + label + split. |
| `EncodedHypervector` | Fixed-width HDC vector + encoder version + OOV flags. |
| `TrajectoryState` | LNN hidden state for one entity at one timestamp. |
| `AnomalyDecision` | Drift score, threshold, verdict, manifold version. |

### Folder Layout

```
hdlnn/
 ├── contracts/      (shared dataclasses + error types)
 ├── common/         (config, logging, seeding)
 ├── data/           (loaders, splitter, quality checks)
 ├── hdc/            (item memory, encoder, IEncoder)
 ├── lnn/            (CfC/LTC models, ISequenceModel)
 ├── divergence/     (manifold, scorer, IDivergenceScorer)
 ├── baselines/      (LSTM, CNN wrappers)
 ├── eval/           (harness, metrics, ablation configs)
 ├── report/         (tables, plots, dashboard)
 ├── deploy/         (SIMD backend, Kafka sidecar, SOAR connector)  ← new, Phase 4
 └── pipeline.py     (single orchestrator: train | eval | stream | ablate)
runs/<run_id>/       (config snapshot, git commit, metrics.json, decisions.parquet)
```

The `deploy/` directory is the only structural addition to HDC-LNN's original layout — it houses the Phase 4 production backend from HYDRA-LNN without touching the research core it depends on.

> **ℹ Document provenance**
> This proposal supersedes both "HYDRA-LNN — Hyper-Dimensional Real-Time Temporal Anomaly Engine" (AI & Cybersecurity Systems Engineering) and "HDC-LNN Architecture" (research specification). Where the two disagreed, the research document's rigor was preferred for algorithmic and evaluation design; the product document's framing was preferred for scope, narrative, and deployment target.
