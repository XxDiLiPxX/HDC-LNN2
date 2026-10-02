# Dataset Provenance, Schema & Acquisition Reference

This folder holds the network telemetry benchmark datasets used for training and evaluating the models. To provide immediate out-of-the-box reproducibility, benchmark datasets are pre-packaged directly in this repository.

---

## 1. Primary Benchmark Datasets

### A. UNSW-NB15
- **Primary Source**: University of New South Wales (UNSW) Cyber Research Group / Australian Centre for Cyber Security (ACCS).
- **Canonical Download**: [UNSW-NB15 Dataset Repository](https://research.unsw.edu.au/projects/unsw-nb15-dataset) or [Kaggle UNSW-NB15](https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15)
- **Key Files**:
  - `UNSW_NB15_training-set.csv` (175,341 flows)
  - `UNSW_NB15_testing-set.csv` (82,332 flows; primary benchmark evaluating 30,000 contiguous sequence flows)
  - `UNSW-NB15_1.csv` through `UNSW-NB15_4.csv` (raw CSV flow dumps)
- **Target Config**: `configs/dataset_unsw_nb15.yaml` and `configs/dataset_unsw_nb15_raw49.yaml`

### B. NSL-KDD & KDD Cup 99
- **Primary Source**: University of New Brunswick (UNB) Canadian Institute for Cybersecurity.
- **Canonical Download**: [UNB NSL-KDD Dataset Repository](https://www.unb.ca/cic/datasets/nsl.html) or [Kaggle NSL-KDD](https://www.kaggle.com/datasets/hassan06/nslkdd)
- **Key Files**:
  - `kdd_test.csv` (22,544 flows)
  - `kdd_train.csv` (125,973 flows)
  - `NSL_KDD_Test.csv` (22,543 flows)
  - `NSL_KDD_Train.csv` (125,973 flows)
  - `kddcup99.csv` (original 10% benchmark subset)
- **Target Config**: `configs/dataset_kdd.yaml`

### C. ToN-IoT Telemetry
- **Primary Source**: Cyber Range and IoT Labs, UNSW Canberra.
- **Canonical Download**: [UNSW Canberra ToN-IoT Repository](https://research.unsw.edu.au/projects/toniot-datasets)
- **Key Files**:
  - `ton-iot.csv`
- **Target Config**: `configs/dataset_ton_iot.yaml`

### D. Stratosphere IPS IoT-23
- **Primary Source**: Stratosphere Laboratory, Czech Technical University (CTU) in Prague.
- **Canonical Download**: [IoT-23 Dataset](https://www.stratosphereips.org/datasets-iot23)
- **Key Files**:
  - `dataset1.csv` through `dataset23.csv` (Zeek-formatted network telemetry flows capturing IoT malware and benign captures)
- **Target Config**: `configs/dataset_iot23.yaml`

### E. UNSW 2018 IoT Botnet Dataset
- **Primary Source**: University of New South Wales IoT Security Laboratory.
- **Canonical Download**: [UNSW IoT Botnet Dataset](https://research.unsw.edu.au/projects/botnet-datasets)
- **Key Files**:
  - `UNSW_2018_IoT_Botnet_Dataset_*.csv`
- **Target Config**: `configs/dataset_unsw_botnet.yaml`

---

## 2. Directory Usage & Verification
- All benchmark datasets reside in `datasets/` with standard CSV encoding.
- File headers and schemas match the field mappings defined in `configs/dataset_<name>.yaml`.
- Automated schema detection is provided by `run_all_datasets.py` (`get_config_for_dataset`).
