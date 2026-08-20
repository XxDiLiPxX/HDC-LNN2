# Dataset Acquisition Sources & Directory Setup

This folder holds the raw network telemetry benchmark datasets used for training and evaluating the models. Due to file size constraints (individual CSV files range up to ~150MB+), raw dataset files are strictly **excluded from version control (`.gitignore`)** and must be acquired locally.

---

## 1. Primary Benchmark Datasets

### A. UNSW-NB15
- **Primary Source**: University of New South Wales (UNSW) Cyber Research Group / Australian Centre for Cyber Security (ACCS).
- **Canonical Download**: [UNSW-NB15 Dataset Repository](https://research.unsw.edu.au/projects/unsw-nb15-dataset) or [Kaggle UNSW-NB15](https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15)
- **Required Files**:
  - `UNSW_NB15_training-set.csv` (175,341 flows)
  - `UNSW_NB15_testing-set.csv` (82,332 flows, primary benchmark evaluating 30,000 contiguous sequence flows)
- **Target Config**: `configs/dataset_unsw_nb15.yaml`

### B. NSL-KDD (KDD Cup 99 Improved)
- **Primary Source**: University of New Brunswick (UNB) Canadian Institute for Cybersecurity.
- **Canonical Download**: [UNB NSL-KDD Dataset Repository](https://www.unb.ca/cic/datasets/nsl.html) or [Kaggle NSL-KDD](https://www.kaggle.com/datasets/hassan06/nslkdd)
- **Required Files**:
  - `kdd_test.csv` (22,544 flows)
  - `kdd_train.csv` (125,973 flows)
- **Target Config**: `configs/dataset_kdd.yaml`

### C. ToN-IoT Telemetry
- **Primary Source**: Cyber Range and IoT Labs, UNSW Canberra.
- **Canonical Download**: [UNSW Canberra ToN-IoT Repository](https://research.unsw.edu.au/projects/toniot-datasets)
- **Required Files**:
  - `ton-iot.csv` / `Network_dataset.csv`
- **Target Config**: `configs/dataset_ton_iot.yaml`

### D. UNSW 2018 IoT Botnet Dataset
- **Primary Source**: University of New South Wales IoT Security Laboratory.
- **Canonical Download**: [UNSW IoT Botnet Dataset](https://research.unsw.edu.au/projects/botnet-datasets)
- **Target Config**: `configs/dataset_unsw_botnet.yaml`

---

## 2. Directory Hygiene & Verification
- Place downloaded CSV files directly inside this `datasets/` directory.
- Verify file naming matches the `download_urls` and `columns` schema defined in `configs/dataset_<name>.yaml`.
- All `.csv` files inside `datasets/` are ignored by git to keep repository clones lightweight and reproducible.
