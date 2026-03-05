<<<<<<< HEAD
# NIDS-XAI — Network Intrusion Detection with Explainable AI
=======
# NIDS-XAI — Network Intrusion Detection System
>>>>>>> d8a984beaf6f7f7dafe575ba5d3734e7b4f242ea

[![Live Demo](https://img.shields.io/badge/Live%20Demo-nids--xai.streamlit.app-00e5b0?style=flat&logo=streamlit&logoColor=white)](https://nids-xai.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat&logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-orange?style=flat)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-0.46.0-purple?style=flat)](https://shap.readthedocs.io)
[![Dataset](https://img.shields.io/badge/Dataset-CICIDS2017-lightgrey?style=flat)](https://www.unb.ca/cic/datasets/ids-2017.html)

A machine learning system that classifies network traffic as **Benign** or **DDoS**, enhanced with **Explainable AI (SHAP)** to provide human-interpretable justifications for every prediction. Trained and evaluated on the CICIDS2017 benchmark from the Canadian Institute for Cybersecurity.

---

## Live Demo

**[nids-xai.streamlit.app](https://nids-xai.streamlit.app)**

Upload a CICIDS2017-format CSV or click **Load sample data** to run live inference with SHAP explanations.

---

## Results

| Metric | Value |
|---|---|
| F1-Score (Macro) | **0.9999** |
| Precision-Recall AUC | **1.0000** |
| ROC-AUC | **1.0000** |
| Precision (DDoS) | 0.9998 |
| Recall (DDoS) | 1.0000 |
| Test Set Size | 44,617 flows |
| Inference Time | < 100 ms / 100 flows |

---

## SHAP Feature Attribution

Top features by mean absolute SHAP value across the test set:

| Rank | Feature | Mean \|SHAP\| | Attribution |
|---|---|---|---|
| 1 | Fwd Packet Length Max | 3.6388 | 23.18% |
| 2 | Total Length of Fwd Packets | 1.8861 | 12.01% |
| 3 | Fwd Packet Length Mean | 1.5988 | 10.18% |
| 4 | Destination Port | 1.4602 | 9.30% |
| 5 | Bwd Packet Length Std | 0.6582 | 4.19% |

The top 4 features account for **54.7%** of all classification decisions.

---

## Project Structure

```
nids-xai/
<<<<<<< HEAD
├── app/
│   └── dashboard.py              # Streamlit Command Center dashboard
├── src/
│   ├── preprocessing/            # Data cleaning and scaling pipeline
│   ├── models/                   # XGBoost training scripts
│   └── explainability/           # SHAP analysis and plot generation
├── data/
│   ├── raw/                      # Original CICIDS2017 CSV files (not tracked)
│   └── processed/                # Preprocessed .parquet files
├── outputs/
│   ├── models/                   # Trained model artifacts (.pkl)
│   ├── figures/                  # SHAP plots, confusion matrices
│   └── reports/                  # Classification reports, SHAP rankings
├── requirements.txt
└── README.md
=======
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ raw/                  # Original CICIDS2017 CSV files (not tracked by Git)
â”‚   â””â”€â”€ processed/            # Cleaned, preprocessed .parquet files
â”œâ”€â”€ notebooks/                # Exploratory analysis and experiments
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ preprocessing/        # Data cleaning, SMOTE, scaling pipeline
â”‚   â”œâ”€â”€ models/               # Random Forest, XGBoost, LSTM training scripts
â”‚   â””â”€â”€ explainability/       # SHAP analysis and plot generation
â”œâ”€â”€ outputs/
â”‚   â”œâ”€â”€ figures/              # SHAP plots, confusion matrices, PR curves
â”‚   â”œâ”€â”€ models/               # Saved trained models (.pkl / .h5)
â”‚   â””â”€â”€ reports/              # Classification reports in CSV
â”œâ”€â”€ app/                      # Streamlit dashboard
â”œâ”€â”€ requirements.txt
â””â”€â”€ README.md
>>>>>>> d8a984beaf6f7f7dafe575ba5d3734e7b4f242ea
```

---

<<<<<<< HEAD
## How to Run Locally
=======
## Dataset
**CICIDS2017** â€” Canadian Institute for Cybersecurity Intrusion Detection Dataset 2017
- Download: https://www.unb.ca/cic/datasets/ids-2017.html
- Place CSV files in: `data/raw/`
- Recommended starting file: `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`
>>>>>>> d8a984beaf6f7f7dafe575ba5d3734e7b4f242ea

```bash
# 1. Clone the repo
git clone https://github.com/fnurani/nids-xai.git
cd nids-xai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the dashboard
streamlit run app/dashboard.py
```

Opens at `http://localhost:8501`

---

## Dataset

**CICIDS2017** — Canadian Institute for Cybersecurity Intrusion Detection System 2017

- **Source:** [unb.ca/cic/datasets/ids-2017.html](https://www.unb.ca/cic/datasets/ids-2017.html)
- **Subset used:** Friday — DDoS vs Benign traffic
- **Raw records:** 225,745 flows
- **After preprocessing:** 223,082 flows
- **Features:** 68 numeric network flow statistics

---

## Tech Stack

| Component | Technology |
|---|---|
| Classifier | XGBoost 2.0.3 |
| Explainability | SHAP 0.46.0 (TreeExplainer) |
| Dashboard | Streamlit 1.39.0 |
| Data processing | Pandas, NumPy, Scikit-learn |
| Visualisation | Matplotlib |
| Deployment | Streamlit Community Cloud |

---

## Author
<<<<<<< HEAD
=======
**Farhan Nurani** â€”  Final Year Student
- Project developed as a supplementary portfolio project alongside FYP: *A Hybrid Machine Learning Approach for DDoS Attack Detection in 5G Network Slicing*
>>>>>>> d8a984beaf6f7f7dafe575ba5d3734e7b4f242ea

**Farhan Nurani**
farhannurani02@gmail.com
