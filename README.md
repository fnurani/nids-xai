# ??? NIDS-XAI � Network Intrusion Detection System

[![Live Demo](https://img.shields.io/badge/Live%20Demo-nids--xai.streamlit.app-00e5b0?style=flat&logo=streamlit)](https://nids-xai.streamlit.app)

# Network Intrusion Detection System with Explainable AI (NIDS-XAI)

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4-orange)
![SHAP](https://img.shields.io/badge/XAI-SHAP-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## Overview
A machine learning-based Network Intrusion Detection System (NIDS) that classifies network traffic as benign or malicious, enhanced with **Explainable AI (SHAP)** to provide human-interpretable explanations for each prediction.

Trained and evaluated on the **CICIDS2017** dataset from the Canadian Institute for Cybersecurity.

---

## Project Structure
```
nids-xai/
├── data/
│   ├── raw/                  # Original CICIDS2017 CSV files (not tracked by Git)
│   └── processed/            # Cleaned, preprocessed .parquet files
├── notebooks/                # Exploratory analysis and experiments
├── src/
│   ├── preprocessing/        # Data cleaning, SMOTE, scaling pipeline
│   ├── models/               # Random Forest, XGBoost, LSTM training scripts
│   └── explainability/       # SHAP analysis and plot generation
├── outputs/
│   ├── figures/              # SHAP plots, confusion matrices, PR curves
│   ├── models/               # Saved trained models (.pkl / .h5)
│   └── reports/              # Classification reports in CSV
├── app/                      # Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## Dataset
**CICIDS2017** — Canadian Institute for Cybersecurity Intrusion Detection Dataset 2017
- Download: https://www.unb.ca/cic/datasets/ids-2017.html
- Place CSV files in: `data/raw/`
- Recommended starting file: `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`

---

## Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/YOUR_USERNAME/nids-xai.git
cd nids-xai
pip install -r requirements.txt
```

### 2. Run EDA
```bash
jupyter notebook notebooks/01_eda.ipynb
# or
python src/preprocessing/eda.py --input data/raw/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
```

### 3. Preprocess Data
```bash
python src/preprocessing/preprocess.py
```

### 4. Train Models
```bash
python src/models/train.py
```

### 5. Run SHAP Analysis
```bash
python src/explainability/shap_analysis.py
```

### 6. Launch Dashboard
```bash
streamlit run app/dashboard.py
```

---

## Results
| Model | Accuracy | F1-Score (Macro) | PR-AUC |
|-------|----------|------------------|--------|
| Random Forest | TBD | TBD | TBD |
| XGBoost | TBD | TBD | TBD |
| LSTM | TBD | TBD | TBD |

*Results will be updated upon training completion.*

---

## Tech Stack
- **ML:** Scikit-learn, XGBoost, TensorFlow
- **XAI:** SHAP
- **Imbalance Handling:** SMOTE (imbalanced-learn)
- **Dashboard:** Streamlit
- **Data I/O:** Pandas, PyArrow (Parquet)

---

## Author
**Farhan Nurani** —  Final Year Student
- Project developed as a supplementary portfolio project alongside FYP: *A Hybrid Machine Learning Approach for DDoS Attack Detection in 5G Network Slicing*

