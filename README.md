
# NIDS-XAI — Network Intrusion Detection with Explainable AI

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
├── app/
│   └── dashboard.py
├── src/
│   ├── preprocessing/
│   ├── models/
│   └── explainability/
├── data/
│   ├── raw/
│   └── processed/
├── outputs/
│   ├── models/
│   ├── figures/
│   └── reports/
├── requirements.txt
└── README.md
```

---

## How to Run Locally
```bash
git clone https://github.com/fnurani/nids-xai.git
cd nids-xai
pip install -r requirements.txt
streamlit run app/dashboard.py
```

---

## Dataset

**CICIDS2017** — Canadian Institute for Cybersecurity Intrusion Detection System 2017

- **Source:** [unb.ca/cic/datasets/ids-2017.html](https://www.unb.ca/cic/datasets/ids-2017.html)
- **Subset:** Friday DDoS vs Benign
- **Raw records:** 225,745 flows
- **After preprocessing:** 223,082 flows
- **Features:** 68 numeric flow statistics

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

**Farhan Nurani**
farhannurani02@gmail.com
'@ | Set-Content README.md -Encoding UTF8
