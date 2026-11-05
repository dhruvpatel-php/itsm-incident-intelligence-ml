# ABC Tech — ITSM Predictive Intelligence Platform (PR-0012)
### Senior Data Science Capstone Project | ITIL V3 Process Optimization Suite

[![Python](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9.1-orange.svg)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.7.0-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.4.1-red.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66-FF4B4B.svg)](https://streamlit.io/)

---

## 📌 Executive Summary & Business Case

**Client**: ABC Tech  
**Category**: IT Service Management (ITSM) Machine Learning  
**Project Reference**: PM-PR-0012  

**ABC Tech** is a mid-sized organization operating in the IT-enabled services sector for over a decade. Handling **22,000 to 25,000 IT incidents/tickets annually**, ABC Tech implemented the ITIL framework covering Incident Management, Problem Management, Change Management, and Configuration Management. 

Despite achieving an audited mature ITIL process level, recent customer surveys revealed that **incident management was rated as poor**. Internal audits confirmed that further traditional process improvement initiatives yield diminishing returns on investment.

This project delivers an **end-to-end Machine Learning suite** addressing the **4 key business areas** identified by ABC Tech leadership:

1. **Predicting High Priority Tickets (Pillar 1)**: Early detection of critical Priority 1 & 2 incidents at inception to trigger preventive intervention and eliminate triage delay.
2. **Forecasting Incident Volume (Pillar 2)**: Quarterly and annual forecasting across diverse IT categories to enable data-driven resource allocation and capacity planning.
3. **Auto-Tagging Tickets & SLA Optimization (Pillar 3)**: Automated classification into the correct department and priority level, reducing multi-hop reassignments by up to **69.4%** and cutting resolution delay by **53.3%**.
4. **Predicting Requests for Change & Asset Misconfigurations (Pillar 4)**: Detecting change-inducing incidents (RFCs) and establishing an **Asset Risk Index (ARI)** across 3,019 Configuration Items (CIs).

---

## 🏆 Key Performance Highlights

| Pillar | Business Objective | Champion Model | Key Metrics | Business Impact |
|---|---|---|---|---|
| **Pillar 1** | Early Warning for P1/P2 Incidents | **LightGBM (Weighted)** | **ROC-AUC: 0.9416**<br>**PR-AUC: 0.5660**<br>**Recall: 74%** | **36.5x lift** over random baseline; captures 74% of critical outages before triage delay. |
| **Pillar 2** | Quarterly & Annual Volume Forecasting | **SARIMA + Holt-Winters Ensemble** | **MAPE: 11.0% - 12.1%**<br>**MAE: 184 - 208** | **2014 Volume: 80,021 tickets**<br>**2015 Capacity: 84,022 tickets**; establishes staffing quotas. |
| **Pillar 3** | Automated Ticket Tagging & Routing | **LightGBM Multi-Class** | **Routing Acc: 69.4%**<br>**Priority Acc: 76.6%** | Eliminates **69.4%** of misrouted hops; cuts MTTR delay by **53.3%** on misrouted tickets. |
| **Pillar 4** | RFC Prediction & Asset Risk Profiling | **LightGBM + Composite ARI** | **RFC ROC-AUC: 0.8857**<br>**3,019 Assets Ranked** | Proactively identifies top high-risk CIs (e.g., `SUB000544`, `WSR000008`) for preventive patching. |

---

## 📂 Project Architecture & Directory Structure

```
PRCL-0012/
├── app.py                                  # Interactive Streamlit Web Application
├── requirements.txt                        # Production Python dependencies
├── README.md                               # Project documentation and architectural guide
├── REPORT.md                               # Comprehensive Technical & Client Deliverable Report
├── data/
│   ├── raw/
│   │   └── dataset_list.csv                # 46,606 raw incident records queried from MySQL
│   └── processed/
│       ├── cleaned_itsm.csv                # Cleaned, standardized, leakage-free dataset
│       ├── daily_incident_volume.csv       # Resampled daily volume series (455 days)
│       └── quarterly_incident_volume.csv   # Historical quarterly volume segmented by field
├── figures/
│   ├── 01_priority_distribution.png        # Ticket priority distribution (P1 - P5)
│   ├── 02_reassignments_vs_handletime.png  # Reassignments vs median handling duration
│   ├── 03_incident_trend_monthly.png       # Monthly incident inflow trend (2012-2014)
│   ├── 04_department_breakdown.png         # Distribution across IT service departments
│   ├── 04b_closure_codes_root_causes.png   # Root causes and misconfiguration breakdown
│   ├── 05_pillar1_roc_pr_curves.png        # Pillar 1 ROC and PR curves
│   ├── 06_pillar1_feature_importance.png   # Pillar 1 Top 15 predictive features
│   ├── 07_pillar2_volume_forecast.png      # Pillar 2 Weekly forecast with 95% CI bands
│   ├── 08_pillar2_ci_forecasts.png         # Pillar 2 Segmented quarterly forecast
│   ├── 09_pillar3_department_confusion_matrix.png # Pillar 3 Department routing confusion matrix
│   ├── 10_pillar3_priority_confusion_matrix.png   # Pillar 3 Priority classification matrix
│   ├── 11_pillar4_rfc_roc_curve.png        # Pillar 4 RFC detection ROC and PR curves
│   └── 12_pillar4_ci_asset_risk_matrix.png # Pillar 4 CI Asset Risk Matrix (Volume vs Risk)
├── models/
│   ├── pillar1_high_priority_model.joblib  # Trained champion model & encoders for Pillar 1
│   ├── pillar1_threshold.json              # Calibrated optimal decision threshold (0.760)
│   ├── pillar2_volume_forecaster.joblib    # Fitted time-series forecaster models & projections
│   ├── pillar3_department_router.joblib    # Fitted department routing classifier
│   ├── pillar3_priority_classifier.joblib  # Fitted multi-class priority classifier
│   ├── pillar4_rfc_predictor.joblib        # Fitted RFC binary predictor
│   └── pillar4_asset_risk_scores.csv       # Profiled risk registry for all 3,019 IT assets
├── notebooks/
│   └── ITSM_Machine_Learning_Capstone.ipynb# Fully documented end-to-end Jupyter Notebook
└── src/
    ├── __init__.py
    ├── config.py                           # Paths, random seeds, and global configurations
    ├── data_loader.py                      # Data ingestion, handle time parser, priority matrix
    ├── eda.py                              # Exploratory data analysis & figure generator
    ├── feature_engineering.py              # Leakage-free temporal & categorical encoders
    ├── model_pillar1_priority.py           # High-priority classification & threshold tuning
    ├── model_pillar2_forecasting.py        # Time-series forecasting (SARIMA / Holt-Winters)
    ├── model_pillar3_routing.py            # Department routing & reassignment simulation
    ├── model_pillar4_rfc_assets.py         # RFC prediction & Configuration Item Risk Index
    └── run_all.py                          # Master end-to-end execution pipeline
```

---

## 🚀 Quickstart Guide

### 1. Environment Installation
Ensure Python 3.10+ is installed. Clone or navigate to the repository directory and install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Run the Complete End-to-End Pipeline
Execute the full pipeline with a single command to perform data ingestion, cleaning, EDA visualization generation, model training for all 4 pillars, and artifact persistence:

```bash
python -m src.run_all
```

### 3. Launch the Interactive Web Dashboard
Run the Streamlit application to explore the executive KPI dashboard, test real-time ticket priority risk assessment, inspect quarterly volume projections, and review the CI Asset Risk Registry:

```bash
streamlit run app.py
```

### 4. Run the Jupyter Notebook
Open and explore the comprehensive, self-contained capstone notebook:

```bash
jupyter notebook notebooks/ITSM_Machine_Learning_Capstone.ipynb
```

---

## 🔍 Methodological Insights & Domain Solutions

### 1. Handle Time Anomaly Reconstruction
The raw `Handle_Time_hrs` contained values such as `'3,87,16,91,111'`. Analysis revealed that exporting European decimals (where `,` denotes decimal separator) on systems with Indian regional formatting stripped decimals and added comma groupings. The last 6 digits represent micro-fractions of an hour ($3600 \text{ seconds} \approx 6 \text{ decimal places}$). The parser successfully reconstructed `3871.691111` hours with 100% precision.

### 2. Deconstructing the ITIL Priority Matrix
Inspection revealed that when `Impact` was `'NS'` (Not Specified), `Priority` was left as `NaN` (1,380 tickets). For all valid pairs of `Impact` and `Urgency`, the official ITIL matrix matched with **100.00% consistency**. This uncovered a critical ITIL vulnerability: when users omit Impact, tickets sit unprioritized. Our ML model predicts Priority directly from ticket metadata, resolving this bottleneck.

### 3. Severe Class Imbalance (1.55% Positive Class)
High Priority tickets (P1/P2) account for only 700 out of 45,226 tickets. Standard accuracy (98.5%) would simply predict all tickets as low priority. We resolved this via:
- Stratified 80/20 train/test splitting.
- Cost-sensitive weighting (`scale_pos_weight = 64.6`).
- PR-AUC evaluation (**0.5660** vs **0.0155** baseline).
- Decision threshold calibration optimizing $F_2$-Score (`threshold = 0.760`), achieving **74% recall** on critical outages.

### 4. Reassignment Escalation Impact
Tickets with 0 reassignments exhibit a median handling duration of **1,331 hours**, whereas reassigned tickets exhibit a median handling duration of **2,851 hours** — an escalation of **+1,520 hours (+114%)**. Auto-routing achieves **69.4% first-touch precision**, cutting misrouted tickets and reducing resolution delays by **53.3%**.

---

## 📊 Client Deliverables
- Detailed Executive & Technical Report: [`REPORT.md`](file:///c:/Users/hp/Downloads/PRCL-0012/REPORT.md)
- Jupyter Notebook: [`notebooks/ITSM_Machine_Learning_Capstone.ipynb`](file:///c:/Users/hp/Downloads/PRCL-0012/notebooks/ITSM_Machine_Learning_Capstone.ipynb)
- Streamlit Web Application: [`app.py`](file:///c:/Users/hp/Downloads/PRCL-0012/app.py)
- Configuration Item Risk Registry: [`models/pillar4_asset_risk_scores.csv`](file:///c:/Users/hp/Downloads/PRCL-0012/models/pillar4_asset_risk_scores.csv)
