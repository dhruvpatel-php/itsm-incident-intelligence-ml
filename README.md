# ABC Tech — ITSM Predictive Intelligence Platform (PR-0012)
### Senior Data Science Capstone Project | ITIL V3 Process Optimization Suite

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9.1-orange.svg)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.7.0-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![Statsmodels](https://img.shields.io/badge/Statsmodels-0.15.0-purple.svg)](https://www.statsmodels.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

---

## 📌 Executive Summary & Deliverables

**Client**: ABC Tech  
**Category**: IT Service Management (ITSM) Machine Learning  
**Project Reference**: PM-PR-0012  
**Final Deliverable**: [`PRCL-0012_ITSM_Machine_Learning.ipynb`](PRCL-0012_ITSM_Machine_Learning.ipynb)

This repository contains the finalized, audited, and self-contained machine learning capstone deliverable for **ABC Tech**. The suite addresses ITIL incident management optimization across four core business pillars, executed strictly with leak-free temporal validation, computed empirical metrics, and rigorous governance.

---

## 🏆 Key Performance Highlights (Audited & Verified)

| Pillar | Business Objective | Production Model | Key Verified Metrics | Operational Impact |
|---|---|---|---|---|
| **Pillar 1** | Early Warning for P1/P2 Incidents | **LightGBM (Cost-Weighted)** | **PR-AUC: 0.2615** [95% CI: 0.1989, 0.3503]<br>**ROC-AUC: 0.9148**<br>**Recall: 50.93% @ Thresh 0.713** | **21.4x lift** over 1.224% prevalence; captures 51% of critical outages at intake. Calibrated Brier score: 0.01014 (-16.3% error). |
| **Pillar 2** | Incident Volume Forecasting | **Ensemble / SARIMAX / Naive** | **6-Wk Rolling CV MAPE: 12.65%**<br>**Naive MAPE: 12.12%** | **2014 Base Volume: 72,714 tickets** (Q1 actual 21,979 + 39-wk forecast 50,735). Block Bootstrap 95% PI: [67,163, 76,519]. |
| **Pillar 3 (Priority)** | Deterministic ITIL Matrix & Imputation | **Exact Matrix Lookup + Imputer** | **Matrix Match: 100.00%**<br>**1,380 NS Imputed: 78.8% P3 / 21.2% P4** | Priority is a deterministic 2D lookup (ML redundant). Imputed P3 range: [9.9%, 96.7%] due to covariate shift. |
| **Pillar 3 (Routing)** | Automated Department Routing | **Variant D (Cat + WBS + Timing)** | **Accuracy: 80.66% (59.42% Bal Acc)**<br>**Balanced: 70.37% (80.57% Bal Acc)** | **+11.89% lift** over 68.77% majority baseline using intake-safe features. Saves **514 hrs/yr (0.26 FTEs)**. |
| **Pillar 4** | Proactive RFC & Asset Risk Profiling | **Bayesian Asset Risk Index (ARI)** | **RFC Signal: AUC 0.5364 (Negative Result)**<br>**ARI Rank Stability: rho >= 0.8213** | Proves scientific integrity: tabular metadata lacks intake RFC signal once leakage is dropped. Profiles 287 CIs by risk tier. |

---

## 📂 Deliverable Package Contents

```
PRCL-0012/
├── PRCL-0012_ITSM_Machine_Learning.ipynb           # Primary Capstone Deliverable (40 cells, fully executed)
├── dataset_list.csv                                # Raw 46,606 ITSM incident records queried from MySQL
├── requirements.txt                                # Certified Python dependencies
├── README.md                                       # Project guide & architecture overview
├── CHANGELOG.md                                    # Comprehensive audit log of all 4-pillar improvements
├── CHEAT_SHEET.md                                  # Defense guide with exact empirical metrics
├── build_and_execute_v3.py                         # Master deterministic reproduction script
├── figures/                                        # Generated high-resolution visualization figures
├── models/                                         # Serialized production models & asset risk profiles
├── run.bat / run.ps1                               # One-click execution helpers for Streamlit & Notebook
└── .gitignore                                      # Clean version-control exclusions
```

---

## 🚀 Quickstart Guide

### 1. Environment Setup
Install the verified dependencies:

```bash
pip install -r requirements.txt
```

### 2. Run the Final Deliverable Notebook
Open and run all cells in [`PRCL-0012_ITSM_Machine_Learning.ipynb`](PRCL-0012_ITSM_Machine_Learning.ipynb):

```bash
jupyter notebook PRCL-0012_ITSM_Machine_Learning.ipynb
```
Select **Kernel -> Restart & Run All**. The notebook runs from top to bottom in under 2 minutes and passes all 19 hard runtime assertions and static governance checks in Cell 39.

### 3. Launch Local Streamlit Application
To interact with the ITSM Predictive Intelligence dashboard:

```bash
streamlit run app.py
```
Or execute `./run.bat` (Windows command line) / `./run.ps1` (PowerShell).

### 4. Programmatic Notebook Reproduction (Optional)
To programmatically regenerate and execute the notebook from scratch with fresh kernel outputs:

```bash
python build_and_execute_v3.py
```

---

## 🔍 Core Methodological Fixes

1. **Handle_Time Anomaly**: Decoded active engineering handle time (median: **1.95 hours**) vs elapsed calendar duration (median: **18.40 hours**). Documented that 21.7% of tickets have handle time exceeding calendar duration ($r = 0.1168$), reflecting asynchronous multi-engineer logging.
2. **ITIL Priority Matrix**: Reverse-engineered the 100% deterministic Impact x Urgency matrix, proving ML priority classification is redundant. For 1,380 tickets lacking Impact, documented severe covariate shift (97.7% are Urgency 3 vs 11.5% in known tickets) and reported the full bootstrap percentile range ([9.9%, 96.7%] P3).
3. **Pillar 1 Early Warning**: Strictly evaluated on a 20% prospective temporal holdout (8,824 tickets). Achieved **21.4x PR-AUC lift** (0.2615) over chance without looking at manual triage fields. Out-of-sample permutation importance proved `WBS_Freq` and `CI_Name_Freq` dominate escalation.
4. **Pillar 2 Volume Forecasting**: Corrected structural break by training strictly on the stable enterprise regime (Oct 2013 on). Combined Q1 2014 actual volume (21,979 tickets) with 39-week forecast (50,735 tickets) to establish a 2014 full-year base of **72,714 tickets** bounded by moving-block bootstrap (95% PI: [67,163, 76,519]).
5. **Pillar 3 Routing**: Removed data leakage from post-triage fields (`CI_Subcat`, `CI_Name`, constant `Alert_Status`). Established deployable Variant D benchmark (**80.66% accuracy**, **+11.89% lift** over 68.77% majority baseline) based on intake WBS billing code frequency (74.6% of WBS map to a single department).
6. **Data-Linked Econometric ROI**: Controlled for incident complexity via statsmodels OLS regression, confirming each reassignment hop causes **+7.41% handle delay** (0.14 hours, $p = 1.22 \times 10^{-200}$). Deployable Variant D saves **514 engineering hours/year (0.26 FTEs)**.
7. **Pillar 4 Honest Negative Result & ARI**: Removed circular post-investigation leakage (`Related_Change`, `No_of_Related_Changes`). Proved that intake metadata lacks predictive signal for RFCs (ROC-AUC 0.5364). Built a Bayesian-shrunk Asset Risk Index ($N \ge 10$) with rank order stability $\rho \ge 0.8213$ across 4 weighting schemes.
8. **Automated Governance Audit**: Dual-layer verification in Cell 39: 19 runtime assertions ensuring 100% claim-to-variable match, plus a strict AST/regex static scan guaranteeing 0 typed numbers in markdown and 0 unwhitelisted numeric literals in print statements.
