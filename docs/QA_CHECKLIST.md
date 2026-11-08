# QA Checklist: PM-PR-0012 ITSM Machine Learning Pipeline

## Overview
This document provides the formal quality assurance and engineering verification checklist for the ABC Tech ITSM Predictive Intelligence Platform (`PRCL-0012`). The assessment reviews all components against engineering standards, reproducibility gates, data leakage guards, and production delivery criteria.

---

## Detailed QA Checklist

### 1. Data Loader & Preprocessing

| Criteria | Status | Evidence & Verification |
|---|:---:|---|
| **Indian grouping comma handled** | **PASS** | `src/data_loader.py:27-46` parses the first comma as decimal separator and strips trailing commas (`3,87,16,91,111` $\to$ `3.8716911`). Verified in `tests/test_data_loader.py::test_parse_handle_time_indian_format_exact`. |
| **Clean handling of NaN / empty / 'nan'** | **PASS** | `src/data_loader.py:30-34` checks `pd.isna()`, strips whitespace, and catches string `"nan"` returning `np.nan` without warnings. Verified in `tests/test_data_loader.py::test_parse_handle_time_edge_cases`. |
| **Handle Time reproduction metrics match** | **PASS** | Verified on `dataset_list.csv` (46,606 rows): Active Handle Time median = **1.95 h** (1.9511h), mean = **4.25 h** (4.2461h), p75 = **4.37 h** (4.3689h), handle > calendar rate = **21.7%** (21.70%). Verified in `src/verify_metrics.py` and `tests/test_data_loader.py`. |
| **Handle Time vs Calendar duration correlations** | **PASS** | Pearson correlation $r = 0.1168$, Spearman rank correlation $\rho = 0.5684$. Verified in `src/verify_metrics.py` (0 error delta). |
| **Calendar duration median matches** | **PASS** | Calendar duration (`Close_DT - Open_DT`) median = **18.40 h**, mean = 123.13 h. Verified in `reports/metrics.json:calendar_duration_median_hours`. |

---

### 2. Pillar 1: High-Priority Incident Early Warning

| Criteria | Status | Evidence & Verification |
|---|:---:|---|
| **Strictly chronological temporal split** | **PASS** | `src/model_pillar1_priority.py:51-63`: Stable regime (from 2013-10-01) sorted by `Open_DT`. Train ($n=28,672$, 65%), Val ($n=6,616$, 15%), Test ($n=8,824$, 20%). Timestamps strictly ordered: $\max(Train) \le \min(Val) \le \min(Test)$. Verified in `tests/test_splits.py::test_pillar1_splits_strictly_chronological`. |
| **Validation-only threshold selection** | **PASS** | `src/model_pillar1_priority.py:115-135`: Optimal $F_2$ decision threshold $\tau^* = 0.7130$ tuned exclusively on validation fold to prioritize outage recall over precision. Evaluated exactly once on test holdout. |
| **Zero post-triage data leakage** | **PASS** | `src/model_pillar1_priority.py:64-77`: Strictly drops post-triage fields (`KB_number`, `Status`, `Closure_Code`, `Impact`, `Urgency`). `CI_Name_Freq` is a legitimate train-only frequency feature computed strictly on the training partition (unseen CIs mapped to 0 count). Guarded in `tests/test_leakage.py::test_pillar1_no_post_triage_leakage`. |
| **PR-AUC with 95% bootstrap CI reproduced** | **PASS** | Test Holdout ROC-AUC = **0.9148**, PR-AUC = **0.2615** with 1,000-fold moving block bootstrap 95% CI of **[0.1989, 0.3503]**. Recall @ $\tau^*=0.713$ is **50.93%** [95% CI: 42.20%, 60.36%]. Statistical lift is **21.4x** over prevalence baseline (1.224%). |
| **Seen vs Unseen asset breakdown reported** | **PASS** | `src/model_pillar1_priority.py:146-170`: Seen CI assets ($n=8,245$, 93.44% of test): prevalence = 0.800%, PR-AUC = 0.2254, recall = 34.85%. Unseen CI assets ($n=579$, 6.56% of test): prevalence = **7.254%** (9x higher outage risk!), PR-AUC = **0.3317**, recall = **76.19%**. |

---

### 3. Pillar 2: Incident Volume Forecasting

| Criteria | Status | Evidence & Verification |
|---|:---:|---|
| **Stable regime isolation (post-Oct-2013)** | **PASS** | `src/model_pillar2_forecasting.py:35-45`: Isolates the 26 weekly observations from 2013-10-01 to 2014-03-31, discarding the 39 pilot weeks before October 2013 where incident capture was incomplete. Verified in `tests/test_forecasting.py::test_forecasting_regime_and_horizon`. |
| **Naive persistence baseline included & competitive** | **PASS** | `src/model_pillar2_forecasting.py:75-95`: Multi-step rolling-origin backtest includes Naive persistence baseline, achieving **12.12% MAPE** (beats SARIMAX 13.17%, Holt-Winters 15.77%, and Seasonal Naive 20.17%). Ensemble achieves 12.65% MAPE. Verified in `tests/test_forecasting.py`. |
| **Shared rolling-origin backtest** | **PASS** | Identical 6-fold rolling backtest ($H=2$ weeks horizon) evaluated systematically across all candidate models. |
| **Moving-block bootstrap prediction interval** | **PASS** | `src/model_pillar2_forecasting.py:130-170`: 1,000-iteration moving-block bootstrap ($L=4$) generates honest 95% prediction intervals for 2014 total volume: $67,163 \le 72,714 \le 76,519$ ($Low < Base < High$). Verified in `tests/test_forecasting.py::test_prediction_interval_invariance`. |
| **Zero hardcoded volume numbers** | **PASS** | All forecasts, bounds, scenarios, and capacity requirements are computed directly from time series models. Verified 0 occurrences of hardcoded constants in `src/` and `app.py`. |

---

### 4. Pillar 3: ITIL Priority Resolution & Department Routing

| Criteria | Status | Evidence & Verification |
|---|:---:|---|
| **Deterministic ITIL matrix 100.00% match** | **PASS** | `src/model_pillar3_routing.py:80-95`: Validated across all 45,226 tickets with explicit Impact $\times$ Urgency mapping. Match rate = **100.00%**. Verified in `tests/test_data_loader.py::test_data_contract_and_handle_time`. |
| **Impact imputation on 1,380 NS tickets only with bootstrap range** | **PASS** | `src/model_pillar3_routing.py:115-155`: LightGBM classifier (5-fold CV accuracy 98.51%, macro-F1 0.8479) imputes Impact for the 1,380 `"NS"` tickets. Resulting distribution: **78.8% Priority 3** / **21.2% Priority 4**; 20-bagging bootstrap range: **[9.9%, 96.7%]** for P3. Identified severe covariate shift (97.7% Urgency 3 in NS vs 11.5% in known). |
| **Variant D routing evaluated vs majority baseline** | **PASS** | `src/model_pillar3_routing.py:200-245`: Variant D (Category + WBS Frequency + Timing arrival cues) achieves **80.66% unweighted accuracy** (**+11.89% lift** over 68.77% majority baseline) and **70.37% balanced accuracy** (80.57% balanced recall). |
| **Econometric delay penalty from regression** | **PASS** | `src/model_pillar3_routing.py:265-310`: Controlled log-linear regression ($\ln(\text{Handle\_Time}) \sim \text{Reassignment\_Count} + \text{Controls}$) with HC3 heteroskedasticity-robust errors yields bounce penalty $\beta = 0.0715$ ($p = 1.22 \times 10^{-200}$), equivalent to a **+7.41% delay penalty** or **+0.14 hours saved per bounce**. |
| **ROI conversion sensitivity analysis** | **PASS** | `src/model_pillar3_routing.py:315-345`: Annualized savings based on base volume 72,714 tickets: 25% realization = **128 hours (0.06 FTE)**, 50% realization = **257 hours (0.13 FTE)**, 100% realization = **514 hours (0.26 FTE)**. Full sensitivity range: 217 h to 1,227 h. |

---

### 5. Pillar 4: RFC Prediction & ITSM Asset Risk Profiling

| Criteria | Status | Evidence & Verification |
|---|:---:|---|
| **Honest negative result for proactive RFC prediction** | **PASS** | `src/model_pillar4_rfc_assets.py:60-120`: Prospective test holdout achieves **ROC-AUC = 0.5364** and **PR-AUC = 0.0137** (near random chance baseline of 1.0%), reported transparently as a scientific negative result proving that ticket metadata at triage does not indicate change requirement. |
| **No month or post-investigation leakage** | **PASS** | Excludes `Related_Change`, `No_of_Related_Changes`, and calendar month. Verified in `tests/test_leakage.py::test_pillar4_rfc_leakage_guards`. |
| **Empirical Bayes shrinkage on CIs ($N \ge 10$)** | **PASS** | `src/model_pillar4_rfc_assets.py:140-180`: Empirical Bayes shrinkage ($m=10.0$) applied to failure rate, high-priority rate, and reassignment rate across 287 qualifying assets with $N \ge 10$ incidents. Effectively eliminates single-incident outlier noise. |
| **Rank-stability check across weighting schemes** | **PASS** | `src/model_pillar4_rfc_assets.py:205-230`: Evaluated across 4 weighting profiles (Balanced, Outage-Focused, Operational Flapping, Infrastructure Reliability). Minimum pairwise Spearman $\rho = \mathbf{0.8213} > 0.80$, proving robust asset prioritization. |
| **43 Critical assets identified with top asset SUB000508** | **PASS** | Risk tiers categorized into 43 Critical (top 15%), 43 High (next 15%), 58 Medium (next 20%), and 143 Low (remaining 50%). Top asset is `SUB000508` (ARI_Final = 92.4, Risk Tier: Critical). Stored in `models/pillar4_asset_risk_scores.csv`. |

---

### 6. Code Quality & Tooling

| Criteria | Status | Evidence & Verification |
|---|:---:|---|
| **pytest passes clean** | **PASS** | `pytest -q`: All tests pass clean across test modules in `tests/`. |
| **ruff check passes clean** | **PASS** | `ruff check .`: All checks passed with 0 errors. |
| **tasks.ps1 implements all required commands** | **PASS** | `scripts/tasks.ps1` supports `setup`, `run`, `test`, `app`, `notebook`, `report`, `lint`, and `hashes`. |
| **pyproject.toml present with proper config** | **PASS** | Contains PEP 518/621 project metadata, pytest testpaths and options, and ruff lint configuration. |
| **CI workflow present** | **PASS** | `.github/workflows/ci.yml` configures automated linting (`ruff`) and testing (`pytest`) on Python 3.11 and 3.12. |

---

### 7. Single Source of Truth & Governance

| Criteria | Status | Evidence & Verification |
|---|:---:|---|
| **metrics.json matches metrics_reference.json (100% PASS)** | **PASS** | `python -m src.verify_metrics`: All metrics PASS with 0 failures and 0 warnings. |
| **build_report.py generates REPORT.md with zero manual typing** | **PASS** | `scripts/build_report.py` programmatically compiles `REPORT.md`, updates `README.md`, and builds `models/MODEL_CARD.md` directly from `reports/metrics.json`. |
| **models/MODEL_CARD.md present with all required sections** | **PASS** | Comprehensive model cards covering all 4 pillars with intended use, training data, temporal splits, audited metrics with 95% CIs, limitations, and failure modes. |
| **reports/artifact_hashes.json records hashes of all models & data** | **PASS** | `reports/artifact_hashes.json` stores SHA-256 digests and file sizes for all artifacts. |
| **No hardcoded superseded numbers anywhere** | **PASS** | Grep scan confirmed 0 occurrences of deprecated numbers (80,021, 69.4%, 0.8857, 225.2%, 1,331 h, 0.5660) across `app.py`, `REPORT.md`, and documentation. |

---

### 8. Deliverables & Streamlit App

| Criteria | Status | Evidence & Verification |
|---|:---:|---|
| **app.py runs without errors and renders all views** | **PASS** | Streamlit AppTest smoke tests render all 5 views (`Executive Dashboard`, `Pillar 1: High Priority Triage`, `Pillar 2: Volume Forecasting`, `Pillar 3: Auto-Tagging & Routing`, `Pillar 4: Asset Risk & RFC`) with 0 unhandled exceptions. |
| **app.py dynamically binds metrics from JSON & models** | **PASS** | All metrics displayed in KPI cards, tables, and charts are bound directly to `reports/metrics.json` and saved joblib payloads. |
| **requirements.txt pinned matching README badges** | **PASS** | `scikit-learn==1.9.1`, `lightgbm==4.7.0`, `statsmodels==0.15.0`. Developer tools isolated in `requirements-dev.txt`. |
| **No duplicate notebooks present** | **PASS** | `PRCL-0012_ITSM_Machine_Learning.ipynb` retained as sole canonical delivery notebook. Duplicate `Improved.ipynb` safely removed. |
| **Cell 39 runtime governance audit passes** | **PASS** | Governance checks in `PRCL-0012_ITSM_Machine_Learning.ipynb` pass all methodology checks with 0 unwhitelisted hardcoded print numbers. |

