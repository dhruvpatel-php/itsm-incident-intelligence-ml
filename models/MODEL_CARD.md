# Enterprise Model Cards: ABC Tech ITSM Predictive Intelligence Platform
### Reference: PM-PR-0012 | Source of Truth: `reports/metrics.json`
**Pipeline Commit**: `731213b1ee` | **Python Version**: `3.14.7`

---

## Pillar 1: High-Priority Incident Early Warning (P1/P2)

### Intended Use
- **Primary Use Case**: Automated real-time intake triage for incoming IT tickets to flag critical outages (P1/P2) before manual dispatch.
- **Target Population**: All enterprise IT incidents submitted to ABC Tech ticketing system.
- **Out-of-Scope**: Resolution time estimation or root-cause assignment.

### Model Architecture & Hyperparameters
- **Architecture**: Cost-Weighted LightGBM Classifier (`LGBMClassifier`).
- **Validated Hyperparameters**: `num_leaves=31, n_estimators=80, max_depth=3, learning_rate=0.05, class_weight='balanced'`.
- **Calibration**: Isotonic Regression (`IsotonicRegression(out_of_bounds='clip')`) fit on validation set.

### Training & Validation Split
- **Methodology**: Prospective chronological temporal split on the stable operating regime (Open_DT >= 2013-10-01).
- **Split Proportions**: Train = 65% (8,245 seen / 579 unseen in holdout), Validation = 15%, Holdout Test = 20% (8,824 incidents).
- **Class Prevalence**: 1.796% (Train), 1.149% (Validation), 1.224% (Test).

### Verified Empirical Performance
- **ROC-AUC**: **0.9148**
- **PR-AUC**: **0.2615** [95% Bootstrap CI: 0.1989 - 0.3503]
- **Statistical Lift**: **21.4x** over random guessing
- **Optimal Decision Threshold (Validation F2)**: **0.713**
- **Holdout Test Recall**: **50.93%** [95% Bootstrap CI: 42.20% - 60.36%]
- **Holdout Test Precision**: **38.73%** (F2-Score: 0.4791)
- **Probability Calibration**: Brier score = 0.01014 (-16.3% error vs prevalence baseline 0.01212)
- **Generalization Audit**:
  - Seen Assets (93.4%): PR-AUC = 0.2254, Recall = 34.8%, ROC-AUC = 0.8874
  - Unseen Assets (6.6%): PR-AUC = 0.3317, Recall = 76.2%, ROC-AUC = 0.8949

### Limitations & Failure Modes
- **Unseen Asset Drift**: Tickets from novel CI assets rely heavily on WBS frequency and submission timing.
- **Sparse Feature Inputs**: Narrative ticket descriptions are absent in tabular feeds; model relies on categorical metadata.

---

## Pillar 2: Incident Volume & Capacity Forecaster

### Intended Use
- **Primary Use Case**: Multi-quarter enterprise staffing, shift allocation, and ITIL capacity planning.
- **Target Horizon**: 39 weeks ahead (Q2 - Q4 2014) following Q1 actuals.
- **Out-of-Scope**: Intraday or hourly queue routing.

### Model Architecture
- **Production Architecture**: Production Ensemble combining `SARIMAX(1, 1, 0)` and `Naive` persistence.
- **Uncertainty Quantification**: Moving-Block Bootstrap ($L=4$ continuous weeks, $B=1,000$ iterations) capturing residual autocorrelation.

### Data Regime & Backtest Design
- **Regime Selection**: Restricted strictly to post-rollout stable regime (26 continuous weekly observations from Oct 2013 to Mar 2014). Historical 2012-2013 pilot data excluded to prevent structural distortion.
- **Backtesting**: 6-Fold Rolling-Origin Cross-Validation ($H=2$ weeks).
- **Comparative Backtest MAPE**:
  - Naive Persistence: 12.12%
  - Production Ensemble: 12.65%
  - SARIMAX(1,1,0): 13.17%
  - Holt-Winters: 15.77%
  - Seasonal Naive: 20.17%

### Verified Forecast Metrics
- **Q1 2014 Actual Recorded**: **21,979 tickets**
- **Future 39-Week Base Forecast**: **50,735 tickets**
- **2014 Full-Year Projected Volume**: **72,714 tickets**
- **95% Moving-Block Bootstrap PI**: **[67,163, 76,519] tickets**
- **Invariance Guarantee**: Strictly satisfies `Low < Base < High`.
- **2015 Scenario (+5% expansion)**: **76,349 tickets**

### Limitations & Failure Modes
- **Regime Length**: Only 26 weekly observations exist in the initial stable cutover, resulting in wider long-horizon bootstrap bands.

---

## Pillar 3: Department Auto-Router & ITIL Priority Engine

### Intended Use
- **Primary Use Case**: First-touch service queue routing (Variant D) and deterministic ITIL priority resolution at intake.
- **Target Population**: All incoming enterprise incidents.

### Part A: Deterministic ITIL Priority Resolution & Impact Imputer
- **ITIL Deterministic Matrix**: Priority is a deterministic 2D lookup of (Impact, Urgency). Match rate = **100.00%** (0 discrepancies across 45,226 tickets).
- **Missing Impact Imputation (1,380 NS Tickets)**:
  - Architecture: LightGBM Classifier (`LGBMClassifier(n_estimators=80, max_depth=5, class_weight='balanced')`).
  - 5-Fold CV Accuracy: **98.51%** (Macro-F1: 0.8479).
  - Champion Imputed Distribution: **78.8% P3 (1,088 tickets)** / **21.2% P4 (292 tickets)**.
  - Covariate Shift Audit: 97.7% of NS tickets have Urgency 3 (vs 11.5% in known tickets). Bootstrap split range: [9.9%, 96.7%] P3. Exactly 269 tickets (19.5%) flagged for human review.

### Part B: Department Auto-Routing (Variant D)
- **Features Used**: Category, WBS billing code frequency, opening hour, day of week, weekend indicator (strictly intake-safe; alert status excluded as constant).
- **Temporal Split Evaluation**:
  - Majority Baseline: 68.77%
  - Variant D Unweighted Accuracy: **80.66%** (+11.89% lift; 59.42% balanced acc)
  - Variant D Balanced Accuracy: **70.37%** (+1.59% lift; **80.57% balanced acc**)
  - WBS Signal Audit: 74.6% of distinct WBS codes map to exactly 1 queue (37.5% of all tickets).

### Part C: Controlled Econometric Regression & Recovered Capacity
- **Model**: `log(Handle_Time) ~ No_of_Reassignments + C(Priority) + C(Category) + C(CI_Cat)` with HC3 robust covariance.
- **Delay Penalty**: Reassignments increase active duration by **+7.41% per bounce** (0.14 hours, $p = 1.2240e-200$).
- **Capacity Recovered (Variant D Lift)**:
  - Base: **514 engineering hours/year (0.26 FTEs)**.
  - Bounds: [217 hrs (0.11 FTEs), 1,227 hrs (0.61 FTEs)].

---

## Pillar 4: Proactive RFC Detection & Empirical Bayes Asset Risk Index

### Part A: Request for Change (RFC) Detection (Negative Result)
- **Intended Use**: Evaluating whether intake metadata can anticipate emergency change requests.
- **Audit Finding**: Reported honestly as a **negative result**.
  - Holdout Test ROC-AUC: **0.5364** (Near random chance 0.5000)
  - Holdout Test PR-AUC: **0.0137**
- **Governance**: Prior models claiming >88% AUC relied on post-investigation circular fields (`Related_Change`, `No_of_Related_Changes`). Without leakage, tabular intake metadata lacks proactive RFC signal.

### Part B: Empirical Bayes Asset Risk Index (ARI)
- **Intended Use**: Proactive Configuration Item (CI) risk profiling, preventative maintenance scheduling, and CMDB risk tagging.
- **Activity Filter**: Applied strictly to enterprise assets with sufficient incident volume ($N \ge 10$), evaluating **287 Configuration Items**.
- **Empirical Bayes Shrinkage**: $m = 10.0$ shrinks raw rates toward the global mean, eliminating false small-sample artifacts.
- **Rank Stability**: Minimum Spearman correlation $\rho = 0.8213$ across 4 distinct weighting schemes.
- **Risk Tiers**:
  - Critical (score >= 85th percentile): 43 assets
  - High (score >= 70th percentile): 43 assets
  - Medium (score >= 50th percentile): 58 assets
  - Low (score < 50th percentile): 143 assets
- **Top 10 Critical Assets**: SUB000508, WBA000124, WBA000082, WBA000107, WBA000122, WBA000008, SUB000002, SBA000626, SUB000479, SUB000505.

---
*Certified Enterprise Model Card. Generated by `scripts/build_report.py` from `reports/metrics.json`.*
