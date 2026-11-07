# CHANGELOG: PM-PR-0012 ITSM Machine Learning Pipeline (Audited Enterprise Revision)

This document records the exact changes, empirical findings, and metric shifts implemented in `PRCL-0012_ITSM_Machine_Learning.ipynb`. All outputs in the notebook are generated strictly by in-process code execution from top to bottom, with zero hardcoding and full reproducibility.

---

## Executive Summary of Latest Reviewer Requirements & Changes

| Category / Requirement | Description of Change in Improved Notebook | Impact & Exact Computed Values (Which Numbers Changed) |
|---|---|---|
| **1. Imputation Uncertainty & Covariate Shift** (Cells 24, 30, 32, 34, 37) | Stored bootstrap min, median, and max shares for Priority 3 and Priority 4 under training set bagging ($B=20$). Displayed range next to champion value across all summary tables, model cards, limitations, and roadmap. Documented covariate shift between known and NS tickets. Added deployment recommendation line to roadmap. | • **Bootstrap Priority Range**: P3 range is **[9.9%, 96.7%]** (median 78.8%); P4 range is **[3.3%, 90.1%]** (median 21.2%). Replaces static champion-only reporting.<br>• **Covariate Shift**: Known tickets are 11.5% Urgency 3 vs **97.7% in NS tickets** (Kolmogorov-Smirnov $KS=0.8466, p \approx 0$).<br>• **Deployment Recommendation**: Default NS tickets to Priority 3 (ITIL matrix result for Impact 1–4 at Urgency 3); flag only the **269 tickets (19.5%)** imputed as Impact 5 for human review. |
| **2. Typed Number Elimination & Strict Audit Regex** (Cells 34, 39) | Replaced literal string `"74.5% of WBS map to 1 queue"` in Cell 34 with dynamic f-string using `{wbs_unique_share:.1f}%`. Replaced all remaining typed numeric constants in print strings across cells with variables. Upgraded static audit in Cell 39 to catch ANY digit-containing number (`\b\d+(?:\.\d+)?\b`) with strict structural whitelist. Added runtime assertions verifying every claim in model cards. | • **Static Audit**: 0 numeric literals in markdown cells, **0 unwhitelisted hardcoded numeric literals** in print/f-strings.<br>• **WBS Unique Queue Share**: Exact computed value is **74.63%** of distinct WBS codes mapping to 1 queue (37.48% of all tickets).<br>• **Runtime Assertions**: 19/19 runtime methodology checks passed; 100% claim-to-variable match verified across all 5 model card panels. |
| **3. Pillar 1 Seen vs Unseen Asset Risk Audit** (Cells 16, 17, 34) | Split prospective holdout test tickets into CI assets seen during training vs unseen assets (`CI_Name_Freq == 0`). Evaluated $n$, prevalence, PR-AUC, and recall at tuned threshold ($0.713$). Added asset risk and taxonomy drift warnings to model card panels. | • **Seen CI Assets** ($n=8{,}245$, 93.44% of test): Prevalence = **0.800%** (66 positives), PR-AUC = **0.2254**, Recall = **34.85%**, ROC-AUC = **0.8874**.<br>• **Unseen CI Assets** ($n=579$, 6.56% of test): Prevalence = **7.254%** (42 positives), PR-AUC = **0.3317**, Recall = **76.19%**, ROC-AUC = **0.8949**.<br>• **Insight**: Unseen new infrastructure assets exhibit 9x higher incident risk, captured effectively by WBS cost code frequency and diurnal arrival cues. |
| **4. Routing Caveat & ROI Sensitivity Analysis** (Cells 25, 26, 34) | Removed constant `Alert_Status` from Variant D; relabeled "Category + WBS + timing". Displayed both unweighted and balanced Variant D metrics in model card, explicitly stating that balanced Variant D gains only **+1.6 points** of accuracy over majority baseline. Added conversion sensitivity rows in Cell 26 ROI simulation for 25%, 50%, and 100% realization. | • **Variant D Unweighted**: Accuracy = **80.66%**, Balanced Accuracy = **59.42%** (+11.89% lift over 68.77% baseline).<br>• **Variant D Balanced**: Accuracy = **70.37%**, Balanced Accuracy = **80.57%** (gains only **+1.59 percentage points** over baseline).<br>• **ROI Conversion Sensitivity**: At base delay (0.14h/bounce):<br>&nbsp;&nbsp;- 25% Realization: **129 engineering hours/yr** (**0.06 FTEs**)<br>&nbsp;&nbsp;- 50% Realization: **257 engineering hours/yr** (**0.13 FTEs**)<br>&nbsp;&nbsp;- 100% Realization: **514 engineering hours/yr** (**0.26 FTEs**). |
| **5. Notebook Integrity & Figures** (Global) | Verified top-to-bottom restart and execution of all 40 cells with `plt.close(fig)` after every single plot to prevent duplicate displays. | • All 40 cells executed with 0 errors.<br>• File size: **1,099,649 bytes**.<br>• Scorecard Badge: **19 / 19 CHECKS PASSED (100% COMPLIANT)**. |

---

## Detailed Section-by-Section Pillar Audit

### 1. Data Ingestion, Cleaning & Handle_Time Reconciliation
* **Handle_Time Units**: Decoded active engineering handle time has a realistic median of **1.95 hours** (mean 4.25h, 75th percentile 4.37h), while calendar duration has a median of **18.40 hours** (mean 123.13h). Handle time exceeds calendar duration for **21.7%** of tickets (Pearson $r = 0.1168$, Spearman $\rho = 0.5684$), reflecting asynchronous multi-engineer logging.
* **Structural Break**: Verified pre-rollout pilot regime (1,150 tickets prior to Oct 2013) vs enterprise cutover (45,456 tickets Oct 2013–Mar 2014). All downstream ML models are trained strictly on the stable regime.
* **Association Matrix**: Highest non-derived association exists between `Impact_Clean` and `Priority_Clean` (Cramér's V = 0.99), while arrival hour shows 0.00 correlation with handle time.

### 2. Pillar 1: High-Priority Incident Early Warning (P1/P2)
* **Temporal Prospective Holdout Evaluation**: Test ROC-AUC = **0.9148**, Test PR-AUC = **0.2615** [95% CI: 0.1989, 0.3503] (**21.4x lift** over 1.224% prevalence baseline).
* **Threshold Tuning on Validation Only**: Tuned operating threshold = **0.713** on validation fold, achieving **50.93% test recall** [95% CI: 42.20%, 60.36%] and 38.73% precision.
* **Out-of-Sample Permutation Feature Importance**: `WBS_Freq` causes the largest PR-AUC drop ($\Delta = 0.3278 \pm 0.0121$), followed by `CI_Name_Freq` ($\Delta = 0.2930 \pm 0.0312$), confirming asset operational history drives early warning.
* **Seen vs Unseen Asset Audit**: Unseen assets represent 6.56% of volume ($n=579$) with 7.25% high-priority prevalence; model achieves PR-AUC 0.3317 and 76.19% recall on unseen assets.

### 3. Pillar 2: Incident Volume Forecasting & Capacity Planning
* **Stable Regime Horizon**: 26 weekly historical points in stable regime (mean 1,748 incidents/week).
* **6-Fold Rolling-Origin Backtesting**: Naive persistence MAPE = **12.12%**, Ensemble MAPE = **12.65%**, SARIMAX MAPE = **13.17%**, Holt-Winters MAPE = **15.77%**.
* **2014 Full-Year Capacity Planning**:
  * Q1 2014 Observed Actual: **21,979 incidents**.
  * 39-Week Future Base Forecast: **50,735 incidents**.
  * Full-Year 2014 Base Volume: **72,714 incidents** (95% Moving-Block Bootstrap Prediction Interval: **[67,163, 76,519]**, strictly satisfying $\text{Low} < \text{Base} < \text{High}$).
  * 2015 Capacity Scenario (+5% assumed growth): **76,349 incidents**.

### 4. Pillar 3: ITIL Priority Matrix, Imputation & Department Routing
* **ITIL Deterministic Matrix Verification**: 100.00% match rate across all 45,226 tickets with recorded triage; proves ML priority classification is redundant.
* **Impact Imputation Uncertainty**: Champion model predicts 78.8% P3 ($1{,}088$ tickets) and 21.2% P4 ($292$ tickets). Under bagging resampling ($B=20$), P3 share ranges from **9.9% to 96.7%** due to covariate shift (NS tickets are 97.7% Urgency 3 vs 11.5% in known tickets).
* **Department Auto-Routing Leakage Ablation**:
  * Majority Baseline: 68.77%
  * Variant C (Category + Timing): 68.73% (+-0.04% lift)
  * Variant D Unweighted (Category + WBS + Timing): **80.66% accuracy**, 59.42% balanced accuracy (**+11.89% lift**)
  * Variant D Balanced: **70.37% accuracy**, 80.57% balanced accuracy (+1.59% lift)
  * Variant B (with CI Name): 93.44% accuracy
  * Variant A Upper Bound (with CI Subcategory): 96.73% accuracy.
* **Data-Linked Econometric ROI**:
  * Controlled OLS regression confirms each reassignment bounce adds **+7.41% active handle duration** ($\beta = +0.0715, p = 1.22 \times 10^{-200}$), equivalent to **0.14 hours/hop**.
  * Deployable Variant D (Base case) recovers **514 engineering hours/year** (**0.26 FTEs**).
  * Sensitivity analysis confirms that even at 25% to 50% conversion efficiency, the model recovers **129 to 257 engineering hours/year** (0.06 to 0.13 FTEs).

### 5. Pillar 4: Proactive RFC Prediction & Asset Risk Index (ARI)
* **RFC Feature Leakage Audit**: Confirmed `Related_Change` is recorded post-investigation and constitutes a circular 100% leak. With intake-only features, out-of-sample ROC-AUC is **0.5364** (confirmed negative result; tabular metadata lacks predictive intake signal).
* **Empirical Bayes Asset Risk Index**: Filtered to $N \ge 10$ incidents (287 assets); shrank proportions toward enterprise prior ($m=10$).
* **Spearman Rank Order Stability**: Minimum Spearman rank correlation across 4 weighting schemes is $\rho \ge \mathbf{0.8213}$, confirming robustness of risk rankings. Standardized assets into 4 tiers: Critical (43 assets), High (43), Medium (58), Low (143).

### 6. Automated Governance Scorecard & Static Audit
* **Static Code & Markdown Scan**: 0 numeric literals in markdown cells; 0 unwhitelisted numeric literals in print/f-strings.
* **Runtime Methodology Assertions**: 19/19 checks passed.
* **Claim-to-Variable Verification**: 100% match between printed text, model cards, roadmap claims, and actual runtime variables.
