# PM-PR-0012: Viva Defense & Executive Review Cheat Sheet
## ABC Tech ITSM Capstone (Machine Learning & Operational Analytics)

This cheat sheet summarizes the key defense points, architectural decisions, and honest metric changes between the original flawed analysis and the improved production-grade implementation (`PRCL-0012_ITSM_Machine_Learning_Improved.ipynb`).

---

### 1. The Core Philosophy of the Improved Architecture
> **"Honest, leak-free, defensible data science over vanity metrics."**
> In the original notebook, high accuracy figures were achieved through data leakage (using post-incident fields), lookahead bias (random splits on time-series), and bugs (misreading regional commas). The improved architecture eliminates all leakage, uses temporal backtesting, and presents realistic, audit-verified metrics.

---

### 2. High-Frequency Viva Questions & Model Answers

#### Q1: "Why did your P1/P2 PR-AUC drop from 0.74 in v1 to 0.26 in Improved?"
- **The Trap**: An examiner will ask why the score went down.
- **Your Answer**:
  1. **Random vs Temporal Split**: v1 used a random 80/20 train/test split. On time-ordered incident streams, random splits leak future ticket vocabulary and seasonal patterns into the training set. We enforced a strict **chronological 3-way temporal split** (Train: Oct 2013–Jan 2014, Val: Jan–Feb 2014, Test: Feb–Mar 2014).
  2. **Triage Decoupling (The Critical Insight)**: In v1, the model used `Impact` and `Urgency`. But in ITIL, Priority is defined *by* Impact and Urgency. A simple rule baseline `(Impact <= 2 & Urgency <= 2)` achieves **0.9934 ROC-AUC**. However, Impact and Urgency require human triage.
  3. **Early Warning Value**: Our ML model predicts P1/P2 **at ticket inception** before triage occurs. Given the extreme class imbalance (1.22% prevalence), a test PR-AUC of **0.2615** is a **21.4x lift** over the no-skill baseline (0.0122).
  4. **Threshold Selection**: In v1, the threshold was tuned on the test set. We tuned the $F_2$-score threshold strictly on the **validation set** (`0.713`), giving double weight to recall to avoid missed major outages, achieving **50.93% test recall** [95% CI: 42.20%, 60.36%].

#### Q2: "What was wrong with `Handle_Time` in the original dataset?"
- **Your Answer**:
  1. The raw string in the dataset looks like `"3,87,16,91,111"`. The original author stripped all commas and took the last 6 digits as decimals (`digits[:-6]`), interpreting it as `3,871.69` hours, leading to a median of **1,331.0 hours (~55 days)**.
  2. 55 days of active touch time is physically impossible.
  3. We cross-validated against timestamps: `Close_Time - Open_Time` has a median elapsed duration of **18.42 hours**.
  4. The raw string is an Indian regional database export where the **first comma is the decimal point** (`3.8717` hours).
  5. Decoded correctly, active engineering handle time has a realistic **median of 1.95 hours** (mean 4.25h). All ROI capacity calculations were corrected using this true touch time.

#### Q3: "Why did you delete the '+225% YoY volume growth' claim in Pillar 2?"
- **Your Answer**:
  1. The data covers 2012 to early 2014 (46,606 tickets).
  2. However, from 2012 to September 2013 (20 months), there were only **1,316 tickets** recorded. In October 2013, the enterprise cutover occurred, jumping to **~23,000 tickets per quarter** (45,290 tickets in 6 months).
  3. The 225% growth was a **system rollout artifact**, not real business growth.
  4. Modeling across this structural break produces invalid trend slopes. We isolated the time series strictly to the stable post-rollout regime (26 continuous weeks).
  5. In 6-fold rolling backtesting, the simple **Naive persistence model achieved the lowest MAPE (12.12%)**, competitive with SARIMAX (13.17%) and Holt-Winters (15.77%). In noisy operational ITSM data, naive persistence is a robust competitive benchmark.

#### Q4: "Why did you replace the Random Forest Priority Classifier with an ITIL Matrix in Pillar 3?"
- **Your Answer**:
  1. The original notebook trained a Random Forest to classify Priority (P1–P5) and got 76.6% accuracy.
  2. In ITIL standards, `Priority = f(Impact, Urgency)` via a deterministic $5 \times 5$ lookup matrix.
  3. We proved in Python that the dataset conforms to this matrix with a **100.00% match rate**. Using ML to guess a deterministic business rule introduces error.
  4. The true ML task was imputing missing `Impact` for **1,380 unassigned ('NS') tickets**, where LightGBM achieved an **84.79% Macro-F1 score**, after which the deterministic matrix assigns priority with 100% precision.

#### Q5: "What is your department routing benchmark, and what did the ablation show?"
- **Your Answer**:
  1. In v1, the model reported 69.44% accuracy. But `application` accounts for **68.77%** of all tickets! The v1 model was basically equal to guessing the majority class.
  2. We conducted a 4-way feature ablation under a temporal split with rare-class merging:
     - **Variant A (All Intake Metadata)**: Reaches **96.73% accuracy** (Macro-F1: 0.9731, +27.96% net lift over baseline).
     - **Variant B (Without Subcategory)**: Reaches **93.44% accuracy** (+24.67% lift).
     - **Variant C (Minimal First-Touch, Category + Timing only)**: Reaches **68.73% accuracy** unweighted (or 17.06% accuracy with 20.41% balanced recall).
     - **Variant D (Legitimate Intake with WBS)**: Reaches **80.66% accuracy** (+11.89% lift over baseline) and **59.42% balanced recall** (80.57% balanced model).
  3. **Strategic Takeaway**: When Work Breakdown Structure (WBS) cost codes are captured at creation, first-touch auto-routing (Variant D) legitimately beats the majority baseline without lookahead leakage. In our enterprise model card, **Variant D is listed as the deployable production model**, and Variant A as the aspirational upper bound.

#### Q6: "How did you calculate the FTE and cost savings in Pillar 3?"
- **Your Answer**:
  1. **Controlled Econometric Regression**: We estimated $\log(\text{Handle\_Time}) = \beta \cdot \text{Reassignments} + \gamma X + \epsilon$ using HC3 robust standard errors. Each bounce incurs a **$+7.41\%$ delay penalty** ($\beta = +0.0715, p = 1.22 \times 10^{-200}$).
  2. **Data-Linked Savings**: Applied to the median touch time ($1.95$h), each avoided bounce saves **$0.14$ hours** (Range: $0.06$h to $0.35$h).
  3. **Realistic ROI**: Using the deployable Variant D lift (+11.89% over baseline) across the 72,714 full-year volume forecast:
     - **474 engineering hours/year** recovered at base (**0.24 FTEs**).
     - Range: **217 to 1,142 engineering hours/year** (**0.11 to 0.57 FTEs**).
     - Upper-bound Variant A yields **1,114 hours/year** (**0.56 FTEs**).

#### Q7: "How does Pillar 4 predict RFCs and rank asset risk?"
- **Your Answer**:
  1. **RFC Prediction Leakage Audit**: `Related_Change` (Change Request ID) and `No_of_Related_Changes` are linked *after* an engineer investigates the incident. Including them is a 100% circular leak! Excluding them, test ROC-AUC is **0.5364** and PR-AUC is **0.0137**, honestly showing that proactive RFC prediction at intake is challenging without full diagnostic text.
  2. **Asset Risk Index (ARI)**: In v1, assets with only 1 incident topped the risk list. We applied an activity threshold ($N \ge 10$ incidents, isolating 287 enterprise assets) and **Empirical Bayes shrinkage** ($m=10$) toward the global mean.
  3. **Stability**: Spearman rank correlation across 4 weighting schemes showed $\rho \ge 0.8213$, proving the rankings are robust and tiering assets into Critical (43 CIs), High (43 CIs), Medium (58 CIs), and Low (143 CIs).

#### Q8: "How does probability calibration work, and why compare Brier score against prevalence rather than class_weight='balanced' output?"
- **The Answer**: Raw probabilities from tree ensembles trained with `class_weight='balanced'` produce artificially inflated scores shifted toward minority prevalence.
- **Methodology**: We fit Isotonic Regression ($cv='prefit'$) strictly on the validation set.
- **Fair Comparison**: Comparing calibrated probabilities against `balanced` raw output is misleading because balancing intentionally distorts marginal probabilities. We benchmark against the true prospective test prevalence baseline ($Brier = p(1-p) = 0.01212$) and unweighted LightGBM. The calibrated model achieves $Brier = 0.01014$ (a **16.3% error reduction** over prevalence).

#### Q9: "Why was the association between CI_Cat and Dept_Consolidated removed from the Cramér's V analysis?"
- **The Answer**: `Dept_Consolidated` is derived directly via a rule-based mapping from `CI_Cat` (top 5 categories kept as department queues, remainder consolidated as `other_infrastructure`).
- **Tautology Elimination**: Reporting Cramér's V between a feature and its own deterministic transformation is circular and misleading ($V \approx 0.99$).
- **Valid Associations**: We compute bias-corrected Cramér's V strictly across non-derived categorical attributes: Category $\times$ CI_Cat ($V = 0.115$), Priority $\times$ Impact ($V = 0.994$), and Priority $\times$ Urgency ($V = 0.925$).

#### Q10: "What is the operational effect of missing Impact ('NS') tickets on ITIL Priority?"
- **The Finding**: In the raw dataset, all 1,380 'NS' (No Severity) tickets have Urgency clean = 3 (1,348 tickets) or Urgency clean = 4 (32 tickets).
- **Active Model Imputation**:
  - Training LightGBM on known tickets yields an imputation Macro-F1 of **0.8479** (Accuracy: 98.51%).
  - The model resolves **78.8% (1,088 tickets)** to Priority 3 and **21.2% (292 tickets)** to Priority 4.
  - Multi-seed testing across 5 independent seeds confirms that the distribution is completely invariant: P3 is identically 1,088 tickets and P4 is identically 292 tickets across all 5 seeds.
- **Operational Reality**: The ML imputation model actively changes priority assignment for 21.2% of unassigned tickets by routing them to Priority 4 based on service signals rather than defaulting them all to Priority 3.

#### Q11: "How did you calculate the 2014 capacity forecast and guarantee Low < Base < High?"
- **Corrected Calendar Window**: Q1 2014 actual is strictly computed from tickets with `Open_DT` in `2014-01-01` to `2014-03-31` (**21,979 incidents**), not the full historical regime.
- **Ensemble Forecast**: The remaining 39 weeks of 2014 are projected using the top-performing Ensemble (SARIMAX + Naive), totaling **50,735 incidents**.
- **Full-Year 2014 Base**: $21,979 + 50,735 =$ **72,714 incidents** (with 2015 planning baseline at **76,349 incidents** under a +5% assumed growth scenario).
- **Moving-Block Bootstrap ($L=4$)**: We simulate 1,000 whole 39-week trajectories using 4-week overlapping residual blocks to preserve empirical autocorrelation. The resulting 95% Prediction Interval for full-year 2014 is **[67,163, 76,519] incidents**, guaranteeing $Low < Base < High$.
- **Backtest Error Consistency**: Weekly out-of-sample backtest MAPE is **12.65%**. Over 39 weeks, partial diversification across 4-week blocks yields an aggregate uncertainty half-width of $\pm 4,678$ incidents ($\pm 9.22\%$ on the future portion; $\pm 6.43\%$ on full-year base), fully consistent with the ~12% weekly backtest error.

#### Q12: "How does your automated governance audit guarantee reproducibility and zero hardcoded figures?"
- **Dynamic Programmatic Rendering**: Every number quoted in summary tables, model cards, limitations, and takeaways is generated dynamically from runtime variables via Python code.
- **Dual-Layer Governance (Cell 36)**:
  1. *Hard Runtime Assertions*: Formally asserts that Q1 2014 actual matches the calendar window (21,979), that 2014 base equals Q1 + 39w forecast (72,714), that $Low < Base < High$, that imputation counts are stable across seeds, that Variant D beats majority baseline, and that every claim in limitations, model cards, and roadmap matches runtime variables.
  2. *Runtime State Checks*: Verifies all 17 methodological requirements top-to-bottom.
  3. *Static AST & Markdown Scan*: Parses notebook JSON and Python AST, scanning all markdown cells and print string templates to guarantee 0 typed numeric metrics. Both audits pass with 100% compliance.

---

### 3. Summary Scorecard (What Changed from v1 to Improved)

| Dimension | Original Notebook (v1) | Improved Architecture (`Improved.ipynb`) |
|---|---|---|
| **P1/P2 PR-AUC** | 0.74 (random split, leakage) | **0.2615** (temporal split, early warning, 21.4x lift) |
| **P1/P2 Decision Threshold** | 0.76 (tuned on test set) | **0.713** (validation $F_2$-tuned, test evaluated once) |
| **Handle Time Median** | 1,331.0 hours (string parsing bug) | **1.95 hours** (reconciled touch time vs 18.42h duration; 21.7% active > calendar limitation noted) |
| **Forecasting Method** | Static fit over structural break | **6-fold rolling backtest**, Naive competitive (MAPE 12.12%, Ensemble 12.65%) |
| **2014 Volume Target** | Hardcoded 80,000 | Dynamic ensemble forecast: **72,714** [67,163 – 76,519] ($Low < Base < High$, Moving-Block L=4) |
| **Priority Determination** | Random Forest (76.6% accuracy) | **Deterministic ITIL matrix (100.00%)** + ML imputation (78.8% P3 vs 21.2% P4, 5-seed stable) |
| **Routing Lift** | -1.16% below majority baseline | **+11.89% net lift** with intake WBS (Variant D: 80.66% vs 68.77% base); Upper Bound Variant A: 96.73% |
| **Deployable Routing Result** | Not reported | **Variant D (with WBS)**: 80.66% accuracy / 59.42% balanced recall (beats 68.77% baseline) |
| **Annualized ROI** | 34.6M hours (impossible) | **Data-linked**: 474 hrs / 0.24 FTEs (Variant D); 1,114 hrs / 0.56 FTEs (Variant A) |
| **Asset Risk Index** | Distorted by $N=1$ outliers | **Empirical Bayes shrinkage ($N \ge 10$)**, $\rho \ge 0.8213$ rank stability |
| **Integrity & Auditing** | Hardcoded print numbers | **Zero hardcoded numbers**, automated claim-to-variable runtime assertions + AST scan (ALL PASSED) |
