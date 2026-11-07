import sys
import json
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import statsmodels.api as sm

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix
)
import lightgbm as lgb

from src.config import (
    CLEANED_DATA_PATH,
    FIGURES_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    STABLE_REGIME_START,
    PRIORITY_MATRIX,
    setup_logger
)

logger = setup_logger("Pillar3_Routing")

# Top-level class for pickling compatibility
class ITILPriorityEngine:
    def __init__(self, imputer_model, imp_cols, ci_freq):
        self.imputer = imputer_model
        self.imp_cols = imp_cols
        self.ci_freq = ci_freq
        self.matrix = PRIORITY_MATRIX

    def resolve(self, row):
        imp = row.get('Impact_Num')
        urg = row.get('Urgency_Num')
        if pd.notnull(imp) and pd.notnull(urg):
            return self.matrix.get((int(imp), int(urg)), 3)
        return 3

def build_routing(df=None, base_vol_2014=72714):
    logger.info("=" * 70)
    logger.info("PILLAR 3: ITIL PRIORITY RESOLUTION, AUTO-ROUTING & ROI ECONOMETRICS")
    logger.info("=" * 70)

    if df is None:
        df = pd.read_csv(CLEANED_DATA_PATH, low_memory=False)
        df['Open_DT'] = pd.to_datetime(df['Open_DT'])

    # 1. Stable Regime strictly sorted by timestamp
    df_stable = df[df['Open_DT'] >= STABLE_REGIME_START].sort_values('Open_DT').reset_index(drop=True)
    df_p1 = df_stable[df_stable['Is_High_Priority'].notnull()].copy().reset_index(drop=True)
    n_tr = int(len(df_p1) * 0.65)
    train_df = df_p1.iloc[:n_tr].copy()
    ci_freq = train_df['CI_Name'].value_counts().to_dict()

    # -------------------------------------------------------------
    # PART A: Deterministic ITIL Matrix & Impact Imputation
    # -------------------------------------------------------------
    logger.info("--- PART A: Deterministic ITIL Matrix Lookup Verification ---")
    valid_iu_mask = df['Impact_Num'].notnull() & df['Urgency_Num'].notnull() & df['Priority'].notnull()
    valid_iu_df = df[valid_iu_mask].copy()

    def verify_priority_matrix(row):
        return PRIORITY_MATRIX.get((int(row['Impact_Num']), int(row['Urgency_Num'])), -1) == int(row['Priority'])

    matches = valid_iu_df.apply(verify_priority_matrix, axis=1)
    match_rate = float(matches.mean())
    mismatches = int((~matches).sum())
    n_valid = len(valid_iu_df)

    logger.info(f"Verified tickets: {n_valid:,} | Match Rate: {match_rate*100:.2f}% | Discrepancies: {mismatches}")

    # Impact Imputation for NS Tickets
    logger.info("--- Training LightGBM Impact Imputation Model for NS Tickets ---")
    known_imp_mask = df['Impact_Num'].notnull()
    df_known_imp = df[known_imp_mask].copy()
    df_ns_imp = df[~known_imp_mask].copy()
    n_ns = len(df_ns_imp)

    df_known_imp['Urgency_Str'] = df_known_imp['Urgency_Num'].astype('Int64').astype(str)
    df_ns_imp['Urgency_Str'] = df_ns_imp['Urgency_Num'].astype('Int64').astype(str)

    X_imp = pd.get_dummies(df_known_imp[['Category', 'CI_Cat', 'Urgency_Str']], drop_first=True, dtype=float)
    X_imp['CI_Name_Freq'] = df_known_imp['CI_Name'].map(ci_freq).fillna(0).values
    y_imp = df_known_imp['Impact_Num'].astype(int).values

    # 5-fold Stratified CV
    skf_imp = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    imp_accs, imp_f1s = [], []

    for tr_k, te_k in skf_imp.split(X_imp, y_imp):
        m_k = lgb.LGBMClassifier(n_estimators=80, max_depth=5, class_weight='balanced', random_state=RANDOM_STATE, verbose=-1, n_jobs=-1)
        m_k.fit(X_imp.iloc[tr_k], y_imp[tr_k])
        preds_k = m_k.predict(X_imp.iloc[te_k])
        imp_accs.append(accuracy_score(y_imp[te_k], preds_k))
        imp_f1s.append(f1_score(y_imp[te_k], preds_k, average='macro'))

    imp_cv_acc = float(np.mean(imp_accs))
    imp_cv_f1 = float(np.mean(imp_f1s))
    logger.info(f"Impact Imputer 5-Fold CV Accuracy: {imp_cv_acc*100:.2f}% | Macro-F1: {imp_cv_f1:.4f}")

    X_ns = pd.get_dummies(df_ns_imp[['Category', 'CI_Cat', 'Urgency_Str']], drop_first=True, dtype=float)
    X_ns['CI_Name_Freq'] = df_ns_imp['CI_Name'].map(ci_freq).fillna(0).values
    X_ns = X_ns.reindex(columns=X_imp.columns, fill_value=0.0)
    ns_urgencies = df_ns_imp['Urgency_Num'].values

    # Bootstrap bagging (B=20) for uncertainty range
    b_impute = 20
    boot_p3_counts, boot_p4_counts = [], []
    for b in range(b_impute):
        idx_b = np.random.RandomState(RANDOM_STATE + b).choice(len(X_imp), size=len(X_imp), replace=True)
        m_b = lgb.LGBMClassifier(
            n_estimators=70, max_depth=5, subsample=0.8, colsample_bytree=0.8, subsample_freq=1,
            class_weight='balanced', random_state=RANDOM_STATE + b, verbose=-1, n_jobs=-1
        )
        m_b.fit(X_imp.iloc[idx_b], y_imp[idx_b])
        p_b = m_b.predict(X_ns)
        mapped_b = [PRIORITY_MATRIX.get((int(imp), int(urg))) for imp, urg in zip(p_b, ns_urgencies)]
        boot_p3_counts.append(sum(p == 3 for p in mapped_b))
        boot_p4_counts.append(sum(p == 4 for p in mapped_b))

    boot_p3_range = [round(float(np.min(boot_p3_counts) / n_ns * 100.0), 1), round(float(np.max(boot_p3_counts) / n_ns * 100.0), 1)]
    boot_p4_range = [round(float(np.min(boot_p4_counts) / n_ns * 100.0), 1), round(float(np.max(boot_p4_counts) / n_ns * 100.0), 1)]

    # Champion Model Fit
    m_imp_full = lgb.LGBMClassifier(n_estimators=80, max_depth=5, class_weight='balanced', random_state=RANDOM_STATE, verbose=-1, n_jobs=-1)
    m_imp_full.fit(X_imp, y_imp)
    ns_preds = m_imp_full.predict(X_ns)
    imputed_impact_counts = {str(k): int(v) for k, v in pd.Series(ns_preds).value_counts().to_dict().items()}

    mapped_priorities = [PRIORITY_MATRIX.get((int(imp), int(urg))) for imp, urg in zip(ns_preds, ns_urgencies)]
    imputed_priority_counts = {str(k): int(v) for k, v in pd.Series(mapped_priorities).value_counts().to_dict().items()}
    p3_share = round(float(imputed_priority_counts.get("3", 0) / n_ns), 3)
    p4_share = round(float(imputed_priority_counts.get("4", 0) / n_ns), 3)

    # Covariate shift stats
    urg_known = df_known_imp['Urgency_Num'].dropna()
    urg_ns = df_ns_imp['Urgency_Num'].dropna()
    known_u3_share = round(float((urg_known == 3).mean()), 3)
    ns_u3_share = round(float((urg_ns == 3).mean()), 3)
    flagged_review_count = int(imputed_impact_counts.get("5", 0))
    flagged_review_pct = round(float(flagged_review_count / n_ns * 100.0), 1)

    logger.info(f"Champion Imputed Priorities: P3={p3_share*100:.1f}%, P4={p4_share*100:.1f}%")
    logger.info(f"Bootstrap P3 Range: {boot_p3_range}% | P4 Range: {boot_p4_range}%")
    logger.info(f"Covariate Shift: Urgency 3 is {ns_u3_share*100:.1f}% in NS tickets vs {known_u3_share*100:.1f}% in known tickets")

    priority_metrics = {
        "verified_tickets": n_valid,
        "deterministic_match_rate": match_rate,
        "discrepancies": mismatches,
        "ns_tickets": n_ns,
        "imputation_cv_accuracy": round(imp_cv_acc, 4),
        "imputation_macro_f1": round(imp_cv_f1, 4),
        "imputed_impact_counts": imputed_impact_counts,
        "imputed_priority_counts": imputed_priority_counts,
        "champion_p3_share": p3_share,
        "champion_p4_share": p4_share,
        "bootstrap_p3_range": boot_p3_range,
        "bootstrap_p4_range": boot_p4_range,
        "covariate_shift_ns_urgency3": ns_u3_share,
        "covariate_shift_known_urgency3": known_u3_share,
        "flagged_human_review_count": flagged_review_count,
        "flagged_human_review_pct": flagged_review_pct
    }

    # -------------------------------------------------------------
    # PART B: Department Auto-Routing (Temporal Split & Ablation)
    # -------------------------------------------------------------
    logger.info("--- PART B: Department Auto-Routing (Temporal Split) ---")
    top_depts = ['application', 'subapplication', 'computer', 'storage', 'hardware']
    df_stable['Dept_Consolidated'] = df_stable['CI_Cat'].apply(lambda x: x if x in top_depts else 'other_infrastructure')

    wbs_dept_nunique = df_stable.groupby('WBS')['Dept_Consolidated'].nunique()
    wbs_unique_share = float((wbs_dept_nunique == 1).mean() * 100.0)
    ticket_wbs_unique_share = float(df_stable['WBS'].map(wbs_dept_nunique == 1).mean() * 100.0)

    n_st = len(df_stable)
    n_tr_d = int(n_st * 0.65)
    n_val_d = int(n_st * 0.15)

    train_d = df_stable.iloc[:n_tr_d]
    val_d = df_stable.iloc[n_tr_d:n_tr_d + n_val_d]
    test_d = df_stable.iloc[n_tr_d + n_val_d:]

    y_tr_d = train_d['Dept_Consolidated'].values
    y_te_d = test_d['Dept_Consolidated'].values
    maj_baseline_route = float((y_te_d == 'application').mean())

    ci_f_d = train_d['CI_Name'].value_counts().to_dict()
    wbs_f_d = train_d['WBS'].value_counts().to_dict()

    # Feature Builders for Ablation
    def prep_variant_a(data):
        X = pd.get_dummies(data[['Category', 'CI_Subcat']], drop_first=True, dtype=float)
        X['CI_Name_Freq'] = data['CI_Name'].map(ci_f_d).fillna(0).values
        X['WBS_Freq'] = data['WBS'].map(wbs_f_d).fillna(0).values
        X['Open_Hour'] = data['Open_DT'].dt.hour.values
        X['Open_DayOfWeek'] = data['Open_DT'].dt.day_of_week.values
        X['Is_Weekend'] = X['Open_DayOfWeek'].isin([5, 6]).astype(int).values
        return X

    def prep_variant_b(data):
        X = pd.get_dummies(data[['Category']], drop_first=True, dtype=float)
        X['CI_Name_Freq'] = data['CI_Name'].map(ci_f_d).fillna(0).values
        X['WBS_Freq'] = data['WBS'].map(wbs_f_d).fillna(0).values
        X['Open_Hour'] = data['Open_DT'].dt.hour.values
        X['Open_DayOfWeek'] = data['Open_DT'].dt.day_of_week.values
        X['Is_Weekend'] = X['Open_DayOfWeek'].isin([5, 6]).astype(int).values
        return X

    def prep_variant_c(data):
        X = pd.get_dummies(data[['Category']], drop_first=True, dtype=float)
        X['Open_Hour'] = data['Open_DT'].dt.hour.values
        X['Open_DayOfWeek'] = data['Open_DT'].dt.day_of_week.values
        X['Is_Weekend'] = X['Open_DayOfWeek'].isin([5, 6]).astype(int).values
        return X

    def prep_variant_d(data):
        X = pd.get_dummies(data[['Category']], drop_first=True, dtype=float)
        X['WBS_Freq'] = data['WBS'].map(wbs_f_d).fillna(0).values
        X['Open_Hour'] = data['Open_DT'].dt.hour.values
        X['Open_DayOfWeek'] = data['Open_DT'].dt.day_of_week.values
        X['Is_Weekend'] = X['Open_DayOfWeek'].isin([5, 6]).astype(int).values
        return X

    # Variant A
    X_tr_a = prep_variant_a(train_d)
    X_te_a = prep_variant_a(test_d).reindex(columns=X_tr_a.columns, fill_value=0.0)
    m_a = lgb.LGBMClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=RANDOM_STATE, verbose=-1, n_jobs=-1)
    m_a.fit(X_tr_a, y_tr_d)
    acc_a = float(accuracy_score(y_te_d, m_a.predict(X_te_a)))

    # Variant B
    X_tr_b = prep_variant_b(train_d)
    X_te_b = prep_variant_b(test_d).reindex(columns=X_tr_b.columns, fill_value=0.0)
    m_b = lgb.LGBMClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=RANDOM_STATE, verbose=-1, n_jobs=-1)
    m_b.fit(X_tr_b, y_tr_d)
    acc_b = float(accuracy_score(y_te_d, m_b.predict(X_te_b)))

    # Variant C
    X_tr_c = prep_variant_c(train_d)
    X_te_c = prep_variant_c(test_d).reindex(columns=X_tr_c.columns, fill_value=0.0)
    m_c = lgb.LGBMClassifier(n_estimators=100, max_depth=5, random_state=RANDOM_STATE, verbose=-1, n_jobs=-1)
    m_c.fit(X_tr_c, y_tr_d)
    acc_c = float(accuracy_score(y_te_d, m_c.predict(X_te_c)))

    # Variant D (Unweighted and Balanced)
    X_tr_d = prep_variant_d(train_d)
    X_te_d = prep_variant_d(test_d).reindex(columns=X_tr_d.columns, fill_value=0.0)

    m_d_unw = lgb.LGBMClassifier(n_estimators=100, max_depth=5, random_state=RANDOM_STATE, verbose=-1, n_jobs=-1)
    m_d_unw.fit(X_tr_d, y_tr_d)
    p_d_unw = m_d_unw.predict(X_te_d)
    acc_d_unw = float(accuracy_score(y_te_d, p_d_unw))
    bal_d_unw = float(balanced_accuracy_score(y_te_d, p_d_unw))
    lift_d_unw = float(acc_d_unw - maj_baseline_route)

    m_d_bal = lgb.LGBMClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=RANDOM_STATE, verbose=-1, n_jobs=-1)
    m_d_bal.fit(X_tr_d, y_tr_d)
    p_d_bal = m_d_bal.predict(X_te_d)
    acc_d_bal = float(accuracy_score(y_te_d, p_d_bal))
    bal_d_bal = float(balanced_accuracy_score(y_te_d, p_d_bal))
    lift_d_bal = float(acc_d_bal - maj_baseline_route)

    logger.info(f"Majority Baseline: {maj_baseline_route*100:.2f}%")
    logger.info(f"Variant D Unweighted: Acc={acc_d_unw*100:.2f}%, BalAcc={bal_d_unw*100:.2f}%, Lift=+{lift_d_unw*100:.2f}%")
    logger.info(f"Variant D Balanced  : Acc={acc_d_bal*100:.2f}%, BalAcc={bal_d_bal*100:.2f}%, Lift=+{lift_d_bal*100:.2f}%")

    # -------------------------------------------------------------
    # PART C: Controlled OLS Regression & ROI
    # -------------------------------------------------------------
    logger.info("--- PART C: Controlled OLS Econometric Regression & ROI ---")
    sub_reg = df[df['Handle_Time_Clean'].notnull() & (df['Handle_Time_Clean'] > 0) & df['Priority_Clean'].notnull()].copy()
    sub_reg['log_handle'] = np.log1p(sub_reg['Handle_Time_Clean'])
    X_reg = pd.get_dummies(sub_reg[['Priority_Clean', 'Category', 'CI_Cat']], drop_first=True, dtype=float)
    X_reg['No_of_Reassignments'] = sub_reg['No_of_Reassignments_Clean']
    X_reg = sm.add_constant(X_reg)
    y_reg = sub_reg['log_handle']

    ols_model = sm.OLS(y_reg, X_reg).fit(cov_type='HC3')
    beta_reassign = float(ols_model.params['No_of_Reassignments'])
    ci_reassign = ols_model.conf_int().loc['No_of_Reassignments']
    p_reassign = float(ols_model.pvalues['No_of_Reassignments'])

    pct_delay_base = float(np.expm1(beta_reassign))
    pct_delay_low = float(np.expm1(ci_reassign[0]))
    pct_delay_high = float(np.expm1(ci_reassign[1]))

    med_handle_time = float(df['Handle_Time_Clean'].dropna().median())
    q25_handle_time = float(df['Handle_Time_Clean'].dropna().quantile(0.25))
    q75_handle_time = float(df['Handle_Time_Clean'].dropna().quantile(0.75))

    hours_saved_hop_base = med_handle_time * pct_delay_base
    hours_saved_hop_low = q25_handle_time * pct_delay_low
    hours_saved_hop_high = q75_handle_time * pct_delay_high

    pct_reassigned_tickets = float((df['No_of_Reassignments_Clean'] > 0).mean())
    annual_misrouted_tickets = float(base_vol_2014 * pct_reassigned_tickets)

    lift_d = float(max(0.001, acc_d_unw - maj_baseline_route))
    tickets_saved_d = annual_misrouted_tickets * lift_d
    annual_hours_d_base = tickets_saved_d * hours_saved_hop_base
    annual_hours_d_low = tickets_saved_d * hours_saved_hop_low
    annual_hours_d_high = tickets_saved_d * hours_saved_hop_high

    fte_d_base = annual_hours_d_base / 2000.0
    fte_d_low = annual_hours_d_low / 2000.0
    fte_d_high = annual_hours_d_high / 2000.0

    # Sensitivity Analysis
    sens_25_h = round(annual_misrouted_tickets * lift_d * 0.25 * hours_saved_hop_base)
    sens_50_h = round(annual_misrouted_tickets * lift_d * 0.50 * hours_saved_hop_base)
    sens_100_h = round(annual_misrouted_tickets * lift_d * 1.00 * hours_saved_hop_base)

    routing_metrics = {
        "majority_baseline": round(maj_baseline_route, 4),
        "variant_d_unweighted_accuracy": round(acc_d_unw, 4),
        "variant_d_unweighted_balanced_acc": round(bal_d_unw, 4),
        "variant_d_unweighted_lift": round(lift_d_unw, 4),
        "variant_d_balanced_accuracy": round(acc_d_bal, 4),
        "variant_d_balanced_balanced_acc": round(bal_d_bal, 4),
        "variant_d_balanced_lift": round(lift_d_bal, 4),
        "variant_c_accuracy": round(acc_c, 4),
        "variant_b_accuracy": round(acc_b, 4),
        "variant_a_accuracy": round(acc_a, 4),
        "wbs_unique_share": round(wbs_unique_share, 2),
        "wbs_unique_ticket_share": round(ticket_wbs_unique_share, 2),
        "delay_beta": round(beta_reassign, 4),
        "delay_p_value": p_reassign,
        "delay_penalty_pct": round(pct_delay_base * 100.0, 2),
        "delay_hours_saved_per_bounce": round(hours_saved_hop_base, 2),
        "delay_hours_low": round(hours_saved_hop_low, 2),
        "delay_hours_high": round(hours_saved_hop_high, 2),
        "annual_hours_saved_base": int(round(annual_hours_d_base)),
        "annual_fte_saved_base": round(fte_d_base, 2),
        "annual_hours_saved_low": int(round(annual_hours_d_low)),
        "annual_fte_saved_low": round(fte_d_low, 2),
        "annual_hours_saved_high": int(round(annual_hours_d_high)),
        "annual_fte_saved_high": round(fte_d_high, 2),
        "roi_sensitivity_25": {
            "effective_lift": round(lift_d * 0.25, 4),
            "hours": int(sens_25_h),
            "fte": round(sens_25_h / 2000.0, 2)
        },
        "roi_sensitivity_50": {
            "effective_lift": round(lift_d * 0.50, 4),
            "hours": int(sens_50_h),
            "fte": round(sens_50_h / 2000.0, 2)
        },
        "roi_sensitivity_100": {
            "effective_lift": round(lift_d * 1.00, 4),
            "hours": int(sens_100_h),
            "fte": round(sens_100_h / 2000.0, 2)
        }
    }

    logger.info(f"Delay beta: {beta_reassign:.4f} (p={p_reassign:.4e}) | Penalty: +{pct_delay_base*100:.2f}%")
    logger.info(f"Base ROI: {annual_hours_d_base:,.0f} hours saved ({fte_d_base:.2f} FTEs)")

    # -------------------------------------------------------------
    # Visualizations
    # -------------------------------------------------------------
    classes_d = sorted(list(set(y_te_d)))
    cm = confusion_matrix(y_te_d, p_d_unw, labels=classes_d)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes_d, yticklabels=classes_d, ax=ax)
    ax.set_title("Pillar 3: Department Auto-Routing Confusion Matrix (Variant D)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Predicted Department", fontsize=11)
    ax.set_ylabel("Actual Department", fontsize=11)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "09_pillar3_department_confusion_matrix.png", dpi=300)
    plt.close(fig)

    # Per-Class Recall Comparison Plot
    rep_a = classification_report(y_te_d, m_a.predict(X_te_a), output_dict=True, zero_division=0)
    rep_c = classification_report(y_te_d, m_c.predict(X_te_c), output_dict=True, zero_division=0)
    rep_d = classification_report(y_te_d, p_d_bal, output_dict=True, zero_division=0)

    rec_a_vals = [float(rep_a[c]['recall']) for c in classes_d]
    rec_c_vals = [float(rep_c.get(c, {'recall': 0.0})['recall']) for c in classes_d]
    rec_d_vals = [float(rep_d[c]['recall']) for c in classes_d]

    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = np.arange(len(classes_d))
    width = 0.25
    ax.bar(x_pos - width, rec_a_vals, width, label='Variant A (Full Asset Metadata)', color='#2563eb', edgecolor='black', alpha=0.85)
    ax.bar(x_pos, rec_d_vals, width, label='Variant D (Category + WBS + timing: Balanced)', color='#10b981', edgecolor='black', alpha=0.85)
    ax.bar(x_pos + width, rec_c_vals, width, label='Variant C (Category + Timing)', color='#f59e0b', edgecolor='black', alpha=0.85)
    ax.set_title("Pillar 3: Per-Queue Routing Recall Across Intake Variants (Holdout Set)", fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel("Recall Rate", fontsize=11)
    ax.set_xlabel("Consolidated IT Service Department", fontsize=11)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(classes_d, rotation=15, ha='right', fontsize=10)
    ax.legend(loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "10_pillar3_priority_confusion_matrix.png", dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # Save Model Artifacts
    # -------------------------------------------------------------
    router_payload = {
        "model": m_d_unw,
        "model_balanced": m_d_bal,
        "wbs_freq": wbs_f_d,
        "train_cols": list(X_tr_d.columns),
        "classes": classes_d,
        "accuracy": acc_d_unw,
        "balanced_accuracy": bal_d_unw,
        "metrics": routing_metrics,
        "provenance": {
            "regime_start": STABLE_REGIME_START,
            "n_train": len(train_d),
            "n_test": len(test_d),
            "random_state": RANDOM_STATE
        }
    }
    joblib.dump(router_payload, MODELS_DIR / "pillar3_department_router.joblib")

    # Priority Classifier Payload
    prio_payload = {
        "engine": ITILPriorityEngine(m_imp_full, list(X_imp.columns), ci_freq),
        "impact_imputer": m_imp_full,
        "imputer_cols": list(X_imp.columns),
        "accuracy": imp_cv_acc,
        "macro_f1": imp_cv_f1,
        "deterministic_match_rate": match_rate,
        "metrics": priority_metrics
    }
    joblib.dump(prio_payload, MODELS_DIR / "pillar3_priority_classifier.joblib")
    joblib.dump(prio_payload, MODELS_DIR / "pillar3_impact_imputer.joblib")

    logger.info("Pillar 3 routing & priority artifacts successfully saved.")
    return {"pillar3_priority": priority_metrics, "pillar3_routing": routing_metrics}

if __name__ == "__main__":
    build_routing()
