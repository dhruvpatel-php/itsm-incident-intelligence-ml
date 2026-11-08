import sys
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
import joblib

from sklearn.metrics import (
    roc_auc_score, average_precision_score, recall_score,
    precision_score, roc_curve
)
import lightgbm as lgb

from src.config import (
    ARI_MIN_INCIDENTS,
    ARI_SHRINKAGE_M,
    CLEANED_DATA_PATH,
    FIGURES_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    STABLE_REGIME_START,
    setup_logger
)

logger = setup_logger("Pillar4_RFC_Assets")

def build_rfc_and_assets(df=None):
    logger.info("=" * 70)
    logger.info("PILLAR 4: PROACTIVE RFC DETECTION (NEGATIVE RESULT) & ASSET RISK INDEX")
    logger.info("=" * 70)

    if df is None:
        df = pd.read_csv(CLEANED_DATA_PATH, low_memory=False)
        df['Open_DT'] = pd.to_datetime(df['Open_DT'])

    df_stable = df[df['Open_DT'] >= STABLE_REGIME_START].sort_values('Open_DT').reset_index(drop=True)

    # -------------------------------------------------------------
    # PART A: Request for Change (RFC) Leakage-Free Evaluation
    # -------------------------------------------------------------
    logger.info("--- PART A: Leakage-Free Proactive RFC Prediction ---")
    logger.info("Strict Leakage Prevention: 'Related_Change', 'No_of_Related_Changes', 'Close_Code', 'Resolved_Time', 'Handle_Time' are EXCLUDED.")

    n_rfc = len(df_stable)
    n_tr_rfc = int(n_rfc * 0.65)
    n_val_rfc = int(n_rfc * 0.15)

    train_rfc = df_stable.iloc[:n_tr_rfc]
    val_rfc = df_stable.iloc[n_tr_rfc:n_tr_rfc + n_val_rfc]
    test_rfc = df_stable.iloc[n_tr_rfc + n_val_rfc:]

    ci_freq_rfc = train_rfc['CI_Name'].value_counts().to_dict()
    wbs_freq_rfc = train_rfc['WBS'].value_counts().to_dict()

    def prep_rfc_features(data):
        X = pd.get_dummies(data[['CI_Cat', 'Category']], drop_first=True, dtype=float)
        X['CI_Name_Freq'] = data['CI_Name'].map(ci_freq_rfc).fillna(0).values
        X['WBS_Freq'] = data['WBS'].map(wbs_freq_rfc).fillna(0).values
        X['Open_Hour'] = data['Open_DT'].dt.hour.values
        X['Open_DayOfWeek'] = data['Open_DT'].dt.day_of_week.values
        X['Is_Weekend'] = X['Open_DayOfWeek'].isin([5, 6]).astype(int).values
        return X

    X_tr_rfc = prep_rfc_features(train_rfc)
    X_val_rfc = prep_rfc_features(val_rfc).reindex(columns=X_tr_rfc.columns, fill_value=0.0)
    X_te_rfc = prep_rfc_features(test_rfc).reindex(columns=X_tr_rfc.columns, fill_value=0.0)

    y_tr_rfc = train_rfc['Has_RFC'].values
    y_val_rfc = val_rfc['Has_RFC'].values
    y_te_rfc = test_rfc['Has_RFC'].values

    # Validated RFC Hyperparameters from audited notebook
    rfc_lgb = lgb.LGBMClassifier(
        num_leaves=15, n_estimators=80, max_depth=5, learning_rate=0.1,
        class_weight='balanced', random_state=RANDOM_STATE, verbose=-1, n_jobs=-1
    )
    rfc_lgb.fit(X_tr_rfc, y_tr_rfc)

    probs_val_rfc = rfc_lgb.predict_proba(X_val_rfc)[:, 1]
    probs_te_rfc = rfc_lgb.predict_proba(X_te_rfc)[:, 1]

    # Cost-optimal threshold tuning under asymmetric CAB penalty (Cost Ratio = 5.0)
    cost_ratio = 5.0
    t_grid_rfc = np.linspace(0.1, 0.9, 81)
    costs_val_rfc = []
    for t in t_grid_rfc:
        preds = (probs_val_rfc >= t).astype(int)
        fn = np.sum((y_val_rfc == 1) & (preds == 0))
        fp = np.sum((y_val_rfc == 0) & (preds == 1))
        costs_val_rfc.append(cost_ratio * fn + 1.0 * fp)

    opt_t_rfc = float(t_grid_rfc[np.argmin(costs_val_rfc)])

    roc_rfc = float(roc_auc_score(y_te_rfc, probs_te_rfc))
    pr_rfc = float(average_precision_score(y_te_rfc, probs_te_rfc))
    preds_te_rfc = (probs_te_rfc >= opt_t_rfc).astype(int)
    rec_rfc = float(recall_score(y_te_rfc, preds_te_rfc, zero_division=0))
    prec_rfc = float(precision_score(y_te_rfc, preds_te_rfc, zero_division=0))
    rfc_prev = round(float(np.mean(y_te_rfc)), 4)

    # Bootstrap 95% CI
    np.random.seed(RANDOM_STATE)
    boot_prs = []
    for _ in range(1000):
        idx = np.random.randint(0, len(y_te_rfc), len(y_te_rfc))
        y_b = y_te_rfc[idx]
        if len(np.unique(y_b)) < 2:
            continue
        boot_prs.append(average_precision_score(y_b, probs_te_rfc[idx]))

    pr_ci = [round(float(np.percentile(boot_prs, 2.5)), 4), round(float(np.percentile(boot_prs, 97.5)), 4)]

    logger.info(f"Test ROC-AUC: {roc_rfc:.4f} (Near random chance 0.5000)")
    logger.info(f"Test PR-AUC : {pr_rfc:.4f} [95% CI: {pr_ci[0]:.4f} - {pr_ci[1]:.4f}]")
    logger.info(f"Operating Threshold: {opt_t_rfc:.2f} | Recall: {rec_rfc*100:.2f}% | Precision: {prec_rfc*100:.2f}%")

    # Plot RFC Performance Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(t_grid_rfc, costs_val_rfc, color='#dc2626', linewidth=2, label='Expected Operational Cost')
    ax1.axvline(opt_t_rfc, color='k', linestyle='--', label=f'Cost-Optimal Threshold = {opt_t_rfc:.2f}')
    ax1.set_title("Pillar 4: Cost Curve vs Decision Threshold (Negative Result)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Decision Threshold", fontsize=10)
    ax1.set_ylabel("Total Operational Cost (Relative Units)", fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right')

    fpr_r, tpr_r, _ = roc_curve(y_te_rfc, probs_te_rfc)
    ax2.plot(fpr_r, tpr_r, color='#984ea3', lw=2, label=f"Leakage-Free LGBM (ROC-AUC = {roc_rfc:.4f})")
    ax2.plot([0, 1], [0, 1], 'k--', lw=1.5, label='No-Skill Baseline (0.5000)')
    ax2.set_title("Pillar 4: Prospective Test ROC Curve (Negative Result)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("False Positive Rate", fontsize=10)
    ax2.set_ylabel("True Positive Rate", fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='lower right')
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "11_pillar4_rfc_roc_curve.png", dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # PART B: Empirical Bayes Asset Risk Index (ARI)
    # -------------------------------------------------------------
    logger.info("--- PART B: Empirical Bayes-Smoothed Asset Risk Index (ARI) ---")
    ci_stats = df_stable.groupby('CI_Name').agg(
        total_incidents=('Incident_ID', 'count'),
        p1p2_incidents=('Is_High_Priority', 'sum'),
        rfc_incidents=('Has_RFC', 'sum'),
        reassigned_incidents=('No_of_Reassignments_Clean', lambda s: (s > 0).sum()),
        mean_handle_time=('Handle_Time_Clean', 'mean')
    ).reset_index()

    # Activity Filter: Minimum incidents threshold
    ci_stable_ari = ci_stats[ci_stats['total_incidents'] >= ARI_MIN_INCIDENTS].copy()
    n_eval_assets = len(ci_stable_ari)
    logger.info(f"Assets evaluated with N >= {ARI_MIN_INCIDENTS} incidents: {n_eval_assets}")

    # Empirical Bayes Shrinkage
    m_shrink = ARI_SHRINKAGE_M
    global_p1p2_rate = float(df_stable['Is_High_Priority'].mean())
    global_rfc_rate = float(df_stable['Has_RFC'].mean())
    global_reassign_rate = float((df_stable['No_of_Reassignments_Clean'] > 0).mean())

    ci_stable_ari['p1p2_rate_bayes'] = (ci_stable_ari['p1p2_incidents'] + m_shrink * global_p1p2_rate) / (ci_stable_ari['total_incidents'] + m_shrink)
    ci_stable_ari['rfc_rate_bayes'] = (ci_stable_ari['rfc_incidents'] + m_shrink * global_rfc_rate) / (ci_stable_ari['total_incidents'] + m_shrink)
    ci_stable_ari['reassign_rate_bayes'] = (ci_stable_ari['reassigned_incidents'] + m_shrink * global_reassign_rate) / (ci_stable_ari['total_incidents'] + m_shrink)

    # Percentile Normalization [0, 1]
    for col in ['total_incidents', 'p1p2_rate_bayes', 'rfc_rate_bayes', 'reassign_rate_bayes', 'mean_handle_time']:
        ci_stable_ari[f'{col}_pctl'] = ci_stable_ari[col].rank(pct=True)

    # 4 Weighting Schemes
    weights_config = {
        'P1_Heavy': {'total_incidents_pctl': 0.15, 'p1p2_rate_bayes_pctl': 0.40, 'rfc_rate_bayes_pctl': 0.20, 'reassign_rate_bayes_pctl': 0.15, 'mean_handle_time_pctl': 0.10},
        'Balanced': {'total_incidents_pctl': 0.20, 'p1p2_rate_bayes_pctl': 0.25, 'rfc_rate_bayes_pctl': 0.25, 'reassign_rate_bayes_pctl': 0.15, 'mean_handle_time_pctl': 0.15},
        'RFC_Heavy': {'total_incidents_pctl': 0.15, 'p1p2_rate_bayes_pctl': 0.20, 'rfc_rate_bayes_pctl': 0.40, 'reassign_rate_bayes_pctl': 0.15, 'mean_handle_time_pctl': 0.10},
        'Equal': {'total_incidents_pctl': 0.20, 'p1p2_rate_bayes_pctl': 0.20, 'rfc_rate_bayes_pctl': 0.20, 'reassign_rate_bayes_pctl': 0.20, 'mean_handle_time_pctl': 0.20}
    }

    for w_name, w_dict in weights_config.items():
        ci_stable_ari[f'ARI_{w_name}'] = sum(ci_stable_ari[col] * w for col, w in w_dict.items()) * 100.0

    rank_df = pd.DataFrame({w_name: ci_stable_ari[f'ARI_{w_name}'].rank(ascending=False) for w_name in weights_config.keys()})
    spearman_corr_matrix = rank_df.corr(method='spearman')
    min_spearman = float(spearman_corr_matrix.values[np.triu_indices_from(spearman_corr_matrix.values, k=1)].min())

    ci_stable_ari['ARI_Final'] = ci_stable_ari['ARI_Balanced']
    p85 = float(ci_stable_ari['ARI_Final'].quantile(0.85))
    p70 = float(ci_stable_ari['ARI_Final'].quantile(0.70))
    p50 = float(ci_stable_ari['ARI_Final'].quantile(0.50))

    def assign_tier(score):
        if score >= p85: return 'Critical'
        if score >= p70: return 'High'
        if score >= p50: return 'Medium'
        return 'Low'

    ci_stable_ari['Risk_Tier'] = ci_stable_ari['ARI_Final'].apply(assign_tier)
    tier_dist = ci_stable_ari['Risk_Tier'].value_counts().to_dict()

    # Get dominant CI_Cat and CI_Subcat for metadata display
    ci_cat_map = df_stable.groupby('CI_Name')['CI_Cat'].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown').to_dict()
    ci_subcat_map = df_stable.groupby('CI_Name')['CI_Subcat'].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown').to_dict()
    ci_stable_ari['CI_Cat'] = ci_stable_ari['CI_Name'].map(ci_cat_map)
    ci_stable_ari['CI_Subcat'] = ci_stable_ari['CI_Name'].map(ci_subcat_map)

    # Renaming for backward compatibility with app.py
    ci_registry = ci_stable_ari.sort_values('ARI_Final', ascending=False).copy()
    ci_registry['Asset_Risk_Index'] = ci_registry['ARI_Final']
    ci_registry['Total_Incidents'] = ci_registry['total_incidents']
    ci_registry['P1_P2_Incidents'] = ci_registry['p1p2_incidents']
    ci_registry['RFC_Incidents'] = ci_registry['rfc_incidents']
    ci_registry['Misconfig_Rate'] = ci_registry['rfc_rate_bayes'] # Clean shrunken proxy

    # Save registry CSV
    ci_registry.to_csv(MODELS_DIR / "pillar4_asset_risk_scores.csv", index=False)

    top10_critical = list(ci_registry.head(10)['CI_Name'].values)
    logger.info(f"Minimum Spearman correlation across weighting schemes: {min_spearman:.4f}")
    logger.info(f"Risk Tier Counts: {tier_dist}")
    logger.info(f"Top 10 Critical Assets: {top10_critical}")

    # Plot CI Asset Risk Matrix
    fig, ax = plt.subplots(figsize=(11, 6))
    scatter = ax.scatter(
        ci_registry['total_incidents'],
        ci_registry['ARI_Final'],
        c=ci_registry['p1p2_rate_bayes'],
        cmap='YlOrRd',
        s=45,
        alpha=0.85,
        edgecolor='black',
        linewidth=0.6
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Shrunk High-Priority Rate (Bayesian)", fontsize=11)

    top5 = ci_registry.head(5)
    offsets = [
        (-135, 16),   # SUB000508: top-left (280, 92.4)
        (20, 16),     # WBA000124: top-right (477, 91.1)
        (20, 16),     # WBA000082: mid-right (89, 85.2)
        (-135, 14),   # WBA000107: mid-left  (58, 82.5)
        (20, -28),    # WBA000122: lower-right (99, 81.7)
    ]
    for (_, row), (dx, dy) in zip(top5.iterrows(), offsets):
        ax.annotate(
            f"{row['CI_Name']} ({row['CI_Cat']})\nRisk: {row['ARI_Final']:.1f}",
            xy=(row['total_incidents'], row['ARI_Final']),
            xytext=(dx, dy),
            textcoords='offset points',
            fontsize=8.5, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.35", fc="#ffffff", ec="#475569", alpha=0.95),
            arrowprops=dict(arrowstyle="->", color='#0f172a', lw=1.2)
        )

    ax.set_xscale('log')
    ax.set_ylim(15, 106)
    ax.set_title("Pillar 4: Configuration Item (CI) Bayesian Asset Risk Matrix (N >= 10)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Total Incident Volume (Log Scale, Min N=10)", fontsize=11)
    ax.set_ylabel("Composite Asset Risk Index (0 - 100)", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "12_pillar4_ci_asset_risk_matrix.png", dpi=300)
    plt.close(fig)

    metrics = {
        "rfc_roc_auc": round(roc_rfc, 4),
        "rfc_pr_auc": round(pr_rfc, 4),
        "rfc_prevalence": rfc_prev,
        "min_incidents_threshold": ARI_MIN_INCIDENTS,
        "bayes_shrinkage_m": ARI_SHRINKAGE_M,
        "evaluated_assets_count": int(n_eval_assets),
        "min_spearman_rho": round(min_spearman, 4),
        "tier_distribution": {
            "Critical": int(tier_dist.get('Critical', 0)),
            "High": int(tier_dist.get('High', 0)),
            "Medium": int(tier_dist.get('Medium', 0)),
            "Low": int(tier_dist.get('Low', 0))
        },
        "top_critical_assets": top10_critical
    }

    payload = {
        "model": rfc_lgb,
        "ci_freq": ci_freq_rfc,
        "wbs_freq": wbs_freq_rfc,
        "train_cols": list(X_tr_rfc.columns),
        "threshold": opt_t_rfc,
        "roc_auc": roc_rfc,
        "pr_auc": pr_rfc,
        "metrics": metrics
    }
    joblib.dump(payload, MODELS_DIR / "pillar4_rfc_predictor.joblib")
    logger.info("Pillar 4 RFC and Asset Risk artifacts successfully saved.")
    return metrics

if __name__ == "__main__":
    build_rfc_and_assets()
