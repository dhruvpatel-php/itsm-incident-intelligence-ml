import sys
import json
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
    precision_score, fbeta_score, brier_score_loss,
    roc_curve, precision_recall_curve
)
from sklearn.isotonic import IsotonicRegression
from sklearn.inspection import permutation_importance
import lightgbm as lgb

from src.config import (
    CLEANED_DATA_PATH,
    FIGURES_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    STABLE_REGIME_START,
    setup_logger
)
from src.feature_engineering import build_early_warning_features

logger = setup_logger("Pillar1_Priority")

def train_pillar1(df=None):
    logger.info("=" * 70)
    logger.info("PILLAR 1: EARLY WARNING FOR HIGH PRIORITY INCIDENTS (P1/P2)")
    logger.info("=" * 70)

    if df is None:
        df = pd.read_csv(CLEANED_DATA_PATH, low_memory=False)
        df['Open_DT'] = pd.to_datetime(df['Open_DT'])

    # 1. Temporal split on stable operating regime
    df_stable = df[df['Open_DT'] >= STABLE_REGIME_START].sort_values('Open_DT').reset_index(drop=True)
    df_p1 = df_stable[df_stable['Is_High_Priority'].notnull()].copy().reset_index(drop=True)

    n_total = len(df_p1)
    n_tr = int(n_total * 0.65)
    n_val = int(n_total * 0.15)

    train_df = df_p1.iloc[:n_tr].copy()
    val_df = df_p1.iloc[n_tr:n_tr + n_val].copy()
    test_df = df_p1.iloc[n_tr + n_val:].copy()

    logger.info(f"Temporal Split: Train={len(train_df):,} ({len(train_df)/n_total*100:.1f}%), "
                f"Val={len(val_df):,} ({len(val_df)/n_total*100:.1f}%), "
                f"Test={len(test_df):,} ({len(test_df)/n_total*100:.1f}%)")

    # 2. Build feature matrices strictly avoiding post-triage fields
    ci_freq = train_df['CI_Name'].value_counts().to_dict()
    wbs_freq = train_df['WBS'].value_counts().to_dict()

    X_tr = build_early_warning_features(train_df, ci_freq, wbs_freq)
    train_cols = list(X_tr.columns)
    X_val = build_early_warning_features(val_df, ci_freq, wbs_freq, train_cols=train_cols)
    X_te = build_early_warning_features(test_df, ci_freq, wbs_freq, train_cols=train_cols)

    y_tr = train_df['Is_High_Priority'].astype(int).values
    y_val = val_df['Is_High_Priority'].astype(int).values
    y_te = test_df['Is_High_Priority'].astype(int).values

    logger.info(f"High-Priority Prevalence: Train={y_tr.mean()*100:.3f}%, Val={y_val.mean()*100:.3f}%, Test={y_te.mean()*100:.3f}%")

    # 3. Fit Champion Model: Cost-Weighted LightGBM (Tuned hyperparams from audited notebook)
    lgb_model = lgb.LGBMClassifier(
        n_estimators=80, max_depth=3, num_leaves=31, learning_rate=0.05,
        class_weight='balanced', random_state=RANDOM_STATE, verbose=-1, n_jobs=-1
    )
    lgb_model.fit(X_tr, y_tr)

    # 4. Probability Calibration on Validation Set
    val_probs = lgb_model.predict_proba(X_val)[:, 1]
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(val_probs, y_val)

    # 5. Tune Threshold on Validation Set strictly (F2-Score optimization)
    threshold_grid = np.linspace(0.01, 0.99, 100)
    f2_val = [fbeta_score(y_val, (val_probs >= t).astype(int), beta=2.0, zero_division=0) for t in threshold_grid]
    best_idx = int(np.argmax(f2_val))
    opt_threshold = float(threshold_grid[best_idx])
    logger.info(f"Optimal Decision Threshold (Validation F2): {opt_threshold:.3f}")

    # 6. Single Holdout Test Evaluation
    test_probs = lgb_model.predict_proba(X_te)[:, 1]
    test_preds = (test_probs >= opt_threshold).astype(int)

    test_roc = float(roc_auc_score(y_te, test_probs))
    test_pr = float(average_precision_score(y_te, test_probs))
    test_rec = float(recall_score(y_te, test_preds))
    test_prec = float(precision_score(y_te, test_preds, zero_division=0))
    test_f2 = float(fbeta_score(y_te, test_preds, beta=2.0))
    test_prev = float(y_te.mean())
    test_lift = float(test_pr / test_prev)

    # 1,000-fold Bootstrap Confidence Intervals
    np.random.seed(RANDOM_STATE)
    boot_prs = []
    boot_recs = []
    for _ in range(1000):
        b_idx = np.random.randint(0, len(y_te), len(y_te))
        y_b = y_te[b_idx]
        if len(np.unique(y_b)) < 2:
            continue
        boot_prs.append(average_precision_score(y_b, test_probs[b_idx]))
        boot_recs.append(recall_score(y_b, test_preds[b_idx], zero_division=0))

    pr_ci_low, pr_ci_high = float(np.percentile(boot_prs, 2.5)), float(np.percentile(boot_prs, 97.5))
    rec_ci_low, rec_ci_high = float(np.percentile(boot_recs, 2.5)), float(np.percentile(boot_recs, 97.5))

    # Brier Scores
    brier_prev = float(brier_score_loss(y_te, np.repeat(y_tr.mean(), len(y_te))))
    cal_test_probs = iso.predict(test_probs)
    brier_cal = float(brier_score_loss(y_te, cal_test_probs))
    brier_err_reduc = float((brier_prev - brier_cal) / brier_prev * 100.0)

    # 7. Seen vs Unseen Asset Generalization Audit
    seen_mask = test_df['CI_Name'].isin(train_df['CI_Name'].unique()).values
    y_seen, p_seen, pred_seen = y_te[seen_mask], test_probs[seen_mask], test_preds[seen_mask]
    y_unseen, p_unseen, pred_unseen = y_te[~seen_mask], test_probs[~seen_mask], test_preds[~seen_mask]

    seen_stats = {
        "n": int(len(y_seen)),
        "pct": float(len(y_seen) / len(y_te) * 100.0),
        "prevalence": float(y_seen.mean()),
        "pr_auc": float(average_precision_score(y_seen, p_seen)),
        "recall": float(recall_score(y_seen, pred_seen, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_seen, p_seen))
    }
    unseen_stats = {
        "n": int(len(y_unseen)),
        "pct": float(len(y_unseen) / len(y_te) * 100.0),
        "prevalence": float(y_unseen.mean()),
        "pr_auc": float(average_precision_score(y_unseen, p_unseen)),
        "recall": float(recall_score(y_unseen, pred_unseen, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_unseen, p_unseen))
    }

    logger.info(f"Test ROC-AUC: {test_roc:.4f} | Test PR-AUC: {test_pr:.4f} [95% CI: {pr_ci_low:.4f}-{pr_ci_high:.4f}]")
    logger.info(f"Test Recall @ F2 Thresh: {test_rec*100:.2f}% [95% CI: {rec_ci_low*100:.2f}%-{rec_ci_high*100:.2f}%]")
    logger.info(f"Statistical Lift: {test_lift:.1f}x over prevalence baseline ({test_prev*100:.3f}%)")
    logger.info(f"Seen Assets PR-AUC={seen_stats['pr_auc']:.4f}, Recall={seen_stats['recall']*100:.1f}%")
    logger.info(f"Unseen Assets PR-AUC={unseen_stats['pr_auc']:.4f}, Recall={unseen_stats['recall']*100:.1f}%")

    # 8. Visualizations
    fpr, tpr, _ = roc_curve(y_te, test_probs)
    prec_c, rec_c, _ = precision_recall_curve(y_te, test_probs)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].plot(fpr, tpr, color='#2563eb', linewidth=2, label=f'LightGBM Early Warning (ROC-AUC = {test_roc:.4f})')
    axes[0].plot([0, 1], [0, 1], linestyle='--', color='gray', label='No-Skill (0.5000)')
    axes[0].set_title("Pillar 1: Prospective ROC Curve (Holdout Test Set)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("False Positive Rate", fontsize=10)
    axes[0].set_ylabel("True Positive Rate (Recall)", fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend(loc='lower right')

    axes[1].plot(rec_c, prec_c, color='#16a34a', linewidth=2, label=f'LightGBM (PR-AUC = {test_pr:.4f})')
    axes[1].axhline(test_prev, linestyle='--', color='gray', label=f'No-Skill Prevalence ({test_prev*100:.2f}%)')
    axes[1].scatter([test_rec], [test_prec], color='#dc2626', s=80, zorder=5, label=f'Operating Point @ {opt_threshold:.3f}')
    axes[1].set_title(f"Pillar 1: Precision-Recall Curve (Lift = {test_lift:.1f}x)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Recall", fontsize=10)
    axes[1].set_ylabel("Precision", fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend(loc='upper right')
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "05_pillar1_roc_pr_curves.png", dpi=300)
    plt.close(fig)

    # Permutation feature importance
    perm = permutation_importance(lgb_model, X_te, y_te, scoring='average_precision', n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1)
    feat_series = pd.Series(perm.importances_mean, index=train_cols).sort_values(ascending=True).tail(10)

    fig, ax = plt.subplots(figsize=(10, 5))
    feat_series.plot(kind='barh', ax=ax, color='#2563eb', edgecolor='black', alpha=0.85)
    ax.set_title("Pillar 1: Out-of-Sample Permutation Feature Importance (PR-AUC Loss)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Average PR-AUC Decrease when Feature Shuffled", fontsize=10)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "06_pillar1_feature_importance.png", dpi=300)
    plt.close(fig)

    # 9. Save Artifacts & Metrics
    metrics = {
        "model_name": "LightGBM (Cost-Weighted)",
        "test_roc_auc": round(test_roc, 4),
        "test_pr_auc": round(test_pr, 4),
        "test_pr_auc_ci_low": round(pr_ci_low, 4),
        "test_pr_auc_ci_high": round(pr_ci_high, 4),
        "prevalence_baseline": round(test_prev, 5),
        "statistical_lift": round(test_lift, 1),
        "validation_threshold": round(opt_threshold, 3),
        "test_recall": round(test_rec, 4),
        "test_recall_ci_low": round(rec_ci_low, 4),
        "test_recall_ci_high": round(rec_ci_high, 4),
        "test_precision": round(test_prec, 4),
        "test_f2_score": round(test_f2, 4),
        "brier_calibrated": round(brier_cal, 5),
        "brier_prevalence": round(brier_prev, 5),
        "brier_error_reduction_pct": round(brier_err_reduc, 1),
        "seen_assets": seen_stats,
        "unseen_assets": unseen_stats
    }

    model_payload = {
        "model": lgb_model,
        "calibrated_model": iso,
        "model_name": "LightGBM (Cost-Weighted)",
        "threshold": opt_threshold,
        "ci_freq": ci_freq,
        "wbs_freq": wbs_freq,
        "train_cols": train_cols,
        "metrics": metrics,
        "provenance": {
            "training_samples": len(train_df),
            "validation_samples": len(val_df),
            "test_samples": len(test_df),
            "random_state": RANDOM_STATE
        }
    }
    joblib.dump(model_payload, MODELS_DIR / "pillar1_high_priority_model.joblib")

    with open(MODELS_DIR / "pillar1_threshold.json", "w") as f:
        json.dump({"threshold": opt_threshold, "f2_score": test_f2, "pr_auc": test_pr}, f, indent=4)

    logger.info("Pillar 1 model, threshold and plots successfully saved.")
    return metrics

if __name__ == "__main__":
    train_pillar1()
