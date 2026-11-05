import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    roc_curve, precision_recall_curve, f1_score, fbeta_score, confusion_matrix
)
import xgboost as xgb
import lightgbm as lgb

from src.config import CLEANED_DATA_PATH, FIGURES_DIR, MODELS_DIR, RANDOM_STATE
from src.feature_engineering import extract_features

def train_pillar1():
    print("=" * 70)
    print("PILLAR 1: HIGH PRIORITY TICKET (P1/P2) PREDICTION")
    print("=" * 70)
    
    df = pd.read_csv(CLEANED_DATA_PATH, low_memory=False)
    # Filter valid target rows
    df_valid = df[df['Is_High_Priority'].notnull()].copy()
    y = df_valid['Is_High_Priority'].astype(int).values
    
    # Train / Test Stratified Split (80/20)
    train_idx, test_idx = train_test_split(
        df_valid.index, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    
    df_train = df_valid.loc[train_idx]
    df_test = df_valid.loc[test_idx]
    
    y_train = df_train['Is_High_Priority'].astype(int).values
    y_test = df_test['Is_High_Priority'].astype(int).values
    
    print(f"[Pillar 1] Training Samples: {len(y_train)} (Positives: {y_train.sum()} - {y_train.mean()*100:.2f}%)")
    print(f"[Pillar 1] Test Samples:     {len(y_test)} (Positives: {y_test.sum()} - {y_test.mean()*100:.2f}%)")
    
    # Feature extraction (fit on train only)
    X_train, encoders = extract_features(df_train, is_train=True)
    X_test = extract_features(df_test, is_train=False, encoders=encoders)
    
    feature_names = list(X_train.columns)
    
    # Class imbalance weight ratio
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    
    # Models dictionary
    models = {
        "Logistic Regression (Balanced)": LogisticRegression(
            class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE
        ),
        "Random Forest (Balanced)": RandomForestClassifier(
            n_estimators=150, max_depth=12, class_weight='balanced_subsample',
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "LightGBM (Weighted)": lgb.LGBMClassifier(
            n_estimators=150, max_depth=6, scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE, verbose=-1, n_jobs=-1
        ),
        "XGBoost (Weighted)": xgb.XGBClassifier(
            n_estimators=150, max_depth=5, scale_pos_weight=scale_pos_weight,
            learning_rate=0.05, random_state=RANDOM_STATE, eval_metric='logloss', n_jobs=-1
        )
    }
    
    results = {}
    best_pr_auc = 0
    best_model_name = None
    best_model_obj = None
    
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(15, 6))
    
    for name, model in models.items():
        print(f"\n[Training] {name}...")
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        
        results[name] = {
            "ROC-AUC": roc_auc,
            "PR-AUC": pr_auc,
            "y_prob": y_prob,
            "model": model
        }
        print(f"   --> ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")
        
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})", lw=2)
        
        # Plot PR curve
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        ax_pr.plot(rec, prec, label=f"{name} (PR-AUC = {pr_auc:.3f})", lw=2)
        
        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_model_name = name
            best_model_obj = model

    # Finalize ROC Plot
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7)
    ax_roc.set_title("ROC Curves - High Priority Ticket Prediction", fontsize=13, fontweight='bold')
    ax_roc.set_xlabel("False Positive Rate", fontsize=11)
    ax_roc.set_ylabel("True Positive Rate (Recall)", fontsize=11)
    ax_roc.legend(loc="lower right", fontsize=9)
    ax_roc.grid(True, linestyle='--', alpha=0.5)

    # Finalize PR Plot
    ax_pr.axhline(y=pos_count / len(y_train), color='k', linestyle='--', label=f"Baseline ({pos_count/len(y_train):.3f})")
    ax_pr.set_title("Precision-Recall Curves - Severe Imbalance Evaluation", fontsize=13, fontweight='bold')
    ax_pr.set_xlabel("Recall (P1/P2 Coverage)", fontsize=11)
    ax_pr.set_ylabel("Precision", fontsize=11)
    ax_pr.legend(loc="upper right", fontsize=9)
    ax_pr.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "05_pillar1_roc_pr_curves.png", dpi=300)
    plt.close(fig)

    print(f"\n[Pillar 1 Champion Model]: {best_model_name} (PR-AUC: {best_pr_auc:.4f})")
    best_probs = results[best_model_name]["y_prob"]
    
    # Threshold Tuning for F2-Score (Prioritize capturing critical outages)
    thresholds = np.linspace(0.1, 0.9, 81)
    f2_scores = [fbeta_score(y_test, (best_probs >= t).astype(int), beta=2.0, zero_division=0) for t in thresholds]
    best_idx = np.argmax(f2_scores)
    best_threshold = thresholds[best_idx]
    best_f2 = f2_scores[best_idx]
    
    y_pred_opt = (best_probs >= best_threshold).astype(int)
    print(f"\n[Optimal Decision Threshold]: {best_threshold:.3f}")
    print(f"Optimal F2-Score (Recall-focused): {best_f2:.4f}")
    print("\nClassification Report at Optimal Threshold:")
    print(classification_report(y_test, y_pred_opt, target_names=["Standard (P3-P5)", "High Priority (P1/P2)"]))
    
    cm = confusion_matrix(y_test, y_pred_opt)
    print("Confusion Matrix:")
    print(cm)
    
    # Feature Importance Plot
    print("\n[Pillar 1] Generating Feature Importance chart...")
    if hasattr(best_model_obj, "feature_importances_"):
        importances = best_model_obj.feature_importances_
    elif hasattr(best_model_obj, "coef_"):
        importances = np.abs(best_model_obj.coef_[0])
    else:
        importances = np.ones(len(feature_names))
        
    feat_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values(by="Importance", ascending=True).tail(15)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(feat_df['Feature'], feat_df['Importance'], color='#1f77b4', edgecolor='black', alpha=0.85)
    ax.set_title(f"Top 15 Predictive Features - {best_model_name}", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Feature Importance Score", fontsize=11)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "06_pillar1_feature_importance.png", dpi=300)
    plt.close(fig)

    # Save Champion Model, Encoders, and Optimal Threshold
    model_payload = {
        "model": best_model_obj,
        "model_name": best_model_name,
        "encoders": encoders,
        "feature_names": feature_names,
        "threshold": float(best_threshold),
        "pr_auc": float(best_pr_auc),
        "roc_auc": float(results[best_model_name]["ROC-AUC"])
    }
    joblib.dump(model_payload, MODELS_DIR / "pillar1_high_priority_model.joblib")
    
    with open(MODELS_DIR / "pillar1_threshold.json", "w") as f:
        json.dump({"best_model": best_model_name, "threshold": float(best_threshold), "f2_score": float(best_f2)}, f, indent=4)
        
    print("[Pillar 1] Champion model and artifacts saved to 'models/' successfully!")
    return results

if __name__ == "__main__":
    train_pillar1()
