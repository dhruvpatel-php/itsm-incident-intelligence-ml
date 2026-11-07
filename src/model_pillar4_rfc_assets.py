import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, roc_curve, precision_recall_curve, confusion_matrix
import xgboost as xgb
import lightgbm as lgb

from src.config import CLEANED_DATA_PATH, FIGURES_DIR, MODELS_DIR, RANDOM_STATE
from src.feature_engineering import extract_features

def build_rfc_and_assets():
    print("=" * 70)
    print("PILLAR 4: RFC PREDICTION & ITSM ASSET RISK SCORING")
    print("=" * 70)
    
    df = pd.read_csv(CLEANED_DATA_PATH, low_memory=False)
    
    # -------------------------------------------------------------
    # PART A: Request for Change (RFC) Prediction
    # -------------------------------------------------------------
    print("\n--- PART A: Request for Change (RFC) Prediction Model ---")
    y_rfc = df['Has_RFC'].astype(int).values
    print(f"Total Tickets: {len(y_rfc)} | Requiring RFC: {y_rfc.sum()} ({y_rfc.mean()*100:.2f}%)")
    
    train_idx, test_idx = train_test_split(
        df.index, test_size=0.20, random_state=RANDOM_STATE, stratify=y_rfc
    )
    
    df_train = df.loc[train_idx]
    df_test = df.loc[test_idx]
    
    X_train, encoders = extract_features(df_train, is_train=True)
    X_test = extract_features(df_test, is_train=False, encoders=encoders)
    
    y_train_rfc = df_train['Has_RFC'].astype(int).values
    y_test_rfc = df_test['Has_RFC'].astype(int).values
    
    scale_pos = (y_train_rfc == 0).sum() / (y_train_rfc == 1).sum()
    
    rfc_model = lgb.LGBMClassifier(
        n_estimators=150, max_depth=6, scale_pos_weight=scale_pos,
        learning_rate=0.05, random_state=RANDOM_STATE, verbose=-1, n_jobs=-1
    )
    rfc_model.fit(X_train, y_train_rfc)
    
    y_prob_rfc = rfc_model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test_rfc, y_prob_rfc)
    pr_auc = average_precision_score(y_test_rfc, y_prob_rfc)
    
    print(f"\n[RFC Prediction ROC-AUC]: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")
    
    # Classification at default & tuned threshold
    prec, rec, thresh = precision_recall_curve(y_test_rfc, y_prob_rfc)
    f1_scores = 2 * (prec * rec) / (prec + rec + 1e-9)
    best_t = thresh[np.argmax(f1_scores)]
    y_pred_rfc = (y_prob_rfc >= best_t).astype(int)
    
    print(f"[Optimal RFC Threshold]: {best_t:.3f} (Max F1: {np.max(f1_scores):.4f})")
    print(classification_report(y_test_rfc, y_pred_rfc, target_names=["Standard Incident", "RFC Required"]))
    
    # Plot RFC ROC & PR Curves
    print("[Pillar 4] Generating 11_pillar4_rfc_roc_curve.png...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fpr, tpr, _ = roc_curve(y_test_rfc, y_prob_rfc)
    ax1.plot(fpr, tpr, color='#984ea3', lw=2.5, label=f"LGBM (ROC-AUC = {roc_auc:.3f})")
    ax1.plot([0, 1], [0, 1], 'k--', lw=1.5)
    ax1.set_title("ROC Curve - Request for Change (RFC) Detection", fontsize=13, fontweight='bold')
    ax1.set_xlabel("False Positive Rate", fontsize=11)
    ax1.set_ylabel("True Positive Rate", fontsize=11)
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2.plot(rec, prec, color='#e41a1c', lw=2.5, label=f"LGBM (PR-AUC = {pr_auc:.3f})")
    ax2.axhline(y=y_train_rfc.mean(), color='k', linestyle='--', label=f"Baseline ({y_train_rfc.mean():.3f})")
    ax2.set_title("Precision-Recall Curve - RFC Prediction", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Recall", fontsize=11)
    ax2.set_ylabel("Precision", fontsize=11)
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "11_pillar4_rfc_roc_curve.png", dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # PART B: ITSM Asset Failure & Misconfiguration Risk Scoring
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PART B: CONFIGURATION ITEM (CI) ASSET RISK REGISTRY")
    print("=" * 70)
    
    # Define misconfiguration closure codes
    misconfig_codes = ['Operator error', 'User error', 'No error - works as designed', 'Unknown', 'User manual not used']
    df['Is_Misconfig'] = df['Closure_Code'].isin(misconfig_codes).astype(int)
    
    # Group by CI_Name
    ci_groups = df.groupby('CI_Name').agg(
        Total_Incidents=('Incident_ID', 'count'),
        CI_Cat=('CI_Cat', lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'),
        CI_Subcat=('CI_Subcat', lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'),
        P1_P2_Incidents=('Is_High_Priority', 'sum'),
        RFC_Incidents=('Has_RFC', 'sum'),
        Repeat_Incidents=('No_of_Related_Incidents_Clean', 'sum'),
        Misconfig_Incidents=('Is_Misconfig', 'sum'),
        Avg_Reassignments=('No_of_Reassignments_Clean', 'mean'),
        Median_Handle_Time=('Handle_Time_Clean', 'median')
    ).reset_index()
    
    # Calculate operational risk rates
    ci_groups['P1_P2_Rate'] = ci_groups['P1_P2_Incidents'] / ci_groups['Total_Incidents']
    ci_groups['RFC_Rate'] = ci_groups['RFC_Incidents'] / ci_groups['Total_Incidents']
    ci_groups['Misconfig_Rate'] = ci_groups['Misconfig_Incidents'] / ci_groups['Total_Incidents']
    ci_groups['Repeat_Rate'] = ci_groups['Repeat_Incidents'] / ci_groups['Total_Incidents']
    
    # Normalize volume with log scale
    max_log_vol = np.log1p(ci_groups['Total_Incidents'].max())
    ci_groups['Norm_Volume'] = np.log1p(ci_groups['Total_Incidents']) / max_log_vol
    
    # Composite Asset Risk Index (0 - 100)
    ci_groups['Asset_Risk_Index'] = (
        0.30 * ci_groups['Norm_Volume'] +
        0.25 * ci_groups['P1_P2_Rate'] +
        0.20 * ci_groups['RFC_Rate'] +
        0.15 * ci_groups['Misconfig_Rate'] +
        0.10 * np.clip(ci_groups['Repeat_Rate'], 0, 1)
    ) * 100.0
    
    ci_registry = ci_groups.sort_values(by='Asset_Risk_Index', ascending=False)
    
    # Save Risk Scores
    risk_csv = MODELS_DIR / "pillar4_asset_risk_scores.csv"
    ci_registry.to_csv(risk_csv, index=False)
    print(f"[Pillar 4] Asset Risk Registry saved to {risk_csv} ({len(ci_registry)} assets profiled)")
    
    print("\n[Top 10 Highest Risk ITSM Configuration Items (CIs)]:")
    cols_display = ['CI_Name', 'CI_Cat', 'Total_Incidents', 'P1_P2_Incidents', 'RFC_Incidents', 'Misconfig_Rate', 'Asset_Risk_Index']
    print(ci_registry[cols_display].head(10).to_string(index=False))
    
    # Asset Risk Matrix Plot
    print("\n[Pillar 4] Generating 12_pillar4_ci_asset_risk_matrix.png...")
    fig, ax = plt.subplots(figsize=(12, 7))
    scatter = ax.scatter(
        ci_groups['Total_Incidents'],
        ci_groups['Asset_Risk_Index'],
        c=ci_groups['P1_P2_Rate'],
        cmap='YlOrRd',
        s=40,
        alpha=0.75,
        edgecolor='black',
        linewidth=0.5
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("High Priority Ticket Rate", fontsize=11)
    
    # Annotate distinct top assets with staggered offsets to prevent overlapping labels
    chosen_assets = []
    seen_coords = set()
    for _, row in ci_registry.iterrows():
        coord = (row['Total_Incidents'], round(row['Asset_Risk_Index'], 0))
        if coord not in seen_coords:
            seen_coords.add(coord)
            chosen_assets.append(row)
        if len(chosen_assets) >= 6:
            break
            
    top_assets = pd.DataFrame(chosen_assets)
    offsets = [
        (15, 15),     # Asset 1: upper right
        (18, -25),    # Asset 2: lower right
        (15, 10),     # Asset 3: right
        (18, -20),    # Asset 4: lower right
        (-135, 15),   # Asset 5: upper left
        (15, -25)     # Asset 6: lower right
    ]
    
    for i, (_, row) in enumerate(top_assets.iterrows()):
        offset = offsets[i % len(offsets)]
        ax.annotate(
            f"{row['CI_Name']} ({row['CI_Cat']})\nRisk: {row['Asset_Risk_Index']:.1f}",
            (row['Total_Incidents'], row['Asset_Risk_Index']),
            xytext=offset, textcoords='offset points',
            fontsize=9, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#94a3b8", alpha=0.92, lw=0.8),
            arrowprops=dict(arrowstyle="->", color='#334155', lw=1.2, connectionstyle="arc3,rad=0.08")
        )
        
    ax.set_xscale('log')
    ax.set_title("ITSM Configuration Item (CI) Failure & Misconfiguration Risk Matrix", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Total Incident Frequency (Log Scale)", fontsize=12)
    ax.set_ylabel("Asset Risk Index (0 - 100)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "12_pillar4_ci_asset_risk_matrix.png", dpi=300)
    plt.close(fig)

    # Save RFC Model
    joblib.dump({"model": rfc_model, "encoders": encoders, "threshold": float(best_t), "roc_auc": float(roc_auc)}, MODELS_DIR / "pillar4_rfc_predictor.joblib")
    print("[Pillar 4] RFC Predictor model saved successfully!")

if __name__ == "__main__":
    build_rfc_and_assets()
