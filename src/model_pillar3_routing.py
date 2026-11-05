import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

from src.config import CLEANED_DATA_PATH, FIGURES_DIR, MODELS_DIR, RANDOM_STATE
from src.feature_engineering import extract_features

def build_routing():
    print("=" * 70)
    print("PILLAR 3: AUTO-TAGGING TICKETS & REASSIGNMENT DELAY REDUCTION")
    print("=" * 70)
    
    df = pd.read_csv(CLEANED_DATA_PATH, low_memory=False)
    
    # -------------------------------------------------------------
    # PART A: Department / CI_Cat Auto-Routing
    # -------------------------------------------------------------
    print("\n--- PART A: Department Auto-Routing Classifier ---")
    valid_dept = df[df['Department'].notnull()].copy()
    y_dept = valid_dept['Department']
    
    train_idx, test_idx = train_test_split(
        valid_dept.index, test_size=0.20, random_state=RANDOM_STATE, stratify=y_dept
    )
    
    df_train = valid_dept.loc[train_idx]
    df_test = valid_dept.loc[test_idx]
    
    # Feature extraction excluding CI_Cat
    X_train, encoders = extract_features(df_train, is_train=True)
    X_test = extract_features(df_test, is_train=False, encoders=encoders)
    
    # Remove CI_Cat dummy columns from features since CI_Cat is the target!
    ci_cat_cols = [c for c in X_train.columns if c.startswith('ci_cat_')]
    X_train_dept = X_train.drop(columns=ci_cat_cols)
    X_test_dept = X_test.drop(columns=ci_cat_cols)
    
    y_train_dept = df_train['Department'].values
    y_test_dept = df_test['Department'].values
    
    print(f"[Pillar 3] Department Training samples: {len(y_train_dept)}, Test samples: {len(y_test_dept)}")
    print(f"Target Department classes: {sorted(list(set(y_train_dept)))}")
    
    dept_model = lgb.LGBMClassifier(
        n_estimators=150, max_depth=8, learning_rate=0.08,
        random_state=RANDOM_STATE, verbose=-1, n_jobs=-1
    )
    dept_model.fit(X_train_dept, y_train_dept)
    
    y_pred_dept = dept_model.predict(X_test_dept)
    dept_acc = accuracy_score(y_test_dept, y_pred_dept)
    dept_f1 = f1_score(y_test_dept, y_pred_dept, average='weighted')
    
    print(f"\n[Department Auto-Routing Accuracy]: {dept_acc*100:.2f}% | Weighted F1: {dept_f1:.4f}")
    print("\nClassification Report (Department Routing):")
    print(classification_report(y_test_dept, y_pred_dept, zero_division=0))
    
    # Department Confusion Matrix
    print("[Pillar 3] Generating 09_pillar3_department_confusion_matrix.png...")
    labels_dept = sorted(list(set(y_test_dept)))
    cm_dept = confusion_matrix(y_test_dept, y_pred_dept, labels=labels_dept)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_dept, annot=True, fmt='d', cmap='Blues', xticklabels=labels_dept, yticklabels=labels_dept, ax=ax)
    ax.set_title("Department Auto-Routing Confusion Matrix", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Predicted Department", fontsize=11)
    ax.set_ylabel("Actual Department", fontsize=11)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "09_pillar3_department_confusion_matrix.png", dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # PART B: Multi-Class Priority Auto-Tagging (P1 to P5)
    # -------------------------------------------------------------
    print("\n--- PART B: Priority Level Auto-Tagging (Multi-Class) ---")
    valid_prio = df[df['Priority_Clean'].notnull()].copy()
    y_prio = valid_prio['Priority_Clean'].astype(int)
    
    train_prio_idx, test_prio_idx = train_test_split(
        valid_prio.index, test_size=0.20, random_state=RANDOM_STATE, stratify=y_prio
    )
    
    df_train_p = valid_prio.loc[train_prio_idx]
    df_test_p = valid_prio.loc[test_prio_idx]
    
    X_train_p, enc_p = extract_features(df_train_p, is_train=True)
    X_test_p = extract_features(df_test_p, is_train=False, encoders=enc_p)
    
    y_train_p = df_train_p['Priority_Clean'].astype(int).values
    y_test_p = df_test_p['Priority_Clean'].astype(int).values
    
    prio_model = RandomForestClassifier(
        n_estimators=150, max_depth=12, class_weight='balanced',
        random_state=RANDOM_STATE, n_jobs=-1
    )
    prio_model.fit(X_train_p, y_train_p)
    
    y_pred_p = prio_model.predict(X_test_p)
    prio_acc = accuracy_score(y_test_p, y_pred_p)
    prio_f1 = f1_score(y_test_p, y_pred_p, average='weighted')
    
    print(f"\n[Priority Auto-Tagging Accuracy]: {prio_acc*100:.2f}% | Weighted F1: {prio_f1:.4f}")
    print("\nClassification Report (Priority Auto-Tagging):")
    print(classification_report(y_test_p, y_pred_p, zero_division=0))
    
    # Priority Confusion Matrix
    print("[Pillar 3] Generating 10_pillar3_priority_confusion_matrix.png...")
    labels_prio = sorted(list(set(y_test_p)))
    cm_prio = confusion_matrix(y_test_p, y_pred_p, labels=labels_prio)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_prio, annot=True, fmt='d', cmap='Oranges', xticklabels=[f"P{p}" for p in labels_prio], yticklabels=[f"P{p}" for p in labels_prio], ax=ax)
    ax.set_title("Multi-Class Ticket Priority Classification Confusion Matrix", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Predicted Priority", fontsize=11)
    ax.set_ylabel("Actual Priority", fontsize=11)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "10_pillar3_priority_confusion_matrix.png", dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # PART C: Reassignment and SLA Delay Impact Simulation
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PART C: REASSIGNMENT & SLA DELAY REDUCTION SIMULATION")
    print("=" * 70)
    
    reassign_df = df[df['Handle_Time_Clean'].notnull() & (df['Handle_Time_Clean'] > 0)].copy()
    zero_hops = reassign_df[reassign_df['No_of_Reassignments_Clean'] == 0]['Handle_Time_Clean']
    multi_hops = reassign_df[reassign_df['No_of_Reassignments_Clean'] > 0]['Handle_Time_Clean']
    
    med_zero = zero_hops.median()
    med_multi = multi_hops.median()
    mean_hops = df['No_of_Reassignments_Clean'].mean()
    reassigned_pct = (df['No_of_Reassignments_Clean'] > 0).mean() * 100
    
    # Estimated delay per reassignment hop (difference in median handle time)
    delay_saved_per_ticket = max(0, med_multi - med_zero)
    annual_volume = 80000 # Projected 2014 volume
    annual_reassigned_tickets = annual_volume * (reassigned_pct / 100)
    
    # Auto-routing routing precision (from dept_acc)
    # Tickets correctly routed on first touch avoid reassignments
    annual_hours_saved = annual_reassigned_tickets * dept_acc * delay_saved_per_ticket
    
    print(f"Current Reassigned Incident Rate:      {reassigned_pct:.1f}% ({df['No_of_Reassignments_Clean'].gt(0).sum():,} tickets)")
    print(f"Current Mean Reassignments per Ticket: {mean_hops:.2f} hops")
    print(f"Median Handle Time (Zero Hops):        {med_zero:.2f} hours")
    print(f"Median Handle Time (>=1 Hops):         {med_multi:.2f} hours")
    print(f"Resolution Delay Due to Reassignments: {delay_saved_per_ticket:.2f} hours/ticket")
    print(f"\n[Projected Business Value of Auto-Routing]:")
    print(f"  • Annual Reassignment Reduction:     -{dept_acc*100:.1f}%")
    print(f"  • Estimated Annual Engineering Hours Saved: {annual_hours_saved:,.0f} hours")
    print(f"  • Resolution Time Acceleration:      Up to {((med_multi - med_zero)/med_multi)*100:.1f}% MTTR reduction for misrouted tickets")
    
    # Save Models
    joblib.dump({"model": dept_model, "encoders": encoders, "accuracy": dept_acc}, MODELS_DIR / "pillar3_department_router.joblib")
    joblib.dump({"model": prio_model, "encoders": enc_p, "accuracy": prio_acc}, MODELS_DIR / "pillar3_priority_classifier.joblib")
    print("\n[Pillar 3] Models and simulation artifacts successfully saved!")

if __name__ == "__main__":
    build_routing()
