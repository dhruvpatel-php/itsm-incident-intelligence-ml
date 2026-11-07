import sys
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REPORTS_DIR = PROJECT_ROOT / "reports"
REFERENCE_PATH = REPORTS_DIR / "metrics_reference.json"
REGENERATED_PATH = REPORTS_DIR / "metrics.json"

# Defined strict tolerances per metric
DEFAULT_TOLERANCES = {
    # Data Loader
    "total_rows": 0,
    "total_columns": 0,
    "handle_time_median_hours": 0.01,
    "handle_time_mean_hours": 0.05,
    "handle_time_p75_hours": 0.05,
    "handle_exceeds_calendar_pct": 0.05,
    "pearson_r": 0.002,
    "spearman_rho": 0.002,
    "calendar_duration_median_hours": 0.05,

    # Pillar 1
    "test_roc_auc": 0.002,
    "test_pr_auc": 0.002,
    "test_pr_auc_ci_low": 0.01,
    "test_pr_auc_ci_high": 0.01,
    "prevalence_baseline": 0.001,
    "statistical_lift": 0.2,
    "validation_threshold": 0.005,
    "test_recall": 0.005,
    "test_recall_ci_low": 0.01,
    "test_recall_ci_high": 0.01,
    "test_precision": 0.005,
    "test_f2_score": 0.005,
    "brier_calibrated": 0.001,
    "brier_prevalence": 0.001,
    "brier_error_reduction_pct": 0.5,

    # Pillar 2
    "stable_regime_weeks": 0,
    "q1_2014_actual": 0,
    "future_39w_base": 2,
    "base_vol_2014": 2,
    "low_vol_2014": 2,
    "high_vol_2014": 2,
    "uncertainty_half_width": 2,
    "scenario_2015_plus5pct": 2,
    "Naive": 0.001,
    "Ensemble": 0.001,
    "SARIMAX": 0.001,
    "Holt_Winters": 0.001,
    "Seasonal_Naive": 0.001,

    # Pillar 3 Priority
    "verified_tickets": 0,
    "deterministic_match_rate": 0.0001,
    "discrepancies": 0,
    "ns_tickets": 0,
    "imputation_cv_accuracy": 0.005,
    "imputation_macro_f1": 0.005,
    "champion_p3_share": 0.005,
    "champion_p4_share": 0.005,
    "covariate_shift_ns_urgency3": 0.005,
    "covariate_shift_known_urgency3": 0.005,
    "flagged_human_review_count": 0,
    "flagged_human_review_pct": 0.1,

    # Pillar 3 Routing
    "majority_baseline": 0.001,
    "variant_d_unweighted_accuracy": 0.001,
    "variant_d_unweighted_balanced_acc": 0.001,
    "variant_d_unweighted_lift": 0.001,
    "variant_d_balanced_accuracy": 0.001,
    "variant_d_balanced_balanced_acc": 0.001,
    "variant_d_balanced_lift": 0.001,
    "variant_c_accuracy": 0.001,
    "variant_b_accuracy": 0.001,
    "variant_a_accuracy": 0.001,
    "wbs_unique_share": 0.05,
    "wbs_unique_ticket_share": 0.05,
    "delay_beta": 0.001,
    "delay_penalty_pct": 0.05,
    "delay_hours_saved_per_bounce": 0.01,
    "delay_hours_low": 0.01,
    "delay_hours_high": 0.01,
    "annual_hours_saved_base": 2,
    "annual_fte_saved_base": 0.01,
    "annual_hours_saved_low": 2,
    "annual_fte_saved_low": 0.01,
    "annual_hours_saved_high": 2,
    "annual_fte_saved_high": 0.01,

    # Pillar 4
    "rfc_roc_auc": 0.002,
    "rfc_pr_auc": 0.002,
    "evaluated_assets_count": 0,
    "min_spearman_rho": 0.002,
    "Critical": 0,
    "High": 0,
    "Medium": 0,
    "Low": 0
}

def verify_all_metrics():
    if not REFERENCE_PATH.exists():
        print(f"ERROR: Reference metrics file not found at {REFERENCE_PATH}")
        sys.exit(1)

    if not REGENERATED_PATH.exists():
        print(f"ERROR: Regenerated metrics file not found at {REGENERATED_PATH}")
        print("Please run `python -m src.run_all` first to generate metrics.json")
        sys.exit(1)

    with open(REFERENCE_PATH, "r", encoding="utf-8") as f:
        ref_data = json.load(f)

    with open(REGENERATED_PATH, "r", encoding="utf-8") as f:
        reg_data = json.load(f)

    print("=" * 105)
    print("EMPIRICAL METRICS REPRODUCTION AUDIT & VERIFICATION REPORT")
    print(f"Reference Source : {REFERENCE_PATH.name} (Audited Ground Truth from Notebook)")
    print(f"Regenerated Data : {REGENERATED_PATH.name}")
    print("=" * 105)
    print(f"{'Metric / Feature':<48} | {'Reference':<15} | {'Regenerated':<15} | {'Tol':<7} | {'Status'}")
    print("-" * 105)

    records = []
    failed_rows = []

    def check_value(path, ref_val, reg_val):
        key = path.split(".")[-1]
        tol = DEFAULT_TOLERANCES.get(key, 0.005)

        # Skip non-metric descriptors
        if key in ["model_name", "source_notebook", "description", "version", "pipeline_name", "execution_timestamp", "git_commit", "python_version", "platform", "raw_data_sha256", "library_versions"]:
            return

        status = "PASS"
        if isinstance(ref_val, (int, float)) and isinstance(reg_val, (int, float)):
            diff = abs(ref_val - reg_val)
            if tol == 0:
                is_pass = (ref_val == reg_val)
            else:
                is_pass = diff <= (tol + 1e-9)
            status = "PASS" if is_pass else "FAIL"
            ref_str = f"{ref_val:.4f}" if isinstance(ref_val, float) else f"{ref_val}"
            reg_str = f"{reg_val:.4f}" if isinstance(reg_val, float) else f"{reg_val}"
            records.append((path, ref_str, reg_str, str(tol), status))
            if not is_pass:
                failed_rows.append((path, ref_val, reg_val, tol))

        elif isinstance(ref_val, list) and isinstance(reg_val, list):
            if all(isinstance(x, (int, float)) for x in ref_val):
                diffs = [abs(r - g) for r, g in zip(ref_val, reg_val)]
                is_pass = len(ref_val) == len(reg_val) and all(d <= tol for d in diffs)
                status = "PASS" if is_pass else "FAIL"
                records.append((path, str(ref_val), str(reg_val), str(tol), status))
                if not is_pass:
                    failed_rows.append((path, ref_val, reg_val, tol))
            else:
                # String lists (e.g. top critical assets)
                is_pass = (ref_val == reg_val)
                status = "PASS" if is_pass else "FAIL"
                ref_preview = f"[{len(ref_val)} items: {', '.join(ref_val[:3])}...]"
                reg_preview = f"[{len(reg_val)} items: {', '.join(reg_val[:3])}...]"
                records.append((path, ref_preview, reg_preview, "exact", status))
                if not is_pass:
                    failed_rows.append((path, ref_val, reg_val, "exact"))

        elif isinstance(ref_val, dict) and isinstance(reg_val, dict):
            for sub_k, sub_ref in ref_val.items():
                if sub_k in reg_val:
                    check_value(f"{path}.{sub_k}", sub_ref, reg_val[sub_k])
                else:
                    records.append((f"{path}.{sub_k}", str(sub_ref), "MISSING", str(tol), "FAIL"))
                    failed_rows.append((f"{path}.{sub_k}", sub_ref, "MISSING", tol))

    for pillar_key, pillar_content in ref_data.items():
        if pillar_key == "metadata":
            continue
        if pillar_key not in reg_data:
            records.append((pillar_key, "PRESENT", "MISSING", "0", "FAIL"))
            failed_rows.append((pillar_key, "PRESENT", "MISSING", 0))
            continue
        if isinstance(pillar_content, dict):
            for k, val in pillar_content.items():
                if k in reg_data[pillar_key]:
                    check_value(f"{pillar_key}.{k}", val, reg_data[pillar_key][k])
                else:
                    records.append((f"{pillar_key}.{k}", str(val), "MISSING", "0", "FAIL"))
                    failed_rows.append((f"{pillar_key}.{k}", val, "MISSING", 0))

    # Print results table
    for path, ref_str, reg_str, tol_str, status in records:
        print(f"{path:<48} | {ref_str:<15} | {reg_str:<15} | {tol_str:<7} | {status}")

    print("-" * 105)
    total_checks = len(records)
    total_passes = total_checks - len(failed_rows)
    pass_rate = (total_passes / total_checks) * 100.0
    print(f"TOTAL METRICS CHECKED : {total_checks}")
    print(f"PASSED               : {total_passes} ({pass_rate:.1f}%)")
    print(f"FAILED               : {len(failed_rows)}")
    print("=" * 105)

    if failed_rows:
        print("\nFAILURE AUDIT DETAILS:")
        for path, ref_val, reg_val, tol in failed_rows:
            print(f"  ❌ {path}: expected {ref_val}, got {reg_val} (allowed tol: {tol})")
        print("\nGATE STATUS: FAILED. Please rectify discrepancy before proceeding to Phase 2.")
        sys.exit(1)
    else:
        print("\nGATE STATUS: ALL METRICS PASSED WITH 100% AUDITED REPRODUCIBILITY!")
        sys.exit(0)

if __name__ == "__main__":
    verify_all_metrics()
