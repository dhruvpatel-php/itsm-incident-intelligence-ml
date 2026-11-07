import json
import joblib

from src.config import MODELS_DIR, REPORTS_DIR
from src.verify_metrics import verify_all_metrics

def test_metrics_json_matches_reference():
    # Must exit cleanly with returncode 0
    try:
        verify_all_metrics()
    except SystemExit as e:
        assert e.code == 0, f"verify_all_metrics failed with exit code {e.code}"

def test_model_artifacts_contain_synced_metrics():
    with open(REPORTS_DIR / "metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)

    # Pillar 1
    p1 = joblib.load(MODELS_DIR / "pillar1_high_priority_model.joblib")
    assert "metrics" in p1
    assert p1["metrics"]["test_roc_auc"] == metrics["pillar1_early_warning"]["test_roc_auc"]
    assert p1["metrics"]["test_pr_auc"] == metrics["pillar1_early_warning"]["test_pr_auc"]

    # Pillar 2
    p2 = joblib.load(MODELS_DIR / "pillar2_volume_forecaster.joblib")
    assert "metrics" in p2
    assert p2["metrics"]["base_vol_2014"] == metrics["pillar2_forecasting"]["base_vol_2014"]

    # Pillar 3
    p3_router = joblib.load(MODELS_DIR / "pillar3_department_router.joblib")
    assert "metrics" in p3_router
    assert p3_router["metrics"]["variant_d_unweighted_accuracy"] == metrics["pillar3_routing"]["variant_d_unweighted_accuracy"]

    p3_prio = joblib.load(MODELS_DIR / "pillar3_priority_classifier.joblib")
    assert "metrics" in p3_prio
    assert p3_prio["metrics"]["imputation_cv_accuracy"] == metrics["pillar3_priority"]["imputation_cv_accuracy"]

    # Pillar 4
    p4 = joblib.load(MODELS_DIR / "pillar4_rfc_predictor.joblib")
    assert "metrics" in p4
    assert p4["metrics"]["rfc_roc_auc"] == metrics["pillar4_rfc_and_assets"]["rfc_roc_auc"]
