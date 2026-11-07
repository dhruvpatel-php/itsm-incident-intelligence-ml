import json
import joblib
import numpy as np

from src.config import MODELS_DIR, REPORTS_DIR

def test_forecasting_regime_and_horizon():
    p2 = joblib.load(MODELS_DIR / "pillar2_volume_forecaster.joblib")
    provenance = p2['provenance']

    assert provenance['regime_start'] == "2013-10-01"
    assert provenance['n_weeks'] == 26
    assert provenance['horizon'] == 2
    assert provenance['n_folds'] == 6
    assert provenance['block_length'] == 4
    assert provenance['bootstrap_b'] == 1000

def test_prediction_interval_invariance():
    with open(REPORTS_DIR / "metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)
    p2_m = metrics["pillar2_forecasting"]

    low = p2_m["low_vol_2014"]
    base = p2_m["base_vol_2014"]
    high = p2_m["high_vol_2014"]
    q1 = p2_m["q1_2014_actual"]
    future_39w = p2_m["future_39w_base"]

    assert low < base < high, f"Prediction interval violated: {low} < {base} < {high}"
    assert np.isclose(base, q1 + future_39w), f"Base volume {base} must equal Q1 {q1} + 39w {future_39w}"
    assert p2_m["scenario_2015_plus5pct"] > base

def test_naive_persistence_in_backtest():
    with open(REPORTS_DIR / "metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)
    mapes = metrics["pillar2_forecasting"]["backtest_mape"]

    assert "Naive" in mapes, "Naive persistence baseline must be evaluated in backtest"
    assert np.isclose(mapes["Naive"], 12.121370, atol=0.01)
    assert np.isclose(mapes["Ensemble"], 12.648016, atol=0.01)
    assert np.isclose(mapes["SARIMAX"], 13.174663, atol=0.01)
