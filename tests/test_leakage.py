import joblib

from src.config import MODELS_DIR

def test_pillar1_no_leakage_features():
    p1 = joblib.load(MODELS_DIR / "pillar1_high_priority_model.joblib")
    train_cols = p1['train_cols']

    # Strictly forbidden post-intake fields
    forbidden = [
        'Close_Time', 'Resolved_Time', 'Handle_Time', 'Handle_Time_hrs',
        'Handle_Time_Clean', 'Closure_Code', 'No_of_Reassignments',
        'No_of_Reassignments_Clean', 'KB_number', 'KB_number_freq',
        'Is_High_Priority', 'Priority', 'Priority_Clean',
        'CI_Subcat', 'CI_Name' # Raw asset text strings must not be passed bare
    ]

    for col in forbidden:
        assert col not in train_cols, f"Forbidden leakage feature '{col}' detected in Pillar 1 features!"

def test_pillar3_routing_no_leakage_features():
    p3 = joblib.load(MODELS_DIR / "pillar3_department_router.joblib")
    train_cols = p3['train_cols']

    # Target and post-triage fields
    forbidden = [
        'Dept_Consolidated', 'CI_Cat', 'CI_Subcat', 'CI_Name',
        'Alert_Status', 'Handle_Time_Clean', 'No_of_Reassignments_Clean'
    ]
    for col in forbidden:
        assert col not in train_cols, f"Forbidden leakage feature '{col}' detected in Pillar 3 routing features!"

def test_pillar4_rfc_no_leakage_features():
    p4 = joblib.load(MODELS_DIR / "pillar4_rfc_predictor.joblib")
    train_cols = p4['train_cols']

    # Circular change fields and lookahead artifacts
    forbidden = [
        'Related_Change', 'No_of_Related_Changes', 'Has_RFC',
        'Close_Code', 'Resolved_Time', 'Handle_Time', 'No_of_Reassignments',
        'Open_Month' # Excluded to prevent month artifact exploitation
    ]
    for col in forbidden:
        assert col not in train_cols, f"Forbidden leakage feature '{col}' detected in Pillar 4 RFC features!"
