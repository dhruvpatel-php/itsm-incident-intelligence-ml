import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def build_early_warning_features(data, ci_freq, wbs_freq, train_cols=None):
    """
    Builds leakage-free intake features for Pillar 1 Early Warning.
    Excludes manual triage fields (Impact, Urgency, KB_number, CI_Subcat).
    """
    X = pd.get_dummies(data[['Category', 'CI_Cat']], drop_first=True, dtype=float)
    X['CI_Name_Freq'] = data['CI_Name'].map(ci_freq).fillna(0).values
    X['WBS_Freq'] = data['WBS'].map(wbs_freq).fillna(0).values
    X['Open_Hour'] = data['Open_DT'].dt.hour.values
    X['Open_DayOfWeek'] = data['Open_DT'].dt.day_of_week.values
    X['Is_Weekend'] = X['Open_DayOfWeek'].isin([5, 6]).astype(int).values

    if train_cols is not None:
        X = X.reindex(columns=train_cols, fill_value=0.0)
    return X

def build_routing_variant_d_features(data, wbs_freq, train_cols=None):
    """
    Builds intake-safe features for Pillar 3 Variant D auto-routing:
    Category + WBS Frequency + Arrival Timing.
    Alert_Status is dropped because it is constant ('closed').
    """
    X = pd.get_dummies(data[['Category']], drop_first=True, dtype=float)
    X['WBS_Freq'] = data['WBS'].map(wbs_freq).fillna(0).values
    X['Open_Hour'] = data['Open_DT'].dt.hour.values
    X['Open_DayOfWeek'] = data['Open_DT'].dt.day_of_week.values
    X['Is_Weekend'] = X['Open_DayOfWeek'].isin([5, 6]).astype(int).values

    if train_cols is not None:
        X = X.reindex(columns=train_cols, fill_value=0.0)
    return X

def build_rfc_features(data, ci_freq, wbs_freq, train_cols=None):
    """
    Builds intake-safe features for Pillar 4 Proactive RFC Prediction.
    Excludes circular leakage fields (Related_Change, No_of_Related_Changes, Open_Month).
    """
    X = pd.get_dummies(data[['CI_Cat', 'Category']], drop_first=True, dtype=float)
    X['CI_Name_Freq'] = data['CI_Name'].map(ci_freq).fillna(0).values
    X['WBS_Freq'] = data['WBS'].map(wbs_freq).fillna(0).values
    X['Open_Hour'] = data['Open_DT'].dt.hour.values
    X['Open_DayOfWeek'] = data['Open_DT'].dt.day_of_week.values
    X['Is_Weekend'] = X['Open_DayOfWeek'].isin([5, 6]).astype(int).values

    if train_cols is not None:
        X = X.reindex(columns=train_cols, fill_value=0.0)
    return X
