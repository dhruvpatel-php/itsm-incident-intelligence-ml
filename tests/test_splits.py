import pandas as pd
import pytest

from src.config import CLEANED_DATA_PATH, STABLE_REGIME_START

@pytest.fixture(scope="module")
def df_clean():
    df = pd.read_csv(CLEANED_DATA_PATH, low_memory=False)
    df['Open_DT'] = pd.to_datetime(df['Open_DT'])
    return df

def test_pillar1_temporal_split(df_clean):
    df_stable = df_clean[df_clean['Open_DT'] >= STABLE_REGIME_START].sort_values('Open_DT').reset_index(drop=True)
    df_p1 = df_stable[df_stable['Is_High_Priority'].notnull()].copy().reset_index(drop=True)

    n_total = len(df_p1)
    n_tr = int(n_total * 0.65)
    n_val = int(n_total * 0.15)

    train_df = df_p1.iloc[:n_tr]
    val_df = df_p1.iloc[n_tr:n_tr + n_val]
    test_df = df_p1.iloc[n_tr + n_val:]

    # Assert strictly temporal chronological boundaries
    assert train_df['Open_DT'].max() <= val_df['Open_DT'].min(), "Pillar 1 Train timestamps must be <= Validation timestamps"
    assert val_df['Open_DT'].max() <= test_df['Open_DT'].min(), "Pillar 1 Validation timestamps must be <= Test timestamps"
    assert len(train_df) == 28672
    assert len(val_df) == 6616
    assert len(test_df) == 8824

def test_pillar3_routing_temporal_split(df_clean):
    df_stable = df_clean[df_clean['Open_DT'] >= STABLE_REGIME_START].sort_values('Open_DT').reset_index(drop=True)
    n_st = len(df_stable)
    n_tr = int(n_st * 0.65)
    n_val = int(n_st * 0.15)

    train_d = df_stable.iloc[:n_tr]
    val_d = df_stable.iloc[n_tr:n_tr + n_val]
    test_d = df_stable.iloc[n_tr + n_val:]

    assert train_d['Open_DT'].max() <= val_d['Open_DT'].min(), "Pillar 3 Train timestamps must be <= Validation timestamps"
    assert val_d['Open_DT'].max() <= test_d['Open_DT'].min(), "Pillar 3 Validation timestamps must be <= Test timestamps"

def test_pillar4_rfc_temporal_split(df_clean):
    df_stable = df_clean[df_clean['Open_DT'] >= STABLE_REGIME_START].sort_values('Open_DT').reset_index(drop=True)
    n_rfc = len(df_stable)
    n_tr = int(n_rfc * 0.65)
    n_val = int(n_rfc * 0.15)

    train_rfc = df_stable.iloc[:n_tr]
    val_rfc = df_stable.iloc[n_tr:n_tr + n_val]
    test_rfc = df_stable.iloc[n_tr + n_val:]

    assert train_rfc['Open_DT'].max() <= val_rfc['Open_DT'].min(), "Pillar 4 Train timestamps must be <= Validation timestamps"
    assert val_rfc['Open_DT'].max() <= test_rfc['Open_DT'].min(), "Pillar 4 Validation timestamps must be <= Test timestamps"
