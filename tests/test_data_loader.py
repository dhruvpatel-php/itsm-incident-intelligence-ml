import numpy as np
import pandas as pd

from src.data_loader import parse_handle_time, load_and_clean_data
from src.config import RAW_DATA_PATH, PRIORITY_MATRIX

def test_parse_handle_time_cases():
    # Valid Indian comma grouped representations of decimal numbers
    assert np.isclose(parse_handle_time("3,87,16,91,111"), 3.871691111)
    assert np.isclose(parse_handle_time("0,862777778"), 0.862777778)
    assert np.isclose(parse_handle_time("1,954166667"), 1.954166667)
    assert np.isclose(parse_handle_time("12,345"), 12.345)

    # Standard numbers
    assert np.isclose(parse_handle_time(4.25), 4.25)
    assert np.isclose(parse_handle_time("4.25"), 4.25)
    assert np.isclose(parse_handle_time("0"), 0.0)
    assert np.isclose(parse_handle_time(0), 0.0)

    # Null and invalid values
    assert np.isnan(parse_handle_time(None))
    assert np.isnan(parse_handle_time(np.nan))
    assert np.isnan(parse_handle_time("nan"))
    assert np.isnan(parse_handle_time("None"))
    assert np.isnan(parse_handle_time(""))
    assert np.isnan(parse_handle_time("   "))
    assert np.isnan(parse_handle_time("invalid_text"))

def test_data_contract():
    raw_df = pd.read_csv(RAW_DATA_PATH, low_memory=False)
    # Shape contract: 46,606 rows x 25 columns
    assert raw_df.shape == (46606, 25), f"Expected shape (46606, 25), got {raw_df.shape}"

    # Impact contract: exactly 1,380 'NS' records
    ns_count = (raw_df['Impact'].astype(str).str.strip() == 'NS').sum()
    assert ns_count == 1380, f"Expected 1,380 NS tickets, got {ns_count}"

    # Priority contract: exactly 1,380 missing Priority values in raw data
    assert raw_df['Priority'].isnull().sum() == 1380

def test_itil_priority_matrix_match():
    raw_df = pd.read_csv(RAW_DATA_PATH, low_memory=False)
    raw_df['Impact_Num'] = pd.to_numeric(raw_df['Impact'].astype(str).str.strip(), errors='coerce')
    raw_df['Urgency_Num'] = pd.to_numeric(raw_df['Urgency'].astype(str).str.extract(r'(\d)')[0], errors='coerce')

    valid_mask = raw_df['Impact_Num'].notnull() & raw_df['Urgency_Num'].notnull() & raw_df['Priority'].notnull()
    valid_df = raw_df[valid_mask]
    assert len(valid_df) == 45226

    matches = valid_df.apply(
        lambda r: PRIORITY_MATRIX.get((int(r['Impact_Num']), int(r['Urgency_Num'])), -1) == int(r['Priority']),
        axis=1
    )
    assert matches.mean() == 1.0, f"Deterministic ITIL match rate was {matches.mean()*100:.2f}%, expected 100.00%"
    assert (~matches).sum() == 0, "Discrepancies found in deterministic ITIL matrix"

def test_handle_time_reproduction():
    df, metrics = load_and_clean_data(return_metrics=True)
    assert np.isclose(metrics['handle_time_median_hours'], 1.954167, atol=0.01)
    assert np.isclose(metrics['handle_time_mean_hours'], 4.2519, atol=0.05)
    assert np.isclose(metrics['handle_time_p75_hours'], 4.3711, atol=0.05)
    assert np.isclose(metrics['calendar_duration_median_hours'], 18.40, atol=0.05)
    assert np.isclose(metrics['handle_exceeds_calendar_pct'], 21.71, atol=0.05)
    assert np.isclose(metrics['pearson_r'], 0.1168, atol=0.002)
    assert np.isclose(metrics['spearman_rho'], 0.5684, atol=0.002)
