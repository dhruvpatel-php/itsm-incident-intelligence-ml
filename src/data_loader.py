import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from src.config import (
    RAW_DATA_PATH,
    CLEANED_DATA_PATH,
    PRIORITY_MATRIX,
    setup_logger
)

logger = setup_logger("DataLoader")

def parse_handle_time(val):
    """
    Validated handle time parser where the first comma serves as decimal point,
    and any subsequent commas are stripped (regional Indian numeral format export).
    e.g. '3,87,16,91,111' -> 3.871691111 hours
         '0,862777778'    -> 0.862777778 hours
         '2,04'           -> 2.04 hours
         '15.5'           -> 15.5 hours
    """
    if pd.isnull(val):
        return np.nan
    val = str(val).strip()
    if not val or val.lower() == 'nan':
        return np.nan
    if ',' in val:
        parts = val.split(',')
        int_part = parts[0].strip()
        dec_part = ''.join(parts[1:]).strip()
        try:
            return float(f"{int_part}.{dec_part}")
        except ValueError:
            return np.nan
    try:
        return float(val)
    except ValueError:
        return np.nan

def load_and_clean_data(raw_path=RAW_DATA_PATH, save_cleaned=True, return_metrics=False):
    logger.info(f"Loading raw dataset from {raw_path}...")
    df = pd.read_csv(raw_path, low_memory=False)
    n_rows, n_cols = df.shape
    logger.info(f"Raw shape: {n_rows:,} incidents x {n_cols} attributes.")

    # 1. Standardize string fields
    str_cols = ['CI_Name', 'CI_Cat', 'CI_Subcat', 'WBS', 'Incident_ID', 'Category', 'Closure_Code', 'KB_number']
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({'nan': np.nan, 'None': np.nan, '': np.nan})

    # 2. Datetime parsing
    df['Open_DT'] = pd.to_datetime(df['Open_Time'], format='%d-%m-%Y %H:%M', errors='coerce')
    df['Resolved_DT'] = pd.to_datetime(df['Resolved_Time'], format='%d-%m-%Y %H:%M', errors='coerce')
    df['Close_DT'] = pd.to_datetime(df['Close_Time'], format='%d-%m-%Y %H:%M', errors='coerce')
    df['Reopen_DT'] = pd.to_datetime(df['Reopen_Time'], format='%d-%m-%Y %H:%M', errors='coerce')

    # 3. Handle Time and Calendar Duration
    df['Handle_Time_Clean'] = df['Handle_Time_hrs'].apply(parse_handle_time)
    df['Calendar_Duration_hrs'] = (df['Close_DT'] - df['Open_DT']).dt.total_seconds() / 3600.0

    # 4. Impact, Urgency and Priority (Deterministic ITIL Matrix)
    df['Impact_Clean'] = df['Impact'].astype(str).str.strip()
    df['Urgency_Clean'] = df['Urgency'].astype(str).str.extract(r'(\d)')[0].fillna(np.nan)
    df['Urgency_Num'] = pd.to_numeric(df['Urgency_Clean'], errors='coerce')
    df['Impact_Num'] = pd.to_numeric(df['Impact_Clean'], errors='coerce')

    def get_priority(row):
        if pd.notnull(row['Priority']):
            try:
                return int(row['Priority'])
            except (ValueError, TypeError):
                pass
        imp, urg = row['Impact_Num'], row['Urgency_Num']
        if pd.notnull(imp) and pd.notnull(urg):
            return PRIORITY_MATRIX.get((int(imp), int(urg)), np.nan)
        return np.nan

    df['Priority_Clean'] = df.apply(get_priority, axis=1)
    df['Is_High_Priority'] = df['Priority_Clean'].apply(
        lambda p: 1 if p in [1, 2] else (0 if pd.notnull(p) else np.nan)
    )

    # 5. Targets and Operational Features
    df['Has_RFC'] = (
        (df['No_of_Related_Changes'] > 0) |
        (df['Related_Change'].notnull() & (df['Related_Change'] != ''))
    ).astype(int)

    df['No_of_Reassignments_Clean'] = pd.to_numeric(df['No_of_Reassignments'], errors='coerce').fillna(0).astype(int)

    # Department consolidation (Top 5 + other_infrastructure)
    df['Department'] = df['CI_Cat']
    top_depts = ['application', 'subapplication', 'computer', 'storage', 'hardware']
    df['Dept_Consolidated'] = df['CI_Cat'].apply(
        lambda x: x if x in top_depts else ('other_infrastructure' if pd.notnull(x) else 'other_infrastructure')
    )

    # Temporal feature engineering
    df['Open_Hour'] = df['Open_DT'].dt.hour
    df['Open_DayOfWeek'] = df['Open_DT'].dt.day_of_week
    df['Open_Month'] = df['Open_DT'].dt.month
    df['Is_Weekend'] = df['Open_DayOfWeek'].isin([5, 6]).astype(int)

    # 6. Verification Metrics
    ht_v = df['Handle_Time_Clean'].dropna()
    cal_v = df['Calendar_Duration_hrs'].dropna()
    sub_paired = df[['Handle_Time_Clean', 'Calendar_Duration_hrs']].dropna()
    pct_ht_exceeds_cal = float((sub_paired['Handle_Time_Clean'] > sub_paired['Calendar_Duration_hrs']).mean() * 100.0)
    r_pearson = float(sub_paired.corr().iloc[0, 1])
    r_spearman = float(sub_paired.corr(method='spearman').iloc[0, 1])

    logger.info("=== HANDLE TIME VALIDATION ===")
    logger.info(f"Active Handle Time (hours) : Median = {ht_v.median():.2f}h, Mean = {ht_v.mean():.2f}h, 75th = {ht_v.quantile(0.75):.2f}h")
    logger.info(f"Calendar Duration (hours)  : Median = {cal_v.median():.2f}h, Mean = {cal_v.mean():.2f}h")
    logger.info(f"Handle > Calendar Rate     : {pct_ht_exceeds_cal:.1f}%")
    logger.info(f"Correlation (Pearson/Spearman) : {r_pearson:.4f} / {r_spearman:.4f}")

    dl_metrics = {
        "total_rows": int(n_rows),
        "total_columns": int(n_cols),
        "handle_time_median_hours": round(float(ht_v.median()), 6),
        "handle_time_mean_hours": round(float(ht_v.mean()), 4),
        "handle_time_p75_hours": round(float(ht_v.quantile(0.75)), 4),
        "handle_exceeds_calendar_pct": round(float(pct_ht_exceeds_cal), 2),
        "pearson_r": round(float(r_pearson), 4),
        "spearman_rho": round(float(r_spearman), 4),
        "calendar_duration_median_hours": round(float(cal_v.median()), 2)
    }

    if save_cleaned:
        logger.info(f"Saving cleaned dataset to {CLEANED_DATA_PATH}...")
        df.to_csv(CLEANED_DATA_PATH, index=False)
        logger.info("Cleaned dataset successfully saved.")

    if return_metrics:
        return df, dl_metrics
    return df

if __name__ == "__main__":
    load_and_clean_data()
