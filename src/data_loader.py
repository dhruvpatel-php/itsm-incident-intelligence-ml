import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from src.config import RAW_DATA_PATH, CLEANED_DATA_PATH, PRIORITY_MATRIX

def parse_handle_time(val):
    """
    Reconstructs handle time from corrupted/Indian number formatted string.
    e.g. '3,87,16,91,111' -> 3871.691111 hours
         '0,862777778'    -> 0.862778 hours
         '2,04'           -> 2.04 hours
    """
    if pd.isnull(val):
        return np.nan
    val = str(val).strip()
    if not val or val == 'nan':
        return np.nan
    if val.startswith('0,'):
        return float('0.' + val[2:].replace(',', ''))
    if ',' not in val:
        try:
            return float(val)
        except ValueError:
            pass
    if re.match(r'^\d+,\d{1,2}$', val):
        return float(val.replace(',', '.'))
    
    digits = re.sub(r'[^\d]', '', val)
    if len(digits) >= 7:
        int_part = digits[:-6]
        dec_part = digits[-6:]
        return float(f"{int_part}.{dec_part}")
    elif len(digits) > 0:
        return float(digits)
    return np.nan

def load_and_clean_data(raw_path=RAW_DATA_PATH, save_cleaned=True):
    print(f"[DataLoader] Reading raw dataset from {raw_path}...")
    df = pd.read_csv(raw_path, low_memory=False)
    initial_rows, initial_cols = df.shape
    print(f"[DataLoader] Raw shape: {initial_rows} rows x {initial_cols} columns")

    # 1. Standardize text strings
    for col in ['CI_Name', 'CI_Cat', 'CI_Subcat', 'WBS', 'Incident_ID', 'Category', 'Closure_Code', 'KB_number']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({'nan': np.nan, 'None': np.nan, '': np.nan})

    # 2. Parse Datetime fields
    print("[DataLoader] Parsing datetime fields...")
    df['Open_DT'] = pd.to_datetime(df['Open_Time'], format='%d-%m-%Y %H:%M', errors='coerce')
    df['Resolved_DT'] = pd.to_datetime(df['Resolved_Time'], format='%d-%m-%Y %H:%M', errors='coerce')
    df['Close_DT'] = pd.to_datetime(df['Close_Time'], format='%d-%m-%Y %H:%M', errors='coerce')
    df['Reopen_DT'] = pd.to_datetime(df['Reopen_Time'], format='%d-%m-%Y %H:%M', errors='coerce')

    # 3. Clean Handle Time
    print("[DataLoader] Cleaning Handle_Time_hrs...")
    df['Handle_Time_Clean'] = df['Handle_Time_hrs'].apply(parse_handle_time)

    # Calculate actual calendar duration in hours as benchmark
    df['Calendar_Duration_hrs'] = (df['Close_DT'] - df['Open_DT']).dt.total_seconds() / 3600.0

    # 4. Clean Impact & Urgency
    print("[DataLoader] Processing Impact, Urgency, and Priority...")
    df['Impact_Clean'] = df['Impact'].astype(str).str.strip()
    df['Urgency_Clean'] = df['Urgency'].astype(str).str.extract(r'(\d)')[0].fillna(np.nan)
    df['Urgency_Num'] = pd.to_numeric(df['Urgency_Clean'], errors='coerce')
    df['Impact_Num'] = pd.to_numeric(df['Impact_Clean'], errors='coerce')

    # Impute missing Priority using Priority Matrix where possible
    def get_priority(row):
        if pd.notnull(row['Priority']):
            return int(row['Priority'])
        imp, urg = row['Impact_Num'], row['Urgency_Num']
        if pd.notnull(imp) and pd.notnull(urg):
            return PRIORITY_MATRIX.get((int(imp), int(urg)), np.nan)
        return np.nan

    df['Priority_Clean'] = df.apply(get_priority, axis=1)

    # 5. Define Targets for the 4 Business Pillars
    # Pillar 1 Target: High Priority Ticket (P1 / P2)
    df['Is_High_Priority'] = df['Priority_Clean'].apply(lambda p: 1 if p in [1, 2] else (0 if pd.notnull(p) else np.nan))

    # Pillar 3 Target: Department (Consolidate low-frequency CI_Cat)
    top_departments = ['application', 'subapplication', 'computer', 'storage', 'hardware', 'software', 'database', 'displaydevice', 'officeelectronics', 'networkcomponents']
    df['Department'] = df['CI_Cat'].apply(lambda x: x if x in top_departments else ('Other' if pd.notnull(x) else 'Unknown'))

    # Pillar 4 Target: RFC (Request for Change)
    df['Has_RFC'] = ((df['No_of_Related_Changes'] > 0) | (df['Related_Change'].notnull() & (df['Related_Change'] != '')) | (df['Category'] == 'request for change')).astype(int)

    # Reopening and related interactions
    df['Has_Reopened'] = df['Reopen_DT'].notnull().astype(int)
    df['No_of_Reassignments_Clean'] = df['No_of_Reassignments'].fillna(0).astype(int)
    df['No_of_Related_Interactions_Clean'] = df['No_of_Related_Interactions'].fillna(0).astype(int)
    df['No_of_Related_Incidents_Clean'] = df['No_of_Related_Incidents'].fillna(0).astype(int)

    # Temporal feature engineering
    df['Open_Year'] = df['Open_DT'].dt.year
    df['Open_Month'] = df['Open_DT'].dt.month
    df['Open_Quarter'] = df['Open_DT'].dt.to_period('Q').astype(str)
    df['Open_DayOfWeek'] = df['Open_DT'].dt.day_of_week # 0=Mon, 6=Sun
    df['Open_Hour'] = df['Open_DT'].dt.hour
    df['Open_DayOfMonth'] = df['Open_DT'].dt.day
    df['Is_Weekend'] = df['Open_DayOfWeek'].apply(lambda d: 1 if d in [5, 6] else 0)

    print(f"[DataLoader] High Priority Tickets: {df['Is_High_Priority'].sum()} / {df['Is_High_Priority'].count()} ({(df['Is_High_Priority'].mean()*100):.2f}%)")
    print(f"[DataLoader] Tickets requiring RFC: {df['Has_RFC'].sum()} / {len(df)} ({(df['Has_RFC'].mean()*100):.2f}%)")

    if save_cleaned:
        print(f"[DataLoader] Saving cleaned data to {CLEANED_DATA_PATH}...")
        df.to_csv(CLEANED_DATA_PATH, index=False)
        print("[DataLoader] Successfully saved!")

    return df

if __name__ == "__main__":
    load_and_clean_data()
