import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.config import CLEANED_DATA_PATH, FIGURES_DIR, setup_logger

logger = setup_logger("EDA")

def run_eda(data_path=CLEANED_DATA_PATH):
    logger.info(f"Loading cleaned data from {data_path}...")
    df = pd.read_csv(data_path, low_memory=False)
    df['Open_DT'] = pd.to_datetime(df['Open_DT'])

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # 1. Priority Distribution Plot
    logger.info("Generating 01_priority_distribution.png...")
    fig, ax = plt.subplots(figsize=(8, 5))
    priority_counts = df['Priority_Clean'].value_counts().sort_index()
    colors = ['#d9534f', '#f0ad4e', '#5bc0de', '#0275d8', '#5cb85c']
    bars = ax.bar(priority_counts.index.astype(int).astype(str), priority_counts.values, color=colors, edgecolor='black', alpha=0.85)
    ax.set_title("ABC Tech ITIL Ticket Priority Distribution (P1 - P5)", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Priority Level", fontsize=12, labelpad=8)
    ax.set_ylabel("Number of Incidents", fontsize=12, labelpad=8)

    total_valid = priority_counts.sum()
    for bar in bars:
        height = bar.get_height()
        pct = (height / total_valid) * 100
        ax.annotate(f"{height:,}\n({pct:.1f}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylim(0, max(priority_counts.values) * 1.15)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "01_priority_distribution.png", dpi=300)
    plt.close(fig)

    # 2. Reassignments vs Resolution Time (Handle Time)
    logger.info("Generating 02_reassignments_vs_handletime.png...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    reassign_counts = df['No_of_Reassignments_Clean'].value_counts().sort_index()
    ax1.bar(reassign_counts.index[:10], reassign_counts.values[:10], color='#4575b4', edgecolor='black', alpha=0.85)
    ax1.set_title("Distribution of Ticket Reassignments (Hops 0 - 9)", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Number of Reassignments", fontsize=11)
    ax1.set_ylabel("Ticket Count", fontsize=11)
    for x, y in zip(reassign_counts.index[:10], reassign_counts.values[:10]):
        ax1.annotate(f"{y:,}", (x, y), textcoords="offset points", xytext=(0, 3), ha='center', fontsize=9)

    sub_df = df[df['Handle_Time_Clean'].notnull() & (df['No_of_Reassignments_Clean'] <= 8)].copy()
    median_time = sub_df.groupby('No_of_Reassignments_Clean')['Handle_Time_Clean'].median()
    ax2.plot(median_time.index, median_time.values, marker='o', linewidth=2.5, markersize=8, color='#d73027')
    ax2.set_title("Impact of Reassignments on Median Handle Time (Hours)", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Number of Reassignments", fontsize=11)
    ax2.set_ylabel("Median Handle Time (Hours)", fontsize=11)
    for x, y in zip(median_time.index, median_time.values):
        ax2.annotate(f"{y:.2f}h", (x, y), textcoords="offset points", xytext=(0, 6), ha='center', fontsize=9, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "02_reassignments_vs_handletime.png", dpi=300)
    plt.close(fig)

    # 3. Monthly Incident Volume Trend
    logger.info("Generating 03_incident_trend_monthly.png...")
    fig, ax = plt.subplots(figsize=(12, 5))
    df_ts = df.dropna(subset=['Open_DT']).copy()
    monthly = df_ts.set_index('Open_DT').resample('ME')['Incident_ID'].count()
    ax.plot(monthly.index.strftime('%Y-%m'), monthly.values, marker='s', linewidth=2.5, markersize=7, color='#2b83ba')
    ax.fill_between(range(len(monthly)), 0, monthly.values, color='#2b83ba', alpha=0.15)
    ax.set_title("Monthly IT Incident Inflow Trend (ABC Tech: 2012 - 2014)", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Year-Month", fontsize=12)
    ax.set_ylabel("Incident Inflow Volume", fontsize=12)
    ax.set_xticks(range(len(monthly)))
    ax.set_xticklabels(monthly.index.strftime('%Y-%m'), rotation=45, ha='right')
    for i, val in enumerate(monthly.values):
        ax.annotate(f"{val:,}", (i, val), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=9)
    ax.set_ylim(0, max(monthly.values) * 1.15)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "03_incident_trend_monthly.png", dpi=300)
    plt.close(fig)

    # 4. Department / CI_Cat Breakdown
    logger.info("Generating 04_department_breakdown.png...")
    fig, ax = plt.subplots(figsize=(10, 6))
    dept_col = 'CI_Cat' if 'CI_Cat' in df.columns else 'Department'
    dept_counts = df[dept_col].value_counts().head(8)
    bars = ax.barh(dept_counts.index[::-1], dept_counts.values[::-1], color='#31a354', edgecolor='black', alpha=0.85)
    ax.set_title("Incident Volume by IT Service Department (CI Category)", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Incident Count", fontsize=12)
    for bar in bars:
        width = bar.get_width()
        pct = (width / len(df)) * 100
        ax.annotate(f" {width:,} ({pct:.1f}%)",
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    va='center', ha='left', fontsize=9, fontweight='bold')
    ax.set_xlim(0, max(dept_counts.values) * 1.22)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "04_department_breakdown.png", dpi=300)
    plt.close(fig)

    # 5. Closure Codes / Misconfiguration Analysis
    logger.info("Generating 04b_closure_codes_root_causes.png...")
    fig, ax = plt.subplots(figsize=(10, 6))
    closure_counts = df['Closure_Code'].value_counts().head(8)
    bars = ax.barh(closure_counts.index[::-1], closure_counts.values[::-1], color='#756bb1', edgecolor='black', alpha=0.85)
    ax.set_title("Top Incident Closure Codes & Root Cause Categorization", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Incident Count", fontsize=12)
    for bar in bars:
        width = bar.get_width()
        pct = (width / len(df)) * 100
        ax.annotate(f" {width:,} ({pct:.1f}%)",
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    va='center', ha='left', fontsize=9, fontweight='bold')
    ax.set_xlim(0, max(closure_counts.values) * 1.20)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "04b_closure_codes_root_causes.png", dpi=300)
    plt.close(fig)

    logger.info("All EDA charts successfully generated and saved to 'figures/'!")

if __name__ == "__main__":
    run_eda()
