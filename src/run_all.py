import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_and_clean_data
from src.eda import run_eda
from src.model_pillar1_priority import train_pillar1
from src.model_pillar2_forecasting import build_forecasting
from src.model_pillar3_routing import build_routing
from src.model_pillar4_rfc_assets import build_rfc_and_assets

def main():
    start_time = time.time()
    print("=" * 80)
    print("ABC TECH - ITSM MACHINE LEARNING END-TO-END PIPELINE (PR-0012)")
    print("Senior Data Scientist Solution Suite")
    print("=" * 80)
    
    print("\n>>> STEP 1: DATA INGESTION, AUDITING & CLEANING...")
    load_and_clean_data()
    
    print("\n>>> STEP 2: STATISTICAL EXPLORATORY DATA ANALYSIS (EDA)...")
    run_eda()
    
    print("\n>>> STEP 3: PILLAR 1 - HIGH PRIORITY INCIDENT EARLY WARNING...")
    train_pillar1()
    
    print("\n>>> STEP 4: PILLAR 2 - INCIDENT VOLUME FORECASTING (QUARTERLY & ANNUAL)...")
    build_forecasting()
    
    print("\n>>> STEP 5: PILLAR 3 - INTELLIGENT TICKET ROUTING & SLA OPTIMIZATION...")
    build_routing()
    
    print("\n>>> STEP 6: PILLAR 4 - RFC PREDICTION & ITSM ASSET RISK PROFILING...")
    build_rfc_and_assets()
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETE IN {elapsed:.2f} SECONDS!")
    print("All production models, forecasts, figures, and risk registries generated.")
    print("=" * 80)

if __name__ == "__main__":
    main()
