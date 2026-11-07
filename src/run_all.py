import sys
import os
import time
import json
import hashlib
import platform
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CLEANED_DATA_PATH,
    REPORTS_DIR,
    setup_logger
)

logger = setup_logger("RunAll")

def get_file_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def get_git_commit():
    try:
        res = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown_or_uncommitted"

def run_pipeline(pillar_filter=None):
    start_time = time.time()
    logger.info("=" * 80)
    logger.info("ABC TECH - ITSM MACHINE LEARNING END-TO-END PIPELINE")
    logger.info("Rigorous Temporal Validation, Zero Data Leakage, ITIL Alignment")
    logger.info("=" * 80)

    # Lazy imports with informative dependency guard
    try:
        import sklearn
        import lightgbm
        import statsmodels
        import pandas as pd
        import numpy as np
    except ImportError as e:
        logger.error(f"Critical dependency missing: {e.name}. Please install dependencies via: pip install -r requirements.txt")
        sys.exit(1)

    from src.data_loader import load_and_clean_data
    from src.eda import run_eda
    from src.model_pillar1_priority import train_pillar1
    from src.model_pillar2_forecasting import build_forecasting
    from src.model_pillar3_routing import build_routing
    from src.model_pillar4_rfc_assets import build_rfc_and_assets

    metrics_payload = {
        "metadata": {
            "pipeline_name": "ABC Tech ITSM Machine Learning Suite",
            "execution_timestamp": datetime.now().isoformat(),
            "git_commit": get_git_commit(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "library_versions": {
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "scikit_learn": sklearn.__version__,
                "lightgbm": lightgbm.__version__,
                "statsmodels": statsmodels.__version__
            }
        }
    }

    # Step 1: Data Ingestion & Cleaning
    logger.info("\n>>> STEP 1: DATA INGESTION, AUDITING & CLEANING...")
    df, dl_metrics = load_and_clean_data(return_metrics=True)
    metrics_payload["data_loader"] = dl_metrics

    # Compute raw dataset hash
    raw_data_file = PROJECT_ROOT / "dataset_list.csv"
    if not raw_data_file.exists():
        raw_data_file = PROJECT_ROOT / "data" / "raw" / "dataset_list.csv"
    if raw_data_file.exists():
        metrics_payload["metadata"]["raw_data_sha256"] = get_file_sha256(raw_data_file)

    # Step 2: Exploratory Data Analysis
    if pillar_filter is None:
        logger.info("\n>>> STEP 2: STATISTICAL EXPLORATORY DATA ANALYSIS (EDA)...")
        run_eda()

    # Step 3: Pillar 1 Early Warning
    if pillar_filter is None or pillar_filter == 1:
        logger.info("\n>>> STEP 3: PILLAR 1 - HIGH PRIORITY INCIDENT EARLY WARNING...")
        p1_metrics = train_pillar1(df)
        metrics_payload["pillar1_early_warning"] = p1_metrics

    # Step 4: Pillar 2 Forecasting
    base_vol_2014 = 72714
    if pillar_filter is None or pillar_filter == 2:
        logger.info("\n>>> STEP 4: PILLAR 2 - INCIDENT VOLUME FORECASTING (QUARTERLY & ANNUAL)...")
        p2_metrics = build_forecasting(df)
        metrics_payload["pillar2_forecasting"] = p2_metrics
        base_vol_2014 = p2_metrics.get("base_vol_2014", 72714)

    # Step 5: Pillar 3 Priority & Routing
    if pillar_filter is None or pillar_filter == 3:
        logger.info("\n>>> STEP 5: PILLAR 3 - ITIL PRIORITY RESOLUTION & ROUTING...")
        p3_res = build_routing(df, base_vol_2014=base_vol_2014)
        metrics_payload["pillar3_priority"] = p3_res["pillar3_priority"]
        metrics_payload["pillar3_routing"] = p3_res["pillar3_routing"]

    # Step 6: Pillar 4 RFC & Asset Risk
    if pillar_filter is None or pillar_filter == 4:
        logger.info("\n>>> STEP 6: PILLAR 4 - RFC PREDICTION & ITSM ASSET RISK PROFILING...")
        p4_metrics = build_rfc_and_assets(df)
        metrics_payload["pillar4_rfc_and_assets"] = p4_metrics

    # If running full pipeline, write single source of truth: reports/metrics.json
    if pillar_filter is None:
        metrics_file = REPORTS_DIR / "metrics.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics_payload, f, indent=2)
        logger.info(f"\n[Provenance] Single source of truth metrics written to: {metrics_file}")

    elapsed = time.time() - start_time
    logger.info("=" * 80)
    logger.info(f"PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS")
    logger.info("=" * 80)
    return metrics_payload

def main():
    parser = argparse.ArgumentParser(description="Run ABC Tech ITSM ML Pipeline")
    parser.add_argument(
        "--pillar",
        type=int,
        choices=[1, 2, 3, 4],
        default=None,
        help="Run only a specific pillar (1: Priority Early Warning, 2: Volume Forecasting, 3: Routing & ITIL, 4: RFC & Asset Risk)"
    )
    args = parser.parse_args()
    run_pipeline(pillar_filter=args.pillar)

if __name__ == "__main__":
    main()
