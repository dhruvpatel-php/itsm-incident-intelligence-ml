import os
import sys
import logging
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "dataset_list.csv"
if not RAW_DATA_PATH.exists():
    RAW_DATA_PATH = PROJECT_ROOT / "dataset_list.csv"

PROCESSED_DATA_DIR = DATA_DIR / "processed"
CLEANED_DATA_PATH = PROCESSED_DATA_DIR / "cleaned_itsm.csv"
DAILY_VOLUME_PATH = PROCESSED_DATA_DIR / "daily_incident_volume.csv"
QUARTERLY_VOLUME_PATH = PROCESSED_DATA_DIR / "quarterly_incident_volume.csv"

FIGURES_DIR = PROJECT_ROOT / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

for p in [PROCESSED_DATA_DIR, FIGURES_DIR, MODELS_DIR, REPORTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Matplotlib cache setup
os.environ["MPLCONFIGDIR"] = os.environ.get("TEMP", str(PROJECT_ROOT / ".matplotlib_cache"))

# Central Deterministic Seed
RANDOM_STATE = 42

# Stable Regime Date Cutoff (Post-Rollout Enterprise Cutover)
STABLE_REGIME_START = "2013-10-01"

# Official ITIL Priority Matrix (Impact, Urgency) -> Priority
PRIORITY_MATRIX = {
    (1, 1): 1, (1, 2): 2, (1, 3): 3, (1, 4): 3, (1, 5): 3,
    (2, 1): 2, (2, 2): 2, (2, 3): 2, (2, 4): 3, (2, 5): 4,
    (3, 1): 2, (3, 2): 2, (3, 3): 3, (3, 4): 3, (3, 5): 4,
    (4, 1): 3, (4, 2): 3, (4, 3): 3, (4, 4): 4, (4, 5): 4,
    (5, 1): 3, (5, 2): 3, (5, 3): 4, (5, 4): 4, (5, 5): 5,
}

# Standardized Logging Setup
def setup_logger(name="ITSM_Pipeline"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
