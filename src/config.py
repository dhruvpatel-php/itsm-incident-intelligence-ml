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

# Pillar 4 Asset Risk Shrinkage Parameters
ARI_MIN_INCIDENTS = 10
ARI_SHRINKAGE_M = 10.0

# Official ITIL Priority Matrix (Impact, Urgency) -> Priority
PRIORITY_MATRIX = {
    (1, 1): 1, (1, 2): 2, (1, 3): 3, (1, 4): 3, (1, 5): 3,
    (2, 1): 2, (2, 2): 2, (2, 3): 2, (2, 4): 3, (2, 5): 4,
    (3, 1): 2, (3, 2): 2, (3, 3): 3, (3, 4): 3, (3, 5): 4,
    (4, 1): 3, (4, 2): 3, (4, 3): 3, (4, 4): 4, (4, 5): 4,
    (5, 1): 3, (5, 2): 3, (5, 3): 4, (5, 4): 4, (5, 5): 5,
}

class ITILPriorityEngine:
    def __init__(self, imputer_model=None, imp_cols=None, ci_freq=None):
        self.imputer = imputer_model
        self.imp_cols = imp_cols
        self.ci_freq = ci_freq
        self.matrix = PRIORITY_MATRIX

    def resolve(self, row):
        imp = row.get('Impact_Num')
        urg = row.get('Urgency_Num')
        if imp is not None and urg is not None and not (imp != imp or urg != urg):
            return self.matrix.get((int(imp), int(urg)), 3)
        return 3

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
