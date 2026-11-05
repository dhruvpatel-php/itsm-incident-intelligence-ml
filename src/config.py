import os
import sys
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "dataset_list.csv"
if not RAW_DATA_PATH.exists():
    # Fallback to project root if not in data/raw
    RAW_DATA_PATH = PROJECT_ROOT / "dataset_list.csv"

PROCESSED_DATA_DIR = DATA_DIR / "processed"
CLEANED_DATA_PATH = PROCESSED_DATA_DIR / "cleaned_itsm.csv"
DAILY_VOLUME_PATH = PROCESSED_DATA_DIR / "daily_incident_volume.csv"
QUARTERLY_VOLUME_PATH = PROCESSED_DATA_DIR / "quarterly_incident_volume.csv"

FIGURES_DIR = PROJECT_ROOT / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

for p in [PROCESSED_DATA_DIR, FIGURES_DIR, MODELS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Set Matplotlib cache dir to prevent warnings
os.environ["MPLCONFIGDIR"] = os.environ.get("TEMP", str(PROJECT_ROOT / ".cache"))

RANDOM_STATE = 42

# Official ITIL Priority Matrix (Urgency vs Impact)
PRIORITY_MATRIX = {
    (1, 1): 1, (1, 2): 2, (1, 3): 3, (1, 4): 3, (1, 5): 3,
    (2, 1): 2, (2, 2): 2, (2, 3): 2, (2, 4): 3, (2, 5): 4,
    (3, 1): 2, (3, 2): 2, (3, 3): 3, (3, 4): 3, (3, 5): 4,
    (4, 1): 3, (4, 2): 3, (4, 3): 3, (4, 4): 4, (4, 5): 4,
    (5, 1): 3, (5, 2): 3, (5, 3): 4, (5, 4): 4, (5, 5): 5,
}

# Column Definitions
DROP_COLUMNS = ["Alert_Status", "Status"] # Alert_Status is single-value ('closed'); Status is 99.98% 'Closed'
