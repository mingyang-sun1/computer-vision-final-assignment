"""Central configuration for the COMP90086 house-price project.

Everything that affects a run lives here: paths, the random seed, and the
split settings. Paths are resolved relative to the repository root so the code
runs on any machine without editing a personal absolute path.

To point the code at the course dataset on your machine, either
  * set the CV_DATA_ROOT environment variable, or
  * place the dataset at <repo>/../comp-90086-2026/house_dataset (default).
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# --- Dataset location -------------------------------------------------------
# Default matches the layout used during development:
#   <parent>/comp-90086-2026/house_dataset/{train.csv,test.csv,train/,test/}
DATA_ROOT = Path(
    os.environ.get(
        "CV_DATA_ROOT",
        REPO_ROOT.parent / "comp-90086-2026" / "house_dataset",
    )
).resolve()

TRAIN_CSV = DATA_ROOT / "train.csv"
TEST_CSV = DATA_ROOT / "test.csv"
TRAIN_IMG_DIR = DATA_ROOT / "train"
TEST_IMG_DIR = DATA_ROOT / "test"

# --- Output locations -------------------------------------------------------
SPLIT_DIR = REPO_ROOT / "data" / "splits"
OUTPUT_DIR = REPO_ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
LOG_DIR = OUTPUT_DIR / "logs"
PRED_DIR = OUTPUT_DIR / "predictions"

# --- Reproducibility --------------------------------------------------------
# Every experiment in this project uses this seed and this split.
SEED = 42
VAL_FRACTION = 0.125  # 1000 of the 8000 training images
N_PRICE_BINS = 10  # stratification granularity (price deciles)

# --- Target ------------------------------------------------------------------
# Prices are stored in units of $1000 USD, matching the Kaggle metric (MSE).
PRICE_COL = "price"
IMAGE_COL = "imageid"

# --- Phase 2 baseline --------------------------------------------------------
# Frozen ImageNet-pretrained backbone + a small regression head. Features are
# extracted once and cached, because extraction is the only expensive step
# without a GPU; everything downstream then trains in seconds.
BACKBONE = "resnet50"
IMG_SIZE = 224
BATCH_SIZE = 32
FEATURE_DIR = OUTPUT_DIR / "features"


def describe() -> str:
    """Human-readable summary, useful in logs and the reproducibility check."""
    return (
        f"REPO_ROOT   = {REPO_ROOT}\n"
        f"DATA_ROOT   = {DATA_ROOT}\n"
        f"SEED        = {SEED}\n"
        f"VAL_FRACTION= {VAL_FRACTION}\n"
        f"N_PRICE_BINS= {N_PRICE_BINS}\n"
    )


if __name__ == "__main__":
    print(describe())
