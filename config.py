"""Shared paths/constants. Edit DATA_DIR if the dataset moves."""
from pathlib import Path

# Root of the unzipped Kaggle dataset
DATA_DIR = Path(r"C:/Users/Jeff/Downloads/Data Mining Assignment 3 Kaggle/nycu-data-mining-assignment-3")

TRAIN_DIR = DATA_DIR / "train" / "train"
TEST_DIR  = DATA_DIR / "test"  / "test"
SAMPLE_SUB = DATA_DIR / "sample_submission.csv"

# Output folder for figures / submissions / cached features
OUT_DIR = Path(r"C:/Users/Jeff/Downloads/Data Mining Assignment 3 Kaggle/code/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
(FIG_DIR := OUT_DIR / "figures").mkdir(exist_ok=True)
(SUB_DIR := OUT_DIR / "submissions").mkdir(exist_ok=True)
(CACHE_DIR := OUT_DIR / "cache").mkdir(exist_ok=True)

SEED = 42
N_CLASSES = 6
SEQ_LEN = 300
FEATURE_COLS = ["mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z"]
