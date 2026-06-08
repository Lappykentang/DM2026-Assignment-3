"""Load train/test CSVs into memory once, cache as parquet for speed."""
from __future__ import annotations
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from config import TRAIN_DIR, TEST_DIR, CACHE_DIR, FEATURE_COLS, SEQ_LEN


def _read_user_dir(user_dir: Path, has_label: bool) -> list[pd.DataFrame]:
    frames = []
    for csv_path in user_dir.glob("*.csv"):
        df = pd.read_csv(csv_path)
        # Some files might not match SEQ_LEN; we keep whatever they have but warn.
        df["user_id"] = user_dir.name
        if not has_label and "label" not in df.columns:
            df["label"] = -1
        frames.append(df)
    return frames


def load_train(use_cache: bool = True) -> pd.DataFrame:
    cache = CACHE_DIR / "train_long.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)
    all_frames = []
    user_dirs = sorted([d for d in TRAIN_DIR.iterdir() if d.is_dir()])
    for ud in tqdm(user_dirs, desc="train users"):
        all_frames.extend(_read_user_dir(ud, has_label=True))
    df = pd.concat(all_frames, ignore_index=True)
    df.to_parquet(cache, index=False)
    return df


def load_test(use_cache: bool = True) -> pd.DataFrame:
    cache = CACHE_DIR / "test_long.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)
    all_frames = []
    user_dirs = sorted([d for d in TEST_DIR.iterdir() if d.is_dir()])
    for ud in tqdm(user_dirs, desc="test users"):
        all_frames.extend(_read_user_dir(ud, has_label=False))
    df = pd.concat(all_frames, ignore_index=True)
    df.to_parquet(cache, index=False)
    return df


def file_meta(df: pd.DataFrame) -> pd.DataFrame:
    """One row per file_id with label + user_id."""
    cols = ["file_id", "user_id"]
    if "label" in df.columns:
        cols.append("label")
    return df[cols].drop_duplicates("file_id").reset_index(drop=True)


if __name__ == "__main__":
    tr = load_train()
    te = load_test()
    print("Train long rows:", len(tr), "files:", tr["file_id"].nunique(), "users:", tr["user_id"].nunique())
    print("Test  long rows:", len(te), "files:", te["file_id"].nunique(), "users:", te["user_id"].nunique())
    print("Label dist (per file):")
    print(file_meta(tr)["label"].value_counts().sort_index())
