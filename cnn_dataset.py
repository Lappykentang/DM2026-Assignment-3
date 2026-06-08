"""PyTorch Dataset for the HAR competition.

Each sample = one file -> (6, 300) tensor (channels-first for Conv1d).
Augmentations applied at __getitem__ time when augment=True:
  - jitter   : add small Gaussian noise
  - time-shift : circular roll along time axis
  - magnitude warp : scale by ~U(0.95, 1.05)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import FEATURE_COLS, SEQ_LEN


def build_data_dict(long_df: pd.DataFrame) -> dict[int, np.ndarray]:
    """Convert long-form DF to {file_id: (SEQ_LEN, 6) float32 array}.
    Pads with zeros if shorter than SEQ_LEN, truncates if longer.
    """
    out: dict[int, np.ndarray] = {}
    for fid, sub in long_df.groupby("file_id", sort=False):
        sub = sub.sort_values("index")
        arr = sub[FEATURE_COLS].to_numpy(dtype=np.float32)
        if arr.shape[0] < SEQ_LEN:
            pad = np.zeros((SEQ_LEN - arr.shape[0], len(FEATURE_COLS)), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=0)
        elif arr.shape[0] > SEQ_LEN:
            arr = arr[:SEQ_LEN]
        out[fid] = arr
    return out


def compute_normalizer(data_dict: dict[int, np.ndarray], file_ids) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean/std over the listed file_ids only."""
    stack = np.stack([data_dict[f] for f in file_ids], axis=0)  # (N, 300, 6)
    mean = stack.reshape(-1, 6).mean(axis=0).astype(np.float32)
    std  = stack.reshape(-1, 6).std(axis=0).astype(np.float32) + 1e-6
    return mean, std


class HARDataset(Dataset):
    def __init__(self,
                 data_dict: dict[int, np.ndarray],
                 file_ids,
                 labels_dict: dict[int, int] | None = None,
                 normalizer: tuple[np.ndarray, np.ndarray] | None = None,
                 augment: bool = False,
                 jitter_sigma: float = 0.02,
                 shift_range: int = 15,
                 scale_range: tuple[float, float] = (0.95, 1.05),
                 p_jitter: float = 0.5,
                 p_shift: float = 0.5,
                 p_scale: float = 0.3,
                 rng_seed: int | None = None):
        self.data = data_dict
        self.file_ids = np.asarray(list(file_ids))
        self.labels = labels_dict
        self.normalizer = normalizer
        self.augment = augment
        self.jitter_sigma = jitter_sigma
        self.shift_range = shift_range
        self.scale_range = scale_range
        self.p_jitter = p_jitter
        self.p_shift = p_shift
        self.p_scale = p_scale
        # Each worker can have its own RNG; seed default is fine.
        self.rng = np.random.default_rng(rng_seed)

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        fid = int(self.file_ids[idx])
        x = self.data[fid].copy()  # (300, 6)

        if self.normalizer is not None:
            mean, std = self.normalizer
            x = (x - mean) / std

        if self.augment:
            if self.rng.random() < self.p_jitter:
                x = x + self.rng.normal(0.0, self.jitter_sigma, size=x.shape).astype(np.float32)
            if self.rng.random() < self.p_shift:
                k = int(self.rng.integers(-self.shift_range, self.shift_range + 1))
                if k != 0:
                    x = np.roll(x, k, axis=0)
            if self.rng.random() < self.p_scale:
                s = float(self.rng.uniform(*self.scale_range))
                x = x * s

        # channels-first for Conv1d: (6, 300)
        x = torch.from_numpy(np.ascontiguousarray(x.T))
        if self.labels is not None:
            return x, torch.tensor(self.labels[fid], dtype=torch.long)
        return x, torch.tensor(fid, dtype=torch.long)
