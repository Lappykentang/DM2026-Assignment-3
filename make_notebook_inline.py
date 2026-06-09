"""Builds HAR_pipeline_inline.ipynb — every line of code inlined.

No imports from local .py files; the notebook is self-contained except for
external packages (numpy, pandas, sklearn, lightgbm, torch, fpdf2).

Run: python make_notebook_inline.py
"""
import json
import uuid
from pathlib import Path

cells = []


def md(text: str):
    cells.append({
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": text.splitlines(keepends=True),
    })


def code(text: str):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    })


# ============================================================================
# NOTEBOOK CONTENT
# ============================================================================

md(r"""# NYCU Data Mining Assignment 3 — HAR (Inline Notebook)

Fully self-contained notebook. Every line of code is in this file.

**How to use:**
1. Open in VS Code, select the `.venv` Python kernel (top-right).
2. Run cells top → bottom (Shift+Enter).
3. Variables persist between cells. If you restart the kernel, re-run from the top.
4. Cells flagged ⏰ are long-running (LGBM ablation, CNN training).

**Before starting**, make sure these packages are installed:
```
pip install numpy pandas scikit-learn matplotlib seaborn tqdm pyarrow scipy lightgbm fpdf2
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Dataset path:** edit the `DATA_DIR` variable in the first code cell to point
to your unzipped Kaggle data folder.
""")

# ============================== INSTALL ==============================
md("## 0a. Install dependencies (run this cell once, then restart the kernel)")

code(r"""# Install all required packages including PyTorch (CPU build).
# Run this cell once. If it finishes without errors, restart the kernel,
# then run all cells top-to-bottom.
import subprocess, sys

def _pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet"] + list(args))

# Core data science packages (I had to install xgboost after I added it
# as a 3rd base model. python-docx is for the Word report at the end.)
_pip("numpy", "pandas", "scikit-learn", "matplotlib", "seaborn",
     "tqdm", "pyarrow", "scipy", "lightgbm", "xgboost", "catboost",
     "python-docx", "tabulate", "fpdf2")

# PyTorch — CPU build (works on any machine, no CUDA needed)
# If you have an NVIDIA GPU, replace the index URL with:
#   https://download.pytorch.org/whl/cu121
_pip("torch", "--index-url", "https://download.pytorch.org/whl/cpu")

import torch
print("All packages installed successfully!")
print(f"  torch version : {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
print()
print("Next step: restart the kernel (Kernel > Restart), then run all cells top-to-bottom.")
""")

# ============================== SETUP ==============================
md("## 0b. Setup — paths, seeds, constants")

code(r"""# Imports + paths + constants
import os, sys, random, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === EDIT THIS PATH if your data is elsewhere ===
DATA_DIR = Path(r"C:/Users/Jeff/Downloads/Data Mining Assignment 3 Kaggle/nycu-data-mining-assignment-3")

TRAIN_DIR  = DATA_DIR / "train" / "train"
TEST_DIR   = DATA_DIR / "test"  / "test"
SAMPLE_SUB = DATA_DIR / "sample_submission.csv"

# Outputs (auto-created)
OUT_DIR   = Path.cwd() / "outputs"
FIG_DIR   = OUT_DIR / "figures"
SUB_DIR   = OUT_DIR / "submissions"
CACHE_DIR = OUT_DIR / "cache"
REPORT_DIR = OUT_DIR / "report"
for d in (OUT_DIR, FIG_DIR, SUB_DIR, CACHE_DIR, REPORT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Constants
SEED = 42
N_CLASSES = 6
SEQ_LEN = 300
FEATURE_COLS = ["mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z"]
MEAN_COLS = ["mean_x", "mean_y", "mean_z"]
STD_COLS  = ["std_x", "std_y", "std_z"]

random.seed(SEED); np.random.seed(SEED)

print("DATA_DIR     :", DATA_DIR)
print("Train exists :", TRAIN_DIR.exists())
print("Test exists  :", TEST_DIR.exists())
print("Sample sub   :", SAMPLE_SUB.exists())
plt.rcParams["figure.dpi"] = 110
""")

# ============================== DATA LOADER ==============================
md("## 1. Load + cache the dataset")

code(r"""# Reads all train/test CSVs into memory and caches as parquet.
from tqdm import tqdm

def _read_user_dir(user_dir: Path, has_label: bool):
    frames = []
    for csv_path in user_dir.glob("*.csv"):
        df = pd.read_csv(csv_path)
        df["user_id"] = user_dir.name
        if not has_label and "label" not in df.columns:
            df["label"] = -1
        frames.append(df)
    return frames

def load_train(use_cache=True):
    cache = CACHE_DIR / "train_long.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)
    all_frames = []
    for ud in tqdm(sorted([d for d in TRAIN_DIR.iterdir() if d.is_dir()]), desc="train users"):
        all_frames.extend(_read_user_dir(ud, has_label=True))
    df = pd.concat(all_frames, ignore_index=True)
    df.to_parquet(cache, index=False)
    return df

def load_test(use_cache=True):
    cache = CACHE_DIR / "test_long.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)
    all_frames = []
    for ud in tqdm(sorted([d for d in TEST_DIR.iterdir() if d.is_dir()]), desc="test users"):
        all_frames.extend(_read_user_dir(ud, has_label=False))
    df = pd.concat(all_frames, ignore_index=True)
    df.to_parquet(cache, index=False)
    return df

def file_meta(df):
    cols = ["file_id", "user_id"]
    if "label" in df.columns:
        cols.append("label")
    return df[cols].drop_duplicates("file_id").reset_index(drop=True)

tr_long = load_train()
te_long = load_test()
tr_meta = file_meta(tr_long)
te_meta = file_meta(te_long)

print(f"Train: {tr_long['file_id'].nunique():,} files, {tr_long['user_id'].nunique()} users")
print(f"Test : {te_long['file_id'].nunique():,} files, {te_long['user_id'].nunique()} users")
print("\nLabel distribution (per file):")
print(tr_meta['label'].value_counts().sort_index())
""")

# ============================== EDA ==============================
md("## 2. EDA — quick figures")

code(r"""# Label distribution + signal example figures
def fig_label_distribution(train_long):
    meta = file_meta(train_long)
    counts = meta["label"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index.astype(str), counts.values, color="steelblue")
    ax.set_title(f"Label distribution (files) - total = {len(meta)}")
    ax.set_xlabel("label"); ax.set_ylabel("# files")
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG_DIR / "01_label_distribution.png"); plt.show()
    return counts

def fig_signal_examples(train_long):
    meta = file_meta(train_long)
    fig, axes = plt.subplots(N_CLASSES, 1, figsize=(10, 2.0 * N_CLASSES), sharex=True)
    for label in range(N_CLASSES):
        fid = meta[meta['label'] == label]['file_id'].iloc[0]
        sub = train_long[train_long['file_id'] == fid].sort_values("index")
        ax = axes[label]
        ax.plot(sub['index'], sub['mean_x'], label='mean_x', lw=0.8)
        ax.plot(sub['index'], sub['mean_y'], label='mean_y', lw=0.8)
        ax.plot(sub['index'], sub['mean_z'], label='mean_z', lw=0.8)
        ax.set_ylabel(f"label {label}")
        ax.legend(loc="upper right", fontsize=7)
    axes[-1].set_xlabel("second within window")
    fig.suptitle("Example signal per class (mean_xyz)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "04_signal_examples_mean.png"); plt.show()

counts = fig_label_distribution(tr_long)
fig_signal_examples(tr_long)

# Quick sanity check
tr_n = tr_long.groupby('file_id').size()
te_n = te_long.groupby('file_id').size()
print(f"\nrows-per-file (train): min={tr_n.min()}  max={tr_n.max()}  median={int(tr_n.median())}")
print(f"rows-per-file (test) : min={te_n.min()}  max={te_n.max()}  median={int(te_n.median())}")
print(f"NaNs in train features: {tr_long[FEATURE_COLS].isna().sum().sum()}")
print(f"NaNs in test  features: {te_long[FEATURE_COLS].isna().sum().sum()}")
overlap = set(tr_long['user_id'].unique()) & set(te_long['user_id'].unique())
print(f"User overlap train/test: {len(overlap)}")
""")

md("""**Things I noticed during EDA (took me a while to convince myself):**

- Every file is exactly 300 rows. So I don't need to handle variable-length sequences. Good.
- Each file has 6 columns: `mean_xyz` (the 1-second average accel) and `std_xyz` (within-second variance).
- Train has 60 users, test has 40, and they're **disjoint**. That's why I can't just do a random 80/20 split — I'd be testing on users the model already saw and inflating my CV score. I use `GroupKFold(user_id)` everywhere.
- Class imbalance is real: classes 2 and 4 have very few examples (~140-360 files). That's why class-balanced sample weights + label smoothing matter.
- I tried a quick "what if I just check which axis has the highest variance per file" idea early on — it kinda separated walking/running from sitting/lying but was useless for distinguishing similar classes.""")

md("""**Per-user file count + per-(user, label) heatmap** — gives an idea of how lopsided the data is across users.""")

code(r"""# Files per user (sorted descending). Some users dominate the dataset.
per_user = file_meta(tr_long).groupby('user_id').size().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.bar(range(len(per_user)), per_user.values, color="darkorange")
ax.set_title("Files per user (train)")
ax.set_xlabel("user (sorted)"); ax.set_ylabel("# files")
fig.tight_layout(); fig.savefig(FIG_DIR / "02_files_per_user.png"); plt.show()
print("Files-per-user describe:\n", per_user.describe())

# User x Label heatmap
pivot = file_meta(tr_long).pivot_table(index='user_id', columns='label',
                                        values='file_id', aggfunc='count', fill_value=0)
fig, ax = plt.subplots(figsize=(7, 10))
im = ax.imshow(pivot.values, aspect='auto', cmap='viridis')
ax.set_xticks(range(pivot.shape[1])); ax.set_xticklabels(pivot.columns)
ax.set_yticks(range(pivot.shape[0])); ax.set_yticklabels(pivot.index, fontsize=6)
ax.set_title("Files per (user, label)")
fig.colorbar(im, ax=ax)
fig.tight_layout(); fig.savefig(FIG_DIR / "03_user_label_heatmap.png"); plt.show()
""")

# ============================== NAIVE BASELINE ==============================
md("## 3. Naive baseline (Prompt 1) — 12 features + LogReg/RF")

code(r"""# Aggregate each file to 12 simple stats (mean & std over time of 6 columns)
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def aggregate_naive(long_df):
    g = long_df.groupby("file_id")
    agg_mean = g[FEATURE_COLS].mean().add_suffix("__tmean")
    agg_std  = g[FEATURE_COLS].std().add_suffix("__tstd").fillna(0.0)
    return pd.concat([agg_mean, agg_std], axis=1).reset_index()

def build_xy(long_df, with_label):
    feats = aggregate_naive(long_df)
    meta = file_meta(long_df)
    df = feats.merge(meta, on="file_id")
    feat_cols = [c for c in df.columns if c.endswith(("__tmean", "__tstd"))]
    X = df[feat_cols].values.astype(np.float32)
    groups = df["user_id"].values
    y = df["label"].values if with_label else None
    return X, y, groups, df["file_id"].values, feat_cols

def cv_score_naive(model, X, y, groups, n_splits=5, model_name=""):
    gkf = GroupKFold(n_splits=n_splits)
    f1s, oof = [], np.full(len(y), -1, dtype=int)
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        m = clone(model)
        m.fit(X[tr_idx], y[tr_idx])
        pred = m.predict(X[va_idx])
        oof[va_idx] = pred
        f1 = f1_score(y[va_idx], pred, average="macro")
        f1s.append(f1)
        print(f"  [{model_name}] fold {fold+1}: macro-F1 = {f1:.4f}")
    mean_f1 = float(np.mean(f1s))
    print(f"  [{model_name}] CV macro-F1 = {mean_f1:.4f} ± {np.std(f1s):.4f}")
    return mean_f1, oof

X_tr_naive, y_tr, groups_tr, ids_tr, _ = build_xy(tr_long, with_label=True)
X_te_naive, _,    _,         ids_te, _ = build_xy(te_long, with_label=False)
print(f"X_train_naive: {X_tr_naive.shape}  X_test_naive: {X_te_naive.shape}")
""")

code(r"""# Train naive baseline with 5-fold GroupKFold by user
logreg = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=2000, C=1.0, random_state=SEED, n_jobs=-1)),
])
rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                            n_jobs=-1, random_state=SEED)

print("--- Logistic Regression ---")
f1_lr, oof_lr = cv_score_naive(logreg, X_tr_naive, y_tr, groups_tr, model_name="logreg")
print("\n--- Random Forest ---")
f1_rf, oof_rf = cv_score_naive(rf, X_tr_naive, y_tr, groups_tr, model_name="rf")

best_name = "logreg" if f1_lr >= f1_rf else "rf"
print(f"\nBest naive model: {best_name}  (CV macro-F1 = {max(f1_lr, f1_rf):.4f})")
""")

code(r"""# Refit best naive model on all train, predict test, write submission_naive.csv
final = clone(logreg if f1_lr >= f1_rf else rf)
final.fit(X_tr_naive, y_tr)
preds_naive = final.predict(X_te_naive)

sub_template = pd.read_csv(SAMPLE_SUB)
pred_df = pd.DataFrame({"Id": ids_te.astype(sub_template['Id'].dtype), "Label": preds_naive})
sub = sub_template[["Id"]].merge(pred_df, on="Id", how="left")
sub["Label"] = sub["Label"].fillna(pd.Series(y_tr).mode().iloc[0]).astype(int)
sub.to_csv(SUB_DIR / "submission_naive.csv", index=False)
print(f"Saved {SUB_DIR / 'submission_naive.csv'}")

np.savez(CACHE_DIR / "naive_oof.npz", oof_lr=oof_lr, oof_rf=oof_rf,
         y=y_tr, ids=ids_tr, groups=groups_tr)
print("OOF cached.")
""")

# ============================== FEATURE ENGINEERING ==============================
md("## 4. Feature engineering — ~334 features per file (organized in groups)")

code(r"""# Per-file feature builder organized into named groups for ablation.
from scipy.stats import skew, kurtosis

def _basic(arr, prefix):
    return {
        f"{prefix}_mean":   float(np.mean(arr)),
        f"{prefix}_std":    float(np.std(arr)),
        f"{prefix}_min":    float(np.min(arr)),
        f"{prefix}_max":    float(np.max(arr)),
        f"{prefix}_median": float(np.median(arr)),
    }

def _pct(arr, prefix):
    p10, p25, p75, p90 = np.percentile(arr, [10, 25, 75, 90])
    return {f"{prefix}_p10": float(p10), f"{prefix}_p25": float(p25),
            f"{prefix}_p75": float(p75), f"{prefix}_p90": float(p90),
            f"{prefix}_iqr": float(p75 - p25)}

def _moments(arr, prefix):
    s = skew(arr, bias=False, nan_policy="omit")
    k = kurtosis(arr, bias=False, nan_policy="omit")
    s = 0.0 if not np.isfinite(s) else float(s)
    k = 0.0 if not np.isfinite(k) else float(k)
    return {f"{prefix}_skew": s, f"{prefix}_kurt": k}

def _fft_features(arr, prefix, n_bands=5):
    arr = arr - np.mean(arr)
    spec = np.abs(np.fft.rfft(arr))[1:]
    if spec.size == 0 or spec.sum() == 0:
        d = {f"{prefix}_fft_b{i}": 0.0 for i in range(n_bands)}
        d[f"{prefix}_fft_entropy"] = 0.0; d[f"{prefix}_fft_dom"] = 0.0
        return d
    bands = np.array_split(spec, n_bands)
    total = spec.sum()
    out = {f"{prefix}_fft_b{i}": float(b.sum() / total) for i, b in enumerate(bands)}
    p = spec / total
    out[f"{prefix}_fft_entropy"] = float(-np.sum(p * np.log(p + 1e-12)))
    out[f"{prefix}_fft_dom"]     = float(np.argmax(spec))
    return out

def _mean_crossings(arr):
    a = arr - np.mean(arr)
    return int(np.sum(np.diff(np.sign(a)) != 0))

def _autocorr_feats(arr, prefix, lags=(1, 2, 5, 10, 20)):
    # Autocorrelation at multiple lags - key for detecting step periodicity
    n = len(arr)
    a = arr - arr.mean()
    std = a.std()
    denom = (std ** 2 * n) if std > 1e-9 else 1.0
    out = {}
    for lag in lags:
        if lag >= n:
            out[f"{prefix}_ac{lag}"] = 0.0
        else:
            c = float(np.dot(a[lag:], a[:-lag]) / denom)
            out[f"{prefix}_ac{lag}"] = float(np.clip(c, -1.0, 1.0))
    return out

def _segment_feats(arr, prefix, n_segs=3):
    # Mean, std, range within each temporal third - captures activity evolution
    segs = np.array_split(arr, n_segs)
    out = {}
    for i, seg in enumerate(segs):
        out[f"{prefix}_seg{i}_mean"] = float(np.mean(seg))
        out[f"{prefix}_seg{i}_std"]  = float(np.std(seg))
        out[f"{prefix}_seg{i}_rng"]  = float(np.ptp(seg))
    return out

def _peak_feats(arr, prefix):
    # Count of local maxima above threshold mean+0.5*std.
    # I added this hoping it'd separate walk/run (rhythmic peaks) from
    # sit/lie (none). Doesn't always work but helps a bit.
    from scipy.signal import find_peaks
    thr = float(arr.mean() + 0.5 * arr.std())
    try:
        peaks, props = find_peaks(arr, height=thr, distance=5)
        n = len(peaks)
        h = float(np.mean(props["peak_heights"])) if n > 0 else 0.0
        d = float(np.mean(np.diff(peaks))) if n >= 2 else 0.0
    except Exception:
        n, h, d = 0, 0.0, 0.0
    return {f"{prefix}_npeaks": float(n),
            f"{prefix}_peakh_mean": h,
            f"{prefix}_peak_period": d}

def _energy_feats(arr, prefix):
    # Energy + histogram entropy. Active vs passive separator.
    energy = float(np.sum(arr ** 2) / max(len(arr), 1))
    hist, _ = np.histogram(arr, bins=10, density=False)
    p = hist.astype(np.float64)
    p = p / (p.sum() + 1e-12)
    ent = float(-np.sum(p * np.log(p + 1e-12)))
    rms = float(np.sqrt(np.mean(arr ** 2)))
    return {f"{prefix}_energy": energy,
            f"{prefix}_hist_ent": ent,
            f"{prefix}_rms": rms}

def _ratio_feats(mx, my, mz):
    # Cross-axis ratios - dominant-axis clue.
    vx, vy, vz = float(np.var(mx)), float(np.var(my)), float(np.var(mz))
    rx, ry, rz = float(np.ptp(mx)), float(np.ptp(my)), float(np.ptp(mz))
    ex, ey, ez = float(np.sum(mx**2)), float(np.sum(my**2)), float(np.sum(mz**2))
    eps = 1e-9
    return {
        "ratio__var_xy": vx / (vy + eps), "ratio__var_xz": vx / (vz + eps), "ratio__var_yz": vy / (vz + eps),
        "ratio__rng_xy": rx / (ry + eps), "ratio__rng_xz": rx / (rz + eps), "ratio__rng_yz": ry / (rz + eps),
        "ratio__e_xy":   ex / (ey + eps), "ratio__e_xz":   ex / (ez + eps), "ratio__e_yz":   ey / (ez + eps),
        "ratio__dominant_axis": float(np.argmax([vx, vy, vz])),
    }

def _tail_feats(arr, prefix):
    p5, p95, p99 = np.percentile(arr, [5, 95, 99])
    return {f"{prefix}_p5": float(p5),
            f"{prefix}_p95": float(p95),
            f"{prefix}_p99": float(p99),
            f"{prefix}_p95_minus_p5": float(p95 - p5)}

def _file_features(df):
    df = df.sort_values("index")
    feats = {}
    for col in FEATURE_COLS:
        arr = df[col].to_numpy(dtype=np.float64)
        feats.update(_basic(arr, f"basic__{col}"))
        feats.update(_pct(arr,   f"pct__{col}"))
        feats.update(_moments(arr, f"moments__{col}"))
        feats.update(_fft_features(arr, f"fft__{col}"))
        feats[f"crossings__{col}"] = _mean_crossings(arr)

    mx = df["mean_x"].to_numpy(); my = df["mean_y"].to_numpy(); mz = df["mean_z"].to_numpy()
    mag = np.sqrt(mx**2 + my**2 + mz**2)
    feats.update(_basic(mag, "mag__m"))
    p10, p90 = np.percentile(mag, [10, 90])
    feats["mag__m_p10"] = float(p10); feats["mag__m_p90"] = float(p90)

    feats["sma__total"] = float(np.mean(np.abs(mx) + np.abs(my) + np.abs(mz)))
    feats["sma__x"] = float(np.mean(np.abs(mx)))
    feats["sma__y"] = float(np.mean(np.abs(my)))
    feats["sma__z"] = float(np.mean(np.abs(mz)))

    def _corr(a, b):
        if np.std(a) < 1e-9 or np.std(b) < 1e-9: return 0.0
        return float(np.corrcoef(a, b)[0, 1])
    feats["corr__xy"] = _corr(mx, my)
    feats["corr__xz"] = _corr(mx, mz)
    feats["corr__yz"] = _corr(my, mz)

    for col, arr in zip(MEAN_COLS, [mx, my, mz]):
        j = np.diff(arr)
        feats[f"jerk__{col}_meanabs"] = float(np.mean(np.abs(j))) if j.size else 0.0
        feats[f"jerk__{col}_std"]     = float(np.std(j)) if j.size else 0.0
        feats[f"jerk__{col}_maxabs"]  = float(np.max(np.abs(j))) if j.size else 0.0

    # jerk magnitude (combined 3-axis jerk energy)
    jx = np.diff(mx); jy = np.diff(my); jz = np.diff(mz)
    jmag = np.sqrt(jx**2 + jy**2 + jz**2)
    feats["jerk__mag_meanabs"] = float(np.mean(jmag)) if jmag.size else 0.0
    feats["jerk__mag_std"]     = float(np.std(jmag)) if jmag.size else 0.0
    feats["jerk__mag_maxabs"]  = float(np.max(jmag)) if jmag.size else 0.0

    for col in STD_COLS:
        arr = df[col].to_numpy(dtype=np.float64)
        feats[f"std_sum__{col}_mean"] = float(np.mean(arr))
        feats[f"std_sum__{col}_max"]  = float(np.max(arr))
        feats[f"std_sum__{col}_p90"]  = float(np.percentile(arr, 90))

    # --- Autocorrelation features (periodicity / step cadence) ---
    for col, arr_ac in zip(MEAN_COLS, [mx, my, mz]):
        feats.update(_autocorr_feats(arr_ac, f"autocorr__{col}"))
    feats.update(_autocorr_feats(mag, "autocorr__mag"))

    # --- Temporal segment features (how activity evolves over time) ---
    for col in FEATURE_COLS:
        arr_s = df[col].to_numpy(dtype=np.float64)
        feats.update(_segment_feats(arr_s, f"seg__{col}"))

    # --- Peak features (rhythm indicator) ---
    for col, arr_p in zip(MEAN_COLS, [mx, my, mz]):
        feats.update(_peak_feats(arr_p, f"peaks__{col}"))
    feats.update(_peak_feats(mag, "peaks__mag"))

    # --- Energy features (active vs passive separator) ---
    for col in FEATURE_COLS:
        arr_e = df[col].to_numpy(dtype=np.float64)
        feats.update(_energy_feats(arr_e, f"energy__{col}"))
    feats.update(_energy_feats(mag, "energy__mag"))

    # --- Cross-axis ratios ---
    feats.update(_ratio_feats(mx, my, mz))

    # --- Extreme percentile features (tails) ---
    for col in MEAN_COLS:
        arr_t = df[col].to_numpy(dtype=np.float64)
        feats.update(_tail_feats(arr_t, f"tails__{col}"))
    feats.update(_tail_feats(mag, "tails__mag"))

    # --- Std-channel temporal features (v4 - biggest late feature gain) ---
    # std_x/y/z are within-second motion intensity. Running autocorr / peaks /
    # tails over them (not just the mean channels) captures motion-intensity
    # rhythm (step cadence). This lifted every tabular model by ~0.004-0.015 OOF.
    sx = df["std_x"].to_numpy(dtype=np.float64)
    sy = df["std_y"].to_numpy(dtype=np.float64)
    sz = df["std_z"].to_numpy(dtype=np.float64)
    for col, arr_sd in zip(STD_COLS, [sx, sy, sz]):
        feats.update(_autocorr_feats(arr_sd, f"autocorr__{col}"))
        feats.update(_peak_feats(arr_sd,     f"peaks__{col}"))
        feats.update(_tail_feats(arr_sd,     f"tails__{col}"))
    for ax, m_arr, s_arr in zip(["x", "y", "z"], [mx, my, mz], [sx, sy, sz]):
        feats[f"covmix__cov_{ax}"]    = float(s_arr.mean() / (np.abs(m_arr).mean() + 1e-9))
        feats[f"covmix__mscorr_{ax}"] = _corr(m_arr, s_arr)
    feats["covmix__cov_mag"]   = float(mag.std() / (np.abs(mag).mean() + 1e-9))
    feats["covmix__std_cc_xy"] = _corr(sx, sy)
    feats["covmix__std_cc_xz"] = _corr(sx, sz)
    feats["covmix__std_cc_yz"] = _corr(sy, sz)

    return feats

def group_of(col_name):
    return col_name.split("__", 1)[0]

def build_features(long_df, cache_name=None):
    cache_path = (CACHE_DIR / f"{cache_name}.parquet") if cache_name else None
    if cache_path is not None and cache_path.exists():
        feats = pd.read_parquet(cache_path)
    else:
        rows = []
        for file_id, sub in tqdm(long_df.groupby("file_id", sort=False),
                                  desc=cache_name or "build features"):
            row = _file_features(sub)
            row["file_id"] = file_id
            row["user_id"] = sub["user_id"].iloc[0]
            if "label" in sub.columns and sub["label"].iloc[0] != -1:
                row["label"] = int(sub["label"].iloc[0])
            rows.append(row)
        feats = pd.DataFrame(rows)
        if cache_path is not None:
            feats.to_parquet(cache_path, index=False)
    meta_cols = {"file_id", "user_id", "label"}
    feat_cols = [c for c in feats.columns if c not in meta_cols]
    groups = {}
    for c in feat_cols:
        groups.setdefault(group_of(c), []).append(c)
    return feats, groups

f_tr, group_map = build_features(tr_long, cache_name="feats_train_v4")
f_te, _         = build_features(te_long, cache_name="feats_test_v4")
print(f"\ntrain features: {f_tr.shape}")
print(f"test  features: {f_te.shape}")
print("\nFeature groups:")
for k, v in group_map.items():
    print(f"  {k:10s} -> {len(v)} cols")
""")

# ============================== LIGHTGBM ==============================
md("## 5. LightGBM training + ablation (Prompt 2)")

code(r"""# LightGBM with 5-fold GroupKFold by user
import lightgbm as lgb

LGB_PARAMS = dict(
    objective="multiclass", num_class=N_CLASSES, metric="multi_logloss",
    learning_rate=0.03, num_leaves=127, min_child_samples=15,
    feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=1,
    reg_alpha=0.0, reg_lambda=0.1, verbose=-1, seed=SEED, num_threads=-1,
)
N_BOOST   = 4000
EARLY_STOP = 150
N_SPLITS  = 5

def _train_one_fold_lgbm(X_tr, y_tr, X_va, y_va, feat_cols, w_tr=None):
    dtr = lgb.Dataset(X_tr, label=y_tr, feature_name=feat_cols, weight=w_tr)
    dva = lgb.Dataset(X_va, label=y_va, feature_name=feat_cols, reference=dtr)
    return lgb.train(
        LGB_PARAMS, dtr, num_boost_round=N_BOOST,
        valid_sets=[dtr, dva], valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                   lgb.log_evaluation(0)],
    )

def cv_lgbm(X, y, groups, feat_cols, sample_weights=None, verbose=True):
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros((len(y), N_CLASSES), dtype=np.float32)
    fold_f1, models = [], []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        w_tr = sample_weights[tr_idx] if sample_weights is not None else None
        m = _train_one_fold_lgbm(X[tr_idx], y[tr_idx], X[va_idx], y[va_idx], feat_cols, w_tr)
        proba = m.predict(X[va_idx], num_iteration=m.best_iteration)
        oof[va_idx] = proba
        f = f1_score(y[va_idx], proba.argmax(1), average="macro")
        fold_f1.append(f); models.append(m)
        if verbose:
            print(f"  fold {fold+1}: best_iter={m.best_iteration:4d}  macro-F1={f:.4f}")
    mean_f1, std_f1 = float(np.mean(fold_f1)), float(np.std(fold_f1))
    if verbose:
        print(f"  CV macro-F1 = {mean_f1:.4f} ± {std_f1:.4f}")
    return mean_f1, std_f1, oof, models

feat_cols = [c for c in f_tr.columns if c not in {"file_id", "user_id", "label"}]
X_tr_full = f_tr[feat_cols].to_numpy(dtype=np.float32)
y_tr_full = f_tr["label"].to_numpy(dtype=int)
groups_full = f_tr["user_id"].to_numpy()
ids_tr_full = f_tr["file_id"].to_numpy()
X_te_full = f_te[feat_cols].to_numpy(dtype=np.float32)
ids_te_full = f_te["file_id"].to_numpy()

# Class-balanced sample weights for LightGBM (inverse frequency, mean-normalised)
_lgbm_counts = np.bincount(y_tr_full, minlength=N_CLASSES).astype(float)
_lgbm_inv    = 1.0 / np.clip(_lgbm_counts, 1, None)
lgbm_sample_weights = (_lgbm_inv / _lgbm_inv.mean())[y_tr_full].astype(np.float32)
print("LightGBM class counts:", _lgbm_counts.astype(int).tolist())
print("Sample weight range: [{:.3f}, {:.3f}]".format(
      lgbm_sample_weights.min(), lgbm_sample_weights.max()))

def predict_test_lgbm(models, X_te):
    probs = np.zeros((X_te.shape[0], N_CLASSES), dtype=np.float32)
    for m in models:
        probs += m.predict(X_te, num_iteration=m.best_iteration)
    return probs / len(models)

# Bag 3 random seeds — different subsampling produces slightly different errors,
# and averaging them out gives a consistent ~0.003-0.005 OOF boost for free.
LGBM_SEEDS = [7, 13, 42]
oof_seeds_lgbm, test_seeds_lgbm = [], []
lgbm_models = None
for _s in LGBM_SEEDS:
    LGB_PARAMS["seed"] = _s
    print(f"\n=== LGBM CV (seed={_s}) ===")
    _, _, _oof, _models = cv_lgbm(
        X_tr_full, y_tr_full, groups_full, feat_cols,
        sample_weights=lgbm_sample_weights)
    oof_seeds_lgbm.append(_oof)
    test_seeds_lgbm.append(predict_test_lgbm(_models, X_te_full))
    lgbm_models = _models   # keep last seed's models for feature importance
    print(f"  OOF F1 (seed={_s}): {f1_score(y_tr_full, _oof.argmax(1), average='macro'):.4f}")

oof_proba_lgbm  = np.mean(oof_seeds_lgbm,  axis=0).astype(np.float32)
test_proba_lgbm = np.mean(test_seeds_lgbm, axis=0).astype(np.float32)
f1_lgbm_cv = f1_score(y_tr_full, oof_proba_lgbm.argmax(1), average="macro")
print(f"\nLGBM bagged ({len(LGBM_SEEDS)} seeds) OOF macro-F1 = {f1_lgbm_cv:.4f}")
""")

code(r"""# Feature-group ablation. This takes a while (~10-15 min), so I cache it
# and reuse on subsequent runs - just delete lgbm_ablation.csv to recompute.
def ablation_lgbm(X, y, groups, feat_cols, group_map, baseline_f1, sw=None):
    col_to_idx = {c: i for i, c in enumerate(feat_cols)}
    rows = []
    for g, cols in sorted(group_map.items()):
        keep_idx_minus = [i for c, i in col_to_idx.items() if group_of(c) != g]
        keep_cols_minus = [feat_cols[i] for i in keep_idx_minus]
        f1_m, _, _, _ = cv_lgbm(X[:, keep_idx_minus], y, groups, keep_cols_minus,
                                 sample_weights=sw, verbose=False)
        keep_idx_only = [col_to_idx[c] for c in cols]
        f1_o, _, _, _ = cv_lgbm(X[:, keep_idx_only], y, groups, cols,
                                 sample_weights=sw, verbose=False)
        delta = baseline_f1 - f1_m
        print(f"  [{g:10s}] only={f1_o:.4f}  all-minus={f1_m:.4f}  drop={delta:+.4f}  ({len(cols)} feats)")
        rows.append(dict(group=g, n_features=len(cols), only_f1=f1_o,
                         all_minus_f1=f1_m, drop_when_removed=delta))
    return pd.DataFrame(rows).sort_values("drop_when_removed", ascending=False)

_abl_cache = CACHE_DIR / "lgbm_ablation.csv"
if _abl_cache.exists():
    print(f"Using cached ablation: {_abl_cache.name}")
    abl_df = pd.read_csv(_abl_cache)
else:
    abl_df = ablation_lgbm(X_tr_full, y_tr_full, groups_full, feat_cols, group_map,
                            f1_lgbm_cv, sw=lgbm_sample_weights)
    abl_df.to_csv(_abl_cache, index=False)
print("\nAblation (sorted by drop when removed):")
print(abl_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
""")

code(r"""# Save submission_lgbm.csv + OOF cache.
# test_proba_lgbm was already computed (seed-averaged) in the bagging loop above.
test_pred = test_proba_lgbm.argmax(axis=1)

sub_template = pd.read_csv(SAMPLE_SUB)
pred_df = pd.DataFrame({"Id": ids_te_full.astype(sub_template['Id'].dtype),
                        "Label": test_pred.astype(int)})
sub = sub_template[["Id"]].merge(pred_df, on="Id", how="left")
sub["Label"] = sub["Label"].fillna(pd.Series(y_tr_full).mode().iloc[0]).astype(int)
sub.to_csv(SUB_DIR / "submission_lgbm.csv", index=False)
print(f"Saved {SUB_DIR / 'submission_lgbm.csv'}")

# Per-group feature importance
imp = np.zeros(len(feat_cols), dtype=np.float64)
for m in lgbm_models:
    imp += m.feature_importance(importance_type="gain")
imp_df = pd.DataFrame({"feature": feat_cols, "gain": imp,
                       "group": [group_of(c) for c in feat_cols]})
imp_df.sort_values("gain", ascending=False, inplace=True)
imp_df.to_csv(CACHE_DIR / "lgbm_feature_importance.csv", index=False)
print("\nGain by group:")
print(imp_df.groupby('group')['gain'].sum().sort_values(ascending=False).to_string())

np.savez(CACHE_DIR / "lgbm_oof.npz",
         oof_proba=oof_proba_lgbm, test_proba=test_proba_lgbm,
         y=y_tr_full, ids_train=ids_tr_full, ids_test=ids_te_full,
         groups=groups_full)
print("\nSaved cache/lgbm_oof.npz")

print("\n=== Per-class report on OOF (LGBM) ===")
print(classification_report(y_tr_full, oof_proba_lgbm.argmax(1), digits=4))
""")

# ============================== CNN-LSTM ==============================
md("""## 6. CNN-LSTM training (Prompt 3)

⏰ The training cell takes 30-60 min on CPU (~10 min on GPU).
If you skipped Cell 0a earlier, make sure PyTorch is installed first.""")

code(r"""# Verify PyTorch
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    print("torch", torch.__version__, "  CUDA:", torch.cuda.is_available())
except ImportError as e:
    print("PyTorch NOT installed. In a cell, run:")
    print("  %pip install torch --index-url https://download.pytorch.org/whl/cpu")
    raise
""")

code(r"""# Define HARDataset
def build_data_dict(long_df):
    out = {}
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

def compute_normalizer(data_dict, file_ids):
    stack = np.stack([data_dict[f] for f in file_ids], axis=0)
    mean = stack.reshape(-1, 6).mean(axis=0).astype(np.float32)
    std  = stack.reshape(-1, 6).std(axis=0).astype(np.float32) + 1e-6
    return mean, std

class HARDataset(Dataset):
    def __init__(self, data_dict, file_ids, labels_dict=None,
                 normalizer=None, augment=False,
                 jitter_sigma=0.02, shift_range=15,
                 scale_range=(0.95, 1.05),
                 p_jitter=0.5, p_shift=0.5, p_scale=0.3,
                 rng_seed=None):
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
        self.rng = np.random.default_rng(rng_seed)

    def __len__(self): return len(self.file_ids)

    def __getitem__(self, idx):
        fid = int(self.file_ids[idx])
        x = self.data[fid].copy()
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
        x = torch.from_numpy(np.ascontiguousarray(x.T))
        if self.labels is not None:
            return x, torch.tensor(self.labels[fid], dtype=torch.long)
        return x, torch.tensor(fid, dtype=torch.long)

print("HARDataset defined.")
""")

code(r"""# Define CNN-LSTM model
class CNNLSTM(nn.Module):
    def __init__(self, in_channels=6, n_classes=6, conv_channels=64,
                 lstm_hidden=128, lstm_layers=2, dropout=0.3):
        super().__init__()
        c1, c2 = conv_channels, conv_channels * 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, c1, 7, padding=3),
            nn.BatchNorm1d(c1), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Conv1d(c1, c2, 5, padding=2),
            nn.BatchNorm1d(c2), nn.ReLU(inplace=True),
            nn.MaxPool1d(2), nn.Dropout(dropout),
            nn.Conv1d(c2, c2, 3, padding=1),
            nn.BatchNorm1d(c2), nn.ReLU(inplace=True),
            nn.MaxPool1d(2), nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(c2, lstm_hidden, num_layers=lstm_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if lstm_layers > 1 else 0.0)
        head_in = lstm_hidden * 4
        self.head = nn.Sequential(
            nn.Linear(head_in, 64), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        h = self.conv(x)
        h = h.transpose(1, 2)
        h, _ = self.lstm(h)
        h_avg = h.mean(dim=1)
        h_max = h.max(dim=1).values
        h = torch.cat([h_avg, h_max], dim=1)
        return self.head(h)

print("CNNLSTM defined. Parameters:",
      sum(p.numel() for p in CNNLSTM().parameters() if p.requires_grad), "(approx)")
""")

code(r"""# Training-loop helpers
def set_seed_torch(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_one_fold_cnn(fold, data_dict, labels_dict, train_ids, val_ids,
                       test_data_dict, test_ids, device,
                       epochs, batch_size, lr, weight_decay, patience,
                       class_weights, out_ckpt):
    mean, std = compute_normalizer(data_dict, train_ids)
    train_ds = HARDataset(data_dict, train_ids, labels_dict,
                          normalizer=(mean, std), augment=True, rng_seed=SEED+fold)
    val_ds   = HARDataset(data_dict, val_ids, labels_dict,
                          normalizer=(mean, std), augment=False)
    test_ds  = HARDataset(test_data_dict, test_ids, None,
                          normalizer=(mean, std), augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=(device == "cuda"))
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=(device == "cuda"))

    model = CNNLSTM().to(device)
    cw = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    best_f1, best_state, no_improve = -1.0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            optim.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward(); optim.step()
            tot += loss.item() * xb.size(0); n += xb.size(0)
        sched.step()
        train_loss = tot / max(n, 1)

        model.eval()
        all_pred, all_true = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                all_pred.append(model(xb).argmax(1).cpu().numpy())
                all_true.append(yb.numpy())
        vp = np.concatenate(all_pred); vt = np.concatenate(all_true)
        val_f1 = f1_score(vt, vp, average="macro")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        print(f"  fold {fold+1} ep {epoch:02d}/{epochs} | loss {train_loss:.4f} | val_F1 {val_f1:.4f}"
              + ("  *" if no_improve == 0 else ""))
        if no_improve >= patience:
            print(f"  early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    torch.save(best_state, out_ckpt)

    def _proba(loader):
        model.eval()
        probs = []
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(device)
                probs.append(torch.softmax(model(xb), dim=1).cpu().numpy())
        return np.concatenate(probs)

    return _proba(val_loader), _proba(test_loader), best_f1

print("Training helpers defined.")
""")

code(r"""# CNN config — edit these for faster sanity runs
EPOCHS       = 50   # try 5 for a quick test
BATCH_SIZE   = 64
LR           = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE     = 12
FOLDS        = 5    # try 2 for a quick test
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE, " EPOCHS:", EPOCHS, " FOLDS:", FOLDS)

print("\nBuilding per-file tensors ...")
tr_data = build_data_dict(tr_long)
te_data = build_data_dict(te_long)
labels_dict = {int(r.file_id): int(r.label) for r in tr_meta.itertuples(index=False)}

tr_ids = tr_meta['file_id'].to_numpy()
tr_y_cnn = tr_meta['label'].to_numpy()
tr_users = tr_meta['user_id'].to_numpy()
te_ids = te_meta['file_id'].to_numpy()

counts = np.bincount(tr_y_cnn, minlength=N_CLASSES).astype(float)
inv = 1.0 / np.clip(counts, 1, None)
class_weights = (inv / inv.mean()).astype(np.float32)
print("Class weights:", np.round(class_weights, 3).tolist())
""")

code(r"""# ⏰ TRAIN CNN-LSTM (longest cell — 30-60 min CPU, ~10 min GPU)
set_seed_torch(SEED)
oof_proba_cnn = np.zeros((len(tr_ids), N_CLASSES), dtype=np.float32)
test_proba_cnn_sum = np.zeros((len(te_ids), N_CLASSES), dtype=np.float32)
fold_f1_cnn = []

gkf = GroupKFold(n_splits=FOLDS)
t0 = time.time()
for fold, (tr_idx, va_idx) in enumerate(gkf.split(tr_ids, tr_y_cnn, tr_users)):
    print(f"\n=== Fold {fold+1}/{FOLDS} (train={len(tr_idx)}  val={len(va_idx)}) ===")
    val_proba, test_proba, best_f1 = train_one_fold_cnn(
        fold=fold,
        data_dict=tr_data, labels_dict=labels_dict,
        train_ids=tr_ids[tr_idx], val_ids=tr_ids[va_idx],
        test_data_dict=te_data, test_ids=te_ids,
        device=DEVICE,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        lr=LR, weight_decay=WEIGHT_DECAY, patience=PATIENCE,
        class_weights=class_weights,
        out_ckpt=CACHE_DIR / f"cnn_fold{fold+1}.pt",
    )
    oof_proba_cnn[va_idx] = val_proba
    test_proba_cnn_sum += test_proba
    fold_f1_cnn.append(best_f1)
    print(f"  fold {fold+1} best val macro-F1: {best_f1:.4f}  ({(time.time()-t0)/60:.1f} min total)")

test_proba_cnn = test_proba_cnn_sum / FOLDS
final_cnn_f1 = f1_score(tr_y_cnn, oof_proba_cnn.argmax(1), average="macro")
print(f"\nCNN-LSTM pooled OOF macro-F1: {final_cnn_f1:.4f}")
print(f"Mean of folds: {np.mean(fold_f1_cnn):.4f} ± {np.std(fold_f1_cnn):.4f}")
print("\n=== Per-class report on OOF (CNN) ===")
print(classification_report(tr_y_cnn, oof_proba_cnn.argmax(1), digits=4))
""")

code(r"""# Save CNN submission + OOF cache
test_pred_cnn = test_proba_cnn.argmax(axis=1)
sub_template = pd.read_csv(SAMPLE_SUB)
pred_df = pd.DataFrame({"Id": te_ids.astype(sub_template['Id'].dtype),
                        "Label": test_pred_cnn.astype(int)})
sub = sub_template[["Id"]].merge(pred_df, on="Id", how="left")
sub["Label"] = sub["Label"].fillna(pd.Series(tr_y_cnn).mode().iloc[0]).astype(int)
sub.to_csv(SUB_DIR / "submission_cnn.csv", index=False)
print(f"Saved {SUB_DIR / 'submission_cnn.csv'}")

np.savez(CACHE_DIR / "cnn_oof.npz",
         oof_proba=oof_proba_cnn, test_proba=test_proba_cnn,
         y=tr_y_cnn, ids_train=tr_ids, ids_test=te_ids, groups=tr_users,
         fold_f1=np.array(fold_f1_cnn, dtype=np.float32))
print("Saved cache/cnn_oof.npz")
""")

# ============================== XGBOOST ==============================
md("""## 6.5. XGBoost — third base learner

I added this because LGBM + CNN alone gave me ~0.7256 OOF, which beat baseline 3
but only just. Boosted trees with *different* split mechanics than LightGBM
tend to be slightly orthogonal, which helps the ensemble. Run takes ~3-5 min.""")

code(r"""# XGBoost 5-fold CV with the same GroupKFold splits.
# Tried max_depth=6 first then bumped to 8 - that gave a small boost.
import xgboost as xgb

XGB_PARAMS = dict(
    objective="multi:softprob", num_class=N_CLASSES, eval_metric="mlogloss",
    learning_rate=0.05, max_depth=8, min_child_weight=3,
    subsample=0.85, colsample_bytree=0.8,
    reg_alpha=0.0, reg_lambda=1.0,
    tree_method="hist", verbosity=0, seed=SEED, nthread=-1,
)
N_BOOST_XGB = 3000
EARLY_STOP_XGB = 100

def cv_xgb(X, y, groups, sw=None):
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros((len(y), N_CLASSES), dtype=np.float32)
    fold_scores, models = [], []
    for k, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        w_tr = sw[tr_idx] if sw is not None else None
        dtrain = xgb.DMatrix(X[tr_idx], label=y[tr_idx], weight=w_tr)
        dvalid = xgb.DMatrix(X[va_idx], label=y[va_idx])
        bst = xgb.train(XGB_PARAMS, dtrain, num_boost_round=N_BOOST_XGB,
                        evals=[(dtrain, "train"), (dvalid, "valid")],
                        early_stopping_rounds=EARLY_STOP_XGB, verbose_eval=False)
        proba = bst.predict(xgb.DMatrix(X[va_idx]), iteration_range=(0, bst.best_iteration + 1))
        oof[va_idx] = proba
        fscore = f1_score(y[va_idx], proba.argmax(axis=1), average="macro")
        fold_scores.append(fscore); models.append(bst)
        print(f"  fold {k+1}: best_iter={bst.best_iteration:4d}  macro-F1={fscore:.4f}")
    print(f"  CV macro-F1 = {np.mean(fold_scores):.4f} +/-{np.std(fold_scores):.4f}")
    return np.mean(fold_scores), oof, models

def predict_test_xgb(models, X_te):
    out = np.zeros((X_te.shape[0], N_CLASSES), dtype=np.float32)
    dte = xgb.DMatrix(X_te)
    for m in models:
        out += m.predict(dte, iteration_range=(0, m.best_iteration + 1))
    return out / len(models)

# Bag XGBoost over 3 seeds — same variance-reduction trick as LGBM.
XGB_SEEDS = [7, 13, 42]
oof_seeds_xgb, test_seeds_xgb = [], []
for _s in XGB_SEEDS:
    XGB_PARAMS["seed"] = _s
    print(f"\n=== XGBoost CV (seed={_s}) ===")
    _, _oof, _models = cv_xgb(
        X_tr_full, y_tr_full, groups_full, sw=lgbm_sample_weights)
    oof_seeds_xgb.append(_oof)
    test_seeds_xgb.append(predict_test_xgb(_models, X_te_full))
    print(f"  OOF F1 (seed={_s}): {f1_score(y_tr_full, _oof.argmax(1), average='macro'):.4f}")

oof_proba_xgb  = np.mean(oof_seeds_xgb,  axis=0).astype(np.float32)
test_proba_xgb = np.mean(test_seeds_xgb, axis=0).astype(np.float32)
f1_xgb_cv = f1_score(y_tr_full, oof_proba_xgb.argmax(1), average="macro")
print(f"\nXGB bagged ({len(XGB_SEEDS)} seeds) OOF macro-F1: {f1_xgb_cv:.4f}")
np.savez(CACHE_DIR / "xgb_oof.npz",
         oof_proba=oof_proba_xgb, test_proba=test_proba_xgb,
         y=y_tr_full, ids_train=ids_tr_full, ids_test=ids_te_full, groups=groups_full)
print("Saved cache/xgb_oof.npz")
""")

# ============================== CATBOOST ==============================
md("""## 6.7. CatBoost — fourth base learner

CatBoost uses oblivious-tree splits (symmetric depth-wise), which behave
differently from both LightGBM (leaf-wise) and XGBoost (level-wise).
Different error structure = useful diversity in the ensemble.

Run takes ~5-10 min (depth=6, 1500 iterations with early stopping).""")

code(r"""# CatBoost 5-fold CV — single seed (depth=6 trains fast enough already)
from catboost import CatBoostClassifier, Pool

CAT_PARAMS = dict(
    iterations=1500,
    learning_rate=0.07,
    depth=6,
    l2_leaf_reg=3.0,
    bootstrap_type="Bernoulli",
    subsample=0.85,
    loss_function="MultiClass",
    classes_count=N_CLASSES,
    eval_metric="MultiClass",
    early_stopping_rounds=80,
    use_best_model=True,
    verbose=0,
    allow_writing_files=False,
    thread_count=-1,
    random_seed=SEED,
)

def cv_catboost(X, y, groups, sample_w=None):
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros((len(y), N_CLASSES), dtype=np.float32)
    fold_scores, models = [], []
    for k, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        w_tr = sample_w[tr_idx] if sample_w is not None else None
        train_pool = Pool(X[tr_idx], label=y[tr_idx], weight=w_tr)
        valid_pool = Pool(X[va_idx], label=y[va_idx])
        clf = CatBoostClassifier(**CAT_PARAMS)
        clf.fit(train_pool, eval_set=valid_pool)
        proba = clf.predict_proba(X[va_idx])
        oof[va_idx] = proba
        fscore = f1_score(y[va_idx], proba.argmax(axis=1), average="macro")
        fold_scores.append(fscore); models.append(clf)
        print(f"  fold {k+1}: best_iter={clf.get_best_iteration():4d}  macro-F1={fscore:.4f}")
    print(f"  CV macro-F1 = {np.mean(fold_scores):.4f} +/-{np.std(fold_scores):.4f}")
    return float(np.mean(fold_scores)), oof, models

def predict_test_cat(models, X_te):
    out = np.zeros((X_te.shape[0], N_CLASSES), dtype=np.float32)
    for m in models:
        out += m.predict_proba(X_te)
    return out / len(models)

# Class-balanced sample weights (reuse lgbm_sample_weights since same class balance)
print("=== CatBoost 5-fold CV ===")
f1_cat_cv, oof_proba_cat, cat_models = cv_catboost(
    X_tr_full, y_tr_full, groups_full, sample_w=lgbm_sample_weights)
test_proba_cat = predict_test_cat(cat_models, X_te_full)
print(f"\nCatBoost OOF macro-F1: {f1_cat_cv:.4f}")

np.savez(CACHE_DIR / "cat_oof.npz",
         oof_proba=oof_proba_cat, test_proba=test_proba_cat,
         y=y_tr_full, ids_train=ids_tr_full, ids_test=ids_te_full, groups=groups_full)
print("Saved cache/cat_oof.npz")
print("\n=== Per-class report on OOF (CatBoost) ===")
print(classification_report(y_tr_full, oof_proba_cat.argmax(1), digits=4))
""")

# ============================== MLP ==============================
md("""## 6.8. MLP on rich features — fifth base learner

A 3-layer MLP on the same 334 tabular features. Neural nets with smooth
activations make different prediction errors than boosted trees, so even
a modest MLP (OOF ~0.70) adds useful diversity to the ensemble.
Uses 3-seed bagging + StandardScaler per fold.""")

code(r"""# MLP model definition — 3 hidden layers with BatchNorm and Dropout
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

class SimpleMLP(nn.Module):
    def __init__(self, in_dim, hidden=256, n_classes=N_CLASSES, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_classes),
        )
    def forward(self, x):
        return self.net(x)

def _set_seed_mlp(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def _train_mlp_fold(train_x, train_y, valid_x, valid_y, seed=42,
                    epochs=80, batch_size=128, lr=1e-3, weight_decay=1e-4,
                    class_weights=None, device="cpu"):
    _set_seed_mlp(seed)
    sc = StandardScaler().fit(train_x)
    tx = sc.transform(train_x).astype(np.float32)
    vx = sc.transform(valid_x).astype(np.float32)
    train_ds = TensorDataset(torch.from_numpy(tx), torch.from_numpy(train_y).long())
    valid_ds = TensorDataset(torch.from_numpy(vx), torch.from_numpy(valid_y).long())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    model = SimpleMLP(in_dim=train_x.shape[1]).to(device)
    cw = (torch.tensor(class_weights, dtype=torch.float32, device=device)
          if class_weights is not None else None)
    crit = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    best_f1, best_state, no_imp, patience = -1.0, None, 0, 15
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad(); loss = crit(model(xb), yb); loss.backward(); optim.step()
        sched.step()
        model.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in valid_loader:
                preds.append(model(xb.to(device)).argmax(1).cpu().numpy())
        vp = np.concatenate(preds)
        f1 = f1_score(valid_y, vp, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        if no_imp >= patience:
            break
    model.load_state_dict(best_state)
    return model, sc, best_f1

def _mlp_proba(model, sc, x, device="cpu"):
    model.eval()
    xs = sc.transform(x).astype(np.float32)
    with torch.no_grad():
        logits = model(torch.from_numpy(xs).to(device))
        return torch.softmax(logits, dim=1).cpu().numpy()

print("MLP helpers defined. Parameters in one model:",
      sum(p.numel() for p in SimpleMLP(X_tr_full.shape[1]).parameters()))
""")

code(r"""# ⏰ Train MLP with 3-seed bagging (5 folds × 3 seeds = 15 training runs).
# Each run takes ~30 sec on CPU, so total ~7-8 min.
MLP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MLP_SEEDS  = [7, 13, 42]
print(f"MLP device: {MLP_DEVICE}  seeds: {MLP_SEEDS}")

# Class weights (inverse frequency)
_mlp_counts = np.bincount(y_tr_full, minlength=N_CLASSES).astype(float)
_mlp_inv    = 1.0 / np.clip(_mlp_counts, 1, None)
mlp_class_w = (_mlp_inv / _mlp_inv.mean()).astype(np.float32)

oof_seeds_mlp, test_seeds_mlp = [], []
for _s in MLP_SEEDS:
    print(f"\n=== MLP CV (seed={_s}) ===")
    gkf = GroupKFold(n_splits=5)
    _oof = np.zeros((len(y_tr_full), N_CLASSES), dtype=np.float32)
    _test_acc = np.zeros((X_te_full.shape[0], N_CLASSES), dtype=np.float32)
    for k, (tr_idx, va_idx) in enumerate(gkf.split(X_tr_full, y_tr_full, groups_full)):
        _model, _sc, _f1 = _train_mlp_fold(
            X_tr_full[tr_idx], y_tr_full[tr_idx],
            X_tr_full[va_idx], y_tr_full[va_idx],
            seed=_s, class_weights=mlp_class_w, device=MLP_DEVICE)
        _oof[va_idx] = _mlp_proba(_model, _sc, X_tr_full[va_idx], MLP_DEVICE)
        _test_acc   += _mlp_proba(_model, _sc, X_te_full, MLP_DEVICE)
        print(f"  fold {k+1}: macro-F1={_f1:.4f}")
    oof_seeds_mlp.append(_oof)
    test_seeds_mlp.append(_test_acc / 5)
    print(f"  seed={_s} OOF F1: {f1_score(y_tr_full, _oof.argmax(1), average='macro'):.4f}")

oof_proba_mlp  = np.mean(oof_seeds_mlp,  axis=0).astype(np.float32)
test_proba_mlp = np.mean(test_seeds_mlp, axis=0).astype(np.float32)
f1_mlp_cv = f1_score(y_tr_full, oof_proba_mlp.argmax(1), average="macro")
print(f"\nMLP bagged ({len(MLP_SEEDS)} seeds) OOF macro-F1 = {f1_mlp_cv:.4f}")

np.savez(CACHE_DIR / "mlp_oof.npz",
         oof_proba=oof_proba_mlp, test_proba=test_proba_mlp,
         y=y_tr_full, ids_train=ids_tr_full, ids_test=ids_te_full, groups=groups_full)
print("Saved cache/mlp_oof.npz")
print("\n=== Per-class report on OOF (MLP) ===")
print(classification_report(y_tr_full, oof_proba_mlp.argmax(1), digits=4))
""")

# ============================== ROCKET ==============================
md("""## 6.9. ROCKET — sixth base learner (random convolutional kernels)

This one I added late, and it turned out to matter. The three tree models
(LGBM, XGB, CatBoost) read the **same 334 tabular features**, so they make
nearly the same mistakes — adding more of them did almost nothing for the
blend. The blend was leaning entirely on the CNN for "different" predictions.

ROCKET attacks the raw 6×300 signal in a completely different way: it slides
a few thousand **random** little filters over the signal and, for each filter,
records two numbers — how often the output was positive (PPV) and the largest
output (max). A plain logistic regression on those numbers is a surprisingly
strong, well-known time-series method. Because it's random + linear, its errors
don't line up with the trees or the CNN, which is exactly the diversity the
ensemble was missing. Run takes ~10-15 min (the transform is the slow part).""")

code(r"""# ROCKET random-kernel transform on the raw 6x300 signal.
from sklearn.linear_model import LogisticRegression as _LR

N_KERNELS    = 4000
KERNEL_SEEDS = [42, 7]   # two random filter banks, averaged for stability

def _signal_block(long_df, ids):
    by_file = {}
    for fid, sub in long_df.groupby("file_id", sort=False):
        arr = sub.sort_values("index")[FEATURE_COLS].to_numpy(np.float32)
        if len(arr) < SEQ_LEN:
            arr = np.vstack([arr, np.zeros((SEQ_LEN - len(arr), 6), np.float32)])
        by_file[fid] = arr[:SEQ_LEN]
    return np.stack([by_file[f] for f in ids]).transpose(0, 2, 1)  # (n,6,300)

def _rocket(signal, n_kernels, seed):
    rng = np.random.default_rng(seed)
    n_files, n_ch, length = signal.shape
    out = np.zeros((n_files, 2 * n_kernels), np.float32)
    klens = rng.choice([7, 9, 11], n_kernels)
    for k in range(n_kernels):
        klen = klens[k]
        w = rng.standard_normal(klen).astype(np.float32); w -= w.mean()
        bias = rng.uniform(-1, 1)
        max_dil = int(np.log2((length - 1) / (klen - 1))) if length > klen else 0
        dil = int(2 ** rng.integers(0, max_dil + 1)) if max_dil > 0 else 1
        chans = rng.choice(n_ch, rng.integers(1, 3), replace=False)
        sig = signal[:, chans, :].sum(1)
        span = (klen - 1) * dil
        if span >= length: dil = 1; span = klen - 1
        conv = np.zeros((n_files, length - span), np.float32)
        for j in range(klen):
            conv += w[j] * sig[:, j * dil: j * dil + (length - span)]
        conv += bias
        out[:, 2*k]   = (conv > 0).mean(1)   # PPV
        out[:, 2*k+1] = conv.max(1)          # max
    return out

# Build + normalise raw signal blocks (normalise with TRAIN stats only)
_sig_tr = _signal_block(tr_long, ids_tr_full)
_sig_te = _signal_block(te_long, ids_te_full)
_mu = _sig_tr.mean((0,2), keepdims=True); _sd = _sig_tr.std((0,2), keepdims=True) + 1e-6
_sig_tr = (_sig_tr - _mu) / _sd; _sig_te = (_sig_te - _mu) / _sd
print(f"signal blocks: train {_sig_tr.shape}  test {_sig_te.shape}")

oof_seeds_rk, test_seeds_rk = [], []
for _s in KERNEL_SEEDS:
    print(f"\n=== ROCKET seed={_s} ({N_KERNELS} kernels) — transforming (slow) ===")
    ftr = _rocket(_sig_tr, N_KERNELS, _s)
    fte = _rocket(_sig_te, N_KERNELS, _s)
    _oof = np.zeros((len(y_tr_full), N_CLASSES), np.float32)
    _tacc = np.zeros((X_te_full.shape[0], N_CLASSES), np.float32)
    gkf = GroupKFold(5)
    for k,(tr,va) in enumerate(gkf.split(ftr, y_tr_full, groups_full)):
        sc = StandardScaler().fit(ftr[tr])
        clf = _LR(C=1.0, max_iter=3000, class_weight="balanced")
        clf.fit(sc.transform(ftr[tr]), y_tr_full[tr])
        _oof[va] = clf.predict_proba(sc.transform(ftr[va]))
        _tacc   += clf.predict_proba(sc.transform(fte))
        print(f"  fold {k+1}: macro-F1={f1_score(y_tr_full[va], _oof[va].argmax(1), average='macro'):.4f}")
    oof_seeds_rk.append(_oof); test_seeds_rk.append(_tacc / 5)
    print(f"  seed={_s} OOF F1: {f1_score(y_tr_full, _oof.argmax(1), average='macro'):.4f}")

oof_proba_rocket  = np.mean(oof_seeds_rk,  axis=0).astype(np.float32)
test_proba_rocket = np.mean(test_seeds_rk, axis=0).astype(np.float32)
f1_rocket_cv = f1_score(y_tr_full, oof_proba_rocket.argmax(1), average="macro")
print(f"\nROCKET bagged ({len(KERNEL_SEEDS)} seeds) OOF macro-F1 = {f1_rocket_cv:.4f}")
np.savez(CACHE_DIR / "rocket_oof.npz",
         oof_proba=oof_proba_rocket, test_proba=test_proba_rocket,
         y=y_tr_full, ids_train=ids_tr_full, ids_test=ids_te_full, groups=groups_full)
print("Saved cache/rocket_oof.npz")
""")

# ============================== ENSEMBLE ==============================
md("""## 7. Ensemble — 6-way hill-climbing blend + LR stacker + per-class tuning

Six base learners contribute OOF probability vectors:
- **LGBM** (3-seed bagged) · **XGBoost** (3-seed bagged) · **CatBoost** — tree models on tabular features (all ~0.72–0.73, but they make the *same* mistakes)
- **CNN-LSTM** (raw sequences, ~0.63) and **ROCKET** (random kernels, ~0.66) — the two *diverse* signals that actually move the blend
- **MLP** (tabular features, ~0.70)

Because there are 6 models, a full weight grid is too big, so the blend uses
**hill-climbing** (Caruana): start empty, repeatedly add whichever model most
improves OOF macro-F1; pick counts become the weights. We also try a
logistic-regression meta-learner on all 36 stacked OOF columns and keep the
better of the two. Finally, CV-folded per-class additive offset tuning squeezes
out a little more on the minority classes.""")

code(r"""# Align all OOF probabilities to LGBM file ordering (they may differ slightly)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from itertools import product

def _align(reference_ids, other_ids, other_proba):
    pos = {fid: i for i, fid in enumerate(other_ids.tolist())}
    idx = np.array([pos[f] for f in reference_ids.tolist()], dtype=int)
    return other_proba[idx]

y = y_tr_full
p_lgbm_tr = oof_proba_lgbm
p_lgbm_te = test_proba_lgbm
p_cnn_tr  = _align(ids_tr_full, tr_ids, oof_proba_cnn)
p_cnn_te  = _align(ids_te_full, te_ids, test_proba_cnn)
p_xgb_tr  = oof_proba_xgb
p_xgb_te  = test_proba_xgb
p_cat_tr  = oof_proba_cat
p_cat_te  = test_proba_cat
p_mlp_tr  = oof_proba_mlp
p_mlp_te  = test_proba_mlp
p_rocket_tr = oof_proba_rocket
p_rocket_te = test_proba_rocket

f1_lgbm  = f1_score(y, p_lgbm_tr.argmax(1),  average="macro")
f1_cnn   = f1_score(y, p_cnn_tr.argmax(1),   average="macro")
f1_xgb   = f1_score(y, p_xgb_tr.argmax(1),   average="macro")
f1_cat   = f1_score(y, p_cat_tr.argmax(1),   average="macro")
f1_mlp   = f1_score(y, p_mlp_tr.argmax(1),   average="macro")
f1_rocket= f1_score(y, p_rocket_tr.argmax(1),average="macro")
print(f"  LGBM   OOF macro-F1: {f1_lgbm:.4f}")
print(f"  CNN    OOF macro-F1: {f1_cnn:.4f}")
print(f"  XGB    OOF macro-F1: {f1_xgb:.4f}")
print(f"  CAT    OOF macro-F1: {f1_cat:.4f}")
print(f"  MLP    OOF macro-F1: {f1_mlp:.4f}")
print(f"  ROCKET OOF macro-F1: {f1_rocket:.4f}")

# 6-way stacked features (6 models x 6 classes = 36 columns)
stack_tr = np.hstack([p_lgbm_tr, p_cnn_tr, p_xgb_tr, p_cat_tr, p_mlp_tr, p_rocket_tr])
stack_te = np.hstack([p_lgbm_te, p_cnn_te, p_xgb_te, p_cat_te, p_mlp_te, p_rocket_te])

def stacker_oof(C):
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros((len(y), N_CLASSES), dtype=np.float32)
    for tr_idx, va_idx in gkf.split(stack_tr, y, groups_full):
        sc = StandardScaler().fit(stack_tr[tr_idx])
        clf = LogisticRegression(C=C, solver="lbfgs", max_iter=2000, random_state=SEED)
        clf.fit(sc.transform(stack_tr[tr_idx]), y[tr_idx])
        oof[va_idx] = clf.predict_proba(sc.transform(stack_tr[va_idx]))
    return oof

print(f"\n--- Method A: LR stacker (6 base models -> {stack_tr.shape[1]} stacked features) ---")
best_C, best_stack_f1, best_stack_oof = None, -1.0, None
for C in [0.1, 0.3, 1.0, 3.0, 10.0]:
    oof = stacker_oof(C)
    fs = f1_score(y, oof.argmax(1), average="macro")
    print(f"  C={C:>5}: OOF F1 = {fs:.4f}")
    if fs > best_stack_f1:
        best_C, best_stack_f1, best_stack_oof = C, fs, oof
print(f"  best stacker C={best_C}, OOF F1={best_stack_f1:.4f}")

sc_full = StandardScaler().fit(stack_tr)
final_lr = LogisticRegression(C=best_C, solver="lbfgs", max_iter=2000, random_state=SEED)
final_lr.fit(sc_full.transform(stack_tr), y)
stack_te_proba = final_lr.predict_proba(sc_full.transform(stack_te))

# --- Method B: hill-climbing weighted ensemble (Caruana) ---
# With 6 models a full weight grid is too big, so we hill-climb: start from an
# empty blend and repeatedly add (with replacement) whichever model most
# improves OOF macro-F1. Each model's pick-count becomes its weight.
print("\n--- Method B: hill-climbing weighted ensemble ---")
_blend = [("LGBM", p_lgbm_tr, p_lgbm_te), ("CNN", p_cnn_tr, p_cnn_te),
          ("XGB", p_xgb_tr, p_xgb_te),    ("CAT", p_cat_tr, p_cat_te),
          ("MLP", p_mlp_tr, p_mlp_te),    ("ROCKET", p_rocket_tr, p_rocket_te)]
HILL_STEPS = 25
_acc = np.zeros_like(p_lgbm_tr)
_picks = []
for _ in range(HILL_STEPS):
    _bi, _bf = None, -1.0
    for i, (_, ptr, _t) in enumerate(_blend):
        f = f1_score(y, (_acc + ptr).argmax(1), average="macro")
        if f > _bf: _bf, _bi = f, i
    _acc = _acc + _blend[_bi][1]; _picks.append(_bi)
_counts = {i: _picks.count(i) for i in range(len(_blend))}
blend_oof      = _acc / HILL_STEPS
blend_te_proba = sum(_counts.get(i,0) * _blend[i][2] for i in range(len(_blend))) / HILL_STEPS
best_blend_f1  = f1_score(y, blend_oof.argmax(1), average="macro")
print("  weights: " + "  ".join(f"{_blend[i][0]}={_counts.get(i,0)/HILL_STEPS:.2f}"
                                 for i in range(len(_blend))))
print(f"  OOF F1={best_blend_f1:.4f}")

# Pick the winner
if best_stack_f1 >= best_blend_f1:
    chosen = "stacker"
    final_te_proba = stack_te_proba
    final_oof = best_stack_oof
    final_f1 = best_stack_f1
else:
    chosen = "blend"
    final_te_proba = blend_te_proba
    final_oof = blend_oof
    final_f1 = best_blend_f1
print(f"\n>>> Picked: {chosen}  (OOF F1 = {final_f1:.4f})")
""")

code(r"""# --- Per-class additive prior tuning (CV-folded to avoid overfitting) ---
# For each class we search an additive offset that maximises OOF macro-F1.
# We tune on each CV training fold (not the same fold we evaluate on) so the
# offsets generalise rather than memorising the OOF noise.
print("--- Per-class additive prior tuning (CV-folded greedy search) ---")
_grid = np.linspace(-0.30, 0.30, 31)

def _greedy_search(score_fn, n_iters=3):
    offs = np.zeros(N_CLASSES, dtype=np.float32)
    cur = score_fn(offs)
    for _ in range(n_iters):
        improved = False
        for c in range(N_CLASSES):
            best_l = (cur, offs[c])
            for delta in _grid:
                trial = offs.copy(); trial[c] = delta
                f = score_fn(trial)
                if f > best_l[0] + 1e-6:
                    best_l = (f, delta); improved = True
            if best_l[1] != offs[c]:
                offs[c] = best_l[1]; cur = best_l[0]
        if not improved:
            break
    return offs, cur

gkf5 = GroupKFold(n_splits=5)
fold_offsets = []
for _tr_idx, _va_idx in gkf5.split(final_oof, y, groups_full):
    def _sc_cv(off, _ti=_tr_idx):
        adj = final_oof[_ti] + off
        return f1_score(y[_ti], adj.argmax(1), average="macro")
    _o, _ = _greedy_search(_sc_cv)
    fold_offsets.append(_o)
cv_offsets   = np.mean(fold_offsets, axis=0)
full_offsets, _ = _greedy_search(
    lambda off: f1_score(y, (final_oof + off).argmax(1), average="macro"))

def _cv_score(off):
    scores = []
    for _ti, _vi in gkf5.split(final_oof, y, groups_full):
        scores.append(f1_score(y[_vi], (final_oof[_vi] + off).argmax(1), average="macro"))
    return float(np.mean(scores))

if _cv_score(cv_offsets) >= _cv_score(full_offsets):
    offsets = cv_offsets;  print("  picked CV-averaged offsets")
else:
    offsets = full_offsets; print("  picked full-OOF offsets")
print(f"  offsets = {np.round(offsets, 3).tolist()}")

final_oof      = final_oof      + offsets
final_te_proba = final_te_proba + offsets
final_f1 = f1_score(y, final_oof.argmax(1), average="macro")
print(f"  OOF F1 after per-class tuning: {final_f1:.4f}")

# --- Save final submission ---
final_te_pred = final_te_proba.argmax(1)
sub_template = pd.read_csv(SAMPLE_SUB)
pred_df = pd.DataFrame({"Id": ids_te_full.astype(sub_template['Id'].dtype),
                        "Label": final_te_pred.astype(int)})
sub = sub_template[["Id"]].merge(pred_df, on="Id", how="left")
sub["Label"] = sub["Label"].fillna(pd.Series(y).mode().iloc[0]).astype(int)
sub.to_csv(SUB_DIR / "submission_final.csv", index=False)
print(f"\nSaved {SUB_DIR / 'submission_final.csv'}")

np.savez(CACHE_DIR / "ensemble_oof.npz",
         oof_proba=final_oof, test_proba=final_te_proba,
         y=y, ids_train=ids_tr_full, ids_test=ids_te_full,
         f1_final=np.array([final_f1], dtype=np.float32),
         f1_lgbm=np.array([f1_lgbm], dtype=np.float32),
         f1_cnn=np.array([f1_cnn], dtype=np.float32),
         f1_xgb=np.array([f1_xgb], dtype=np.float32),
         f1_cat=np.array([f1_cat], dtype=np.float32),
         f1_mlp=np.array([f1_mlp], dtype=np.float32),
         f1_rocket=np.array([f1_rocket], dtype=np.float32),
         f1_stacker=np.array([best_stack_f1], dtype=np.float32),
         f1_blend=np.array([best_blend_f1], dtype=np.float32))

best_f1 = final_f1  # alias for downstream report cells

print("\n=== Per-class report (final OOF) ===")
print(classification_report(y, final_oof.argmax(1), digits=4))
print("\n========== SUMMARY ==========")
print(f"  LGBM    OOF macro-F1: {f1_lgbm:.4f}")
print(f"  CNN     OOF macro-F1: {f1_cnn:.4f}")
print(f"  XGB     OOF macro-F1: {f1_xgb:.4f}")
print(f"  CAT     OOF macro-F1: {f1_cat:.4f}")
print(f"  MLP     OOF macro-F1: {f1_mlp:.4f}")
print(f"  ROCKET  OOF macro-F1: {f1_rocket:.4f}")
print(f"  Stacker OOF macro-F1: {best_stack_f1:.4f}")
print(f"  Blend   OOF macro-F1: {best_blend_f1:.4f}")
print(f"  Final ({chosen}) + prior-tuned OOF: {final_f1:.4f}")
""")

# ============================== REPORT ASSETS ==============================
md("## 8. Report assets — confusion matrix + per-class F1 chart (Prompt 5)")

code(r"""# Confusion matrices + per-class F1 bar chart for all 5 models
import seaborn as sns
from sklearn.metrics import confusion_matrix

def confusion_fig(y_true, y_pred, title, fname):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues", cbar=True, ax=ax,
                xticklabels=list(range(N_CLASSES)),
                yticklabels=list(range(N_CLASSES)))
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(FIG_DIR / fname); plt.show()

def per_class_f1_chart(y_true, p_lgbm, p_cnn, p_xgb, p_cat, p_mlp, p_rocket, p_ens):
    def pc(y_t, y_p):
        return [f1_score(y_t == c, y_p == c, average="binary") for c in range(N_CLASSES)]
    bars = [("LGBM",   pc(y_true, p_lgbm.argmax(1))),
            ("CNN",    pc(y_true, p_cnn.argmax(1))),
            ("XGB",    pc(y_true, p_xgb.argmax(1))),
            ("CAT",    pc(y_true, p_cat.argmax(1))),
            ("MLP",    pc(y_true, p_mlp.argmax(1))),
            ("ROCKET", pc(y_true, p_rocket.argmax(1))),
            ("Ens",    pc(y_true, p_ens.argmax(1)))]
    width = 0.115; xs = np.arange(N_CLASSES); n = len(bars)
    fig, ax = plt.subplots(figsize=(12, 4))
    for i, (lbl, vals) in enumerate(bars):
        ax.bar(xs + (i - n/2 + 0.5)*width, vals, width=width, label=lbl)
    ax.set_xticks(xs); ax.set_xlabel("Class label"); ax.set_ylabel("F1 (one-vs-rest)")
    ax.set_title("Per-class F1 by model (OOF)")
    ax.set_ylim(0, 1.0); ax.legend(ncol=4); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG_DIR / "10_per_class_f1.png"); plt.show()

confusion_fig(y, final_oof.argmax(1),      "6-way Ensemble OOF",   "07_confusion_primary.png")
confusion_fig(y, p_lgbm_tr.argmax(1),      "LGBM OOF",             "08_confusion_lgbm.png")
confusion_fig(y, p_cnn_tr.argmax(1),       "CNN-LSTM OOF",         "09_confusion_cnn.png")
confusion_fig(y, p_xgb_tr.argmax(1),       "XGBoost OOF",          "11_confusion_xgb.png")
confusion_fig(y, p_cat_tr.argmax(1),       "CatBoost OOF",         "12_confusion_cat.png")
confusion_fig(y, p_mlp_tr.argmax(1),       "MLP OOF",              "13_confusion_mlp.png")
confusion_fig(y, p_rocket_tr.argmax(1),    "ROCKET OOF",           "14_confusion_rocket.png")
per_class_f1_chart(y, p_lgbm_tr, p_cnn_tr, p_xgb_tr, p_cat_tr, p_mlp_tr, p_rocket_tr, final_oof)

# Save summary markdown
md_scores = "| Stage | OOF macro-F1 |\n|---|---|\n"
md_scores += f"| Naive (best of LogReg/RF)  | {max(f1_lr, f1_rf):.4f} |\n"
md_scores += f"| LightGBM (rich features)   | {f1_lgbm:.4f} |\n"
md_scores += f"| CNN-LSTM (raw sequences)   | {f1_cnn:.4f} |\n"
md_scores += f"| XGBoost (rich features)    | {f1_xgb:.4f} |\n"
md_scores += f"| CatBoost (rich features)   | {f1_cat:.4f} |\n"
md_scores += f"| MLP (tabular features)     | {f1_mlp:.4f} |\n"
md_scores += f"| ROCKET (random kernels)    | {f1_rocket:.4f} |\n"
md_scores += f"| 6-way Ensemble (final)     | {final_f1:.4f} |\n"
(REPORT_DIR / "summary_scores.md").write_text(md_scores, encoding="utf-8")
print(f"\nSaved {REPORT_DIR / 'summary_scores.md'}")
print(md_scores)
""")

# ============================== PDF REPORT ==============================
md("""## 9. Final PDF report (Prompt 6)

**Edit the 4 variables** below with your real student ID, GitHub URL, and
Kaggle public F1 (after you upload `submission_final.csv`).""")

code(r"""# ============== EDIT THIS BLOCK ==============
STUDENT_ID       = "314540061"                       # your student ID
GITHUB_URL       = "https://github.com/Lappykentang/DM2026-Assignment-3"  # your repo
KAGGLE_PUBLIC_F1 = 0.8267                             # final public macro-F1
KAGGLE_RANK      = None                              # optional, e.g. "3/180"
NAIVE_CV_F1      = max(f1_lr, f1_rf)                 # auto-filled from cell above
# =============================================
print("STUDENT_ID  :", STUDENT_ID)
print("GITHUB_URL  :", GITHUB_URL)
print("KAGGLE_F1   :", KAGGLE_PUBLIC_F1)
""")

code(r"""# Generate PDF + README
from fpdf import FPDF

_UNICODE_FIX = {"—": "-", "–": "-", "→": "->", "←": "<-", "×": "x",
                "±": "+/-", "Δ": "Delta", "≈": "~=", "²": "^2", "•": "*"}
def _ascii(s):
    for k, v in _UNICODE_FIX.items():
        s = s.replace(k, v)
    return s

class ReportPDF(FPDF):
    def header(self):
        self.set_font("helvetica", "B", 10)
        self.cell(0, 8, _ascii("NYCU Data Mining Assignment 3 - HAR Report"),
                  new_x="LMARGIN", new_y="NEXT", align="C")
        self.ln(2)
    def footer(self):
        self.set_y(-15); self.set_font("helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")
    def h1(self, t):
        self.set_font("helvetica", "B", 14); self.ln(2)
        self.cell(0, 8, _ascii(t), new_x="LMARGIN", new_y="NEXT"); self.ln(1)
    def h2(self, t):
        self.set_font("helvetica", "B", 11); self.ln(1)
        self.cell(0, 6, _ascii(t), new_x="LMARGIN", new_y="NEXT"); self.ln(0.5)
    def body(self, t):
        self.set_font("helvetica", "", 10)
        self.multi_cell(0, 5, _ascii(t)); self.ln(1)
    def table(self, headers, rows, col_widths=None):
        self.set_font("helvetica", "B", 9)
        n = len(headers); col_widths = col_widths or [self.epw / n] * n
        for h, w in zip(headers, col_widths):
            self.cell(w, 6, _ascii(str(h)), border=1, align="C")
        self.ln(); self.set_font("helvetica", "", 9)
        for r in rows:
            for v, w in zip(r, col_widths):
                self.cell(w, 5.5, _ascii(str(v)), border=1)
            self.ln()
        self.ln(1)
    def image_if_exists(self, path, width_mm=170):
        if path.exists():
            self.image(str(path), w=width_mm); self.ln(2)

scores = {
    "Naive (best of LogReg/RF)":    max(f1_lr, f1_rf),
    "LightGBM (rich features)":     f1_lgbm,
    "CNN-LSTM (raw sequences)":     f1_cnn,
    "XGBoost (rich features)":      f1_xgb,
    "CatBoost (rich features)":     f1_cat,
    "MLP (tabular features)":       f1_mlp,
    "ROCKET (random kernels)":      f1_rocket,
    "6-way Ensemble (final)":       final_f1,
}

pdf = ReportPDF(orientation="P", unit="mm", format="A4")
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()
pdf.set_font("helvetica", "B", 16)
pdf.cell(0, 10, _ascii("Human Activity Recognition - HAR"),
         new_x="LMARGIN", new_y="NEXT", align="C")
pdf.set_font("helvetica", "", 10)
pdf.cell(0, 6, _ascii(f"Student ID: {STUDENT_ID}"), new_x="LMARGIN", new_y="NEXT", align="C")
pdf.cell(0, 6, _ascii(f"GitHub:    {GITHUB_URL}"), new_x="LMARGIN", new_y="NEXT", align="C")
if KAGGLE_PUBLIC_F1 is not None:
    rank = f"   rank {KAGGLE_RANK}" if KAGGLE_RANK else ""
    pdf.cell(0, 6, _ascii(f"Kaggle public F1: {KAGGLE_PUBLIC_F1:.4f}{rank}"),
             new_x="LMARGIN", new_y="NEXT", align="C")
pdf.ln(4)

# Section 1
pdf.h1("1. Preliminary Analysis")
pdf.body(
    "The dataset contains 5-minute wrist-accelerometer recordings aggregated to "
    "300 one-second rows of {mean_x, mean_y, mean_z, std_x, std_y, std_z} per file. "
    "Each file has exactly one of six activity labels. Train: 60 users; "
    "Test: a disjoint 40 users (no user overlap). This makes user-level "
    "generalization the main difficulty, so all CV uses GroupKFold by user. "
    "We confirmed the train/test users are disjoint and no files contain NaNs."
)
pdf.image_if_exists(FIG_DIR / "01_label_distribution.png")
pdf.image_if_exists(FIG_DIR / "04_signal_examples_mean.png", width_mm=170)
pdf.body(
    f"The label distribution is moderately imbalanced. A naive aggregate baseline "
    f"(12 features = time-mean and time-std of the 6 columns) reached macro-F1 "
    f"~ {NAIVE_CV_F1:.4f} under 5-fold GroupKFold(user). This is the starting "
    f"line on top of which we measure all subsequent improvements."
)

# Section 2
pdf.add_page(); pdf.h1("2. Preprocessing Techniques")
pdf.body(
    "We split feature work into named groups and measured the per-group "
    "macro-F1 contribution via leave-one-group-out ablation with LightGBM "
    "under 5-fold GroupKFold:"
)
pdf.body(
    "  basic    : per-column mean / std / min / max / median\n"
    "  pct      : per-column p10, p25, p75, p90, IQR\n"
    "  moments  : per-column skew + kurtosis\n"
    "  mag      : signal magnitude m = sqrt(mx^2+my^2+mz^2) summaries\n"
    "  sma      : signal magnitude area = mean(|mx|+|my|+|mz|)\n"
    "  corr     : pairwise correlations across axes\n"
    "  jerk     : 1st differences of mean_xyz (motion smoothness)\n"
    "  fft      : per-axis FFT energy in 5 bands + spectral entropy + dom freq\n"
    "  crossings: mean-crossings per axis\n"
    "  std_sum  : statistics of per-second within-second std columns"
)
abl_path = CACHE_DIR / "lgbm_ablation.csv"
if abl_path.exists():
    abl = pd.read_csv(abl_path).sort_values("drop_when_removed", ascending=False)
    rows = [[r["group"], int(r["n_features"]),
             f"{r['only_f1']:.4f}", f"{r['all_minus_f1']:.4f}",
             f"{r['drop_when_removed']:+.4f}"] for _, r in abl.iterrows()]
    pdf.table(["Group", "# feats", "only-F1", "all-group F1", "drop when removed"],
              rows, col_widths=[34, 22, 26, 30, 32])
else:
    pdf.body("(LGBM ablation not run - cache file missing.)")

pdf.h2("F1 improvements relative to the naive baseline")
rows = []
base = NAIVE_CV_F1
for k, v in scores.items():
    rows.append([k, f"{v:.4f}", f"{v - base:+.4f}"])
pdf.table(["Stage", "CV macro-F1", "delta vs naive"], rows, col_widths=[80, 40, 40])

# Section 3
pdf.add_page(); pdf.h1("3. Temporal Alignment")
pdf.body(
    "Each file is a sequence of 300 one-second rows, so the model must capture "
    "WHEN events happen, not just summary statistics. We address this in two ways:"
)
pdf.h2("(a) Hand-crafted temporal features (LightGBM input)")
pdf.body(
    "FFT energy in 5 bands per axis + spectral entropy + dominant frequency "
    "(captures periodicity / cadence); jerk = 1st-order time differences of "
    "mean_xyz (captures motion abruptness); mean-crossing rates (oscillation "
    "frequency proxy); cross-axis correlations (postural vs ambulatory)."
)
pdf.h2("(b) Sequence model (CNN-LSTM)")
pdf.body(
    "Architecture on the raw 6x300 signal:\n"
    "  Conv1d(7) -> BN+ReLU -> Conv1d(5) -> BN+ReLU+Pool -> Conv1d(3) -> BN+ReLU+Pool\n"
    "  -> BiLSTM x 2 (hidden 128) -> time-wise avg+max pool -> FC(64) -> FC(6)\n\n"
    "Conv blocks extract local, translation-invariant motion patterns and "
    "downsample 300 -> 75 timesteps. BiLSTM models long-range temporal "
    "dependencies bidirectionally. Concatenated avg+max pooling gives a "
    "position-invariant representation, making the prediction robust to where "
    "in the 5-minute window the activity is most evident. Training uses "
    "per-fold normalization and three augmentations applied only during "
    "training: Gaussian jitter, circular time-shift (+/-15 s), and magnitude "
    "scaling (0.95-1.05)."
)
pdf.h2("(c) CatBoost (oblivious trees)")
pdf.body(
    "CatBoost uses depth-first symmetric (oblivious) trees where every node "
    "at the same depth uses the same split condition. This gives slightly "
    "different decision boundaries than LightGBM (leaf-wise) and XGBoost "
    "(level-wise), contributing orthogonal errors to the ensemble. "
    "Single seed, depth=6, 1500 iterations with early stopping."
)
pdf.h2("(d) MLP on tabular features")
pdf.body(
    "A 3-layer MLP (in->256->128->6) with BatchNorm and Dropout on the same "
    "334 tabular features. Neural networks with smooth activations make "
    "prediction errors that are systematically different from gradient boosting, "
    "so adding even a modest MLP (OOF ~0.70) helps the ensemble. "
    "Trained with StandardScaler per fold, CosineAnnealingLR, and 3-seed bagging."
)
pdf.h2("(e) ROCKET (random convolutional kernels)")
pdf.body(
    "ROCKET convolves the raw 6x300 signal with several thousand RANDOM dilated "
    "filters and summarises each filter's output by two statistics: the "
    "proportion of positive values (PPV) and the maximum. A logistic regression "
    "on these features is a strong, well-known time-series classifier. The three "
    "gradient-boosting models share the same 334 tabular features and therefore "
    "make highly correlated errors; ROCKET reads the raw signal in a completely "
    "different (random + linear) way, so its mistakes are de-correlated from both "
    "the trees and the CNN-LSTM. Adding it lifted the cross-validated ensemble "
    "macro-F1 by about +0.005 - the single most useful late addition. "
    "Two random kernel banks (seeds 42, 7) are averaged for stability."
)

# Section 4
pdf.add_page(); pdf.h1("4. Ablation Study")
pdf.body("Controlled experiments testing the core design choices. All numbers "
         "are 5-fold GroupKFold(user) macro-F1.")
pdf.h2("(a) Feature group contribution (LightGBM)")
pdf.body("See ablation table in Section 2. Positive delta when removed = group "
         "adds discriminative signal.")
pdf.h2("(b) Model class comparison")
pdf.table(["Model", "CV macro-F1"],
          [[k, f"{v:.4f}"] for k, v in scores.items()],
          col_widths=[110, 60])
pdf.h2("(c) Confusion analysis & per-class F1")
pdf.body("The ensemble's confusion matrix and per-class F1 bar chart show which "
         "classes remain confusable. Classes with lower F1 typically share "
         "similar wrist motion signatures.")
pdf.image_if_exists(FIG_DIR / "07_confusion_primary.png", width_mm=120)
pdf.image_if_exists(FIG_DIR / "10_per_class_f1.png", width_mm=170)
pdf.h2("(d) Augmentations and class weighting (CNN-LSTM)")
pdf.body(
    "Jitter + time-shift consistently helps the CNN-LSTM generalize across "
    "users; inverse-frequency class weighting raises macro-F1 on minority "
    "classes; blending CNN-LSTM with tree models via weighted soft-vote "
    "improves macro-F1 over any single model, indicating model families "
    "capture complementary patterns (temporal sequences vs tabular statistics)."
)
pdf.h2("(e) Per-class additive prior tuning")
pdf.body(
    "After blending, we run a greedy search for an additive offset per class "
    "on the combined probability vector. This shifts the decision boundary for "
    "hard/rare classes (2, 4, 5) without retraining. Tuning is done CV-folded "
    "(learn offsets on 4 folds, evaluate on 1) to prevent overfitting to OOF "
    "noise. Gives ~0.002-0.005 additional macro-F1 improvement."
)

# Section 5
pdf.add_page(); pdf.h1("5. Final Summary")
pdf.table(["Stage", "CV macro-F1"],
          [[k, f"{v:.4f}"] for k, v in scores.items()], col_widths=[110, 60])
if KAGGLE_PUBLIC_F1 is not None:
    rank = f", rank {KAGGLE_RANK}" if KAGGLE_RANK else ""
    pdf.h2("Kaggle public leaderboard")
    pdf.body(f"Final submission macro-F1: {KAGGLE_PUBLIC_F1:.4f}{rank}.")
pdf.h2("Reproducibility")
pdf.body(f"SEED=42. Code public at: {GITHUB_URL}.")

out_pdf = REPORT_DIR / f"DM_asg3_{STUDENT_ID}.pdf"
pdf.output(str(out_pdf))
print(f"Saved {out_pdf}")

# README.md (built line-by-line to avoid nested triple-quotes)
lines = []
lines.append("# NYCU Data Mining Assignment 3 - HAR")
lines.append("")
lines.append("| Stage | CV macro-F1 |")
lines.append("|---|---|")
for k, v in scores.items():
    lines.append(f"| {k} | {v:.4f} |")
lines.append("")
if KAGGLE_PUBLIC_F1 is not None:
    rank_str = f" (rank {KAGGLE_RANK})" if KAGGLE_RANK else ""
    lines.append(f"Kaggle public leaderboard: **{KAGGLE_PUBLIC_F1:.4f}**{rank_str}")
    lines.append("")
lines.append("## Run")
lines.append("")
lines.append("Open HAR_pipeline_inline.ipynb in VS Code (or Jupyter) and run cells top-to-bottom.")
lines.append("")
lines.append(f"Student ID: **{STUDENT_ID}**")
readme = "\n".join(lines)
(REPORT_DIR / "README.md").write_text(readme, encoding="utf-8")
print(f"Saved {REPORT_DIR / 'README.md'}")
""")

# ============================== SUBMIT ==============================
md("""## 10. Submit

You're done with the code. Final steps (manual):

1. **Upload** `outputs/submissions/submission_final.csv` to **Kaggle**.
2. **Submit** `outputs/report/DM_asg3_{your_id}.pdf` via **E3**.
3. **Push** this notebook to your **public GitHub repo**.
4. **Set Kaggle display name** = your student ID (otherwise grade = 0!).
""")

# ============================ WRITE NOTEBOOK ============================
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3 (.venv)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = Path("HAR_pipeline_inline.ipynb")
out_path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(f"Wrote {out_path.resolve()} ({len(cells)} cells)")
