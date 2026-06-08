# Per-file feature engineering.
#
# Each file = 300 rows x 6 cols (mean_x/y/z, std_x/y/z) -> one feature row.
# I grouped the features by the kind of statistic they capture so I could
# run an ablation (drop one group at a time, see which one mattered most).
#
# Groups (current count - I added the bottom 4 after seeing weak per-class
# F1 for the hard classes):
#   basic     - mean / std / min / max / median per column
#   pct       - percentiles p10/p25/p75/p90 + IQR per column
#   moments   - skew + kurtosis per column
#   mag       - magnitude m = sqrt(mx^2+my^2+mz^2)
#   sma       - signal magnitude area on |mx|+|my|+|mz|
#   corr      - pairwise corr between mean_x/y/z over time
#   jerk      - 1st difference of mean_xyz over time
#   fft       - per-axis FFT energy in 5 bands + spectral entropy + dom freq
#   std_sum   - stats of std_x/y/z (the within-second variance)
#   crossings - mean-crossing rate per axis
#   autocorr  - autocorrelation at lags 1/2/5/10/20 (NEW - tried this first
#               for step cadence, it helped a tiny bit)
#   seg       - mean/std/range per temporal third (NEW)
#   peaks     - peak count + height per axis (NEW)
#   energy    - signal energy + histogram entropy + rms (NEW)
#   ratio     - cross-axis variance/range/energy ratios (NEW)
#   tails     - extreme percentiles p5/p95/p99 (NEW)
#   covmix    - coeff. of variation + mean/std coupling + std cross-corr (v4)
#
# v4 update: I also ran the autocorr / peak / tail extractors over the
# std_x/y/z channels (previously only the mean channels got them). The std
# channels are the within-second motion intensity, and their periodicity
# (step cadence) turned out to be the most useful late feature addition.
# (Note: I also tried gravity-alignment and multi-scale wavelet features in
# later experiments, but both regressed on the leaderboard, so they were
# dropped. The v4 feature set below is the final one.)
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

from config import FEATURE_COLS, SEQ_LEN, CACHE_DIR

MEAN_COLS = ["mean_x", "mean_y", "mean_z"]
STD_COLS  = ["std_x", "std_y", "std_z"]


# ------------------------- per-group feature builders ------------------------- #
def _basic(arr: np.ndarray, prefix: str) -> dict:
    return {
        f"{prefix}_mean":   float(np.mean(arr)),
        f"{prefix}_std":    float(np.std(arr)),
        f"{prefix}_min":    float(np.min(arr)),
        f"{prefix}_max":    float(np.max(arr)),
        f"{prefix}_median": float(np.median(arr)),
    }


def _pct(arr: np.ndarray, prefix: str) -> dict:
    p10, p25, p75, p90 = np.percentile(arr, [10, 25, 75, 90])
    return {
        f"{prefix}_p10": float(p10), f"{prefix}_p25": float(p25),
        f"{prefix}_p75": float(p75), f"{prefix}_p90": float(p90),
        f"{prefix}_iqr": float(p75 - p25),
    }


def _moments(arr: np.ndarray, prefix: str) -> dict:
    s = skew(arr, bias=False, nan_policy="omit")
    k = kurtosis(arr, bias=False, nan_policy="omit")
    s = 0.0 if not np.isfinite(s) else float(s)
    k = 0.0 if not np.isfinite(k) else float(k)
    return {f"{prefix}_skew": s, f"{prefix}_kurt": k}


def _fft_features(arr: np.ndarray, prefix: str, n_bands: int = 5) -> dict:
    """Energy in n_bands log-spaced freq bins + spectral entropy + dom freq."""
    arr = arr - np.mean(arr)
    spec = np.abs(np.fft.rfft(arr))[1:]  # drop DC
    if spec.size == 0 or spec.sum() == 0:
        d = {f"{prefix}_fft_b{i}": 0.0 for i in range(n_bands)}
        d[f"{prefix}_fft_entropy"] = 0.0
        d[f"{prefix}_fft_dom"] = 0.0
        return d
    bands = np.array_split(spec, n_bands)
    total = spec.sum()
    out = {f"{prefix}_fft_b{i}": float(b.sum() / total) for i, b in enumerate(bands)}
    p = spec / total
    out[f"{prefix}_fft_entropy"] = float(-np.sum(p * np.log(p + 1e-12)))
    out[f"{prefix}_fft_dom"]     = float(np.argmax(spec))  # bin index of dominant freq
    return out


def _mean_crossings(arr: np.ndarray) -> int:
    a = arr - np.mean(arr)
    return int(np.sum(np.diff(np.sign(a)) != 0))


def _autocorr_feats(arr: np.ndarray, prefix: str, lags=(1, 2, 5, 10, 20)) -> dict:
    """Autocorrelation at multiple lags — key for detecting step periodicity."""
    n = len(arr)
    a = arr - arr.mean()
    std = a.std()
    denom = (std ** 2 * n) if std > 1e-9 else 1.0
    out: dict = {}
    for lag in lags:
        if lag >= n:
            out[f"{prefix}_ac{lag}"] = 0.0
        else:
            c = float(np.dot(a[lag:], a[:-lag]) / denom)
            out[f"{prefix}_ac{lag}"] = float(np.clip(c, -1.0, 1.0))
    return out


def _segment_feats(arr: np.ndarray, prefix: str, n_segs: int = 3) -> dict:
    """Mean, std, range within each temporal third — captures activity evolution."""
    segs = np.array_split(arr, n_segs)
    out: dict = {}
    for i, seg in enumerate(segs):
        out[f"{prefix}_seg{i}_mean"] = float(np.mean(seg))
        out[f"{prefix}_seg{i}_std"]  = float(np.std(seg))
        out[f"{prefix}_seg{i}_rng"]  = float(np.ptp(seg))
    return out


def _peak_feats(arr: np.ndarray, prefix: str) -> dict:
    """Count of local maxima above adaptive threshold + average peak height.
    I think this should help distinguish walking/running (lots of rhythmic peaks)
    from sitting/lying (few or no peaks)."""
    from scipy.signal import find_peaks
    thr = float(arr.mean() + 0.5 * arr.std())
    try:
        peaks, props = find_peaks(arr, height=thr, distance=5)
        n = len(peaks)
        h = float(np.mean(props["peak_heights"])) if n > 0 else 0.0
        # also peak-to-peak distance mean (period proxy)
        if n >= 2:
            d = float(np.mean(np.diff(peaks)))
        else:
            d = 0.0
    except Exception:
        n, h, d = 0, 0.0, 0.0
    return {
        f"{prefix}_npeaks": float(n),
        f"{prefix}_peakh_mean": h,
        f"{prefix}_peak_period": d,
    }


def _energy_feats(arr: np.ndarray, prefix: str) -> dict:
    """Total signal energy + Shannon entropy of histogram (10 bins).
    Energy separates active from passive; entropy separates regular from chaotic."""
    energy = float(np.sum(arr ** 2) / max(len(arr), 1))
    hist, _ = np.histogram(arr, bins=10, density=False)
    p = hist.astype(np.float64)
    p = p / (p.sum() + 1e-12)
    ent = float(-np.sum(p * np.log(p + 1e-12)))
    rms = float(np.sqrt(np.mean(arr ** 2)))
    return {
        f"{prefix}_energy": energy,
        f"{prefix}_hist_ent": ent,
        f"{prefix}_rms": rms,
    }


def _ratio_feats(mx, my, mz) -> dict:
    """Cross-axis ratios — orientation-invariant clues about dominant axis."""
    vx, vy, vz = float(np.var(mx)), float(np.var(my)), float(np.var(mz))
    rx, ry, rz = float(np.ptp(mx)), float(np.ptp(my)), float(np.ptp(mz))
    ex = float(np.sum(mx ** 2)); ey = float(np.sum(my ** 2)); ez = float(np.sum(mz ** 2))
    eps = 1e-9
    return {
        "ratio__var_xy": vx / (vy + eps),
        "ratio__var_xz": vx / (vz + eps),
        "ratio__var_yz": vy / (vz + eps),
        "ratio__rng_xy": rx / (ry + eps),
        "ratio__rng_xz": rx / (rz + eps),
        "ratio__rng_yz": ry / (rz + eps),
        "ratio__e_xy": ex / (ey + eps),
        "ratio__e_xz": ex / (ez + eps),
        "ratio__e_yz": ey / (ez + eps),
        "ratio__dominant_axis": float(np.argmax([vx, vy, vz])),
    }


def _tail_feats(arr: np.ndarray, prefix: str) -> dict:
    """Extreme percentiles — tails of distribution."""
    p5, p95, p99 = np.percentile(arr, [5, 95, 99])
    return {
        f"{prefix}_p5":  float(p5),
        f"{prefix}_p95": float(p95),
        f"{prefix}_p99": float(p99),
        f"{prefix}_p95_minus_p5": float(p95 - p5),
    }


# ------------------------- per-file aggregator ------------------------- #
def _file_features(df: pd.DataFrame) -> dict:
    df = df.sort_values("index")
    feats: dict = {}

    # basic / pct / moments / fft / crossings  on every column
    for col in FEATURE_COLS:
        arr = df[col].to_numpy(dtype=np.float64)
        feats.update(_basic(arr, f"basic__{col}"))
        feats.update(_pct(arr,   f"pct__{col}"))
        feats.update(_moments(arr, f"moments__{col}"))
        feats.update(_fft_features(arr, f"fft__{col}"))
        feats[f"crossings__{col}"] = _mean_crossings(arr)

    # magnitude features (group: mag)
    mx, my, mz = df["mean_x"].to_numpy(), df["mean_y"].to_numpy(), df["mean_z"].to_numpy()
    mag = np.sqrt(mx ** 2 + my ** 2 + mz ** 2)
    feats.update(_basic(mag, "mag__m"))
    p10, p90 = np.percentile(mag, [10, 90])
    feats["mag__m_p10"] = float(p10); feats["mag__m_p90"] = float(p90)

    # SMA
    feats["sma__total"] = float(np.mean(np.abs(mx) + np.abs(my) + np.abs(mz)))
    feats["sma__x"] = float(np.mean(np.abs(mx)))
    feats["sma__y"] = float(np.mean(np.abs(my)))
    feats["sma__z"] = float(np.mean(np.abs(mz)))

    # axis correlations
    def _corr(a, b):
        if np.std(a) < 1e-9 or np.std(b) < 1e-9:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])
    feats["corr__xy"] = _corr(mx, my)
    feats["corr__xz"] = _corr(mx, mz)
    feats["corr__yz"] = _corr(my, mz)

    # jerk (first difference of mean_xyz)
    for col, arr in zip(MEAN_COLS, [mx, my, mz]):
        j = np.diff(arr)
        feats[f"jerk__{col}_meanabs"] = float(np.mean(np.abs(j))) if j.size else 0.0
        feats[f"jerk__{col}_std"]     = float(np.std(j)) if j.size else 0.0
        feats[f"jerk__{col}_maxabs"]  = float(np.max(np.abs(j))) if j.size else 0.0

    # jerk magnitude (3-axis combined)
    jx, jy, jz = np.diff(mx), np.diff(my), np.diff(mz)
    jmag = np.sqrt(jx**2 + jy**2 + jz**2)
    feats["jerk__mag_meanabs"] = float(np.mean(jmag)) if jmag.size else 0.0
    feats["jerk__mag_std"]     = float(np.std(jmag)) if jmag.size else 0.0
    feats["jerk__mag_maxabs"]  = float(np.max(jmag)) if jmag.size else 0.0

    # std summary (within-second variance summary)
    for col in STD_COLS:
        arr = df[col].to_numpy(dtype=np.float64)
        feats[f"std_sum__{col}_mean"] = float(np.mean(arr))
        feats[f"std_sum__{col}_max"]  = float(np.max(arr))
        feats[f"std_sum__{col}_p90"]  = float(np.percentile(arr, 90))

    # --- Autocorrelation features (periodicity / step cadence) ---
    for col, arr_ac in zip(MEAN_COLS, [mx, my, mz]):
        feats.update(_autocorr_feats(arr_ac, f"autocorr__{col}"))
    feats.update(_autocorr_feats(mag, "autocorr__mag"))

    # --- Temporal segment features (how activity evolves across time) ---
    for col in FEATURE_COLS:
        arr_s = df[col].to_numpy(dtype=np.float64)
        feats.update(_segment_feats(arr_s, f"seg__{col}"))

    # --- Peak features (rhythmic activity indicator) ---
    for col, arr_p in zip(MEAN_COLS, [mx, my, mz]):
        feats.update(_peak_feats(arr_p, f"peaks__{col}"))
    feats.update(_peak_feats(mag, "peaks__mag"))

    # --- Energy features (active vs. passive separator) ---
    for col in FEATURE_COLS:
        arr_e = df[col].to_numpy(dtype=np.float64)
        feats.update(_energy_feats(arr_e, f"energy__{col}"))
    feats.update(_energy_feats(mag, "energy__mag"))

    # --- Cross-axis ratios ---
    feats.update(_ratio_feats(mx, my, mz))

    # --- Extreme percentiles (tail features) ---
    for col in MEAN_COLS:
        arr_t = df[col].to_numpy(dtype=np.float64)
        feats.update(_tail_feats(arr_t, f"tails__{col}"))
    feats.update(_tail_feats(mag, "tails__mag"))

    # --- Std-channel temporal features (added in v4) ---
    # The std_x/y/z columns are the WITHIN-second motion intensity. Their
    # time-course carries the strongest activity cue (how vigorous + how
    # rhythmic the motion is), but until now I only took simple summaries of
    # them. Running the same autocorr / peak / tail extractors I used on the
    # mean channels over the std channels gave the biggest single feature
    # bump I found late in the project (+~0.004 LGBM OOF). Autocorrelation of
    # the std signal in particular captures step/stride cadence.
    sx = df["std_x"].to_numpy(dtype=np.float64)
    sy = df["std_y"].to_numpy(dtype=np.float64)
    sz = df["std_z"].to_numpy(dtype=np.float64)
    for col, arr_sd in zip(STD_COLS, [sx, sy, sz]):
        feats.update(_autocorr_feats(arr_sd, f"autocorr__{col}"))
        feats.update(_peak_feats(arr_sd,     f"peaks__{col}"))
        feats.update(_tail_feats(arr_sd,     f"tails__{col}"))

    # Coefficient of variation (motion consistency) + mean/std coupling per axis.
    for ax, m_arr, s_arr in zip(["x", "y", "z"], [mx, my, mz], [sx, sy, sz]):
        feats[f"covmix__cov_{ax}"] = float(s_arr.mean() / (np.abs(m_arr).mean() + 1e-9))
        feats[f"covmix__mscorr_{ax}"] = _corr(m_arr, s_arr)
    feats["covmix__cov_mag"] = float(mag.std() / (np.abs(mag).mean() + 1e-9))
    feats["covmix__std_cc_xy"] = _corr(sx, sy)
    feats["covmix__std_cc_xz"] = _corr(sx, sz)
    feats["covmix__std_cc_yz"] = _corr(sy, sz)

    return feats


# ------------------------- group mapping helper ------------------------- #
def group_of(col_name: str) -> str:
    """Map a feature column back to its group via the prefix before '__'."""
    head = col_name.split("__", 1)[0]
    # fft features look like fft__mean_x_fft_b0 -> head is 'fft' which is what we want
    return head


def build_features(long_df: pd.DataFrame, cache_name: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Returns (feature_df_with_file_id_user_id_label, {group -> [cols]})."""
    cache_path = (CACHE_DIR / f"{cache_name}.parquet") if cache_name else None
    if cache_path is not None and cache_path.exists():
        feats = pd.read_parquet(cache_path)
    else:
        rows = []
        for file_id, sub in long_df.groupby("file_id", sort=False):
            row = _file_features(sub)
            row["file_id"] = file_id
            row["user_id"] = sub["user_id"].iloc[0]
            if "label" in sub.columns and sub["label"].iloc[0] != -1:
                row["label"] = int(sub["label"].iloc[0])
            rows.append(row)
        feats = pd.DataFrame(rows)
        if cache_path is not None:
            feats.to_parquet(cache_path, index=False)

    # group -> [cols]
    meta_cols = {"file_id", "user_id", "label"}
    feat_cols = [c for c in feats.columns if c not in meta_cols]
    groups: dict[str, list[str]] = {}
    for c in feat_cols:
        groups.setdefault(group_of(c), []).append(c)
    return feats, groups


if __name__ == "__main__":
    from data_loader import load_train, load_test
    tr = load_train()
    te = load_test()
    f_tr, g = build_features(tr, cache_name="feats_train_v4")
    f_te, _ = build_features(te, cache_name="feats_test_v4")
    print(f"train features: {f_tr.shape}  test features: {f_te.shape}")
    print("groups:")
    for k, v in g.items():
        print(f"  {k:10s} -> {len(v)} cols")
