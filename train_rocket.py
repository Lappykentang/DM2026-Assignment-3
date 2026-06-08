# Sixth base model: ROCKET (RandOm Convolutional KErnel Transform).
#
# Why I added this: my five other models (LGBM, XGB, CatBoost, MLP, CNN) had
# basically stopped helping the blend - the three tree models all make the
# same mistakes because they read the same 288 tabular features, and the blend
# was leaning almost entirely on the CNN for "different" predictions. I needed
# another model that looks at the raw 6x300 signal in a totally different way.
#
# ROCKET does exactly that: it slides a few thousand RANDOM little filters over
# the raw signal and, for each filter, records two numbers - how often the
# filter output was positive (ppv) and the biggest output it ever produced
# (max). Those numbers become features for a plain logistic-regression. It
# sounds too simple to work but it's a well-known strong time-series method,
# and being random + linear it makes errors that don't line up with the trees
# or the CNN, which is what the ensemble was missing.
#
# Outputs:
#   outputs/submissions/submission_rocket.csv
#   outputs/cache/rocket_oof.npz
#
# Run: python train_rocket.py
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import GroupKFold

from config import SEED, SUB_DIR, CACHE_DIR, SAMPLE_SUB, N_CLASSES, SEQ_LEN, FEATURE_COLS
from data_loader import load_train, load_test, file_meta

# How many random filters and which seeds to bag. Two seeds = two independent
# random filter banks; averaging them smooths out the luck of the draw.
N_KERNELS = 4000
KERNEL_SEEDS = [42, 7]
N_SPLITS = 5


def make_signal_block(long_df, ids):
    # Turn the long table into a stack of (n_files, 6, 300) signal blocks,
    # padding/truncating to 300 rows so every file is the same length.
    by_file = {}
    for file_id, sub in long_df.groupby("file_id", sort=False):
        arr = sub.sort_values("index")[FEATURE_COLS].to_numpy(np.float32)
        if len(arr) < SEQ_LEN:
            arr = np.vstack([arr, np.zeros((SEQ_LEN - len(arr), 6), np.float32)])
        by_file[file_id] = arr[:SEQ_LEN]
    block = np.stack([by_file[f] for f in ids])      # (n, 300, 6)
    return block.transpose(0, 2, 1)                  # (n, 6, 300)


def rocket_features(signal, n_kernels, seed):
    # Apply n_kernels random dilated filters; return 2 features per kernel.
    rng = np.random.default_rng(seed)
    n_files, n_ch, length = signal.shape
    out = np.zeros((n_files, 2 * n_kernels), np.float32)
    klens = rng.choice([7, 9, 11], n_kernels)
    for k in range(n_kernels):
        klen = klens[k]
        weight = rng.standard_normal(klen).astype(np.float32)
        weight -= weight.mean()                      # zero-mean weights (ROCKET trick)
        bias = rng.uniform(-1, 1)
        max_dilation = int(np.log2((length - 1) / (klen - 1))) if length > klen else 0
        dilation = int(2 ** rng.integers(0, max_dilation + 1)) if max_dilation > 0 else 1
        # mix 1-2 random channels so the filter can see cross-axis patterns
        n_pick = rng.integers(1, 3)
        chans = rng.choice(n_ch, n_pick, replace=False)
        sig = signal[:, chans, :].sum(1)
        span = (klen - 1) * dilation
        if span >= length:
            dilation = 1
            span = klen - 1
        conv = np.zeros((n_files, length - span), np.float32)
        for j in range(klen):
            conv += weight[j] * sig[:, j * dilation: j * dilation + (length - span)]
        conv += bias
        out[:, 2 * k] = (conv > 0).mean(1)           # ppv  (proportion positive values)
        out[:, 2 * k + 1] = conv.max(1)              # max
    return out


def run_one_seed(feat_tr, feat_te, y, groups, seed):
    # 5-fold GroupKFold: OOF on train, averaged fold-models on test.
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros((len(y), N_CLASSES), np.float32)
    test_acc = np.zeros((feat_te.shape[0], N_CLASSES), np.float32)
    for k, (tr, va) in enumerate(gkf.split(feat_tr, y, groups)):
        scaler = StandardScaler().fit(feat_tr[tr])
        clf = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced")
        clf.fit(scaler.transform(feat_tr[tr]), y[tr])
        oof[va] = clf.predict_proba(scaler.transform(feat_tr[va]))
        test_acc += clf.predict_proba(scaler.transform(feat_te))
        f = f1_score(y[va], oof[va].argmax(1), average="macro")
        print(f"  fold {k+1}: macro-F1={f:.4f}")
    return oof, test_acc / N_SPLITS


def main():
    print("Loading data ...")
    tr_long = load_train()
    te_long = load_test()
    tr_meta = file_meta(tr_long)
    te_meta = file_meta(te_long)

    ids_tr = tr_meta["file_id"].to_numpy()
    ids_te = te_meta["file_id"].to_numpy()
    y = tr_meta["label"].to_numpy(int)
    groups = tr_meta["user_id"].to_numpy()

    print("Building raw signal blocks ...")
    sig_tr = make_signal_block(tr_long, ids_tr)
    sig_te = make_signal_block(te_long, ids_te)

    # Normalise each channel using TRAIN statistics only.
    mu = sig_tr.mean((0, 2), keepdims=True)
    sd = sig_tr.std((0, 2), keepdims=True) + 1e-6
    sig_tr = (sig_tr - mu) / sd
    sig_te = (sig_te - mu) / sd
    print(f"  train signal {sig_tr.shape}   test signal {sig_te.shape}")

    oof_seeds, test_seeds = [], []
    for s in KERNEL_SEEDS:
        print(f"\n=== ROCKET seed={s}  ({N_KERNELS} kernels) ===")
        print("  transforming (this is the slow part) ...")
        feat_tr = rocket_features(sig_tr, N_KERNELS, s)
        feat_te = rocket_features(sig_te, N_KERNELS, s)
        oof_s, test_s = run_one_seed(feat_tr, feat_te, y, groups, s)
        oof_seeds.append(oof_s)
        test_seeds.append(test_s)
        print(f"  seed={s} OOF macro-F1: {f1_score(y, oof_s.argmax(1), average='macro'):.4f}")

    oof_proba = np.mean(oof_seeds, axis=0).astype(np.float32)
    test_proba = np.mean(test_seeds, axis=0).astype(np.float32)
    f1_bag = f1_score(y, oof_proba.argmax(1), average="macro")
    print(f"\nROCKET bagged ({len(KERNEL_SEEDS)} seeds) OOF macro-F1 = {f1_bag:.4f}")

    test_pred = test_proba.argmax(1)
    sub_template = pd.read_csv(SAMPLE_SUB)
    pred_df = pd.DataFrame({"Id": ids_te.astype(sub_template["Id"].dtype), "Label": test_pred})
    sub = sub_template[["Id"]].merge(pred_df, on="Id", how="left")
    miss = sub["Label"].isna().sum()
    if miss:
        print(f"WARNING: {miss} ids missing - filling with majority class.")
        sub["Label"] = sub["Label"].fillna(pd.Series(y).mode().iloc[0])
    sub["Label"] = sub["Label"].astype(int)
    out_path = SUB_DIR / "submission_rocket.csv"
    sub.to_csv(out_path, index=False)
    print(f"Saved {out_path}")

    np.savez(CACHE_DIR / "rocket_oof.npz",
             oof_proba=oof_proba, test_proba=test_proba,
             y=y, ids_train=ids_tr, ids_test=ids_te, groups=groups)
    print("Saved cache/rocket_oof.npz")

    print("\n=== Per-class report on OOF ===")
    print(classification_report(y, oof_proba.argmax(1), digits=4))


if __name__ == "__main__":
    main()
