# LightGBM trainer. Does 5-fold CV (GroupKFold by user so different users go in
# different folds), then trains a leave-one-feature-group-out ablation.
# I tried random forest first but LightGBM was clearly better (~0.65 vs ~0.69).
#
# Run: python train_lgbm.py
#      python train_lgbm.py --redo-ablation   # if you want to rerun ablation
from __future__ import annotations
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import GroupKFold

from config import SEED, SUB_DIR, CACHE_DIR, SAMPLE_SUB, N_CLASSES
from data_loader import load_train, load_test
from feature_engineering import build_features


# I started with lr=0.05 num_leaves=63 (defaults-ish) and CV was ~0.66.
# Lowering lr to 0.03 + raising num_leaves to 127 pushed it to ~0.69.
LGB_PARAMS = dict(
    objective="multiclass",
    num_class=N_CLASSES,
    metric="multi_logloss",
    learning_rate=0.03,
    num_leaves=127,
    min_child_samples=15,
    feature_fraction=0.85,
    bagging_fraction=0.85,
    bagging_freq=1,
    reg_alpha=0.0,
    reg_lambda=0.1,
    verbose=-1,
    seed=SEED,
    num_threads=-1,
)
N_BOOST = 4000
EARLY_STOP = 150
N_SPLITS = 5


def _train_one_fold(X_tr, y_tr, X_va, y_va, feat_cols, w_tr=None):
    dtr = lgb.Dataset(X_tr, label=y_tr, feature_name=feat_cols, weight=w_tr)
    dva = lgb.Dataset(X_va, label=y_va, feature_name=feat_cols, reference=dtr)
    model = lgb.train(
        LGB_PARAMS, dtr, num_boost_round=N_BOOST, valid_sets=[dtr, dva],
        valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                   lgb.log_evaluation(0)],
    )
    return model


def cv_lgbm(X, y, groups, feat_cols, sample_weights=None, verbose=True):
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof_proba = np.zeros((len(y), N_CLASSES), dtype=np.float32)
    fold_f1, models = [], []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        w_tr = sample_weights[tr_idx] if sample_weights is not None else None
        model = _train_one_fold(X[tr_idx], y[tr_idx], X[va_idx], y[va_idx], feat_cols, w_tr)
        proba = model.predict(X[va_idx], num_iteration=model.best_iteration)
        oof_proba[va_idx] = proba
        pred = proba.argmax(axis=1)
        f1 = f1_score(y[va_idx], pred, average="macro")
        fold_f1.append(f1); models.append(model)
        if verbose:
            print(f"  fold {fold+1}: best_iter={model.best_iteration:4d}  macro-F1={f1:.4f}")
    mean = float(np.mean(fold_f1)); std = float(np.std(fold_f1))
    if verbose:
        print(f"  CV macro-F1 = {mean:.4f} +/-{std:.4f}")
    return mean, std, oof_proba, models


def predict_test(models, X_te):
    probs = np.zeros((X_te.shape[0], N_CLASSES), dtype=np.float32)
    for m in models:
        probs += m.predict(X_te, num_iteration=m.best_iteration)
    return probs / len(models)


def ablation(X, y, groups, feat_cols, group_map: dict[str, list[str]], baseline_f1: float,
             sample_weights=None):
    """Leave-one-group-out + only-group. Returns dataframe."""
    col_to_idx = {c: i for i, c in enumerate(feat_cols)}
    rows = []
    for g, cols in sorted(group_map.items()):
        keep_idx_minus = [i for c, i in col_to_idx.items() if group_of(c) != g]
        keep_cols_minus = [feat_cols[i] for i in keep_idx_minus]
        f1_minus, _, _, _ = cv_lgbm(X[:, keep_idx_minus], y, groups, keep_cols_minus,
                                     sample_weights=sample_weights, verbose=False)
        keep_idx_only = [col_to_idx[c] for c in cols]
        f1_only, _, _, _ = cv_lgbm(X[:, keep_idx_only], y, groups, cols,
                                    sample_weights=sample_weights, verbose=False)
        delta = baseline_f1 - f1_minus
        print(f"  [{g:10s}] only={f1_only:.4f}  all-minus={f1_minus:.4f}  delta={delta:+.4f}  ({len(cols)} feats)")
        rows.append(dict(group=g, n_features=len(cols), only_f1=f1_only,
                         all_minus_f1=f1_minus, drop_when_removed=delta))
    return pd.DataFrame(rows).sort_values("drop_when_removed", ascending=False)


def group_of(col_name: str) -> str:
    return col_name.split("__", 1)[0]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default=str(SEED),
                    help="Comma-separated list of seeds for bagging (e.g. 7,13,42). "
                         "Default uses just the config SEED.")
    args = ap.parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    print(f"Bagging seeds: {seeds}")

    print("Loading data ...")
    tr_long = load_train(); te_long = load_test()
    print("Building features (cached after first run) ...")
    f_tr, group_map = build_features(tr_long, cache_name="feats_train_v4")
    f_te, _         = build_features(te_long, cache_name="feats_test_v4")

    feat_cols = [c for c in f_tr.columns if c not in {"file_id", "user_id", "label"}]
    X_tr = f_tr[feat_cols].to_numpy(dtype=np.float32)
    y_tr = f_tr["label"].to_numpy(dtype=int)
    groups_tr = f_tr["user_id"].to_numpy()
    ids_tr = f_tr["file_id"].to_numpy()

    X_te = f_te[feat_cols].to_numpy(dtype=np.float32)
    ids_te = f_te["file_id"].to_numpy()

    print(f"\nfeatures: {len(feat_cols)}   groups: {len(group_map)}")
    print(f"X_train {X_tr.shape}  X_test {X_te.shape}")

    # Class-balanced sample weights
    counts = np.bincount(y_tr, minlength=N_CLASSES).astype(float)
    inv = 1.0 / np.clip(counts, 1, None)
    sample_weights = (inv / inv.mean())[y_tr].astype(np.float32)
    print(f"Sample weights: min={sample_weights.min():.3f}  max={sample_weights.max():.3f}")

    # Bag over seeds. When only one seed, this is equivalent to the old behavior.
    oof_seeds, test_seeds = [], []
    models = None
    last_f1, last_std = None, None
    for s in seeds:
        LGB_PARAMS["seed"] = s
        print(f"\n=== Full-feature CV (seed={s}) ===")
        f1_s, std_s, oof_s, models_s = cv_lgbm(X_tr, y_tr, groups_tr, feat_cols,
                                                sample_weights=sample_weights)
        print(f"LGBM seed={s} CV macro-F1 = {f1_s:.4f} +/-{std_s:.4f}")
        oof_seeds.append(oof_s)
        # Compute test probs for this seed
        test_s = predict_test(models_s, X_te)
        test_seeds.append(test_s)
        models = models_s  # keep the last set for feature importance
        last_f1, last_std = f1_s, std_s

    # Average OOF and test across seeds
    oof_proba  = np.mean(oof_seeds, axis=0).astype(np.float32)
    test_proba = np.mean(test_seeds, axis=0).astype(np.float32)
    f1_bag = f1_score(y_tr, oof_proba.argmax(1), average="macro")
    f1_all, f1_std = f1_bag, last_std  # bag is the headline number now
    print(f"\nLGBM bagged ({len(seeds)} seeds) OOF macro-F1 = {f1_bag:.4f}")

    print("\n=== Per-group feature importance (sum gain across folds, last seed) ===")
    imp = np.zeros(len(feat_cols), dtype=np.float64)
    for m in models:
        imp += m.feature_importance(importance_type="gain")
    imp_df = pd.DataFrame({"feature": feat_cols, "gain": imp,
                           "group": [group_of(c) for c in feat_cols]})
    imp_df.sort_values("gain", ascending=False, inplace=True)
    imp_df.to_csv(CACHE_DIR / "lgbm_feature_importance.csv", index=False)
    by_group = imp_df.groupby("group")["gain"].sum().sort_values(ascending=False)
    print(by_group.to_string())

    # Ablation: re-use cache if exists, only rebuild when user passes --redo-ablation
    import sys as _sys
    abl_cache = CACHE_DIR / "lgbm_ablation.csv"
    if abl_cache.exists() and "--redo-ablation" not in _sys.argv:
        print(f"\n=== Ablation: using cached {abl_cache.name} (pass --redo-ablation to recompute) ===")
        abl = pd.read_csv(abl_cache)
    else:
        print("\n=== Ablation (5-fold CV per variant; this takes a while) ===")
        abl = ablation(X_tr, y_tr, groups_tr, feat_cols, group_map, f1_all,
                       sample_weights=sample_weights)
        abl.to_csv(abl_cache, index=False)
    print("\nAblation table (sorted by delta drop when removed):")
    print(abl.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== Predicting test (already averaged across seeds) ===")
    test_pred = test_proba.argmax(axis=1)

    sub_template = pd.read_csv(SAMPLE_SUB)
    pred_df = pd.DataFrame({"Id": ids_te.astype(sub_template["Id"].dtype), "Label": test_pred})
    sub = sub_template[["Id"]].merge(pred_df, on="Id", how="left")
    miss = sub["Label"].isna().sum()
    if miss:
        print(f"WARNING: {miss} ids missing — filling with majority class.")
        sub["Label"] = sub["Label"].fillna(pd.Series(y_tr).mode().iloc[0])
    sub["Label"] = sub["Label"].astype(int)
    out_path = SUB_DIR / "submission_lgbm.csv"
    sub.to_csv(out_path, index=False)
    print(f"Saved {out_path}")

    np.savez(CACHE_DIR / "lgbm_oof.npz",
             oof_proba=oof_proba, test_proba=test_proba,
             y=y_tr, ids_train=ids_tr, ids_test=ids_te, groups=groups_tr)
    print("Saved OOF + test probabilities to cache/lgbm_oof.npz")

    print("\n=== Per-class report on OOF ===")
    print(classification_report(y_tr, oof_proba.argmax(axis=1), digits=4))

    print("\n=== SUMMARY ===")
    print(f"LGBM full features  CV macro-F1: {f1_all:.4f} +/-{f1_std:.4f}")
    print(f"Top-3 most useful groups (by delta drop when removed):")
    print(abl.head(3).to_string(index=False, float_format=lambda x: f'{x:.4f}'))


if __name__ == "__main__":
    main()
