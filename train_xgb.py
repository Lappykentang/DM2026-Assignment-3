# Third base model: XGBoost. I added this because LGBM alone wasn't enough -
# tried bumping the ensemble higher and a 2nd boosted tree with different
# splits/leafing usually adds a bit of diversity.
#
# Outputs:
#   outputs/submissions/submission_xgb.csv
#   outputs/cache/xgb_oof.npz   (oof_proba, test_proba, y, ids_train, ids_test, groups)
#
# Run: python train_xgb.py
from __future__ import annotations
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import GroupKFold

from config import SEED, SUB_DIR, CACHE_DIR, SAMPLE_SUB, N_CLASSES
from data_loader import load_train, load_test
from feature_engineering import build_features


# I copied LGBM's idea (depth-ish leaves) but used XGBoost defaults more.
# Tried max_depth=6 first then bumped to 8 - 8 seemed to help slightly.
XGB_PARAMS = dict(
    objective="multi:softprob",
    num_class=N_CLASSES,
    eval_metric="mlogloss",
    learning_rate=0.05,
    max_depth=8,
    min_child_weight=3,
    subsample=0.85,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=1.0,
    tree_method="hist",
    verbosity=0,
    seed=SEED,
    nthread=-1,
)
N_BOOST = 3000
EARLY_STOP = 100
N_SPLITS = 5


def _train_one(train_x, train_y, valid_x, valid_y, w_tr=None):
    dtrain = xgb.DMatrix(train_x, label=train_y, weight=w_tr)
    dvalid = xgb.DMatrix(valid_x, label=valid_y)
    bst = xgb.train(
        XGB_PARAMS, dtrain, num_boost_round=N_BOOST,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=EARLY_STOP, verbose_eval=False,
    )
    return bst


def cv_xgb(train_x, train_y, train_groups, sample_w=None):
    splitter = GroupKFold(n_splits=N_SPLITS)
    oof_proba = np.zeros((len(train_y), N_CLASSES), dtype=np.float32)
    fold_scores = []
    models = []
    for k, (tr_idx, va_idx) in enumerate(splitter.split(train_x, train_y, train_groups)):
        w_tr = sample_w[tr_idx] if sample_w is not None else None
        bst = _train_one(train_x[tr_idx], train_y[tr_idx],
                         train_x[va_idx], train_y[va_idx], w_tr)
        proba = bst.predict(xgb.DMatrix(train_x[va_idx]),
                            iteration_range=(0, bst.best_iteration + 1))
        oof_proba[va_idx] = proba
        pred = proba.argmax(axis=1)
        fscore = f1_score(train_y[va_idx], pred, average="macro")
        fold_scores.append(fscore)
        models.append(bst)
        print(f"  fold {k+1}: best_iter={bst.best_iteration:4d}  macro-F1={fscore:.4f}")
    mean_f, std_f = float(np.mean(fold_scores)), float(np.std(fold_scores))
    print(f"  CV macro-F1 = {mean_f:.4f} +/-{std_f:.4f}")
    return mean_f, std_f, oof_proba, models


def predict_test(models, test_x):
    out = np.zeros((test_x.shape[0], N_CLASSES), dtype=np.float32)
    dtest = xgb.DMatrix(test_x)
    for m in models:
        out += m.predict(dtest, iteration_range=(0, m.best_iteration + 1))
    return out / len(models)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default=str(SEED),
                    help="Comma-separated seeds for bagging, e.g. 7,13,42")
    args = ap.parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    print(f"Bagging seeds: {seeds}")

    print("Loading data ...")
    tr_long = load_train(); te_long = load_test()
    print("Building features (using v3 cache from feature_engineering.py) ...")
    f_tr, _ = build_features(tr_long, cache_name="feats_train_v4")
    f_te, _ = build_features(te_long, cache_name="feats_test_v4")

    feat_cols = [c for c in f_tr.columns if c not in {"file_id", "user_id", "label"}]
    train_x = f_tr[feat_cols].to_numpy(dtype=np.float32)
    train_y = f_tr["label"].to_numpy(dtype=int)
    train_groups = f_tr["user_id"].to_numpy()
    ids_tr = f_tr["file_id"].to_numpy()

    test_x = f_te[feat_cols].to_numpy(dtype=np.float32)
    ids_te = f_te["file_id"].to_numpy()
    print(f"features: {len(feat_cols)}   X_train {train_x.shape}  X_test {test_x.shape}")

    # class-balanced sample weights (same idea as LGBM)
    counts = np.bincount(train_y, minlength=N_CLASSES).astype(float)
    inv = 1.0 / np.clip(counts, 1, None)
    sample_w = (inv / inv.mean())[train_y].astype(np.float32)

    # Bag over seeds. With one seed this is the previous behavior.
    oof_seeds, test_seeds = [], []
    last_mean = None
    for s in seeds:
        XGB_PARAMS["seed"] = s
        print(f"\n=== XGBoost 5-fold CV (seed={s}) ===")
        mean_s, _, oof_s, models_s = cv_xgb(train_x, train_y, train_groups, sample_w)
        oof_seeds.append(oof_s)
        test_seeds.append(predict_test(models_s, test_x))
        last_mean = mean_s

    oof_proba  = np.mean(oof_seeds, axis=0).astype(np.float32)
    test_proba = np.mean(test_seeds, axis=0).astype(np.float32)
    f1_bag = f1_score(train_y, oof_proba.argmax(1), average="macro")
    mean_f = f1_bag
    print(f"\nXGB bagged ({len(seeds)} seeds) OOF macro-F1 = {f1_bag:.4f}")

    test_pred = test_proba.argmax(axis=1)

    sub_template = pd.read_csv(SAMPLE_SUB)
    pred_df = pd.DataFrame({"Id": ids_te.astype(sub_template["Id"].dtype), "Label": test_pred})
    sub = sub_template[["Id"]].merge(pred_df, on="Id", how="left")
    miss = sub["Label"].isna().sum()
    if miss:
        print(f"WARNING: {miss} ids missing - filling with majority class.")
        sub["Label"] = sub["Label"].fillna(pd.Series(train_y).mode().iloc[0])
    sub["Label"] = sub["Label"].astype(int)
    out_path = SUB_DIR / "submission_xgb.csv"
    sub.to_csv(out_path, index=False)
    print(f"Saved {out_path}")

    np.savez(CACHE_DIR / "xgb_oof.npz",
             oof_proba=oof_proba, test_proba=test_proba,
             y=train_y, ids_train=ids_tr, ids_test=ids_te, groups=train_groups)
    print("Saved cache/xgb_oof.npz")

    print("\n=== Per-class report on OOF ===")
    print(classification_report(train_y, oof_proba.argmax(axis=1), digits=4))

    print(f"\nXGB bagged OOF macro-F1: {mean_f:.4f}")


if __name__ == "__main__":
    main()
