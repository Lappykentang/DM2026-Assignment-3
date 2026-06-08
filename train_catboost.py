# Fourth base model: CatBoost. I added this after LightGBM + XGBoost + CNN
# alone stalled at ~0.77 on the leaderboard. CatBoost uses oblivious-tree
# splits which behave differently from LightGBM (leaf-wise) and XGBoost
# (level-wise), so it injects different errors into the ensemble.
#
# Outputs:
#   outputs/submissions/submission_cat.csv
#   outputs/cache/cat_oof.npz
#
# Run: python train_catboost.py
#      python train_catboost.py --seeds 7,13,42   # bagging across seeds
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import GroupKFold

from config import SEED, SUB_DIR, CACHE_DIR, SAMPLE_SUB, N_CLASSES
from data_loader import load_train, load_test
from feature_engineering import build_features


# depth=6 trains ~4x faster than depth=8 and loses very little on this dataset.
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
)
N_SPLITS = 5


def cv_catboost(train_x, train_y, train_groups, sample_w=None, seed=SEED):
    CAT_PARAMS["random_seed"] = seed
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof_proba = np.zeros((len(train_y), N_CLASSES), dtype=np.float32)
    fold_scores, models = [], []
    for k, (tr_idx, va_idx) in enumerate(gkf.split(train_x, train_y, train_groups)):
        w_tr = sample_w[tr_idx] if sample_w is not None else None
        train_pool = Pool(train_x[tr_idx], label=train_y[tr_idx], weight=w_tr)
        valid_pool = Pool(train_x[va_idx], label=train_y[va_idx])
        clf = CatBoostClassifier(**CAT_PARAMS)
        clf.fit(train_pool, eval_set=valid_pool)
        proba = clf.predict_proba(train_x[va_idx])
        oof_proba[va_idx] = proba
        fscore = f1_score(train_y[va_idx], proba.argmax(axis=1), average="macro")
        fold_scores.append(fscore); models.append(clf)
        print(f"  fold {k+1}: best_iter={clf.get_best_iteration():4d}  macro-F1={fscore:.4f}")
    mean_f = float(np.mean(fold_scores))
    print(f"  CV macro-F1 = {mean_f:.4f} +/-{np.std(fold_scores):.4f}")
    return mean_f, oof_proba, models


def predict_test(models, test_x):
    out = np.zeros((test_x.shape[0], N_CLASSES), dtype=np.float32)
    for m in models:
        out += m.predict_proba(test_x)
    return out / len(models)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default=str(SEED),
                    help="Comma-separated seeds for bagging, e.g. 7,13,42")
    args = ap.parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    print(f"Bagging seeds: {seeds}")

    print("Loading data ...")
    tr_long = load_train(); te_long = load_test()
    print("Building features (using v3 cache) ...")
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

    counts = np.bincount(train_y, minlength=N_CLASSES).astype(float)
    inv = 1.0 / np.clip(counts, 1, None)
    sample_w = (inv / inv.mean())[train_y].astype(np.float32)

    oof_seeds, test_seeds = [], []
    for s in seeds:
        print(f"\n=== CatBoost CV (seed={s}) ===")
        _, oof_s, models_s = cv_catboost(train_x, train_y, train_groups, sample_w, seed=s)
        oof_seeds.append(oof_s)
        test_seeds.append(predict_test(models_s, test_x))

    oof_proba  = np.mean(oof_seeds, axis=0).astype(np.float32)
    test_proba = np.mean(test_seeds, axis=0).astype(np.float32)
    f1_bag = f1_score(train_y, oof_proba.argmax(1), average="macro")
    print(f"\nCatBoost bagged ({len(seeds)} seeds) OOF macro-F1 = {f1_bag:.4f}")

    test_pred = test_proba.argmax(axis=1)
    sub_template = pd.read_csv(SAMPLE_SUB)
    pred_df = pd.DataFrame({"Id": ids_te.astype(sub_template["Id"].dtype), "Label": test_pred})
    sub = sub_template[["Id"]].merge(pred_df, on="Id", how="left")
    miss = sub["Label"].isna().sum()
    if miss:
        print(f"WARNING: {miss} ids missing - filling with majority class.")
        sub["Label"] = sub["Label"].fillna(pd.Series(train_y).mode().iloc[0])
    sub["Label"] = sub["Label"].astype(int)
    out_path = SUB_DIR / "submission_cat.csv"
    sub.to_csv(out_path, index=False)
    print(f"Saved {out_path}")

    np.savez(CACHE_DIR / "cat_oof.npz",
             oof_proba=oof_proba, test_proba=test_proba,
             y=train_y, ids_train=ids_tr, ids_test=ids_te, groups=train_groups)
    print("Saved cache/cat_oof.npz")

    print("\n=== Per-class report on OOF ===")
    print(classification_report(train_y, oof_proba.argmax(axis=1), digits=4))


if __name__ == "__main__":
    main()
