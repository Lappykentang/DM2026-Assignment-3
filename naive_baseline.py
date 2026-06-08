"""Naive baseline for Prompt 1.

Pipeline:
  1. For each file, aggregate the 300 rows to ONE row of simple summary stats.
     - mean over time of (mean_x, mean_y, mean_z, std_x, std_y, std_z)
     - std  over time of the same 6 columns
     => 12 features per file.
  2. Train Logistic Regression and Random Forest with GroupKFold by user.
  3. Report Macro F1 (CV mean over folds).
  4. Refit the best model on all data and predict the test set.
  5. Write submission_naive.csv matching sample_submission.csv exactly.

Run: python naive_baseline.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import SEED, FEATURE_COLS, SUB_DIR, CACHE_DIR, SAMPLE_SUB
from data_loader import load_train, load_test, file_meta


def aggregate_naive(long_df: pd.DataFrame) -> pd.DataFrame:
    """One row per file: 12 features = mean & std over time of each of 6 columns."""
    g = long_df.groupby("file_id")
    agg_mean = g[FEATURE_COLS].mean().add_suffix("__tmean")
    agg_std  = g[FEATURE_COLS].std().add_suffix("__tstd").fillna(0.0)
    feats = pd.concat([agg_mean, agg_std], axis=1).reset_index()
    return feats


def build_xy(long_df: pd.DataFrame, with_label: bool):
    feats = aggregate_naive(long_df)
    meta = file_meta(long_df)
    df = feats.merge(meta, on="file_id")
    feat_cols = [c for c in df.columns if c.endswith(("__tmean", "__tstd"))]
    X = df[feat_cols].values.astype(np.float32)
    groups = df["user_id"].values
    y = df["label"].values if with_label else None
    return X, y, groups, df["file_id"].values, feat_cols


def cv_score(model, X, y, groups, n_splits=5, model_name=""):
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


def main():
    print("Loading data ...")
    tr_long = load_train()
    te_long = load_test()

    print("Aggregating features ...")
    X_tr, y_tr, groups_tr, ids_tr, feat_cols = build_xy(tr_long, with_label=True)
    X_te, _,    _,         ids_te, _         = build_xy(te_long, with_label=False)
    print(f"X_train: {X_tr.shape}  X_test: {X_te.shape}  features: {len(feat_cols)}")

    logreg = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, C=1.0,
                                   random_state=SEED, n_jobs=-1)),
    ])
    rf = RandomForestClassifier(n_estimators=400, max_depth=None,
                                min_samples_leaf=2, n_jobs=-1, random_state=SEED)

    print("\n--- Logistic Regression (5-fold GroupKFold by user) ---")
    f1_lr, oof_lr = cv_score(logreg, X_tr, y_tr, groups_tr, model_name="logreg")
    print("\n--- Random Forest (5-fold GroupKFold by user) ---")
    f1_rf, oof_rf = cv_score(rf,     X_tr, y_tr, groups_tr, model_name="rf")

    best_name, best_f1 = ("logreg", f1_lr) if f1_lr >= f1_rf else ("rf", f1_rf)
    print(f"\nBest naive model: {best_name}  (CV macro-F1 = {best_f1:.4f})")

    # Refit best on full train, predict test
    final = logreg if best_name == "logreg" else rf
    final.fit(X_tr, y_tr)
    preds = final.predict(X_te)

    # Submission must align with sample_submission Ids
    sub_template = pd.read_csv(SAMPLE_SUB)
    pred_df = pd.DataFrame({"Id": ids_te.astype(sub_template["Id"].dtype), "Label": preds})
    sub = sub_template[["Id"]].merge(pred_df, on="Id", how="left")
    missing = sub["Label"].isna().sum()
    if missing:
        print(f"WARNING: {missing} ids missing prediction — filling with majority class.")
        sub["Label"] = sub["Label"].fillna(pd.Series(y_tr).mode().iloc[0])
    sub["Label"] = sub["Label"].astype(int)

    out_path = SUB_DIR / "submission_naive.csv"
    sub.to_csv(out_path, index=False)
    print(f"\nSaved submission to {out_path}")
    print(sub.head())

    # Save OOF for later ensembling, and a small summary
    np.savez(CACHE_DIR / "naive_oof.npz",
             oof_lr=oof_lr, oof_rf=oof_rf, y=y_tr, ids=ids_tr, groups=groups_tr)
    print("OOF arrays cached.")

    print("\n=== Per-class report (best OOF) ===")
    best_oof = oof_lr if best_name == "logreg" else oof_rf
    print(classification_report(y_tr, best_oof, digits=4))

    print("\n=== SUMMARY ===")
    print(f"LogReg CV macro-F1: {f1_lr:.4f}")
    print(f"RF     CV macro-F1: {f1_rf:.4f}")
    print(f"Best:  {best_name} -> submission_naive.csv ready for Kaggle upload")


if __name__ == "__main__":
    main()
