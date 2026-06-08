# Ensemble step. I started with a simple weighted blend (grid search on w),
# but after adding XGBoost as a 3rd model the search space got bigger so I
# switched to a logistic-regression stacker. I keep both methods here and
# pick whichever scores higher on OOF.
#
# Strategy:
#   1. Load OOF probs from cache for LGBM, CNN, (optional) XGB.
#   2. Align all three to the same file_id ordering.
#   3. Method A - logistic-regression stacker on OOF probs (18 features per row).
#      Evaluate via GroupKFold(user) so the meta-learner doesn't peek.
#   4. Method B - simple weighted average (grid over w_lgbm + w_cnn + w_xgb = 1).
#   5. Pick the better of A or B for the final submission.
#
# Run: python ensemble.py
from __future__ import annotations
import numpy as np
import pandas as pd
from itertools import product
from sklearn.metrics import f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

from config import CACHE_DIR, SUB_DIR, SAMPLE_SUB, N_CLASSES, SEED


def _load(name: str):
    p = CACHE_DIR / name
    if not p.exists():
        return None  # caller decides whether this is fatal
    # allow_pickle=True needed because user_id may be stored as object dtype
    return np.load(p, allow_pickle=True)


def _align(reference_ids, other_ids, other_arr):
    # Re-order other_arr so its rows match reference_ids
    pos = {fid: i for i, fid in enumerate(other_ids.tolist())}
    idx = np.array([pos[f] for f in reference_ids.tolist()], dtype=int)
    return other_arr[idx]


def stacker_oof_score(stack_tr, y, groups, C=1.0):
    # Score the LR stacker honestly via GroupKFold(user)
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros((len(y), N_CLASSES), dtype=np.float32)
    for tr_idx, va_idx in gkf.split(stack_tr, y, groups):
        sc = StandardScaler().fit(stack_tr[tr_idx])
        clf = LogisticRegression(C=C, solver="lbfgs", max_iter=2000,
                                 random_state=SEED)
        clf.fit(sc.transform(stack_tr[tr_idx]), y[tr_idx])
        oof[va_idx] = clf.predict_proba(sc.transform(stack_tr[va_idx]))
    return oof


def main():
    print("Loading OOF caches ...")
    lgbm   = _load("lgbm_oof.npz")
    cnn    = _load("cnn_oof.npz")
    xgb    = _load("xgb_oof.npz")     # may be None if not yet trained
    cat    = _load("cat_oof.npz")     # may be None if CatBoost not yet trained
    mlp    = _load("mlp_oof.npz")     # may be None if MLP not yet trained
    rocket = _load("rocket_oof.npz")  # may be None if ROCKET not yet trained

    if lgbm is None or cnn is None:
        raise FileNotFoundError("Need at least lgbm_oof.npz + cnn_oof.npz. Run train_lgbm.py and train_cnn.py first.")

    # Use LGBM ordering as the reference
    ref_ids_tr = lgbm["ids_train"]
    ref_ids_te = lgbm["ids_test"]
    y          = lgbm["y"]
    groups     = lgbm["groups"]

    p_lgbm_tr = lgbm["oof_proba"].astype(np.float32)
    p_lgbm_te = lgbm["test_proba"].astype(np.float32)
    p_cnn_tr  = _align(ref_ids_tr, cnn["ids_train"], cnn["oof_proba"].astype(np.float32))
    p_cnn_te  = _align(ref_ids_te, cnn["ids_test"],  cnn["test_proba"].astype(np.float32))

    has_xgb = xgb is not None
    if has_xgb:
        p_xgb_tr = _align(ref_ids_tr, xgb["ids_train"], xgb["oof_proba"].astype(np.float32))
        p_xgb_te = _align(ref_ids_te, xgb["ids_test"],  xgb["test_proba"].astype(np.float32))

    has_cat = cat is not None
    if has_cat:
        p_cat_tr = _align(ref_ids_tr, cat["ids_train"], cat["oof_proba"].astype(np.float32))
        p_cat_te = _align(ref_ids_te, cat["ids_test"],  cat["test_proba"].astype(np.float32))

    has_mlp = mlp is not None
    if has_mlp:
        p_mlp_tr = _align(ref_ids_tr, mlp["ids_train"], mlp["oof_proba"].astype(np.float32))
        p_mlp_te = _align(ref_ids_te, mlp["ids_test"],  mlp["test_proba"].astype(np.float32))

    has_rocket = rocket is not None
    if has_rocket:
        p_rocket_tr = _align(ref_ids_tr, rocket["ids_train"], rocket["oof_proba"].astype(np.float32))
        p_rocket_te = _align(ref_ids_te, rocket["ids_test"],  rocket["test_proba"].astype(np.float32))

    # Individual OOF scores
    f1_lgbm = f1_score(y, p_lgbm_tr.argmax(1), average="macro")
    f1_cnn  = f1_score(y, p_cnn_tr.argmax(1),  average="macro")
    print(f"  LGBM OOF macro-F1: {f1_lgbm:.4f}")
    print(f"  CNN  OOF macro-F1: {f1_cnn:.4f}")
    if has_xgb:
        f1_xgb = f1_score(y, p_xgb_tr.argmax(1), average="macro")
        print(f"  XGB  OOF macro-F1: {f1_xgb:.4f}")
    if has_cat:
        f1_cat = f1_score(y, p_cat_tr.argmax(1), average="macro")
        print(f"  CAT  OOF macro-F1: {f1_cat:.4f}")
    if has_mlp:
        f1_mlp = f1_score(y, p_mlp_tr.argmax(1), average="macro")
        print(f"  MLP  OOF macro-F1: {f1_mlp:.4f}")
    if has_rocket:
        f1_rocket = f1_score(y, p_rocket_tr.argmax(1), average="macro")
        print(f"  ROCKET OOF macro-F1: {f1_rocket:.4f}")

    # --------------- Method A: LR stacker ---------------
    stack_parts_tr = [p_lgbm_tr, p_cnn_tr]
    stack_parts_te = [p_lgbm_te, p_cnn_te]
    if has_xgb:
        stack_parts_tr.append(p_xgb_tr); stack_parts_te.append(p_xgb_te)
    if has_cat:
        stack_parts_tr.append(p_cat_tr); stack_parts_te.append(p_cat_te)
    if has_mlp:
        stack_parts_tr.append(p_mlp_tr); stack_parts_te.append(p_mlp_te)
    if has_rocket:
        stack_parts_tr.append(p_rocket_tr); stack_parts_te.append(p_rocket_te)
    stack_tr = np.hstack(stack_parts_tr)
    stack_te = np.hstack(stack_parts_te)

    print(f"\n--- Method A: logistic-regression stacker ({stack_tr.shape[1]} features) ---")
    best_C, best_stack_f1, best_stack_oof = None, -1.0, None
    for C in [0.1, 0.3, 1.0, 3.0, 10.0]:
        oof = stacker_oof_score(stack_tr, y, groups, C=C)
        f = f1_score(y, oof.argmax(1), average="macro")
        print(f"  C={C:>5}: OOF F1 = {f:.4f}")
        if f > best_stack_f1:
            best_C, best_stack_f1, best_stack_oof = C, f, oof
    print(f"  best stacker C={best_C}, OOF F1={best_stack_f1:.4f}")

    # Fit final stacker on full data, predict test
    scaler_final = StandardScaler().fit(stack_tr)
    final_lr = LogisticRegression(C=best_C, solver="lbfgs", max_iter=2000,
                                  random_state=SEED)
    final_lr.fit(scaler_final.transform(stack_tr), y)
    stack_te_proba = final_lr.predict_proba(scaler_final.transform(stack_te))

    # --------------- Method B: hill-climbing weighted ensemble ---------------
    # I switched from grid search to hill-climbing (Caruana et al. 2004) once I
    # had 6 models - a full weight grid blows up exponentially with the number
    # of models, but hill-climbing scales fine. Start from an empty blend and
    # repeatedly add (with replacement) whichever model most improves OOF
    # macro-F1. The number of times each model gets picked becomes its weight.
    print("\n--- Method B: hill-climbing weighted ensemble ---")
    blend_models = [("LGBM", p_lgbm_tr, p_lgbm_te), ("CNN", p_cnn_tr, p_cnn_te)]
    if has_xgb:
        blend_models.append(("XGB", p_xgb_tr, p_xgb_te))
    if has_cat:
        blend_models.append(("CAT", p_cat_tr, p_cat_te))
    if has_mlp:
        blend_models.append(("MLP", p_mlp_tr, p_mlp_te))
    if has_rocket:
        blend_models.append(("ROCKET", p_rocket_tr, p_rocket_te))

    HILL_STEPS = 25
    acc_tr = np.zeros_like(p_lgbm_tr)
    picks = []
    for _ in range(HILL_STEPS):
        best_i, best_f = None, -1.0
        for i, (_, ptr, _te) in enumerate(blend_models):
            f = f1_score(y, (acc_tr + ptr).argmax(1), average="macro")
            if f > best_f:
                best_f, best_i = f, i
        acc_tr = acc_tr + blend_models[best_i][1]
        picks.append(best_i)
    counts = {i: picks.count(i) for i in range(len(blend_models))}
    blend_oof = acc_tr / HILL_STEPS
    blend_te_proba = sum(counts.get(i, 0) * blend_models[i][2]
                         for i in range(len(blend_models))) / HILL_STEPS
    best_blend_f1 = f1_score(y, blend_oof.argmax(1), average="macro")
    weight_str = "  ".join(f"{blend_models[i][0]}={counts.get(i,0)/HILL_STEPS:.2f}"
                           for i in range(len(blend_models)))
    print(f"  weights: {weight_str}")
    print(f"  OOF F1={best_blend_f1:.4f}")

    # --------------- Pick the winner ---------------
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

    # --------------- Per-class additive priors ---------------
    # Greedy search: for each class find an additive offset on the prob that
    # maximizes OOF macro-F1. This boosts the rare/hard classes.
    # I use CV-folded tuning so the offsets don't memorize OOF noise. Otherwise
    # the offsets just memorize quirks in the validation predictions.
    print("\n--- Per-class additive prior tuning (CV-folded) ---")
    grid = np.linspace(-0.30, 0.30, 31)

    def _evaluate(adj, mask):
        return f1_score(y[mask], adj[mask].argmax(1), average="macro")

    def greedy_search(score_fn, n_iters=3):
        offs = np.zeros(N_CLASSES, dtype=np.float32)
        cur = score_fn(offs)
        for _it in range(n_iters):
            improved = False
            for c in range(N_CLASSES):
                best_l = (cur, offs[c])
                for delta in grid:
                    trial = offs.copy(); trial[c] = delta
                    f = score_fn(trial)
                    if f > best_l[0] + 1e-6:
                        best_l = (f, delta); improved = True
                if best_l[1] != offs[c]:
                    offs[c] = best_l[1]; cur = best_l[0]
            if not improved:
                break
        return offs, cur

    # CV-fold the tuning: 5 folds, learn offsets on 4, evaluate on 1, average
    gkf = GroupKFold(n_splits=5)
    fold_offsets = []
    for tr_idx, va_idx in gkf.split(final_oof, y, groups):
        def sc(off):
            adj = final_oof[tr_idx] + off
            return f1_score(y[tr_idx], adj.argmax(1), average="macro")
        o, _ = greedy_search(sc)
        fold_offsets.append(o)
    cv_offsets = np.mean(fold_offsets, axis=0)
    # also do full-OOF tuning as a comparison
    full_offsets, full_f1 = greedy_search(lambda off: f1_score(y, (final_oof + off).argmax(1), average="macro"))

    # pick whichever scores better on a separate validation (use 5-fold cv eval on full OOF)
    def cv_score(off):
        scores = []
        for tr_idx, va_idx in gkf.split(final_oof, y, groups):
            adj = final_oof[va_idx] + off
            scores.append(f1_score(y[va_idx], adj.argmax(1), average="macro"))
        return float(np.mean(scores))

    cv_score_cv = cv_score(cv_offsets)
    cv_score_full = cv_score(full_offsets)
    if cv_score_cv >= cv_score_full:
        offsets = cv_offsets
        print(f"  picked CV-averaged offsets (less overfit)  CV-F1: {cv_score_cv:.4f}")
    else:
        offsets = full_offsets
        print(f"  picked full-OOF offsets  CV-F1: {cv_score_full:.4f}")
    cur_f1 = f1_score(y, (final_oof + offsets).argmax(1), average="macro")
    print(f"  offsets = {np.round(offsets, 3).tolist()}")
    print(f"  OOF F1 after per-class tuning: {cur_f1:.4f}  (vs {final_f1:.4f} before)")

    # Apply to test predictions too
    final_oof = final_oof + offsets
    final_te_proba = final_te_proba + offsets
    final_f1 = cur_f1

    final_te_pred = final_te_proba.argmax(1)
    sub_template = pd.read_csv(SAMPLE_SUB)
    pred_df = pd.DataFrame({"Id": ref_ids_te.astype(sub_template["Id"].dtype),
                            "Label": final_te_pred.astype(int)})
    sub = sub_template[["Id"]].merge(pred_df, on="Id", how="left")
    miss = sub["Label"].isna().sum()
    if miss:
        print(f"WARNING: {miss} test ids unmatched - filling with majority class.")
        sub["Label"] = sub["Label"].fillna(pd.Series(y).mode().iloc[0])
    sub["Label"] = sub["Label"].astype(int)
    out_path = SUB_DIR / "submission_final.csv"
    sub.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")

    np.savez(CACHE_DIR / "ensemble_oof.npz",
             oof_proba=final_oof, test_proba=final_te_proba,
             y=y, ids_train=ref_ids_tr, ids_test=ref_ids_te,
             f1_final=np.array([final_f1], dtype=np.float32),
             f1_lgbm=np.array([f1_lgbm], dtype=np.float32),
             f1_cnn=np.array([f1_cnn], dtype=np.float32),
             f1_xgb=np.array([f1_xgb if has_xgb else -1.0], dtype=np.float32),
             f1_cat=np.array([f1_cat if has_cat else -1.0], dtype=np.float32),
             f1_mlp=np.array([f1_mlp if has_mlp else -1.0], dtype=np.float32),
             f1_rocket=np.array([f1_rocket if has_rocket else -1.0], dtype=np.float32),
             f1_stacker=np.array([best_stack_f1], dtype=np.float32),
             f1_blend=np.array([best_blend_f1], dtype=np.float32))

    print("\nPer-class report (final OOF):")
    print(classification_report(y, final_oof.argmax(1), digits=4))

    print("\n========== SUMMARY ==========")
    print(f"  LGBM    OOF macro-F1: {f1_lgbm:.4f}")
    print(f"  CNN     OOF macro-F1: {f1_cnn:.4f}")
    if has_xgb:
        print(f"  XGB     OOF macro-F1: {f1_xgb:.4f}")
    if has_cat:
        print(f"  CAT     OOF macro-F1: {f1_cat:.4f}")
    if has_mlp:
        print(f"  MLP     OOF macro-F1: {f1_mlp:.4f}")
    if has_rocket:
        print(f"  ROCKET  OOF macro-F1: {f1_rocket:.4f}")
    print(f"  Stacker OOF macro-F1: {best_stack_f1:.4f}")
    print(f"  Blend   OOF macro-F1: {best_blend_f1:.4f}")
    print(f"  Final ({chosen}) OOF: {final_f1:.4f}")
    print("==============================")


if __name__ == "__main__":
    main()
