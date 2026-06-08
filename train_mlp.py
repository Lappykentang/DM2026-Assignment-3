# 5th base model: a simple MLP on the rich 288 features. This is a different
# function class from boosted trees - smooth predictions, different errors.
# I added this after the CNN got stuck at OOF 0.63; needed another diverse signal.
#
# Outputs:
#   outputs/submissions/submission_mlp.csv
#   outputs/cache/mlp_oof.npz
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from config import SEED, SUB_DIR, CACHE_DIR, SAMPLE_SUB, N_CLASSES
from data_loader import load_train, load_test
from feature_engineering import build_features


class MLP(nn.Module):
    # Just a 3-layer MLP with dropout + batchnorm. Nothing fancy - this is a
    # diversity-adding model, not a primary one.
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


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_fold(train_x, train_y, valid_x, valid_y, seed=42,
                   epochs=80, batch_size=128, lr=1e-3, weight_decay=1e-4,
                   class_weights=None, device="cpu"):
    set_seed(seed)
    sc = StandardScaler().fit(train_x)
    train_x_s = sc.transform(train_x).astype(np.float32)
    valid_x_s = sc.transform(valid_x).astype(np.float32)

    train_ds = TensorDataset(torch.from_numpy(train_x_s), torch.from_numpy(train_y).long())
    valid_ds = TensorDataset(torch.from_numpy(valid_x_s), torch.from_numpy(valid_y).long())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

    model = MLP(in_dim=train_x.shape[1]).to(device)
    cw = torch.tensor(class_weights, dtype=torch.float32, device=device) if class_weights is not None else None
    crit = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    best_f1, best_state, no_imp = -1.0, None, 0
    patience = 15
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            optim.step()
        sched.step()
        model.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in valid_loader:
                preds.append(model(xb.to(device)).argmax(1).cpu().numpy())
        val_pred = np.concatenate(preds)
        f1 = f1_score(valid_y, val_pred, average="macro")
        if f1 > best_f1:
            best_f1, best_state, no_imp = f1, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            no_imp += 1
        if no_imp >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    return model, sc, best_f1


def predict_proba(model, sc, x, device="cpu"):
    model.eval()
    x_s = sc.transform(x).astype(np.float32)
    with torch.no_grad():
        logits = model(torch.from_numpy(x_s).to(device))
        return torch.softmax(logits, dim=1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="7,13,42")
    args = ap.parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Bagging seeds: {seeds}")

    print("Loading data ...")
    tr_long = load_train(); te_long = load_test()
    f_tr, _ = build_features(tr_long, cache_name="feats_train_v4")
    f_te, _ = build_features(te_long, cache_name="feats_test_v4")

    feat_cols = [c for c in f_tr.columns if c not in {"file_id", "user_id", "label"}]
    train_x = f_tr[feat_cols].to_numpy(dtype=np.float32)
    train_y = f_tr["label"].to_numpy(dtype=int)
    train_groups = f_tr["user_id"].to_numpy()
    ids_tr = f_tr["file_id"].to_numpy()
    test_x = f_te[feat_cols].to_numpy(dtype=np.float32)
    ids_te = f_te["file_id"].to_numpy()
    print(f"features: {len(feat_cols)}  X_train {train_x.shape}")

    counts = np.bincount(train_y, minlength=N_CLASSES).astype(float)
    inv = 1.0 / np.clip(counts, 1, None)
    class_weights = (inv / inv.mean()).astype(np.float32)

    oof_seeds, test_seeds = [], []
    for s in seeds:
        print(f"\n=== MLP CV (seed={s}) ===")
        gkf = GroupKFold(n_splits=5)
        oof_s = np.zeros((len(train_y), N_CLASSES), dtype=np.float32)
        test_acc = np.zeros((test_x.shape[0], N_CLASSES), dtype=np.float32)
        for k, (tr_idx, va_idx) in enumerate(gkf.split(train_x, train_y, train_groups)):
            model, sc, f1 = train_one_fold(
                train_x[tr_idx], train_y[tr_idx],
                train_x[va_idx], train_y[va_idx],
                seed=s, class_weights=class_weights, device=device,
            )
            oof_s[va_idx] = predict_proba(model, sc, train_x[va_idx], device=device)
            test_acc += predict_proba(model, sc, test_x, device=device)
            print(f"  fold {k+1}: macro-F1={f1:.4f}")
        test_s = test_acc / 5
        oof_seeds.append(oof_s); test_seeds.append(test_s)
        print(f"  seed={s} OOF F1: {f1_score(train_y, oof_s.argmax(1), average='macro'):.4f}")

    oof_proba  = np.mean(oof_seeds, axis=0).astype(np.float32)
    test_proba = np.mean(test_seeds, axis=0).astype(np.float32)
    f1_bag = f1_score(train_y, oof_proba.argmax(1), average="macro")
    print(f"\nMLP bagged ({len(seeds)} seeds) OOF macro-F1 = {f1_bag:.4f}")

    test_pred = test_proba.argmax(axis=1)
    sub_template = pd.read_csv(SAMPLE_SUB)
    pred_df = pd.DataFrame({"Id": ids_te.astype(sub_template["Id"].dtype), "Label": test_pred})
    sub = sub_template[["Id"]].merge(pred_df, on="Id", how="left")
    miss = sub["Label"].isna().sum()
    if miss:
        print(f"WARNING: {miss} ids missing - filling with majority class.")
        sub["Label"] = sub["Label"].fillna(pd.Series(train_y).mode().iloc[0])
    sub["Label"] = sub["Label"].astype(int)
    out_path = SUB_DIR / "submission_mlp.csv"
    sub.to_csv(out_path, index=False)
    print(f"Saved {out_path}")

    np.savez(CACHE_DIR / "mlp_oof.npz",
             oof_proba=oof_proba, test_proba=test_proba,
             y=train_y, ids_train=ids_tr, ids_test=ids_te, groups=train_groups)
    print("Saved cache/mlp_oof.npz")

    print("\n=== Per-class report on OOF ===")
    print(classification_report(train_y, oof_proba.argmax(axis=1), digits=4))


if __name__ == "__main__":
    main()
