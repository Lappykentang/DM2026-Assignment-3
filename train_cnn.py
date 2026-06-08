# CNN-LSTM trained on raw (300, 6) sequences. 5-fold GroupKFold by user.
# I first tried just a CNN, then added an LSTM on top - the LSTM helped a bit.
# Run: python train_cnn.py
#      python train_cnn.py --inference-only   # use cached fold checkpoints (fast)
#
# Files written:
#   outputs/submissions/submission_cnn.csv
#   outputs/cache/cnn_oof.npz   (oof + test probs)
#   outputs/cache/cnn_fold{k}.pt (best checkpoint per fold)
#
# Notes to self:
#  - augmentations (jitter / shift / scale) only on train, never on val/test
#  - class weighting helps a little bit for the rare classes
#  - on my CPU one fold takes ~6-10 min; whole thing 30-60 min
from __future__ import annotations
import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import GroupKFold

from config import SEED, N_CLASSES, SUB_DIR, CACHE_DIR, SAMPLE_SUB
from data_loader import load_train, load_test, file_meta
from cnn_dataset import HARDataset, build_data_dict, compute_normalizer
from cnn_model import CNNLSTM, count_params


# --------------------------- reproducibility --------------------------- #
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Focal loss. I tried plain CE first then class-weighted CE, but class 2 still
# had F1 ~0.3 because the model was getting the easy classes "right enough" and
# ignoring the hard ones. Focal loss with gamma=1.5 down-weights confident-correct
# predictions, so the gradient signal stays on the rare/hard examples.
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.5, label_smoothing=0.05):
        super().__init__()
        self.alpha = alpha  # per-class weight tensor (or None)
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        n_classes = logits.size(1)
        log_prob = torch.log_softmax(logits, dim=1)
        prob = torch.exp(log_prob)
        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth_targets = torch.full_like(log_prob, self.label_smoothing / (n_classes - 1))
                smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            ce = -(smooth_targets * log_prob).sum(dim=1)
            p_t = prob.gather(1, targets.unsqueeze(1)).squeeze(1).clamp(min=1e-7, max=1.0)
            focal = (1.0 - p_t) ** self.gamma
        else:
            ce = nn.functional.nll_loss(log_prob, targets, reduction="none")
            p_t = prob.gather(1, targets.unsqueeze(1)).squeeze(1).clamp(min=1e-7, max=1.0)
            focal = (1.0 - p_t) ** self.gamma

        loss = focal * ce
        if self.alpha is not None:
            loss = loss * self.alpha[targets]
        return loss.mean()


def mixup_batch(x, y, alpha=0.2):
    # Mixup: blend two random samples in a batch. Helps generalization on
    # imbalanced data - the model sees soft labels between classes.
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    # Skew lam away from 0.5 to keep one sample dominant
    lam = max(lam, 1.0 - lam)
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


# --------------------------- training one fold --------------------------- #
def train_one_fold(
    fold: int,
    data_dict, labels_dict,
    train_ids, val_ids,
    test_data_dict, test_ids,
    device: str,
    epochs: int, batch_size: int, lr: float, weight_decay: float,
    patience: int,
    class_weights: np.ndarray | None,
    out_ckpt: Path,
    inference_only: bool = False,
    use_focal_loss: bool = False,
    use_mixup: bool = False,
    mixup_alpha: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Returns (val_oof_proba, test_proba_avg_logits_softmax, best_val_f1)."""
    # Per-fold normalizer fitted on train_ids only (no leakage)
    mean, std = compute_normalizer(data_dict, train_ids)

    train_ds = HARDataset(data_dict, train_ids, labels_dict,
                          normalizer=(mean, std), augment=True,
                          rng_seed=SEED + fold)
    val_ds   = HARDataset(data_dict, val_ids,   labels_dict,
                          normalizer=(mean, std), augment=False)
    test_ds  = HARDataset(test_data_dict, test_ids, labels_dict=None,
                          normalizer=(mean, std), augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=(device == "cuda"))
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=(device == "cuda"))

    model = CNNLSTM(in_channels=6, n_classes=N_CLASSES).to(device)
    if fold == 0:
        print(f"  Model params: {count_params(model):,}")

    # Inference-only path: skip training, just load the cached checkpoint.
    # I added this because retraining CNN for every tweak took forever.
    skip_training = inference_only and out_ckpt.exists()
    if skip_training:
        print(f"  [inference-only] loading {out_ckpt.name} and skipping training")
        best_state = torch.load(out_ckpt, map_location=device)
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            all_pred, all_true = [], []
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                all_pred.append(model(xb).argmax(1).cpu().numpy())
                all_true.append(yb.numpy())
        best_f1 = f1_score(np.concatenate(all_true), np.concatenate(all_pred), average="macro")
        print(f"  cached val macro-F1: {best_f1:.4f}")
    else:
        cw = None if class_weights is None else torch.tensor(class_weights, dtype=torch.float32, device=device)
        if use_focal_loss:
            criterion = FocalLoss(alpha=cw, gamma=1.5, label_smoothing=0.05)
            print(f"  using FocalLoss(gamma=1.5, alpha={None if cw is None else 'class_weights'})")
        else:
            criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)
        optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
        best_f1, best_state, no_improve = -1.0, None, 0

        if use_mixup:
            print(f"  using mixup (alpha=0.2)")

    train_epochs = 0 if skip_training else epochs
    for epoch in range(1, train_epochs + 1):
        # ---- train ----
        model.train()
        tot_loss = 0.0; n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optim.zero_grad()
            if use_mixup:
                xb_m, y_a, y_b, lam = mixup_batch(xb, yb, alpha=mixup_alpha)
                logits = model(xb_m)
                loss = lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
            else:
                logits = model(xb)
                loss = criterion(logits, yb)
            loss.backward()
            optim.step()
            tot_loss += loss.item() * xb.size(0); n += xb.size(0)
        sched.step()
        train_loss = tot_loss / max(n, 1)

        # ---- validate ----
        model.eval()
        all_pred, all_true = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                logits = model(xb)
                all_pred.append(logits.argmax(1).cpu().numpy())
                all_true.append(yb.numpy())
        val_pred = np.concatenate(all_pred)
        val_true = np.concatenate(all_true)
        val_f1 = f1_score(val_true, val_pred, average="macro")

        improved = val_f1 > best_f1
        if improved:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if epoch % 1 == 0:
            print(f"  fold {fold+1} ep {epoch:02d}/{epochs} | train_loss {train_loss:.4f} | val_F1 {val_f1:.4f}"
                  + ("  *" if improved else ""))
        if no_improve >= patience:
            print(f"  early stop at epoch {epoch}")
            break

    # Load best, compute OOF probs and test probs
    model.load_state_dict(best_state)
    torch.save(best_state, out_ckpt)

    def _proba(loader, with_label):
        model.eval()
        probs, ids = [], []
        with torch.no_grad():
            for xb, meta in loader:
                xb = xb.to(device, non_blocking=True)
                logits = model(xb)
                probs.append(torch.softmax(logits, dim=1).cpu().numpy())
                ids.append(meta.numpy())
        return np.concatenate(probs), np.concatenate(ids)

    # Test-time augmentation. I tried just plain inference first, then added TTA.
    # Idea: small jitter + small time-shift, predict N times, average. Adds a small
    # boost (~+0.005-0.010 in my runs).
    def _proba_tta(file_ids, data_dict_src, with_label, n_tta=8):
        model.eval()
        # Clean pass (no augmentation) - gets the same weight as each TTA pass.
        ds_clean = HARDataset(data_dict_src, file_ids,
                              labels_dict=(labels_dict if with_label else None),
                              normalizer=(mean, std), augment=False)
        loader_c = DataLoader(ds_clean, batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=(device == "cuda"))
        collected_probs, collected_ids = [], []
        with torch.no_grad():
            for xb, meta in loader_c:
                xb = xb.to(device, non_blocking=True)
                collected_probs.append(torch.softmax(model(xb), dim=1).cpu().numpy())
                collected_ids.append(meta.numpy())
        p_sum = np.concatenate(collected_probs)
        ids   = np.concatenate(collected_ids)
        # Augmented passes - mild only (don't want to lose signal)
        for k in range(n_tta):
            ds_aug = HARDataset(data_dict_src, file_ids,
                                labels_dict=(labels_dict if with_label else None),
                                normalizer=(mean, std), augment=True,
                                jitter_sigma=0.01, shift_range=5,
                                p_jitter=1.0, p_shift=1.0, p_scale=0.0,
                                rng_seed=SEED + fold * 100 + k)
            loader_a = DataLoader(ds_aug, batch_size=batch_size, shuffle=False,
                                  num_workers=0, pin_memory=(device == "cuda"))
            with torch.no_grad():
                buf = []
                for xb, _ in loader_a:
                    xb = xb.to(device, non_blocking=True)
                    buf.append(torch.softmax(model(xb), dim=1).cpu().numpy())
            p_sum = p_sum + np.concatenate(buf)
        return p_sum / (n_tta + 1), ids

    # 4 TTA passes - tradeoff between speed and gain.
    val_proba, val_ids_arr   = _proba_tta(val_ids,  data_dict,      with_label=True,  n_tta=4)
    test_proba, test_ids_arr = _proba_tta(test_ids, test_data_dict, with_label=False, n_tta=4)

    # Re-order to match val_ids / test_ids inputs (DataLoader keeps order since shuffle=False)
    # val_loader's meta is y, not file_id. We use val_ids list order as-is.
    return val_proba, test_proba, best_f1


# --------------------------- main --------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--device", type=str, default=None,
                    help="cuda | cpu (default: auto)")
    ap.add_argument("--inference-only", action="store_true",
                    help="Skip training, just load fold checkpoints and re-run TTA inference. Fast!")
    ap.add_argument("--focal-loss", action="store_true",
                    help="Use focal loss instead of plain CE. Helps the rare classes.")
    ap.add_argument("--mixup", action="store_true",
                    help="Use mixup augmentation - blend two random samples in each batch.")
    ap.add_argument("--mixup-alpha", type=float, default=0.2,
                    help="Mixup beta distribution parameter (0.2 is mild).")
    ap.add_argument("--boost-rare", type=float, default=1.0,
                    help="Multiply class weights for classes 2, 4, 5 by this factor.")
    args = ap.parse_args()

    set_seed(SEED)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading data ...")
    tr_long = load_train()
    te_long = load_test()
    tr_meta = file_meta(tr_long)
    te_meta = file_meta(te_long)

    print("Building per-file tensors ...")
    tr_data = build_data_dict(tr_long)
    te_data = build_data_dict(te_long)
    labels_dict = {int(r.file_id): int(r.label) for r in tr_meta.itertuples(index=False)}

    # GroupKFold by user
    tr_ids   = tr_meta["file_id"].to_numpy()
    tr_y     = tr_meta["label"].to_numpy()
    tr_users = tr_meta["user_id"].to_numpy()
    te_ids   = te_meta["file_id"].to_numpy()
    print(f"train files: {len(tr_ids)}  test files: {len(te_ids)}")
    print("Label counts:", dict(pd.Series(tr_y).value_counts().sort_index()))

    # Class weights (inverse frequency, normalized to mean 1)
    counts = np.bincount(tr_y, minlength=N_CLASSES).astype(float)
    inv = 1.0 / np.clip(counts, 1, None)
    class_weights = (inv / inv.mean()).astype(np.float32)
    # Boost rare classes - class 2 is the worst (F1 ~0.30), so boost it more.
    # Tried boosting 2/4/5 all at once - that hurt classes 4 and 5 which were
    # already decent. Now just boost class 2 strongly.
    if args.boost_rare != 1.0:
        class_weights[2] *= args.boost_rare
        # 4 and 5 get only half the boost
        class_weights[4] *= max(1.0, (args.boost_rare - 1.0) * 0.5 + 1.0)
        class_weights[5] *= max(1.0, (args.boost_rare - 1.0) * 0.5 + 1.0)
        print(f"Boosted class-2 weight by x{args.boost_rare}, classes 4/5 by x{(args.boost_rare - 1.0) * 0.5 + 1.0:.2f}")
    print("Class weights:", np.round(class_weights, 3).tolist())

    oof_proba = np.zeros((len(tr_ids), N_CLASSES), dtype=np.float32)
    test_proba_sum = np.zeros((len(te_ids), N_CLASSES), dtype=np.float32)
    fold_f1 = []

    gkf = GroupKFold(n_splits=args.folds)
    fold_iter = list(gkf.split(tr_ids, tr_y, tr_users))

    t0 = time.time()
    for fold, (tr_idx, va_idx) in enumerate(fold_iter):
        print(f"\n=== Fold {fold+1}/{args.folds} (train={len(tr_idx)}  val={len(va_idx)}) ===")
        train_ids_f = tr_ids[tr_idx]
        val_ids_f   = tr_ids[va_idx]

        ckpt_path = CACHE_DIR / f"cnn_fold{fold+1}.pt"
        val_proba, test_proba, best_f1 = train_one_fold(
            fold=fold,
            data_dict=tr_data, labels_dict=labels_dict,
            train_ids=train_ids_f, val_ids=val_ids_f,
            test_data_dict=te_data, test_ids=te_ids,
            device=device,
            epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, weight_decay=args.weight_decay,
            patience=args.patience,
            class_weights=class_weights,
            out_ckpt=ckpt_path,
            inference_only=args.inference_only,
            use_focal_loss=args.focal_loss,
            use_mixup=args.mixup,
            mixup_alpha=args.mixup_alpha,
        )
        oof_proba[va_idx] = val_proba
        test_proba_sum += test_proba
        fold_f1.append(best_f1)
        print(f"  fold {fold+1} best val macro-F1: {best_f1:.4f}  ({(time.time()-t0)/60:.1f} min total)")

    test_proba = test_proba_sum / args.folds
    test_pred = test_proba.argmax(axis=1)
    oof_pred = oof_proba.argmax(axis=1)
    final_oof_f1 = f1_score(tr_y, oof_pred, average="macro")

    print("\n=== Summary ===")
    for i, f in enumerate(fold_f1):
        print(f"  fold {i+1}: {f:.4f}")
    print(f"  mean of folds: {np.mean(fold_f1):.4f} +/- {np.std(fold_f1):.4f}")
    print(f"  pooled-OOF macro-F1: {final_oof_f1:.4f}")
    print("\nPer-class report (pooled OOF):")
    print(classification_report(tr_y, oof_pred, digits=4))

    # ---- Save submission ----
    sub_template = pd.read_csv(SAMPLE_SUB)
    pred_df = pd.DataFrame({"Id": te_ids.astype(sub_template["Id"].dtype),
                            "Label": test_pred.astype(int)})
    sub = sub_template[["Id"]].merge(pred_df, on="Id", how="left")
    miss = sub["Label"].isna().sum()
    if miss:
        print(f"WARNING: {miss} ids missing — filling with majority class.")
        sub["Label"] = sub["Label"].fillna(pd.Series(tr_y).mode().iloc[0])
    sub["Label"] = sub["Label"].astype(int)
    out_path = SUB_DIR / "submission_cnn.csv"
    sub.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")

    np.savez(CACHE_DIR / "cnn_oof.npz",
             oof_proba=oof_proba, test_proba=test_proba,
             y=tr_y, ids_train=tr_ids, ids_test=te_ids, groups=tr_users,
             fold_f1=np.array(fold_f1, dtype=np.float32))
    print("Saved cache/cnn_oof.npz (used by Prompt 4 ensemble).")


if __name__ == "__main__":
    main()
