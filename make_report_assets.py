"""Prompt 5 — Build report assets (figures + tables).

Generates the figures and tables your PDF report needs:
  - confusion matrix (ensemble OOF)
  - per-class F1 bar chart  (all 3 models on the same axis)
  - LGBM feature-group ablation table (markdown)
  - Summary scores table (markdown)

Saves PNGs to outputs/figures/ and markdown snippets to outputs/report/.

Run: python make_report_assets.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score

from config import CACHE_DIR, FIG_DIR, OUT_DIR, N_CLASSES

REPORT_DIR = OUT_DIR / "report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def _load_or_none(path):
    # allow_pickle=True because some caches store the user_id groups as object dtype
    return np.load(path, allow_pickle=True) if path.exists() else None


def confusion_fig(y, y_pred, title, fname):
    cm = confusion_matrix(y, y_pred, labels=list(range(N_CLASSES)))
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues", cbar=True, ax=ax,
                xticklabels=list(range(N_CLASSES)),
                yticklabels=list(range(N_CLASSES)))
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(FIG_DIR / fname); plt.close(fig)


def per_class_f1_chart(y, model_oofs):
    """Grouped bar chart of per-class F1. model_oofs = list of (label, oof|None)."""
    def per_class(pred):
        return [f1_score(y == c, pred == c, average="binary") for c in range(N_CLASSES)]
    slots = [(lbl, per_class(oof.argmax(1))) for lbl, oof in model_oofs if oof is not None]
    n_bars = len(slots)
    width = 0.8 / max(n_bars, 1)
    xs = np.arange(N_CLASSES)
    fig, ax = plt.subplots(figsize=(12, 4))
    centers = np.linspace(-(n_bars - 1) / 2, (n_bars - 1) / 2, n_bars) * width
    for offs, (lbl, vals) in zip(centers, slots):
        ax.bar(xs + offs, vals, width=width, label=lbl)
    ax.set_xticks(xs); ax.set_xticklabels([str(i) for i in xs])
    ax.set_xlabel("Class label"); ax.set_ylabel("F1 (one-vs-rest)")
    ax.set_title("Per-class F1 by model (OOF)")
    ax.set_ylim(0, 1.0); ax.legend(ncol=4); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG_DIR / "10_per_class_f1.png"); plt.close(fig)


def ablation_table_markdown():
    p = CACHE_DIR / "lgbm_ablation.csv"
    if not p.exists():
        print(f"  [skip] {p} missing")
        return None
    df = pd.read_csv(p)
    df = df.sort_values("drop_when_removed", ascending=False)
    df["only_f1"]      = df["only_f1"].map(lambda v: f"{v:.4f}")
    df["all_minus_f1"] = df["all_minus_f1"].map(lambda v: f"{v:.4f}")
    df["drop_when_removed"] = df["drop_when_removed"].map(lambda v: f"{v:+.4f}")
    md = df.to_markdown(index=False)
    (REPORT_DIR / "ablation_table.md").write_text(md, encoding="utf-8")
    print("  wrote", REPORT_DIR / "ablation_table.md")
    return md


def summary_scores_markdown(scores: dict[str, float]):
    """scores = {'naive': float, 'lgbm': float, 'cnn': float, 'ensemble': float, 'kaggle': float|None}"""
    rows = [(k, v) for k, v in scores.items() if v is not None]
    md = "| Stage | Score |\n|---|---|\n" + "\n".join(
        [f"| {k} | {v:.4f} |" for k, v in rows]
    )
    (REPORT_DIR / "summary_scores.md").write_text(md, encoding="utf-8")
    print("  wrote", REPORT_DIR / "summary_scores.md")
    return md


def main():
    print("Loading caches ...")
    naive  = _load_or_none(CACHE_DIR / "naive_oof.npz")
    lgbm   = _load_or_none(CACHE_DIR / "lgbm_oof.npz")
    cnn    = _load_or_none(CACHE_DIR / "cnn_oof.npz")
    xgb    = _load_or_none(CACHE_DIR / "xgb_oof.npz")
    cat    = _load_or_none(CACHE_DIR / "cat_oof.npz")
    mlp    = _load_or_none(CACHE_DIR / "mlp_oof.npz")
    rocket = _load_or_none(CACHE_DIR / "rocket_oof.npz")
    ens    = _load_or_none(CACHE_DIR / "ensemble_oof.npz")

    if lgbm is None and cnn is None:
        raise RuntimeError("Run Prompts 2/3 first — no OOF caches found.")

    # Use ensemble OOF if available, else best individual
    primary = ens if ens is not None else lgbm if lgbm is not None else cnn
    y = primary["y"]
    primary_oof = primary["oof_proba"]
    primary_name = "Ensemble" if ens is not None else ("LGBM" if lgbm is not None else "CNN")

    # ---- Confusion matrices ----
    print("Confusion matrices ...")
    confusion_fig(y, primary_oof.argmax(1),
                  f"{primary_name} OOF confusion matrix",
                  "07_confusion_primary.png")
    if lgbm is not None and primary_name != "LGBM":
        confusion_fig(y, lgbm["oof_proba"].argmax(1), "LGBM OOF confusion matrix",
                      "08_confusion_lgbm.png")
    if cnn is not None and primary_name != "CNN":
        confusion_fig(y, cnn["oof_proba"].argmax(1), "CNN-LSTM OOF confusion matrix",
                      "09_confusion_cnn.png")
    if xgb is not None:
        confusion_fig(y, xgb["oof_proba"].argmax(1), "XGBoost OOF confusion matrix",
                      "11_confusion_xgb.png")
    if cat is not None:
        confusion_fig(y, cat["oof_proba"].argmax(1), "CatBoost OOF confusion matrix",
                      "12_confusion_cat.png")
    if mlp is not None:
        confusion_fig(y, mlp["oof_proba"].argmax(1), "MLP OOF confusion matrix",
                      "13_confusion_mlp.png")
    if rocket is not None:
        confusion_fig(y, rocket["oof_proba"].argmax(1), "ROCKET OOF confusion matrix",
                      "14_confusion_rocket.png")

    # ---- Per-class F1 chart ----
    print("Per-class F1 chart ...")
    per_class_f1_chart(y, [
        ("LGBM",   lgbm["oof_proba"]   if lgbm   is not None else None),
        ("CNN",    cnn["oof_proba"]    if cnn    is not None else None),
        ("XGB",    xgb["oof_proba"]    if xgb    is not None else None),
        ("CatBoost", cat["oof_proba"]  if cat    is not None else None),
        ("MLP",    mlp["oof_proba"]    if mlp    is not None else None),
        ("ROCKET", rocket["oof_proba"] if rocket is not None else None),
        ("Ensemble", ens["oof_proba"]  if ens    is not None else None),
    ])

    # ---- Tables ----
    print("Ablation table ...")
    ablation_table_markdown()

    print("Summary scores table ...")
    scores = {}
    if naive is not None:
        try:
            f_lr = f1_score(naive["y"], naive["oof_lr"], average="macro")
            f_rf = f1_score(naive["y"], naive["oof_rf"], average="macro")
            scores["Naive (best of LogReg/RF)"] = max(f_lr, f_rf)
        except Exception:
            pass
    if lgbm is not None:
        scores["LightGBM (rich features)"] = f1_score(lgbm["y"], lgbm["oof_proba"].argmax(1), average="macro")
    if cnn is not None:
        scores["CNN-LSTM (raw sequences)"] = f1_score(cnn["y"], cnn["oof_proba"].argmax(1), average="macro")
    if xgb is not None:
        scores["XGBoost (rich features)"] = f1_score(xgb["y"], xgb["oof_proba"].argmax(1), average="macro")
    if cat is not None:
        scores["CatBoost (rich features)"] = f1_score(cat["y"], cat["oof_proba"].argmax(1), average="macro")
    if mlp is not None:
        scores["MLP (rich features)"] = f1_score(mlp["y"], mlp["oof_proba"].argmax(1), average="macro")
    if rocket is not None:
        scores["ROCKET (raw signal)"] = f1_score(rocket["y"], rocket["oof_proba"].argmax(1), average="macro")
    if ens is not None:
        scores["Ensemble (6-way blend)"] = f1_score(ens["y"], ens["oof_proba"].argmax(1), average="macro")
    summary_scores_markdown(scores)

    print("\nAll assets written.")
    print(f"  Figures: {FIG_DIR}")
    print(f"  Tables : {REPORT_DIR}")


if __name__ == "__main__":
    main()
