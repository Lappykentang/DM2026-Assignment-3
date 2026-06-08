"""Exploratory data analysis: label distribution, per-user stats, signal plots.
Run: python eda.py
Saves figures to outputs/figures/.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import FIG_DIR, FEATURE_COLS, SEQ_LEN, N_CLASSES
from data_loader import load_train, load_test, file_meta

plt.rcParams["figure.dpi"] = 110


def fig1_label_distribution(train_long: pd.DataFrame):
    meta = file_meta(train_long)
    counts = meta["label"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index.astype(str), counts.values, color="steelblue")
    ax.set_title(f"Label distribution (files) — total = {len(meta)}")
    ax.set_xlabel("label"); ax.set_ylabel("# files")
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG_DIR / "01_label_distribution.png"); plt.close(fig)
    return counts


def fig2_files_per_user(train_long: pd.DataFrame):
    meta = file_meta(train_long)
    per_user = meta.groupby("user_id").size().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.bar(range(len(per_user)), per_user.values, color="darkorange")
    ax.set_title("Files per user (train)")
    ax.set_xlabel("user (sorted)"); ax.set_ylabel("# files")
    fig.tight_layout(); fig.savefig(FIG_DIR / "02_files_per_user.png"); plt.close(fig)
    return per_user.describe()


def fig3_label_user_heatmap(train_long: pd.DataFrame):
    meta = file_meta(train_long)
    pivot = meta.pivot_table(index="user_id", columns="label", values="file_id",
                             aggfunc="count", fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 10))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(pivot.shape[1])); ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(pivot.shape[0])); ax.set_yticklabels(pivot.index, fontsize=6)
    ax.set_title("Files per (user, label)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(FIG_DIR / "03_user_label_heatmap.png"); plt.close(fig)


def fig4_signal_examples(train_long: pd.DataFrame):
    """One example file per label, plot mean_x/y/z over 300 seconds."""
    meta = file_meta(train_long)
    fig, axes = plt.subplots(N_CLASSES, 1, figsize=(10, 2.0 * N_CLASSES), sharex=True)
    for label in range(N_CLASSES):
        fid = meta[meta["label"] == label]["file_id"].iloc[0]
        sub = train_long[train_long["file_id"] == fid].sort_values("index")
        ax = axes[label]
        ax.plot(sub["index"], sub["mean_x"], label="mean_x", lw=0.8)
        ax.plot(sub["index"], sub["mean_y"], label="mean_y", lw=0.8)
        ax.plot(sub["index"], sub["mean_z"], label="mean_z", lw=0.8)
        ax.set_ylabel(f"label {label}")
        ax.legend(loc="upper right", fontsize=7)
    axes[-1].set_xlabel("second within window")
    fig.suptitle("Example signal per class (mean_xyz)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "04_signal_examples_mean.png"); plt.close(fig)

    # Std version
    fig, axes = plt.subplots(N_CLASSES, 1, figsize=(10, 2.0 * N_CLASSES), sharex=True)
    for label in range(N_CLASSES):
        fid = meta[meta["label"] == label]["file_id"].iloc[0]
        sub = train_long[train_long["file_id"] == fid].sort_values("index")
        ax = axes[label]
        ax.plot(sub["index"], sub["std_x"], label="std_x", lw=0.8)
        ax.plot(sub["index"], sub["std_y"], label="std_y", lw=0.8)
        ax.plot(sub["index"], sub["std_z"], label="std_z", lw=0.8)
        ax.set_ylabel(f"label {label}")
        ax.legend(loc="upper right", fontsize=7)
    axes[-1].set_xlabel("second within window")
    fig.suptitle("Example signal per class (std_xyz)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "05_signal_examples_std.png"); plt.close(fig)


def fig5_per_class_feature_box(train_long: pd.DataFrame):
    """Per-class boxplot of file-level means for each axis."""
    meta = file_meta(train_long)
    agg = (train_long
           .groupby("file_id")[FEATURE_COLS].mean()
           .reset_index()
           .merge(meta[["file_id", "label"]], on="file_id"))
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, col in zip(axes.flat, FEATURE_COLS):
        data = [agg[agg["label"] == l][col].values for l in range(N_CLASSES)]
        ax.boxplot(data, labels=range(N_CLASSES), showfliers=False)
        ax.set_title(col); ax.set_xlabel("label")
    fig.suptitle("Per-class distribution of file-level mean of each column")
    fig.tight_layout(); fig.savefig(FIG_DIR / "06_per_class_box.png"); plt.close(fig)


def sanity_check(train_long: pd.DataFrame, test_long: pd.DataFrame):
    print("\n=== Sanity ===")
    # rows-per-file distribution
    tr_n = train_long.groupby("file_id").size()
    te_n = test_long.groupby("file_id").size()
    print(f"train rows-per-file: min={tr_n.min()}  max={tr_n.max()}  median={int(tr_n.median())}")
    print(f"test  rows-per-file: min={te_n.min()}  max={te_n.max()}  median={int(te_n.median())}")
    # NaNs
    print("NaNs in train features:", train_long[FEATURE_COLS].isna().sum().sum())
    print("NaNs in test  features:", test_long[FEATURE_COLS].isna().sum().sum())
    # Disjoint users
    tr_users = set(train_long["user_id"].unique())
    te_users = set(test_long["user_id"].unique())
    print(f"train users: {len(tr_users)}  test users: {len(te_users)}  overlap: {len(tr_users & te_users)}")


def main():
    tr = load_train()
    te = load_test()
    print("Train files:", tr["file_id"].nunique(), "| Test files:", te["file_id"].nunique())
    counts = fig1_label_distribution(tr); print("Label counts:\n", counts)
    print("\nFiles-per-user describe:\n", fig2_files_per_user(tr))
    fig3_label_user_heatmap(tr)
    fig4_signal_examples(tr)
    fig5_per_class_feature_box(tr)
    sanity_check(tr, te)
    print(f"\nFigures saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()
