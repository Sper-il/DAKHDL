#!/usr/bin/env python
"""Vẽ đồ thị từ log của một run (cần pandas + matplotlib).
    python scripts/plot_logs.py --run-dir runs/face_resnet18_2026...
Tạo trong <run-dir>/plots: curves.png, grad_norm_heatmap.png, update_ratio_heatmap.png, weight_norm_heatmap.png
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def heatmap(df, value, title, path):
    pv = df.pivot(index="param", columns="epoch", values=value)
    pv = pv.loc[df.drop_duplicates("param")["param"]]  # giữ thứ tự lớp trong mô hình
    data = np.log10(np.clip(pv.to_numpy(dtype=float), 1e-12, None))
    fig, ax = plt.subplots(figsize=(max(6, 0.35 * pv.shape[1] + 4), max(4, 0.22 * pv.shape[0] + 2)))
    im = ax.imshow(data, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(pv.index)))
    ax.set_yticklabels(pv.index, fontsize=6)
    ax.set_xticks(range(len(pv.columns)))
    ax.set_xticklabels(pv.columns, fontsize=6)
    ax.set_xlabel("epoch")
    ax.set_title(title + " (log10)")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    a = ap.parse_args()
    run = Path(a.run_dir)
    out = run / "plots"
    out.mkdir(exist_ok=True)

    ep = pd.read_csv(run / "epochs.csv")
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    ax = ax.ravel()
    ax[0].plot(ep["epoch"], ep["train_loss"], label="train"); ax[0].plot(ep["epoch"], ep["val_loss"], label="val")
    ax[0].set_title("loss"); ax[0].legend()
    ax[1].plot(ep["epoch"], ep["train_acc"], label="train"); ax[1].plot(ep["epoch"], ep["val_acc"], label="val")
    ax[1].set_title("accuracy"); ax[1].legend()
    ax[2].plot(ep["epoch"], ep["val_macro_f1"], label="macro-F1"); ax[2].plot(ep["epoch"], ep["val_bal_acc"], label="balanced acc")
    ax[2].set_title("val macro-F1 / balanced acc"); ax[2].legend()
    ax[3].plot(ep["epoch"], ep["lr"]); ax[3].set_title("learning rate")
    ax[4].plot(ep["epoch"], ep["grad_norm_mean"]); ax[4].set_title("norm gradient trung bình")
    ax[5].plot(ep["epoch"], ep["epoch_sec"]); ax[5].set_title("giây / epoch")
    for x in ax:
        x.set_xlabel("epoch"); x.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "curves.png", dpi=130); plt.close(fig)

    pr = pd.read_csv(run / "params.csv")
    heatmap(pr, "grad_norm_mean", "Norm gradient trung bình từng tham số", out / "grad_norm_heatmap.png")
    heatmap(pr, "update_ratio", "Tỉ lệ thay đổi ||Δw||/||w|| mỗi epoch", out / "update_ratio_heatmap.png")
    heatmap(pr, "w_norm", "Norm trọng số", out / "weight_norm_heatmap.png")
    print(f"Đã lưu đồ thị vào {out}")


if __name__ == "__main__":
    main()
