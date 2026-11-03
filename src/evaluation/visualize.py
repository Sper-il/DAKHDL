"""Vẽ đồ thị và confusion matrix bằng matplotlib (PNG rồi nhúng base64)."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def plot_confusion_matrix(cm, save_path: str | Path, title: str = "Confusion matrix"):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(EMOTIONS)), EMOTIONS, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(EMOTIONS)), EMOTIONS, fontsize=8)
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Thực tế")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return str(save_path)


def plot_loss_curve(train_losses, val_losses, save_path: str | Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(train_losses, label="train")
    ax.plot(val_losses, label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Đường cong loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return str(save_path)
