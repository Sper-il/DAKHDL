"""Chỉ số đánh giá thuần numpy (không cần torch)."""
from __future__ import annotations

import numpy as np


def confusion_matrix(y_true, y_pred, n: int) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    return np.bincount(y_true * n + y_pred, minlength=n * n).reshape(n, n)


def per_class(cm: np.ndarray):
    """Trả về precision, recall, f1, support cho từng lớp (hàng = thật, cột = dự đoán)."""
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(1).astype(np.float64)
    predicted = cm.sum(0).astype(np.float64)
    prec = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    rec = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = prec + rec
    f1 = np.divide(2 * prec * rec, denom, out=np.zeros_like(tp), where=denom > 0)
    return prec, rec, f1, support


def summarize(y_true, probs, n: int) -> dict:
    """acc, macro_f1, bal_acc (trung bình recall), loss (NLL) và báo cáo từng lớp.
    macro_f1 và bal_acc chỉ lấy trung bình trên các lớp có mặt trong y_true."""
    probs = np.asarray(probs, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)
    pred = probs.argmax(1)
    cm = confusion_matrix(y_true, pred, n)
    prec, rec, f1, sup = per_class(cm)
    present = sup > 0
    p_true = np.clip(probs[np.arange(len(y_true)), y_true], 1e-12, 1.0)
    return {
        "acc": float((pred == y_true).mean()),
        "macro_f1": float(f1[present].mean()),
        "bal_acc": float(rec[present].mean()),
        "loss": float(-np.log(p_true).mean()),
        "cm": cm,
        "precision": prec, "recall": rec, "f1": f1, "support": sup.astype(int),
    }


def by_group(y_true, probs, groups, n: int) -> dict:
    """Kết quả tách theo nhóm (vd theo nguồn data: asvp / crema / ravdess)."""
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    groups = np.asarray(groups)
    out = {}
    for g in sorted(set(groups.tolist())):
        m = groups == g
        s = summarize(y_true[m], probs[m], n)
        out[str(g)] = {"n": int(m.sum()), "acc": s["acc"], "macro_f1": s["macro_f1"],
                       "bal_acc": s["bal_acc"]}
    return out


def plot_confusion(cm: np.ndarray, names, path, title: str = "", normalize: bool = True) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(cm, dtype=np.float64)
    shown = cm / np.clip(cm.sum(1, keepdims=True), 1, None) if normalize else cm
    n = len(names)
    fig, ax = plt.subplots(figsize=(0.8 * n + 3, 0.8 * n + 2.5))
    im = ax.imshow(shown, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(names)
    ax.set_xlabel("Dự đoán"); ax.set_ylabel("Thật"); ax.set_title(title)
    for i in range(n):
        for j in range(n):
            txt = f"{shown[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    color="white" if shown[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
