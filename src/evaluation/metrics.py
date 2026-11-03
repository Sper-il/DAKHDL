"""Metric đánh giá 2 lớp: model level + full pipeline."""
from __future__ import annotations
import time
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def evaluate_branch(model, loader, device: str | None = None) -> dict:
    """Đánh giá 1 nhánh: accuracy, F1 macro, confusion matrix, inference time."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    preds, trues, t_total, n_total = [], [], 0.0, 0
    with torch.no_grad():
        for batch in loader:
            x = batch["image"] if "image" in batch else batch["mel"]
            y = batch["label"]
            x = x.to(device)
            t0 = time.perf_counter()
            logits = model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            t_total += (time.perf_counter() - t0) * 1000.0
            n_total += len(y)
            preds += logits.argmax(dim=-1).cpu().tolist()
            trues += y.tolist()
    return {
        "accuracy": float(accuracy_score(trues, preds)),
        "f1_macro": float(f1_score(trues, preds, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(trues, preds).tolist(),
        "inference_time_ms": float(t_total / max(1, n_total)),
    }


def evaluate_fusion(face_model, audio_model, loader_pairs, w_face=0.5, w_audio=0.5,
                    device: str | None = None) -> dict:
    """Đánh giá late fusion trên các cặp (mặt, tiếng) cùng nhãn.

    loader_pairs: iterable các tuple (batch_face, batch_audio) đã căn nhãn.
    """
    from src.models.fusion import fuse_probs
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    face_model, audio_model = face_model.to(device).eval(), audio_model.to(device).eval()
    preds, trues, t_total, n_total = [], [], 0.0, 0
    with torch.no_grad():
        for bf, ba in loader_pairs:
            xf, yf = bf["image"].to(device), bf["label"]
            xa, ya = ba["mel"].to(device), ba["label"]
            assert (yf == ya).all(), "Cặp face/audio lệch nhãn khi eval fusion"
            t0 = time.perf_counter()
            pf = F.softmax(face_model(xf), dim=-1)
            pa = F.softmax(audio_model(xa), dim=-1)
            fused = fuse_probs(pf, pa, w_face, w_audio)
            if device == "cuda":
                torch.cuda.synchronize()
            t_total += (time.perf_counter() - t0) * 1000.0
            n_total += len(yf)
            preds += fused.argmax(dim=-1).cpu().tolist()
            trues += yf.tolist()
    return {
        "accuracy": float(accuracy_score(trues, preds)),
        "f1_macro": float(f1_score(trues, preds, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(trues, preds).tolist(),
        "inference_time_ms": float(t_total / max(1, n_total)),
        "w_face": w_face,
        "w_audio": w_audio,
    }
