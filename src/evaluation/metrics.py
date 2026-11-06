"""Metric đánh giá 2 lớp: model level (hỗ trợ Test-Time Augmentation) + full pipeline Late Fusion."""
from __future__ import annotations
import time
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


@torch.no_grad()
def collect_logits_and_labels(model, loader, device: str | None = None, use_tta: bool = False) -> tuple[torch.Tensor, list[int], float]:
    """Thu thập toàn bộ logits, nhãn thật và thời gian suy luận trung bình (ms/mẫu)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    all_logits = []
    trues: list[int] = []
    t_total = 0.0
    n_total = 0

    for batch in loader:
        is_face = "image" in batch
        x = batch["image"] if is_face else batch["mel"]
        y = batch["label"]
        x = x.to(device)
        t0 = time.perf_counter()
        logits = model(x)
        if use_tta:
            if is_face:
                # TTA cho ảnh mặt: trung bình cộng với ảnh lật ngang
                logits_flip = model(torch.flip(x, dims=[-1]))
                logits = 0.5 * (logits + logits_flip)
            else:
                # TTA cho âm thanh: dịch nhẹ có replicate biên (tránh wrap-around của torch.roll) + biến thiên biên độ nhẹ
                pad_l = F.pad(x[:, :, :, :-2], (2, 0, 0, 0), mode="replicate")
                pad_r = F.pad(x[:, :, :, 2:], (0, 2, 0, 0), mode="replicate")
                logits_s1 = model(pad_l)
                logits_s2 = model(pad_r)
                logits = 0.6 * logits + 0.2 * logits_s1 + 0.2 * logits_s2
        if device == "cuda":
            torch.cuda.synchronize()
        t_total += (time.perf_counter() - t0) * 1000.0
        n_total += len(y)
        all_logits.append(logits.cpu())
        trues += y.tolist()

    cat_logits = torch.cat(all_logits, dim=0) if all_logits else torch.empty((0, 7))
    avg_ms = float(t_total / max(1, n_total))
    return cat_logits, trues, avg_ms


def evaluate_branch(model, loader, device: str | None = None, use_tta: bool = False) -> dict:
    """Đánh giá 1 nhánh: accuracy, F1 macro, confusion matrix, inference time."""
    logits, trues, avg_ms = collect_logits_and_labels(model, loader, device=device, use_tta=use_tta)
    preds = logits.argmax(dim=-1).tolist()
    return {
        "accuracy": round(float(accuracy_score(trues, preds)), 4),
        "f1_macro": round(float(f1_score(trues, preds, average="macro", zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(trues, preds, labels=list(range(7))).tolist(),
        "inference_time_ms": round(avg_ms, 2),
    }


def evaluate_fusion(
    face_model,
    audio_model,
    loader_pairs,
    w_face: float = 0.5,
    w_audio: float = 0.5,
    T_face: float = 1.0,
    T_audio: float = 1.0,
    device: str | None = None,
) -> dict:
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
            pf = F.softmax(face_model(xf) / max(T_face, 1e-3), dim=-1)
            pa = F.softmax(audio_model(xa) / max(T_audio, 1e-3), dim=-1)
            fused = fuse_probs(pf, pa, w_face, w_audio)
            if device == "cuda":
                torch.cuda.synchronize()
            t_total += (time.perf_counter() - t0) * 1000.0
            n_total += len(yf)
            preds += fused.argmax(dim=-1).cpu().tolist()
            trues += yf.tolist()
    return {
        "accuracy": round(float(accuracy_score(trues, preds)), 4),
        "f1_macro": round(float(f1_score(trues, preds, average="macro", zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(trues, preds, labels=list(range(7))).tolist(),
        "inference_time_ms": round(float(t_total / max(1, n_total)), 2),
        "w_face": w_face,
        "w_audio": w_audio,
        "T_face": T_face,
        "T_audio": T_audio,
    }
