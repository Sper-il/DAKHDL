"""Late fusion: gộp xác suất 2 nhánh mặt + tiếng, hỗ trợ Temperature Scaling và tối ưu trọng số có điều chuẩn trên tập Val."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score


EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def fuse_probs(p_face, p_audio, w_face: float = 0.5, w_audio: float = 0.5) -> torch.Tensor:
    """Gộp 2 phân bố 7 lớp thành 1 phân bố.

    Input: p_face [7] hoặc [B,7], p_audio tương tự. Output: p_fused chuẩn hóa tổng bằng 1.
    Công thức: p_fused = w_face * p_face + w_audio * p_audio.
    """
    pf = torch.as_tensor(p_face, dtype=torch.float32)
    pa = torch.as_tensor(p_audio, dtype=torch.float32)
    fused = w_face * pf + w_audio * pa
    fused = fused / fused.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return fused


def predict_fused(
    face_logits,
    audio_logits,
    w_face: float = 0.5,
    w_audio: float = 0.5,
    T_face: float = 1.0,
    T_audio: float = 1.0,
) -> dict:
    """Từ logits 2 nhánh (có hiệu chỉnh nhiệt độ T_face, T_audio) ra nhãn và confidence sau fusion."""
    lf = torch.as_tensor(face_logits, dtype=torch.float32) / max(T_face, 1e-3)
    la = torch.as_tensor(audio_logits, dtype=torch.float32) / max(T_audio, 1e-3)
    p_face = F.softmax(lf, dim=-1)
    p_audio = F.softmax(la, dim=-1)
    p_fused = fuse_probs(p_face, p_audio, w_face, w_audio)
    conf, idx = p_fused.max(dim=-1)
    if p_fused.dim() == 1:
        return {"emotion": EMOTIONS[int(idx)], "confidence": float(conf), "probs_fused": p_fused.tolist()}
    return {
        "emotion": [EMOTIONS[int(i)] for i in idx.tolist()],
        "confidence": conf.tolist(),
        "probs_fused": p_fused.tolist(),
    }


def tune_temperature(logits: torch.Tensor | np.ndarray, labels: torch.Tensor | np.ndarray | list[int]) -> float:
    """Tìm nhiệt độ T ∈ [0.7, 2.0] tối ưu trên tập Validation để hiệu chỉnh độ tự tin của từng nhánh."""
    z = torch.as_tensor(logits, dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.long)
    best_t = 1.0
    best_nll = float("inf")
    for t in np.linspace(0.7, 2.0, 27):
        nll = float(F.cross_entropy(z / float(t), y).item())
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)
    return round(best_t, 3)


def tune_fusion_weights(
    face_logits: torch.Tensor | np.ndarray,
    audio_logits: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray | list[int],
    calibrate: bool = True,
) -> dict[str, float]:
    """Tìm trọng số (w_face, w_audio) trong khoảng an toàn [0.15, 0.85] có điều chuẩn theo F1 từng nhánh đơn lẻ."""
    zf = torch.as_tensor(face_logits, dtype=torch.float32)
    za = torch.as_tensor(audio_logits, dtype=torch.float32)
    y_true = np.asarray(labels, dtype=np.int64)

    t_face = tune_temperature(zf, y_true) if calibrate else 1.0
    t_audio = tune_temperature(za, y_true) if calibrate else 1.0

    pf = F.softmax(zf / t_face, dim=-1)
    pa = F.softmax(za / t_audio, dim=-1)

    # Điểm F1 đơn nhánh để làm prior chống suy biến về 1 nhánh yếu
    f1_f = float(f1_score(y_true, pf.argmax(dim=-1).cpu().numpy(), average="macro", zero_division=0))
    f1_a = float(f1_score(y_true, pa.argmax(dim=-1).cpu().numpy(), average="macro", zero_division=0))
    prior_wf = (f1_f + 0.05) / (f1_f + f1_a + 0.10)
    prior_wf = float(np.clip(prior_wf, 0.2, 0.8))

    best_w = prior_wf
    best_score = -1.0
    best_f1 = 0.0
    best_acc = 0.0

    for w in np.linspace(0.15, 0.85, 36):
        wf = float(w)
        wa = 1.0 - wf
        fused = fuse_probs(pf, pa, wf, wa)
        preds = fused.argmax(dim=-1).cpu().numpy()
        f1 = float(f1_score(y_true, preds, average="macro", zero_division=0))
        acc = float(accuracy_score(y_true, preds))
        # Điều chuẩn nhẹ về prior_wf để tránh chọn trọng số cực đoan khi tập Val nhỏ
        reg_penalty = 0.03 * abs(wf - prior_wf)
        score = 0.6 * f1 + 0.4 * acc - reg_penalty
        if score > best_score:
            best_score = score
            best_w = wf
            best_f1 = f1
            best_acc = acc

    return {
        "w_face": round(best_w, 3),
        "w_audio": round(1.0 - best_w, 3),
        "T_face": t_face,
        "T_audio": t_audio,
        "val_f1_macro": round(best_f1, 4),
        "val_accuracy": round(best_acc, 4),
    }
