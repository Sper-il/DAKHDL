"""Late fusion: gộp xác suất 2 nhánh mặt + tiếng."""
import torch
import torch.nn.functional as F


EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def fuse_probs(p_face, p_audio, w_face: float = 0.5, w_audio: float = 0.5):
    """Gộp 2 phân bố 7 lớp thành 1 phân bố.

    Input: p_face [7] hoặc [B,7], p_audio tương tự. Output: p_fused chuẩn hóa tổng bằng 1.
    Công thức: p_fused = w_face * p_face + w_audio * p_audio.
    """
    pf = torch.as_tensor(p_face, dtype=torch.float32)
    pa = torch.as_tensor(p_audio, dtype=torch.float32)
    fused = w_face * pf + w_audio * pa
    fused = fused / fused.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return fused


def predict_fused(face_logits, audio_logits, w_face: float = 0.5, w_audio: float = 0.5):
    """Từ logits 2 nhánh ra nhãn và confidence sau fusion."""
    p_face = F.softmax(torch.as_tensor(face_logits, dtype=torch.float32), dim=-1)
    p_audio = F.softmax(torch.as_tensor(audio_logits, dtype=torch.float32), dim=-1)
    p_fused = fuse_probs(p_face, p_audio, w_face, w_audio)
    conf, idx = p_fused.max(dim=-1)
    if p_fused.dim() == 1:
        return {"emotion": EMOTIONS[int(idx)], "confidence": float(conf), "probs_fused": p_fused.tolist()}
    return {"emotion": [EMOTIONS[int(i)] for i in idx.tolist()], "confidence": conf.tolist(), "probs_fused": p_fused.tolist()}
