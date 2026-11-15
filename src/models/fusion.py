"""Late fusion: gộp xác suất 2 nhánh mặt + tiếng."""
import torch
import torch.nn.functional as F


EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral",
            "contempt", "boredom", "excited"]

EMOTIONS_7 = EMOTIONS[:7]

EMOTIONS_VI = ["Giận dữ", "Ghê tởm", "Sợ hãi", "Vui vẻ", "Buồn", "Ngạc nhiên",
               "Trung tính", "Khinh bỉ", "Chán nản", "Hào hứng"]


def _align_probs(p_face, p_audio):
    """Căn 2 phân bố về cùng số lớp để fusion.

    Cả 2 nhánh đều 10 lớp (7 gốc + contempt/boredom/excited).
    Dùng `min` để tương thích checkpoint 7 lớp cũ: nếu 1 nhánh còn 7 lớp
    thì cắt cả 2 về 7 lớp chung đầu.
    """
    import torch
    pf = torch.as_tensor(p_face, dtype=torch.float32)
    pa = torch.as_tensor(p_audio, dtype=torch.float32)
    n = min(pf.shape[-1], pa.shape[-1])
    # 7 lớp đầu khớp nhau (ánh xạ ASVP 07 -> contempt riêng nên không trộn)
    pf = pf[..., :n]
    pa = pa[..., :n]
    pf = pf / pf.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    pa = pa / pa.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return pf, pa


def fuse_probs(p_face, p_audio, w_face: float = 0.5, w_audio: float = 0.5):
    """Gộp 2 phân bố mặt + tiếng thành 1 phân bố.

    Input: p_face [10] hoặc [B,10], p_audio [10] hoặc [B,10].
    Output: p_fused cùng số lớp đã căn, chuẩn hóa tổng bằng 1.
    Công thức: p_fused = w_face * p_face + w_audio * p_audio (sau khi căn về lớp chung).
    """
    pf, pa = _align_probs(p_face, p_audio)
    fused = w_face * pf + w_audio * pa
    fused = fused / fused.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return fused


def predict_fused(face_logits, audio_logits, w_face: float = 0.5, w_audio: float = 0.5):
    """Từ logits 2 nhánh ra nhãn và confidence sau fusion."""
    p_face = F.softmax(torch.as_tensor(face_logits, dtype=torch.float32), dim=-1)
    p_audio = F.softmax(torch.as_tensor(audio_logits, dtype=torch.float32), dim=-1)
    p_fused = fuse_probs(p_face, p_audio, w_face, w_audio)
    names = EMOTIONS[: p_fused.shape[-1]]
    conf, idx = p_fused.max(dim=-1)
    if p_fused.dim() == 1:
        return {"emotion": names[int(idx)], "confidence": float(conf), "probs_fused": p_fused.tolist()}
    return {"emotion": [names[int(i)] for i in idx.tolist()], "confidence": conf.tolist(), "probs_fused": p_fused.tolist()}
