"""Loss và builder chung cho 2 nhánh (hỗ trợ Class Weights, Label Smoothing và chọn kiến trúc)."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


def compute_smoothed_class_weights(labels: list[int] | np.ndarray, num_classes: int = 7, power: float = 0.5) -> torch.Tensor:
    """Tính trọng số lớp nghịch đảo có làm mượt (power=0.5) để chống mất cân bằng mà không gây nổ gradient."""
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    total = float(counts.sum())
    weights = (total / (num_classes * counts)) ** power
    weights = weights / weights.mean()
    return torch.as_tensor(weights, dtype=torch.float32)


def build_loss(class_weights=None, label_smoothing: float = 0.1) -> nn.Module:
    """Cross-entropy có trọng số và label smoothing để xử lý mất cân bằng & chống overconfidence."""
    weight = None
    if class_weights is not None:
        weight = torch.as_tensor(class_weights, dtype=torch.float32)
    return nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)


def build_model(
    kind: str = "face",
    num_classes: int = 7,
    pretrained: bool = True,
    dropout: float = 0.35,
    arch: str | None = None,
    stem_type: str = "cifar",
) -> nn.Module:
    """Hàm dựng chung: kind = face | audio."""
    if kind == "face":
        from .face_cnn import build_face_model
        return build_face_model(num_classes=num_classes, pretrained=pretrained, dropout=dropout, stem_type=stem_type)
    if kind == "audio":
        from .audio_cnn import build_audio_model
        return build_audio_model(
            num_classes=num_classes,
            dropout=dropout,
            arch=arch or "audio_cnn_4blocks",
            pretrained=pretrained,
        )
    raise ValueError(f"kind phải là face hoặc audio, nhận được: {kind}")
