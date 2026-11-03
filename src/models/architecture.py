"""Loss và builder chung cho 2 nhánh."""
import torch.nn as nn


def build_loss(class_weights=None) -> nn.Module:
    """Cross-entropy có trọng số để xử lý mất cân bằng FER2013 (disgust rất ít)."""
    weight = None
    if class_weights is not None:
        import torch
        weight = torch.as_tensor(class_weights, dtype=torch.float32)
    return nn.CrossEntropyLoss(weight=weight)


def build_model(kind: str = "face", num_classes: int = 7, pretrained: bool = True, dropout: float = 0.3):
    """Hàm dựng chung: kind = face | audio."""
    if kind == "face":
        from .face_cnn import build_face_model
        return build_face_model(num_classes=num_classes, pretrained=pretrained)
    if kind == "audio":
        from .audio_cnn import build_audio_model
        return build_audio_model(num_classes=num_classes, dropout=dropout)
    raise ValueError(f"kind phải là face hoặc audio, nhận được: {kind}")
