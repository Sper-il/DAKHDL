"""Nhánh mặt: ResNet-18 grayscale cho 7 cảm xúc."""
import torch.nn as nn
from torchvision import models


class FaceCNN(nn.Module):
    """Input [B,1,48,48] -> logits [B,7]."""

    def __init__(self, num_classes: int = 7, pretrained: bool = True):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        # Thích ứng ảnh xám: conv1 từ 3 kênh về 1 kênh, giữ trọng số trung bình
        old = backbone.conv1
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with __import__("torch").no_grad():
            backbone.conv1.weight.copy_(old.weight.mean(dim=1, keepdim=True))
        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)


def build_face_model(num_classes: int = 7, pretrained: bool = True) -> FaceCNN:
    return FaceCNN(num_classes=num_classes, pretrained=pretrained)
