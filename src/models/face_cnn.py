"""Nhánh mặt: ResNet-18 grayscale bảo toàn thống kê Pretrained conv1/bn1 (bỏ maxpool -> giữ 3x3) + SE-Attention + LayerNorm + Multi-Sample Dropout."""
from __future__ import annotations
import torch
import torch.nn as nn
from torchvision import models


class SEBlock(nn.Module):
    """Squeeze-and-Excitation: tăng trọng số các kênh đặc trưng vùng mắt/miệng quan trọng."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(8, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * w


class FaceCNN(nn.Module):
    """Input [B,1,48,48] -> logits [B,7].

    Thiết kế đạt độ chính xác cao & chống overfitting:
    - Giữ nguyên bộ lọc conv1 7x7 pretrained của ImageNet (gộp trung bình 3 kênh về 1 kênh) để bảo toàn 100%
      phân phối kích hoạt của bn1 và layer1..layer4, nhưng thay maxpool bằng Identity khi stem_type='cifar'
      để bản đồ đặc trưng cuối giữ kích thước 3x3 thay vì bị co nhỏ.
    - SEBlock (Channel Attention) + LayerNorm + Multi-Sample Dropout (5 nhánh) giúp hội tụ mượt và chống overfit.
    """

    def __init__(
        self,
        num_classes: int = 7,
        pretrained: bool = True,
        dropout: float = 0.4,
        stem_type: str = "cifar",
    ):
        super().__init__()
        self.stem_type = stem_type
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)

        old = backbone.conv1
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            backbone.conv1.weight.copy_(old.weight.mean(dim=1, keepdim=True))
        if stem_type == "cifar":
            backbone.maxpool = nn.Identity()

        # Tăng nhẹ momentum của BatchNorm để thống kê running_mean/var cập nhật kịp trên tập dữ liệu vừa và nhỏ
        for m in backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = 0.25

        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.se = SEBlock(in_features, reduction=16)
        self.norm = nn.LayerNorm(in_features)
        self.dropouts = nn.ModuleList([nn.Dropout(p=dropout) for _ in range(5)])
        self.fc = nn.Linear(in_features, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.se(x)
        x = self.backbone.avgpool(x)
        return self.norm(torch.flatten(x, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(x)
        if self.training:
            return sum(self.fc(d(feat)) for d in self.dropouts) / len(self.dropouts)
        return self.fc(feat)

    def set_backbone_trainable(self, trainable: bool = True) -> None:
        """Đóng băng hoặc mở khóa các tầng sớm (conv1..layer2) khi fine-tune."""
        for name, param in self.backbone.named_parameters():
            if not (name.startswith("layer3") or name.startswith("layer4")):
                param.requires_grad = trainable

    def get_param_groups(self, lr: float, weight_decay: float, backbone_lr_mult: float = 0.35) -> list[dict]:
        """Chia nhóm tham số với learning rate phân tầng (backbone nhỏ hơn head)."""
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params = [
            p
            for p in (list(self.se.parameters()) + list(self.norm.parameters()) + list(self.fc.parameters()))
            if p.requires_grad
        ]
        return [
            {"params": backbone_params, "lr": lr * backbone_lr_mult, "weight_decay": weight_decay},
            {"params": head_params, "lr": lr, "weight_decay": weight_decay},
        ]


def build_face_model(
    num_classes: int = 7,
    pretrained: bool = True,
    dropout: float = 0.4,
    stem_type: str = "cifar",
) -> FaceCNN:
    return FaceCNN(num_classes=num_classes, pretrained=pretrained, dropout=dropout, stem_type=stem_type)
