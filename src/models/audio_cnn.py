"""Nhánh tiếng: hỗ trợ ResNet-18 Pretrained trên 3 kênh [Mel, ΔMel, ΔΔMel] + Attentive Pooling + Multi-Sample Dropout."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class AttentiveStatsPooling(nn.Module):
    """Gộp đặc trưng theo trục thời gian bằng Attention (Mean + Std có trọng số), rất mạnh cho SER."""

    def __init__(self, in_dim: int, bottleneck: int = 128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(in_dim, bottleneck, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(bottleneck, in_dim, kernel_size=1),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        w = self.attn(x)  # [B, C, T]
        mean = torch.sum(w * x, dim=-1)
        var = torch.sum(w * ((x - mean.unsqueeze(-1)) ** 2), dim=-1).clamp_min(1e-6)
        std = torch.sqrt(var)
        return torch.cat([mean, std], dim=1)  # [B, 2*C]


def compute_delta_channels(mel: torch.Tensor) -> torch.Tensor:
    """Từ Mel [B, 1, F, T] tính thêm ΔMel và ΔΔMel dọc trục thời gian -> [B, 3, F, T]."""
    padded = F.pad(mel, (1, 1, 0, 0), mode="replicate")
    delta1 = (padded[:, :, :, 2:] - padded[:, :, :, :-2]) * 0.5
    padded2 = F.pad(delta1, (1, 1, 0, 0), mode="replicate")
    delta2 = (padded2[:, :, :, 2:] - padded2[:, :, :, :-2]) * 0.5
    return torch.cat([mel, delta1, delta2], dim=1)


class AudioResNetMel(nn.Module):
    """Input [B,1,128,128] -> tự tạo 3 kênh [Mel, Δ, ΔΔ] -> ResNet-18 pretrained + AttentiveStatsPooling -> [B,7]."""

    def __init__(self, num_classes: int = 7, pretrained: bool = True, dropout: float = 0.4):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        for m in backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = 0.25
        in_features = backbone.fc.in_features  # 512
        backbone.maxpool = nn.Identity()
        backbone.fc = nn.Identity()
        backbone.avgpool = nn.Identity()
        self.backbone = backbone
        self.pool = AttentiveStatsPooling(in_features, bottleneck=128)
        self.head = nn.Sequential(
            nn.LayerNorm(in_features * 2),
            nn.Dropout(dropout),
            nn.Linear(in_features * 2, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = compute_delta_channels(x)
        b = self.backbone
        x = b.conv1(x)
        x = b.bn1(x)
        x = b.relu(x)
        x = b.maxpool(x)
        x = b.layer1(x)
        x = b.layer2(x)
        x = b.layer3(x)
        x = b.layer4(x)  # [B, 512, F'=4, T'=4]
        x = x.mean(dim=2)  # Gộp trục tần số -> [B, 512, T']
        return self.pool(x)  # [B, 1024]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))

    def set_backbone_trainable(self, trainable: bool = True) -> None:
        for name, param in self.backbone.named_parameters():
            if not (name.startswith("layer3") or name.startswith("layer4")):
                param.requires_grad = trainable

    def get_param_groups(self, lr: float, weight_decay: float, backbone_lr_mult: float = 0.35) -> list[dict]:
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params = [p for p in list(self.pool.parameters()) + list(self.head.parameters()) if p.requires_grad]
        return [
            {"params": backbone_params, "lr": lr * backbone_lr_mult, "weight_decay": weight_decay},
            {"params": head_params, "lr": lr, "weight_decay": weight_decay},
        ]


class AudioCNN(nn.Module):
    """Input [B,1,128,128] -> logits [B,7] (CNN 4 khối có Dropout chống overfit)."""

    def __init__(self, num_classes: int = 7, dropout: float = 0.35):
        super().__init__()

        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout, momentum=0.25),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout, momentum=0.25),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(1, 32),    # 64x64
            block(32, 64),   # 32x32
            block(64, 128),  # 16x16
            block(128, 256), # 8x8
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(256),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))

    def set_backbone_trainable(self, trainable: bool = True) -> None:
        for idx, layer in enumerate(self.features):
            if idx < 2:
                for p in layer.parameters():
                    p.requires_grad = trainable

    def get_param_groups(self, lr: float, weight_decay: float, backbone_lr_mult: float = 0.35) -> list[dict]:
        backbone_params = [p for p in self.features.parameters() if p.requires_grad]
        head_params = [p for p in self.head.parameters() if p.requires_grad]
        return [
            {"params": backbone_params, "lr": lr * backbone_lr_mult, "weight_decay": weight_decay},
            {"params": head_params, "lr": lr, "weight_decay": weight_decay},
        ]


def build_audio_model(
    num_classes: int = 7,
    dropout: float = 0.35,
    arch: str = "audio_resnet18_mel",
    pretrained: bool = True,
) -> nn.Module:
    if arch in ("audio_resnet18_mel", "resnet18_mel"):
        return AudioResNetMel(num_classes=num_classes, pretrained=pretrained, dropout=dropout)
    return AudioCNN(num_classes=num_classes, dropout=dropout)
