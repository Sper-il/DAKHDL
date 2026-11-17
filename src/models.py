"""Hai mô hình nhẹ, dễ giải thích: FaceCNN (ResNet-18 xám) và AudioCNN (CNN 4 khối trên Mel)."""
from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tvm


class FaceCNN(nn.Module):
    """Input [B,1,48,48] -> logits [B,num_classes]."""

    def __init__(self, num_classes=7, pretrained=True, remove_maxpool=False, input_size=48, dropout=0.0):
        super().__init__()
        backbone, loaded = None, False
        if pretrained:
            try:
                backbone = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)
                loaded = True
            except Exception as e:  # không có mạng / không tải được trọng số
                warnings.warn(f"Không tải được trọng số ImageNet ({type(e).__name__}); dùng khởi tạo ngẫu nhiên.")
        if backbone is None:
            backbone = tvm.resnet18(weights=None)
        self.pretrained_loaded = loaded
        old = backbone.conv1
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():  # ảnh xám: lấy trung bình trọng số 3 kênh màu
            backbone.conv1.weight.copy_(old.weight.mean(dim=1, keepdim=True))
        if remove_maxpool:
            backbone.maxpool = nn.Identity()
        in_f = backbone.fc.in_features
        backbone.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes)) if dropout > 0 \
            else nn.Linear(in_f, num_classes)
        self.backbone = backbone
        self.input_size = int(input_size)

    def forward(self, x):
        if x.shape[-1] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        return self.backbone(x)


class AudioCNN(nn.Module):
    """Input [B,1,128,128] -> logits [B,num_classes]; 4 khối (Conv-BN-ReLU)x2 + MaxPool."""

    def __init__(self, num_classes=7, dropout=0.3):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.MaxPool2d(2))

        self.features = nn.Sequential(block(1, 32), block(32, 64), block(64, 128), block(128, 256),
                                      nn.AdaptiveAvgPool2d(1))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(256, 128), nn.ReLU(inplace=True),
                                  nn.Dropout(dropout), nn.Linear(128, num_classes))

    def forward(self, x):
        return self.head(self.features(x))


def build_model(cfg: dict) -> nn.Module:
    n, m = int(cfg["num_classes"]), cfg["model"]
    if cfg["branch"] == "face":
        return FaceCNN(n, pretrained=bool(m["pretrained"]), remove_maxpool=bool(m["remove_maxpool"]),
                       input_size=int(m["input_size"]), dropout=float(m["dropout"]))
    return AudioCNN(n, dropout=float(m["dropout"]))
