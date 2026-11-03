"""Nhánh tiếng: CNN 4 khối cho Mel 128x128."""
import torch.nn as nn


class AudioCNN(nn.Module):
    """Input [B,1,128,128] -> logits [B,7]."""

    def __init__(self, num_classes: int = 7, dropout: float = 0.3):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
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
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


def build_audio_model(num_classes: int = 7, dropout: float = 0.3) -> AudioCNN:
    return AudioCNN(num_classes=num_classes, dropout=dropout)
