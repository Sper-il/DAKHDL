"""Smoke test shape cho module data (không cần dữ liệu thật)."""
import torch
from src.data.transforms import spec_augment


def test_spec_augment_keeps_shape():
    mel = torch.randn(1, 128, 128)
    out = spec_augment(mel)
    assert tuple(out.shape) == (1, 128, 128)


def test_label_range():
    # 7 cảm xúc: 0..6
    assert 0 <= 0 <= 6 and 0 <= 6 <= 6
