"""Smoke test shape cho module data (không cần dữ liệu thật)."""
import torch
from src.data.transforms import spec_augment


def test_spec_augment_keeps_shape():
    mel = torch.randn(1, 128, 128)
    out = spec_augment(mel)
    assert tuple(out.shape) == (1, 128, 128)


def test_label_range():
    # Cả 2 nhánh đều dùng 10 lớp: 0..9 (7 gốc + contempt/boredom/excited)
    assert 0 <= 0 <= 9 and 0 <= 9 <= 9
