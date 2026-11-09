"""Smoke test shape cho module models."""
import torch
from src.models.fusion import fuse_probs


def test_fuse_probs_shape_and_norm():
    # Mat 7 lop, tieng 10 lop -> fusion can ve 7 lop chung
    pf = torch.softmax(torch.randn(4, 7), dim=-1)
    pa = torch.softmax(torch.randn(4, 10), dim=-1)
    fused = fuse_probs(pf, pa, 0.5, 0.5)
    assert tuple(fused.shape) == (4, 7)
    assert torch.allclose(fused.sum(dim=-1), torch.ones(4), atol=1e-5)


def test_fuse_probs_same_shape():
    pf = torch.softmax(torch.randn(2, 7), dim=-1)
    pa = torch.softmax(torch.randn(2, 7), dim=-1)
    fused = fuse_probs(pf, pa, 0.5, 0.5)
    assert tuple(fused.shape) == (2, 7)
