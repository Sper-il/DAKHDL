"""Smoke test shape cho module models."""
import torch
from src.models.architecture import build_model
from src.models.fusion import fuse_probs


def test_fuse_probs_shape_and_norm():
    # Mat 10 lop, tieng 10 lop -> fusion 10 lop
    pf = torch.softmax(torch.randn(4, 10), dim=-1)
    pa = torch.softmax(torch.randn(4, 10), dim=-1)
    fused = fuse_probs(pf, pa, 0.5, 0.5)
    assert tuple(fused.shape) == (4, 10)
    assert torch.allclose(fused.sum(dim=-1), torch.ones(4), atol=1e-5)


def test_fuse_probs_backward_compat_7_classes():
    # Checkpoint 7 lop cu -> fusion phai cat ve 7 lop chung
    pf = torch.softmax(torch.randn(2, 7), dim=-1)
    pa = torch.softmax(torch.randn(2, 10), dim=-1)
    fused = fuse_probs(pf, pa, 0.5, 0.5)
    assert tuple(fused.shape) == (2, 7)
    assert torch.allclose(fused.sum(dim=-1), torch.ones(2), atol=1e-5)


def test_fuse_probs_same_shape():
    pf = torch.softmax(torch.randn(2, 10), dim=-1)
    pa = torch.softmax(torch.randn(2, 10), dim=-1)
    fused = fuse_probs(pf, pa, 0.5, 0.5)
    assert tuple(fused.shape) == (2, 10)


def test_face_cnn_forward_shape():
    model = build_model("face", num_classes=10, pretrained=False)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 1, 48, 48))
    assert tuple(out.shape) == (2, 10)


def test_audio_cnn_forward_shape():
    model = build_model("audio", num_classes=10)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 1, 128, 128))
    assert tuple(out.shape) == (2, 10)
