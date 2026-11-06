"""Smoke test shape cho module models (FaceCNN stem 3x3, AudioResNetMel, AudioCNN, Fusion & Calibration)."""
import torch
from src.models.architecture import build_model, build_loss, compute_smoothed_class_weights
from src.models.fusion import fuse_probs, tune_fusion_weights


def test_fuse_probs_shape_and_norm():
    pf = torch.softmax(torch.randn(4, 7), dim=-1)
    pa = torch.softmax(torch.randn(4, 7), dim=-1)
    fused = fuse_probs(pf, pa, 0.5, 0.5)
    assert tuple(fused.shape) == (4, 7)
    assert torch.allclose(fused.sum(dim=-1), torch.ones(4), atol=1e-5)


def test_face_and_audio_models_forward():
    face_model = build_model("face", pretrained=False, stem_type="cifar")
    x_face = torch.randn(2, 1, 48, 48)
    out_face = face_model(x_face)
    assert tuple(out_face.shape) == (2, 7)

    audio_res = build_model("audio", pretrained=False, arch="audio_resnet18_mel")
    x_mel = torch.randn(2, 1, 128, 128)
    out_audio = audio_res(x_mel)
    assert tuple(out_audio.shape) == (2, 7)


def test_tune_fusion_weights_and_loss():
    zf = torch.randn(14, 7)
    za = torch.randn(14, 7)
    y = [i % 7 for i in range(14)]
    tuned = tune_fusion_weights(zf, za, y, calibrate=True)
    assert 0.0 <= tuned["w_face"] <= 1.0
    assert abs(tuned["w_face"] + tuned["w_audio"] - 1.0) < 1e-5

    cw = compute_smoothed_class_weights(y, num_classes=7)
    criterion = build_loss(class_weights=cw, label_smoothing=0.1)
    loss = criterion(zf, torch.tensor(y, dtype=torch.long))
    assert float(loss.item()) > 0.0
