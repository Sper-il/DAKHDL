"""Verify nhanh module models: forward 1 batch giả, kiểm tra shape."""
import torch


def main():
    from .architecture import build_model, build_loss
    from .fusion import fuse_probs

    face = build_model("face", num_classes=7, pretrained=False)
    audio = build_model("audio", num_classes=7)
    face.eval()
    audio.eval()
    with torch.no_grad():
        f_out = face(torch.randn(2, 1, 48, 48))
        a_out = audio(torch.randn(2, 1, 128, 128))
    print(f"[face] logits: {tuple(f_out.shape)} (ky vong [2,7])")
    print(f"[audio] logits: {tuple(a_out.shape)} (ky vong [2,7])")
    pf = torch.softmax(f_out, dim=-1)
    pa = torch.softmax(a_out, dim=-1)
    fused = fuse_probs(pf, pa, 0.5, 0.5)
    print(f"[fusion] probs_fused: {tuple(fused.shape)} (ky vong [2,7])")
    loss_fn = build_loss()
    loss = loss_fn(f_out, torch.tensor([3, 4]))
    print(f"[loss] CrossEntropy chay duoc: {float(loss):.4f}")
    n_face = sum(p.numel() for p in face.parameters()) / 1e6
    n_audio = sum(p.numel() for p in audio.parameters()) / 1e6
    print(f"[params] face ~{n_face:.2f}M, audio ~{n_audio:.2f}M")
    print("[models] done.")


if __name__ == "__main__":
    main()
