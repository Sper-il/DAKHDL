"""Demo 1 lệnh: ảnh mặt + wav tiếng -> JSON emotion + index.html rút gọn."""
from __future__ import annotations
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
import yaml


def load_face_tensor(path: str):
    img = Image.open(path).convert("L").resize((48, 48))
    return T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])(img).unsqueeze(0)


def load_mel_tensor(path: str):
    import librosa
    import numpy as np
    y, _ = librosa.load(path, sr=16000, mono=True, duration=4.0)
    target = 16000 * 4
    y = np.pad(y, (0, max(0, target - len(y))))[:target]
    mel = librosa.feature.melspectrogram(y=y, sr=16000, n_mels=128)
    mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    if mel.shape[1] < 128:
        mel = np.pad(mel, ((0, 0), (0, 128 - mel.shape[1])))
    else:
        mel = mel[:, :128]
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    return torch.from_numpy(mel).unsqueeze(0).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser(description="Demo emotion-fusion 1 lệnh")
    parser.add_argument("--image", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--face-ckpt", required=True)
    parser.add_argument("--audio-ckpt", default=None)
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    parser.add_argument("--w-face", type=float, default=0.5)
    parser.add_argument("--w-audio", type=float, default=0.5)
    args = parser.parse_args()

    from src.models.architecture import build_model
    from src.models.fusion import fuse_probs, EMOTIONS

    mcfg = {}
    if Path(args.model_config).exists():
        with open(args.model_config, encoding="utf-8") as f:
            mcfg = yaml.safe_load(f) or {}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    face_cfg = mcfg.get("face", {})
    face_model = build_model(
        "face",
        pretrained=False,
        stem_type=face_cfg.get("stem_type", "cifar"),
    ).to(device).eval()
    ckpt = torch.load(args.face_ckpt, map_location=device)
    face_model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=False)

    t0 = time.perf_counter()
    with torch.no_grad():
        x_face = load_face_tensor(args.image).to(device)
        # TTA lật ngang nhẹ cho demo
        logits_f = 0.5 * (face_model(x_face) + face_model(torch.flip(x_face, dims=[-1])))
        pf = F.softmax(logits_f, dim=-1)
        if args.audio_ckpt:
            audio_cfg = mcfg.get("audio", {})
            audio_model = build_model(
                "audio",
                pretrained=False,
                arch=audio_cfg.get("name", "audio_resnet18_mel"),
            ).to(device).eval()
            ackpt = torch.load(args.audio_ckpt, map_location=device)
            audio_model.load_state_dict(ackpt["model"] if "model" in ackpt else ackpt, strict=False)
            pa = F.softmax(audio_model(load_mel_tensor(args.audio).to(device)), dim=-1)
            fused = fuse_probs(pf, pa, args.w_face, args.w_audio)
        else:
            fused = pf
        conf, idx = fused[0].max(dim=-1)
    e2e = (time.perf_counter() - t0) * 1000.0
    result = {
        "emotion": EMOTIONS[int(idx)],
        "confidence": round(float(conf), 4),
        "probs_fused": [round(float(v), 4) for v in fused[0].tolist()],
        "inference_time_ms": round(e2e, 1),
    }

    ts = datetime.now().strftime("%Y%m%d-%H%M")
    out_dir = Path("results/demo") / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "index.html").write_text(
        f"""<!DOCTYPE html><html lang="vi"><head><meta charset="utf-8">
<title>Demo emotion-fusion — {ts}</title></head><body>
<h1>Kết quả demo — {result['emotion']} ({result['confidence']})</h1>
<p>Ảnh: {args.image} | Audio: {args.audio}</p>
<p>Latency: {result['inference_time_ms']} ms</p>
<pre>{json.dumps(result, ensure_ascii=False, indent=2)}</pre>
</body></html>""",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[demo] lưu tại {out_dir}/result.json và index.html")


if __name__ == "__main__":
    main()
