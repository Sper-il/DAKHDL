"""CLI đánh giá + sinh metrics.json, plots và report.html."""
from __future__ import annotations
import argparse
import json
from datetime import datetime
from pathlib import Path
import yaml


def main():
    parser = argparse.ArgumentParser(description="Eval emotion-fusion + visualize HTML")
    parser.add_argument("--config", default="configs/eval_config.yaml")
    parser.add_argument("--face-ckpt", default=None)
    parser.add_argument("--audio-ckpt", default=None)
    parser.add_argument("--face-test-csv", default=None)
    parser.add_argument("--audio-test-csv", default=None)
    parser.add_argument("--run-dir", default=None, help="Thư mục results/runs/<ts> để ghi kết quả")
    args = parser.parse_args()

    if not args.face_ckpt or not args.face_test_csv:
        print("[evaluation] thiếu --face-ckpt và --face-test-csv (tối thiểu nhánh mặt).")
        print("Ví dụ: python -m src.evaluation --face-ckpt models/checkpoints/face/best.pt "
              "--face-test-csv data/splits/face_test.csv")
        return

    import torch
    from src.data.dataloaders import build_face_loader, build_audio_loader
    from src.data.transforms import build_val_face_transform
    from src.models.architecture import build_model
    from src.evaluation.metrics import evaluate_branch
    from src.evaluation.visualize import plot_confusion_matrix
    from src.evaluation.visualize_html import build_report

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ts = datetime.now().strftime("%Y%m%d-%H%M")
    run_dir = Path(args.run_dir) if args.run_dir else Path("results/runs") / ts
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)

    branches = {}
    # Nhánh mặt
    face_model = build_model("face", num_classes=7, pretrained=False)
    face_model.load_state_dict(torch.load(args.face_ckpt, map_location="cpu")["model"]
                               if "model" in torch.load(args.face_ckpt, map_location="cpu")
                               else torch.load(args.face_ckpt, map_location="cpu"))
    face_loader = build_face_loader(args.face_test_csv, batch_size=64, shuffle=False,
                                    num_workers=0, transform=build_val_face_transform())
    m_face = evaluate_branch(face_model, face_loader)
    m_face["target"] = (cfg.get("targets", {}) or {}).get("accuracy_face", 0.65)
    m_face["passed"] = m_face["accuracy"] >= m_face["target"]
    branches["face"] = m_face
    plot_confusion_matrix(m_face["confusion_matrix"], run_dir / "plots" / "confusion_face.png",
                          title="Confusion matrix — nhánh mặt")

    # Nhánh tiếng (nếu có)
    if args.audio_ckpt and args.audio_test_csv:
        audio_model = build_model("audio", num_classes=7)
        ckpt = torch.load(args.audio_ckpt, map_location="cpu")
        audio_model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        audio_loader = build_audio_loader(args.audio_test_csv, batch_size=32,
                                          shuffle=False, num_workers=0)
        m_audio = evaluate_branch(audio_model, audio_loader)
        m_audio["target"] = (cfg.get("targets", {}) or {}).get("accuracy_audio", 0.70)
        m_audio["passed"] = m_audio["accuracy"] >= m_audio["target"]
        branches["audio"] = m_audio
        plot_confusion_matrix(m_audio["confusion_matrix"], run_dir / "plots" / "confusion_audio.png",
                              title="Confusion matrix — nhánh tiếng")

    out = {"time": ts, "branches": branches,
           "e2e_latency_ms": round(sum(b.get("inference_time_ms", 0) for b in branches.values()), 1),
           "error_analysis": "Xem confusion matrix per-class trong plots/ để biết lớp nào yếu nhất."}
    (run_dir / "metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    html = build_report(run_dir)
    print(f"[evaluation] metrics: {run_dir / 'metrics.json'}")
    print(f"[visualize] report: {html}")


if __name__ == "__main__":
    main()
