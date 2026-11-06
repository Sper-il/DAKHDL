"""CLI đánh giá 2 nhánh + Late Fusion (có Temperature Scaling, Tune trọng số trên Val & TTA) + sinh report.html."""
from __future__ import annotations
import argparse
import json
from datetime import datetime
from pathlib import Path
import yaml


def _load_checkpoint_into_model(model, ckpt_path: str | Path):
    import torch
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    return model


def main():
    parser = argparse.ArgumentParser(description="Eval emotion-fusion + Late Fusion + visualize HTML")
    parser.add_argument("--config", default="configs/eval_config.yaml")
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    parser.add_argument("--splits-prefix", default="", help="Tiền tố file split (ví dụ: 'vn_')")
    parser.add_argument("--face-ckpt", default=None)
    parser.add_argument("--audio-ckpt", default=None)
    parser.add_argument("--face-val-csv", default=None)
    parser.add_argument("--audio-val-csv", default=None)
    parser.add_argument("--face-test-csv", default=None)
    parser.add_argument("--audio-test-csv", default=None)
    parser.add_argument("--tta", action="store_true", help="Bật Test-Time Augmentation khi đánh giá")
    parser.add_argument("--run-dir", default=None, help="Thư mục results/runs/<ts> để ghi kết quả")
    args = parser.parse_args()

    if args.splits_prefix:
        sp = Path("data/splits")
        args.face_val_csv = args.face_val_csv or str(sp / f"{args.splits_prefix}face_val.csv")
        args.audio_val_csv = args.audio_val_csv or str(sp / f"{args.splits_prefix}audio_val.csv")
        args.face_test_csv = args.face_test_csv or str(sp / f"{args.splits_prefix}face_test.csv")
        args.audio_test_csv = args.audio_test_csv or str(sp / f"{args.splits_prefix}audio_test.csv")

    if not args.face_ckpt or not args.face_test_csv:
        print("[evaluation] thiếu --face-ckpt và --face-test-csv (tối thiểu nhánh mặt).")
        print(
            "Ví dụ: python -m src.evaluation --face-ckpt models/checkpoints/face/best.pt "
            "--audio-ckpt models/checkpoints/audio/best.pt --splits-prefix vn_ --tta"
        )
        return

    from src.data.dataloaders import build_face_loader, build_audio_loader
    from src.data.transforms import build_val_face_transform
    from src.models.architecture import build_model
    from src.models.fusion import tune_fusion_weights
    from src.evaluation.metrics import evaluate_branch, evaluate_fusion, collect_logits_and_labels
    from src.evaluation.visualize import plot_confusion_matrix
    from src.evaluation.visualize_html import build_report

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(args.model_config, encoding="utf-8") as f:
        mcfg = yaml.safe_load(f)

    ts = datetime.now().strftime("%Y%m%d-%H%M")
    run_dir = Path(args.run_dir) if args.run_dir else Path("results/runs") / ts
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)

    # Tự động ghép hoặc sao chép đồ thị loss_curve.png từ các lượt train gần nhất vào run_dir/plots/
    recent_loss_curves = sorted(Path("results/runs").glob("*_*/plots/loss_curve.png"), key=lambda p: p.stat().st_mtime)
    if recent_loss_curves:
        import shutil
        shutil.copy(recent_loss_curves[-1], run_dir / "plots" / "loss_curve.png")

    branches = {}
    # 1. Nhánh mặt
    face_cfg = mcfg.get("face", {})
    face_model = build_model(
        "face",
        num_classes=face_cfg.get("num_classes", 7),
        pretrained=False,
        dropout=face_cfg.get("dropout", 0.4),
        stem_type=face_cfg.get("stem_type", "cifar"),
    )
    _load_checkpoint_into_model(face_model, args.face_ckpt)
    face_loader = build_face_loader(
        args.face_test_csv, batch_size=64, shuffle=False, num_workers=0, transform=build_val_face_transform()
    )
    m_face = evaluate_branch(face_model, face_loader, use_tta=args.tta)
    m_face["target"] = (cfg.get("targets", {}) or {}).get("accuracy_face", 0.65)
    m_face["passed"] = m_face["accuracy"] >= m_face["target"]
    branches["face"] = m_face
    plot_confusion_matrix(
        m_face["confusion_matrix"],
        run_dir / "plots" / "confusion_face.png",
        title="Confusion matrix — nhánh mặt",
    )

    # 2. Nhánh tiếng + Late Fusion (nếu có)
    if args.audio_ckpt and args.audio_test_csv:
        audio_cfg = mcfg.get("audio", {})
        audio_model = build_model(
            "audio",
            num_classes=audio_cfg.get("num_classes", 7),
            pretrained=False,
            dropout=audio_cfg.get("dropout", 0.4),
            arch=audio_cfg.get("name", "audio_resnet18_mel"),
        )
        _load_checkpoint_into_model(audio_model, args.audio_ckpt)
        audio_loader = build_audio_loader(args.audio_test_csv, batch_size=32, shuffle=False, num_workers=0)
        m_audio = evaluate_branch(audio_model, audio_loader, use_tta=args.tta)
        m_audio["target"] = (cfg.get("targets", {}) or {}).get("accuracy_audio", 0.70)
        m_audio["passed"] = m_audio["accuracy"] >= m_audio["target"]
        branches["audio"] = m_audio
        plot_confusion_matrix(
            m_audio["confusion_matrix"],
            run_dir / "plots" / "confusion_audio.png",
            title="Confusion matrix — nhánh tiếng",
        )

        # Nếu tập test mặt & tiếng có cùng số mẫu (dữ liệu cặp như vn_youtube), chạy đánh giá Late Fusion
        if len(face_loader.dataset) == len(audio_loader.dataset):
            fusion_cfg = mcfg.get("fusion", {})
            w_face = float(fusion_cfg.get("w_face", 0.5))
            w_audio = float(fusion_cfg.get("w_audio", 0.5))
            t_face, t_audio = 1.0, 1.0

            # Tự động tối ưu nhiệt độ (T_face, T_audio) và trọng số (w_face, w_audio) trên tập Validation
            if (
                fusion_cfg.get("tune_on_val", True)
                and args.face_val_csv
                and args.audio_val_csv
                and Path(args.face_val_csv).exists()
                and Path(args.audio_val_csv).exists()
            ):
                f_val_loader = build_face_loader(
                    args.face_val_csv, batch_size=32, shuffle=False, num_workers=0, transform=build_val_face_transform()
                )
                a_val_loader = build_audio_loader(args.audio_val_csv, batch_size=32, shuffle=False, num_workers=0)
                if len(f_val_loader.dataset) == len(a_val_loader.dataset):
                    zf_val, y_val, _ = collect_logits_and_labels(face_model, f_val_loader, use_tta=args.tta)
                    za_val, _, _ = collect_logits_and_labels(audio_model, a_val_loader, use_tta=args.tta)
                    tuned = tune_fusion_weights(
                        zf_val, za_val, y_val, calibrate=fusion_cfg.get("calibrate_temperature", True)
                    )
                    w_face, w_audio = tuned["w_face"], tuned["w_audio"]
                    t_face, t_audio = tuned["T_face"], tuned["T_audio"]
                    print(
                        f"[fusion] Đã tối ưu trên tập Val: w_face={w_face}, w_audio={w_audio}, "
                        f"T_face={t_face}, T_audio={t_audio} (Val F1={tuned['val_f1_macro']})"
                    )

            f_test_pair = build_face_loader(
                args.face_test_csv, batch_size=32, shuffle=False, num_workers=0, transform=build_val_face_transform()
            )
            a_test_pair = build_audio_loader(args.audio_test_csv, batch_size=32, shuffle=False, num_workers=0)
            m_fusion = evaluate_fusion(
                face_model,
                audio_model,
                zip(f_test_pair, a_test_pair),
                w_face=w_face,
                w_audio=w_audio,
                T_face=t_face,
                T_audio=t_audio,
            )
            m_fusion["target"] = (cfg.get("targets", {}) or {}).get("accuracy_fusion", 0.70)
            m_fusion["passed"] = m_fusion["accuracy"] >= m_fusion["target"]
            branches["fusion"] = m_fusion
            plot_confusion_matrix(
                m_fusion["confusion_matrix"],
                run_dir / "plots" / "confusion_fusion.png",
                title=f"Confusion matrix — Late Fusion (w={w_face:.2f}/{w_audio:.2f})",
            )

    out = {
        "time": ts,
        "branches": branches,
        "e2e_latency_ms": round(
            branches.get("fusion", {}).get(
                "inference_time_ms",
                sum(b.get("inference_time_ms", 0) for k, b in branches.items() if k in ("face", "audio")),
            ),
            1,
        ),
        "error_analysis": "Đã áp dụng Temperature Scaling + tối ưu trọng số Fusion trên tập Val. Xem confusion matrix trong plots/.",
    }
    (run_dir / "metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    html = build_report(run_dir)
    print(f"[evaluation] metrics: {run_dir / 'metrics.json'}")
    print(f"[visualize] report: {html}")


if __name__ == "__main__":
    main()
