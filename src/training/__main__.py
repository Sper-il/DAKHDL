"""CLI train 1 nhánh: face hoặc audio."""
from __future__ import annotations
import argparse
import logging
from datetime import datetime
from pathlib import Path
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train emotion-fusion 1 nhánh")
    parser.add_argument("--branch", choices=["face", "audio"], required=True)
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--data-config", default="configs/data_config.yaml")
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    parser.add_argument("--train-csv", default=None, help="Đường dẫn face_train.csv / audio_train.csv")
    parser.add_argument("--val-csv", default=None, help="Đường dẫn face_val.csv / audio_val.csv")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--sanity", action="store_true", help="Chỉ overfit 1 batch nhỏ để kiểm tra")
    args = parser.parse_args()

    if not args.train_csv or not args.val_csv:
        print("[training] thiếu --train-csv và --val-csv.")
        print("Ví dụ: python -m src.training --branch face "
              "--train-csv data/splits/face_train.csv --val-csv data/splits/face_val.csv")
        print("Hãy tải FER2013 + RAVDESS và chạy split trước (xem docs/data-flow.md).")
        return

    train_cfg = load_yaml(args.config)
    data_cfg = load_yaml(args.data_config)
    model_cfg = load_yaml(args.model_config)

    import torch
    from src.data.dataloaders import build_face_loader, build_audio_loader
    from src.data.transforms import build_train_face_transform, build_val_face_transform, spec_augment
    from src.models.architecture import build_model
    from src.training.trainer import Trainer

    branch_cfg = model_cfg[args.branch]
    model = build_model(args.branch,
                        num_classes=branch_cfg.get("num_classes", 7),
                        pretrained=branch_cfg.get("pretrained", True) if args.branch == "face" else True,
                        dropout=branch_cfg.get("dropout", 0.3) if args.branch == "audio" else 0.3)

    if args.branch == "face":
        bs = data_cfg["face"]["batch_size"]
        train_loader = build_face_loader(args.train_csv, batch_size=bs if not args.sanity else 8,
                                         shuffle=True, num_workers=0,
                                         transform=build_train_face_transform())
        val_loader = build_face_loader(args.val_csv, batch_size=bs,
                                       shuffle=False, num_workers=0,
                                       transform=build_val_face_transform())
    else:
        bs = data_cfg["audio"]["batch_size"]
        train_loader = build_audio_loader(args.train_csv, batch_size=bs if not args.sanity else 8,
                                          shuffle=True, num_workers=0)
        val_loader = build_audio_loader(args.val_csv, batch_size=bs,
                                        shuffle=False, num_workers=0)

    ts = datetime.now().strftime("%Y%m%d-%H%M")
    run_dir = Path("results/runs") / f"{ts}_{args.branch}"
    ckpt_dir = Path(train_cfg.get("checkpoint_dir", "models/checkpoints")) / args.branch
    trainer = Trainer(model, train_loader, val_loader,
                      lr=train_cfg.get("learning_rate", 3e-4),
                      weight_decay=train_cfg.get("weight_decay", 1e-4),
                      epochs=args.epochs or (2 if args.sanity else train_cfg.get("epochs", 30)),
                      patience=train_cfg.get("early_stopping", {}).get("patience", 7),
                      grad_clip=train_cfg.get("gradient_clipping", 1.0),
                      ckpt_dir=ckpt_dir, run_dir=run_dir)
    # Lưu config reproduce
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump({"branch": args.branch, "train": train_cfg,
                        "data": data_cfg, "model": branch_cfg}, allow_unicode=True),
        encoding="utf-8")
    best = trainer.fit()
    print(f"[training] xong nhánh {args.branch}. Best: {best}")


if __name__ == "__main__":
    main()
