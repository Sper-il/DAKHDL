"""CLI train 1 nhánh (face hoặc audio) với chống overfitting, fine-tuning và đánh giá GroupKFold."""
from __future__ import annotations
import argparse
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _train_single_split(
    branch: str,
    train_csv: str | Path,
    val_csv: str | Path,
    train_cfg: dict,
    data_cfg: dict,
    branch_cfg: dict,
    args,
    tag_suffix: str = "",
) -> tuple[Path, float]:
    import torch
    from src.data.dataloaders import build_face_loader, build_audio_loader
    from src.data.transforms import (
        build_train_face_transform,
        build_val_face_transform,
        build_train_audio_transform,
    )
    from src.models.architecture import build_model
    from src.training.trainer import Trainer

    arch = args.arch or branch_cfg.get("name", None)
    stem_type = branch_cfg.get("stem_type", "cifar")
    dropout = args.dropout if args.dropout is not None else branch_cfg.get("dropout", 0.4)

    model = build_model(
        branch,
        num_classes=branch_cfg.get("num_classes", 7),
        pretrained=branch_cfg.get("pretrained", True),
        dropout=dropout,
        arch=arch,
        stem_type=stem_type,
    )

    if args.pretrained_ckpt:
        ckpt_path = Path(args.pretrained_ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Không tìm thấy checkpoint để fine-tune: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)
        print(f"[training] Đã nạp trọng số từ {ckpt_path} để fine-tune!")

    import pandas as pd
    n_train = len(pd.read_csv(train_csv))
    if branch == "face":
        cfg_bs = data_cfg["face"]["batch_size"]
        bs = min(cfg_bs, 16) if n_train < 500 else cfg_bs
        train_loader = build_face_loader(
            str(train_csv),
            batch_size=bs if not args.sanity else 8,
            shuffle=True,
            num_workers=0,
            transform=build_train_face_transform(),
        )
        val_loader = build_face_loader(
            str(val_csv),
            batch_size=bs,
            shuffle=False,
            num_workers=0,
            transform=build_val_face_transform(),
        )
    else:
        cfg_bs = data_cfg["audio"]["batch_size"]
        bs = min(cfg_bs, 16) if n_train < 500 else cfg_bs
        train_loader = build_audio_loader(
            str(train_csv),
            batch_size=bs if not args.sanity else 8,
            shuffle=True,
            num_workers=0,
            transform=build_train_audio_transform(time_mask=20, freq_mask=16, num_masks=2),
            augment_wave=True,
        )
        val_loader = build_audio_loader(
            str(val_csv),
            batch_size=bs,
            shuffle=False,
            num_workers=0,
            transform=None,
            augment_wave=False,
        )

    ts = datetime.now().strftime("%Y%m%d-%H%M")
    run_name = f"{ts}_{branch}{tag_suffix}"
    run_dir = Path("results/runs") / run_name
    ckpt_dir = Path(train_cfg.get("checkpoint_dir", "models/checkpoints")) / f"{branch}{tag_suffix}"

    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        lr=args.lr or train_cfg.get("learning_rate", 3e-4),
        weight_decay=args.weight_decay or train_cfg.get("weight_decay", 1e-2),
        epochs=args.epochs or (2 if args.sanity else train_cfg.get("epochs", 30)),
        patience=train_cfg.get("early_stopping", {}).get("patience", 8),
        grad_clip=train_cfg.get("gradient_clipping", 1.0),
        label_smoothing=args.label_smoothing if args.label_smoothing is not None else train_cfg.get("label_smoothing", 0.1),
        use_class_weights=train_cfg.get("use_class_weights", True),
        mixup_alpha=args.mixup_alpha if args.mixup_alpha is not None else train_cfg.get("mixup_alpha", 0.2),
        use_ema=train_cfg.get("use_ema", True),
        warmup_epochs=train_cfg.get("warmup_epochs", 2),
        freeze_epochs=args.freeze_epochs if args.freeze_epochs is not None else train_cfg.get("freeze_epochs", 2),
        ckpt_dir=ckpt_dir,
        run_dir=run_dir,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump({"branch": branch, "train": train_cfg, "data": data_cfg, "model": branch_cfg}, allow_unicode=True),
        encoding="utf-8",
    )
    best = trainer.fit()
    print(f"[training] xong nhánh {branch}{tag_suffix}. Best F1={trainer.best_f1:.4f} tại: {best}")
    return best, float(trainer.best_f1)


def main():
    parser = argparse.ArgumentParser(description="Train emotion-fusion 1 nhánh (chống overfitting)")
    parser.add_argument("--branch", choices=["face", "audio"], required=True)
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--data-config", default="configs/data_config.yaml")
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    parser.add_argument("--splits-prefix", default="", help="Tiền tố file split trong data/splits/ (ví dụ: 'vn_')")
    parser.add_argument("--train-csv", default=None, help="Đường dẫn face_train.csv / audio_train.csv")
    parser.add_argument("--val-csv", default=None, help="Đường dẫn face_val.csv / audio_val.csv")
    parser.add_argument("--pretrained-ckpt", default=None, help="Đường dẫn checkpoint .pt để fine-tune tiếp")
    parser.add_argument("--arch", default=None, help="Kiến trúc nhánh tiếng: audio_resnet18_mel | audio_cnn_4blocks")
    parser.add_argument("--lr", type=float, default=None, help="Ghi đè learning rate (ví dụ 1e-4 khi fine-tune)")
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--label-smoothing", type=float, default=None)
    parser.add_argument("--mixup-alpha", type=float, default=None)
    parser.add_argument("--freeze-epochs", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--kfold", action="store_true", help="Chạy GroupKFold theo video_id (chống rò rỉ dữ liệu VN)")
    parser.add_argument("--sanity", action="store_true", help="Chỉ overfit 1 batch nhỏ để kiểm tra")
    args = parser.parse_args()

    train_cfg = load_yaml(args.config)
    data_cfg = load_yaml(args.data_config)
    model_cfg = load_yaml(args.model_config)
    branch_cfg = model_cfg[args.branch]

    # Chế độ GroupKFold Cross-Validation
    if args.kfold:
        prefix = args.splits_prefix or "vn_"
        kfold_dir = Path("data/splits/kfold")
        fold_trains = sorted(kfold_dir.glob(f"{prefix}fold*_{args.branch}_train.csv"))
        if not fold_trains:
            raise FileNotFoundError(
                f"Không tìm thấy các file GroupKFold trong {kfold_dir}. Hãy chạy: python -m tools.yt_crawler export"
            )
        f1_scores = []
        for idx, tr_path in enumerate(fold_trains):
            va_path = kfold_dir / f"{prefix}fold{idx}_{args.branch}_val.csv"
            print(f"\n========== [GroupKFold {idx + 1}/{len(fold_trains)}] {tr_path.name} ==========")
            _, best_f1 = _train_single_split(
                args.branch, tr_path, va_path, train_cfg, data_cfg, branch_cfg, args, tag_suffix=f"_fold{idx}"
            )
            f1_scores.append(best_f1)
        mean_f1 = float(np.mean(f1_scores))
        std_f1 = float(np.std(f1_scores))
        print(f"\n[GroupKFold] Kết quả {len(f1_scores)} folds nhánh {args.branch}: F1-macro = {mean_f1:.4f} ± {std_f1:.4f}")
        return

    # Tự động suy ra đường dẫn train_csv / val_csv nếu người dùng truyền --splits-prefix hoặc đã có sẵn trong data/splits
    if not args.train_csv or not args.val_csv:
        cand_tr = Path("data/splits") / f"{args.splits_prefix}{args.branch}_train.csv"
        cand_va = Path("data/splits") / f"{args.splits_prefix}{args.branch}_val.csv"
        if cand_tr.exists() and cand_va.exists():
            args.train_csv = str(cand_tr)
            args.val_csv = str(cand_va)
        else:
            print("[training] thiếu --train-csv và --val-csv.")
            print(
                "Ví dụ: python -m src.training --branch face "
                "--train-csv data/splits/face_train.csv --val-csv data/splits/face_val.csv"
            )
            print("Hoặc dùng bộ dữ liệu Tiếng Việt: python -m src.training --branch face --splits-prefix vn_")
            return

    _train_single_split(args.branch, args.train_csv, args.val_csv, train_cfg, data_cfg, branch_cfg, args)


if __name__ == "__main__":
    main()
