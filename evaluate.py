#!/usr/bin/env python
"""Đánh giá lại một checkpoint tốt nhất.  Ví dụ:
    python evaluate.py --run-dir runs/face_resnet18_2026... --splits val test
    python evaluate.py --run-dir runs/audio_x --set data_root=/content/data
"""
import argparse
from pathlib import Path

from src.config import apply_overrides
from src.datasets import build_datasets
from src.labels import class_names
from src.models import build_model
from src.trainer import final_report, load_ckpt, make_loaders, pick_device


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--set", nargs="*", default=[], metavar="KHÓA=GIÁ_TRỊ")
    ap.add_argument("--out", help="thư mục ghi kết quả (mặc định <run-dir>/eval)")
    a = ap.parse_args()

    run_dir = Path(a.run_dir)
    device = pick_device("auto")
    ck = load_ckpt(run_dir / "best.pt", device)
    cfg = apply_overrides(ck["cfg"], a.set)
    cfg["model"]["pretrained"] = False
    cfg["logging"]["tensorboard"] = False
    cfg["train"]["num_workers"] = int(cfg["train"].get("num_workers", 0))
    names = class_names(int(cfg["num_classes"]))
    data = build_datasets(cfg, stats=(ck["mean"], ck["std"]), splits=tuple(a.splits))
    loaders = make_loaders(data, cfg, device)
    model = build_model(cfg).to(device)
    model.load_state_dict(ck["model"])
    use_amp = bool(cfg["train"]["amp"]) and device.type == "cuda"
    final_report(model, loaders, data, cfg, names, a.out or (run_dir / "eval"), device, use_amp,
                 splits=tuple(a.splits))


if __name__ == "__main__":
    main()
