#!/usr/bin/env python
"""Huấn luyện một nhánh.  Ví dụ:
    python train.py --config configs/face.yaml  --set data_root=/content/data
    python train.py --config configs/audio.yaml --set data_root=/content/data train.epochs=40
    python train.py --config configs/face.yaml --run-dir runs/face_x --resume
"""
import argparse
import datetime
from pathlib import Path

from src.config import load_config
from src.trainer import fit


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="file YAML trong configs/")
    ap.add_argument("--set", nargs="*", default=[], metavar="KHÓA=GIÁ_TRỊ",
                    help="ghi đè cấu hình, vd: data_root=/content/data train.epochs=5")
    ap.add_argument("--run-dir", help="thư mục lưu run (mặc định runs/<name>_<thời gian>)")
    ap.add_argument("--resume", action="store_true", help="tiếp tục từ last.pt trong --run-dir")
    ap.add_argument("--init-ckpt", help="nạp trọng số khởi tạo từ best.pt để fine-tune")
    a = ap.parse_args()

    cfg = load_config(a.config, a.set)
    if a.run_dir:
        run_dir = Path(a.run_dir)
    else:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(cfg["logging"]["out_dir"]) / f"{cfg['name']}_{stamp}"
    if a.resume and not a.run_dir:
        ap.error("--resume cần --run-dir")
    fit(cfg, run_dir, resume=a.resume, init_ckpt=a.init_ckpt)


if __name__ == "__main__":
    main()
