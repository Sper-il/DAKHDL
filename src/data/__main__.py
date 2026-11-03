"""CLI kiểm tra nhanh module data: in shape 1 batch mặt và tiếng."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Kiểm tra data pipeline emotion-fusion")
    parser.add_argument("--config", default="configs/data_config.yaml")
    parser.add_argument("--face-csv", default=None, help="Đường dẫn face_{train,val}.csv")
    parser.add_argument("--audio-csv", default=None, help="Đường dẫn audio_{train,val}.csv")
    args = parser.parse_args()

    print(f"[data] config: {args.config}")
    if args.face_csv and Path(args.face_csv).exists():
        from .dataloaders import build_face_loader
        from .transforms import build_val_face_transform
        loader = build_face_loader(args.face_csv, batch_size=4, shuffle=False,
                                   num_workers=0, transform=build_val_face_transform())
        batch = next(iter(loader))
        print(f"[face] image: {tuple(batch['image'].shape)} label: {tuple(batch['label'].shape)}")
    else:
        print("[face] skip (chua co file split). Chay split.py sau khi tai FER2013.")

    if args.audio_csv and Path(args.audio_csv).exists():
        from .dataloaders import build_audio_loader
        loader = build_audio_loader(args.audio_csv, batch_size=4, shuffle=False, num_workers=0)
        batch = next(iter(loader))
        print(f"[audio] mel: {tuple(batch['mel'].shape)} label: {tuple(batch['label'].shape)}")
    else:
        print("[audio] skip (chua co file split). Chay split.py sau khi tai RAVDESS.")
    print("[data] done. Shape ky vong: mat [B,1,48,48], tieng [B,1,128,128].")


if __name__ == "__main__":
    main()
