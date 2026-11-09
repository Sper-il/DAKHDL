"""Vat lieu hoa data/processed: face PNG 48x48 + audio mel .npy 128x128.

Chay: python -m src.data.make_processed [--face-only] [--audio-only] [--limit N]

Doc data/splits/face_*.csv (cot pixels) -> data/processed/face/{train,val,test}/*.png
Doc data/splits/audio_*.csv (cot filepath) -> data/processed/audio/{train,val,test}/*.npy
Dong thoi sinh audio_*_proc.csv ke ben splits de train nhanh khong can librosa.
Bo qua file da ton tai de chay tiep duoc.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image


def process_face(splits: Path, out_root: Path, limit: int | None = None) -> dict:
    out_root.mkdir(parents=True, exist_ok=True)
    stats = {}
    for split in ["train", "val", "test"]:
        csv = splits / f"face_{split}.csv"
        df = pd.read_csv(csv, usecols=["label", "pixels"])
        if limit:
            df = df.head(limit)
        d = out_root / split
        d.mkdir(parents=True, exist_ok=True)
        n_new = 0
        for i, row in df.iterrows():
            dst = d / f"{split}_{i:05d}_label{int(row['label'])}.png"
            if dst.exists():
                continue
            arr = np.fromstring(str(row["pixels"]), sep=" ", dtype=np.uint8).reshape(48, 48)
            Image.fromarray(arr).save(dst)
            n_new += 1
        stats[split] = {"total": len(df), "new": n_new}
        print(f"[processed-face] {split}: {len(df)} mau, moi {n_new} -> {d}/")
    return stats


def process_audio(splits: Path, out_root: Path, limit: int | None = None) -> dict:
    import librosa
    out_root.mkdir(parents=True, exist_ok=True)
    stats = {}
    for split in ["train", "val", "test"]:
        csv = splits / f"audio_{split}.csv"
        df = pd.read_csv(csv)
        if limit:
            df = df.head(limit)
        d = out_root / split
        d.mkdir(parents=True, exist_ok=True)
        rows = []
        n_new = 0
        for i, row in df.iterrows():
            src = str(row["filepath"])
            dst = d / f"{split}_{i:05d}_label{int(row['label'])}.npy"
            if not dst.exists():
                try:
                    y, _ = librosa.load(src, sr=16000, mono=True, duration=4.0)
                    target = 16000 * 4
                    y = np.pad(y, (0, max(0, target - len(y))))[:target]
                    mel = librosa.feature.melspectrogram(y=y, sr=16000, n_mels=128)
                    mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
                    if mel.shape[1] < 128:
                        mel = np.pad(mel, ((0, 0), (0, 128 - mel.shape[1])))
                    else:
                        mel = mel[:, :128]
                    np.save(dst, mel)
                    n_new += 1
                except Exception as e:
                    print(f"[processed-audio] loi {src}: {e}")
                    continue
            rows.append({"mel_path": dst.as_posix(), "label": int(row["label"]),
                         "speaker": str(row.get("speaker", "")),
                         "source": str(row.get("source", ""))})
        proc_csv = splits / f"audio_{split}_proc.csv"
        pd.DataFrame(rows).to_csv(proc_csv, index=False)
        stats[split] = {"total": len(df), "new": n_new}
        print(f"[processed-audio] {split}: {len(df)} mau, moi {n_new} -> {d}/ + {proc_csv.name}")
    return stats


def main():
    ap = argparse.ArgumentParser(description="Vat lieu hoa data/processed")
    ap.add_argument("--face-only", action="store_true")
    ap.add_argument("--audio-only", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="Gioi han mau moi split de test nhanh")
    args = ap.parse_args()
    splits = Path("data/splits")
    do_face = not args.audio_only
    do_audio = not args.face_only
    if do_face:
        process_face(splits, Path("data/processed/face"), args.limit)
    if do_audio:
        process_audio(splits, Path("data/processed/audio"), args.limit)
    print("[processed] xong.")


if __name__ == "__main__":
    main()
