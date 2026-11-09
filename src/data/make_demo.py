"""Tao du lieu demo nhe 3 anh + 3 wav vao data/processed/demo.

Chay: python -m src.data.make_demo
Lay 3 anh dau tien tu face_train.csv (parse pixels -> PNG 48x48)
va 3 file WAV dau tien tu audio_train.csv (copy) de demo 1 lenh.
"""
from __future__ import annotations
from pathlib import Path
import shutil
import pandas as pd
import numpy as np
from PIL import Image


def main():
    demo = Path("data/processed/demo")
    demo.mkdir(parents=True, exist_ok=True)
    # 1. Mat: 3 anh
    face_csv = Path("data/splits/face_train.csv")
    df = pd.read_csv(face_csv, nrows=3)
    for i, row in df.iterrows():
        pix = np.array(list(map(int, str(row["pixels"]).split())), dtype=np.uint8).reshape(48, 48)
        label = int(row["label"])
        Image.fromarray(pix).save(demo / f"face_{i+1:02d}_label{label}.png")
    # 2. Tieng: 3 wav
    audio_csv = Path("data/splits/audio_train.csv")
    adf = pd.read_csv(audio_csv, nrows=3)
    for i, row in adf.iterrows():
        src = Path(str(row["filepath"]))
        label = int(row["label"])
        dst = demo / f"voice_{i+1:02d}_label{label}.wav"
        if src.exists():
            shutil.copy(src, dst)
    print(f"[demo] xong: {sorted(p.name for p in demo.glob('*'))}")


if __name__ == "__main__":
    main()
