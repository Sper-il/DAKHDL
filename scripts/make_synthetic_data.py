#!/usr/bin/env python
"""Tạo data GIẢ cùng cấu trúc với data thật để chạy thử nhanh (smoke test) toàn bộ pipeline.
    python scripts/make_synthetic_data.py --out /tmp/fake_emotion
Mặt: dải sáng dọc theo lớp; tiếng: dải năng lượng theo lớp -> mô hình học được nên loss giảm.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def face_img(label, rng):
    img = rng.normal(90, 20, (48, 48))
    c = 2 + label * 4
    img[:, c:c + 4] += 90
    return np.clip(img, 0, 255).astype(np.uint8)


def mel(label, rng):
    m = rng.normal(-65, 6, (128, 128))
    r = 4 + label * 11
    m[r:r + 8, :] = rng.normal(-25, 5, (8, 128))
    return np.clip(m, -80, 0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-class", type=int, default=24, help="số mẫu mỗi lớp ở tập train")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    root = Path(a.out)
    spl = root / "data" / "splits"
    spl.mkdir(parents=True, exist_ok=True)
    sizes = {"train": a.per_class, "val": max(4, a.per_class // 4), "test": max(4, a.per_class // 4)}
    sources = ["asvp", "crema", "ravdess"]
    for sp, per in sizes.items():
        # mặt: lớp 8, 9 ít mẫu như data thật
        rows = []
        for lab in range(10):
            k = per if lab < 8 else max(2, per // 6)
            for _ in range(k):
                px = " ".join(map(str, face_img(lab, rng).ravel()))
                rows.append({"label": lab, "pixels": px, "Usage": "Training" if sp == "train" else "PrivateTest"})
        pd.DataFrame(rows).sample(frac=1, random_state=a.seed).to_csv(spl / f"face_{sp}.csv", index=False)
        # tiếng
        d = root / "data" / "processed" / "audio" / sp
        d.mkdir(parents=True, exist_ok=True)
        arows, i = [], 0
        for lab in range(10):
            k = per if lab < 8 else max(2, per // 6)
            for _ in range(k):
                fn = f"{sp}_{i:05d}_label{lab}.npy"
                np.save(d / fn, mel(lab, rng))
                arows.append({"mel_path": f"data/processed/audio/{sp}/{fn}", "label": lab,
                              "speaker": f"{sp}_spk{i % 6}", "source": sources[(lab + i) % 3]})
                i += 1
        pd.DataFrame(arows).sample(frac=1, random_state=a.seed).to_csv(spl / f"audio_{sp}_proc.csv", index=False)
    print(f"Đã tạo data giả tại {root}")


if __name__ == "__main__":
    main()
