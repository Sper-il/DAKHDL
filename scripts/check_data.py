#!/usr/bin/env python
"""Kiểm tra data trước khi train (chỉ cần numpy + pandas).
    python scripts/check_data.py --data-root /content/data
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import data_utils as du  # noqa: E402
from src.labels import EMOTIONS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    a = ap.parse_args()
    splits_dir = du.default_splits_dir(a.data_root)
    print(f"splits_dir = {splits_dir}\n")
    for branch in ("face", "audio"):
        print(f"===== {branch} =====")
        ok = True
        for sp in ("train", "val", "test"):
            try:
                p = du.split_csv_path(splits_dir, branch, sp)
            except FileNotFoundError as e:
                print(f"  THIẾU: {e}")
                ok = False
                continue
            df = pd.read_csv(p, usecols=lambda c: c != "pixels")
            lab = "label" if "label" in df.columns else "emotion"
            cnt = df[lab].value_counts().sort_index()
            print(f"  {sp:<5} {len(df):>6} mẫu | "
                  + ", ".join(f"{EMOTIONS[int(k)]}={v}" for k, v in cnt.items() if int(k) < len(EMOTIONS)))
            if branch == "audio":
                if "speaker" in df.columns:
                    print(f"        {df['speaker'].nunique()} người nói")
                if "source" in df.columns:
                    print("        nguồn x nhãn:\n" + pd.crosstab(df["source"], df[lab]).to_string().replace("\n", "\n          "))
                try:
                    paths = du.resolve_paths(a.data_root, df["mel_path"].tolist())
                    rng = np.random.default_rng(0)
                    for i in rng.choice(len(paths), size=min(3, len(paths)), replace=False):
                        x = np.load(paths[i])
                        print(f"        mẫu {paths[i].name}: shape={x.shape} min={x.min():.1f} max={x.max():.1f} mean={x.mean():.1f}")
                except Exception as e:
                    print(f"  LỖI đường dẫn mel: {e}")
                    ok = False
        if branch == "audio":
            try:
                spk = {sp: set(pd.read_csv(du.split_csv_path(splits_dir, "audio", sp))["speaker"]) for sp in ("train", "val", "test")}
                print(f"  người nói trùng giữa các tập: train∩val={len(spk['train'] & spk['val'])}, "
                      f"train∩test={len(spk['train'] & spk['test'])}, val∩test={len(spk['val'] & spk['test'])} (nên = 0)")
            except Exception:
                pass
        print(f"  => {'OK' if ok else 'CÓ VẤN ĐỀ'}\n")


if __name__ == "__main__":
    main()
