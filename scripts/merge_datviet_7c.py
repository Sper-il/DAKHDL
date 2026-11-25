#!/usr/bin/env python
"""Gộp bộ audio 7 lớp gốc (CREMA-D, RAVDESS, ASVP-ESD) với DataViet (đã chuẩn bị)
thành một bộ audio 7 lớp chung để fine-tune (Cách A: giữ đủ 7 cảm xúc + giỏi tiếng Việt).

Trên Colab (sau khi giải nén data.zip -> /content/data và data_datviet.zip -> /content/data_datviet):
    python scripts/merge_datviet_7c.py --base-root /content/data --datviet-root /content/data_datviet --out /content/data_merged
"""
import argparse
import csv
from pathlib import Path

# DataViet 4 lớp (0:angry, 1:happy, 2:sad, 3:neutral) -> 7 lớp chuẩn (0:angry, 3:happy, 4:sad, 6:neutral)
MAP_4_TO_7 = {0: 0, 1: 3, 2: 4, 3: 6}


def find_splits(root: Path) -> Path:
    for cand in (root / "data" / "splits", root / "splits"):
        if cand.is_dir():
            return cand
    raise FileNotFoundError(f"Không thấy thư mục splits trong {root}")


def resolve_mel_abs(root: Path, rel: str) -> str:
    rel = rel.replace("\\", "/")
    for cand in (root / rel, root / "data" / rel):
        if cand.is_file():
            return str(cand.resolve())
    raise FileNotFoundError(f"Không tìm thấy {rel} dưới {root}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-root", required=True, help="Thư mục data 7 lớp gốc (chứa data/splits)")
    ap.add_argument("--datviet-root", required=True, help="Thư mục data_datviet (sau khi giải nén data_datviet.zip)")
    ap.add_argument("--out", required=True, help="Thư mục đầu ra cho CSV gộp")
    args = ap.parse_args()

    base_root = Path(args.base_root)
    dv_root = Path(args.datviet_root)
    base_splits = find_splits(base_root)
    dv_splits = find_splits(dv_root)

    out_splits = Path(args.out) / "data" / "splits"
    out_splits.mkdir(parents=True, exist_ok=True)

    for sp in ("train", "val", "test"):
        merged_rows = []
        # 1. Đọc dữ liệu 7 lớp gốc
        with open(base_splits / f"audio_{sp}_proc.csv", newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                lab = int(r["label"])
                if lab >= 7:
                    continue
                merged_rows.append([
                    resolve_mel_abs(base_root, r["mel_path"]),
                    lab,
                    r.get("speaker", ""),
                    r.get("source", "base"),
                ])

        # 2. Đọc DataViet và ánh xạ nhãn 4 -> 7
        n_dv = 0
        with open(dv_splits / f"audio_{sp}_proc.csv", newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                lab4 = int(r["label"])
                if lab4 not in MAP_4_TO_7:
                    continue
                merged_rows.append([
                    resolve_mel_abs(dv_root, r["mel_path"]),
                    MAP_4_TO_7[lab4],
                    r.get("speaker", ""),
                    r.get("source", "datviet"),
                ])
                n_dv += 1

        out_csv = out_splits / f"audio_{sp}_proc.csv"
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f, lineterminator="\n")
            wr.writerow(["mel_path", "label", "speaker", "source"])
            wr.writerows(merged_rows)
        print(f"[{sp}] Đã gộp {len(merged_rows)} mẫu (gồm {n_dv} mẫu DataViet) -> {out_csv}")


if __name__ == "__main__":
    main()
