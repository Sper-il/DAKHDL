"""Gop anh mat extra (contempt/boredom/excited) vao face splits hien co.

Nguon:
- contempt (7): data/raw/affectnet_face/contempt/* (AffectNet, mat crop san 96x96)
- boredom (8): data/raw/face_extra/boredom/* (Student Engagement, anh rong -> center-crop vuong)
- excited (9): data/raw/face_extra/excited/* (proxy tu AffectNet happy 96x96, mat crop san)

Chia extra theo ty le 70/15/15, seed 42, roi noi vao data/splits/face_{train,val,test}.csv
dang pixels (giong FER2013). Giu nguyen 35.887 mau cu, chi append them.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

SEED = 42
EXTRA = {7: Path("data/raw/affectnet_face/contempt"), 8: Path("data/raw/face_extra/boredom"), 9: Path("data/raw/face_extra/excited")}
WIDE_CLASSES = {8}  # anh rong can center-crop


def img_to_pixels(p: Path, wide: bool) -> str:
    im = Image.open(p).convert("L")
    if wide:
        w, h = im.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        im = im.crop((left, top, left + side, top + side))
    im = im.resize((48, 48), Image.BILINEAR)
    return " ".join(str(v) for v in list(im.getdata()))


def main():
    splits = Path("data/splits")
    rows = []
    for label, d in EXTRA.items():
        files = sorted([f for f in d.glob("*.*") if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".webp")])
        print(f"[face-extra] label {label}: {len(files)} files trong {d}")
        for f in files:
            try:
                px = img_to_pixels(f, wide=(label in WIDE_CLASSES))
            except Exception as e:
                print(f"  bo qua {f.name}: {e}")
                continue
            rows.append({"label": label, "pixels": px, "Usage": "extra", "src": f.as_posix()})
    df = pd.DataFrame(rows)
    print("[face-extra] tong extra hop le:", len(df), df["label"].value_counts().sort_index().to_dict())

    # Chia 70/15/15 stratify theo label
    tr, tmp = train_test_split(df, train_size=0.7, random_state=SEED, stratify=df["label"])
    va, te = train_test_split(tmp, train_size=0.5, random_state=SEED, stratify=tmp["label"])
    print(f"[face-extra] split extra: train {len(tr)} / val {len(va)} / test {len(te)}")
    for name, sub in [("train", tr), ("val", va), ("test", te)]:
        print(f"  extra {name}:", sub["label"].value_counts().sort_index().to_dict())

    for name, sub in [("train", tr), ("val", va), ("test", te)]:
        csv = splits / f"face_{name}.csv"
        base = pd.read_csv(csv)
        add = sub[["label", "pixels", "Usage"]].copy()
        merged = pd.concat([base[["label", "pixels", "Usage"]], add], ignore_index=True)
        merged = merged.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
        merged.to_csv(csv, index=False)
        print(f"[face-extra] {csv.name}: {len(base)} -> {len(merged)} (+{len(add)})")
        print("   labels:", merged["label"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
