"""Chia train/val/test cho 2 nhánh, seed cố định, audio split theo speaker."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42


def make_face_splits(raw_csv: str | Path, out_dir: str | Path,
                      train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=SEED):
    """Chia FER2013 theo stratify label. Ưu tiên cột Usage gốc nếu có đủ 3 tập."""
    df = pd.read_csv(raw_csv)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Chuẩn hóa tên cột label
    if "emotion" in df.columns and "label" not in df.columns:
        df = df.rename(columns={"emotion": "label"})
    if "Usage" in set(df.columns):
        mapping = {"Training": "train", "PublicTest": "val", "PrivateTest": "test"}
        for usage, name in mapping.items():
            sub = df[df["Usage"] == usage].copy()
            sub.to_csv(out / f"face_{name}.csv", index=False)
        return {n: str(out / f"face_{n}.csv") for n in ["train", "val", "test"]}
    # Không có Usage: stratify 70/15/15
    train, temp = train_test_split(df, train_size=train_ratio, random_state=seed, stratify=df["label"])
    val_size = val_ratio / (val_ratio + test_ratio)
    val, test = train_test_split(temp, train_size=val_size, random_state=seed, stratify=temp["label"])
    paths = {}
    for name, sub in [("train", train), ("val", val), ("test", test)]:
        p = out / f"face_{name}.csv"
        sub.to_csv(p, index=False)
        paths[name] = str(p)
    return paths


def make_audio_splits(file_csv: str | Path, out_dir: str | Path,
                       train_ratio=0.7, val_ratio=0.15, seed=SEED):
    """Chia audio theo speaker để chống leakage. file_csv cần cột filepath,label,speaker."""
    df = pd.read_csv(file_csv)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    speakers = sorted(df["speaker"].unique().tolist())
    train_sp, temp_sp = train_test_split(speakers, train_size=train_ratio, random_state=seed)
    val_ratio_adj = val_ratio / (1 - train_ratio)
    val_sp, test_sp = train_test_split(temp_sp, train_size=val_ratio_adj, random_state=seed)
    paths = {}
    for name, sp in [("train", train_sp), ("val", val_sp), ("test", test_sp)]:
        sub = df[df["speaker"].isin(sp)].copy()
        p = out / f"audio_{name}.csv"
        sub.to_csv(p, index=False)
        paths[name] = str(p)
    return paths
