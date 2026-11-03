"""Sinh splits mat + tieng, seed 42, audio split theo speaker.

Chay: python -m src.data.make_splits
Tu tim FER o data/raw/fer2013.csv (uu tien) hoac icml_face_data.csv,
quet WAV o data/raw/audio/, sinh data/splits/face_*.csv va audio_*.csv.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

from .split import make_face_splits, make_audio_splits, SEED


# RAVDESS emotion code -> nhan 7 cam xuc cua project
# 01 neutral->6, 02 calm->6, 03 happy->3, 04 sad->4, 05 angry->0,
# 06 fear->2, 07 disgust->1, 08 surprised->5
RAVDESS_MAP = {"01": 6, "02": 6, "03": 3, "04": 4, "05": 0, "06": 2, "07": 1, "08": 5}


def normalize_fer(raw_dir: Path) -> Path:
    """Tra ve duong dan CSV chuan emotion,pixels,Usage trong data/raw."""
    fer = raw_dir / "fer2013.csv"
    if fer.exists() and fer.stat().st_size > 1_000_000:
        print(f"[splits] dung san {fer.name} ({fer.stat().st_size} bytes)")
        return fer
    icml = raw_dir / "icml_face_data.csv"
    if icml.exists() and icml.stat().st_size > 1_000_000:
        df = pd.read_csv(icml)
        df.columns = [c.strip() for c in df.columns]
        cols = {c.lower(): c for c in df.columns}
        df = df.rename(columns={cols["emotion"]: "emotion",
                                cols["pixels"]: "pixels",
                                cols["usage"]: "Usage"})
        out = raw_dir / "fer2013.csv"
        df[["emotion", "pixels", "Usage"]].to_csv(out, index=False)
        print(f"[splits] chuan hoa {icml.name} -> fer2013.csv ({len(df)} dong)")
        return out
    raise FileNotFoundError("Thieu data/raw/fer2013.csv (hoac icml_face_data.csv >1MB)")


def build_audio_filelist(audio_root: Path, out_csv: Path) -> Path:
    """Quet WAV RAVDESS, sinh file filepath,speaker,label."""
    wavs = sorted(audio_root.rglob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"Khong thay WAV nao trong {audio_root}")
    rows = []
    for w in wavs:
        parts = w.stem.split("-")
        if len(parts) != 7:
            continue
        emo_code, speaker = parts[2], parts[6]
        if emo_code not in RAVDESS_MAP:
            continue
        rows.append({"filepath": str(w.as_posix()), "speaker": f"actor_{speaker}",
                     "label": RAVDESS_MAP[emo_code]})
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[splits] quet {len(df)} WAV, {df['speaker'].nunique()} speaker -> {out_csv.name}")
    print(df["label"].value_counts().sort_index().to_dict())
    return out_csv


def make_audio_filelist_split(filelist: Path, splits: Path):
    return make_audio_splits(filelist, splits, seed=SEED)


def main():
    raw = Path("data/raw")
    splits = Path("data/splits")
    splits.mkdir(parents=True, exist_ok=True)

    # 1. Mat
    fer_csv = normalize_fer(raw)
    face_paths = make_face_splits(fer_csv, splits, seed=SEED)
    for k, p in face_paths.items():
        with open(p, encoding="utf-8") as f:
            n = sum(1 for _ in f) - 1
        print(f"[splits] mat {k}: {n} mau -> {p}")

    # 2. Tieng
    audio_root = raw / "audio"
    filelist = splits / "audio_files.csv"
    build_audio_filelist(audio_root, filelist)
    audio_paths = make_audio_filelist_split(filelist, splits)
    for k, p in audio_paths.items():
        with open(p, encoding="utf-8") as f:
            n = sum(1 for _ in f) - 1
        print(f"[splits] tieng {k}: {n} mau -> {p}")
    print("[splits] xong. Seed=42, audio split theo speaker.")


if __name__ == "__main__":
    main()
