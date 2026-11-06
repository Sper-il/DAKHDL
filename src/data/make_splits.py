"""Sinh splits mặt + tiếng, seed 42, audio split theo speaker (hỗ trợ RAVDESS + CREMA-D + tự động tải RAVDESS).

Chạy:
  python -m src.data.make_splits
  python -m src.data.make_splits --download-ravdess
"""
from __future__ import annotations
import argparse
import urllib.request
import zipfile
from pathlib import Path
import pandas as pd

from .split import make_face_splits, make_audio_splits, SEED


# RAVDESS emotion code -> nhãn 7 cảm xúc của project
# 01 neutral->6, 02 calm->6, 03 happy->3, 04 sad->4, 05 angry->0,
# 06 fear->2, 07 disgust->1, 08 surprised->5
RAVDESS_MAP = {"01": 6, "02": 6, "03": 3, "04": 4, "05": 0, "06": 2, "07": 1, "08": 5}

# CREMA-D emotion code -> nhãn 7 cảm xúc của project
CREMAD_MAP = {"ANG": 0, "DIS": 1, "FEA": 2, "HAP": 3, "SAD": 4, "NEU": 6}

RAVDESS_ZENODO_URL = "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"


def download_ravdess_if_needed(audio_root: Path) -> None:
    """Tải tự động bộ RAVDESS Speech (24 diễn viên, ~200MB) từ Zenodo nếu chưa có."""
    audio_root.mkdir(parents=True, exist_ok=True)
    existing = list(audio_root.rglob("*.wav"))
    if len(existing) >= 100:
        print(f"[splits] Đã có {len(existing)} file WAV trong {audio_root}, bỏ qua tải RAVDESS.")
        return
    zip_path = audio_root / "Audio_Speech_Actors_01-24.zip"
    print(f"[splits] Đang tải RAVDESS từ Zenodo về {zip_path}...")
    urllib.request.urlretrieve(RAVDESS_ZENODO_URL, str(zip_path))
    print("[splits] Đang giải nén RAVDESS...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(audio_root)
    zip_path.unlink(missing_ok=True)
    print(f"[splits] Đã giải nén xong {len(list(audio_root.rglob('*.wav')))} file WAV RAVDESS.")


def normalize_fer(raw_dir: Path) -> Path:
    """Trả về đường dẫn CSV chuẩn emotion,pixels,Usage trong data/raw."""
    fer = raw_dir / "fer2013.csv"
    if fer.exists() and fer.stat().st_size > 1_000_000:
        print(f"[splits] dùng sẵn {fer.name} ({fer.stat().st_size} bytes)")
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
        print(f"[splits] chuẩn hóa {icml.name} -> fer2013.csv ({len(df)} dòng)")
        return out
    raise FileNotFoundError("Thiếu data/raw/fer2013.csv (hoặc icml_face_data.csv >1MB)")


def build_audio_filelist(audio_root: Path, out_csv: Path) -> Path:
    """Quét WAV RAVDESS và CREMA-D (nếu có), sinh file filepath,speaker,label."""
    wavs = sorted(audio_root.rglob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"Không thấy WAV nào trong {audio_root}")
    rows = []
    for w in wavs:
        # 1. Định dạng RAVDESS: 03-01-05-01-01-01-01.wav
        parts_dash = w.stem.split("-")
        if len(parts_dash) == 7:
            emo_code, speaker = parts_dash[2], parts_dash[6]
            if emo_code in RAVDESS_MAP:
                rows.append({
                    "filepath": str(w.as_posix()),
                    "speaker": f"ravdess_{speaker}",
                    "label": RAVDESS_MAP[emo_code],
                })
            continue
        # 2. Định dạng CREMA-D: 1001_DFA_ANG_XX.wav
        parts_under = w.stem.split("_")
        if len(parts_under) == 4 and parts_under[2] in CREMAD_MAP:
            rows.append({
                "filepath": str(w.as_posix()),
                "speaker": f"crema_{parts_under[0]}",
                "label": CREMAD_MAP[parts_under[2]],
            })
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"Không nhận diện được định dạng tên file RAVDESS/CREMA-D trong {audio_root}")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[splits] quét {len(df)} WAV, {df['speaker'].nunique()} speaker -> {out_csv.name}")
    print(df["label"].value_counts().sort_index().to_dict())
    return out_csv


def make_audio_filelist_split(filelist: Path, splits: Path):
    return make_audio_splits(filelist, splits, seed=SEED)


def main():
    parser = argparse.ArgumentParser(description="Tạo splits cho FER2013 + RAVDESS/CREMA-D")
    parser.add_argument("--download-ravdess", action="store_true", help="Tự động tải RAVDESS từ Zenodo nếu chưa có")
    args = parser.parse_args()

    raw = Path("data/raw")
    splits = Path("data/splits")
    splits.mkdir(parents=True, exist_ok=True)

    # 1. Mặt (nếu có fer2013.csv)
    try:
        fer_csv = normalize_fer(raw)
        face_paths = make_face_splits(fer_csv, splits, seed=SEED)
        for k, p in face_paths.items():
            with open(p, encoding="utf-8") as f:
                n = sum(1 for _ in f) - 1
            print(f"[splits] mặt {k}: {n} mẫu -> {p}")
    except FileNotFoundError as e:
        print(f"[splits] Bỏ qua FER2013: {e}")

    # 2. Tiếng (RAVDESS / CREMA-D)
    audio_root = raw / "audio"
    if args.download_ravdess:
        download_ravdess_if_needed(audio_root)
    if audio_root.exists() and any(audio_root.rglob("*.wav")):
        filelist = splits / "audio_files.csv"
        build_audio_filelist(audio_root, filelist)
        audio_paths = make_audio_filelist_split(filelist, splits)
        for k, p in audio_paths.items():
            with open(p, encoding="utf-8") as f:
                n = sum(1 for _ in f) - 1
            print(f"[splits] tiếng {k}: {n} mẫu -> {p}")
        print("[splits] xong. Seed=42, audio split theo speaker.")
    else:
        print("[splits] Chưa có WAV trong data/raw/audio (dùng --download-ravdess để tải tự động).")


if __name__ == "__main__":
    main()
