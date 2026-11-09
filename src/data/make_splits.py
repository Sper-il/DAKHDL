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

# CREMA-D filename: 1001_DFA_ANG_XX.wav -> speaker=1001, emotion=ANG
# 6 cam xuc: ANG->0, DIS->1, FEA->2, HAP->3, SAD->4, NEU->6 (khong co surprise)
CREMA_MAP = {"ANG": 0, "DIS": 1, "FEA": 2, "HAP": 3, "SAD": 4, "NEU": 6}

# ASVP-ESD filename: data/raw/asvp/Speech/actor_0/08_01_90_00_01_02_02_04.wav
# 2 so dau la ma cam xuc 01-13: 01 boredom->8, 02 neutral->6, 03 happy->3,
# 04 sad->4, 05 angry->0, 06 fear->2, 07 disgust/contempt->7,
# 08 surprise->5, 09 excited->9 (bo qua 10 pleasure, 11 pain, 12 disappointment, 13 breath)
ASVP_MAP = {"01": 8, "02": 6, "03": 3, "04": 4, "05": 0,
            "06": 2, "07": 7, "08": 5, "09": 9}


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


def build_audio_filelist(audio_root: Path, out_csv: Path, crema_root: Path | None = None,
                         asvp_root: Path | None = None) -> Path:
    """Quét WAV RAVDESS (+ CREMA-D, ASVP-ESD nếu có), sinh filepath,speaker,label,source."""
    rows = []
    # 1. RAVDESS: data/raw/audio/Actor_XX/03-01-05-01-01-01-03.wav
    if audio_root.exists():
        wavs = sorted(audio_root.rglob("*.wav"))
        for w in wavs:
            parts = w.stem.split("-")
            if len(parts) != 7:
                continue
            emo_code, speaker = parts[2], parts[6]
            if emo_code not in RAVDESS_MAP:
                continue
            rows.append({"filepath": str(w.as_posix()), "speaker": f"actor_{speaker}",
                         "label": RAVDESS_MAP[emo_code], "source": "ravdess"})
    # 2. CREMA-D: data/raw/crema[/AudioWAV]/1001_DFA_ANG_XX.wav
    if crema_root is not None and crema_root.exists():
        wavs = sorted(crema_root.rglob("*.wav"))
        for w in wavs:
            parts = w.stem.split("_")
            if len(parts) != 4:
                continue
            actor_id, _, emo_code, _ = parts
            if emo_code not in CREMA_MAP:
                continue
            rows.append({"filepath": str(w.as_posix()), "speaker": f"crema_{actor_id}",
                         "label": CREMA_MAP[emo_code], "source": "crema"})
    # 3. ASVP-ESD: data/raw/asvp/Speech/actor_0/08_01_90_00_01_02_02_04.wav
    # 2 so dau la ma 01-09 -> label mo rong 0-9 (7 contempt, 8 boredom, 9 excited)
    if asvp_root is not None and asvp_root.exists():
        wavs = sorted(asvp_root.rglob("*.wav"))
        for w in wavs:
            code = w.stem.split("_")[0] if "_" in w.stem else ""
            if code not in ASVP_MAP:
                continue
            # speaker: uu tien thu muc actor_X, fallback theo Speech/NonSpeech
            speaker = "asvp_unknown"
            for part in w.parts:
                if part.startswith("actor_"):
                    speaker = f"asvp_{part}"
                    break
            rows.append({"filepath": str(w.as_posix()), "speaker": speaker,
                         "label": ASVP_MAP[code], "source": "asvp"})
    if not rows:
        raise FileNotFoundError(f"Không thấy WAV nào trong {audio_root} (và {crema_root}, {asvp_root})")
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    n_rav = int((df["source"] == "ravdess").sum()) if "source" in df.columns else len(df)
    n_crema = int((df["source"] == "crema").sum()) if "source" in df.columns else 0
    n_asvp = int((df["source"] == "asvp").sum()) if "source" in df.columns else 0
    print(f"[splits] quét {len(df)} WAV ({n_rav} ravdess + {n_crema} crema + {n_asvp} asvp), "
          f"{df['speaker'].nunique()} speaker -> {out_csv.name}")
    print(df["label"].value_counts().sort_index().to_dict())
    if n_crema or n_asvp:
        print(df.groupby("source")["label"].value_counts().to_dict())
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

    # 2. Tieng (RAVDESS + CREMA-D + ASVP neu co)
    audio_root = raw / "audio"
    crema_root = raw / "crema"
    if not crema_root.exists():
        # Ho tro dat AudioWAV ngay trong data/raw/crema/
        alt = raw / "crema" / "AudioWAV"
        crema_root = crema_root if crema_root.exists() or alt.exists() else None
    asvp_root = raw / "asvp"
    asvp_root = asvp_root if asvp_root.exists() else None
    filelist = splits / "audio_files.csv"
    build_audio_filelist(audio_root, filelist, crema_root=crema_root, asvp_root=asvp_root)
    audio_paths = make_audio_filelist_split(filelist, splits)
    for k, p in audio_paths.items():
        with open(p, encoding="utf-8") as f:
            n = sum(1 for _ in f) - 1
        print(f"[splits] tieng {k}: {n} mau -> {p}")
    print("[splits] xong. Seed=42, audio split theo speaker (gop RAVDESS + CREMA-D + ASVP).")


if __name__ == "__main__":
    main()
