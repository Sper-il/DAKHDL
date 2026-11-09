"""Tai ASVP-ESD (bo sung 4 nhom cam xuc mo rong) vao data/raw/asvp.

Chay: python -m src.data.download_asvp [--out data/raw/asvp]

Nguon: HuggingFace EdwardLin2023/ASVP_ESD (2.150 clip Speech + 5.484 clip
NonSpeech, 13 nhan). Khong can key, chi can mang + `datasets` + `soundfile`.
File wav se nam o data/raw/asvp/Speech/actor_X/*.wav va NonSpeech/...
Xong chay: python -m src.data.make_splits (tu gop RAVDESS + CREMA-D + ASVP)

Ghi chu licence: ASVP-ESD dung cho nghien cuu, ghi cong paper
Landry et al. 2020 khi bao cao.
"""
from __future__ import annotations
import argparse
from pathlib import Path


def download_via_datasets(out: Path) -> bool:
    try:
        from datasets import load_dataset
    except ImportError:
        print("[asvp] thieu datasets, cai: pip install datasets soundfile")
        return False
    try:
        import soundfile as sf
    except ImportError:
        print("[asvp] thieu soundfile, cai: pip install soundfile")
        return False
    try:
        print("[asvp] tai EdwardLin2023/ASVP_ESD qua datasets...")
        ds = load_dataset("EdwardLin2023/ASVP_ESD", split="train")
        print(f"[asvp] thay {len(ds)} dong, cot: {ds.column_names}")
        out.mkdir(parents=True, exist_ok=True)
        n_new = 0
        for i, row in enumerate(ds):
            # Cot duong dan goc co dang .../ASVP_ESD/Speech/actor_0/08_01_....wav
            raw_path = str(row.get("path", f"clip_{i}.wav"))
            # Tach loai Speech/NonSpeech + ten file tu path goc
            p = Path(raw_path.replace("\\", "/"))
            kind = "Speech"
            for k in ("Speech", "NonSpeech"):
                if k.lower() in [x.lower() for x in p.parts]:
                    kind = k
                    break
            fname = p.name if p.suffix.lower() == ".wav" else f"clip_{i}.wav"
            # Speaker: uu tien cot speaker neu co, khong thi lay tu path actor_X
            speaker = str(row.get("speaker", ""))
            if not speaker or speaker == "None":
                speaker = "unknown"
                for part in p.parts:
                    if part.startswith("actor_"):
                        speaker = part
                        break
            dst = out / kind / speaker / fname
            if dst.exists():
                continue
            audio = row.get("audio")
            if audio is None:
                continue
            arr = audio.get("array") if isinstance(audio, dict) else None
            sr = audio.get("sampling_rate", 16000) if isinstance(audio, dict) else 16000
            if arr is None:
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(dst), arr, sr)
            n_new += 1
        print(f"[asvp] xong: {n_new} file moi -> {out}/Speech + NonSpeech/")
        return True
    except Exception as e:
        print(f"[asvp] tai that bai: {e}")
        return False


def main():
    ap = argparse.ArgumentParser(description="Tai ASVP-ESD vao data/raw/asvp")
    ap.add_argument("--out", default="data/raw/asvp")
    args = ap.parse_args()
    out = Path(args.out)
    if download_via_datasets(out):
        return
    print("[asvp] HUONG DAN THU CONG:")
    print("  1. Cai: pip install datasets soundfile")
    print("  2. Chay lai: python -m src.data.download_asvp")
    print("  3. Neu mang chan HF, tai zip tu trang dataset")
    print("     https://huggingface.co/datasets/EdwardLin2023/ASVP_ESD")
    print("     roi giai nen vao data/raw/asvp/ (giu Speech/ + NonSpeech/).")
    print("  4. Chay: python -m src.data.make_splits")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
