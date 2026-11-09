"""Tai ASVP-ESD (bo sung surprise/contempt/boredom/excited) vao data/raw/asvp.

Chay: python -m src.data.download_asvp [--out data/raw/asvp]

Nguon thuc te da dung: Kaggle dejolilandry/asvpesdspeech-nonspeech-emotional-
utterances (ban ASVP-ESD-Update: 13.964 WAV Audio + Bonus, giu 12.227 file
thuoc 9/13 nhan). Ten file dang 03-<01 Speech|02 NonSpeech>-<emotion 01-13>-...
ma cam xuc nam o token thu 3 (pos2): 01 boredom->8, 02 neutral->6, 03 happy->3,
04 sad->4, 05 angry->0, 06 fear->2, 07 disgust/contempt->7, 08 surprise->5,
09 excited->9 (bo qua 10 pleasure, 11 pain, 12 disappointment, 13 breath).

Xong chay: python -m src.data.make_splits (tu gop RAVDESS + CREMA-D + ASVP,
split theo speaker, seed 42).

Ghi chu licence: ASVP-ESD dung cho nghien cuu, ghi cong paper
Landry et al. 2020 khi bao cao.
"""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path


KAGGLE_DATASET = "dejolilandry/asvpesdspeech-nonspeech-emotional-utterances"


def download_via_kagglehub(out: Path) -> bool:
    try:
        import kagglehub
    except ImportError:
        print("[asvp] thieu kagglehub, cai: pip install kagglehub")
        return False
    try:
        print(f"[asvp] tai {KAGGLE_DATASET} qua kagglehub...")
        src = Path(kagglehub.dataset_download(KAGGLE_DATASET))
        print(f"[asvp] cache: {src}")
        # Ban ASVP-ESD-Update chua Audio/ + Bonus/
        root = src / "ASVP-ESD-Update" if (src / "ASVP-ESD-Update").exists() else src
        n_new = 0
        for kind in ("Audio", "Bonus"):
            kdir = root / kind
            if not kdir.exists():
                continue
            for wav in sorted(kdir.rglob("*.wav")):
                dst = out / kind / wav.relative_to(kdir)
                if dst.exists():
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(wav, dst)
                n_new += 1
        print(f"[asvp] xong: {n_new} file moi -> {out}/Audio + Bonus/")
        return True
    except Exception as e:
        print(f"[asvp] tai that bai: {e}")
        return False


def main():
    ap = argparse.ArgumentParser(description="Tai ASVP-ESD vao data/raw/asvp")
    ap.add_argument("--out", default="data/raw/asvp")
    args = ap.parse_args()
    out = Path(args.out)
    if download_via_kagglehub(out):
        return
    print("[asvp] HUONG DAN THU CONG:")
    print("  1. Cai: pip install kagglehub")
    print("  2. Chay: python -c \"import kagglehub; kagglehub.dataset_download(")
    print(f"     '{KAGGLE_DATASET}')\"")
    print("  3. Chep thu muc Audio/ + Bonus/ vao data/raw/asvp/")
    print("  4. Chay: python -m src.data.make_splits")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
