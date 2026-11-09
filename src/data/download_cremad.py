"""Tai CREMA-D (bo sung nhanh tieng) vao data/raw/crema.

Chay: python -m src.data.download_cremad [--method kaggle|hf] [--out data/raw/crema]

Uu tien:
  1. Kaggle: ejlok1/cremad (can kaggle.json hoac KAGGLE_USERNAME/KAGGLE_KEY)
  2. HuggingFace: narad/checkpoint... fallback danwuijing/crema-d (can mang, khong can key)
File wav se nam o data/raw/crema[/AudioWAV]/xxxx_xxx_ANG_xx.wav
Xong chay: python -m src.data.make_splits
"""
from __future__ import annotations
import argparse
from pathlib import Path


def download_kaggle(out: Path) -> bool:
    try:
        import kagglehub
    except ImportError:
        print("[crema] thieu kagglehub, cai: pip install kagglehub")
        return False
    try:
        print("[crema] tai ejlok1/cremad qua kagglehub...")
        p = kagglehub.dataset_download("ejlok1/cremad")
        src = Path(p)
        out.mkdir(parents=True, exist_ok=True)
        import shutil
        # dataset co thu muc AudioWAV
        wav_root = src / "AudioWAV" if (src / "AudioWAV").exists() else src
        n = 0
        for w in wav_root.rglob("*.wav"):
            rel = w.relative_to(wav_root)
            dst = out / "AudioWAV" / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy(w, dst)
                n += 1
        print(f"[crema] xong kaggle: {n} file moi -> {out}/AudioWAV/")
        return True
    except Exception as e:
        print(f"[crema] kaggle that bai: {e}")
        return False


def download_hf(out: Path) -> bool:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[crema] thieu huggingface_hub, cai: pip install huggingface_hub")
        return False
    candidates = ["danwuijing/crema-d", "Narad/crema-d"]
    for repo in candidates:
        try:
            print(f"[crema] thu {repo} qua huggingface_hub...")
            p = snapshot_download(repo, repo_type="dataset", allow_patterns=["*.wav", "*.WAV", "**/*.wav"])
            src = Path(p)
            out.mkdir(parents=True, exist_ok=True)
            import shutil
            n = 0
            for w in list(src.rglob("*.wav")) + list(src.rglob("*.WAV")):
                dst = out / "AudioWAV" / w.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    shutil.copy(w, dst)
                    n += 1
            if n > 0:
                print(f"[crema] xong hf {repo}: {n} file -> {out}/AudioWAV/")
                return True
        except Exception as e:
            print(f"[crema] {repo} loi: {e}")
            continue
    return False


def main():
    ap = argparse.ArgumentParser(description="Tai CREMA-D vao data/raw/crema")
    ap.add_argument("--out", default="data/raw/crema")
    ap.add_argument("--method", choices=["auto", "kaggle", "hf"], default="auto")
    args = ap.parse_args()
    out = Path(args.out)
    ok = False
    if args.method in ("auto", "kaggle"):
        ok = download_kaggle(out)
        if ok:
            return
        if args.method == "kaggle":
            raise SystemExit(1)
    if args.method in ("auto", "hf"):
        ok = download_hf(out)
        if ok:
            return
    if not ok:
        print("[crema] HUONG DAN THU CONG:")
        print("  1. Vao https://www.kaggle.com/datasets/ejlok1/cremad tai zip (605MB).")
        print("  2. Giai nen thu muc AudioWAV vao data/raw/crema/AudioWAV/ (7442 file .wav).")
        print("  3. Chay: python -m src.data.make_splits")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
