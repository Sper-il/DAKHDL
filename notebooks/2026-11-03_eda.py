"""EDA mat + tieng: phan bo lop, speaker, pixel/mel mau, luu plots vao results/eda/."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
LABELS_VI = ["Giận dữ", "Ghê tởm", "Sợ hãi", "Vui vẻ", "Buồn", "Ngạc nhiên", "Trung tính"]
SPLIT_VI = {"train": "huấn luyện", "val": "kiểm định", "test": "kiểm thử"}
OUT = Path("results/eda")
OUT.mkdir(parents=True, exist_ok=True)

def plot_dist(counts: dict, title: str, path: Path):
    ids = sorted(counts)
    vals = [counts[i] for i in ids]
    names = [f"{LABELS_VI[i]}\n({LABELS[i]})" for i in ids]
    plt.figure(figsize=(9, 4.5))
    bars = plt.bar(names, vals)
    plt.bar_label(bars, fmt="%d", fontsize=8)
    plt.xticks(rotation=15, ha="right")
    plt.ylabel("Số mẫu")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()

# 1. Face splits
face_stats = {}
for split in ["train", "val", "test"]:
    df = pd.read_csv(f"data/splits/face_{split}.csv", usecols=["label"])
    vc = df["label"].value_counts().sort_index().to_dict()
    face_stats[split] = {"total": len(df), "counts": vc}
    plot_dist(vc, f"FER2013 {split} — Phân bố 7 cảm xúc (n={len(df)})", OUT / f"face_{split}_dist.png")

# 2. Audio splits
audio_stats = {}
for split in ["train", "val", "test"]:
    df = pd.read_csv(f"data/splits/audio_{split}.csv")
    vc = df["label"].value_counts().sort_index().to_dict()
    audio_stats[split] = {"total": len(df), "counts": vc,
                          "speakers": sorted(df["speaker"].unique().tolist())}
    plot_dist(vc, f"RAVDESS {split} — Phân bố 7 cảm xúc (n={len(df)})", OUT / f"audio_{split}_dist.png")

# 3. Kiem tra leakage speaker
tr = set(pd.read_csv("data/splits/audio_train.csv")["speaker"])
va = set(pd.read_csv("data/splits/audio_val.csv")["speaker"])
te = set(pd.read_csv("data/splits/audio_test.csv")["speaker"])
leak = {"train-val": sorted(tr & va), "train-test": sorted(tr & te), "val-test": sorted(va & te)}

# 4. Mau anh mat: mean/std pixel + 1 anh vi du
import numpy as np
df = pd.read_csv("data/splits/face_train.csv", usecols=["label"], nrows=2000)
# read pixels separately to avoid quoting issues
import csv
px_list = []
with open("data/splits/face_train.csv", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        if i >= 2000:
            break
        px_list.append(np.fromstring(row["pixels"], sep=" ", dtype=np.float32))
px = np.stack(px_list)
print(f"[eda] face sample pixels: mean={px.mean():.1f} std={px.std():.1f} min={px.min():.0f} max={px.max():.0f}")
plt.figure(figsize=(4, 4))
plt.imshow(px[0].reshape(48, 48), cmap="gray")
plt.title(f"Ảnh mặt mẫu train[0] — {LABELS_VI[int(df['label'].iloc[0])]} ({LABELS[int(df['label'].iloc[0])]})")
plt.axis("off")
plt.savefig(OUT / "face_sample.png", dpi=120)
plt.close()

# 5. Mau audio: mel 128x128
try:
    import librosa
    adf = pd.read_csv("data/splits/audio_train.csv", nrows=1)
    w = adf.iloc[0]["filepath"]
    y, _ = librosa.load(w, sr=16000, mono=True, duration=4.0)
    print(f"[eda] audio mau: {w} dai {len(y)/16000:.2f}s")
    mel = librosa.feature.melspectrogram(y=y, sr=16000, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    plt.figure(figsize=(6, 4.5))
    plt.imshow(mel_db, origin="lower", aspect="auto")
    plt.title(f"Mel-spectrogram mẫu — {LABELS_VI[int(adf.iloc[0]['label'])]} ({LABELS[int(adf.iloc[0]['label'])]})")
    plt.xlabel("Khung thời gian")
    plt.ylabel("Băng tần Mel")
    plt.tight_layout()
    plt.savefig(OUT / "audio_mel_sample.png", dpi=120)
    plt.close()
except Exception as e:
    print("[eda] bo qua mel mau:", str(e).encode("ascii", "ignore").decode())

print("[eda] face:", face_stats)
print("[eda] audio:", {k: {"total": v["total"], "counts": v["counts"], "n_speakers": len(v["speakers"])} for k, v in audio_stats.items()})
print("[eda] leakage speaker (phai rong):", leak)
print(f"[eda] plots luu tai {OUT}/")
