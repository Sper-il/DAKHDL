"""Trich anh mat + mel + wav mau cho 7 cam xuc, luu vao results/eda/."""
from pathlib import Path
import csv
import shutil
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "DejaVu Sans"

LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
LABELS_VI = ["Giận dữ", "Ghê tởm", "Sợ hãi", "Vui vẻ", "Buồn", "Ngạc nhiên", "Trung tính"]
EDA = Path("results/eda")
EDA.mkdir(parents=True, exist_ok=True)

# 1. Anh mat mau: lay 1 anh dau tien moi label tu face_train.csv
face_csv = Path("data/splits/face_train.csv")
found = {}
with open(face_csv, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        lab = int(row["label"])
        if lab not in found:
            found[lab] = row["pixels"]
            if len(found) == 7:
                break
for i in range(7):
    arr = np.fromstring(found[i], sep=" ", dtype=np.float32).reshape(48, 48)
    plt.figure(figsize=(2.5, 2.5))
    plt.imshow(arr, cmap="gray")
    plt.title(f"{LABELS_VI[i]} ({LABELS[i]})", fontsize=10)
    plt.axis("off")
    plt.tight_layout()
    p = EDA / f"face_emotion_{i}_{LABELS[i]}.png"
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"[samples] wrote {p}")

# 2. Audio mau: 1 wav dau tien moi label tu audio_train.csv + mel tuong ung
import librosa
adf = pd.read_csv("data/splits/audio_train.csv")
firsts = adf.groupby("label").first().to_dict()["filepath"]
for i in range(7):
    src = Path(firsts[i])
    dst = EDA / f"audio_emotion_{i}_{LABELS[i]}.wav"
    shutil.copy(src, dst)
    print(f"[samples] copied {src} -> {dst}")
    y, _ = librosa.load(str(src), sr=16000, mono=True, duration=4.0)
    mel = librosa.feature.melspectrogram(y=y, sr=16000, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    plt.figure(figsize=(4, 3))
    plt.imshow(mel_db, origin="lower", aspect="auto")
    plt.title(f"Mel {LABELS_VI[i]} ({LABELS[i]})", fontsize=10)
    plt.xlabel("Khung thời gian")
    plt.ylabel("Băng tần Mel")
    plt.tight_layout()
    p = EDA / f"mel_emotion_{i}_{LABELS[i]}.png"
    plt.savefig(p, dpi=120)
    plt.close()
    print(f"[samples] wrote {p}")
print("[samples] xong 7 cam xuc")
