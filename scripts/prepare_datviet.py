#!/usr/bin/env python
"""Chuyển đổi DataViet/Dataset_clean (.wav 16kHz) sang Mel-Spectrogram 128x128 (.npy)
và tạo splits CSV chuẩn cho Model-face-audio-test (4 lớp: angry, happy, sad, neutral).

Cách dùng:
    python scripts/prepare_datviet.py --src DataViet/Dataset_clean --out data_datviet
"""
import argparse
import csv
import struct
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.transforms as T

SAMPLE_RATE = 16000
MEL_SIZE = 128
LABEL_MAP = {"angry": 0, "happy": 1, "sad": 2, "neutral": 3}


def read_wav_float32(path: Path) -> tuple[np.ndarray, int]:
    """Đọc file WAV (PCM_16, PCM_24, PCM_32, IEEE_FLOAT) thành mảng float32 [-1, 1]."""
    raw = path.read_bytes()
    if len(raw) < 12 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError(f"Không phải file WAV hợp lệ: {path}")

    pos = 12
    fmt_tag = channels = sr = bits = None
    data_bytes = None
    while pos + 8 <= len(raw):
        chunk_id = raw[pos:pos + 4]
        chunk_size = struct.unpack_from("<I", raw, pos + 4)[0]
        chunk_data = raw[pos + 8:pos + 8 + chunk_size]
        if chunk_id == b"fmt ":
            fmt_tag, channels, sr, _, _, bits = struct.unpack_from("<HHIIHH", chunk_data, 0)
            if fmt_tag == 0xFFFE and len(chunk_data) >= 26:
                fmt_tag = struct.unpack_from("<H", chunk_data, 24)[0]
        elif chunk_id == b"data":
            data_bytes = chunk_data
            break
        pos += 8 + chunk_size + (chunk_size & 1)

    if fmt_tag is None or data_bytes is None:
        raise ValueError(f"Thiếu chunk fmt/data trong {path}")

    if fmt_tag == 3 and bits == 32:
        arr = np.frombuffer(data_bytes, dtype="<f4").astype(np.float32)
    elif fmt_tag == 1 and bits == 16:
        arr = np.frombuffer(data_bytes, dtype="<i2").astype(np.float32) / 32768.0
    elif fmt_tag == 1 and bits == 32:
        arr = np.frombuffer(data_bytes, dtype="<i4").astype(np.float32) / 2147483648.0
    elif fmt_tag == 1 and bits == 24:
        b = np.frombuffer(data_bytes, dtype=np.uint8).reshape(-1, 3)
        signed = (b[:, 0].astype(np.int32)
                  | (b[:, 1].astype(np.int32) << 8)
                  | (b[:, 2].astype(np.int32) << 16))
        signed = np.where(signed & 0x800000, signed - 0x1000000, signed)
        arr = signed.astype(np.float32) / 8388608.0
    else:
        raise ValueError(f"Định dạng WAV chưa hỗ trợ (fmt={fmt_tag}, bits={bits}): {path}")

    if channels > 1:
        arr = arr[: (len(arr) // channels) * channels].reshape(-1, channels).mean(axis=1)
    return arr, sr


def wav_to_mel128(wav: np.ndarray, sr: int, mel_tf, to_db) -> np.ndarray:
    t = torch.from_numpy(wav).float()
    if sr != SAMPLE_RATE:
        new_len = max(1, int(round(len(wav) * SAMPLE_RATE / sr)))
        t = F.interpolate(t.view(1, 1, -1), size=new_len, mode="linear", align_corners=False).view(-1)
    mel = to_db(mel_tf(t.unsqueeze(0)))
    if mel.shape[-1] != MEL_SIZE:
        mel = F.interpolate(mel.unsqueeze(0), size=(MEL_SIZE, MEL_SIZE), mode="bilinear", align_corners=False).squeeze(0)
    return mel.squeeze(0).numpy().astype(np.float16)


def split_train_val(rows: list[dict], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Chia Train -> train/val theo nhóm (gender, region, emotion, orig_id) để chống rò rỉ."""
    rng = np.random.default_rng(seed)
    strata = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["gender"], r["region"], r["emotion"])
        strata[key][r["orig_id"]].append(r)

    train_rows, val_rows = [], []
    for key in sorted(strata):
        groups = list(strata[key].keys())
        rng.shuffle(groups)
        n_val = max(1, int(round(len(groups) * val_ratio)))
        val_ids = set(groups[:n_val])
        for gid, items in strata[key].items():
            (val_rows if gid in val_ids else train_rows).extend(items)
    return train_rows, val_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="DataViet/Dataset_clean", help="Thư mục Dataset_clean")
    ap.add_argument("--out", default="data_datviet", help="Thư mục đầu ra (chứa data/splits & data/processed)")
    ap.add_argument("--val-ratio", type=float, default=0.2, help="Tỷ lệ tách val từ Train (mặc định 0.2)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src_dir = Path(args.src)
    manifest = src_dir / "manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"Không thấy {manifest}")

    raw_train, raw_test = [], []
    with open(manifest, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if str(row.get("dropped", "")).lower() == "true":
                continue
            rel = row["rel_path"].replace("\\", "/")
            wav_path = src_dir / rel
            if not wav_path.is_file():
                raise FileNotFoundError(f"Thiếu file âm thanh: {wav_path}")
            emo = row["emotion"].strip().lower()
            if emo not in LABEL_MAP:
                continue
            item = {
                "wav_path": wav_path,
                "stem": wav_path.stem,
                "gender": row["gender"].strip(),
                "region": row["region"].strip(),
                "emotion": emo,
                "label": LABEL_MAP[emo],
                "orig_id": row["orig_id"].strip(),
            }
            if row["split"].strip().lower() == "train":
                raw_train.append(item)
            else:
                raw_test.append(item)

    train_rows, val_rows = split_train_val(raw_train, args.val_ratio, args.seed)
    splits = {"train": train_rows, "val": val_rows, "test": raw_test}

    out_root = Path(args.out)
    splits_dir = out_root / "data" / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    mel_tf = T.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=MEL_SIZE)
    to_db = T.AmplitudeToDB(stype="power", top_db=80)

    for sp, items in splits.items():
        proc_dir = out_root / "data" / "processed" / "audio" / sp
        proc_dir.mkdir(parents=True, exist_ok=True)
        csv_path = splits_dir / f"audio_{sp}_proc.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f, lineterminator="\n")
            wr.writerow(["mel_path", "label", "speaker", "source", "emotion_name"])
            for idx, it in enumerate(items, 1):
                wav, sr = read_wav_float32(it["wav_path"])
                mel = wav_to_mel128(wav, sr, mel_tf, to_db)
                npy_name = f"{it['stem']}.npy"
                np.save(proc_dir / npy_name, mel)
                rel_mel = f"data/processed/audio/{sp}/{npy_name}"
                speaker = f"{it['gender']}_{it['region']}_{it['orig_id']}"
                source = f"datviet_{it['region']}"
                wr.writerow([rel_mel, it["label"], speaker, source, it["emotion"]])
                if idx % 2000 == 0:
                    print(f"  [{sp}] đã xử lý {idx}/{len(items)}...")
        print(f"Xong {sp}: {len(items)} mẫu -> {csv_path}")


if __name__ == "__main__":
    main()
