"""Đọc data (numpy/pandas thuần, không cần torch).

Cấu trúc data mong đợi (đúng như data.zip của dự án):
    <data_root>/data/splits/face_{train,val,test}.csv          cột: label, pixels, (Usage)
    <data_root>/data/splits/audio_{train,val,test}_proc.csv    cột: mel_path, label, (speaker, source)
    <data_root>/data/processed/audio/...npy                    Mel-spectrogram 128x128 (dB, -80..0)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FACE_SIZE = 48
MEL_SHAPE = (128, 128)


# ----------------------------------------------------------------- đường dẫn
def default_splits_dir(data_root) -> Path:
    root = Path(data_root)
    for cand in (root / "data" / "splits", root / "splits"):
        if cand.is_dir():
            return cand
    raise FileNotFoundError(
        f"Không thấy thư mục splits trong {root}. Cần <data_root>/data/splits "
        f"(data_root là thư mục chứa thư mục 'data' sau khi giải nén data.zip).")


def split_csv_path(splits_dir, branch: str, split: str) -> Path:
    name = {"face": f"face_{split}.csv", "audio": f"audio_{split}_proc.csv"}[branch]
    p = Path(splits_dir) / name
    if not p.is_file():
        raise FileNotFoundError(f"Thiếu file {p}")
    return p


def resolve_paths(root, rels):
    """mel_path trong CSV là đường dẫn tương đối (vd data/processed/audio/train/x.npy).
    Thử vài cách nối với data_root, chọn cách đúng cho file đầu tiên rồi áp cho tất cả."""
    root = Path(root)
    rels = [str(r).replace("\\", "/") for r in rels]
    if not rels:
        return []
    patterns = [
        lambda r: root / r,
        lambda r: root / "data" / r,
        lambda r: root.parent / r,
        lambda r: root / r[5:] if r.startswith("data/") else root / r,
    ]
    for pat in patterns:
        if pat(rels[0]).exists():
            paths = [pat(r) for r in rels]
            missing = [p for p in paths if not p.exists()]
            if missing:
                raise FileNotFoundError(f"Thiếu {len(missing)} file, ví dụ: {missing[:3]}")
            return paths
    raise FileNotFoundError(
        f"Không tìm thấy {rels[0]!r} dưới data_root={root}. "
        f"Hãy chỉnh data_root cho đúng (thư mục chứa thư mục 'data').")


# ----------------------------------------------------------------- cache
def _cache_file(cache_dir, tag: str, csv_path: Path):
    if not cache_dir:
        return None
    st = csv_path.stat()
    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{tag}_{csv_path.stem}_{st.st_size}_{int(st.st_mtime)}.npy"


# ----------------------------------------------------------------- mặt
def _parse_pixels(series) -> np.ndarray:
    n = len(series)
    out = np.empty((n, FACE_SIZE * FACE_SIZE), dtype=np.uint8)
    for i, s in enumerate(series.values):
        vals = list(map(int, str(s).split()))
        if len(vals) != FACE_SIZE * FACE_SIZE:
            raise ValueError(f"Dòng {i}: cần {FACE_SIZE * FACE_SIZE} pixel, thấy {len(vals)}")
        out[i] = vals
    return out.reshape(n, FACE_SIZE, FACE_SIZE)


def load_face_split(csv_path, num_classes: int, cache_dir=None) -> dict:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    lab = "label" if "label" in df.columns else "emotion"
    if lab not in df.columns or "pixels" not in df.columns:
        raise ValueError(f"{csv_path.name} cần cột label (hoặc emotion) và pixels, có: {list(df.columns)}")
    cf = _cache_file(cache_dir, "facepix", csv_path)
    if cf is not None and cf.exists():
        images = np.load(cf)
    else:
        images = _parse_pixels(df["pixels"])
        if cf is not None:
            np.save(cf, images)
    labels = df[lab].to_numpy(dtype=np.int64)
    keep = labels < num_classes
    return {"images": images[keep], "labels": labels[keep], "meta": None}


# ----------------------------------------------------------------- tiếng
def load_audio_split(csv_path, data_root, num_classes: int, cache_dir=None) -> dict:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    for c in ("mel_path", "label"):
        if c not in df.columns:
            raise ValueError(f"{csv_path.name} thiếu cột {c}, có: {list(df.columns)}")
    df = df[df["label"] < num_classes].reset_index(drop=True)
    cf = _cache_file(cache_dir, f"melc{num_classes}", csv_path)
    if cf is not None and cf.exists():
        mels = np.load(cf)
        if len(mels) != len(df):
            mels = None
    else:
        mels = None
    if mels is None:
        paths = resolve_paths(data_root, df["mel_path"].tolist())
        mels = np.empty((len(paths), *MEL_SHAPE), dtype=np.float16)
        for i, p in enumerate(paths):
            x = np.load(p)
            if x.shape != MEL_SHAPE:
                raise ValueError(f"{p} có shape {x.shape}, cần {MEL_SHAPE}")
            mels[i] = x
        if cf is not None:
            np.save(cf, mels)
    meta = df.drop(columns=["mel_path"])
    return {"images": None, "mels": mels, "labels": df["label"].to_numpy(dtype=np.int64), "meta": meta}


# ----------------------------------------------------------------- thống kê
def compute_stats(arr: np.ndarray, scale: float = 1.0, chunk: int = 1024):
    """Trung bình và độ lệch chuẩn toàn cục (tính theo khối để đỡ tốn RAM)."""
    s = 0.0
    ss = 0.0
    n = 0
    for i in range(0, len(arr), chunk):
        x = arr[i:i + chunk].astype(np.float64) / scale
        s += x.sum()
        ss += (x ** 2).sum()
        n += x.size
    mean = s / n
    var = max(ss / n - mean ** 2, 1e-12)
    return float(mean), float(np.sqrt(var))


def class_counts(labels, num_classes: int) -> np.ndarray:
    return np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes)[:num_classes]


def class_weights(counts, mode: str) -> np.ndarray:
    """none: 1 | inv: N/(C*n_c) | inv_sqrt: căn bậc hai của inv. Chuẩn hóa để trung bình = 1."""
    counts = np.maximum(np.asarray(counts, dtype=np.float64), 1.0)
    if mode == "none":
        return np.ones_like(counts, dtype=np.float32)
    w = counts.sum() / (len(counts) * counts)
    if mode == "inv_sqrt":
        w = np.sqrt(w)
    w = w / w.mean()
    return w.astype(np.float32)
