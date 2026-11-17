"""Dataset PyTorch cho nhánh mặt và nhánh tiếng + hàm dựng toàn bộ data từ cấu hình."""
from __future__ import annotations

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

from . import data_utils as du


class FaceDataset(Dataset):
    """Ảnh xám 48x48 -> tensor [1,48,48], chuẩn hóa theo mean/std của tập train."""

    def __init__(self, images, labels, mean, std, train=False, aug=None):
        self.images, self.labels = images, labels
        self.mean, self.std, self.train = float(mean), float(std), train
        aug = aug or {}
        tfs = []
        if train:
            if aug.get("hflip", True):
                tfs.append(T.RandomHorizontalFlip())
            rot, tr = float(aug.get("rotation", 0)), float(aug.get("translate", 0))
            if rot > 0 or tr > 0:
                tfs.append(T.RandomAffine(degrees=rot, translate=(tr, tr) if tr > 0 else None))
        self.geo = T.Compose(tfs) if tfs else None
        p_erase = float(aug.get("erasing", 0))
        self.erase = T.RandomErasing(p=p_erase, scale=(0.02, 0.15), value=0) if (train and p_erase > 0) else None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        x = torch.from_numpy(np.ascontiguousarray(self.images[i])).float().div_(255.0).unsqueeze(0)
        if self.geo is not None:
            x = self.geo(x)
        x = (x - self.mean) / self.std
        if self.erase is not None:
            x = self.erase(x)
        return x, int(self.labels[i]), i


class AudioDataset(Dataset):
    """Mel (dB) 128x128 -> tensor [1,128,128]; train có SpecAugment (che dải tần + đoạn thời gian)."""

    def __init__(self, mels, labels, mean, std, train=False, aug=None):
        self.mels, self.labels = mels, labels
        self.mean, self.std, self.train = float(mean), float(std), train
        aug = aug or {}
        self.freq_mask = int(aug.get("freq_mask", 0))
        self.time_mask = int(aug.get("time_mask", 0))
        self.n_masks = int(aug.get("n_masks", 2))

    def __len__(self):
        return len(self.labels)

    def _spec_augment(self, x):
        h, w = x.shape
        for _ in range(self.n_masks):
            if self.freq_mask > 0:
                m = int(torch.randint(0, self.freq_mask + 1, (1,)))
                s = int(torch.randint(0, max(1, h - m + 1), (1,)))
                x[s:s + m, :] = 0.0
            if self.time_mask > 0:
                m = int(torch.randint(0, self.time_mask + 1, (1,)))
                s = int(torch.randint(0, max(1, w - m + 1), (1,)))
                x[:, s:s + m] = 0.0
        return x

    def __getitem__(self, i):
        x = torch.from_numpy(self.mels[i].astype(np.float32))
        x = (x - self.mean) / self.std
        if self.train:
            x = self._spec_augment(x)
        return x.unsqueeze(0), int(self.labels[i]), i


def build_datasets(cfg: dict, log=print, stats=None, splits=("train", "val", "test")) -> dict:
    """Đọc CSV/npy, tính mean/std (nếu chưa có), dựng Dataset cho từng split.
    Trả về dict: datasets, meta (DataFrame theo split), mean, std, counts (train), class_counts_all."""
    branch, n = cfg["branch"], int(cfg["num_classes"])
    root = cfg["data_root"]
    splits_dir = cfg.get("splits_dir") or du.default_splits_dir(root)
    cache = cfg.get("cache_dir")
    raw = {}
    for sp in splits:
        csv = du.split_csv_path(splits_dir, branch, sp)
        log(f"Đọc {csv.name} ...")
        if branch == "face":
            raw[sp] = du.load_face_split(csv, n, cache)
        else:
            raw[sp] = du.load_audio_split(csv, root, n, cache)
        counts = du.class_counts(raw[sp]["labels"], n)
        log(f"  {sp}: {len(raw[sp]['labels'])} mẫu | số mẫu mỗi lớp = {counts.tolist()}")

    key = "images" if branch == "face" else "mels"
    if stats is None:
        if "train" not in raw:
            raise ValueError("Cần split train để tính mean/std (hoặc truyền stats)")
        scale = 255.0 if branch == "face" else 1.0
        mean, std = du.compute_stats(raw["train"][key], scale=scale)
    else:
        mean, std = stats
    log(f"Chuẩn hóa: mean={mean:.5f}, std={std:.5f}")

    cls = FaceDataset if branch == "face" else AudioDataset
    datasets = {}
    for sp in raw:
        datasets[sp] = cls(raw[sp][key], raw[sp]["labels"], mean, std,
                           train=(sp == "train"), aug=cfg.get("augment"))
    return {
        "datasets": datasets,
        "meta": {sp: raw[sp]["meta"] for sp in raw},
        "labels": {sp: raw[sp]["labels"] for sp in raw},
        "mean": mean, "std": std,
        "counts": {sp: du.class_counts(raw[sp]["labels"], n).tolist() for sp in raw},
    }
